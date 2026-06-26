import base64
import io
import os
import tempfile
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import fitsio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


class PhotometryValidationError(ValueError):
    pass


class PhotometryCalibrationError(RuntimeError):
    pass


DEFAULT_CONFIG = {
    "photometry": {
        "catalog": "PanSTARRS1",
        "obs_band": "g",
        "cal_band": "r",
    },
    "detection": {
        "snr": 3.0,
        "aperture_radius": 7.0,
    },
    "output": {
        "make_plots": False,
        "write_fits": True,
    },
    "meta": {
        "plot_type": "color_correction",
    },
}


def merge_config(user_config: Optional[dict], default_config: dict) -> dict:
    result = deepcopy(default_config)

    if not user_config:
        return result

    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(value, result[key])
        else:
            result[key] = value

    return result


def _as_float(body: Dict[str, Any], name: str, default=None, required: bool = False):
    value = body.get(name, default)

    if value is None:
        if required:
            raise PhotometryValidationError(f"{name} is required")
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PhotometryValidationError(f"{name} must be a number") from exc


def _as_bool(body: Dict[str, Any], name: str, default: bool = False) -> bool:
    value = body.get(name, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}

    return bool(value)


def _safe_filename_from_url(file_url: str) -> str:
    parsed = urllib.parse.urlparse(file_url)
    name = os.path.basename(parsed.path) or "input.fits"

    if not name.lower().endswith((".fits", ".fit", ".fts")):
        name = f"{name}.fits"

    safe = "".join(
        ch if ch.isalnum() or ch in {".", "_", "-"} else "_"
        for ch in name
    )

    return safe or "input.fits"


def _resolve_input_fits(file_url: str) -> str:
    parsed = urllib.parse.urlparse(file_url)

    if parsed.scheme in {"http", "https"}:
        out_dir = Path(tempfile.gettempdir()) / "catch_analysis_tools"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / _safe_filename_from_url(file_url)
        urllib.request.urlretrieve(file_url, out_path)

        return str(out_path)

    if parsed.scheme == "file":
        local_path = urllib.request.url2pathname(parsed.path)
    else:
        local_path = file_url

    if not os.path.exists(local_path):
        raise PhotometryValidationError(
            f"WCS-solved FITS file not found: {file_url}"
        )

    return local_path


def validate_and_normalize(body: Dict[str, Any]) -> Dict[str, Any]:
    if body is None:
        raise PhotometryValidationError("Request body is required")

    wcs_image_url = body.get("wcs_image_url")

    if not wcs_image_url:
        raise PhotometryValidationError("wcs_image_url is required")

    color_index = body.get("color_term")
    obs_band = body.get("obs_band")
    cal_band = body.get("cal_band", "r")

    if obs_band is None and isinstance(color_index, str) and "-" in color_index:
        obs_band = color_index.split("-", 1)[0]

    if obs_band is None:
        obs_band = "g"

    return {
        "wcs_image_url": wcs_image_url,
        "photometry": {
            "catalog": body.get("catalog", "PanSTARRS1"),
            "obs_band": obs_band,
            "cal_band": cal_band,
        },
        "detection": {
            "snr": _as_float(body, "snr_threshold", 3.0),
            "aperture_radius": _as_float(body, "aperture_radius", 7.0),
        },
        "output": {
            "make_plots": _as_bool(body, "return_plot", False),
            "write_fits": _as_bool(body, "write_fits", True),
        },
        "meta": {
            "plot_type": body.get("plot_type", "color_correction"),
        },
    }


def _detect_sources_on_background_subtracted_image(
    image_sub: np.ndarray,
    bkg_err: float,
    snr: float,
    aperture_radius: float,
) -> pd.DataFrame:
    sep.set_sub_object_limit(500)

    sources = sep.extract(
        image_sub,
        thresh=snr,
        err=bkg_err,
        deblend_nthresh=16,
    )

    source_list = pd.DataFrame(sources)

    if source_list.empty:
        raise PhotometryValidationError(
            "No sources were detected in the WCS-solved image."
        )

    flux, flux_err, _ = sep.sum_circle(
        image_sub,
        source_list["x"],
        source_list["y"],
        aperture_radius,
        err=bkg_err,
    )

    source_list["aperture_sum"] = flux
    source_list["aperture_err"] = flux_err

    source_list = source_list[source_list["aperture_sum"] > 0].reset_index(
        drop=True
    )

    if source_list.empty:
        raise PhotometryValidationError(
            "No detected sources have positive aperture_sum."
        )

    return source_list


def _load_wcs_from_fits(input_fits: str) -> WCS:
    with fits.open(input_fits) as hdul:
        wcs_solution = WCS(hdul[0].header, relax=True)

    if not wcs_solution.has_celestial:
        raise PhotometryValidationError(
            "Input FITS does not contain a celestial WCS."
        )

    return wcs_solution


def _attach_sky_coordinates(
    source_list: pd.DataFrame,
    wcs_solution: WCS,
) -> tuple[pd.DataFrame, SkyCoord]:
    x = source_list["x"].to_numpy(dtype=float)
    y = source_list["y"].to_numpy(dtype=float)

    world = wcs_solution.pixel_to_world(x, y)

    source_list = source_list.copy()
    source_list["RA"] = world.ra.deg
    source_list["Dec"] = world.dec.deg

    sky_coords = SkyCoord(source_list["RA"].to_numpy(), source_list["Dec"].to_numpy(), unit="deg")

    return source_list, sky_coords


def _load_photometric_assets(input_fits: str, det_cfg: dict) -> dict:
    image = fitsio.read(input_fits).astype(np.float32)

    bkg = sep.Background(image)
    image_sub = image - bkg.back()

    source_list = _detect_sources_on_background_subtracted_image(
        image_sub=image_sub,
        bkg_err=float(bkg.globalrms),
        snr=det_cfg.get("snr", 3.0),
        aperture_radius=det_cfg.get("aperture_radius", 7.0),
    )

    wcs_solution = _load_wcs_from_fits(input_fits)
    source_list, sky_coords = _attach_sky_coordinates(source_list, wcs_solution)

    return {
        "image": image,
        "image_sub": image_sub,
        "bkg": bkg,
        "source_list": source_list,
        "sky_coords": sky_coords,
        "wcs_solution": wcs_solution,
    }


def _encode_figure(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _public_file_reference(path: str) -> str:
    return path


def run_photometry_from_config(input_fits: str, user_config: dict) -> dict:
    from catch_analysis_tools.calibration.photometry import (
        calibrate_photometric_zero_point,
        get_color_corrected_indices,
        get_matched_indices,
        plot_color_correction,
        plot_photometric_matches,
        write_photometric_calibration_output,
    )

    config = merge_config(user_config, DEFAULT_CONFIG)

    phot_cfg = config["photometry"]
    det_cfg = config["detection"]
    out_cfg = config["output"]
    meta_cfg = config["meta"]

    assets = _load_photometric_assets(input_fits, det_cfg)

    image = assets["image"]
    image_sub = assets["image_sub"]
    source_list = assets["source_list"]
    sky_coords = assets["sky_coords"]
    wcs_solution = assets["wcs_solution"]

    try:
        calibration = calibrate_photometric_zero_point(
            sky_coords=sky_coords,
            source_list=source_list,
            catalog=phot_cfg["catalog"],
            obs_band=phot_cfg["obs_band"],
            cal_band=phot_cfg["cal_band"],
        )
    except Exception as exc:
        raise PhotometryCalibrationError(
            f"Photometric calibration failed: {exc}"
        ) from exc

    zp = calibration["zp"]
    color_term = calibration.get("color_term", calibration.get("C"))
    unc = calibration["unc"]
    m = calibration["m"]
    m_inst = calibration["m_inst"]
    color_mags = calibration["color_mags"]
    color_index = calibration["color_index"]
    objids = calibration["objids"]

    matched_idx = get_matched_indices(objids, len(source_list))
    colored_idx = get_color_corrected_indices(color_mags, len(source_list))

    plots = {}

    if out_cfg.get("make_plots", False):
        requested_plot = meta_cfg.get("plot_type", "color_correction")

        make_color = requested_plot in {"color_correction", "all"}
        make_overlay = requested_plot in {"image_overlay", "all"}

        if make_color:
            fig, _ = plot_color_correction(
                color_mags,
                m,
                m_inst,
                color_term,
                zp,
                color_index,
            )
            plots["color_correction"] = _encode_figure(fig)
            plt.close(fig)

        if make_overlay:
            fig, _ = plot_photometric_matches(
                image_sub,
                source_list,
                matched_idx,
                colored_idx,
            )
            plots["image_overlay"] = _encode_figure(fig)
            plt.close(fig)

    photometry_image_url = None

    if out_cfg.get("write_fits", True):
        file_base = os.path.splitext(input_fits)[0]
        output_fits = f"{file_base}_photometry.fits"

        write_photometric_calibration_output(
            image=image,
            wcs_solution=wcs_solution,
            source_list=source_list,
            matched_idx=matched_idx,
            color_corrected_idx=colored_idx,
            output_fits=output_fits,
            zero_point=zp,
            zero_point_uncertainty=unc,
            catalog=phot_cfg["catalog"],
            obs_band=phot_cfg["obs_band"],
            cal_band=phot_cfg["cal_band"],
            color_index=color_index,
            color_term=color_term,
        )

        photometry_image_url = _public_file_reference(output_fits)

    result = {
        "photometry": {
            "zero_point": float(zp),
            "color_term": float(color_term),
            "uncertainty": float(unc),
        },
        "sources": {
            "detected": int(len(source_list)),
            "matched": int(len(matched_idx)),
        },
    }

    if photometry_image_url is not None:
        result["photometry_image_url"] = photometry_image_url

    if plots:
        result["plots"] = plots

    return result


def run_photometry_pipeline(body=None, **kwargs):
    if body is None:
        body = kwargs

    config = validate_and_normalize(body)
    input_fits = _resolve_input_fits(config["wcs_image_url"])

    result = run_photometry_from_config(input_fits, config)

    if config["output"]["make_plots"]:
        plot_type = config["meta"]["plot_type"]
        png_b64 = result["plots"][plot_type]
        png_bytes = base64.b64decode(png_b64)

        return png_bytes, 200, {"Content-Type": "image/png"}

    return result, 200, {"Content-Type": "application/json"}