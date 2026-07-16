import numpy as np
import os
import io
import base64
import fitsio
import sep
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from typing import Dict, Any
from copy import deepcopy
from catch_analysis_tools.calibration.astrometry import run_astrometry_calibration
from catch_analysis_tools.calibration.photometry import (
    calibrate_photometric_zero_point,
    get_matched_indices,
    get_color_corrected_indices,
    plot_color_correction,
    plot_photometric_matches,
    write_photometric_calibration_output
)

class AstrometrySolveError(RuntimeError):
    pass


class AstrometryValidationError(ValueError):
    pass


DEFAULT_CONFIG = {
    "wcs": {
        "use_ra_dec": True,
        "pixel_scale": 2.5,
        "scale_low": None,
        "scale_high": None,
        "search_radius": 2.0,
    },
    "detection": {
        "snr": 3.0,
        "aperture_radius": 7.0,
    },
    "photometry": {
        "catalog": "PanSTARRS1",
        "obs_band": "g",
        "cal_band": "r",
    },
    "output": {
        "make_plots": False,
        "write_fits": False,
    }
}


def merge_config(user_config, default_config):
    result = deepcopy(default_config)
    for k, v in user_config.items():
        if isinstance(v, dict) and k in result:
            result[k] = merge_config(v, result[k])
        else:
            result[k] = v
    return result


def run_pipeline(input_fits: str, user_config: dict) -> dict:
    config = merge_config(user_config, DEFAULT_CONFIG)

    wcs_cfg = config["wcs"]
    det_cfg = config["detection"]
    phot_cfg = config["photometry"]
    out_cfg = config["output"]

    file_base = os.path.splitext(input_fits)[0]
    image = fitsio.read(input_fits).astype(np.float32)

    # Compute global background RMS needed by your colleagues' updated astrometry code
    bkg = sep.Background(image)
    bkg_err = bkg.globalrms

    # --- 1. Run Modular Astrometry Calibration ---
    astrom_res = run_astrometry_calibration(
        input_fits=input_fits,
        ra_deg=wcs_cfg.get("ra"),
        dec_deg=wcs_cfg.get("dec"),
        bkg_err=bkg_err,
        pixel_scale=wcs_cfg.get("pixel_scale", 1.86),
        snr=det_cfg.get("snr", 7.0),
        output_fits=None
    )

    source_list = astrom_res["source_list"]
    sky_coords = astrom_res["sky_coords"]
    wcs_solution = astrom_res["wcs_solution"]

    # Calculate center coordinates using the returned WCS
    ny, nx = image.shape
    center_world = wcs_solution.pixel_to_world(nx / 2.0, ny / 2.0)
    center_ra = float(center_world.ra.deg)
    center_dec = float(center_world.dec.deg)

    # --- 2. Run Modular Photometry Calibration ---
    calibration = calibrate_photometric_zero_point(
        sky_coords=sky_coords,
        source_list=source_list,
        catalog=phot_cfg["catalog"],
        obs_band=phot_cfg["obs_band"],
        cal_band=phot_cfg["cal_band"]
    )

    zp = calibration["zp"]
    color_term = calibration["color_term"]
    unc = calibration["unc"]
    m = calibration["m"]
    m_inst = calibration["m_inst"]
    color_mags = calibration["color_mags"]
    color_index = calibration["color_index"]
    objids = calibration["objids"]

    # Filter source indices using modular photometry filters
    matched_idx = get_matched_indices(objids, len(source_list))
    colored_idx = get_color_corrected_indices(color_mags, len(source_list))

    # --- 3. Dynamic Plotting Layer ---
    plots = {}
    if out_cfg["make_plots"]:
        fig1, _ = plot_color_correction(color_mags, m, m_inst, color_term, zp, color_index)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=150, bbox_inches="tight")
        plots["color_correction"] = base64.b64encode(buf1.getvalue()).decode()
        plt.close(fig1)

        # Get the background subtracted image array for the stars overlay
        image_sub = image - bkg.back()
        fig2, _ = plot_photometric_matches(image_sub, source_list, matched_idx, colored_idx)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
        plots["image_overlay"] = base64.b64encode(buf2.getvalue()).decode()
        plt.close(fig2)

    # --- 4. Write Calibrated Output FITS ---
    if out_cfg["write_fits"]:
        write_photometric_calibration_output(
            image=image,
            wcs_solution=wcs_solution,
            source_list=source_list,
            matched_idx=matched_idx,
            color_corrected_idx=colored_idx,
            output_fits=input_fits,
            zero_point=zp,
            zero_point_uncertainty=unc,
            catalog=phot_cfg["catalog"],
            obs_band=phot_cfg["obs_band"],
            cal_band=phot_cfg["cal_band"],
            color_index=color_index,
            color_term=color_term
        )

    results = {
        "photometry": {
            "zero_point": float(zp),
            "color_term": float(color_term),
            "uncertainty": float(unc),
        },
        "sources": {
            "detected": int(len(source_list)),
            "matched": int(len(matched_idx)),
        },
        "astrometry": {
            "center_ra_deg": center_ra,
            "center_dec_deg": center_dec,
            "pixel_scale": wcs_cfg["pixel_scale"],
        }
    }

    if out_cfg["make_plots"]:
        results["plots"] = plots

    return results


def validate_and_normalize(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal validation + normalization layer.
    Keeps API robust without adding heavy dependencies.
    """
    def get_float(name, default=None, required=False):
        val = body.get(name, default)
        if val is None:
            if required:
                raise AstrometryValidationError(f"{name} is required")
            return None
        try:
            return float(val)
        except Exception:
            raise AstrometryValidationError(f"{name} must be a number")

    def get_bool(name, default=False):
        return bool(body.get(name, default))

    image_url = body.get("image_url")
    if not image_url:
        raise AstrometryValidationError("image_url is required")

    use_ra_dec = get_bool("use_ra_dec", False)
    ra = get_float("ra")
    dec = get_float("dec")

    if use_ra_dec and (ra is None or dec is None):
        raise AstrometryValidationError("ra/dec required when use_ra_dec=True")

    pixel_scale = get_float("pixel_scale")
    scale_low = get_float("scale_low")
    scale_high = get_float("scale_high")

    if pixel_scale is None:
        if scale_low is None or scale_high is None:
            raise AstrometryValidationError(
                "Provide pixel_scale OR (scale_low & scale_high)"
            )
        if scale_low > scale_high:
            raise AstrometryValidationError("scale_low must be <= scale_high")

    snr = get_float("snr_threshold", 3.0)
    aperture_radius = get_float("aperture_radius", 7.0)

    catalog = body.get("catalog", "PanSTARRS1")
    obs_band = body.get("obs_band", "g")
    cal_band = body.get("cal_band", "r")

    return_plot = get_bool("return_plot", False)
    plot_type = body.get("plot_type", "color_correction")

    return {
        "image_url": image_url,
        "wcs": {
            "ra": ra,
            "dec": dec,
            "pixel_scale": pixel_scale,
            "scale_low": scale_low,
            "scale_high": scale_high,
            "search_radius": get_float("search_radius", 2.0),
            "use_ra_dec": use_ra_dec,
        },
        "detection": {
            "snr": snr,
            "aperture_radius": aperture_radius,
        },
        "photometry": {
            "catalog": catalog,
            "obs_band": obs_band,
            "cal_band": cal_band,
        },
        "output": {
            "make_plots": return_plot,
            "write_fits": False,
        },
        "meta": {
            "return_plot": return_plot,
            "plot_type": plot_type,
        }
    }


if __name__ == "__main__":
    # --- Local Integration Testing ---
    body = {
        "image_url": "Comet_65P_Gunn_LONEOS.fits",
        "ra": 51.0,
        "dec": 17.0,
        "use_ra_dec": True,
        "pixel_scale": 2.5,
        "snr_threshold": 3.0,
        "aperture_radius": 7.0,
        "catalog": "PanSTARRS1",
        "obs_band": "g",
        "cal_band": "r",
        "return_plot": True,
        "plot_type": "color_correction",
    }

    cfg = validate_and_normalize(body)
    input_fits = "Comet_65P_Gunn_LONEOS.fits"

    if os.path.exists(input_fits):
        results = run_pipeline(input_fits=input_fits, user_config=cfg)
        print("\n=== REFACTORED PIPELINE PIPELINE SUCCESS ===")
        print(results)
    else:
        print(f"Skipping local run execution: '{input_fits}' file not found.")