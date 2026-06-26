import os
from typing import Any, Dict
import fitsio
import numpy as np
import sep
import tempfile
import urllib.request
from urllib.parse import urlparse


from catch_analysis_tools.calibration.astrometry import run_astrometry_calibration


class AstrometryValidationError(ValueError):
    pass


class AstrometrySolveError(RuntimeError):
    pass


def _get_float(body: Dict[str, Any], key: str, default=None):
    value = body.get(key, default)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AstrometryValidationError(f"{key} must be a number") from exc


def _get_bool(body: Dict[str, Any], key: str, default=False):
    value = body.get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y"}

    return bool(value)

def materialize_input_fits(image_url: str) -> str:
    """
    Convert a local FITS path or remote FITS URL into a clean local FITS path.
    """
    parsed = urlparse(image_url)

    if parsed.scheme in {"http", "https", "ftp"}:
        tmpdir = tempfile.mkdtemp(prefix="catch_astrometry_")
        local_path = os.path.join(tmpdir, "input.fits")

        urllib.request.urlretrieve(image_url, local_path)
        return local_path

    return image_url

def validate_and_normalize_astrometry(body: Dict[str, Any]) -> Dict[str, Any]:
    image_url = body.get("image_url")

    if not image_url:
        raise AstrometryValidationError("image_url is required")

    use_ra_dec = _get_bool(body, "use_ra_dec", False)

    ra = _get_float(body, "ra")
    dec = _get_float(body, "dec")

    if use_ra_dec and (ra is None or dec is None):
        raise AstrometryValidationError(
            "ra and dec are required when use_ra_dec=True"
        )

    pixel_scale = _get_float(body, "pixel_scale", 2.5)
    scale_low = _get_float(body, "scale_low")
    scale_high = _get_float(body, "scale_high")

    if pixel_scale is None:
        if scale_low is None or scale_high is None:
            raise AstrometryValidationError(
                "Provide pixel_scale or both scale_low and scale_high"
            )
        if scale_low > scale_high:
            raise AstrometryValidationError("scale_low must be <= scale_high")

    return {
        "image_url": image_url,
        "ra": ra,
        "dec": dec,
        "use_ra_dec": use_ra_dec,
        "pixel_scale": pixel_scale,
        "scale_low": scale_low,
        "scale_high": scale_high,
        "search_radius": _get_float(body, "search_radius", 2.0),
        "snr_threshold": _get_float(body, "snr_threshold", 3.0),
    }


def run_astrometry_pipeline(body=None, **kwargs) -> Dict[str, Any]:
    """
    OpenAPI/Connexion endpoint for /astrometry.

    Compatible with:
        catch_analysis_tools.calibration.astrometry.run_astrometry_calibration
    """
    if body is None:
        body = kwargs

    cfg = validate_and_normalize_astrometry(body)
    input_fits = materialize_input_fits(cfg["image_url"])

    image = fitsio.read(input_fits).astype(np.float32)
    bkg = sep.Background(image)
    bkg_err = float(bkg.globalrms)

    ra_deg = cfg["ra"] if cfg["use_ra_dec"] else None
    dec_deg = cfg["dec"] if cfg["use_ra_dec"] else None

    try:
        astrom_res = run_astrometry_calibration(
            input_fits=input_fits,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            bkg_err=bkg_err,
            pixel_scale=cfg["pixel_scale"],
            snr=cfg["snr_threshold"],
            output_fits=None,
        )
    except Exception as exc:
        raise AstrometrySolveError(f"Astrometry calibration failed: {exc}") from exc

    source_list = astrom_res["source_list"]
    wcs_solution = astrom_res["wcs_solution"]
    output_fits = astrom_res["output_fits"]

    ny, nx = image.shape
    center_world = wcs_solution.pixel_to_world(nx / 2.0, ny / 2.0)

    return {
        "wcs_image_url": output_fits,
        "sources_detected": int(len(source_list)),
        "center_ra_deg": float(center_world.ra.deg),
        "center_dec_deg": float(center_world.dec.deg),
        "pixel_scale": float(cfg["pixel_scale"]),
    }