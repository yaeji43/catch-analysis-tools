import argparse
import os
import subprocess

import fitsio
import numpy as np
import pandas as pd
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


def run_solve_field(
    input_fits, output_wcs, pixel_scale, Ra_deg, Dec_deg, scale_units="arcsecperpix"
):
    """
    Execute the `solve-field` command to compute a WCS solution.


    Parameters
    ----------
    input_fits : str
        Path to the input FITS image.

    output_wcs : str
        Path for the output WCS solution file.

    pixel_scale : float
        Approximate pixel scale (e.g., arcsec/pixel).

    scale_units : str, optional
        Units for pixel scale (default is "arcsecperpix").


    Returns
    -------
    success : bool
        True if the solve-field command succeeded or file already exists.

    """

    if os.path.exists(output_wcs):
        print(
            f"Output file '{output_wcs}' already exists. "
            "Skipping solve-field execution."
        )
        return True

    config_file = os.environ.get("ASTROMETRY_CONFIG")
    if config_file is None:
        raise RuntimeError(
            "ASTROMETRY_CONFIG is not set. This is required to run solve-field."
        )

    command = [
        "solve-field",
        "--overwrite",
        "--config", config_file,
        "--scale-units", scale_units,
        "--scale-low", str(pixel_scale * 0.5),
        "--scale-high", str(pixel_scale * 2.0),
        "--downsample", "1",
    ]

    if Ra_deg is not None and Dec_deg is not None:
        command.extend([
            "--ra", str(Ra_deg),
            "--dec", str(Dec_deg),
            "--radius", "2",
        ])

    command.append(input_fits)

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"solve-field failed: {e}")


def find_sources(image_sub, bkg_err, snr, aperture_radius=7.0):
    """
    Detect sources in an image using SEP background subtraction and extraction.


    Parameters
    ----------
    image_sub : array_like
        2D numpy array after background subtraction (cleaned image).

    bkg_err : float or array_like
        Background noise estimate (global RMS or per‐pixel error map).

    snr : float
        Minimum signal-to-noise ratio threshold for source extraction.

    aperture_radius : float, optional
        Radius of the circular aperture in pixels for flux summation (default is 7.0).


    Returns
    -------
    source_list : pd.DataFrame
        Table of detected sources with aperture photometry columns.

    image_sub : np.ndarray
        Background-subtracted image array.

    """

    sep.set_sub_object_limit(500)
    sources = sep.extract(image_sub, thresh=snr, err=bkg_err, deblend_nthresh=16)
    source_list = pd.DataFrame(sources)
    flux, flux_err, _ = sep.sum_circle(
        image_sub, source_list["x"], source_list["y"], aperture_radius, err=bkg_err
    )
    source_list["aperture_sum"] = flux
    source_list["aperture_err"] = flux_err
    source_list = source_list[source_list["aperture_sum"] > 0].reset_index(drop=True)
    return source_list, image_sub


def load_wcs(output_wcs):
    """
    Load a WCS solution from a FITS file header.


    Parameters
    ----------
    output_wcs : str
        Path to the FITS file containing the WCS header from astrometry.net().


    Returns
    -------
    wcs_solution : astropy.wcs.WCS
        World coordinate system solution object.

    """

    if not os.path.exists(output_wcs):
        raise FileNotFoundError(f"WCS file not found: {output_wcs}")
    with fits.open(output_wcs) as hdul:
        wcs_solution = WCS(hdul[0].header)
    return wcs_solution


def retrieve_sources(source_list, wcs_solution):
    """
    Convert pixel coordinates to sky coordinates using a WCS.


    Parameters
    ----------
    source_list : pd.DataFrame
        Table with 'x' and 'y' pixel positions of detected sources.

    wcs_solution : astropy.wcs.WCS
        World coordinate system solution object.


    Returns
    -------
    source_list : pd.DataFrame
        Updated table including 'RA' and 'Dec' columns in degrees.

    sky_coords : astropy.coordinates.SkyCoord
        SkyCoord object with celestial coordinates of sources.

    """
    world = wcs_solution.pixel_to_world(
    np.asarray(source_list["x"], dtype=float),
    np.asarray(source_list["y"], dtype=float),
    )
    source_list["RA"] = world.ra.deg
    source_list["Dec"] = world.dec.deg
    sky_coords = SkyCoord(source_list['RA'], source_list['Dec'], unit='deg')
    return source_list, sky_coords


def cleanup_files(file_base):
    """
    Remove temporary files generated during the processing pipeline.


    Parameters
    ----------
    file_base : str
        Base filename (without extension) for the files to remove.


    Returns
    -------
    None

    """

    extensions = [
        ".axy",
        ".corr",
        ".match",
        ".new",
        ".rdls",
        ".solved",
        "-ngc.png",
        "-objs.png",
        "-indx.png",
        "-indx.xyls",
    ]
    for ext in extensions:
        fname = f"{file_base}{ext}"
        if os.path.exists(fname):
            os.remove(fname)


def write_astrometry_output(
    image,
    wcs_solution,
    source_list,
    output_fits,
):
    """
    Write an astrometrically calibrated FITS file.

    This writes WCS information and detected source coordinates only. It
    intentionally does not write photometric calibration metadata such as zero
    point, color term, reference catalog, or reference filter.

    """

    image_arr = np.asarray(image)

    primary_hdu = fits.PrimaryHDU(
        data=image_arr,
        header=wcs_solution.to_header(),
    )

    source_list_clean = source_list.map(
        lambda x: x.filled(np.nan) if hasattr(x, "filled") else x
    )

    detected_hdu = fits.BinTableHDU(
        Table.from_pandas(source_list_clean),
        name="DETECTED_SOURCES",
    )

    hdul = fits.HDUList([primary_hdu, detected_hdu])
    hdul.writeto(output_fits, overwrite=True)


def run_astrometry_calibration(
    input_fits,
    ra_deg,
    dec_deg,
    bkg_err,
    pixel_scale=1.86,
    snr=7.0,
    output_fits=None,
):
    """
    Run astrometric calibration only.
    """

    file_base = os.path.splitext(input_fits)[0]

    if output_fits is None:
        output_fits = f"{file_base}_astrometry.fits"

    output_wcs = f"{file_base}.wcs"

    image = fitsio.read(input_fits).astype(np.float32)

    if run_solve_field(input_fits, output_wcs, pixel_scale, ra_deg, dec_deg):
        wcs_solution = load_wcs(output_wcs)
    else:
        raise RuntimeError("solve-field did not produce a WCS solution.")

    bkg = sep.Background(image)
    image_sub = image - bkg.back()
    source_list, image_sub = find_sources(image_sub, bkg.globalrms, snr)
    source_list, sky_coords = retrieve_sources(source_list, wcs_solution)

    write_astrometry_output(
        image=image,
        wcs_solution=wcs_solution,
        source_list=source_list,
        output_fits=output_fits,
    )

    cleanup_files(file_base)

    return {
        "output_fits": output_fits,
        "output_wcs": output_wcs,
        "source_list": source_list,
        "sky_coords": sky_coords,
        "wcs_solution": wcs_solution,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Astrometric calibration for a FITS image."
    )

    parser.add_argument(
        "input_fits",
        help="Path to input FITS image.",
    )

    parser.add_argument(
        "--Ra",
        type=float,
        required=True,
        help="Initial RA estimate in degrees.",
    )

    parser.add_argument(
        "--Dec",
        type=float,
        required=True,
        help="Initial Dec estimate in degrees.",
    )

    parser.add_argument(
        "--bkg_err",
        type=float,
        required=True,
        help="Global background RMS.",
    )

    parser.add_argument(
        "--pixel_scale",
        type=float,
        default=1.86,
        help="Pixel scale in arcsec/pixel. Default: 1.86.",
    )

    parser.add_argument(
        "--snr",
        type=float,
        default=7.0,
        help="Detection S/N threshold. Default: 7.0.",
    )

    parser.add_argument(
        "--output_fits",
        default=None,
        help="Output astrometrically calibrated FITS file.",
    )

    args = parser.parse_args()

    try:
        result = run_astrometry_calibration(
            input_fits=args.input_fits,
            ra_deg=args.Ra,
            dec_deg=args.Dec,
            bkg_err=args.bkg_err,
            pixel_scale=args.pixel_scale,
            snr=args.snr,
            output_fits=args.output_fits,
        )

        print(f"Astrometric calibration complete: {result['output_fits']}")

    except Exception as exc:
        raise SystemExit(f"Astrometric calibration failed: {exc}")
import os
from typing import Dict, Any
import numpy as np
import fitsio
import sep
from astropy.io import fits
from catch_analysis_tools.calibration.astrometry import run_astrometry_calibration

class AstrometryValidationError(Exception):
    pass

def validate_and_normalize_astrometry(body: Dict[str, Any]) -> Dict[str, Any]:
    def get_float(key: str, default: Any = None) -> Any:
        val = body.get(key, default)
        return float(val) if val is not None else None

    def get_bool(key: str, default: bool = False) -> bool:
        val = body.get(key, default)
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    image_url = body.get("image_url")
    if not image_url:
        raise AstrometryValidationError("image_url is required")

    return {
        "image_url": image_url,
        "ra": get_float("ra"),
        "dec": get_float("dec"),
        "pixel_scale": get_float("pixel_scale", 2.5),
        "scale_low": get_float("scale_low"),
        "scale_high": get_float("scale_high"),
        "search_radius": get_float("search_radius", 2.0),
        "use_ra_dec": get_bool("use_ra_dec", False),
        "snr_threshold": get_float("snr_threshold", 3.0),
    }

def run_pipeline(body: dict) -> dict:
    cfg = validate_and_normalize_astrometry(body)
    input_fits = cfg["image_url"]

    image = fitsio.read(input_fits).astype(np.float32)
    bkg = sep.Background(image)

    astrom_res = run_astrometry_calibration(
        input_fits=input_fits,
        ra_deg=cfg["ra"] if cfg["use_ra_dec"] else None,
        dec_deg=cfg["dec"] if cfg["use_ra_dec"] else None,
        bkg_err=bkg.globalrms,
        pixel_scale=cfg["pixel_scale"],
        snr=cfg["snr_threshold"],
        output_fits=None
    )

    source_list = astrom_res["source_list"]
    wcs_solution = astrom_res["wcs_solution"]

    ny, nx = image.shape
    center_world = wcs_solution.pixel_to_world(nx / 2.0, ny / 2.0)

    output_filename = input_fits.replace(".fits", ".wcs.fits")
    with fits.open(input_fits) as hdul:
        header = hdul[0].header
        header.update(wcs_solution.to_header())
        fits.writeto(output_filename, hdul[0].data, header, overwrite=True)

    return {
        "wcs_image_url": output_filename,
        "sources_detected": int(len(source_list)),
        "center_ra_deg": float(center_world.ra.deg),
        "center_dec_deg": float(center_world.dec.deg),
        "pixel_scale": cfg["pixel_scale"],
    }