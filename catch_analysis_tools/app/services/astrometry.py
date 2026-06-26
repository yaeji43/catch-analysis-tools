import base64
import io
import os
import subprocess
from copy import deepcopy
from typing import Any

import calviacat as cvc
import fitsio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

matplotlib.use("Agg")


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
    },
}


def merge_config(user_config, default_config):
    result = deepcopy(default_config)
    for k, v in user_config.items():
        if isinstance(v, dict) and k in result:
            result[k] = merge_config(v, result[k])
        else:
            result[k] = v
    return result


def run_solve_field(input_fits, output_wcs, wcs_cfg):
    """
    Execute the `solve-field` command to compute a WCS solution.
    """
    if os.path.exists(output_wcs):
        return True

    if wcs_cfg["use_ra_dec"]:
        if wcs_cfg.get("ra") is None or wcs_cfg.get("dec") is None:
            raise RuntimeError("RA/Dec missing while use_ra_dec=True")

    pixel_scale = wcs_cfg["pixel_scale"]
    scale_low = wcs_cfg["scale_low"] or pixel_scale * 0.5
    scale_high = wcs_cfg["scale_high"] or pixel_scale * 2.0

    config_file = os.environ.get("ASTROMETRY_CONFIG")
    if config_file is None:
        raise RuntimeError(
            "ASTROMETRY_CONFIG is not set. This is required to run solve-field."
        )

    command = [
        "solve-field",
        "--overwrite",
        "--config",
        config_file,
        "--fits-image",
        "--wcs",
        output_wcs,
        "--no-plots",
    ]

    # --- RA/Dec only if enabled ---
    if wcs_cfg["use_ra_dec"]:
        command += [
            "--ra",
            str(wcs_cfg["ra"]),
            "--dec",
            str(wcs_cfg["dec"]),
        ]

    command += [
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        str(scale_low),
        "--scale-high",
        str(scale_high),
    ]

    if wcs_cfg["use_ra_dec"]:
        command += [
            "--radius",
            str(int(wcs_cfg["search_radius"])),
        ]

    command += [
        "--downsample",
        "1",
        input_fits,
    ]

    subprocess.run(command, check=True)
    if not os.path.exists(output_wcs):
        raise AstrometrySolveError(
            f"solve-field did not produce a WCS solution: {output_wcs}"
        )
    return True


def find_sources(image, det_cfg):
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
    bkg = sep.Background(image)
    image_sub = image - bkg.back()

    sep.set_sub_object_limit(500)

    sources = sep.extract(
        image_sub, thresh=det_cfg["snr"], err=bkg.globalrms, deblend_nthresh=16
    )

    source_list = pd.DataFrame(sources)

    flux, flux_err, _ = sep.sum_circle(
        image_sub,
        source_list["x"],
        source_list["y"],
        det_cfg["aperture_radius"],
        err=bkg.globalrms,
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
        wcs_solution = WCS(hdul[0].header, relax=True)
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
    world = wcs_solution.pixel_to_world(source_list["x"], source_list["y"])
    source_list["RA"] = [c.ra.deg for c in world]
    source_list["Dec"] = [c.dec.deg for c in world]
    sky_coords = SkyCoord(source_list["RA"], source_list["Dec"], unit="deg")
    return source_list, sky_coords


def calibrate_photometry(sky_coords, source_list, phot_cfg):
    """
    Calibrate instrumental magnitudes against a Pan-STARRS1 catalog.


    Parameters
    ----------
    sky_coords : astropy.coordinates.SkyCoord
        Celestial coordinates of detected sources.

    source_list : pd.DataFrame
        Table of detected sources containing 'aperture_sum'.

    catalog : str, optional
        Name of the photometric catalog class in calviacat (default 'PanSTARRS1').

    obs_band : str, optional
        Filter of the observed image (used for labeling and color index only;
        default: 'obs_band').

    cal_band : str, optional
        Reference catalog filter for color term (e.g. 'g', 'r', 'i'; default 'g').


    Returns
    -------
    calibration : dict
        Dictionary with keys:
        - zp           : float, zero-point magnitude
        - C            : float, color-term coefficient
        - unc          : float, uncertainty of zero-point
        - m            : array_like, calibrated magnitudes in the observed band
        - m_inst       : array_like, instrumental magnitudes
        - obs_band     : str, label of the observed band
        - cal_band     : str, same as input cal_band
        - color_mags   : array_like, color indices (obs_band - cal_band)
        - color_index  : str, the color string used (e.g. 'r-g')
        - objids       : array_like, matched catalog object IDs
        - distances    : array_like, matching distances

    """
    catalog = phot_cfg["catalog"]
    obs_band = phot_cfg["obs_band"]
    cal_band = phot_cfg["cal_band"]

    color_index = f"{obs_band}-{cal_band}"

    CatalogClass = getattr(cvc, catalog)
    ref = CatalogClass("cat.db")

    results = ref.search(sky_coords)
    if len(results[0]) < 500:
        ref.fetch_field(sky_coords)

    objids, distances = ref.xmatch(sky_coords)

    m_inst = -2.5 * np.log10(source_list["aperture_sum"].values)

    zp, C, unc, m_cal, color_mags, _ = ref.cal_color(
        objids,
        m_inst,
        cal_band,
        color_index,
    )

    return {
        "zp": zp,
        "C": C,
        "unc": unc,
        "m": m_cal,
        "m_inst": m_inst,
        "obs_band": obs_band,
        "cal_band": cal_band,
        "color_mags": color_mags,
        "color_index": color_index,
        "objids": objids,
        "distances": distances,
    }


def plot_color_correction(color_mags, m, m_inst, C, zp, color_index: str):
    """
    Plot the relation between instrumental and calibrated magnitudes.

    Parameters
    ----------
    color_mags : array_like
        Color indices (obs_band - cal_band) of matched stars.
    m : array_like
        Calibrated magnitudes from reference catalog.
    m_inst : array_like
        Instrumental magnitudes measured.
    C : float
        Color term coefficient.
    zp : float
        Zero-point magnitude.
    color_index : str
        Label for the color axis (e.g. 'r-g').

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis objects for the plot.
    """
    fig, ax = plt.subplots()
    ax.scatter(color_mags, m - m_inst, marker=".")
    x = np.linspace(0, 1.5, 100)
    ax.plot(x, C * x + zp, color="red", label=f"$m = C\\times({color_index}) + ZP$")
    ax.set_xlabel(f"${color_index}$ (mag)")
    ax.set_ylabel(r"$m - m_{\mathrm{inst}}$ (mag)")
    plt.tight_layout()
    return fig, ax


def plot_image(telescope_image_sub, source_list, matched_idx, colored_idx):
    """
    Overlay detected and matched sources on the background-subtracted image.

    Parameters
    ----------
    telescope_image_sub : np.ndarray
        Background-subtracted image array.
    source_list : pd.DataFrame
        Table of detected sources with 'x' and 'y' pixel positions.
    matched_idx : array_like
        Indices of matched catalog sources in source_list.
    colored_idx : array_like
        Indices of sources selected for color correction.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis objects for the plot.
    """
    fig, ax = plt.subplots()
    m, s = np.mean(telescope_image_sub), np.std(telescope_image_sub)
    im = ax.imshow(
        telescope_image_sub, interpolation="nearest", origin="lower", cmap="gray"
    )
    im.set_clim(vmin=m - s, vmax=m + s)
    fig.colorbar(im, ax=ax)
    ax.plot(
        source_list["x"],
        source_list["y"],
        "+",
        markersize=5,
        label="Detected",
        color="red",
    )
    ax.plot(
        source_list["x"].iloc[matched_idx],
        source_list["y"].iloc[matched_idx],
        "o",
        markersize=10,
        color="blue",
        markerfacecolor="none",
        label="Matched",
    )
    ax.plot(
        source_list["x"].iloc[colored_idx],
        source_list["y"].iloc[colored_idx],
        "o",
        markersize=15,
        color="yellow",
        markerfacecolor="none",
        label="Selected for Color Corr",
    )
    ax.legend()

    return fig, ax


def create_header(
    image,
    wcs_solution,
    zp,
    unc,
    source_list,
    matched_idx,
    colored_idx,
    input_fits,
    cal_band: str,
    catalog: str,
    obj_band: str,
):
    """
    Create and write a FITS file with calibrated header and source tables.

    Parameters
    ----------
    image : array_like
        Original 2D image data array.
    wcs_solution : astropy.wcs.WCS
        WCS solution for the image.
    zp : float
        Zero-point magnitude.
    unc : float
        Uncertainty of the zero-point.
    source_list : pd.DataFrame
        Table of detected sources.
    matched_idx : array_like
        Indices for matched catalog sources.
    colored_idx : array_like
        Indices for color-correction sources.
    input_fits : str, optional
        Filename for the input FITS file.

    Returns
    -------
    None
    """
    image_arr = np.asarray(image)
    primary_hdu = fits.PrimaryHDU(data=image_arr, header=wcs_solution.to_header())
    primary_hdu.header["ZP"] = zp
    primary_hdu.header["ZP_STD"] = unc
    primary_hdu.header["SUV_FLT"] = cal_band
    primary_hdu.header["REF_CATA"] = catalog
    primary_hdu.header["REF_FLT"] = obj_band
    primary_hdu.header["CAT_COR"] = f"{cal_band}-{obj_band}"
    source_list_clean = source_list.map(
        lambda x: x.filled(np.nan) if hasattr(x, "filled") else x
    )
    detected_hdu = fits.BinTableHDU(
        Table.from_pandas(source_list_clean), name="DETECTED_SOURCES"
    )
    if not source_list_clean.empty:
        matched_hdu = fits.BinTableHDU(
            Table.from_pandas(
                source_list_clean.iloc[matched_idx].reset_index(drop=True)
            ),
            name="SELECTED_STARS",
        )
        colored_hdu = fits.BinTableHDU(
            Table.from_pandas(
                source_list_clean.iloc[colored_idx].reset_index(drop=True)
            ),
            name="START_COLOR_CORRECTION",
        )
    else:
        matched_hdu = fits.BinTableHDU(name="SELECTED_STARS")
        colored_hdu = fits.BinTableHDU(name="START_COLOR_CORRECTION")
    hdul = fits.HDUList([primary_hdu, detected_hdu, matched_hdu, colored_hdu])
    hdul.writeto(input_fits, overwrite=True)


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


def run_pipeline(input_fits: str, user_config: dict) -> dict:
    config = merge_config(user_config, DEFAULT_CONFIG)

    wcs_cfg = config["wcs"]
    det_cfg = config["detection"]
    phot_cfg = config["photometry"]
    out_cfg = config["output"]

    file_base = os.path.splitext(input_fits)[0]
    image = fitsio.read(input_fits).astype(np.float32)
    output_wcs = f"{file_base}.wcs"

    # --- WCS solution ---
    run_solve_field(input_fits, output_wcs, wcs_cfg)

    wcs_solution = load_wcs(output_wcs)

    ny, nx = image.shape
    center_world = wcs_solution.pixel_to_world(nx / 2.0, ny / 2.0)
    center_ra = float(center_world.ra.deg)
    center_dec = float(center_world.dec.deg)

    # --- Source detection ---
    source_list, telescope_image_sub = find_sources(image, det_cfg)
    source_list, sky_coords = retrieve_sources(source_list, wcs_solution)

    astrometry_results = {
        "status": "success",
        "center_ra_deg": center_ra,
        "center_dec_deg": center_dec,
        "pixel_scale": wcs_cfg["pixel_scale"],
    }

    # --- Photometry ---
    try:
        calibration = calibrate_photometry(sky_coords, source_list, phot_cfg)
    except Exception as exc:
        cleanup_files(file_base)
        return {
            "status": "partial_success",
            "astrometry": astrometry_results,
            "photometry": {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "stage": "calibrate_photometry",
            },
            "sources": {
                "detected": int(len(source_list)),
                "matched": None,
            },
        }

    zp = calibration["zp"]
    C = calibration["C"]
    unc = calibration["unc"]
    m = calibration["m"]
    m_inst = calibration["m_inst"]
    color_mags = calibration["color_mags"]
    color_index = calibration["color_index"]
    objids = calibration["objids"]

    # --- Matching indices ---
    matched_idx = (
        np.where(~objids.mask)[0]
        if hasattr(objids, "mask")
        else np.arange(len(source_list))
    )
    colored_idx = (
        np.where(~color_mags.mask)[0]
        if hasattr(color_mags, "mask")
        else np.arange(len(source_list))
    )

    # --- Plotting ---
    plots = {}
    if out_cfg["make_plots"]:
        fig1, _ = plot_color_correction(color_mags, m, m_inst, C, zp, color_index)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=150, bbox_inches="tight")
        plots["color_correction"] = base64.b64encode(buf1.getvalue()).decode()
        plt.close(fig1)

        fig2, _ = plot_image(telescope_image_sub, source_list, matched_idx, colored_idx)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
        plots["image_overlay"] = base64.b64encode(buf2.getvalue()).decode()
        plt.close(fig2)

    # --- Optional FITS ---
    if out_cfg["write_fits"]:
        create_header(
            image,
            wcs_solution,
            zp,
            unc,
            source_list,
            matched_idx,
            colored_idx,
            input_fits,
            phot_cfg["obs_band"],
            phot_cfg["catalog"],
            phot_cfg["cal_band"],
        )

    cleanup_files(file_base)

    results = {
        "status": "success",
        "photometry": {
            "status": "success",
            "zero_point": float(zp),
            "color_term": float(C),
            "uncertainty": float(unc),
        },
        "sources": {
            "detected": int(len(source_list)),
            "matched": int(len(matched_idx)),
        },
        "astrometry": astrometry_results,
    }

    if out_cfg["make_plots"]:
        results["plots"] = plots

    return results


def validate_and_normalize(body: dict[str, Any]) -> dict[str, Any]:
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

    # --- Required ---
    image_url = body.get("image_url")
    if not image_url:
        raise AstrometryValidationError("image_url is required")

    # --- WCS ---
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

    # --- Detection ---
    snr = get_float("snr_threshold", 3.0)
    aperture_radius = get_float("aperture_radius", 7.0)

    # --- Photometry ---
    catalog = body.get("catalog", "PanSTARRS1")
    obs_band = body.get("obs_band", "g")
    cal_band = body.get("cal_band", "r")

    # --- Output ---
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
        },
    }


# ## when testing the code locally
if __name__ == "__main__":
    # --- Simulated API request body ---
    body = {
        "image_url": "Comet_65P_Gunn_LONEOS.fits",  # not used in local test
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

    # --- Validate + normalize (same as API path) ---
    cfg = validate_and_normalize(body)

    # --- Use local FITS file instead of downloading ---
    input_fits = "Comet_65P_Gunn_LONEOS.fits"

    # --- Run pipeline ---
    results = run_pipeline(
        input_fits=input_fits,
        user_config=cfg,
    )

    # --- Print summary ---
    print("\n=== RESULTS ===")
    print(results)

    # --- Optional: visualize plot locally ---
    if cfg["output"]["make_plots"]:
        import base64
        import io

        import matplotlib.pyplot as plt

        plot_type = cfg["meta"]["plot_type"]
        img_bytes = base64.b64decode(results["plots"][plot_type])

        img = plt.imread(io.BytesIO(img_bytes), format="png")
        plt.imshow(img)
        plt.axis("off")
        plt.title(plot_type)
        plt.show()
