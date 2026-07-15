import calviacat as cvc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


def calibrate_photometric_zero_point(
    sky_coords,
    source_list: pd.DataFrame,
    catalog: str = "PanSTARRS1",
    color_term: str = "g-r",
    cal_band: str = "r",
    catalog_db: str = "cat.db",
):
    """
    Calibrate instrumental magnitudes using a reference catalog.

    This performs photometric calibration only:
    catalog matching, instrumental magnitude calculation,
    zero-point estimation, and color-term correction.

    Parameters
    ----------
    sky_coords : astropy.coordinates.SkyCoord
        Sky coordinates of detected sources.

    source_list : pandas.DataFrame
        Source table containing aperture fluxes in the
        ``aperture_sum`` column.

    catalog : str, optional
        Name of the reference catalog class in calviacat.
        Default is ``"PanSTARRS1"``.

    obs_band : str, optional
        Observed image band label. Used to construct the color index.

    cal_band : str, optional
        Reference catalog band used for calibration.

    catalog_db : str, optional
        Local catalog database path.

    Returns
    -------
    dict
        Photometric calibration results containing zero point,
        color term, calibrated magnitudes, matched object IDs,
        and match distances.
    """
    if "aperture_sum" not in source_list.columns:
        raise ValueError("source_list must contain an 'aperture_sum' column.")

    aperture_sum = source_list["aperture_sum"].to_numpy()

    if np.any(aperture_sum <= 0):
        raise ValueError(
            "All aperture_sum values must be positive before magnitude calibration."
        )

    color_index = color_term

    try:
        CatalogClass = getattr(cvc, catalog)
    except AttributeError as exc:
        raise ValueError(f"Catalog '{catalog}' not found in calviacat.") from exc

    ref = CatalogClass(catalog_db)

    results = ref.search(sky_coords)
    if len(results[0]) < 500:
        ref.fetch_field(sky_coords)

    xmatch_result = ref.xmatch(sky_coords)

    if xmatch_result is None:
        raise RuntimeError("Photometric calibration failed: fewer than 10 catalog matches.")

    objids, distances = xmatch_result

    aperture_sum = np.asarray(source_list["aperture_sum"].values, dtype=float)
    valid_flux = np.isfinite(aperture_sum) & (aperture_sum > 0)

    objids = objids[valid_flux]
    distances = distances[valid_flux]
    aperture_sum = aperture_sum[valid_flux]

    m_inst = -2.5 * np.log10(aperture_sum)

    valid_m_inst = np.isfinite(m_inst)

    objids = objids[valid_m_inst]
    distances = distances[valid_m_inst]
    m_inst = m_inst[valid_m_inst]

    if len(m_inst) < 10:
        raise RuntimeError(
            f"Photometric calibration failed: only {len(m_inst)} valid matched sources remain after filtering."
        )

    zp, color_term, zp_unc, m_cal, color_mags, _ = ref.cal_color(
        objids,
        m_inst,
        cal_band,
        color_term,
    )

    return {
        "zp": zp,
        "color_term": color_term,
        "unc": zp_unc,
        "m": m_cal,
        "m_inst": m_inst,
        "cal_band": cal_band,
        "color_mags": color_mags,
        "color_index": color_term,
        "objids": objids,
        "distances": distances,
    }


def get_matched_indices(objids, source_count: int):
    """
    Return indices of sources matched to catalog objects.
    """
    if hasattr(objids, "mask"):
        return np.where(~objids.mask)[0]

    return np.arange(source_count)


def get_color_corrected_indices(color_mags, source_count: int):
    """
    Return indices of sources used for color correction.
    """
    if hasattr(color_mags, "mask"):
        return np.where(~color_mags.mask)[0]

    return np.arange(source_count)


def plot_color_correction(
    color_mags,
    calibrated_magnitude,
    instrumental_magnitude,
    color_term,
    zero_point,
    color_index: str,
):
    """
    Plot color correction relation for photometric calibration.
    """
    fig, ax = plt.subplots()

    ax.scatter(
        color_mags,
        calibrated_magnitude - instrumental_magnitude,
        marker=".",
    )

    x = np.linspace(0, 1.5, 100)
    ax.plot(
        x,
        color_term * x + zero_point,
        color="red",
        label=f"$m = C\\times({color_index}) + ZP$",
    )

    ax.set_xlabel(f"${color_index}$ (mag)")
    ax.set_ylabel(r"$m - m_{\mathrm{inst}}$ (mag)")
    ax.legend()

    plt.tight_layout()

    return fig, ax


def plot_photometric_matches(
    image_sub,
    source_list: pd.DataFrame,
    matched_idx,
    color_corrected_idx,
):
    """
    Overlay detected, catalog-matched, and color-corrected sources.
    """
    fig, ax = plt.subplots()

    mean_value = np.mean(image_sub)
    std_value = np.std(image_sub)

    im = ax.imshow(
        image_sub,
        interpolation="nearest",
        origin="lower",
        cmap="gray",
    )
    im.set_clim(vmin=mean_value - std_value, vmax=mean_value + std_value)

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
        source_list["x"].iloc[color_corrected_idx],
        source_list["y"].iloc[color_corrected_idx],
        "o",
        markersize=15,
        color="yellow",
        markerfacecolor="none",
        label="Selected for Color Correction",
    )

    ax.legend()

    return fig, ax


def write_photometric_calibration_output(
    image,
    wcs_solution,
    source_list: pd.DataFrame,
    matched_idx,
    color_corrected_idx,
    output_fits,
    zero_point,
    zero_point_uncertainty,
    catalog: str,
    cal_band: str,
    color_index: str,
    color_term=None,
):
    """
    Write a FITS file with photometric calibration metadata and source tables.
    """
    image_arr = np.asarray(image)

    primary_hdu = fits.PrimaryHDU(
        data=image_arr,
        header=wcs_solution.to_header(),
    )

    primary_hdu.header["ZP"] = zero_point
    primary_hdu.header["ZP_STD"] = zero_point_uncertainty
    primary_hdu.header["REF_CATA"] = catalog
    primary_hdu.header["REF_FLT"] = cal_band
    primary_hdu.header["CAT_COR"] = color_index

    if color_term is not None:
        primary_hdu.header["COLORTRM"] = color_term

    source_list_clean = source_list.map(
        lambda x: x.filled(np.nan) if hasattr(x, "filled") else x
    )

    detected_hdu = fits.BinTableHDU(
        Table.from_pandas(source_list_clean),
        name="DETECTED_SOURCES",
    )

    if not source_list_clean.empty:
        matched_hdu = fits.BinTableHDU(
            Table.from_pandas(
                source_list_clean.iloc[matched_idx].reset_index(drop=True)
            ),
            name="SELECTED_STARS",
        )

        color_hdu = fits.BinTableHDU(
            Table.from_pandas(
                source_list_clean.iloc[color_corrected_idx].reset_index(drop=True)
            ),
            name="COLOR_CORRECTION_STARS",
        )
    else:
        matched_hdu = fits.BinTableHDU(name="SELECTED_STARS")
        color_hdu = fits.BinTableHDU(name="COLOR_CORRECTION_STARS")

    hdul = fits.HDUList(
        [
            primary_hdu,
            detected_hdu,
            matched_hdu,
            color_hdu,
        ]
    )

    hdul.writeto(output_fits, overwrite=True)
