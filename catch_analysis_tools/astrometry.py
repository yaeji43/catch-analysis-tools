import os
import glob
import subprocess
import argparse
import numpy as np
import pandas as pd
import sep
import fitsio
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import calviacat as cvc

def run_solve_field(input_fits, output_wcs, pixel_scale, Ra_deg, Dec_deg, scale_units="arcsecperpix"):
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
        print(f"Output file '{output_wcs}' already exists. Skipping solve-field execution.")          
        return True
    
    config_file = os.environ.get("ASTROMETRY_CONFIG")
    if config_file is None:
        raise RuntimeError(
            "ASTROMETRY_CONFIG is not set. "
            "This is required to run solve-field."
        )

    command = [
        "solve-field",
        "--overwrite",
        "--config", config_file,
        "--ra", str(Ra_deg),
        "--dec", str(Dec_deg),
        "--scale-units", scale_units,
        "--scale-low", str(pixel_scale * 0.5),
        "--scale-high", str(pixel_scale * 2.0),
        "--radius", "2",
        "--downsample", "1",
        input_fits,
    ]

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
    sources = sep.extract(
        image_sub,
        thresh=snr,
        err=bkg_err,
        deblend_nthresh=16
    )
    source_list = pd.DataFrame(sources)
    flux, flux_err, _ = sep.sum_circle(
        image_sub,
        source_list['x'], source_list['y'],
        aperture_radius,
        err=bkg_err
    )
    source_list['aperture_sum'] = flux
    source_list['aperture_err'] = flux_err
    source_list = source_list[source_list['aperture_sum'] > 0].reset_index(drop=True)
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
    world = wcs_solution.pixel_to_world(source_list['x'], source_list['y'])
    source_list['RA'] = [c.ra.deg for c in world]
    source_list['Dec'] = [c.dec.deg for c in world]
    sky_coords = SkyCoord(source_list['RA'], source_list['Dec'], unit='deg')
    return source_list, sky_coords


def calibrate_photometry(
    sky_coords,
    source_list,
    catalog: str   = 'PanSTARRS1',
    obs_band: str  = 'obs_band',
    cal_band: str  = 'g'):
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
        Filter of the observed image (used for labeling and color index only; default: 'obs_band').
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
    color_index = f"{obs_band}-{cal_band}"
    
    try:
        CatalogClass = getattr(cvc, catalog)
    except AttributeError:
        raise ValueError(f"Catalog '{catalog}' not found in calviacat")
    
    ref = CatalogClass('cat.db')
    results = ref.search(sky_coords)
    if len(results[0]) < 500:
        ref.fetch_field(sky_coords)
    objids, distances = ref.xmatch(sky_coords)
    m_inst = -2.5 * np.log10(source_list['aperture_sum'].values)
    zp, C, unc, m_cal, color_mags, _ = ref.cal_color(
            objids,
            m_inst,
            cal_band,
            color_index,
        )
    return {
        'zp'          : zp,
        'C'           : C,
        'unc'         : unc,
        'm'           : m_cal,
        'm_inst'      : m_inst,
        'obs_band'    : obs_band,
        'cal_band'    : cal_band,
        'color_mags'  : color_mags,
        'color_index' : color_index,
        'objids'      : objids,
        'distances'   : distances,
    }


def plot_color_correction(
    color_mags,
    m,
    m_inst,
    C,
    zp,
    color_index: str
):
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
    ax.scatter(color_mags, m - m_inst, marker='.')
    x = np.linspace(0, 1.5, 100)
    ax.plot(x, C * x + zp, color='red', label=f'$m = C\\times({color_index}) + ZP$')
    ax.set_xlabel(f'${color_index}$ (mag)')
    ax.set_ylabel(r'$m - m_{\mathrm{inst}}$ (mag)')
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
    im = ax.imshow(telescope_image_sub, interpolation='nearest', origin='lower', cmap='gray')
    im.set_clim(vmin=m-s, vmax=m+s)
    fig.colorbar(im, ax=ax)
    ax.plot(source_list['x'], source_list['y'], '+', markersize=5, label='Detected', color='red',)
    ax.plot(source_list['x'].iloc[matched_idx], source_list['y'].iloc[matched_idx], 'o', markersize=10, color='blue', markerfacecolor='none', label='Matched')
    ax.plot(source_list['x'].iloc[colored_idx], source_list['y'].iloc[colored_idx], 'o', markersize=15, color='yellow', markerfacecolor='none', label='Selected for Color Corr')
    ax.legend()

    return fig, ax


def create_header(image, wcs_solution, zp, unc, source_list, matched_idx, colored_idx, input_fits, cal_band: str, catalog: str, obj_band: str):
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
    primary_hdu.header['ZP'] = zp
    primary_hdu.header['ZP_STD'] = unc
    primary_hdu.header['SUV_FLT']  = cal_band
    primary_hdu.header['REF_CATA'] = catalog
    primary_hdu.header['REF_FLT']  = obj_band
    primary_hdu.header['CAT_COR']  = f"{cal_band}-{obj_band}"
    source_list_clean = source_list.applymap(lambda x: x.filled(np.nan) if hasattr(x, 'filled') else x)
    detected_hdu = fits.BinTableHDU(Table.from_pandas(source_list_clean), name='DETECTED_SOURCES')
    if not source_list_clean.empty:
        matched_hdu = fits.BinTableHDU(Table.from_pandas(source_list_clean.iloc[matched_idx].reset_index(drop=True)), name='SELECTED_STARS')
        colored_hdu = fits.BinTableHDU(Table.from_pandas(source_list_clean.iloc[colored_idx].reset_index(drop=True)), name='START_COLOR_CORRECTION')
    else:
        matched_hdu = fits.BinTableHDU(name='SELECTED_STARS')
        colored_hdu = fits.BinTableHDU(name='START_COLOR_CORRECTION')
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
    extensions = ['.axy', '.corr', '.match', '.new', '.rdls', '.solved',
                  '-ngc.png', '-objs.png', '-indx.png', '-indx.xyls']
    for ext in extensions:
        fname = f"{file_base}{ext}"
        if os.path.exists(fname):
            os.remove(fname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Photometric calibration on a background-subtracted image.")
    parser.add_argument("input_fits", help="Path to background-subtracted FITS image")
    parser.add_argument("--Ra", type=float, required=True, help="RA from CATCH")
    parser.add_argument("--Dec", type=float, required=True, help="DEC from CATCH")
    parser.add_argument("--bkg_err", type=float, required=True,
                        help="Global background RMS (float, required)")
    parser.add_argument("--pixel_scale", type=float, default=1.86,
                        help="Pixel scale in arcsec/pixel (default: 1.86)")
    parser.add_argument("--snr", type=float, default=7.0,
                        help="Detection S/N threshold (default: 7.0)")
    parser.add_argument("--catalog", default="PanSTARRS1",
                        help="Photometric reference catalog (default: PanSTARRS1)")
    parser.add_argument("--obs_band", default="obs_band", help="Observed image bandpass (used for labeling only; default: 'obs_band')")
    parser.add_argument("--cal_band", default="r", help="Reference catalog bandpass (default: r)")
    args = parser.parse_args()

    input_fits  = args.input_fits
    file_base   = os.path.splitext(input_fits)[0]
    image       = fitsio.read(input_fits).astype(np.float32)
    Ra          = args.Ra
    Dec         = args.Dec
    bkg_err     = args.bkg_err
    pixel_scale = args.pixel_scale
    snr         = args.snr
    catalog     = args.catalog
    obs_band    = args.obs_band
    cal_band    = args.cal_band
 
    
    output_wcs = f"{file_base}.wcs"
    try:
        if run_solve_field(input_fits, output_wcs, pixel_scale, Ra, Dec):
            wcs_solution = load_wcs(output_wcs)
        else:
            raise RuntimeError("solve-field did not produce a WCS solution")
    except Exception as e:
        raise SystemExit(f"WCS calibration failed: {e}")

    source_list, telescope_image_sub = find_sources(image, bkg_err, snr)
    source_list, sky_coords = retrieve_sources(source_list, wcs_solution)
    calibration = calibrate_photometry(
        sky_coords,
        source_list,
        catalog=catalog,
        obs_band=obs_band,
        cal_band=cal_band,
    )
    zp     = calibration["zp"]
    C      = calibration["C"]
    unc    = calibration["unc"]
    m      = calibration["m"]
    m_inst = calibration["m_inst"]
    color_mags  = calibration['color_mags']
    color_index = calibration['color_index']
    objids      = calibration['objids']

    if hasattr(objids, "mask"):
        matched_idx = np.where(~objids.mask)[0]
    else:
        matched_idx = np.arange(len(source_list))
    if hasattr(color_mags, "mask"):
        colored_idx = np.where(~color_mags.mask)[0]
    else:
        colored_idx = np.arange(len(source_list))

    fig1, ax1 = plot_color_correction(color_mags, m, m_inst, C, zp, color_index)
    fig2, ax2 = plot_image(telescope_image_sub, source_list, matched_idx, colored_idx)
    plt.show()

    create_header(
        image,
        wcs_solution,
        zp,
        unc,
        source_list,
        matched_idx,
        colored_idx,
        input_fits,         
        obs_band,
        catalog,
        cal_band,
    )

    cleanup_files(file_base)