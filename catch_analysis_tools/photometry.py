import numpy as np
from astropy.convolution import convolve
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    RectangularAperture,
)
from photutils.centroids import centroid_quadratic, centroid_sources
from photutils.segmentation import (
    SourceCatalog,
    SourceFinder,
    make_2dgaussian_kernel,
)
from photutils.utils import circular_footprint


# here are functions for grabbing the data, doing background subtractions and
# manipulating source extractions
def get_image(url):
    """Access a cutout via a call to a CATCH URL


    Parameters
    ----------
    url : string
        CATCH-generated URL from a query.


    Returns
    -------
    data : array_like
        This is a 2D array containing the image data returned by the CATCH query

    header :
        FITS header class, from astropy.io.fits.Header

    """

    fits_hdu = fits.open(url)
    data = fits_hdu[0].data
    header = fits_hdu[0].header
    return data, header


def id_good_sources(data, bkg):
    """
    Uses a segmentation image to identify reliable sources in image that can be
    snapped to.

    Coincidentally, computes baseline photometry that could be used as a quality
    comparison user results, though this flux isn't always a good comparison as
    it often underestimates the source size


    Parameters
    ----------
    data : array_like
        2D image array to be background subtracted

    bkg :
        background object returned from get_background() or global_subtraction()


    Returns
    -------
    cat :
        Astropy Table class, from SourceCatalog output giving source locations,
        fluxes

    """

    source_threshold = bkg.background_median + 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=5, progress_bar=False)
    segment_map = finder(convolved_data, source_threshold)

    # vmax = np.percentile(np.ndarray.flatten(data), 99)
    # vmin = np.percentile(np.ndarray.flatten(data), 1)

    # make a plot to show the background subtracted frame and the resulting segment map
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    # ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=vmin,vmax=vmax)
    # ax1.set_title('Original Data')

    # ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
    #       interpolation='nearest')
    # ax2.set_title('Segmentation Image')

    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)

    return cat


def create_user_aperture(position, size):
    """Simple placeholder function for making user-selected circular apertures


    Parameters
    ----------
    position : array_like
        [x,y] location of aperture center

    size : Int
        Radius of aperture, given in pixels


    Returns
    -------
    aperture :
        Photutils circular aperture object with the specified size, location
        parameters

    """

    aperture = CircularAperture(position, r=size)
    return aperture


def define_aperture(aperture_params):
    """
    Defines an aperture from user supplied parameters, used both to create
    target and background apertures
    """

    if aperture_params["shape"] == "Circular":
        aperture = CircularAperture(
            aperture_params["position"], r=aperture_params["size"]
        )
    elif aperture_params["shape"] == "Circular_Annulus":
        aperture = CircularAnnulus(
            aperture_params["position"],
            r_in=aperture_params["inner_r"],
            r_out=aperture_params["outer_r"],
        )
    elif aperture_params["shape"] == "Rectangular":
        aperture = RectangularAperture(
            aperture_params["position"],
            aperture_params["w"],
            aperture_params["h"],
            theta=aperture_params["theta"],
        )
    else:
        aperture = 0
    return aperture


def subpixel_centroid(user_point, data, radius):
    """
    Takes in a user defined point and returns the location of the brightest
    pixel within radius pixels.


    Parameters
    ----------
    user_point : array_like
        [x,y] location of desired point, to be snapped from

    data : array_like
        image to be searched for sources to be snapped to

    radius : Int
        number of pixels within which objects can be snapped to from user_point


    Returns
    -------
    target_position: array_like
        [x,y] location of the source that is closest to user_point found in data

    """

    footprint = circular_footprint(radius)
    x, y = centroid_sources(
        data,
        user_point[0],
        user_point[1],
        footprint=footprint,
        centroid_func=centroid_quadratic,
    )

    target_position = np.array([x[0], y[0]])
    return target_position


def do_aperture_photometry(data, source_aperture, bkg_aperture):
    """
    Takes in an image, a source aperture, and outputs from the calc_annulus_bkg
    function.

    Returns the source flux (background subtracted, per-pixel background median)
    and the uncertainty as defined at the quoted link

    method='center' means pixels are either in or out, no interpolation to a
    perfect circle (in other words, areas will be in whole pixels)


    Parameters
    ----------
    data : array_like
        image data to be used for photometry

    source_aperture :
        Photutils aperture object containing desired source

    bkg_median : float
        median value of pixel background,

    bkg_var : float
        variance of pixel background, ideally from output of
        calc_annulus_background

    bkg_aperture :
        Photutils aperture object containing background, ideally as
        annulus_aperture output from calc_annulus_background


    Returns
    -------
    source_sum : float
        background subtracted flux of the targeted source

    source_err : float
        error on source flux

    """

    aperture_mask = source_aperture.to_mask(method="center")
    aperture_data = aperture_mask.multiply(data)
    aperture_sum = np.nansum(aperture_data)
    aperture_area = np.nansum(aperture_mask)

    bkg_mask = bkg_aperture.to_mask(method="center")
    bkg_area = np.nansum(bkg_mask)
    bkg_data = bkg_mask.multiply(data)
    bkg_median = np.nanmedian(bkg_data)
    bkg_var = np.nanvar(bkg_data[bkg_data != 0])
    background = bkg_median * aperture_area
    source_sum = aperture_sum - background

    # Using uncertainty as derived by https://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
    # Setting the gain g=1, N_i = 1. Assumes data has already been converted to e-
    ### TODO: FIND THE GAINS FOR EACH SURVEY
    term1 = source_sum
    term2 = (
        aperture_area + (np.pi / 2) * (np.square(aperture_area) / bkg_area)
    ) * bkg_var
    source_err = np.sqrt(term1 + term2)

    return source_sum, source_err


def load_thumbnail(url):
    """Access a cutout via a call to a CATCH URL, returning WCSobject


    Parameters
    ----------
    url : string
        CATCH-generated URL from a query.


    Returns
    -------
    data : array_like
        This is a 2D array containing the image data returned by the CATCH query

    header :
        FITS header class, from astropy.io.fits.Header

    img_WCS:
        Astropy WCS object

    """
    fits_hdu = fits.open(url)
    data = fits_hdu[0].data
    header = fits_hdu[0].header
    img_WCS = WCS(header)
    return data, header, img_WCS


def source_instr_mag(ap_flux, ap_fluxerr, exposure_time):
    """Quick function to return instrumental magnitudes from a source flux
    Does not force magnitude uncertainties to be symmetric

    To be used by calibrated_mag() function


    Parameters
    ----------
    ap_flux: float
        Flux (in counts) of source

    ap_fluxerr: float
        Flux error (in counts)

    exposure_time: float
        integration time of the frame (s)

    Returns
    -------
    instr_mag_array: array_like
        array containing source instrumental magnitude and uncertainties, as
        [Magnitude, Upper Magnitude Uncertainty, Lower Magnitude Uncertainty]

    """

    instr_mag = -2.5 * np.log10(ap_flux / exposure_time)
    instr_mag_hi = -2.5 * np.log10((ap_flux - ap_fluxerr) / exposure_time)
    instr_mag_lo = -2.5 * np.log10((ap_flux + ap_fluxerr) / exposure_time)

    instr_mag_hi_uncert = instr_mag_hi - instr_mag
    instr_mag_lo_uncert = instr_mag_lo - instr_mag

    instr_mag_array = np.array([instr_mag, instr_mag_hi_uncert, instr_mag_lo_uncert])

    return instr_mag_array


def calibrated_mag(instr_mag_array, zero_point, zero_point_uncert):
    """
    Takes in the array from source_instr_mag, converts to derived magnitude,
    propagating uncertainties from both


    Parameters
    ----------
    instr_mag_array: array_like
        From source_instr_mag() output: array containing source instrumental
        magnitude and uncertainties, as [Magnitude, Upper Magnitude Uncertainty,
        Lower Magnitude Uncertainty]

    zero_point: float
        zero point magnitude of image (ideally taken from metadata in header
        given by astrometry solution)

    zero_point_uncert: float
        zero point uncertainty (mags) of image (ideally taken from metadata in
        header given by astrometry solution)


    Returns
    -------

    calib_mag_array: array_like
        array containing source CALIBRATED magnitude and uncertainties, as
        [Magnitude, Upper Magnitude Uncertainty, Lower Magnitude Uncertainty]

    """

    calib_mag = zero_point + instr_mag_array[0]
    calib_mag_hi = np.sqrt(np.square(zero_point_uncert) + np.square(instr_mag_array[1]))
    calib_mag_lo = np.sqrt(np.square(zero_point_uncert) + np.square(instr_mag_array[2]))
    calib_mag_array = {
        "cal_mag": calib_mag,
        "cal_mag_hi_uncert": calib_mag_hi,
        "cal_mag_lo_uncert": calib_mag_lo,
    }
    return calib_mag_array
