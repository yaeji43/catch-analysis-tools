import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D
from photutils.background import SExtractorBackground as SourceExtractorBackground
from photutils.segmentation import (
    detect_sources,
    detect_threshold,
)
from photutils.utils import circular_footprint


def get_background(data):
    """
    Computes and returns a global background subtraction, masking sources using
    image segmentation IDs. Does NOT return background subtracted data


    Parameters
    ----------
    data : array_like
        2D image array to compute background on


    Returns
    -------
    bkg :
        background object returned from photutils Background2D

    """

    # performs a global background subtraction, masking sources using image
    # segmentation IDs
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)

    threshold = detect_threshold(data, nsigma=2.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(data, threshold, npixels=10)

    footprint = circular_footprint(radius=10)
    mask = segment_img.make_source_mask(footprint=footprint)

    bkg_estimator = SourceExtractorBackground(sigma_clip)
    bkg = Background2D(
        data,
        (50, 50),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        mask=mask,
        bkg_estimator=bkg_estimator,
    )

    return bkg


def global_subtraction(data):
    """
    Performs a global background subtraction, masking sources using image
    segmentation IDs.


    Parameters
    ----------
    data : array_like
        2D image array to be background subtracted


    Returns
    -------
    data_sub : array_like
        Background subtracted data array

    bkg :
        background object returned from get_background(data)

    """

    bkg = get_background(data)
    data_sub = data - bkg.background_median

    return data_sub, bkg


def calc_bkg(data, background_aperture, method, sigma_clip):
    """
    Takes in an image array and aperture for background estimate. Determines
    background estimator per specified method (mean or median) and variance of
    pixels, with optional sigma clipping.


    Parameters
    ----------
    data : array_like
        image data to be used for photometry

    position : array_like
        [x,y] pixel location to define center CircularAnnulus object

    inner_r : float
        Distance (in pixels) from annulus center to the inner edge

    outer_r : float
        Distance (in pixels) from annulus center to the outer edge


    Returns
    -------
    bkg_estimator : float
        Estimate of true background level in background aperture, per "method"

    bkg_var : float
        Variance of pixels within the background aperture. If sigma clipped,
        square of clipped stddev. Otherwise, as square of range of values from
        50-16th percentiles.

    """

    background_mask = background_aperture.to_mask(method="center")

    background_data = background_mask.multiply(data)
    background_data_1d = background_data[background_mask.data > 0]

    # Todo: Add flexibility for the stats for computing background. Include
    # rectangular aperture.
    if sigma_clip is not None:
        bkg_mean, bkg_median, bkg_stddev = sigma_clipped_stats(
            background_data_1d, sigma=sigma_clip, maxiters=10
        )
        bkg_var = bkg_stddev**2
    else:
        bkg_median = np.nanmedian(background_data_1d)
        bkg_mean = np.nanmean(background_data_1d)
        # robust approximation via https://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
        bkg_var = np.square(
            np.percentile(background_data_1d, 50)
            - np.percentile(background_data_1d, 16)
        )
        bkg_stddev = np.sqrt(bkg_var)

    if method == "mean":
        bkg_estimator = bkg_mean
    elif method == "median":
        bkg_estimator = bkg_median
    else:
        raise ValueError("Method must be 'mean' or 'median'")

    # plt.figure(figsize=(8,8))
    # optional code to check the standard deviation estimate of the background
    # plt.axvspan(bkg_estimator-bkg_stddev,bkg_estimator+bkg_stddev,alpha=0.5)
    # plt.axvline(bkg_estimator,color='red')
    # plt.hist(annulus_data_1d)

    return bkg_estimator, bkg_var
