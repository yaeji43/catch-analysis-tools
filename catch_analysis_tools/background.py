import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from photutils.segmentation import detect_sources, detect_threshold, make_2dgaussian_kernel, SourceFinder, SourceCatalog, make_2dgaussian_kernel
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import SExtractorBackground as SourceExtractorBackground
from photutils.background import Background2D, MedianBackground
from photutils.utils import circular_footprint
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord, ICRS
from astropy.wcs import WCS
from photutils.aperture import aperture_photometry, CircularAnnulus, CircularAperture
from photutils.centroids import centroid_quadratic, centroid_sources, centroid_com
from astropy import units as u

def get_background(data):

    """computes and returns a global background subtraction, masking sources using image segmentation IDs.
       Does NOT return background subtracted data


    Parameters
    ----------
    data : array_like
        2D image array to compute background on

   

    Returns
    -------
    bkg :
        background object returned from photutils Background2D


    """

    # performs a global background subtraction, masking sources using image segmentation IDs
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    
    threshold = detect_threshold(data, nsigma=2.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(data, threshold, npixels=10)

    footprint = circular_footprint(radius=10)
    mask = segment_img.make_source_mask(footprint=footprint)
     
    
    bkg_estimator = SourceExtractorBackground(sigma_clip)
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, mask=mask, bkg_estimator=bkg_estimator)
    

    return bkg

def global_subtraction(data):
    """performs a global background subtraction, masking sources using image segmentation IDs


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
    data_sub = data-bkg.background_median

    return data_sub, bkg    

def calc_annulus_bkg(data,position,inner_r,outer_r):
    
    """ Takes in an image array and position + inner/outer radii for a circular annulus
        Computes and returns background median, variance from pixels within this annulus
     
        Currently exports the annulus object, too, although this might not be necessary



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
    bkg_median : float
                 median value of pixels within the annulus
    
    bkg_var : float
              variance of pixels within the annulus, as square of range of values from 50-16th percentiles

    annulus_aperture : 
                       Photutils annulus aperture object with the specified size, location parameters


    """
    annulus_aperture = CircularAnnulus(position, r_in=inner_r, r_out=outer_r)
    annulus_mask = annulus_aperture.to_mask(method='center')
    
    annulus_data = annulus_mask.multiply(data)
    annulus_data_1d = annulus_data[annulus_mask.data > 0]
    
    
    
    bkg_median = np.nanmedian(annulus_data_1d)
    # robust approximation via https://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
    bkg_var = np.square(np.percentile(annulus_data_1d,50)-np.percentile(annulus_data_1d,16))
    
    # optional code to check the standard deviation estimate of the background
    plt.axvspan(bkg_median-np.sqrt(bkg_var),bkg_median+np.sqrt(bkg_var),alpha=0.5)
    plt.axvline(bkg_median)
    plt.hist(annulus_data_1d)
    
    
    return bkg_median, bkg_var, annulus_aperture
