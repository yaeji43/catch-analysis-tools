import os

import requests
from astropy.io import fits

from ...background import global_subtraction
from ...photometry import get_image


def download_file(url):
    """
    Downloads a file from a given URL and saves it locally, returning the base
    filename (without extension).


    Parameters:
    ----------
    url: str
        URL to the file to be downloaded


    Returns:
    -------
    file_base: str
        Base filename (without extension) of the downloaded file

    """
    response = requests.get(url)
    if "content-disposition" in response.headers:
        content_disposition = response.headers["content-disposition"]
        filename = content_disposition.split("filename=")[1]
    else:
        filename = url.split("/")[-1]
    cleaned_filename = filename.replace(" ", "")
    cleaned_filename = cleaned_filename.replace('"', "")
    with open(cleaned_filename, mode="wb") as file:
        file.write(response.content)
        print(cleaned_filename)
    print(f"Downloaded file {filename} as {cleaned_filename}")

    return os.path.splitext(cleaned_filename)[0]


def perform_median_subtraction(url):
    """
    Takes a cutout URL from CATCH, opens the file, performs a median background
    subtraction after masking sources. Returns background subtracted image.

    Parameters:
    ----------
    url: str
        URL to CATCH cutout image


    Returns:
    -------
    subt_fname: str
        Filename of background subtracted image saved locally

    """

    # Download file and get file base name
    file_base = download_file(url)
    # Get image data
    data, header = get_image(file_base + ".fits")

    # Perform global background subtraction
    data_sub, bkg = global_subtraction(data)
    header["BKG_MED"] = bkg.background_median

    # Save background subtracted image
    subtract_fname = file_base + "_subtracted.fits"

    fits.writeto(subtract_fname, data_sub, header, overwrite=True)

    return subtract_fname
