import matplotlib
from astropy.wcs import WCS

from ...photometry import (
    define_aperture,
    do_aperture_photometry,
    get_image,
    subpixel_centroid,
)

matplotlib.use("Agg")
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import pyplot as plt


def get_world_coordinates(WCS_file, x, y):
    """
    Accepts an astropy-readable WCS object (a .wcs or fits file ) and outputs
    the world coordinates of a 0-indexed (x,y) pixel point in decimal degrees.
    """
    try:
        world_coords = WCS(fits.open(WCS_file)[0].header)
    except Exception:
        try:
            world_coords = WCS(WCS_file)
        except Exception:
            try:
                world_coords = WCS_file
            except Exception:
                raise ValueError(
                    "Could not read WCS file. Please ensure that the file is either a "
                    ".wcs file or a .fits file with a valid WCS solution in the header."
                )

    loc = world_coords.pixel_to_world(x, y)
    print(loc)
    transform_results = {"x": x, "y": y, "ra": loc.ra.deg, "dec": loc.dec.deg}
    return transform_results


def get_pixel_coordinates(WCS_file, ra, dec):
    """
    Accepts an astropy-readable WCS object (a .wcs or fits file) and outputs the
    0-indexed (x,y) pixel coordinates of an (ra,dec) point in decimal degrees.
    """
    try:
        world_coords = WCS(fits.open(WCS_file)[0].header)
    except Exception:
        try:
            world_coords = WCS(WCS_file)
        except Exception:
            try:
                world_coords = WCS_file
            except Exception:
                raise ValueError(
                    "Could not read WCS file. Please ensure that the file is either a "
                    ".wcs file or a .fits file with a valid WCS solution in the header."
                )

    sky_loc = SkyCoord(ICRS(ra=ra * u.deg, dec=dec * u.deg))
    loc = world_coords.world_to_pixel(sky_loc)
    x = loc[0].item()
    y = loc[1].item()

    transform_results = {"x": x, "y": y, "ra": ra, "dec": dec}
    return transform_results


def centroid(file, target_x, target_y, search_radius):
    """
    Searches for source nearby to expected ephemeris location in image, assumes
    that astrometry solution has been rerun and that both the cutout .fits file
    and the redone .wcs file exist.


    Parameters
    ----------
    file : string
        Base filename (without .fits or .wcs extension) taken from
        CATCH-generated query URL.

    target_x : float
        Target x pixel location, to be used as initial guess for the object.

    target_y : float
        Target y pixel ephemeris location, to be used as initial guess for the
        object.

    search_radius : float
        Radius in pixels of search area (centered on (x,y) ) for centroiding


    Returns
    -------
    search_results : array_like

    figname : string
        Output plot showing the default aperture + annulus extraction onto the
        cutout image.

    """

    img, header = get_image(file)

    cent_pix = subpixel_centroid([target_x, target_y], img, search_radius)

    search_results = {
        "init_guess_x": target_x,
        "init_guess_y": target_y,
        "search_radius": search_radius,
        "cent_x": cent_pix[0],
        "cent_y": cent_pix[1],
    }

    norm = ImageNormalize(img, interval=ZScaleInterval())

    plt.figure(figsize=(8, 8))
    plt.imshow(img, norm=norm, cmap="gray_r")
    plt.scatter(
        img.shape[0] / 2, img.shape[1] / 2, s=100, marker="x", label="Image Center"
    )
    plt.scatter(target_x, target_y, s=100, marker="+", label="Initial Guess")
    plt.scatter(
        cent_pix[0],
        cent_pix[1],
        s=50,
        marker=".",
        c="yellow",
        label="Centroid Location",
    )
    plt.xlim(target_x - 20, target_x + 20)
    plt.ylim(target_y - 20, target_y + 20)
    plt.legend()

    figname = "centroid.png"
    plt.savefig(figname)
    plt.close()
    centroid_results = {"search_results": search_results, "centroid_figure": figname}

    return centroid_results


# todo: create function that takes target location, background location and
# outputs aperture objects this function can be fed the outputs of
# centroid_location


def target_extraction(body):
    """
    Searches for source nearby to expected ephemeris location in image, assumes
    that astrometry solution has been rerun and that both the cutout .fits file
    and the redone .wcs file exist.


    Parameters
    ----------
    body : dict
        Request body containing file, target_aperture_params, and
        background_aperture_params.

    """
    file = body["file"]
    target_aperture_params = body["target_aperture_params"]
    background_aperture_params = body["background_aperture_params"]

    img, header = get_image(file)

    target_aperture = define_aperture(target_aperture_params)
    background_aperture = define_aperture(background_aperture_params)

    norm = ImageNormalize(img, interval=ZScaleInterval())

    target_flux, target_fluxerr = do_aperture_photometry(
        img, target_aperture, background_aperture
    )

    # could put code to filter out frames where targ_loc and targ_cent vary by
    # more than a couple pixels here (would mean star hit)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, norm=norm, cmap="gray_r")
    target_aperture.plot(color="blue", lw=1.5, alpha=0.5)
    background_aperture.plot(color="yellow", lw=1.5)
    plt.scatter(
        target_aperture_params["position"][0],
        target_aperture_params["position"][1],
        s=100,
        marker="+",
    )
    plt.scatter(
        background_aperture_params["position"][0],
        background_aperture_params["position"][1],
        s=50,
        marker=".",
        c="yellow",
    )
    plt.xlim(
        target_aperture_params["position"][0] - 20,
        target_aperture_params["position"][0] + 20,
    )
    plt.ylim(
        target_aperture_params["position"][1] - 20,
        target_aperture_params["position"][1] + 20,
    )

    figname = "aperture_extraction.png"
    plt.savefig(figname)
    plt.close()
    extraction_results = {
        "aperture_flux": target_flux,
        "aperture_fluxerr": target_fluxerr,
        "aperture_figure": figname,
    }

    return extraction_results
