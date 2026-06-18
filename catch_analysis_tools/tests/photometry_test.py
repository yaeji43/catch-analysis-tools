import numpy as np
import pytest
from photutils.datasets import make_wcs
from pytest import approx

from ..app.services.photometry import (
    get_image,
    get_pixel_coordinates,
    get_world_coordinates,
)
from ..background import get_background

# here are test functions for grabbing the data, doing background subtractions
# and manipulating source extractions
from ..photometry import (
    calibrated_mag,
    create_user_aperture,
    define_aperture,
    do_aperture_photometry,
    id_good_sources,
    source_instr_mag,
    subpixel_centroid,
)


@pytest.mark.remote_data
def test_image():
    test_url = "https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013"
    data, header = get_image(test_url)
    # test some basic values checked manually against above image values
    assert approx(data[10, 10]) == 1558.4028
    assert approx(header["SECZ"]) == 1.055449


@pytest.mark.remote_data
def test_id_good_sources():
    test_url = "https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013"
    data, header = get_image(test_url)
    bkg = get_background(data)
    cat = id_good_sources(data, bkg)
    assert len(cat.to_table()) == 125


def test_create_aperture():
    aperture = create_user_aperture((11, 54), 12)
    assert approx(aperture.area) == 452.3893421169302


def test_define_aperture():
    aperture = define_aperture(
        ({"shape": "Circular", "position": [11, 54], "size": 12})
    )
    assert approx(aperture.area) == 452.3893421169302


def test_subpixel_centroid():
    data = np.zeros((100, 100))
    data[40:50, 40:50] = 5 + data[40:50, 40:50]
    data[45, 45] = 100
    source = subpixel_centroid([41, 48], data, 20)
    assert approx(source) == np.array([45.0, 45.0])


def test_do_aperture_photometry():
    import photutils.datasets

    noise = photutils.datasets.make_noise_image(
        (100, 100), distribution="gaussian", mean=0, stddev=1, seed=1
    )
    noise[50, 50] = (
        noise[50, 50] + 100
    )  # give us a "source" with flux of 100 to recover
    source_aperture = define_aperture(
        {"shape": "Circular", "position": [50, 50], "size": 5}
    )
    bkg_aperture = define_aperture(
        {
            "shape": "Circular_Annulus",
            "position": [50, 50],
            "size": 5,
            "inner_r": 5,
            "outer_r": 10,
        }
    )
    source_sum, source_err = do_aperture_photometry(
        noise, source_aperture, bkg_aperture
    )
    assert approx([source_sum, source_err]) == [89.2212216869083, 14.261818958632185]


def test_get_world_coordinates():
    shape = (100, 100)
    wcs = make_wcs(shape)

    skycoord = get_world_coordinates(wcs, 42, 57)
    assert approx([skycoord["ra"], skycoord["dec"]]) == [197.89278975, -1.36561284]


def test_get_pixel_coordinates():
    shape = (100, 100)
    wcs = make_wcs(shape)

    pixcoord = get_pixel_coordinates(wcs, 197.89278975, -1.36561284)

    assert approx([np.round(pixcoord["x"]), np.round(pixcoord["y"])]) == [42.0, 57.0]


def test_source_instr_mag():
    mag = source_instr_mag(10, 1, 1)
    assert approx(mag[0] + mag[1]) == -2.38560627


def test_calibrated_mag():
    calib_mag = calibrated_mag(source_instr_mag(10, 1, 1), 22, 0.5)
    print(calib_mag)
    assert approx(calib_mag["cal_mag"] + calib_mag["cal_mag_hi_uncert"]) == 20.01291902
