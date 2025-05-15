import numpy as np
import pytest
from pytest import approx
from photutils.datasets import make_wcs

# here are test functions for grabbing the data, doing background subtractions and manipulating source extractions

from ..photometry import *
from ..background import *

@pytest.mark.remote_data
def test_image():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    # test some basic values checked manually against above image values
    assert approx(data[10,10]) == 1558.4028
    assert approx(header['SECZ']) == 1.055449

@pytest.mark.remote_data
def test_id_good_sources():
    test_url = 'https://sbnsurveys.astro.umd.edu/api/images/urn%3Anasa%3Apds%3Agbo.ast.neat.survey%3Adata_tricam%3Ap20020121_obsdata_20020121132624c?format=fits&size=10.00arcmin&ra=177.51011&dec=15.25013'
    data, header = get_image(test_url)
    bkg = get_background(data)
    cat = id_good_sources(data,bkg)
    assert len(cat.to_table()) == 125

def test_create_aperture():
    aperture = create_user_aperture((11,54),12)
    assert approx(aperture.area) == 452.3893421169302

def test_subpixel_centroid():
    data = np.zeros((100,100))
    data[40:50,40:50] = 5+data[40:50,40:50]
    #data[45,45] = 100
    source = subpixel_centroid([41,48],data,20)
    plt.imshow(data)
    print(source)
    assert approx(source) == np.array([44.5, 44.5])

def test_do_aperture_photometry():
    import photutils.datasets
    noise = photutils.datasets.make_noise_image((100,100), distribution='gaussian', mean=0, stddev=1, seed=1)
    noise[50,50] = 100 # give us a "source" with flux of 100 to recover
    bkg_median, bkg_var, bkg_aperture = calc_annulus_bkg(noise,(50,50),5,10)
    source_aperture = create_user_aperture((50,50),5)
    source_sum, source_err = do_aperture_photometry(noise,source_aperture,bkg_median,bkg_var,bkg_aperture)
    assert approx([source_sum, source_err]) == [92.28235162232964, 15.19308011549977]

def test_get_pixel_WCS():
    shape = (100, 100)
    wcs = make_wcs(shape)

    skycoord = get_pixel_WCS(wcs,[42, 57])
    assert approx([skycoord.ra.value,skycoord.dec.value]) == [197.89278975, -1.36561284]

def test_get_WCS_pixel():
    shape = (100, 100)
    wcs = make_wcs(shape)

    x, y = get_WCS_pixel(wcs,[197.89278975, -1.36561284])

    assert all(np.round([x, y]) == [42.0, 57.0])

def test_source_instr_mag():
    mag = source_instr_mag(10,1,1)
    assert approx(mag[0] + mag[1]) == -2.38560627

def test_calibrated_mag():
    calib_mag = calibrated_mag(source_instr_mag(10,1,1),22,0.5)
    print (calib_mag)
    assert approx(calib_mag[0] + calib_mag[1]) == 20.01291902

