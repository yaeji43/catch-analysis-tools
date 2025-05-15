import numpy as np
import pytest
from pytest import approx

from ..background import *

def test_global_subtraction():
    data = np.ones((100,100))
    data[40:50,40:50] = 5*data[40:50,40:50]
    data_sub, bkg = global_subtraction(data)

    errors = []
    # test some basic values checked manually against above image values
    assert np.mean(data_sub) == 0.04
    assert np.mean(bkg.background) == 1.0
    

def test_get_background():
    data = np.ones((100,100))
    data[40:50,40:50] = 5*data[40:50,40:50]
    bkg = get_background(data)
    assert np.mean(bkg.background) == 1.0

def test_calc_annulus_bkg():
    import photutils.datasets
    noise = photutils.datasets.make_noise_image((100,100), distribution='gaussian', mean=5, stddev=1, seed=1)
    bkg_median, bkg_var, annulus_aperture = calc_annulus_bkg(noise,(50,50),1,40)
    assert approx([bkg_median,bkg_var]) == [4.990637730701752, 0.9217670741324576]
