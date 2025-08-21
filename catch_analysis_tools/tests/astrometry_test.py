import pytest
import tempfile
from unittest.mock import patch, MagicMock

from ..astrometry import *

@pytest.fixture
def synthetic_image():
    image = np.random.normal(loc=0, scale=1.0, size=(100, 100)).astype(np.float32)
    for i in range(-2, 3):
        for j in range(-2, 3):
            image[50 + i, 50 + j] += 20 * np.exp(-(i**2 + j**2) / (2 * 1.5**2))
    return image


@pytest.fixture
def synthetic_wcs():
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 100
    header['NAXIS2'] = 100
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRPIX1'] = 50.0
    header['CRPIX2'] = 50.0
    header['CRVAL1'] = 150.0
    header['CRVAL2'] = 2.0
    header['CD1_1'] = -0.000277778
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = 0.000277778
    return WCS(header)

@pytest.mark.parametrize("file_exists, should_call_run", [
    (True, False),  # WCS already exists → skip
    (False, True),  # WCS doesn't exist → run solve-field
])

def test_run_solve_field_conditional_execution(file_exists, should_call_run):
    with patch("catch_analysis_tools.astrometry.os.path.exists", return_value=file_exists) as mock_exists, \
         patch("catch_analysis_tools.astrometry.subprocess.run") as mock_run:
        
        result = run_solve_field("input.fits", "output.wcs", pixel_scale=2.0)

        assert result is True
        mock_exists.assert_called_once_with("output.wcs")
        if should_call_run:
            mock_run.assert_called_once()
        else:
            mock_run.assert_not_called()


def test_run_solve_field_raises_if_subprocess_fails():
    with patch("catch_analysis_tools.astrometry.os.path.exists", return_value=False), \
         patch("catch_analysis_tools.astrometry.subprocess.run", side_effect=subprocess.CalledProcessError(1, "solve-field")):
        with pytest.raises(RuntimeError, match="solve-field failed"):
            run_solve_field("input.fits", "output.wcs", pixel_scale=1.5)


def test_find_sources(synthetic_image):
    bkg_err, snr = 1.0, 5.0
    source_list, image_out = find_sources(synthetic_image, bkg_err, snr)
    assert isinstance(source_list, pd.DataFrame)
    assert image_out.shape == synthetic_image.shape
    assert len(source_list) >= 1
    assert "x" in source_list.columns and "aperture_sum" in source_list.columns


def test_load_wcs(synthetic_wcs):
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        filename = tmp.name
    try:
        fits.PrimaryHDU(header=synthetic_wcs.to_header()).writeto(filename, overwrite=True)
        wcs = load_wcs(filename)
        assert isinstance(wcs, WCS)
        assert wcs.wcs.ctype[0] == 'RA---TAN'
    finally:
        os.remove(filename)


def test_retrieve_sources(synthetic_wcs):
    df = pd.DataFrame({'x': [50.0], 'y': [50.0]})
    df_out, sky_coords = retrieve_sources(df.copy(), synthetic_wcs)
    assert "RA" in df_out.columns and "Dec" in df_out.columns
    assert isinstance(sky_coords, SkyCoord)


def test_calibrate_photometry():
    df = pd.DataFrame({'aperture_sum': [1000, 2000, 500]})
    coords = SkyCoord([150.0, 150.01, 150.02], [2.0, 2.01, 2.02], unit='deg')
    mock_cat = MagicMock()
    mock_cat.search.return_value = (np.arange(3),)
    mock_cat.xmatch.return_value = (np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
    mock_cat.cal_color.return_value = (25.0, 0.05, 0.01, np.array([20.1, 20.2, 20.3]), np.array([0.3, 0.2, 0.1]), None)
    with patch("catch_analysis_tools.astrometry.cvc") as mock_cvc:
        mock_cvc.PanSTARRS1.return_value = mock_cat
        result = calibrate_photometry(coords, df)
    assert result['zp'] == 25.0 and len(result['m']) == 3


def test_plot_color_correction():
    fig, ax = plot_color_correction(np.array([0.1, 0.4]), np.array([18.3, 18.8]), np.array([18.0, 18.5]), 0.1, 25.0, "r-g")
    assert fig and ax
    plt.close(fig)


def test_plot_image():
    image = np.random.normal(100, 10, (50, 50))
    df = pd.DataFrame({'x': [10, 20, 30], 'y': [10, 25, 35]})
    fig, ax = plot_image(image, df, [0, 1], [1, 2])
    assert fig and ax
    plt.close(fig)


def test_create_header_writes_fits(tmp_path, synthetic_wcs):
    image = np.random.rand(10, 10)
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'aperture_sum': [100, 200, 300]})
    matched_idx, colored_idx = [0, 1], [1, 2]
    output_path = tmp_path / "output.fits"
    create_header(image, synthetic_wcs, 25.0, 0.02, df, matched_idx, colored_idx, str(output_path), "r", "PanSTARRS1", "g")
    assert output_path.exists()
    with fits.open(output_path) as hdul:
        assert hdul[0].header['ZP'] == 25.0 and hdul[1].name == 'DETECTED_SOURCES'


def test_cleanup_files(tmp_path):
    file_base = tmp_path / "testfile"
    extensions = ['.axy', '.corr', '.match', '.new', '.rdls', '.solved', '-ngc.png', '-objs.png', '-indx.png', '-indx.xyls']
    for ext in extensions:
        (file_base.with_name(file_base.name + ext)).write_text("tmp")
    cleanup_files(str(file_base))
    for ext in extensions:
        assert not (file_base.with_name(file_base.name + ext)).exists()
