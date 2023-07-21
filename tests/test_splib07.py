import warnings

# Temporarily silence deprecated alias warnings from nptyping 2.5.0 for numpy>=1.24.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping")

import pathlib
from typing import Final

import numpy as np
import numpy.testing
import pytest

import splib07

DATA_DIR: Final = pathlib.Path(__file__).parent.parent / "data"

USGS_ARCHIVE_FILE: Final = DATA_DIR / "usgs_splib07.zip"

skip_if_no_archive = pytest.mark.skipif(
    not USGS_ARCHIVE_FILE.exists(), reason="splib07 archive not found"
)


@pytest.fixture
def library() -> splib07.Splib07:
    return splib07.Splib07(USGS_ARCHIVE_FILE)


@skip_if_no_archive
def test_scan_archive(library) -> None:
    assert len(library.list_spectra()) == 2457
    assert len(library.list_resamplings()) == 29


@skip_if_no_archive
def test_experimental_resample(library) -> None:
    # spectrum_name = random.choice(library.list_spectra())
    spectrum_name = "Seawater_Coast_Chl_SW1"

    for resampling in library.list_resamplings():
        print(f"testing {resampling}")

        expected_spectrum = library.load(spectrum_name, resample=resampling)
        resampled_spectrum = library.load(
            spectrum_name,
            resample=(expected_spectrum.wavelengths, expected_spectrum.fwhm),
        )
        nan_mask = np.isnan(expected_spectrum.spectrum) | np.isnan(
            resampled_spectrum.spectrum
        )

        np.testing.assert_allclose(
            resampled_spectrum.spectrum[~nan_mask],
            expected_spectrum.spectrum[~nan_mask],
            atol=0.01,
            rtol=0.01,
        )
