import numpy as np


def test_scan_archive(library) -> None:
    assert len(library.list_spectra()) == 2457
    assert len(library.list_resamplings()) == 29


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
