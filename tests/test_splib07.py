import numpy as np

import splib07
import splib07._util


def test_manual_resample_agrees_with_precomputed(library) -> None:
    # spectrum_name = random.choice(library.list_spectra())
    spectrum_name = "Seawater_Coast_Chl_SW1_BECKa_AREF"

    for resampling in splib07.Sampling:
        print(f"testing {resampling}")

        expected_spectrum = library.load_spectrum(spectrum_name, resample=resampling)
        resampled_spectrum = library.load_spectrum(
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
