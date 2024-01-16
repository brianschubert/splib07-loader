import numpy as np

import splib07
import splib07._util


def test_manual_resample_agrees_with_precomputed(library) -> None:
    # spectrum_name = random.choice(library.list_spectra())
    spectrum_name = "Seawater_Coast_Chl_SW1_BECKa_AREF"

    for resampling in splib07.Sampling:
        print(f"testing {spectrum_name} {resampling}")

        expected_spectrum = library.load_spectrum(spectrum_name, resample=resampling)
        resampled_spectrum = library.load_spectrum(
            spectrum_name,
            resample=(expected_spectrum.wavelengths, expected_spectrum.fwhm),
        )
        nan_mask = np.isnan(expected_spectrum.spectrum)

        try:
            np.testing.assert_allclose(
                resampled_spectrum.spectrum[~nan_mask],
                expected_spectrum.spectrum[~nan_mask],
                atol=0.05,
                rtol=0.05,
            )
        except AssertionError:
            print(
                f"{resampled_spectrum.spectrum.tolist()=}\n{expected_spectrum.spectrum.tolist()=}"
            )
            raise
