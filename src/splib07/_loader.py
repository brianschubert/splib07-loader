from __future__ import annotations

import importlib
import pathlib
import re
import zipfile
from functools import cache
from typing import TYPE_CHECKING, Final, Iterable, Literal, NamedTuple, TextIO, overload

import numpy as np
import spectral
from nptyping import Bool, Float, NDArray
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import spectral.io.envi


# Value used to signify a deleted chanel.
# DS1035 p.19
_DeletedChannelMarker: Final = -1.23e34
_DeletedChannelRange: Final = (-1.23001e34, -1.22999e34)

# Special names for the splib07a and splib07b resamplings.
_RESAMPLING_FIXED_NAMES: Final = {
    "measured": "splib07a",
    "oversampled": "splib07b",
}

_VirtualPath: TypeAlias = "pathlib.Path | zipfile.Path"

_FloatArray: TypeAlias = NDArray[Literal["*"], Float]


class Spectrum(NamedTuple):
    spectrum: _FloatArray
    wavelengths: _FloatArray
    fwhm: _FloatArray


class Splib07:
    """
    Interface to a local archive of the USGS Spectral Library Version 7.
    """

    root: _VirtualPath
    """
    Path to directory containing the extracted USGS Spectral Library Version 7 files.
    """

    def __init__(self, root: str | pathlib.Path) -> None:
        path: _VirtualPath
        if zipfile.is_zipfile(root):
            path = zipfile.Path(root, at="")
        else:
            path = pathlib.Path(root)

        _assert_splib07_path(path)
        self.root = path

    @cache
    def list_spectra(self) -> list[str]:
        """
        Return list of all available spectra names.
        """
        measured_dir = self.root.joinpath("ASCIIdata").joinpath("ASCIIdata_splib07a")
        spectra_basenames = [
            "_".join(f.name.split("_")[1:-2]) for f in _scan_spectra(measured_dir)
        ]
        spectra_basenames.sort()
        return spectra_basenames

    @cache
    def list_resamplings(self) -> list[str]:
        """
        Return list of all available resamplings.
        """
        named_resamplings = [
            d.name.removeprefix("ASCIIdata_splib07b_")
            for d in self.root.joinpath("ASCIIdata").iterdir()
            if d.name.startswith("ASCIIdata_splib07b_")
        ]
        return list(_RESAMPLING_FIXED_NAMES.keys()) + named_resamplings

    def search_spectra(self, regex: str | re.Pattern[str]) -> list[str]:
        """
        Return list of all spectra names that match the given pattern.
        """
        if isinstance(regex, re.Pattern):
            pattern = regex
        else:
            pattern = re.compile(regex, re.IGNORECASE)
        return [s for s in self.list_spectra() if pattern.search(s) is not None]

    @overload
    def load(
        self,
        spectra_name: str,
        resample: str | _FloatArray | tuple[_FloatArray, _FloatArray],
        *,
        deleted: Literal["sigil", "nan", "drop"] = ...,
    ) -> Spectrum:
        ...

    @overload
    def load(
        self,
        spectra_name: str,
        resample: str | _FloatArray | tuple[_FloatArray, _FloatArray],
        *,
        deleted: Literal["sigil", "nan", "drop"] = ...,
        format: Literal["raw"] = ...,
    ) -> Spectrum:
        ...

    @overload
    def load(
        self,
        spectra_name: str,
        resample: str | _FloatArray | tuple[_FloatArray, _FloatArray],
        *,
        deleted: Literal["sigil", "nan", "drop"] = ...,
        format: Literal["spectral"] = ...,
    ) -> spectral.io.envi.SpectralLibrary:
        ...

    def load(
        self,
        spectra_name: str,
        resample: str | _FloatArray | tuple[_FloatArray, _FloatArray],
        *,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
        format: Literal["raw", "spectral"] = "raw",
    ) -> Spectrum | spectral.io.envi.SpectralLibrary:
        """
        Load the given spectrum with the specified resampling.
        """
        if spectra_name not in self.list_spectra():
            raise ValueError(f"unknown spectra {spectra_name}")

        if isinstance(resample, str):
            resample_source = resample
        else:
            resample_source = "oversampled"

        if resample_source not in self.list_resamplings():
            raise ValueError(f"unknown resampling {resample_source}")

        if deleted == "drop":
            # TODO add logic for handling deleted bands when returning wavelengths/bandwidths.
            raise NotImplementedError

        loaded_spectrum = self._load(spectra_name, resample_source, deleted)

        if not isinstance(resample, str):
            loaded_spectrum = _resample(loaded_spectrum, resample)

        if format == "raw":
            return loaded_spectrum

        if format == "spectral":
            return spectral.io.envi.SpectralLibrary(
                data=loaded_spectrum.spectrum.reshape(1, -1),
                header={
                    "wavelength": loaded_spectrum.wavelengths,
                    "fwhm": loaded_spectrum.fwhm,
                    "wavelength units": "micrometer",
                    "spectra names": [spectra_name],
                },
            )

        raise ValueError(f"unknown format - {format}")

    def _load(
        self,
        spectra_name: str,
        resample_source: str,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
    ) -> Spectrum:
        resampling_label = _RESAMPLING_FIXED_NAMES.get(
            resample_source, f"splib07b_{resample_source}"
        )
        resampling_dir = self.root.joinpath("ASCIIdata").joinpath(
            f"ASCIIdata_{resampling_label}"
        )

        # TODO direct lookup from file format instead of manually searching.
        for file in _scan_spectra(resampling_dir):
            if spectra_name in file.name:
                with file.open("r") as fd:
                    spectra = _load_asciidata(fd, deleted)  # type: ignore
                break
        else:
            # Should never happen.
            raise RuntimeError(
                f"missing {resample_source} resampling for {spectra_name} - is the splib07 data directory incomplete?"
            )

        # TODO test stability.
        # TODO tidy wavelength matching hack.
        # We use the first four letters of the second to last component in the spectrum's
        # filename identify its associated wavelength/bandwidth files.
        sampling_label = file.name.split("_")[-2][:4]

        # Attempt to locate the associated FWHM bandwidths file.
        fwhm_candidates = [
            f
            for f in resampling_dir.iterdir()
            if f.name.endswith(".txt")
            and ("andpass" in f.name or "esolution" in f.name)
            and not f.name.endswith("_nm.txt")
        ]
        if len(fwhm_candidates) > 1:
            fwhm_candidates = [f for f in fwhm_candidates if sampling_label in f.name]
        if len(fwhm_candidates) != 1:
            raise RuntimeError(
                f"could not determine bandwidths/FWHMs for {spectra_name} "
                f"with resampling {resample_source}"
            )
        with fwhm_candidates[0].open("r") as fd:
            fwhm = _load_asciidata(fd, deleted)  # type: ignore

        # Attempt to locate the associated wavelengths file.
        wavelength_candidates = [
            f
            for f in resampling_dir.iterdir()
            if f.name.endswith(".txt")
            and ("avelength" in f.name or "aves" in f.name)
            and not f.name.endswith("_SRFs.txt")
            and not f.name.endswith("Function.txt")
            and not f.name.endswith("Functions.txt")
        ]
        if len(wavelength_candidates) > 1:
            wavelength_candidates = [
                f for f in wavelength_candidates if sampling_label in f.name
            ]
        if len(wavelength_candidates) != 1:
            # Special case - ASD{HR,NG} in splib07{a,b}
            if sampling_label.startswith("ASD"):
                wavelength_candidates = [
                    resampling_dir.joinpath(
                        "splib07b_Wavelengths_ASDFR_0.35-2.5microns_2151ch.txt"
                        if resample_source == "oversampled"
                        else "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"
                    )
                ]
            else:
                raise RuntimeError(
                    f"could not determine wavelengths for {spectra_name} "
                    f"with resampling {resample_source}"
                )
        with wavelength_candidates[0].open("r") as fd:
            wavelengths = _load_asciidata(fd, deleted)  # type: ignore

        return Spectrum(spectra, wavelengths, fwhm)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.root!r})"


def _scan_spectra(path: _VirtualPath) -> Iterable[_VirtualPath]:
    """Iterate the spectrum files in the specified resampling directory."""
    for directory in path.iterdir():
        if not directory.name.startswith("Chapter"):
            continue
        yield from directory.iterdir()


def _load_asciidata(
    file: pathlib.Path | TextIO,
    deleted: Literal["sigil", "nan", "drop"] = "nan",
) -> _FloatArray:
    """Load array from ASCIIdata file."""
    data = np.loadtxt(file, skiprows=1)

    if deleted == "sigil":
        return data

    mask = _usgs_deleted_mask(data)

    if deleted == "drop":
        return data[~mask]

    if deleted == "nan":
        data[mask] = np.nan
        return data

    raise ValueError(f"unknown deleted behavior: {deleted}")


def _usgs_deleted_mask(arr: _FloatArray) -> NDArray[Literal["*"], Bool]:
    """Generate mask of bands marked as deleted in the given array."""
    return _mask_in_range(arr, *_DeletedChannelRange)


def _assert_splib07_path(path: _VirtualPath) -> None:
    """
    Validate that the given directory contains Spectral Library Version 7 files.
    """
    if not path.exists():
        raise FileNotFoundError(f"no such directory or archive file: '{path}'")

    # TODO check for existence of other key files.
    expected_subdirs = ["ASCIIdata"]
    actual_contents = [entry.name for entry in path.iterdir()]

    missing = [
        expected for expected in expected_subdirs if expected not in actual_contents
    ]

    if missing:
        raise ValueError(
            f"invalid splib07 dataset root '{path}': missing expected subdirectories {missing}. Actual contents: {actual_contents}"
        )


def _mask_in_range(
    arr: _FloatArray, start: float, end: float
) -> NDArray[Literal["*"], Bool]:
    """
    Return mask that is True for all entries that are in the range [start, end].
    """
    return (start <= arr) & (arr <= end)


def _resample(
    spectrum: Spectrum, to: _FloatArray | tuple[_FloatArray, _FloatArray]
) -> Spectrum:
    if isinstance(to, tuple):
        target_wavelengths, target_fwhm = to
    else:
        # Need to use importlib instead of normal import since spectral seems to shadow
        # some submodules with module attributes.
        resampling = importlib.import_module("spectral.algorithms.resampling")

        target_wavelengths = to
        target_fwhm = resampling.build_fwhm(to)

    resampler = spectral.BandResampler(
        centers1=spectrum.wavelengths,
        centers2=target_wavelengths,
        fwhm1=spectrum.fwhm,
        fwhm2=target_fwhm,
    )

    return Spectrum(resampler(spectrum.spectrum), target_wavelengths, target_fwhm)
