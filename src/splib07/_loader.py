# Copyright (C) 2023 Brian Schubert.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import pathlib
import re
from functools import cache
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TextIO, overload

import numpy as np
import spectral
from nptyping import Bool, Float, NDArray
from typing_extensions import TypeAlias

from ._index import Splib07Index, Sampling, load_cached_index, SpectrumIdentifier
from ._util import PathLike, VirtualPath, resolve_zip_path

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


_FloatArray: TypeAlias = NDArray[Literal["*"], Float]


class Spectrum(NamedTuple):
    spectrum: _FloatArray
    wavelengths: _FloatArray
    fwhm: _FloatArray


class Splib07:
    """
    Interface to a local archive of the USGS Spectral Library Version 7.
    """

    _root: VirtualPath
    """
    Path to directory containing the extracted USGS Spectral Library Version 7 files.
    """

    _index: Splib07Index

    def __init__(self, root: PathLike, index: Splib07Index | None = None) -> None:
        root_path = resolve_zip_path(root)

        self._index = load_cached_index() if index is None else index

        _assert_splib07_path(root_path)
        self._root = root_path

    @cache
    def list_spectra(self) -> list[str]:
        """
        Return list of all available spectra names.
        """
        return list(
            self._index._sampling_indices[Sampling.MEASURED].all_chapters.keys()
        )

    def search_spectra(self, pattern: str | re.Pattern[str]) -> list[str]:
        """
        Return list of all spectra names that match the given pattern.
        """
        if isinstance(pattern, re.Pattern):
            regex = pattern
        else:
            regex = re.compile(pattern, re.IGNORECASE)
        return [s for s in self.list_spectra() if regex.search(s) is not None]

    @overload
    def load(
        self,
        spectrum: SpectrumIdentifier,
        *,
        resample: Sampling | _FloatArray | tuple[_FloatArray, _FloatArray],
        deleted: Literal["sigil", "nan", "drop"] = ...,
    ) -> Spectrum:
        ...

    @overload
    def load(
        self,
        spectrum: SpectrumIdentifier,
        *,
        resample: Sampling | _FloatArray | tuple[_FloatArray, _FloatArray],
        deleted: Literal["sigil", "nan", "drop"] = ...,
        format: Literal["raw"] = ...,
    ) -> Spectrum:
        ...

    @overload
    def load(
        self,
        spectrum: SpectrumIdentifier,
        *,
        resample: Sampling | _FloatArray | tuple[_FloatArray, _FloatArray],
        deleted: Literal["sigil", "nan", "drop"] = ...,
        format: Literal["spectral"] = ...,
    ) -> spectral.io.envi.SpectralLibrary:
        ...

    def load(
        self,
        spectrum: SpectrumIdentifier,
        *,
        resample: Sampling
        | _FloatArray
        | tuple[_FloatArray, _FloatArray] = Sampling.MEASURED,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
        format: Literal["raw", "spectral"] = "raw",
    ) -> Spectrum | spectral.io.envi.SpectralLibrary:
        """
        Load the given spectrum with the specified resampling.
        """
        if spectrum not in self.list_spectra():
            raise ValueError(f"unknown spectra {spectrum}")

        if isinstance(resample, Sampling):
            resample_source = resample
        else:
            resample_source = Sampling.OVERSAMPLED

        if deleted == "drop":
            # TODO add logic for handling deleted bands when returning wavelengths/bandwidths.
            raise NotImplementedError

        loaded_spectrum = self._load(spectrum, resample_source, deleted)

        if not isinstance(resample, Sampling):
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
                    "spectra names": [spectrum],
                },
            )

        raise ValueError(f"unknown format - {format}")

    def _load(
        self,
        spectrum: str,
        resample_source: Sampling,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
    ) -> Spectrum:
        entry = self._index._sampling_indices[resample_source].all_chapters[spectrum]

        if entry.spectrum_asciidata is None:
            raise ValueError(
                f"spectrum '{spectrum}' missing spectrum data in '{resample_source.value}'"
            )

        with self._root.joinpath(entry.spectrum_asciidata).open("r") as fd:
            spec_data = _load_asciidata(fd, deleted)

        with self._root.joinpath(entry.wavelengths_asciidata).open("r") as fd:
            wavelengths = _load_asciidata(fd, deleted)

        with self._root.joinpath(entry.bandpass_asciidata).open("r") as fd:
            fwhm = _load_asciidata(fd, deleted)

        return Spectrum(spec_data, wavelengths, fwhm)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._root!r})"


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


def _assert_splib07_path(path: VirtualPath) -> None:
    """
    Validate that the given directory contains Spectral Library Version 7 files.
    """
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
