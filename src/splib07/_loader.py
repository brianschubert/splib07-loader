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
from typing import TYPE_CHECKING, Final, Iterable, Literal, NamedTuple, TextIO, overload

import numpy as np
import spectral
from nptyping import Bool, Float, NDArray
from typing_extensions import TypeAlias, deprecated

from ._common import Chapter, Sampling, SpectrumIdentifier
from ._index import Splib07Index, load_cached_index
from ._util import PathLike, VirtualPath, resolve_zip_path

if TYPE_CHECKING:
    import spectral.io.envi


# Value used to signify a deleted chanel.
# DS1035 p.19
_DeletedChannelMarker: Final = -1.23e34
_DeletedChannelRange: Final = (-1.23001e34, -1.22999e34)


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

    def list_spectra(
        self, chapters: Chapter | Iterable[Chapter] | None = None
    ) -> list[str]:
        """
        Return list of all available spectra names.
        """
        sampling_index = self._index._sampling_indices[Sampling.MEASURED]

        if chapters is None:
            chapter_index = sampling_index.all_chapters
        else:
            chapter_index = sampling_index.only_chapters(chapters)

        return list(chapter_index.keys())

    def search_spectra(
        self,
        pattern: str | re.Pattern[str],
        chapters: Chapter | Iterable[Chapter] | None = None,
    ) -> list[str]:
        """
        Return list of all spectra names that match the given pattern.
        """
        if isinstance(pattern, re.Pattern):
            regex = pattern
        else:
            regex = re.compile(pattern, re.IGNORECASE)
        return [s for s in self.list_spectra(chapters) if regex.search(s) is not None]

    @deprecated(
        "'Splib07.load' is deprecated, use 'Splib07.load_spectrum' instead. 'Splib07.load_spectrum' implements a new"
        " interface that allows the library to be searched much faster, but uses slightly different identifiers for the"
        " library spectra. Use 'Splib07.search_spectra' to find the new identifier for a spectrum that 'Splib07.load'"
        " could load. The current implementation of 'Splib07.load' attempts to reimplement the old behavior of "
        " 'Splib07.load' terms of 'Splib07.load_spectrum'. If you run into issues with using this implementation of "
        " 'Splib07.load' or with migrating to 'Splib07.load_spectrum', consider downgrading to 'splib07-loader==0.4.0'."
    )
    def load(
        self,
        spectra_name: str,
        resample: str | _FloatArray | tuple[_FloatArray, _FloatArray],
        *,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
        format: Literal["raw", "spectral"] = "raw",
    ) -> Spectrum | spectral.io.envi.SpectralLibrary:
        candidates = [
            name for name in self.list_spectra() if name.startswith(spectra_name)
        ]

        try:
            [resolved_name] = candidates
        except ValueError:
            raise ValueError(
                f"old spectra prefix {spectra_name} matched {len(candidates)}"
                f" spectrum identifiers - match must be unique"
            )

        resolved_resample: Sampling | _FloatArray | tuple[_FloatArray, _FloatArray]
        if isinstance(resample, str):
            if resample == "measured":
                resolved_resample = Sampling.MEASURED
            elif resample == "oversampled":
                resolved_resample = Sampling.OVERSAMPLED
            else:
                resolved_resample = Sampling(f"splib07b_{resample}")
        else:
            resolved_resample = resample

        resolved_format = {"raw": "tuple", "spectral": "spectral"}[format]

        return self.load_spectrum(  # type: ignore
            resolved_name,
            resample=resolved_resample,
            deleted=deleted,
            format=resolved_format,
        )

    @overload
    def load_spectrum(
        self,
        spectrum: SpectrumIdentifier,
        *,
        resample: Sampling | _FloatArray | tuple[_FloatArray, _FloatArray],
        deleted: Literal["sigil", "nan", "drop"] = ...,
    ) -> Spectrum:
        ...

    @overload
    def load_spectrum(
        self,
        spectrum: SpectrumIdentifier,
        *,
        resample: Sampling | _FloatArray | tuple[_FloatArray, _FloatArray],
        deleted: Literal["sigil", "nan", "drop"] = ...,
        format: Literal["tuple"] = ...,
    ) -> Spectrum:
        ...

    @overload
    def load_spectrum(
        self,
        spectrum: SpectrumIdentifier,
        *,
        resample: Sampling | _FloatArray | tuple[_FloatArray, _FloatArray],
        deleted: Literal["sigil", "nan", "drop"] = ...,
        format: Literal["spectral"] = ...,
    ) -> spectral.io.envi.SpectralLibrary:
        ...

    def load_spectrum(
        self,
        spectrum: SpectrumIdentifier,
        *,
        resample: Sampling
        | _FloatArray
        | tuple[_FloatArray, _FloatArray] = Sampling.MEASURED,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
        format: Literal["tuple", "spectral"] = "tuple",
    ) -> Spectrum | spectral.io.envi.SpectralLibrary:
        """
        Load the given spectrum with the specified resampling.
        """

        if not isinstance(resample, (Sampling, np.ndarray, tuple)):
            raise ValueError(
                f"got unexpected type for resample: {type(resample)}, expected Sampling, array, or tuple[array, array]"
            )

        if isinstance(resample, Sampling):
            load_sampling = resample
        else:
            load_sampling = Sampling.OVERSAMPLED

        loaded_spectrum = self._load(spectrum, load_sampling, deleted)

        if not isinstance(resample, Sampling):
            loaded_spectrum = _resample(loaded_spectrum, resample)

        if format == "tuple":
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
        deleted: Literal["sigil", "nan", "drop"],
    ) -> Spectrum:
        sampling_index = self._index._sampling_indices[resample_source]
        try:
            entry = sampling_index.all_chapters[spectrum]
        except KeyError as ex:
            raise ValueError(f"unknown spectra {spectrum}") from ex

        if entry.spectrum_asciidata is None:
            raise ValueError(
                f"spectrum '{spectrum}' missing spectrum data in '{resample_source.value}'"
            )

        with self._root.joinpath(entry.spectrum_asciidata).open("r") as fd:
            spec_data = _load_asciidata(fd)

        with self._root.joinpath(entry.wavelengths_asciidata).open("r") as fd:
            wavelengths = _load_asciidata(fd)

        with self._root.joinpath(entry.bandpass_asciidata).open("r") as fd:
            fwhm = _load_asciidata(fd)

        if deleted != "sigil":
            deleted_mask = _usgs_deleted_mask(spec_data)

            if deleted == "nan":
                spec_data[deleted_mask] = np.nan

            if deleted == "drop":
                keep_mask = ~deleted_mask
                spec_data = spec_data[keep_mask]
                wavelengths = wavelengths[keep_mask]
                fwhm = fwhm[keep_mask]

        return Spectrum(spec_data, wavelengths, fwhm)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._root!r})"


def _load_asciidata(file: pathlib.Path | TextIO) -> _FloatArray:
    """Load array from ASCIIdata file."""
    return np.loadtxt(file, skiprows=1)


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
