from __future__ import annotations

import pathlib
import re
import zipfile
from functools import cache
from typing import TYPE_CHECKING, Final, Iterable, Literal, TextIO

import numpy as np
from nptyping import Bool, Float, NDArray
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import spectral.io.envi

# Distribution version, PEP-440 compatible.
# Should be kept in sync with 'tool.poetry.version' in pyproject.toml.
__version__ = "0.1.0-dev"

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


class Splib07:
    """
    Interface to a local archive of the USGS Spectral Library Version 7.
    """

    root: _VirtualPath
    """
    Path to directory containing the extracted USGS Spectral Library Version 7 files.
    """

    def __init__(self, root: pathlib.Path) -> None:
        path: _VirtualPath
        if zipfile.is_zipfile(root):
            path = zipfile.Path(root, at="")
        else:
            path = root

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

    def search_spectra(self, regex: str | re.Pattern) -> list[str]:
        """
        Return list of all spectra names that match the given pattern.
        """
        if isinstance(regex, re.Pattern):
            pattern = regex
        else:
            pattern = re.compile(regex, re.IGNORECASE)
        return [s for s in self.list_spectra() if pattern.search(s) is not None]

    def load(
        self,
        spectra_name: str,
        resampling: str,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
        format: Literal["onlyspectra", "tuple", "spectral"] = "onlyspectra",
    ) -> (
        _FloatArray
        | tuple[_FloatArray, _FloatArray, _FloatArray]
        | spectral.io.envi.SpectralLibrary
    ):
        """
        Load the given spectrum with the specified resampling.
        """
        if spectra_name not in self.list_spectra():
            raise ValueError(f"unknown spectra {spectra_name}")

        if resampling not in self.list_resamplings():
            raise ValueError(f"unknown resampling {resampling}")

        resampling_label = _RESAMPLING_FIXED_NAMES.get(
            resampling, f"splib07b_{resampling}"
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
                f"missing {resampling} resampling for {spectra_name} - is the splib07 data directory incomplete?"
            )

        if format == "onlyspectra":
            return spectra

        if deleted == "drop":
            # TODO add logic for handling deleted bands when returning wavelengths/bandwidths.
            raise NotImplementedError

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
        ]
        if len(fwhm_candidates) > 1:
            fwhm_candidates = [f for f in fwhm_candidates if sampling_label in f.name]
        if len(fwhm_candidates) != 1:
            raise RuntimeError(
                f"could not determine bandwidths/FWHMs for {spectra_name}"
            )
        with fwhm_candidates[0].open("r") as fd:
            fwhm = _load_asciidata(fd, deleted)  # type: ignore

        # Attempt to locate the associated wavelengths file.
        wavelength_candidates = [
            f
            for f in resampling_dir.iterdir()
            if f.name.endswith(".txt") and "avelength" in f.name
        ]
        if len(wavelength_candidates) > 1:
            wavelength_candidates = [
                f for f in wavelength_candidates if sampling_label in f.name
            ]
        if len(wavelength_candidates) != 1:
            raise RuntimeError(f"could not determine wavelengths for {spectra_name}")
        with wavelength_candidates[0].open("r") as fd:
            wavelengths = _load_asciidata(fd, deleted)  # type: ignore

        if format == "tuple":
            return spectra, wavelengths, fwhm

        if format == "spectral":
            try:
                import spectral.io.envi
            except ImportError as ex:
                raise ValueError(
                    "unable to use spectral format - spectral package could not be loaded"
                ) from ex
            return spectral.io.envi.SpectralLibrary(
                data=spectra.reshape(1, -1),
                header={
                    "wavelength": wavelengths,
                    "fwhm": fwhm,
                    "wavelength units": "micrometer",
                    "spectra names": [spectra_name],
                },
            )

        raise ValueError(f"unknown format {format}")

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
