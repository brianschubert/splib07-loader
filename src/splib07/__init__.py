from __future__ import annotations

import io
import pathlib
import zipfile
from functools import cache
from typing import Final, Iterable, Literal
from typing_extensions import TypeAlias

import numpy as np

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


class Splib07:
    """
    Interface to a local archive of the USGS Spectral Library Version 7.
    """

    root: _VirtualPath
    """
    Path to directory containing the extracted USGS Spectral Library Version 7 files.
    """

    def __init__(self, root: pathlib.Path) -> None:
        if zipfile.is_zipfile(root):
            path = zipfile.Path(root, at="")
        else:
            path = root

        _assert_splib07_path(path)
        self.root = path

    @cache
    def list_spectra(self) -> list[str]:
        """
        Return list of all spectra container in the library.
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
        Return list of all generated resamplings of the spectra.
        """
        named_resamplings = [
            d.name.removeprefix("ASCIIdata_splib07b_")
            for d in self.root.joinpath("ASCIIdata").iterdir()
            if d.name.split("ASCIIdata_splib07")
        ]
        return list(_RESAMPLING_FIXED_NAMES.keys()) + named_resamplings

    def load(
        self,
        spectra: str,
        resampling: str,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
    ):
        """
        Load the given spectra with the specified resampling.
        """
        if spectra not in self.list_spectra():
            raise ValueError(f"unknown spectra {spectra}")

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
            if spectra in file.name:
                with file.open("r") as fd:
                    return _load_asciidata(fd, deleted)
        else:
            # Should never happen.
            raise RuntimeError(
                f"missing {resampling} resampling for {spectra} - is the splib07 data directory incomplete?"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.root!r})"


def _scan_spectra(path: _VirtualPath) -> Iterable[_VirtualPath]:
    for directory in path.iterdir():
        if not directory.name.startswith("Chapter"):
            continue
        yield from directory.iterdir()


def _load_asciidata(
    file: pathlib.Path | io.IOBase, deleted: Literal["sigil", "nan", "drop"] = "nan"
) -> np.ndarray:
    """Load array from ASCIIdata file."""
    data = np.loadtxt(file, skiprows=1)

    if deleted == "sigil":
        return data

    mask = _usgs_delated_mask(data)

    if deleted == "drop":
        return data[~mask]

    if deleted == "nan":
        data[mask] = np.nan
        return data

    raise ValueError(f"unknown deleted behavior: {deleted}")


def _usgs_delated_mask(arr: np.ndarray) -> np.ndarray:
    """Generate mask of bands marked as deleted in the given array."""
    return _mask_in_range(arr, *_DeletedChannelRange)


def _assert_splib07_path(path: pathlib.Path) -> None:
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


def _mask_in_range(arr: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Return mask that is True for all entries that are in the range [start, end].
    """
    return (start <= arr) & (arr <= end)
