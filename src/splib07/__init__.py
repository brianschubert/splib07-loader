import pathlib
from functools import cache
from typing import Final, Literal

import numpy as np

# Distribution version, PEP-440 compatible.
# Should be kept in sync with 'tool.poetry.version' in pyproject.toml.
__version__ = "0.1.0-dev"

# Value used to signify a deleted chanel.
# DS1035 p.19
_DeletedChannelMarker: Final = -1.23e34
_DeletedChannelRange: Final = (-1.23001e34, -1.22999e34)

_RESAMPLING_FIXED_NAMES: Final = {
    "measured": "splib07a",
    "oversampled": "splib07b",
}


class Splib07:
    root_dir: pathlib.Path

    def __init__(self, root_dir: pathlib.Path) -> None:
        _assert_splib07_path(root_dir)
        self.root_dir = root_dir

    @cache
    def list_spectra(self) -> list[str]:
        measured_dir = self.root_dir.joinpath("ASCIIdata").joinpath(
            "ASCIIdata_splib07a"
        )
        spectra_basenames = [
            "_".join(f.name.split("_")[1:-2])
            for f in measured_dir.glob("Chapter*/*.txt")
        ]
        spectra_basenames.sort()
        return spectra_basenames

    @cache
    def list_resamplings(self) -> list[str]:
        named_resamplings = [
            d.name.removeprefix("ASCIIdata_splib07b_")
            for d in self.root_dir.joinpath("ASCIIdata").glob("ASCIIdata_splib07*/")
        ]
        return list(_RESAMPLING_FIXED_NAMES.keys()) + named_resamplings

    def load(
        self,
        spectra: str,
        resampling: str,
        deleted: Literal["sigil", "nan", "drop"] = "nan",
    ):
        if spectra not in self.list_spectra():
            raise ValueError(f"unknown spectra {spectra}")

        if resampling not in self.list_resamplings():
            raise ValueError(f"unknown resampling {resampling}")

        resampling_label = _RESAMPLING_FIXED_NAMES.get(
            resampling, f"splib07b_{resampling}"
        )
        resampling_dir = self.root_dir.joinpath("ASCIIdata").joinpath(
            f"ASCIIdata_{resampling_label}"
        )

        for file in resampling_dir.glob("Chapter*/*.txt"):
            if spectra in file.name:
                return load_asciidata(file, deleted)
        else:
            raise RuntimeError(
                f"missing {resampling} resampling for {spectra} - is the splib07 archive incomplete?"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.root_dir!r})"


def load_asciidata(
    file: pathlib.Path, deleted: Literal["sigil", "nan", "drop"] = "nan"
) -> np.ndarray:
    data = np.loadtxt(file, skiprows=1)

    if deleted == "sigil":
        return data

    mask = usgs_delated_mask(data)

    if deleted == "drop":
        return data[~mask]

    if deleted == "nan":
        data[mask] = np.nan
        return data

    raise ValueError(f"unknown deleted behavior: {deleted}")


def usgs_delated_mask(arr: np.ndarray) -> np.ndarray:
    return _mask_in_range(arr, *_DeletedChannelRange)


def _assert_splib07_path(path: pathlib.Path) -> None:
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
    return (start <= arr) & (arr <= end)
