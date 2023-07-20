import pathlib
from typing import Final

import numpy as np

# Distribution version, PEP-440 compatible.
# Should be kept in sync with 'tool.poetry.version' in pyproject.toml.
__version__ = "0.1.0-dev"

# Value used to signify a deleted chanel.
# DS1035 p.19
_DeletedChannelMarker: Final = -1.23e-34
_DeletedChannelRange: Final = (-1.23001e-34, -1.22999e-34)


class Splib07:
    root_dir: pathlib.Path

    def __init__(self, root_dir: pathlib.Path) -> None:
        _assert_splib07_path(root_dir)
        self.root_dir = root_dir

    def list_libraries(self) -> list[str]:
        return [
            d.name.removeprefix("ASCIIdata_")
            for d in self.root_dir.joinpath("ASCIIdata").glob("ASCIIdata_splib07*/")
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.root_dir!r})"


def usgs_deleted2nan(arr: np.ndarray) -> None:
    mask = _mask_in_range(arr, *_DeletedChannelRange)
    arr[mask] = np.nan


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
