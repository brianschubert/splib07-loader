import warnings

# Temporarily silence deprecated alias warnings from nptyping 2.5.0 for numpy>=1.24.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping")

import pathlib
from typing import Final

import pytest

import splib07

DATA_DIR: Final = pathlib.Path(__file__).parent.parent / "data"
USGS_ARCHIVE_DIR: Final = DATA_DIR / "usgs_splib07"
USGS_ARCHIVE_ZIP: Final = DATA_DIR / "usgs_splib07.zip"


@pytest.fixture(params=[USGS_ARCHIVE_DIR, USGS_ARCHIVE_ZIP], ids=["directory", "zip"])
def library_path(request) -> pathlib.Path:
    path = pathlib.Path(request.param)

    if not path.exists():
        pytest.skip(f"splib07 {request.node.callspec.id} not found at '{path}'")

    return path


@pytest.fixture
def library(library_path) -> splib07.Splib07:
    return splib07.Splib07(library_path)
