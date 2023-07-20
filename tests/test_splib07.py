import warnings

# Temporarily silence deprecated alias warnings from nptyping 2.5.0 for numpy>=1.24.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping")

import pathlib
from typing import Final

import pytest

import splib07

DATA_DIR: Final = pathlib.Path(__file__).parent.parent / "data"

USGS_ARCHIVE_FILE: Final = DATA_DIR / "usgs_splib07.zip"

skip_if_no_archive = pytest.mark.skipif(
    not USGS_ARCHIVE_FILE.exists(), reason="splib07 archive not found"
)


@pytest.fixture
def library() -> splib07.Splib07:
    return splib07.Splib07(USGS_ARCHIVE_FILE)


@skip_if_no_archive
def test_scan_archive(library):
    assert len(library.list_spectra()) == 2457
    assert len(library.list_resamplings()) == 29
