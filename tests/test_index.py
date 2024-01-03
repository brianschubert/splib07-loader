import pathlib
import zipfile
from typing import Final

import pytest

import splib07

_EXPECTED_CHAPTER_LENGTHS: Final = (1275, 208, 11, 23, 359, 289, 285)


@pytest.fixture(scope="module")
def generated_index(library_path) -> splib07._index.Splib07Index:
    return splib07._index.Splib07Index.generate_index(library_path)


def test_resolve_zip_path_identifies_zip(library_path) -> None:
    resolved_path = splib07._util.resolve_zip_path(library_path)

    expected_type = zipfile.Path if library_path.suffix == ".zip" else pathlib.Path
    assert isinstance(resolved_path, expected_type)


def test_read_index_finds_all_sampling_datatables(library_path) -> None:
    sampling_datatables = splib07._index._read_toc_sampling_paths(
        splib07._util.resolve_zip_path(library_path)
        / "indexes"
        / "table_of_contents.html"
    )

    assert len(sampling_datatables) == len(splib07._index.Sampling)


def test_generate_index_finds_all_entries(generated_index) -> None:
    # Check that there is an index for every available sampling.
    assert len(generated_index._sampling_indices) == len(splib07._index.Sampling)

    for sampling_index in generated_index._sampling_indices.values():
        chapter_lengths = tuple(map(len, sampling_index))

        # Check that each sampling contains the expected number of spectra in each chapter.
        assert chapter_lengths == _EXPECTED_CHAPTER_LENGTHS
