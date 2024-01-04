import os
import pathlib
import zipfile
from typing import Final

import pytest

import splib07
import splib07._util

_EXPECTED_CHAPTER_LENGTHS: Final = (1275, 208, 11, 23, 359, 289, 285)


@pytest.fixture(scope="module")
def generated_index(library_path) -> splib07._index.Splib07Index:
    if os.environ.get("TEST_GENERATED_INDEX") != "1":
        pytest.skip(
            "set the environment variable TEST_GENERATED_INDEX=1 to run test involving generating the index"
        )
    return splib07._index.Splib07Index.generate_index(library_path)


@pytest.fixture(scope="module")
def cached_index() -> splib07._index.Splib07Index:
    return splib07._index.load_cached_index()


@pytest.fixture(params=[generated_index.__name__, cached_index.__name__])
def index(request) -> splib07._index.Splib07Index:
    # Note: library_path must also be requested for the value of generated_index to be retrieved correctly.
    # See https://github.com/pytest-dev/pytest/issues/4666.
    return request.getfixturevalue(request.param)


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


def test_generate_index_finds_all_entries(index, library_path) -> None:
    # Check that there is an index for every available sampling.
    assert len(index._sampling_indices) == len(splib07._index.Sampling)

    for sampling_index in index._sampling_indices.values():
        chapter_lengths = tuple(map(len, sampling_index))

        # Check that each sampling contains the expected number of spectra in each chapter.
        assert chapter_lengths == _EXPECTED_CHAPTER_LENGTHS


def test_index_all_paths_exist(index, library_path) -> None:
    resolved_path = splib07._util.resolve_zip_path(library_path)

    for sampling_index in index._sampling_indices.values():
        for entry in sampling_index.all_chapters.values():
            if entry.spectrum_asciidata is not None:
                assert resolved_path.joinpath(entry.spectrum_asciidata).is_file()

            if entry.error_asciidata is not None:
                assert resolved_path.joinpath(entry.error_asciidata).is_file()

            assert resolved_path.joinpath(entry.wavelengths_asciidata).is_file()

            assert resolved_path.joinpath(entry.bandpass_asciidata).is_file()

            if entry.range_plot is not None:
                assert resolved_path.joinpath(entry.range_plot).is_file()

            assert resolved_path.joinpath(entry.wavelength_plot).is_file()

            assert resolved_path.joinpath(entry.bandpass_plot).is_file()

            for extra_plot in entry.extra_range_plots:
                if extra_plot is not None:
                    assert resolved_path.joinpath(extra_plot).is_file()
