from __future__ import annotations

import collections
import enum
import functools
import itertools
import lzma
import pathlib
import pickle
import re
from typing import (
    Final,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    overload,
)
import importlib.resources

import bs4
from typing_extensions import Self, TypeAlias

from splib07._common import Chapter, Sampling, SpectrumIdentifier
from splib07._util import PathLike, VirtualPath, resolve_zip_path


_ChapterIndex: TypeAlias = MutableMapping[SpectrumIdentifier, "_SpectrumEntry"]
"""Mapping of all spectra contained in a chapter."""

_SPACES_PATTERN: Final = re.compile(" +")
"""Regex for locating sequences of repeated spaces."""


class Splib07Index:
    """
    Index of all spectra contained in an splib07 archive.
    """

    _sampling_indices: Mapping[Sampling, _SamplingIndex]

    def __init__(self, sampling_indices: Mapping[Sampling, _SamplingIndex]) -> None:
        self._sampling_indices = sampling_indices

    @classmethod
    def generate_index(cls, library_root: PathLike) -> Self:
        root_path = resolve_zip_path(library_root)
        sampling_datatables = _read_toc_sampling_paths(
            root_path / "indexes" / "table_of_contents.html"
        )

        sampling_indices = {}

        for sampling, datatable_path in sampling_datatables.items():
            sampling_indices[sampling] = _read_datatable(
                root_path.joinpath("indexes").joinpath(datatable_path)
            )

        return cls(sampling_indices)


class _SamplingIndex(NamedTuple):
    minerals: _ChapterIndex
    soils_and_mixtures: _ChapterIndex
    coatings: _ChapterIndex
    liquids: _ChapterIndex
    organics: _ChapterIndex
    artificial: _ChapterIndex
    vegetation: _ChapterIndex

    def only_chapters(self, chapters: Chapter | Iterable[Chapter]) -> _ChapterIndex:
        if isinstance(chapters, Chapter):
            return self[chapters.value - 1]
        else:
            return collections.ChainMap(*[self[c.value - 1] for c in chapters])

    @property
    def all_chapters(self) -> _ChapterIndex:
        return collections.ChainMap(*self)


class _SpectrumEntry(NamedTuple):
    """
    An entry (row) in an index datatable.
    """

    # some entries are missing spectra, error plots, or range plots
    # e.g. Landsat8 Ilmenite HS231.3B NIC4bcu

    name: SpectrumIdentifier
    description: pathlib.PurePath
    spectrum_asciidata: pathlib.PurePath | None
    error_asciidata: pathlib.PurePath | None
    wavelengths_asciidata: pathlib.PurePath
    bandpass_asciidata: pathlib.PurePath
    range_plot: pathlib.PurePath | None
    wavelength_plot: pathlib.PurePath
    bandpass_plot: pathlib.PurePath
    extra_range_plots: list[pathlib.PurePath | None]


@functools.cache
def load_cached_index() -> Splib07Index:
    index_file = importlib.resources.files(__package__).joinpath("index.pickle.xz")

    with index_file.open("rb") as temp, lzma.open(temp) as fd:
        return pickle.load(fd)  # type: ignore


def _read_toc_sampling_paths(toc_path: VirtualPath) -> dict[Sampling, pathlib.PurePath]:
    """
    Extract available samplings from TOC HTML file.

    The TOC file is located at ``usgs_splib07/indexes/table_of_contents.html``.

    The returned dictionary *should* include a file path for each available sampling.
    Checking whether this is true is the responsibility of the caller.
    """
    soup = bs4.BeautifulSoup(toc_path.read_text(), features="html.parser")

    datatables = {}

    for li_tags in soup.find_all(["li"]):
        link = li_tags.find("a")

        datatable_path = pathlib.PurePath(link["href"])

        # Extract name of sampling
        sampling_id = datatable_path.stem.removeprefix("datatable_")

        # Will raise on unknown sampling.
        sampling = Sampling(sampling_id)

        # Duplicate entries would have the same file name, so no need to check.
        datatables[sampling] = datatable_path

    return datatables


def _read_datatable(
    path: VirtualPath,
) -> _SamplingIndex:
    soup = bs4.BeautifulSoup(path.read_text(), features="html.parser")

    table_tags = soup.find_all("table")

    expected_num_tables = len(Chapter) + 1
    if len(table_tags) != expected_num_tables:
        raise ValueError(
            f"expected to find exactly {expected_num_tables} table tags in datatable file,"
            f" found {len(table_tags)} in '{path}'"
        )

    chapter_indices = _SamplingIndex(*({} for _ in range(len(Chapter))))

    # First table is the header banner.
    # The remaining 7 datatables are the spectra chapters.
    for table, current_index in zip(table_tags[1:], chapter_indices):
        all_rows = table.find_all("tr")

        # Skip first 3 rows (header).
        for row in itertools.islice(all_rows, 4, None):
            data = row.find_all("td")

            try:
                [
                    title,
                    description,
                    spectrum_asciidata,
                    error_asciidata,
                    wavelengths_asciidata,
                    bandpass_asciidata,
                    range_plot,
                    *extra_plots,
                    wavelength_plot,
                    bandpass_plot,
                ] = data

                spectrum_identifier = re.sub(_SPACES_PATTERN, "_", title.text)

                current_index[spectrum_identifier] = _SpectrumEntry(
                    name=spectrum_identifier,
                    description=_extract_link_path(description, False),
                    spectrum_asciidata=_extract_link_path(spectrum_asciidata, True),
                    error_asciidata=_extract_link_path(error_asciidata, True),
                    wavelengths_asciidata=_extract_link_path(
                        wavelengths_asciidata, False
                    ),
                    bandpass_asciidata=_extract_link_path(bandpass_asciidata, False),
                    range_plot=_extract_link_path(range_plot, True),
                    wavelength_plot=_extract_link_path(wavelength_plot, False),
                    bandpass_plot=_extract_link_path(bandpass_plot, False),
                    extra_range_plots=[
                        _extract_link_path(p, True) for p in extra_plots
                    ],
                )
            except ValueError as ex:
                raise ValueError(
                    f"""failed to interpret datatable row {data}"""
                ) from ex

    return chapter_indices


@overload
def _extract_link_path(
    tag: bs4.Tag, missing_ok: Literal[True]
) -> pathlib.PurePath | None:
    ...


@overload
def _extract_link_path(tag: bs4.Tag, missing_ok: Literal[False]) -> pathlib.PurePath:
    ...


def _extract_link_path(tag: bs4.Tag, missing_ok: bool) -> pathlib.PurePath | None:
    anchor = tag.find("a")
    if anchor is None:
        if not missing_ok:
            raise ValueError(f"missing anchor in tag {tag}")
        return None
    return pathlib.PurePath(anchor["href"].removeprefix("../"))  # type: ignore
