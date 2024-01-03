from __future__ import annotations

import collections
import enum
import functools
import itertools
import lzma
import pathlib
import pickle
import re
import sys
import zipfile
from typing import Final, Literal, Mapping, NamedTuple, overload
import importlib.resources

import bs4
from typing_extensions import Self, TypeAlias

from splib07._util import PathLike, VirtualPath, resolve_zip_path

_SpectrumIdentifier: TypeAlias = str
"""Unique identifier for a particular spectrum, shared across all available samplings."""

_ChapterIndex: TypeAlias = dict[_SpectrumIdentifier, "_SpectrumEntry"]
"""Mapping of all spectra contained in a chapter."""

_SPACES_PATTERN: Final = re.compile(" +")
"""Regex for locating sequences of repeated spaces."""


@enum.unique
class Sampling(enum.Enum):
    """Available samplings in splib07."""

    MEASURED = "splib07a"
    OVERSAMPLED = "splib07b"
    ASD = "splib07b_cvASD"
    AVIRIS_1995 = "splib07b_cvAVIRISc1995"
    AVIRIS_1996 = "splib07b_cvAVIRISc1996"
    AVIRIS_1997 = "splib07b_cvAVIRISc1997"
    AVIRIS_1998 = "splib07b_cvAVIRISc1998"
    AVIRIS_1999 = "splib07b_cvAVIRISc1999"
    AVIRIS_2000 = "splib07b_cvAVIRISc2000"
    AVIRIS_2001 = "splib07b_cvAVIRISc2001"
    AVIRIS_2005 = "splib07b_cvAVIRISc2005"
    AVIRIS_2006 = "splib07b_cvAVIRISc2006"
    AVIRIS_2009 = "splib07b_cvAVIRISc2009"
    AVIRIS_2010 = "splib07b_cvAVIRISc2010"
    AVIRIS_2011 = "splib07b_cvAVIRISc2011"
    AVIRIS_2012 = "splib07b_cvAVIRISc2012"
    AVIRIS_2013 = "splib07b_cvAVIRISc2013"
    AVIRIS_2014 = "splib07b_cvAVIRISc2014"
    CRISM_GLOBAL = "splib07b_cvCRISM-global"
    CRISM_TARGET = "splib07b_cvCRISMjMTR3"
    HYMAP_2007 = "splib07b_cvHYMAP2007"
    HYMAP_2014 = "splib07b_cvHYMAP2014"
    HYPERION = "splib07b_cvHYPERION"
    M_3 = "splib07b_cvM3-target"
    VIMS = "splib07b_cvVIMS"
    ASTER = "splib07b_rsASTER"
    LANDSAT_8 = "splib07b_rsLandsat8"
    SENTINEL_2 = "splib07b_rsSentinel2"
    WORLD_VIEW_3 = "splib07b_rsWorldView3"


@enum.unique
class Chapter(enum.IntEnum):
    MINERALS = 1
    SOILS_AND_MIXTURES = 2
    COATINGS = 3
    LIQUIDS = 4
    ORGANICS = 5
    ARTIFICIAL = 6
    VEGETATION = 7

    @property
    def initial(self) -> str:
        return self.name[0]


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


class _SpectrumEntry(NamedTuple):
    """
    An entry (row) in an index datatable.
    """

    # some entries are missing spectra, error plots, or range plots
    # e.g. Landsat8 Ilmenite HS231.3B NIC4bcu

    name: _SpectrumIdentifier
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
    index_file = importlib.resources.files(__package__).joinpath("index.pickle")

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
    return pathlib.PurePath(anchor["href"])  # type: ignore
