from __future__ import annotations

import enum
import pathlib
import zipfile

import bs4

from splib07._util import VirtualPath


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


class Splib07Index:
    """
    Index of all spectra contained in a splib07 archive.
    """

    root_path: VirtualPath

    _sampling_datatables: dict[Sampling, VirtualPath]

    def __init__(self, root: str | pathlib.Path) -> None:
        path: VirtualPath
        if zipfile.is_zipfile(root):
            path = zipfile.Path(root, at="")
        else:
            path = pathlib.Path(root)

        self.root_path = path

        self._sampling_datatables = _read_toc_sampling_paths(
            self.root_path / "indexes" / "table_of_contents.html"
        )


def _read_toc_sampling_paths(toc_path: VirtualPath) -> dict[Sampling, VirtualPath]:
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
        datatables[sampling] = toc_path.parent.joinpath(datatable_path)

    return datatables
