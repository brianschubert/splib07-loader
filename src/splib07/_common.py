"""
Common types.
"""

import enum

from typing_extensions import TypeAlias

SpectrumIdentifier: TypeAlias = str
"""Unique identifier for a particular spectrum, shared across all available samplings."""


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
    """Datatable chapters in splib07."""

    MINERALS = 1

    SOILS_AND_MIXTURES = 2
    COATINGS = 3
    LIQUIDS = 4
    ORGANICS = 5
    ARTIFICIAL = 6
    VEGETATION = 7
