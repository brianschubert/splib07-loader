"""
Tools for loading spectra from a local archive of the `USGS Spectral Library Version 7`_.

.. _USGS Spectral Library Version 7: https://pubs.er.usgs.gov/publication/ds1035
"""

import importlib.metadata

from ._loader import Spectrum, Splib07

# Distribution version, PEP-440 compatible.
# Automatically retrieved from installed distribution metadata.
__version__ = importlib.metadata.version("splib07-loader")

__all__ = ["Spectrum", "Splib07"]
