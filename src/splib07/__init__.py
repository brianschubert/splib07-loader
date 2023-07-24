import importlib.metadata

from ._loader import Spectrum, Splib07

# Distribution version, PEP-440 compatible.
# Automatically retrieved from installed distribution metadata.
__version__ = importlib.metadata.version("splib07-loader")

__all__ = ["Spectrum", "Splib07"]
