"""Misc utilities."""

import pathlib
import zipfile
from typing import Union

from typing_extensions import TypeAlias

VirtualPath: TypeAlias = Union[pathlib.Path, zipfile.Path]
"""Filesystem or zipfile path."""
