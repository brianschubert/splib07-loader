"""Misc utilities."""
from __future__ import annotations

import os
import pathlib
import zipfile
from typing import Union

from typing_extensions import TypeAlias

VirtualPath: TypeAlias = Union[pathlib.Path, zipfile.Path]
"""Concrete filesystem or zipfile path."""

PathLike: TypeAlias = Union[str, os.PathLike[str]]
"""Path-representing type, excluding bytes."""


def resolve_zip_path(path: PathLike) -> VirtualPath:
    resolved_path: VirtualPath
    if zipfile.is_zipfile(path):
        resolved_path = zipfile.Path(path, at="")
    else:
        resolved_path = pathlib.Path(path)

    return resolved_path
