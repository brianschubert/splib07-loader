"""Misc utilities."""
# Copyright (C) 2023 Brian Schubert.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
