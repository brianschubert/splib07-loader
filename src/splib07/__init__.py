"""
Tools for loading spectra from a local archive of the `USGS Spectral Library Version 7`_.

.. _USGS Spectral Library Version 7: https://pubs.er.usgs.gov/publication/ds1035
"""
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

import importlib.metadata

from ._loader import Spectrum, Splib07
from ._index import Sampling

# Distribution version, PEP-440 compatible.
# Automatically retrieved from installed distribution metadata.
__version__ = importlib.metadata.version("splib07-loader")

__all__ = ["Spectrum", "Splib07", "Sampling"]
