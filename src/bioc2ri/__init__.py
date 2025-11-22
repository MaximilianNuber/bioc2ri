import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .engine import Engine
from .base_plugin import base_plugin
from .numpy_plugin import numpy_plugin
from .pandas_plugin import pandas_plugin
from .biocpy_plugin import biocpy_plugin
from .scipy_sparse_plugin import scipy_sparse_plugin

__all__ = [
    "Engine",
    "numpy_plugin",
    "pandas_plugin",
    "biocpy_plugin",
]
