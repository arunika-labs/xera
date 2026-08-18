from . import initializers
from . import loom
from . import weave
from . import serialize

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("xera")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "initializers",
    "loom",
    "weave",
    "serialize",
]
