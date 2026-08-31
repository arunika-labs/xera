from . import functional
from . import loom
from . import weave
from . import io

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("xera")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "functional",
    "loom",
    "weave",
    "io",
]
