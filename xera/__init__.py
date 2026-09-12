from . import functional
from . import loom
from . import optimizer
from . import io
from ._kernel.shard import shard

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("xera")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "functional",
    "loom",
    "optimizer",
    "io",
    "shard",
]
