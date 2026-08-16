

from . import initializers
from . import loom
from . import weave
from . import serialize

# Convenience aliases: `import xera; xera.L.Module`, `xera.W.Train`,
# so the high-level namespace mirrors `import xera.loom as L` / `import xera.weave as W`.
L = loom
W = weave
S = serialize

__version__ = "0.0.2"

__all__ = [
    "initializers",
    "loom",
    "weave",
    "serialize",
    "L",
    "W",
    "S",
]