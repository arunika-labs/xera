

from . import initializers
from . import loom
from . import weave

# Convenience aliases: `import xera; xera.L.Module`, `xera.W.Train`,
# so the high-level namespace mirrors `import xera.loom as L` / `import xera.weave as W`.
L = loom
W = weave

__version__ = "0.0.2"

__all__ = [
    "initializers",
    "loom",
    "weave",
    "L",
    "W",
]