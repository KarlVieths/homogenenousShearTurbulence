"""Analysis tools for the ``hst`` DNS output.

Each tool is a module exposing a ``run(config, args) -> int`` entry point and
registered here in ``TOOLS``.  Adding a tool is adding one file in this package
and one line in ``TOOLS``; the dispatcher in :mod:`post.__main__` then calls it
as ``python3 -m post <name> [deck] [args]``.

Tools that need heavy dependencies (``matplotlib``, ``scipy``) import them
locally inside ``run`` so the base package stays importable with only numpy.
"""

from . import plot_stats
from . import slices

TOOLS = {
    "plot_stats": plot_stats,
    "slices": slices,
}

__all__ = ["TOOLS", "plot_stats", "slices"]
