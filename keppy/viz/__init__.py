"""
keppy.viz — Matplotlib-based visualization for N-body simulations.

Static plots
------------
    plot_trajectory(records, ...)   → axes  2-D or 3-D body paths
    plot_orbit(elements, mu, ...)   → axes  orbital ellipse from elements
    plot_system(system, ...)        → axes  snapshot of current positions
    plot_energy(records, mus, ...)  → axes  energy conservation diagnostic

Interactive animated viewer
---------------------------
    OrbitViewer(records, ...)       — live orbit plot with controls:
                                      • CheckButtons: toggle body labels
                                      • RadioButtons: change center body
                                      • FuncAnimation: animated playback

All static functions return the ``matplotlib.axes.Axes`` they drew on.
``OrbitViewer.animate()`` returns a ``FuncAnimation`` — assign it to a
variable before calling ``plt.show()`` to prevent garbage collection.

Optional dependency
-------------------
matplotlib is not installed by keppy by default.  Install it with::

    pip install "keppy[viz]"

A ``ModuleNotFoundError`` is raised at import time if matplotlib is absent.
"""

from keppy.viz.plot import (
    plot_trajectory,
    plot_orbit,
    plot_system,
    plot_energy,
)
from keppy.viz.interactive import OrbitViewer

__all__ = [
    "plot_trajectory",
    "plot_orbit",
    "plot_system",
    "plot_energy",
    "OrbitViewer",
]
