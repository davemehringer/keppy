"""
keppy.viz — Matplotlib-based visualization for N-body simulations.

Provides four plotting functions:

    plot_trajectory(records, ...)   → axes  2-D or 3-D body paths
    plot_orbit(elements, mu, ...)   → axes  orbital ellipse from elements
    plot_system(system, ...)        → axes  snapshot of current positions
    plot_energy(records, mus, ...)  → axes  energy conservation diagnostic

All functions return the ``matplotlib.axes.Axes`` (or ``Axes3D``) they drew
on, so callers can compose multiple plots onto a single figure and control
``plt.show()`` / ``fig.savefig()`` themselves.

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

__all__ = [
    "plot_trajectory",
    "plot_orbit",
    "plot_system",
    "plot_energy",
]
