"""
keppy.viz.plot — Matplotlib-based visualization functions.

Four public functions:

    plot_trajectory(records, ...)   — body paths from a StepRecord list
    plot_orbit(elements, mu, ...)   — orbital ellipse from Elements
    plot_system(system, ...)        — snapshot of NBodySystem positions
    plot_energy(records, mus, ...)  — total-energy conservation diagnostic

Unit helpers
------------
Pass ``unit="au"`` / ``"km"`` / ``"m"`` (default ``"m"``) to rescale
spatial axes.  For time axes (``plot_energy``), use ``time_unit="day"``
/ ``"year"`` / ``"h"`` / ``"s"`` (default ``"s"``).
"""

from __future__ import annotations

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3D projection
except ModuleNotFoundError as _exc:
    raise ModuleNotFoundError(
        "keppy.viz requires matplotlib.  "
        "Install it with: pip install \"keppy[viz]\""
    ) from _exc

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from keppy.timestep import StepRecord
    from keppy.orbital import Elements
    from keppy.nbody_system import NBodySystem

# ---------------------------------------------------------------------------
# Unit tables
# ---------------------------------------------------------------------------

_SPATIAL_UNITS: dict[str, tuple[float, str]] = {
    "m":  (1.0,                    "m"),
    "km": (1e-3,                   "km"),
    "au": (1.0 / 1.495_978_707e11, "AU"),
}

_TIME_UNITS: dict[str, tuple[float, str]] = {
    "s":    (1.0,          "s"),
    "h":    (1.0 / 3600,   "h"),
    "day":  (1.0 / 86400,  "days"),
    "year": (1.0 / 3.156e7,"years"),
}

_DEFAULT_COLORS = list(mcolors.TABLEAU_COLORS.values())


def _spatial_scale(unit: str) -> tuple[float, str]:
    key = unit.lower()
    if key not in _SPATIAL_UNITS:
        raise ValueError(
            f"Unknown spatial unit {unit!r}. "
            f"Choose from: {list(_SPATIAL_UNITS)}"
        )
    return _SPATIAL_UNITS[key]


def _time_scale(unit: str) -> tuple[float, str]:
    key = unit.lower()
    if key not in _TIME_UNITS:
        raise ValueError(
            f"Unknown time unit {unit!r}. "
            f"Choose from: {list(_TIME_UNITS)}"
        )
    return _TIME_UNITS[key]


# ---------------------------------------------------------------------------
# plot_trajectory
# ---------------------------------------------------------------------------

def plot_trajectory(
    records: Sequence["StepRecord"],
    *,
    body_names: list[str] | None = None,
    plane: str = "xy",
    ax=None,
    unit: str = "au",
    title: str = "Trajectory",
    legend: bool = True,
    mark_start: bool = True,
) -> "matplotlib.axes.Axes":
    """
    Plot body trajectories from a list of :class:`~keppy.timestep.StepRecord`
    snapshots.

    Parameters
    ----------
    records     : Snapshots from :func:`~keppy.timestep.run`.
    body_names  : Display names for each body.  Defaults to ``"Body 0"``,
                  ``"Body 1"``, etc.
    plane       : Projection plane — ``"xy"``, ``"xz"``, ``"yz"``, or
                  ``"3d"`` for a full 3-D plot.
    ax          : Existing :class:`~matplotlib.axes.Axes` to draw on.
                  Created automatically if ``None``.  For ``plane="3d"``
                  a 3-D axes is required; pass ``None`` to auto-create one.
    unit        : Spatial unit for axis labels: ``"m"``, ``"km"``, ``"au"``.
    title       : Figure title.
    legend      : Whether to add a legend.
    mark_start  : Whether to mark the starting position of each body with
                  a filled circle.

    Returns
    -------
    matplotlib.axes.Axes or Axes3D
    """
    if not records:
        raise ValueError("records is empty.")

    scale, unit_label = _spatial_scale(unit)
    n_bodies = records[0].positions.shape[0]

    if body_names is None:
        body_names = [f"Body {i}" for i in range(n_bodies)]
    elif len(body_names) != n_bodies:
        raise ValueError(
            f"body_names has {len(body_names)} entries but records have "
            f"{n_bodies} bodies."
        )

    is_3d = plane.lower() == "3d"

    if ax is None:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
    elif is_3d and not hasattr(ax, "set_zlabel"):
        raise ValueError(
            "plane='3d' requires a 3-D axes (Axes3D).  "
            "Create one with: fig.add_subplot(111, projection='3d')."
        )

    # Shape: (n_steps, n_bodies, 3)
    positions = np.stack([r.positions for r in records])  # (n_steps, n_bodies, 3)

    plane_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    axis_labels = ["x", "y", "z"]

    for bi in range(n_bodies):
        color = _DEFAULT_COLORS[bi % len(_DEFAULT_COLORS)]
        pos_b = positions[:, bi, :] * scale   # (n_steps, 3)

        if is_3d:
            ax.plot(pos_b[:, 0], pos_b[:, 1], pos_b[:, 2],
                    color=color, label=body_names[bi], linewidth=0.8)
            if mark_start:
                ax.scatter(*pos_b[0], color=color, s=30, zorder=5)
        else:
            ix, iy = plane_map[plane.lower()]
            ax.plot(pos_b[:, ix], pos_b[:, iy],
                    color=color, label=body_names[bi], linewidth=0.8)
            if mark_start:
                ax.scatter(pos_b[0, ix], pos_b[0, iy],
                           color=color, s=30, zorder=5)

    if is_3d:
        ax.set_xlabel(f"x ({unit_label})")
        ax.set_ylabel(f"y ({unit_label})")
        ax.set_zlabel(f"z ({unit_label})")
    else:
        ix, iy = plane_map[plane.lower()]
        ax.set_xlabel(f"{axis_labels[ix]} ({unit_label})")
        ax.set_ylabel(f"{axis_labels[iy]} ({unit_label})")
        ax.set_aspect("equal")

    ax.set_title(title)
    if legend:
        ax.legend(fontsize="small")

    return ax


# ---------------------------------------------------------------------------
# plot_orbit
# ---------------------------------------------------------------------------

def plot_orbit(
    elements: "Elements",
    mu: float,
    *,
    n_points: int = 360,
    plane: str = "xy",
    ax=None,
    unit: str = "au",
    label: str | None = None,
    **plot_kwargs,
) -> "matplotlib.axes.Axes":
    """
    Plot an orbital ellipse (or arc) from :class:`~keppy.orbital.Elements`.

    The orbit is sampled at *n_points* equally-spaced mean anomalies.
    Hyperbolic orbits (e ≥ 1) are skipped with a ``ValueError``.

    Parameters
    ----------
    elements   : Keplerian orbital elements (a in meters, angles in degrees).
    mu         : Gravitational parameter G·M_total (m³ s⁻²).
    n_points   : Number of sample points around the orbit.
    plane      : Projection plane — ``"xy"``, ``"xz"``, ``"yz"``, or ``"3d"``.
    ax         : Existing axes to draw on; created automatically if ``None``.
    unit       : Spatial unit for axis labels.
    label      : Line label for the legend.  Defaults to ``"orbit"``.
    **plot_kwargs : Passed directly to ``ax.plot()``.

    Returns
    -------
    matplotlib.axes.Axes or Axes3D

    Raises
    ------
    ValueError
        For hyperbolic orbits (e ≥ 1).
    """
    from keppy.orbital import Elements, elements_to_vectors

    if elements.e >= 1.0:
        raise ValueError(
            f"plot_orbit only supports elliptic orbits (e < 1); got e={elements.e}."
        )

    scale, unit_label = _spatial_scale(unit)
    is_3d = plane.lower() == "3d"

    if ax is None:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    # Sample positions around the full orbit
    M_values = np.linspace(0.0, 360.0, n_points, endpoint=False)
    positions = []
    for M_deg in M_values:
        el = Elements(
            a    = elements.a,
            e    = elements.e,
            i    = elements.i,
            node = elements.node if not math.isnan(elements.node) else 0.0,
            peri = elements.peri if not math.isnan(elements.peri) else 0.0,
            M    = M_deg,
        )
        pos, _ = elements_to_vectors(mu, el)
        positions.append(pos)

    positions = np.array(positions) * scale   # (n_points, 3)
    # Close the curve
    positions = np.vstack([positions, positions[[0]]])

    kw = {"linewidth": 1.0}
    kw.update(plot_kwargs)
    lbl = label if label is not None else "orbit"

    plane_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    axis_labels = ["x", "y", "z"]

    if is_3d:
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                label=lbl, **kw)
        ax.set_xlabel(f"x ({unit_label})")
        ax.set_ylabel(f"y ({unit_label})")
        ax.set_zlabel(f"z ({unit_label})")
    else:
        ix, iy = plane_map[plane.lower()]
        ax.plot(positions[:, ix], positions[:, iy], label=lbl, **kw)
        ax.set_xlabel(f"{axis_labels[ix]} ({unit_label})")
        ax.set_ylabel(f"{axis_labels[iy]} ({unit_label})")
        ax.set_aspect("equal")

    return ax


# ---------------------------------------------------------------------------
# plot_system
# ---------------------------------------------------------------------------

def plot_system(
    system: "NBodySystem",
    *,
    plane: str = "xy",
    ax=None,
    unit: str = "au",
    title: str = "System snapshot",
    annotate: bool = True,
) -> "matplotlib.axes.Axes":
    """
    Plot the current positions of all bodies in an
    :class:`~keppy.NBodySystem`.

    Parameters
    ----------
    system   : The N-body system to snapshot.
    plane    : Projection plane — ``"xy"``, ``"xz"``, ``"yz"``, or ``"3d"``.
    ax       : Existing axes; created automatically if ``None``.
    unit     : Spatial unit for axis labels.
    title    : Axes title.
    annotate : Label each body with its name.

    Returns
    -------
    matplotlib.axes.Axes or Axes3D
    """
    scale, unit_label = _spatial_scale(unit)
    is_3d = plane.lower() == "3d"

    if ax is None:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    plane_map  = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    axis_labels = ["x", "y", "z"]

    for bi, body in enumerate(system.bodies):
        pos   = body.position * scale
        color = _DEFAULT_COLORS[bi % len(_DEFAULT_COLORS)]

        # Marker size proportional to log(radius) when radius is known
        ms = 8
        if body.radius > 0:
            ms = max(4, min(20, 4 + 4 * math.log10(body.radius / 1e6 + 1)))

        if is_3d:
            ax.scatter(*pos, color=color, s=ms ** 2, zorder=5)
            if annotate:
                ax.text(pos[0], pos[1], pos[2], f"  {body.name}",
                        fontsize=7, color=color)
        else:
            ix, iy = plane_map[plane.lower()]
            ax.scatter(pos[ix], pos[iy], color=color, s=ms ** 2, zorder=5)
            if annotate:
                ax.annotate(body.name, (pos[ix], pos[iy]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=color)

    if is_3d:
        ax.set_xlabel(f"x ({unit_label})")
        ax.set_ylabel(f"y ({unit_label})")
        ax.set_zlabel(f"z ({unit_label})")
    else:
        ix, iy = plane_map[plane.lower()]
        ax.set_xlabel(f"{axis_labels[ix]} ({unit_label})")
        ax.set_ylabel(f"{axis_labels[iy]} ({unit_label})")
        ax.set_aspect("equal")

    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# plot_energy
# ---------------------------------------------------------------------------

def _compute_total_energy(records: Sequence["StepRecord"], mus: np.ndarray) -> np.ndarray:
    """
    Compute total mechanical energy (KE + PE) at each step.

    Parameters
    ----------
    records : Trajectory snapshots.
    mus     : Gravitational parameters for each body, shape (n_bodies,).

    Returns
    -------
    energies : shape (n_steps,), Joules (SI).
    """
    from keppy.constants import G

    mus    = np.asarray(mus, dtype=float)
    masses = mus / G
    n      = len(masses)
    energies = np.empty(len(records))

    for step_i, r in enumerate(records):
        pos = r.positions   # (n, 3)
        vel = r.velocities  # (n, 3)

        # Kinetic energy
        ke = 0.5 * float(np.dot(masses, np.sum(vel ** 2, axis=1)))

        # Potential energy  (O(n²) pair sum)
        pe = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(pos[i] - pos[j]))
                pe -= G * masses[i] * masses[j] / d

        energies[step_i] = ke + pe

    return energies


def plot_energy(
    records: Sequence["StepRecord"],
    mus: Sequence[float] | np.ndarray,
    *,
    ax=None,
    time_unit: str = "day",
    relative: bool = True,
    title: str = "Energy conservation",
    **plot_kwargs,
) -> "matplotlib.axes.Axes":
    """
    Plot total mechanical energy (or relative energy error) vs time.

    This is the primary diagnostic for checking integrator quality:
    a symplectic integrator (Leapfrog, Yoshida) shows bounded oscillations;
    a dissipative one (RK4) shows secular drift.

    Parameters
    ----------
    records     : Trajectory snapshots from :func:`~keppy.timestep.run`.
    mus         : Gravitational parameters (m³ s⁻²) for each body,
                  in the same order as ``records[0].positions``.
    ax          : Existing axes; created automatically if ``None``.
    time_unit   : Time axis unit — ``"s"``, ``"h"``, ``"day"``, ``"year"``.
    relative    : If ``True`` (default), plot ``(E - E₀) / |E₀|``
                  (dimensionless relative error).  If ``False``, plot the
                  raw energy in Joules.
    title       : Axes title.
    **plot_kwargs : Passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not records:
        raise ValueError("records is empty.")

    t_scale, t_label = _time_scale(time_unit)
    times    = np.array([r.time for r in records]) * t_scale
    energies = _compute_total_energy(records, mus)

    if ax is None:
        _, ax = plt.subplots()

    kw = {"linewidth": 1.0, "color": "steelblue"}
    kw.update(plot_kwargs)

    if relative:
        E0 = energies[0]
        if E0 == 0.0:
            raise ValueError("Initial energy is zero; cannot compute relative error.")
        y     = (energies - E0) / abs(E0)
        ylabel = "(E − E₀) / |E₀|"
    else:
        y      = energies
        ylabel = "Total energy (J)"

    ax.plot(times, y, **kw)
    ax.set_xlabel(f"Time ({t_label})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")

    return ax
