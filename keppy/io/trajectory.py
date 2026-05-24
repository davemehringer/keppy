"""
keppy.io.trajectory — Save and load simulation trajectory snapshots.

Two output formats:

``save_trajectory`` / ``load_trajectory``
    NumPy ``.npz`` compressed binary.  Zero additional dependencies,
    compact file size, exact floating-point round-trip.  Recommended for
    programmatic use.

``save_trajectory_csv``
    Comma-separated text.  Human-readable, easy to open in a spreadsheet
    or plotting tool, but produces much larger files for long runs.

Column layout of CSV output
---------------------------
    step, time_s, dt_used_s, body_index, body_name,
    x_m, y_m, z_m, vx_mps, vy_mps, vz_mps

One row per (step × body) pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from keppy.timestep import StepRecord


# ---------------------------------------------------------------------------
# NumPy .npz format
# ---------------------------------------------------------------------------

def save_trajectory(
    records: Sequence[StepRecord],
    path: str | Path,
) -> None:
    """
    Save a list of :class:`~keppy.timestep.StepRecord` snapshots to a
    compressed NumPy ``.npz`` file.

    Arrays stored
    -------------
    ``times``      : shape (n_steps,)            — simulation time in seconds.
    ``dt_used``    : shape (n_steps,)            — actual step taken in seconds.
    ``positions``  : shape (n_steps, n_bodies, 3) — positions in meters.
    ``velocities`` : shape (n_steps, n_bodies, 3) — velocities in m s⁻¹.

    Parameters
    ----------
    records : Snapshots returned by :func:`~keppy.timestep.run`.
    path    : Destination path.  NumPy appends ``.npz`` if the path does not
              already end with that extension.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("records is empty; nothing to save.")

    times      = np.array([r.time     for r in records])
    dt_used    = np.array([r.dt_used  for r in records])
    positions  = np.stack([r.positions  for r in records])   # (n_steps, n_bodies, 3)
    velocities = np.stack([r.velocities for r in records])   # (n_steps, n_bodies, 3)

    np.savez_compressed(
        path,
        times      = times,
        dt_used    = dt_used,
        positions  = positions,
        velocities = velocities,
    )


def load_trajectory(path: str | Path) -> list[StepRecord]:
    """
    Load :class:`~keppy.timestep.StepRecord` snapshots from a ``.npz``
    trajectory file written by :func:`save_trajectory`.

    Parameters
    ----------
    path : Path to the ``.npz`` file.

    Returns
    -------
    list of StepRecord
    """
    data       = np.load(path)
    times      = data["times"]
    dt_used    = data["dt_used"]
    positions  = data["positions"]
    velocities = data["velocities"]

    return [
        StepRecord(
            time       = float(times[i]),
            dt_used    = float(dt_used[i]),
            positions  = positions[i].copy(),
            velocities = velocities[i].copy(),
        )
        for i in range(len(times))
    ]


# ---------------------------------------------------------------------------
# CSV format
# ---------------------------------------------------------------------------

def save_trajectory_csv(
    records: Sequence[StepRecord],
    path: str | Path,
    *,
    body_names: list[str] | None = None,
) -> None:
    """
    Save a list of :class:`~keppy.timestep.StepRecord` snapshots to a CSV
    file.

    Each accepted step produces one row per body:

        step, time_s, dt_used_s, body_index, body_name,
        x_m, y_m, z_m, vx_mps, vy_mps, vz_mps

    Parameters
    ----------
    records     : Snapshots returned by :func:`~keppy.timestep.run`.
    path        : Destination path.  The file is overwritten if it exists.
    body_names  : Display names for the bodies.  If ``None``, bodies are
                  labelled ``"0"``, ``"1"``, etc.  Length must match the
                  number of bodies in the records.

    Raises
    ------
    ValueError
        If *records* is empty, or if the length of *body_names* does not
        match the number of bodies.
    """
    if not records:
        raise ValueError("records is empty; nothing to save.")

    n_bodies = records[0].positions.shape[0]
    if body_names is None:
        body_names = [str(i) for i in range(n_bodies)]
    elif len(body_names) != n_bodies:
        raise ValueError(
            f"body_names has {len(body_names)} entries but records have "
            f"{n_bodies} bodies."
        )

    rows = [
        "step,time_s,dt_used_s,body_index,body_name,"
        "x_m,y_m,z_m,vx_mps,vy_mps,vz_mps"
    ]
    for step_idx, r in enumerate(records):
        for body_idx in range(n_bodies):
            pos = r.positions[body_idx]
            vel = r.velocities[body_idx]
            # Cast to plain Python float so repr() gives clean decimal literals,
            # not the numpy-annotated form "np.float64(...)".
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
            rows.append(
                f"{step_idx},{float(r.time)!r},{float(r.dt_used)!r},{body_idx},"
                f"{body_names[body_idx]},"
                f"{px!r},{py!r},{pz!r},"
                f"{vx!r},{vy!r},{vz!r}"
            )

    Path(path).write_text("\n".join(rows) + "\n")
