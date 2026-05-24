"""
TimeStepManager — controls adaptive time step sizing for fixed-step integrators.

The C++ keplerpp code used a class hierarchy with pure-virtual modify() that
returned an enum (NO_CHANGE / INCREASE / DECREASE), requiring the integrator to
query a getDeltaT() accessor and handle the retry loop inline.

Improvements over C++:
  * TimeStepManager is a Protocol — any callable object with the right method
    satisfies it without forced inheritance.
  * update() returns a ChangeType AND updates self.dt atomically; no separate
    getDeltaT() call needed.
  * Both threshold-based (AccelerationTimeStepManager, faithful to the C++ logic)
    and continuous-scaling (ScaledTimeStepManager, smoother) variants included.
  * min_dt / max_dt guard rails on all concrete managers — the C++ code had none.
  * A standalone run() function encapsulates the retry loop so callers do not
    need to reimplement it.

Usage
-----
    from keppy import NBodySystem, LeapfrogIntegrator, PairwiseAccelerationCalculator
    from keppy.timestep import AccelerationTimeStepManager, run
    from keppy.constants import DAY

    sys   = NBodySystem(...)
    calc  = PairwiseAccelerationCalculator()
    integ = LeapfrogIntegrator(calc)
    tsm   = AccelerationTimeStepManager(dt_init=DAY, a_min=0.01, a_max=0.1)

    trajectory = run(sys, integ, tsm, t_end=365 * DAY)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from keppy.nbody_system import NBodySystem
    from keppy.integrator import Integrator


# ---------------------------------------------------------------------------
# ChangeType enum
# ---------------------------------------------------------------------------

class ChangeType(Enum):
    """Result of a TimeStepManager.update() call."""
    NO_CHANGE = auto()
    INCREASE  = auto()
    DECREASE  = auto()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TimeStepManager(Protocol):
    """
    Interface for time step managers.

    Any object that exposes a ``dt`` property and an ``update()`` method
    satisfies this protocol.
    """

    @property
    def dt(self) -> float:
        """Current recommended time step (seconds)."""
        ...

    def update(
        self,
        acc_before: np.ndarray,
        acc_after: np.ndarray,
    ) -> ChangeType:
        """
        Evaluate the quality of the last step and update *dt*.

        Parameters
        ----------
        acc_before : shape (n, 3)  Accelerations at the start of the step.
        acc_after  : shape (n, 3)  Accelerations at the end of the step.

        Returns
        -------
        ChangeType
            NO_CHANGE — step acceptable, dt unchanged.
            INCREASE  — step acceptable, dt was increased for next step.
            DECREASE  — step was too large, dt was decreased; caller should
                        roll back the system state and retry.
        """
        ...


# ---------------------------------------------------------------------------
# ConstantTimeStepManager
# ---------------------------------------------------------------------------

class ConstantTimeStepManager:
    """
    No-op manager: dt never changes.

    Satisfies the TimeStepManager protocol and is the right choice when you
    want a fixed step size but still want to use the run() helper.

    Parameters
    ----------
    dt : Fixed step size in seconds.
    """

    def __init__(self, dt: float) -> None:
        self._dt = float(dt)

    @property
    def dt(self) -> float:
        return self._dt

    def update(
        self,
        acc_before: np.ndarray,
        acc_after: np.ndarray,
    ) -> ChangeType:
        return ChangeType.NO_CHANGE


# ---------------------------------------------------------------------------
# AccelerationTimeStepManager
# ---------------------------------------------------------------------------

class AccelerationTimeStepManager:
    """
    Threshold-based adaptive step manager (faithful to C++ keplerpp logic).

    After each step, computes the *fractional acceleration change* for each
    body:

        q_i = max_component( |a_after_i - a_before_i| / |a_before_i| )

    where the maximum is taken over the three Cartesian components.

    * If ANY body has q_i > ``a_max``  → decrease dt by ``decrease_factor``
      and signal DECREASE (caller should retry the step).
    * If ALL bodies have q_i < ``a_min`` → increase dt by ``increase_factor``
      and signal INCREASE.
    * Otherwise → NO_CHANGE.

    Improvement over C++: ``min_dt`` and ``max_dt`` guard rails prevent dt
    from leaving a sensible range.

    Parameters
    ----------
    dt_init         : Initial time step (seconds).
    a_min           : Lower threshold; below this the step is "too easy" and
                      dt is increased (dimensionless fraction, e.g. 0.01).
    a_max           : Upper threshold; above this the step is "too hard" and
                      dt is decreased (dimensionless fraction, e.g. 0.1).
    increase_factor : Multiplicative factor when increasing dt (default 2.0).
    decrease_factor : Multiplicative factor when decreasing dt (default 2.0).
    min_dt          : Hard lower bound on dt (default 1.0 s).
    max_dt          : Hard upper bound on dt (default None = unlimited).
    """

    def __init__(
        self,
        dt_init: float,
        a_min: float,
        a_max: float,
        increase_factor: float = 2.0,
        decrease_factor: float = 2.0,
        min_dt: float = 1.0,
        max_dt: float | None = None,
    ) -> None:
        if a_min >= a_max:
            raise ValueError(f"a_min ({a_min}) must be less than a_max ({a_max})")
        self._dt        = float(dt_init)
        self._a_min     = float(a_min)
        self._a_max     = float(a_max)
        self._inc       = float(increase_factor)
        self._dec       = float(decrease_factor)
        self._min_dt    = float(min_dt)
        self._max_dt    = float(max_dt) if max_dt is not None else np.inf

    @property
    def dt(self) -> float:
        return self._dt

    def update(
        self,
        acc_before: np.ndarray,
        acc_after: np.ndarray,
    ) -> ChangeType:
        """
        Evaluate step quality and update dt.

        Uses the same threshold logic as keplerpp's AccelerationTimeStepManager.
        """
        acc_before = np.asarray(acc_before, dtype=float)
        acc_after  = np.asarray(acc_after,  dtype=float)

        # Magnitude of initial accelerations per body, shape (n,)
        mag = np.linalg.norm(acc_before, axis=1)

        # Avoid division by zero for stationary / zero-force bodies
        nonzero = mag > 0.0
        if not np.any(nonzero):
            return ChangeType.NO_CHANGE

        # Fractional change per component, shape (n, 3)
        diff = np.abs(acc_after - acc_before)
        frac = np.zeros_like(diff)
        frac[nonzero] = diff[nonzero] / mag[nonzero, np.newaxis]

        # Maximum fractional change over all components, per body, shape (n,)
        q = frac.max(axis=1)

        if np.any(q[nonzero] > self._a_max):
            self._dt = max(self._dt / self._dec, self._min_dt)
            return ChangeType.DECREASE

        if np.all(q[nonzero] < self._a_min):
            self._dt = min(self._dt * self._inc, self._max_dt)
            return ChangeType.INCREASE

        return ChangeType.NO_CHANGE


# ---------------------------------------------------------------------------
# ScaledTimeStepManager
# ---------------------------------------------------------------------------

class ScaledTimeStepManager:
    """
    Smooth continuous-scaling adaptive step manager.

    Instead of binary thresholds the step size is continuously scaled to
    drive the RMS fractional acceleration change toward a target value:

        q = rms_over_bodies( rms_over_components( |Δa| / |a| ) )

        dt_new = dt * (q_target / q) ** exponent

    This gives smoother step-size evolution than the factor-of-2 jumps in
    ``AccelerationTimeStepManager`` and avoids oscillating between INCREASE
    and DECREASE states near the threshold boundary.

    A step is declared too large (DECREASE) when q > ``reject_factor *
    q_target``; in that case the caller should retry with the new (smaller)
    dt.

    Parameters
    ----------
    dt_init       : Initial time step (seconds).
    q_target      : Target RMS fractional acceleration change per step
                    (dimensionless; 0.02–0.05 is a good range for orbits).
    exponent      : Controls scaling aggressiveness (default 0.3; higher =
                    more aggressive, but can lead to oscillation).
    reject_factor : Step is rejected when q > reject_factor * q_target
                    (default 3.0).
    min_dt        : Hard lower bound on dt (default 1.0 s).
    max_dt        : Hard upper bound on dt (default None = unlimited).
    max_factor    : Maximum growth per step (default 5.0).
    min_factor    : Minimum shrinkage per step (default 0.1).
    """

    def __init__(
        self,
        dt_init: float,
        q_target: float = 0.03,
        exponent: float = 0.3,
        reject_factor: float = 3.0,
        min_dt: float = 1.0,
        max_dt: float | None = None,
        max_factor: float = 5.0,
        min_factor: float = 0.1,
    ) -> None:
        self._dt           = float(dt_init)
        self._q_target     = float(q_target)
        self._exponent     = float(exponent)
        self._reject       = float(reject_factor)
        self._min_dt       = float(min_dt)
        self._max_dt       = float(max_dt) if max_dt is not None else np.inf
        self._max_factor   = float(max_factor)
        self._min_factor   = float(min_factor)

    @property
    def dt(self) -> float:
        return self._dt

    def update(
        self,
        acc_before: np.ndarray,
        acc_after: np.ndarray,
    ) -> ChangeType:
        acc_before = np.asarray(acc_before, dtype=float)
        acc_after  = np.asarray(acc_after,  dtype=float)

        mag = np.linalg.norm(acc_before, axis=1)
        nonzero = mag > 0.0
        if not np.any(nonzero):
            return ChangeType.NO_CHANGE

        # RMS fractional change per body, then RMS over bodies → scalar q
        diff = np.abs(acc_after[nonzero] - acc_before[nonzero])
        frac = diff / mag[nonzero, np.newaxis]          # (n_nz, 3)
        q_per_body = np.sqrt((frac ** 2).mean(axis=1))  # (n_nz,)
        q = float(np.sqrt((q_per_body ** 2).mean()))

        if q == 0.0:
            # Perfectly linear step: allow growth up to max_factor
            scale = self._max_factor
        else:
            scale = (self._q_target / q) ** self._exponent
            scale = np.clip(scale, self._min_factor, self._max_factor)

        reject_threshold = self._reject * self._q_target

        if q > reject_threshold:
            # Step too large — decrease and signal retry
            self._dt = max(self._dt * scale, self._min_dt)
            return ChangeType.DECREASE

        old_dt = self._dt
        self._dt = float(np.clip(self._dt * scale, self._min_dt, self._max_dt))

        if self._dt > old_dt * 1.001:
            return ChangeType.INCREASE
        if self._dt < old_dt * 0.999:
            return ChangeType.DECREASE   # within tolerance, no retry needed
        return ChangeType.NO_CHANGE


# ---------------------------------------------------------------------------
# Simulation run() helper
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Single-step snapshot stored by run()."""
    time      : float
    dt_used   : float
    positions : np.ndarray   # shape (n, 3)
    velocities: np.ndarray   # shape (n, 3)


def run(
    system: "NBodySystem",
    integrator: "Integrator",
    tsm: "TimeStepManager",
    t_end: float,
    *,
    output_every: int = 1,
    max_retries: int = 10,
) -> list[StepRecord]:
    """
    Run an adaptive simulation from *system.time* to *t_end*.

    The loop:
      1. Records the system state (so a retry can roll back).
      2. Reads the current acceleration from body objects (or computes it
         on the first call via the integrator's calculator).
      3. Calls ``system.step(tsm.dt, integrator)``.
      4. Reads the new acceleration from body objects.
      5. Calls ``tsm.update(a_before, a_after)``.
      6. If DECREASE and the step can be retried, rolls back and repeats.
      7. Appends a ``StepRecord`` every *output_every* accepted steps.

    Parameters
    ----------
    system      : NBodySystem to integrate.
    integrator  : Fixed-step integrator (Leapfrog, Yoshida, RK4, …).
    tsm         : TimeStepManager controlling dt.
    t_end       : Stop time in seconds.
    output_every: Record a snapshot every N accepted steps (default 1).
    max_retries : Maximum consecutive retries per step before giving up
                  and accepting the step anyway (default 10).

    Returns
    -------
    list of StepRecord
        Snapshots at every *output_every* accepted steps, plus the final step.

    Notes
    -----
    *Leapfrog* and *Yoshida* caches are reset on rollback so they
    recompute their initial accelerations cleanly.

    The integrators must expose ``body.acceleration`` fields (Leapfrog and
    Yoshida do; RK4 does not — RK4 is best paired with RK45Integrator for
    adaptive stepping instead).
    """
    from keppy.integrator import LeapfrogIntegrator, YoshidaIntegrator

    records: list[StepRecord] = []
    step_count = 0

    # Ensure body.acceleration fields are populated before the first step.
    # Leapfrog / Yoshida update them; for other integrators we warm them up.
    _ensure_accelerations(system, integrator)

    while system.time < t_end:
        # Clamp dt so we do not overshoot t_end
        dt_req = min(tsm.dt, t_end - system.time)

        # Save rollback state
        pos_saved, vel_saved = system.get_state()
        t_saved = system.time
        a_saved = _get_accelerations(system)

        retries = 0
        while True:
            # Reset integrator cache on retry or first call
            if retries > 0:
                _reset_integrator(integrator)
                system.set_state(pos_saved, vel_saved)
                system.time = t_saved

            a_before = _get_accelerations(system)
            system.step(dt_req, integrator)
            a_after = _get_accelerations(system)

            change = tsm.update(a_before, a_after)

            if change != ChangeType.DECREASE or retries >= max_retries:
                break

            retries += 1
            dt_req = min(tsm.dt, t_end - t_saved)

        step_count += 1
        if step_count % output_every == 0 or system.time >= t_end:
            pos, vel = system.get_state()
            records.append(StepRecord(
                time=system.time,
                dt_used=system.time - t_saved,
                positions=pos.copy(),
                velocities=vel.copy(),
            ))

    return records


# ---------------------------------------------------------------------------
# Internal helpers for run()
# ---------------------------------------------------------------------------

def _get_accelerations(system: "NBodySystem") -> np.ndarray:
    """Return stacked body accelerations, shape (n, 3)."""
    return np.stack([b.acceleration.copy() for b in system.bodies])


def _ensure_accelerations(system: "NBodySystem", integrator) -> None:
    """
    Make sure body.acceleration fields are non-zero before the first step.
    Leapfrog / Yoshida populate them during a step; if the integrator has
    a calculator attached we warm it up here.
    """
    from keppy.integrator import LeapfrogIntegrator, YoshidaIntegrator

    calc = getattr(integrator, "_calc", None)
    if calc is None:
        return

    # Only warm up if all accelerations are still zero
    accs = _get_accelerations(system)
    if np.any(accs != 0.0):
        return

    positions, _ = system.get_state()
    mus = np.array([b.mu for b in system.bodies])
    result = calc.compute(positions, mus)
    for i, b in enumerate(system.bodies):
        b.acceleration[:] = result[i]


def _reset_integrator(integrator) -> None:
    """Clear any cached state so the next step starts fresh."""
    if hasattr(integrator, "reset"):
        integrator.reset()
