"""
Integrators — numerical methods for advancing an N-body system through time.

Each integrator implements the Protocol:

    step(system: NBodySystem, dt: float) -> float

It updates all body positions and velocities in-place and returns the actual
time step taken (which may differ from *dt* for adaptive methods).

Integrators provided
--------------------
RK4Integrator        Fixed-step classical 4th-order Runge-Kutta.
RK45Integrator       Adaptive Dormand-Prince RK4(5); automatically shrinks /
                     grows the step to meet a tolerance target.
LeapfrogIntegrator   Fixed-step symplectic 2nd-order (Velocity Verlet /
                     Störmer-Verlet KDK form).  Bounded energy error over
                     arbitrarily long integrations.
YoshidaIntegrator    Fixed-step symplectic 4th-order (Yoshida 1990).  Better
                     accuracy than leapfrog at the same step count, still
                     symplectic.

Choosing an integrator
----------------------
* Short integrations, arbitrary forces   → RK4Integrator
* Varying time scales, need accuracy     → RK45Integrator
* Long orbital integrations              → LeapfrogIntegrator  (safe default)
* Long + high accuracy                   → YoshidaIntegrator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from keppy.nbody_system import NBodySystem


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Integrator(Protocol):
    """
    Interface for integrators.

    Any object with a ``step(system, dt)`` method satisfies this protocol.
    """

    def step(self, system: "NBodySystem", dt: float) -> float:
        """
        Advance *system* by (up to) *dt* seconds.

        Returns the actual time step taken.  For fixed-step integrators this
        equals *dt*; for adaptive integrators it may be smaller.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers shared by all integrators
# ---------------------------------------------------------------------------

def _mus(system: "NBodySystem") -> np.ndarray:
    """Return gravitational parameters for all bodies, shape (n,)."""
    return np.array([b.mu for b in system.bodies])


# ---------------------------------------------------------------------------
# RK4 — fixed-step 4th-order Runge-Kutta
# ---------------------------------------------------------------------------

class RK4Integrator:
    """
    Classical 4th-order Runge-Kutta with a fixed step size.

    Global truncation error is O(dt⁴).  Suitable for short integrations or
    test/validation runs.  Not symplectic: energy error grows secularly over
    long runs.

    Parameters
    ----------
    calc : AccelerationCalculator
        Object with a ``compute(positions, mus)`` method.
    """

    def __init__(self, calc) -> None:
        self._calc = calc

    def step(self, system: "NBodySystem", dt: float) -> float:
        pos, vel = system.get_state()
        mus = _mus(system)

        def a(p: np.ndarray) -> np.ndarray:
            return self._calc.compute(p, mus)

        # Stage 1
        k1x = vel
        k1v = a(pos)

        # Stage 2
        k2x = vel + 0.5 * dt * k1v
        k2v = a(pos + 0.5 * dt * k1x)

        # Stage 3
        k3x = vel + 0.5 * dt * k2v
        k3v = a(pos + 0.5 * dt * k2x)

        # Stage 4
        k4x = vel + dt * k3v
        k4v = a(pos + dt * k3x)

        new_pos = pos + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        new_vel = vel + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        system.set_state(new_pos, new_vel)
        return dt


# ---------------------------------------------------------------------------
# RK45 — adaptive Dormand-Prince
# ---------------------------------------------------------------------------

# Dormand-Prince Butcher tableau (FSAL property: k7 of step n = k1 of step n+1)
_DP_A21 = 1 / 5
_DP_A31, _DP_A32 = 3 / 40, 9 / 40
_DP_A41, _DP_A42, _DP_A43 = 44 / 45, -56 / 15, 32 / 9
_DP_A51, _DP_A52, _DP_A53, _DP_A54 = (
    19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729,
)
_DP_A61, _DP_A62, _DP_A63, _DP_A64, _DP_A65 = (
    9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656,
)

# 5th-order weights
_DP_B1, _DP_B3, _DP_B4, _DP_B5, _DP_B6 = (
    35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84,
)

# Error coefficients  E = y5 - y4
_DP_E1  =  71 / 57600
_DP_E3  = -71 / 16695
_DP_E4  =  71 / 1920
_DP_E5  = -17253 / 339200
_DP_E6  =  22 / 525
_DP_E7  = -1 / 40


class RK45Integrator:
    """
    Adaptive Dormand-Prince RK4(5) integrator.

    On each call to ``step(system, dt)``, attempts a step of size *dt*.  If
    the local truncation error exceeds *tol*, the step is retried with a
    smaller size.  The method never takes a step larger than *dt* (it does
    not advance time beyond the requested point).

    The returned value is the actual step taken; callers that need output at
    exact times should accumulate leftover time themselves.

    Parameters
    ----------
    calc       : AccelerationCalculator
    rtol       : Relative tolerance (default 1e-9).
    atol       : Absolute tolerance in meters / m s⁻¹ (default 1e-3, ~ mm).
    min_dt     : Minimum allowed step size in seconds (default 1.0 s).
    safety     : Safety factor on step-size estimate (default 0.9).
    max_factor : Maximum step-size growth per step (default 5.0).
    min_factor : Minimum step-size reduction per step (default 0.2).
    """

    def __init__(
        self,
        calc,
        rtol: float = 1e-9,
        atol: float = 1e-3,
        min_dt: float = 1.0,
        safety: float = 0.9,
        max_factor: float = 5.0,
        min_factor: float = 0.2,
    ) -> None:
        self._calc = calc
        self._rtol = rtol
        self._atol = atol
        self._min_dt = min_dt
        self._safety = safety
        self._max_factor = max_factor
        self._min_factor = min_factor
        self._k1: np.ndarray | None = None  # FSAL cache

    def step(self, system: "NBodySystem", dt: float) -> float:
        pos, vel = system.get_state()
        mus = _mus(system)
        h = dt

        while True:
            pos_new, vel_new, err = self._dp_step(pos, vel, mus, h)
            if err <= 1.0 or h <= self._min_dt:
                break
            # Reduce step and retry
            factor = max(self._min_factor, self._safety * err ** -0.2)
            h = max(h * factor, self._min_dt)

        system.set_state(pos_new, vel_new)
        self._k1 = None  # clear FSAL cache (positions changed)
        return h

    def _dp_step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mus: np.ndarray,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Single Dormand-Prince step; returns (pos_new, vel_new, err_norm)."""

        def a(p: np.ndarray) -> np.ndarray:
            return self._calc.compute(p, mus)

        # Flatten state to a single array for uniform error scaling
        # State layout: first half = positions, second half = velocities
        k1x = vel
        k1v = a(pos)

        k2x = vel + h * _DP_A21 * k1v
        k2v = a(pos + h * _DP_A21 * k1x)

        k3x = vel + h * (_DP_A31 * k1v + _DP_A32 * k2v)
        k3v = a(pos + h * (_DP_A31 * k1x + _DP_A32 * k2x))

        k4x = vel + h * (_DP_A41 * k1v + _DP_A42 * k2v + _DP_A43 * k3v)
        k4v = a(pos + h * (_DP_A41 * k1x + _DP_A42 * k2x + _DP_A43 * k3x))

        k5x = vel + h * (_DP_A51*k1v + _DP_A52*k2v + _DP_A53*k3v + _DP_A54*k4v)
        k5v = a(pos + h * (_DP_A51*k1x + _DP_A52*k2x + _DP_A53*k3x + _DP_A54*k4x))

        k6x = vel + h * (
            _DP_A61*k1v + _DP_A62*k2v + _DP_A63*k3v +
            _DP_A64*k4v + _DP_A65*k5v
        )
        k6v = a(pos + h * (
            _DP_A61*k1x + _DP_A62*k2x + _DP_A63*k3x +
            _DP_A64*k4x + _DP_A65*k5x
        ))

        # 5th-order solution
        pos_new = pos + h * (_DP_B1*k1x + _DP_B3*k3x + _DP_B4*k4x +
                              _DP_B5*k5x + _DP_B6*k6x)
        vel_new = vel + h * (_DP_B1*k1v + _DP_B3*k3v + _DP_B4*k4v +
                              _DP_B5*k5v + _DP_B6*k6v)

        # k7 for error estimate
        k7v = a(pos_new)
        k7x = vel_new

        # Error vector  E = y5 - y4
        err_pos = h * (_DP_E1*k1x + _DP_E3*k3x + _DP_E4*k4x +
                        _DP_E5*k5x + _DP_E6*k6x + _DP_E7*k7x)
        err_vel = h * (_DP_E1*k1v + _DP_E3*k3v + _DP_E4*k4v +
                        _DP_E5*k5v + _DP_E6*k6v + _DP_E7*k7v)

        # RMS error norm scaled by tolerance
        scale_pos = self._atol + self._rtol * np.maximum(np.abs(pos), np.abs(pos_new))
        scale_vel = self._atol + self._rtol * np.maximum(np.abs(vel), np.abs(vel_new))

        err_norm = float(np.sqrt(
            0.5 * (
                np.mean((err_pos / scale_pos) ** 2) +
                np.mean((err_vel / scale_vel) ** 2)
            )
        ))
        return pos_new, vel_new, err_norm


# ---------------------------------------------------------------------------
# Leapfrog — fixed-step symplectic 2nd-order (KDK Velocity Verlet)
# ---------------------------------------------------------------------------

class LeapfrogIntegrator:
    """
    Kick-Drift-Kick (KDK) Velocity Verlet / Störmer-Verlet integrator.

    Second-order symplectic method.  Because it is symplectic it preserves a
    shadow Hamiltonian exactly, so energy error oscillates but does not drift
    secularly — making it far superior to RK4 for long orbital integrations.

    The first call computes the initial acceleration.  Subsequent calls reuse
    the acceleration from the end of the previous step (one evaluation per
    step after initialization, same as RK4's four evaluations per step).

    Parameters
    ----------
    calc : AccelerationCalculator
    """

    def __init__(self, calc) -> None:
        self._calc = calc
        self._accel: np.ndarray | None = None   # cached end-of-step acceleration

    def reset(self) -> None:
        """Clear the cached acceleration (call after any external state change)."""
        self._accel = None

    def step(self, system: "NBodySystem", dt: float) -> float:
        pos, vel = system.get_state()
        mus = _mus(system)

        # Compute or reuse initial acceleration
        a0 = self._accel if self._accel is not None else self._calc.compute(pos, mus)

        # KDK: half-kick → drift → half-kick
        v_half   = vel     + 0.5 * dt * a0
        pos_new  = pos     + dt  * v_half
        a_new    = self._calc.compute(pos_new, mus)
        vel_new  = v_half  + 0.5 * dt * a_new

        self._accel = a_new
        system.set_state(pos_new, vel_new)

        # Also update body acceleration fields for diagnostics
        for i, b in enumerate(system.bodies):
            b.acceleration[:] = a_new[i]

        return dt


# ---------------------------------------------------------------------------
# Yoshida — fixed-step symplectic 4th-order
# ---------------------------------------------------------------------------

# Yoshida (1990) coefficients
_cbrt2 = 2.0 ** (1.0 / 3.0)
_Y_W1  = 1.0 / (2.0 - _cbrt2)
_Y_W0  = -_cbrt2 * _Y_W1

# Position (drift) coefficients c1..c4
_Y_C1 = _Y_W1 / 2.0
_Y_C2 = (_Y_W0 + _Y_W1) / 2.0
_Y_C3 = _Y_C2
_Y_C4 = _Y_C1

# Velocity (kick) coefficients d1..d3
_Y_D1 = _Y_W1
_Y_D2 = _Y_W0
_Y_D3 = _Y_W1


class YoshidaIntegrator:
    """
    Yoshida (1990) 4th-order symplectic integrator.

    Uses 3 force evaluations per step (vs 4 for RK4) while being 4th-order
    *and* symplectic — long-run energy error is bounded, not drifting.  The
    recommended integrator for high-accuracy long orbital integrations.

    Parameters
    ----------
    calc : AccelerationCalculator
    """

    def __init__(self, calc) -> None:
        self._calc = calc

    def step(self, system: "NBodySystem", dt: float) -> float:
        pos, vel = system.get_state()
        mus = _mus(system)

        def a(p: np.ndarray) -> np.ndarray:
            return self._calc.compute(p, mus)

        # Drift-Kick-Drift-Kick-Drift-Kick-Drift  (7 sub-steps, 3 force evals)
        p1 = pos + _Y_C1 * dt * vel
        v1 = vel + _Y_D1 * dt * a(p1)

        p2 = p1  + _Y_C2 * dt * v1
        v2 = v1  + _Y_D2 * dt * a(p2)

        p3 = p2  + _Y_C3 * dt * v2
        v3 = v2  + _Y_D3 * dt * a(p3)

        p4 = p3  + _Y_C4 * dt * v3   # final position

        system.set_state(p4, v3)

        # Update acceleration fields for diagnostics
        a_final = a(p4)
        for i, b in enumerate(system.bodies):
            b.acceleration[:] = a_final[i]

        return dt
