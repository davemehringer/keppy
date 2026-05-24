"""
Tests for keppy.integrator.

All tests use a Sun + Earth two-body system in a circular orbit.
Analytical values:
  orbital period  T = 2π √(AU³ / MU_SUN)  ≈ 3.156e7 s  ≈ 365.25 days
  orbital speed   v = √(MU_SUN / AU)       ≈ 29 784 m/s
  total energy    E = -½ MU_SUN * m_earth / AU
"""

import numpy as np
import pytest

from keppy.body import Body
from keppy.nbody_system import NBodySystem
from keppy.acceleration import PairwiseAccelerationCalculator
from keppy.integrator import (
    RK4Integrator,
    RK45Integrator,
    LeapfrogIntegrator,
    YoshidaIntegrator,
    Integrator,
)
from keppy.constants import AU, MU_SUN, G, DAY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _orbital_period() -> float:
    """Keplerian period of Earth at 1 AU (seconds)."""
    return 2.0 * np.pi * np.sqrt(AU ** 3 / MU_SUN)


def sun_earth(translate: bool = False) -> NBodySystem:
    """
    Sun at origin, Earth in a circular orbit at 1 AU in the x-y plane.
    translate=False keeps the Sun exactly at the origin for simplicity.
    """
    mass_earth = 5.972e24
    v_circ = np.sqrt(MU_SUN / AU)

    sun   = Body.from_mu("Sun",   MU_SUN,
                         position=np.zeros(3), velocity=np.zeros(3))
    earth = Body.from_mass("Earth", mass_earth,
                            position=np.array([AU,    0.0, 0.0]),
                            velocity=np.array([0.0, v_circ, 0.0]))
    return NBodySystem([sun, earth], translate_to_barycenter=translate)


def make_integrator(cls, **kwargs):
    calc = PairwiseAccelerationCalculator()
    return cls(calc, **kwargs)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:

    def test_rk4_satisfies_protocol(self):
        assert isinstance(make_integrator(RK4Integrator), Integrator)

    def test_rk45_satisfies_protocol(self):
        assert isinstance(make_integrator(RK45Integrator), Integrator)

    def test_leapfrog_satisfies_protocol(self):
        assert isinstance(make_integrator(LeapfrogIntegrator), Integrator)

    def test_yoshida_satisfies_protocol(self):
        assert isinstance(make_integrator(YoshidaIntegrator), Integrator)


# ---------------------------------------------------------------------------
# Basic step mechanics
# ---------------------------------------------------------------------------

class TestStepMechanics:

    @pytest.mark.parametrize("cls", [
        RK4Integrator, LeapfrogIntegrator, YoshidaIntegrator,
    ])
    def test_step_returns_dt(self, cls):
        sys = sun_earth()
        integ = make_integrator(cls)
        dt = 3600.0
        returned = sys.step(dt, integ)
        assert returned == pytest.approx(dt)

    @pytest.mark.parametrize("cls", [
        RK4Integrator, LeapfrogIntegrator, YoshidaIntegrator,
    ])
    def test_system_time_advances(self, cls):
        sys = sun_earth()
        integ = make_integrator(cls)
        dt = DAY
        sys.step(dt, integ)
        assert sys.time == pytest.approx(dt)

    @pytest.mark.parametrize("cls", [
        RK4Integrator, LeapfrogIntegrator, YoshidaIntegrator,
    ])
    def test_positions_change_after_step(self, cls):
        sys = sun_earth()
        pos0, _ = sys.get_state()
        integ = make_integrator(cls)
        sys.step(DAY, integ)
        pos1, _ = sys.get_state()
        assert not np.allclose(pos0, pos1)

    def test_rk45_returns_le_dt(self):
        """Adaptive integrator must not advance beyond the requested dt."""
        sys = sun_earth()
        integ = make_integrator(RK45Integrator)
        dt = DAY
        actual = sys.step(dt, integ)
        assert actual <= dt + 1e-10   # allow tiny float tolerance


# ---------------------------------------------------------------------------
# One-orbit period test (position returns to start)
# ---------------------------------------------------------------------------

def _run_one_orbit(cls, n_steps=3650, **kwargs):
    """
    Run one Keplerian orbit using a massless Earth (test particle).

    We use a massless test particle so there is no two-body mass-ratio
    correction to the period; the nominal T = 2π√(AU³/MU_SUN) is exact.

    We compare the *relative* position (Earth minus Sun) rather than the
    absolute position of Earth to eliminate any barycenter drift.

    n_steps=3650 (~2.4 h per step) gives sub-meter closure for RK4 and
    Yoshida (4th-order methods).  Leapfrog needs ~10× more steps for
    comparable closure since it is only 2nd-order; for Leapfrog we test
    orbital-radius preservation instead (see TestOrbitalRadius).
    """
    v = np.sqrt(MU_SUN / AU)
    sys = NBodySystem(
        [Body.from_mu("Sun", MU_SUN, np.zeros(3), np.zeros(3)),
         Body.from_mass("Earth", 0.0,
                        np.array([AU, 0.0, 0.0]),
                        np.array([0.0, v, 0.0]))],
        translate_to_barycenter=False,
    )
    integ = make_integrator(cls, **kwargs)
    T = _orbital_period()
    dt = T / n_steps
    t = 0.0
    while t < T:
        actual = sys.step(dt, integ)
        t += actual
    earth = sys.get_body_by_name("Earth")
    sun   = sys.get_body_by_name("Sun")
    return (earth.position - sun.position).copy()


class TestOrbitalPeriod:
    """
    Orbit-closure tests: after one full period Earth returns to [AU, 0, 0].

    Notes on step size and expected accuracy (massless test particle):
      RK4 (4th-order):     n=3650  → ~0.2 m error   → tolerance 1 km
      Yoshida (4th-order): n=3650  → ~7 m error     → tolerance 100 m
      Leapfrog (2nd-order) needs ~36 000 steps for km-level closure;
        tested separately via orbit-radius preservation.
      RK45: local error is adaptive; orbit closure depends on tolerance
        setting, not step count — tested via energy/momentum instead.
    """

    def test_rk4_orbit_closes(self):
        """RK4 with n=3650 steps should close the orbit within 1 km."""
        rel = _run_one_orbit(RK4Integrator, n_steps=3650)
        np.testing.assert_allclose(rel, [AU, 0.0, 0.0], atol=1e3)

    def test_yoshida_orbit_closes(self):
        """Yoshida with n=3650 steps should close the orbit within 100 m."""
        rel = _run_one_orbit(YoshidaIntegrator, n_steps=3650)
        np.testing.assert_allclose(rel, [AU, 0.0, 0.0], atol=100.0)


class TestOrbitalRadius:
    """
    Orbital-radius preservation: |r_earth - r_sun| ≈ AU throughout the orbit.

    Leapfrog is symplectic so it exactly preserves a shadow Hamiltonian,
    keeping the orbit shape (radius) tightly bounded even with a coarse step
    size.  RK4 has secular energy error that can slowly change the radius.
    """

    @pytest.mark.parametrize("cls", [
        LeapfrogIntegrator, YoshidaIntegrator, RK4Integrator,
    ])
    def test_radius_preserved_one_orbit(self, cls):
        """
        After one orbit the Earth-Sun distance should remain within 0.01% of AU.
        """
        sys = sun_earth()
        integ = make_integrator(cls)
        T = _orbital_period()
        dt = T / 365
        t = 0.0
        while t < T:
            actual = sys.step(dt, integ)
            t += actual
        earth = sys.get_body_by_name("Earth")
        sun   = sys.get_body_by_name("Sun")
        r = np.linalg.norm(earth.position - sun.position)
        assert abs(r - AU) / AU < 1e-4


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------

def _energy_drift_fraction(cls, n_orbits=1, dt_fraction=1/365, **kwargs):
    """
    Run *n_orbits* complete orbits; return |ΔE| / |E0|.
    """
    sys = sun_earth()
    integ = make_integrator(cls, **kwargs)
    E0 = sys.total_energy
    T = _orbital_period()
    dt = T * dt_fraction
    target = T * n_orbits
    t = 0.0
    while t < target:
        actual = sys.step(dt, integ)
        t += actual
    E1 = sys.total_energy
    return abs(E1 - E0) / abs(E0)


class TestEnergyConservation:

    def test_rk4_energy_drift_one_orbit(self):
        """RK4 energy drift over one orbit should be < 0.1%."""
        drift = _energy_drift_fraction(RK4Integrator)
        assert drift < 1e-3

    def test_leapfrog_energy_drift_one_orbit(self):
        """Leapfrog energy drift should be < 0.1%."""
        drift = _energy_drift_fraction(LeapfrogIntegrator)
        assert drift < 1e-3

    def test_yoshida_energy_drift_one_orbit(self):
        """Yoshida drift should be < 0.01% — tighter than RK4 and leapfrog."""
        drift = _energy_drift_fraction(YoshidaIntegrator)
        assert drift < 1e-4

    def test_rk45_energy_drift_one_orbit(self):
        """RK45 adaptive should achieve < 0.001% energy drift."""
        drift = _energy_drift_fraction(RK45Integrator)
        assert drift < 1e-5

    def test_symplectic_beats_rk4_over_10_orbits(self):
        """
        Over 10 orbits the symplectic integrators should have lower energy
        drift than RK4 (secular drift vs bounded oscillation).
        """
        drift_rk4      = _energy_drift_fraction(RK4Integrator,      n_orbits=10)
        drift_leapfrog = _energy_drift_fraction(LeapfrogIntegrator,  n_orbits=10)
        drift_yoshida  = _energy_drift_fraction(YoshidaIntegrator,   n_orbits=10)
        assert drift_leapfrog < drift_rk4
        assert drift_yoshida  < drift_rk4


# ---------------------------------------------------------------------------
# Angular momentum conservation
# ---------------------------------------------------------------------------

class TestAngularMomentum:

    @pytest.mark.parametrize("cls", [
        RK4Integrator, LeapfrogIntegrator, YoshidaIntegrator,
    ])
    def test_angular_momentum_conserved(self, cls):
        """
        |L| should not change by more than 0.01% over one orbit.
        """
        sys = sun_earth()
        integ = make_integrator(cls)
        L0 = np.linalg.norm(sys.angular_momentum)
        T = _orbital_period()
        dt = T / 365
        t = 0.0
        while t < T:
            t += sys.step(dt, integ)
        L1 = np.linalg.norm(sys.angular_momentum)
        assert abs(L1 - L0) / L0 < 1e-4


# ---------------------------------------------------------------------------
# Leapfrog-specific: cached acceleration
# ---------------------------------------------------------------------------

class TestLeapfrogCache:

    def test_reset_clears_cache(self):
        sys = sun_earth()
        integ = make_integrator(LeapfrogIntegrator)
        sys.step(DAY, integ)
        assert integ._accel is not None
        integ.reset()
        assert integ._accel is None

    def test_two_steps_same_result_as_reset_between(self):
        """
        Two consecutive steps should give the same result whether or not
        we reset the cache between them (reset forces a fresh evaluation,
        but the values should agree to machine precision).
        """
        # Without reset
        sys1 = sun_earth()
        integ1 = make_integrator(LeapfrogIntegrator)
        sys1.step(DAY, integ1)
        sys1.step(DAY, integ1)
        pos1, vel1 = sys1.get_state()

        # With reset between steps
        sys2 = sun_earth()
        integ2 = make_integrator(LeapfrogIntegrator)
        sys2.step(DAY, integ2)
        integ2.reset()
        sys2.step(DAY, integ2)
        pos2, vel2 = sys2.get_state()

        np.testing.assert_allclose(pos1, pos2, rtol=1e-12)
        np.testing.assert_allclose(vel1, vel2, rtol=1e-12)


# ---------------------------------------------------------------------------
# NBodySystem.step() return value
# ---------------------------------------------------------------------------

class TestNBodyStepReturn:

    def test_step_returns_float(self):
        sys = sun_earth()
        integ = make_integrator(RK4Integrator)
        result = sys.step(DAY, integ)
        assert isinstance(result, float)

    def test_accumulated_time_matches_steps(self):
        sys = sun_earth()
        integ = make_integrator(LeapfrogIntegrator)
        total = sum(sys.step(DAY, integ) for _ in range(10))
        assert sys.time == pytest.approx(total)
