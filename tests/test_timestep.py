"""Tests for keppy.timestep."""

import numpy as np
import pytest

from keppy.body import Body
from keppy.nbody_system import NBodySystem
from keppy.acceleration import PairwiseAccelerationCalculator
from keppy.integrator import LeapfrogIntegrator, YoshidaIntegrator, RK4Integrator
from keppy.timestep import (
    ChangeType,
    ConstantTimeStepManager,
    AccelerationTimeStepManager,
    ScaledTimeStepManager,
    TimeStepManager,
    StepRecord,
    run,
)
from keppy.constants import AU, MU_SUN, G, DAY, YEAR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sun_earth() -> NBodySystem:
    v = np.sqrt(MU_SUN / AU)
    sun   = Body.from_mu("Sun",   MU_SUN, np.zeros(3), np.zeros(3))
    earth = Body.from_mass("Earth", 5.972e24,
                            np.array([AU, 0., 0.]),
                            np.array([0., v, 0.]))
    return NBodySystem([sun, earth])


def make_calc():
    return PairwiseAccelerationCalculator()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_constant_satisfies_protocol(self):
        assert isinstance(ConstantTimeStepManager(DAY), TimeStepManager)

    def test_acceleration_satisfies_protocol(self):
        assert isinstance(
            AccelerationTimeStepManager(DAY, 0.01, 0.1), TimeStepManager
        )

    def test_scaled_satisfies_protocol(self):
        assert isinstance(ScaledTimeStepManager(DAY), TimeStepManager)


# ---------------------------------------------------------------------------
# ConstantTimeStepManager
# ---------------------------------------------------------------------------

class TestConstantTimeStepManager:
    def test_dt_unchanged(self):
        tsm = ConstantTimeStepManager(DAY)
        a = np.ones((2, 3))
        tsm.update(a, a * 2)
        assert tsm.dt == pytest.approx(DAY)

    def test_always_returns_no_change(self):
        tsm = ConstantTimeStepManager(DAY)
        a_before = np.ones((2, 3))
        # Even with wildly different accelerations
        assert tsm.update(a_before, a_before * 100) == ChangeType.NO_CHANGE
        assert tsm.update(a_before, a_before * 0.001) == ChangeType.NO_CHANGE


# ---------------------------------------------------------------------------
# AccelerationTimeStepManager
# ---------------------------------------------------------------------------

class TestAccelerationTimeStepManager:

    def test_construction_invalid_thresholds(self):
        with pytest.raises(ValueError, match="a_min"):
            AccelerationTimeStepManager(DAY, a_min=0.1, a_max=0.05)

    def test_no_change_in_dead_zone(self):
        """When all q_i are between a_min and a_max → NO_CHANGE."""
        tsm = AccelerationTimeStepManager(DAY, a_min=0.01, a_max=0.1)
        a_before = np.array([[10., 0., 0.], [0., 10., 0.]])
        # 5% change → in [0.01, 0.1] → no change
        a_after  = np.array([[10.5, 0., 0.], [0., 10.5, 0.]])
        result = tsm.update(a_before, a_after)
        assert result == ChangeType.NO_CHANGE
        assert tsm.dt == pytest.approx(DAY)

    def test_decrease_when_change_too_large(self):
        """q > a_max → DECREASE; dt halved."""
        tsm = AccelerationTimeStepManager(DAY, a_min=0.01, a_max=0.1,
                                          decrease_factor=2.0)
        a_before = np.array([[10., 0., 0.]])
        a_after  = np.array([[12., 0., 0.]])   # 20% change > 10%
        result = tsm.update(a_before, a_after)
        assert result == ChangeType.DECREASE
        assert tsm.dt == pytest.approx(DAY / 2)

    def test_increase_when_change_too_small(self):
        """All q_i < a_min → INCREASE; dt doubled."""
        tsm = AccelerationTimeStepManager(DAY, a_min=0.01, a_max=0.1,
                                          increase_factor=2.0)
        a_before = np.array([[10., 0., 0.], [0., 10., 0.]])
        a_after  = a_before * 1.001   # 0.1% change < 1%
        result = tsm.update(a_before, a_after)
        assert result == ChangeType.INCREASE
        assert tsm.dt == pytest.approx(DAY * 2)

    def test_min_dt_clamps_decrease(self):
        """dt should not drop below min_dt."""
        tsm = AccelerationTimeStepManager(
            dt_init=10.0, a_min=0.01, a_max=0.1,
            decrease_factor=100.0, min_dt=5.0,
        )
        a_before = np.array([[10., 0., 0.]])
        a_after  = np.array([[15., 0., 0.]])   # huge change
        tsm.update(a_before, a_after)
        assert tsm.dt >= 5.0

    def test_max_dt_clamps_increase(self):
        """dt should not exceed max_dt."""
        tsm = AccelerationTimeStepManager(
            dt_init=DAY, a_min=0.01, a_max=0.1,
            increase_factor=1000.0, max_dt=2 * DAY,
        )
        a_before = np.array([[10., 0., 0.]])
        a_after  = a_before * 1.0001
        tsm.update(a_before, a_after)
        assert tsm.dt <= 2 * DAY

    def test_zero_acceleration_bodies_ignored(self):
        """
        Bodies with zero initial acceleration are excluded from the quality
        metric.  Only the nonzero body is considered; if its q < a_min, the
        step is INCREASE (all nonzero bodies are "easy").
        """
        tsm = AccelerationTimeStepManager(DAY, a_min=0.01, a_max=0.1)
        a_before = np.array([[0., 0., 0.], [10., 0., 0.]])
        a_after  = np.array([[5., 5., 5.], [10.05, 0., 0.]])  # 0.5% on body 1
        result = tsm.update(a_before, a_after)
        # Only body[1] is nonzero; q=0.5% < a_min=1% → INCREASE (step too easy)
        assert result == ChangeType.INCREASE


# ---------------------------------------------------------------------------
# ScaledTimeStepManager
# ---------------------------------------------------------------------------

class TestScaledTimeStepManager:

    def test_rejects_large_change(self):
        """q >> q_target should trigger a DECREASE."""
        tsm = ScaledTimeStepManager(DAY, q_target=0.01, reject_factor=3.0)
        a_before = np.array([[10., 0., 0.]])
        a_after  = np.array([[15., 0., 0.]])   # 50% change >> 3% target
        result = tsm.update(a_before, a_after)
        assert result == ChangeType.DECREASE
        assert tsm.dt < DAY

    def test_increases_on_tiny_change(self):
        """q << q_target should trigger an INCREASE."""
        tsm = ScaledTimeStepManager(DAY, q_target=0.05, exponent=0.3,
                                    max_factor=5.0)
        a_before = np.array([[10., 0., 0.]])
        a_after  = a_before * 1.00001   # 0.001% << 5% target
        result = tsm.update(a_before, a_after)
        assert result == ChangeType.INCREASE
        assert tsm.dt > DAY

    def test_min_dt_respected(self):
        tsm = ScaledTimeStepManager(10.0, q_target=0.01, min_dt=5.0)
        a_before = np.array([[1., 0., 0.]])
        a_after  = np.array([[2., 0., 0.]])   # 100% change → large decrease
        tsm.update(a_before, a_after)
        assert tsm.dt >= 5.0

    def test_max_dt_respected(self):
        tsm = ScaledTimeStepManager(DAY, q_target=0.05,
                                    max_dt=2 * DAY, max_factor=100.0)
        a_before = np.array([[10., 0., 0.]])
        a_after  = a_before * 1.000001
        tsm.update(a_before, a_after)
        assert tsm.dt <= 2 * DAY


# ---------------------------------------------------------------------------
# run() helper — integration tests
# ---------------------------------------------------------------------------

class TestRun:

    def test_run_returns_records(self):
        sys   = sun_earth()
        integ = LeapfrogIntegrator(make_calc())
        tsm   = ConstantTimeStepManager(DAY)
        records = run(sys, integ, tsm, t_end=10 * DAY)
        assert len(records) == 10
        assert all(isinstance(r, StepRecord) for r in records)

    def test_run_advances_time(self):
        sys   = sun_earth()
        integ = LeapfrogIntegrator(make_calc())
        tsm   = ConstantTimeStepManager(DAY)
        run(sys, integ, tsm, t_end=30 * DAY)
        assert sys.time == pytest.approx(30 * DAY)

    def test_run_record_fields(self):
        sys   = sun_earth()
        integ = LeapfrogIntegrator(make_calc())
        tsm   = ConstantTimeStepManager(DAY)
        records = run(sys, integ, tsm, t_end=5 * DAY)
        r = records[0]
        assert r.positions.shape  == (2, 3)
        assert r.velocities.shape == (2, 3)
        assert r.time > 0
        assert r.dt_used > 0

    def test_run_output_every(self):
        """output_every=5 should produce ⌈n_steps / 5⌉ records."""
        sys   = sun_earth()
        integ = LeapfrogIntegrator(make_calc())
        tsm   = ConstantTimeStepManager(DAY)
        records = run(sys, integ, tsm, t_end=10 * DAY, output_every=5)
        # 10 steps / 5 = 2 records (steps 5 and 10)
        assert len(records) == 2

    def test_run_with_yoshida(self):
        sys   = sun_earth()
        integ = YoshidaIntegrator(make_calc())
        tsm   = ConstantTimeStepManager(DAY)
        records = run(sys, integ, tsm, t_end=5 * DAY)
        assert len(records) == 5

    def test_adaptive_run_decreases_dt_on_close_approach(self):
        """
        On an eccentric orbit (e=0.5, apoapsis at 1 AU, periapsis at AU/3)
        the acceleration near periapsis grows by 9× compared to apoapsis.
        Running for a half-period (~100 days) should trigger at least one
        DECREASE so that dt_used differs across steps.
        """
        v_esc = np.sqrt(2 * MU_SUN / AU)
        v_ecc = 0.5 * v_esc   # e=0.5, starting at apoapsis
        # Orbital period ≈ 199 days; half-period ≈ 99 days
        T_half = np.pi * np.sqrt((2 * AU / 3) ** 3 / MU_SUN)
        sun   = Body.from_mu("Sun", MU_SUN, np.zeros(3), np.zeros(3))
        earth = Body.from_mass("Earth", 5.972e24,
                                np.array([AU,  0., 0.]),
                                np.array([0., v_ecc, 0.]))
        sys   = NBodySystem([sun, earth])
        integ = LeapfrogIntegrator(make_calc())
        # a_max=0.02 (2%) is small enough to trigger DECREASE near periapsis
        tsm   = AccelerationTimeStepManager(
            dt_init=DAY, a_min=0.001, a_max=0.02,
            min_dt=600.0,
        )
        records = run(sys, integ, tsm, t_end=T_half, output_every=1)
        dt_values = [r.dt_used for r in records]
        # At least one step should differ from the initial DAY
        assert not all(abs(dt - DAY) < 1.0 for dt in dt_values)

    def test_run_does_not_overshoot_t_end(self):
        sys   = sun_earth()
        integ = LeapfrogIntegrator(make_calc())
        tsm   = ConstantTimeStepManager(DAY * 3)  # step larger than t_end
        run(sys, integ, tsm, t_end=7 * DAY)
        assert sys.time <= 7 * DAY + 1.0   # 1 s tolerance

    def test_run_energy_conserved_leapfrog(self):
        """Energy should be conserved to 0.1% over 30 days with Leapfrog."""
        sys   = sun_earth()
        integ = LeapfrogIntegrator(make_calc())
        tsm   = ConstantTimeStepManager(DAY)
        E0    = sys.total_energy
        run(sys, integ, tsm, t_end=30 * DAY)
        E1    = sys.total_energy
        assert abs(E1 - E0) / abs(E0) < 1e-3
