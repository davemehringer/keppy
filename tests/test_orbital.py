"""Tests for keppy.orbital — Keplerian orbital element conversions."""

import math

import numpy as np
import pytest

from keppy.orbital import (
    Elements,
    elements_to_vectors,
    solve_kepler,
    vectors_to_elements,
)
from keppy.constants import AU, MU_SUN, DAY, YEAR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_vectors_close(a, b, rtol=1e-9, atol=0.0):
    """Assert two numpy arrays are close element-wise."""
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def _circular_earth_elements() -> Elements:
    """Circular Earth-like orbit: a=1 AU, e=0, i=0."""
    return Elements(a=AU, e=0.0, i=0.0, node=float("nan"), peri=float("nan"), M=float("nan"))


def _earth_elements() -> Elements:
    """Earth-like elliptic orbit: a=1 AU, e=0.0167, i=0.0°."""
    return Elements(a=AU, e=0.0167, i=0.0, node=0.0, peri=0.0, M=0.0)


# ---------------------------------------------------------------------------
# Elements dataclass
# ---------------------------------------------------------------------------

class TestElements:

    def test_periapsis(self):
        el = Elements(a=AU, e=0.2, i=0.0, node=0.0, peri=0.0, M=0.0)
        assert el.periapsis == pytest.approx(AU * 0.8)

    def test_apoapsis_elliptic(self):
        el = Elements(a=AU, e=0.2, i=0.0, node=0.0, peri=0.0, M=0.0)
        assert el.apoapsis == pytest.approx(AU * 1.2)

    def test_apoapsis_hyperbolic(self):
        el = Elements(a=-AU, e=1.5, i=10.0, node=0.0, peri=0.0, M=float("nan"))
        assert el.apoapsis == float("inf")

    def test_apoapsis_parabolic(self):
        el = Elements(a=AU, e=1.0, i=0.0, node=0.0, peri=0.0, M=float("nan"))
        assert el.apoapsis == float("inf")

    def test_period_property_returns_nan(self):
        """period without mu is always nan."""
        el = _earth_elements()
        assert math.isnan(el.period)

    def test_period_from_mu_elliptic(self):
        """Earth's period should be ~365.25 days."""
        el = _earth_elements()
        T = el.period_from_mu(MU_SUN)
        assert T == pytest.approx(YEAR, rel=1e-3)

    def test_period_from_mu_hyperbolic(self):
        el = Elements(a=-AU, e=1.5, i=0.0, node=0.0, peri=0.0, M=float("nan"))
        assert math.isnan(el.period_from_mu(MU_SUN))

    def test_str_representation(self):
        el = _earth_elements()
        s = str(el)
        assert "Elements(" in s
        assert "m" in s   # meters unit present


# ---------------------------------------------------------------------------
# solve_kepler
# ---------------------------------------------------------------------------

class TestSolveKepler:

    def test_zero_eccentricity(self):
        """e=0: Kepler's equation trivially gives E=M."""
        for M in [0.0, 1.0, math.pi, 2 * math.pi - 0.1]:
            E = solve_kepler(M, 0.0)
            assert E == pytest.approx(M, abs=1e-13)

    def test_nearly_circular(self):
        """Very small eccentricity: E ≈ M + e*sin(M) to first order in e."""
        M = 1.0
        e = 0.001
        E = solve_kepler(M, e)
        # First-order approximation accurate to O(e²) ≈ 1e-6
        assert E == pytest.approx(M + e * math.sin(M), abs=1e-6)

    def test_high_eccentricity(self):
        """e=0.9: verify by substituting back into Kepler's equation."""
        M = 1.0
        e = 0.9
        E = solve_kepler(M, e)
        residual = E - e * math.sin(E) - M
        assert abs(residual) < 1e-13

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.99])
    @pytest.mark.parametrize("M", [0.01, 0.5, 1.0, 2.5, 6.0])
    def test_residual_parametric(self, M, e):
        """For all (M, e) pairs: |E - e*sin(E) - M| < tol."""
        E = solve_kepler(M, e)
        residual = E - e * math.sin(E) - M
        assert abs(residual) < 1e-13

    def test_raises_for_e_ge_1(self):
        with pytest.raises(ValueError, match="e < 1"):
            solve_kepler(1.0, 1.0)
        with pytest.raises(ValueError, match="e < 1"):
            solve_kepler(1.0, 1.5)

    def test_mean_anomaly_zero(self):
        """M=0 → E=0 for all e."""
        for e in [0.0, 0.3, 0.7, 0.99]:
            assert solve_kepler(0.0, e) == pytest.approx(0.0, abs=1e-14)

    def test_mean_anomaly_pi(self):
        """M=π → E=π (periapsis + π is apoapsis by symmetry)."""
        for e in [0.0, 0.3, 0.9]:
            E = solve_kepler(math.pi, e)
            assert E == pytest.approx(math.pi, abs=1e-10)


# ---------------------------------------------------------------------------
# elements_to_vectors — circular equatorial (degenerate)
# ---------------------------------------------------------------------------

class TestElementsToVectorsCircularEquatorial:
    """Circular equatorial: node=nan, peri=nan, M=nan."""

    def test_position_radius(self):
        """Position magnitude should equal semi-major axis."""
        el = _circular_earth_elements()
        pos, _ = elements_to_vectors(MU_SUN, el)
        assert np.linalg.norm(pos) == pytest.approx(AU, rel=1e-12)

    def test_velocity_magnitude(self):
        """Speed should equal circular orbital velocity sqrt(mu/a)."""
        el = _circular_earth_elements()
        _, vel = elements_to_vectors(MU_SUN, el)
        v_circ = math.sqrt(MU_SUN / AU)
        assert np.linalg.norm(vel) == pytest.approx(v_circ, rel=1e-12)

    def test_position_velocity_perpendicular(self):
        """For a circular orbit pos·vel = 0."""
        el = _circular_earth_elements()
        pos, vel = elements_to_vectors(MU_SUN, el)
        assert float(np.dot(pos, vel)) == pytest.approx(0.0, abs=1.0)  # 1 m²/s tol


# ---------------------------------------------------------------------------
# elements_to_vectors — general elliptic
# ---------------------------------------------------------------------------

class TestElementsToVectorsElliptic:

    def test_position_radius_at_M0(self):
        """At M=0, the body is at periapsis: r = a(1-e)."""
        el = Elements(a=AU, e=0.3, i=30.0, node=45.0, peri=60.0, M=0.0)
        pos, _ = elements_to_vectors(MU_SUN, el)
        r = np.linalg.norm(pos)
        assert r == pytest.approx(AU * (1.0 - 0.3), rel=1e-12)

    def test_vis_viva(self):
        """Speed must satisfy vis-viva: v² = mu*(2/r - 1/a)."""
        el = Elements(a=AU, e=0.3, i=20.0, node=50.0, peri=80.0, M=90.0)
        pos, vel = elements_to_vectors(MU_SUN, el)
        r = np.linalg.norm(pos)
        v2 = np.dot(vel, vel)
        expected_v2 = MU_SUN * (2.0 / r - 1.0 / AU)
        assert v2 == pytest.approx(expected_v2, rel=1e-12)

    def test_angular_momentum_direction_inclination(self):
        """The angular momentum h = r×v should match the expected inclination."""
        i_deg = 45.0
        el = Elements(a=AU, e=0.1, i=i_deg, node=0.0, peri=0.0, M=45.0)
        pos, vel = elements_to_vectors(MU_SUN, el)
        h = np.cross(pos, vel)
        h_mag = np.linalg.norm(h)
        cos_i = h[2] / h_mag
        i_computed = math.degrees(math.acos(max(-1.0, min(1.0, cos_i))))
        assert i_computed == pytest.approx(i_deg, abs=1e-8)

    def test_position_at_M180_near_apoapsis(self):
        """At M=180°, the body should be near apoapsis: r ≈ a(1+e)."""
        el = Elements(a=AU, e=0.3, i=0.0, node=0.0, peri=0.0, M=180.0)
        pos, _ = elements_to_vectors(MU_SUN, el)
        r = np.linalg.norm(pos)
        # M=180 → E=π → r = a(1 - e*cos(π)) = a(1+e)
        assert r == pytest.approx(AU * 1.3, rel=1e-12)


# ---------------------------------------------------------------------------
# vectors_to_elements — round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """
    Convert elements → vectors → elements and check we recover the inputs.
    """

    def _check_round_trip(self, el_in: Elements, rtol=1e-9):
        pos, vel = elements_to_vectors(MU_SUN, el_in)
        el_out = vectors_to_elements(MU_SUN, pos, vel)
        assert el_out.a == pytest.approx(el_in.a, rel=rtol)
        assert el_out.e == pytest.approx(el_in.e, abs=1e-12)
        assert el_out.i == pytest.approx(el_in.i, abs=1e-9)
        # node and peri only compared when defined
        if not math.isnan(el_in.node):
            assert el_out.node == pytest.approx(el_in.node, abs=1e-8)
        if not math.isnan(el_in.peri):
            assert el_out.peri == pytest.approx(el_in.peri, abs=1e-8)
        if not math.isnan(el_in.M):
            assert el_out.M == pytest.approx(el_in.M, abs=1e-8)

    def test_earth_like(self):
        # i=0 is equatorial; node is undefined (nan) — skip the node check
        self._check_round_trip(Elements(
            a=AU, e=0.0167, i=10.0, node=0.0, peri=102.9, M=100.0,
        ))

    def test_inclined_moderate_eccentricity(self):
        self._check_round_trip(Elements(
            a=1.5 * AU, e=0.3, i=45.0, node=60.0, peri=120.0, M=200.0,
        ))

    def test_high_inclination(self):
        self._check_round_trip(Elements(
            a=2.0 * AU, e=0.5, i=80.0, node=200.0, peri=300.0, M=90.0,
        ))

    def test_retrograde(self):
        """Retrograde orbit: i > 90°."""
        self._check_round_trip(Elements(
            a=AU, e=0.1, i=150.0, node=30.0, peri=45.0, M=270.0,
        ))

    def test_small_eccentricity(self):
        """Near-circular but non-circular orbit."""
        self._check_round_trip(Elements(
            a=AU, e=0.01, i=10.0, node=90.0, peri=45.0, M=135.0,
        ))

    @pytest.mark.parametrize("e", [0.001, 0.1, 0.4, 0.7, 0.9])
    @pytest.mark.parametrize("i_deg", [1.0, 30.0, 60.0, 120.0, 170.0])
    def test_parametric_round_trip(self, e, i_deg):
        """Broad parametric check over eccentricity and inclination."""
        el = Elements(
            a=AU, e=e, i=i_deg, node=75.0, peri=130.0, M=60.0,
        )
        pos, vel = elements_to_vectors(MU_SUN, el)
        el_out = vectors_to_elements(MU_SUN, pos, vel)
        assert el_out.a == pytest.approx(AU, rel=1e-9)
        assert el_out.e == pytest.approx(e, abs=1e-10)
        assert el_out.i == pytest.approx(i_deg, abs=1e-8)


# ---------------------------------------------------------------------------
# vectors_to_elements — edge cases
# ---------------------------------------------------------------------------

class TestVectorsToElementsEdgeCases:

    def test_equatorial_orbit_node_is_nan(self):
        """i=0 → n_mag=0 → node should be nan."""
        pos = np.array([AU, 0.0, 0.0])
        vel = np.array([0.0, math.sqrt(MU_SUN / AU) * 1.1, 0.0])  # e > 0
        el = vectors_to_elements(MU_SUN, pos, vel)
        assert el.i == pytest.approx(0.0, abs=1e-8)
        assert math.isnan(el.node)

    def test_circular_orbit_peri_is_nan(self):
        """e=0 → peri and M should be nan."""
        v_circ = math.sqrt(MU_SUN / AU)
        pos = np.array([AU, 0.0, 0.0])
        vel = np.array([0.0, v_circ, 0.0])
        el = vectors_to_elements(MU_SUN, pos, vel)
        assert el.e == pytest.approx(0.0, abs=1e-10)
        assert math.isnan(el.peri)
        assert math.isnan(el.M)

    def test_circular_inclination_recovered(self):
        """Circular inclined orbit: inclination should still be recoverable."""
        i_deg = 30.0
        i_rad = math.radians(i_deg)
        v_circ = math.sqrt(MU_SUN / AU)
        pos = np.array([AU, 0.0, 0.0])
        # Tilt velocity into the i=30 plane
        vel = np.array([0.0, v_circ * math.cos(i_rad), v_circ * math.sin(i_rad)])
        el = vectors_to_elements(MU_SUN, pos, vel)
        assert el.i == pytest.approx(i_deg, abs=1e-8)
        assert el.e == pytest.approx(0.0, abs=1e-10)
        assert math.isnan(el.peri)

    def test_positive_semi_major_axis_bound_orbit(self):
        """Bound orbit must have a > 0."""
        pos = np.array([AU, 0.0, 0.0])
        vel = np.array([0.0, math.sqrt(MU_SUN / AU), 0.0])
        el = vectors_to_elements(MU_SUN, pos, vel)
        assert el.a > 0.0

    def test_hyperbolic_orbit_negative_a(self):
        """Escape velocity gives e > 1, a < 0."""
        v_esc = math.sqrt(2.0 * MU_SUN / AU)
        pos = np.array([AU, 0.0, 0.0])
        vel = np.array([0.0, v_esc * 1.1, 0.0])  # above escape velocity
        el = vectors_to_elements(MU_SUN, pos, vel)
        assert el.e > 1.0
        assert el.a < 0.0


# ---------------------------------------------------------------------------
# vectors_to_elements — physical properties
# ---------------------------------------------------------------------------

class TestVectorsToElementsPhysics:

    def test_eccentricity_from_radii(self):
        """
        Put body exactly at apoapsis with the right velocity.
        e should be (r_a - r_p) / (r_a + r_p).
        """
        r_a = 1.5 * AU   # apoapsis
        r_p = 0.5 * AU   # periapsis
        e_expected = (r_a - r_p) / (r_a + r_p)
        a = 0.5 * (r_a + r_p)
        # At apoapsis the velocity is purely tangential: v_a = sqrt(mu*(2/r_a - 1/a))
        v_a = math.sqrt(MU_SUN * (2.0 / r_a - 1.0 / a))
        pos = np.array([r_a, 0.0, 0.0])
        vel = np.array([0.0, v_a, 0.0])
        el = vectors_to_elements(MU_SUN, pos, vel)
        assert el.e == pytest.approx(e_expected, rel=1e-9)
        assert el.a == pytest.approx(a, rel=1e-9)

    def test_inclination_various_angles(self):
        """Check inclination is correctly recovered for several values."""
        for i_deg in [0.0, 30.0, 45.0, 90.0, 120.0, 180.0]:
            i_rad = math.radians(i_deg)
            v_circ = math.sqrt(MU_SUN / AU)
            pos = np.array([AU, 0.0, 0.0])
            # Velocity in the direction perpendicular to pos, tilted by i
            vel = np.array([0.0, v_circ * math.cos(i_rad), v_circ * math.sin(i_rad)])
            el = vectors_to_elements(MU_SUN, pos, vel)
            assert el.i == pytest.approx(i_deg, abs=1e-6), f"Failed for i={i_deg}°"

    def test_mean_anomaly_at_periapsis(self):
        """At periapsis, M=0."""
        e = 0.4
        a = AU
        r_p = a * (1.0 - e)
        # Velocity at periapsis: purely tangential, magnitude sqrt(mu*(2/r_p - 1/a))
        v_p = math.sqrt(MU_SUN * (2.0 / r_p - 1.0 / a))
        pos = np.array([r_p, 0.0, 0.0])
        vel = np.array([0.0, v_p, 0.0])
        el = vectors_to_elements(MU_SUN, pos, vel)
        assert el.M == pytest.approx(0.0, abs=1e-8)

    def test_mean_anomaly_at_apoapsis(self):
        """At apoapsis, M=180°."""
        e = 0.4
        a = AU
        r_a = a * (1.0 + e)
        v_a = math.sqrt(MU_SUN * (2.0 / r_a - 1.0 / a))
        pos = np.array([r_a, 0.0, 0.0])
        vel = np.array([0.0, v_a, 0.0])
        # At apoapsis velocity is in -y direction for the standard orientation,
        # but here with positive x position and positive y velocity we are going away
        # from periapsis. Let's check M is 180.
        # Actually at apoapsis with pos=[r_a, 0, 0] and vel=[0, v_a, 0] (prograde),
        # the body is receding from x-axis periapsis. M should be 180°.
        el = vectors_to_elements(MU_SUN, pos, vel)
        # e=0.4 orbit with peri at +x: apoapsis is at +x too (at r_a).
        # Wait — if peri is at x=r_p, apoapsis is at x=-r_a for peri=0.
        # Actually: at M=0 (periapsis), position is along the periapsis direction.
        # If peri=0 and node=0, periapsis direction is +x.
        # But apoapsis at M=180 is at the OPPOSITE end of the major axis, i.e., -x.
        # So to be at apoapsis we need pos=[-r_a, 0, 0].
        r_a_pos = np.array([-r_a, 0.0, 0.0])
        vel_a = np.array([0.0, -v_a, 0.0])
        el2 = vectors_to_elements(MU_SUN, r_a_pos, vel_a)
        assert el2.M == pytest.approx(180.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Full vector round-trip: vectors → elements → vectors
# ---------------------------------------------------------------------------

class TestVectorRoundTrip:
    """Convert state vectors → elements → state vectors and check recovery."""

    def _check(self, pos_in, vel_in, rtol=1e-9, atol=1.0):
        el = vectors_to_elements(MU_SUN, pos_in, vel_in)
        pos_out, vel_out = elements_to_vectors(MU_SUN, el)
        np.testing.assert_allclose(pos_out, pos_in, rtol=rtol, atol=atol)
        np.testing.assert_allclose(vel_out, vel_in, rtol=rtol, atol=1e-6)

    def test_circular_equatorial(self):
        v_circ = math.sqrt(MU_SUN / AU)
        self._check(
            np.array([AU, 0.0, 0.0]),
            np.array([0.0, v_circ, 0.0]),
        )

    def test_general_orbit(self):
        pos = np.array([AU * 0.8, AU * 0.3, AU * 0.1])
        # Use a velocity that gives a bound orbit
        v_mag = math.sqrt(MU_SUN * (2.0 / np.linalg.norm(pos) - 1.0 / AU))
        vel = np.array([v_mag * 0.2, v_mag * 0.9, v_mag * 0.1])
        vel *= v_mag / np.linalg.norm(vel)  # normalize to v_mag
        self._check(pos, vel, atol=100.0)  # 100 m tolerance for complex orbit

    def test_inclined_eccentric(self):
        el = Elements(a=2.0 * AU, e=0.6, i=60.0, node=120.0, peri=200.0, M=300.0)
        pos, vel = elements_to_vectors(MU_SUN, el)
        el2 = vectors_to_elements(MU_SUN, pos, vel)
        pos2, vel2 = elements_to_vectors(MU_SUN, el2)
        np.testing.assert_allclose(pos2, pos, rtol=1e-9)
        np.testing.assert_allclose(vel2, vel, rtol=1e-9)


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------

class TestPackageExports:

    def test_orbital_exports(self):
        import keppy
        assert hasattr(keppy, "Elements")
        assert hasattr(keppy, "elements_to_vectors")
        assert hasattr(keppy, "vectors_to_elements")
        assert hasattr(keppy, "solve_kepler")
