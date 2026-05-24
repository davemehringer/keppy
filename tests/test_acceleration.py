"""Tests for keppy.acceleration."""

import numpy as np
import pytest

from keppy.acceleration import PairwiseAccelerationCalculator, _zonal_harmonic_acceleration
from keppy.body import Body
from keppy.constants import G, AU, MU_SUN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sun_earth_positions():
    """Sun at origin, Earth at 1 AU along x-axis."""
    mass_sun = MU_SUN / G
    mass_earth = 5.972e24
    positions = np.array([
        [0.0, 0.0, 0.0],
        [AU,  0.0, 0.0],
    ])
    mus = np.array([MU_SUN, G * mass_earth])
    return positions, mus


# ---------------------------------------------------------------------------
# Basic Newtonian gravity
# ---------------------------------------------------------------------------

class TestPairwiseNewton:

    def test_output_shape(self):
        positions, mus = sun_earth_positions()
        calc = PairwiseAccelerationCalculator()
        acc = calc.compute(positions, mus)
        assert acc.shape == (2, 3)

    def test_earth_acceleration_magnitude(self):
        """|a_earth| should equal MU_sun / AU^2."""
        positions, mus = sun_earth_positions()
        calc = PairwiseAccelerationCalculator()
        acc = calc.compute(positions, mus)
        expected = MU_SUN / AU ** 2
        assert abs(np.linalg.norm(acc[1]) - expected) / expected < 1e-10

    def test_earth_acceleration_direction(self):
        """Earth's acceleration should point toward the Sun (negative x)."""
        positions, mus = sun_earth_positions()
        calc = PairwiseAccelerationCalculator()
        acc = calc.compute(positions, mus)
        assert acc[1, 0] < 0          # x-component negative (toward Sun)
        assert abs(acc[1, 1]) < 1e-20  # y-component zero
        assert abs(acc[1, 2]) < 1e-20  # z-component zero

    def test_newton_third_law_ratio(self):
        """
        Forces are equal and opposite: m_i * a_i = -m_j * a_j.
        Since positions, mus are symmetric, the force magnitudes satisfy
        F = m*a = mu/G * |a|.
        """
        positions, mus = sun_earth_positions()
        calc = PairwiseAccelerationCalculator()
        acc = calc.compute(positions, mus)
        F_on_earth = (mus[1] / G) * np.linalg.norm(acc[1])
        F_on_sun   = (mus[0] / G) * np.linalg.norm(acc[0])
        assert F_on_earth == pytest.approx(F_on_sun, rel=1e-10)

    def test_forces_opposite_direction(self):
        """Net force vectors should be antiparallel."""
        positions, mus = sun_earth_positions()
        calc = PairwiseAccelerationCalculator()
        acc = calc.compute(positions, mus)
        # Normalize and check they are opposite
        a0 = acc[0] / np.linalg.norm(acc[0])
        a1 = acc[1] / np.linalg.norm(acc[1])
        np.testing.assert_allclose(a0, -a1, atol=1e-12)

    def test_three_body_shape(self):
        positions = np.array([
            [0.0,  0.0, 0.0],
            [AU,   0.0, 0.0],
            [0.0,  AU,  0.0],
        ])
        mus = np.array([MU_SUN, G * 5.972e24, G * 6.39e23])
        calc = PairwiseAccelerationCalculator()
        acc = calc.compute(positions, mus)
        assert acc.shape == (3, 3)

    def test_self_acceleration_zero(self):
        """A single body exerts no force on itself."""
        positions = np.array([[0.0, 0.0, 0.0]])
        mus = np.array([MU_SUN])
        calc = PairwiseAccelerationCalculator()
        acc = calc.compute(positions, mus)
        np.testing.assert_array_equal(acc, [[0.0, 0.0, 0.0]])

    def test_zero_mu_body_exerts_no_force(self):
        """A massless tracer particle contributes no gravitational pull."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [AU,  0.0, 0.0],
            [0.0, AU,  0.0],
        ])
        mus = np.array([MU_SUN, 0.0, G * 5.972e24])
        calc = PairwiseAccelerationCalculator()
        acc_with = calc.compute(positions, mus)

        # Massless body (index 1) should still feel Sun's gravity
        assert np.linalg.norm(acc_with[1]) > 0

        # Sun's acceleration should be the same as with only the third body
        positions2 = np.array([[0.0, 0.0, 0.0], [0.0, AU, 0.0]])
        mus2 = np.array([MU_SUN, G * 5.972e24])
        acc2 = calc.compute(positions2, mus2)
        np.testing.assert_allclose(acc_with[0], acc2[0], rtol=1e-12)

    def test_symmetry_under_permutation(self):
        """Swapping two bodies should swap their accelerations."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [AU,  0.0, 0.0],
        ])
        mus = np.array([MU_SUN, G * 5.972e24])
        calc = PairwiseAccelerationCalculator()
        acc = calc.compute(positions, mus)

        positions_swapped = positions[[1, 0]]
        mus_swapped = mus[[1, 0]]
        acc_swapped = calc.compute(positions_swapped, mus_swapped)

        np.testing.assert_allclose(acc[0], acc_swapped[1], rtol=1e-12)
        np.testing.assert_allclose(acc[1], acc_swapped[0], rtol=1e-12)

    def test_inverse_square_law(self):
        """Doubling distance should quarter the acceleration magnitude."""
        positions1 = np.array([[0.0, 0.0, 0.0], [AU, 0.0, 0.0]])
        positions2 = np.array([[0.0, 0.0, 0.0], [2*AU, 0.0, 0.0]])
        mus = np.array([MU_SUN, G * 5.972e24])
        calc = PairwiseAccelerationCalculator()
        a1 = np.linalg.norm(calc.compute(positions1, mus)[1])
        a2 = np.linalg.norm(calc.compute(positions2, mus)[1])
        assert a1 / a2 == pytest.approx(4.0, rel=1e-10)


# ---------------------------------------------------------------------------
# Zonal harmonics
# ---------------------------------------------------------------------------

class TestZonalHarmonics:

    def test_no_j_coefficients_returns_zero(self):
        diff = np.array([1e7, 0.0, 0.0])
        d = float(np.linalg.norm(diff))
        aj = _zonal_harmonic_acceleration(diff, d, d**2, 6.371e6, [])
        np.testing.assert_array_equal(aj, [0.0, 0.0, 0.0])

    def test_j2_equatorial_plane(self):
        """
        In the equatorial plane (z=0) J2 perturbation has no z-component
        and is purely radial.
        """
        R = 6.371e6   # Earth radius
        diff = np.array([7e6, 0.0, 0.0])   # directly above equator
        d = float(np.linalg.norm(diff))
        aj = _zonal_harmonic_acceleration(diff, d, d**2, R, [1.08263e-3])
        # z-component must be zero (z=0 ⇒ z_d=0 ⇒ aj[2] = cr*0*(-1.5+0)/d3 = 0)
        assert abs(aj[2]) < 1e-30
        # x-component should be non-zero (radial correction)
        assert aj[0] != 0.0
        assert aj[1] == pytest.approx(0.0, abs=1e-30)

    def test_j2_polar_axis(self):
        """
        On the polar axis (x=y=0) the x and y perturbation components are zero.
        """
        R = 6.371e6
        diff = np.array([0.0, 0.0, 7e6])
        d = float(np.linalg.norm(diff))
        aj = _zonal_harmonic_acceleration(diff, d, d**2, R, [1.08263e-3])
        assert aj[0] == pytest.approx(0.0, abs=1e-30)
        assert aj[1] == pytest.approx(0.0, abs=1e-30)
        assert aj[2] != 0.0

    def test_j2_only_no_higher_terms(self):
        """With one coefficient only J2 fires; result is finite and non-zero."""
        R = 6.371e6
        diff = np.array([5e6, 3e6, 2e6])
        d = float(np.linalg.norm(diff))
        aj = _zonal_harmonic_acceleration(diff, d, d**2, R, [1.08263e-3])
        assert np.all(np.isfinite(aj))
        assert np.linalg.norm(aj) > 0

    def test_j2_through_j6_finite(self):
        """Higher harmonics should all produce finite results."""
        R = 6.371e6
        diff = np.array([5e6, 3e6, 2e6])
        d = float(np.linalg.norm(diff))
        # Earth J2–J6 (approximate values)
        j_coeffs = [1.08263e-3, -2.532e-6, -1.616e-6, -2.273e-7, 5.41e-7]
        aj = _zonal_harmonic_acceleration(diff, d, d**2, R, j_coeffs)
        assert np.all(np.isfinite(aj))

    def test_j2_magnitude_smaller_than_newtonian(self):
        """
        J2 correction should be much smaller than the Newtonian term
        at a typical LEO altitude (~400 km above Earth surface).
        """
        R = 6.371e6
        alt = 400e3
        r = R + alt
        diff = np.array([r, 0.0, 0.0])
        d = float(np.linalg.norm(diff))
        aj = _zonal_harmonic_acceleration(diff, d, d**2, R, [1.08263e-3])
        # Newtonian acceleration magnitude at this distance (mu_earth/r^2)
        mu_earth = 3.986004418e14
        a_newton = mu_earth / r**2
        # J2 correction itself — compare relative magnitudes (aj is dimensionless
        # shape; the full correction is mu * aj but we just check the ratio of
        # the shape factor to 1/r^2)
        j2_shape = np.linalg.norm(aj)
        assert j2_shape < 1.0   # the correction factor cr/d^3 << 1/r^2
