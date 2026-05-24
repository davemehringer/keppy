"""Tests for keppy.nbody_system."""

import numpy as np
import pytest

from keppy.body import Body
from keppy.nbody_system import NBodySystem
from keppy.constants import G


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def two_body_system(translate: bool = True) -> NBodySystem:
    """
    Sun + Earth in circular orbit, separated by 1 AU.
    Placed symmetrically about their shared barycentre so that
    translate_to_barycenter() is effectively a no-op.
    """
    mu_sun = 1.327_124_4e20  # m^3 s^-2
    mass_sun = mu_sun / G
    mass_earth = 5.972e24

    M = mass_sun + mass_earth
    r = 1.495_978_707e11          # 1 AU in metres
    # Circular orbital speed for the relative motion
    v_rel = (G * M / r) ** 0.5

    # Place each body at its barycentre offset
    x_sun = -r * mass_earth / M
    x_earth = r * mass_sun / M
    v_sun = -v_rel * mass_earth / M
    v_earth = v_rel * mass_sun / M

    sun = Body.from_mass("Sun", mass_sun,
                         position=np.array([x_sun, 0.0, 0.0]),
                         velocity=np.array([0.0, v_sun, 0.0]))
    earth = Body.from_mass("Earth", mass_earth,
                            position=np.array([x_earth, 0.0, 0.0]),
                            velocity=np.array([0.0, v_earth, 0.0]))
    return NBodySystem([sun, earth], translate_to_barycenter=translate)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestNBodySystemConstruction:

    def test_body_count(self):
        sys = two_body_system()
        assert sys.n == 2

    def test_body_ids_assigned(self):
        sys = two_body_system()
        ids = [b.body_id for b in sys.bodies]
        assert 0 in ids and 1 in ids

    def test_duplicate_id_raises(self):
        sun = Body.from_mass("Sun", 2e30, np.zeros(3), np.zeros(3), body_id=0)
        earth = Body.from_mass("Earth", 6e24, np.array([1e11, 0, 0]),
                               np.array([0, 3e4, 0]), body_id=0)
        with pytest.raises(ValueError, match="id=0"):
            NBodySystem([sun, earth], translate_to_barycenter=False)


# ---------------------------------------------------------------------------
# Barycentre
# ---------------------------------------------------------------------------

class TestBarycenter:

    def test_barycenter_near_origin_after_translation(self):
        sys = two_body_system(translate=True)
        com = sys.barycenter()
        np.testing.assert_allclose(com, [0.0, 0.0, 0.0], atol=1.0)  # 1 m tolerance

    def test_linear_momentum_near_zero_after_translation(self):
        sys = two_body_system(translate=True)
        p = sys.linear_momentum
        np.testing.assert_allclose(p, [0.0, 0.0, 0.0], atol=1e10)  # generous for floats


# ---------------------------------------------------------------------------
# Conserved quantities
# ---------------------------------------------------------------------------

class TestConservedQuantities:

    def test_total_energy_is_negative(self):
        """Bound gravitational system has negative total energy."""
        sys = two_body_system()
        assert sys.total_energy < 0

    def test_kinetic_energy_positive(self):
        sys = two_body_system()
        assert sys.kinetic_energy > 0

    def test_potential_energy_negative(self):
        sys = two_body_system()
        assert sys.potential_energy < 0

    def test_angular_momentum_shape(self):
        sys = two_body_system()
        L = sys.angular_momentum
        assert L.shape == (3,)

    def test_angular_momentum_mostly_z(self):
        """Orbit in x-y plane → L should be predominantly in z."""
        sys = two_body_system()
        L = sys.angular_momentum
        assert abs(L[2]) > abs(L[0]) + abs(L[1])

    def test_total_mass(self):
        sys = two_body_system()
        expected = (1.327_124_4e20 / G) + 5.972e24
        assert sys.total_mass == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# State vector I/O
# ---------------------------------------------------------------------------

class TestStateIO:

    def test_get_state_shape(self):
        sys = two_body_system()
        pos, vel = sys.get_state()
        assert pos.shape == (2, 3)
        assert vel.shape == (2, 3)

    def test_set_state_roundtrip(self):
        sys = two_body_system()
        pos0, vel0 = sys.get_state()
        # perturb
        sys.set_state(pos0 * 2, vel0 * 2)
        pos1, vel1 = sys.get_state()
        np.testing.assert_allclose(pos1, pos0 * 2)
        np.testing.assert_allclose(vel1, vel0 * 2)

    def test_set_state_wrong_shape_raises(self):
        sys = two_body_system()
        with pytest.raises(ValueError):
            sys.set_state(np.zeros((3, 3)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

class TestTime:

    def test_initial_time_zero(self):
        sys = two_body_system()
        assert sys.time == pytest.approx(0.0)

    def test_time_days_years_conversion(self):
        from keppy.constants import DAY, YEAR
        sys = two_body_system()
        sys.time = 365.25 * DAY
        assert sys.time_days == pytest.approx(365.25)
        assert sys.time_years == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Body lookup
# ---------------------------------------------------------------------------

class TestBodyLookup:

    def test_get_body_by_name(self):
        sys = two_body_system()
        earth = sys.get_body_by_name("Earth")
        assert earth.name == "Earth"

    def test_get_body_missing_raises(self):
        sys = two_body_system()
        with pytest.raises(KeyError):
            sys.get_body_by_name("Jupiter")

    def test_remove_body(self):
        sys = two_body_system()
        earth = sys.get_body_by_name("Earth")
        sys.remove_body(earth.body_id)
        assert sys.n == 1
        assert sys.bodies[0].name == "Sun"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class TestDisplay:

    def test_repr_contains_n(self):
        sys = two_body_system()
        assert "n=2" in repr(sys)

    def test_str_contains_body_names(self):
        sys = two_body_system()
        s = str(sys)
        assert "Sun" in s
        assert "Earth" in s
