"""Tests for keppy.body."""

import numpy as np
import pytest

from keppy.body import Body, Frame, Origin
from keppy.constants import G


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBodyConstruction:

    def test_from_mass(self):
        b = Body.from_mass("Earth", 5.972e24,
                           position=np.array([1.5e11, 0.0, 0.0]),
                           velocity=np.array([0.0, 29_780.0, 0.0]))
        assert b.name == "Earth"
        assert b.mass == pytest.approx(5.972e24)
        assert b.mu == pytest.approx(G * 5.972e24)

    def test_from_mu(self):
        mu = 1.327_124_4e20  # Solar mu
        b = Body.from_mu("Sun", mu,
                         position=np.zeros(3),
                         velocity=np.zeros(3))
        assert b.mu == pytest.approx(mu)
        assert b.mass == pytest.approx(mu / G)

    def test_mass_mu_roundtrip(self):
        """Setting mass should update mu and vice-versa."""
        b = Body.from_mass("Test", 1e24,
                           position=np.zeros(3), velocity=np.zeros(3))
        b.mass = 2e24
        assert b.mu == pytest.approx(G * 2e24)

        b.mu = G * 3e24
        assert b.mass == pytest.approx(3e24)

    def test_vectors_are_float64(self):
        b = Body.from_mass("X", 1e10,
                           position=[1, 2, 3], velocity=[4, 5, 6])
        assert b.position.dtype == np.float64
        assert b.velocity.dtype == np.float64

    def test_bad_position_shape_raises(self):
        with pytest.raises(ValueError, match="3-vector"):
            Body.from_mass("bad", 1.0,
                           position=np.array([1.0, 2.0]),
                           velocity=np.zeros(3))

    def test_acceleration_initialized_to_zero(self):
        b = Body.from_mass("Z", 1e20, np.zeros(3), np.zeros(3))
        np.testing.assert_array_equal(b.acceleration, [0.0, 0.0, 0.0])

    def test_default_frame_and_origin(self):
        b = Body.from_mass("X", 1.0, np.zeros(3), np.zeros(3))
        assert b.frame is Frame.ECLIPTIC
        assert b.origin is Origin.SYSTEM_BARYCENTER

    def test_j_coefficients_default_empty(self):
        b = Body.from_mass("X", 1.0, np.zeros(3), np.zeros(3))
        assert b.j_coefficients == []

    def test_copy_is_independent(self):
        b = Body.from_mass("A", 1e20,
                           np.array([1.0, 0.0, 0.0]),
                           np.array([0.0, 1.0, 0.0]))
        c = b.copy()
        c.position[0] = 999.0
        assert b.position[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestBodyProperties:

    def test_speed(self):
        b = Body.from_mass("X", 1e10,
                           position=np.zeros(3),
                           velocity=np.array([3.0, 4.0, 0.0]))
        assert b.speed == pytest.approx(5.0)

    def test_kinetic_energy(self):
        """KE = ½ |v|²  (specific, per unit mass)."""
        b = Body.from_mass("X", 1e10,
                           position=np.zeros(3),
                           velocity=np.array([3.0, 4.0, 0.0]))
        assert b.kinetic_energy == pytest.approx(0.5 * 25.0)


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------

class TestBodyStr:

    def test_repr_contains_name(self):
        b = Body.from_mass("Mars", 6.39e23, np.zeros(3), np.zeros(3))
        assert "Mars" in repr(b)

    def test_str_contains_sections(self):
        b = Body.from_mass("Venus", 4.867e24, np.zeros(3), np.zeros(3))
        s = str(b)
        assert "Venus" in s
        assert "mass" in s
        assert "pos" in s
