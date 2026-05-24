"""Tests for keppy.solar_system — bodies catalogue and Horizons fetcher."""

import math
import socket
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from keppy.body import Body
from keppy.nbody_system import NBodySystem
from keppy.constants import AU, MU_SUN, G
from keppy.solar_system.bodies import (
    BodyData,
    BODY_DATA,
    make_body,
    sun, mercury, venus, earth, moon,
    mars, jupiter, saturn, uranus, neptune, pluto, ceres,
)
from keppy.solar_system.horizons import (
    HorizonsError,
    HORIZONS_IDS,
    _next_day,
    _parse_vectors,
    fetch_body,
    fetch_system,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Realistic mock responses matching the actual Horizons VECTORS text format.
# X, Y, Z in km; VX, VY, VZ in km/s.
_EARTH_RESPONSE = """\
*******************************************************************************
 Revised: Apr 12, 2021             Earth                                    399
*******************************************************************************
$$SOE
2451545.000000000 = A.D. 2000-Jan-01 12:00:00.0000 TDB \
 X =-2.622975507869803E+07 Y = 1.445063726040879E+08 Z =-5.595440773060024E+03
 VX=-2.978452369399764E+01 VY=-5.205463699379028E+00 VZ= 6.984209786391832E-04
 LT= 4.994877012988E+02 RG= 1.496178174854E+08 RR=-3.131424498E-01
$$EOE
*******************************************************************************
"""

_SUN_RESPONSE = """\
*******************************************************************************
 Revised: Jan 18, 2022              Sun                                       10
*******************************************************************************
$$SOE
2451545.000000000 = A.D. 2000-Jan-01 12:00:00.0000 TDB \
 X =-3.897930219018695E+05 Y = 6.427044289696960E+05 Z = 1.100384527578082E+04
 VX=-8.206665649543551E-03 VY=-7.210665982920861E-03 VZ= 2.380291936543019E-04
 LT= 2.504628001395E+00 RG= 7.503553049082E+05 RR=-5.023047462E-03
$$EOE
*******************************************************************************
"""

_JUPITER_RESPONSE = """\
*******************************************************************************
$$SOE
2451545.000000000 = A.D. 2000-Jan-01 12:00:00.0000 TDB \
 X = 5.974987428432618E+08 Y = 4.387489614764048E+08 Z =-1.521127987899905E+07
 VX=-7.957994660451476E+00 VY= 1.101800419049843E+01 VZ= 1.324438975660636E-01
 LT= 2.409601021879E+03 RG= 7.223960285268E+08 RR=-9.178698088E-01
$$EOE
*******************************************************************************
"""


def _mock_urlopen(response_text: str):
    """Return a context manager that stubs urlopen to yield *response_text*."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_text.encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return patch("urllib.request.urlopen", return_value=mock_resp)


def _horizons_live() -> bool:
    """Return True if JPL Horizons API responds (not just TCP-connectable)."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND=10"
            "&OBJ_DATA=NO&MAKE_EPHEM=NO",
            headers={"User-Agent": "keppy-test/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


_live = _horizons_live()
requires_live = pytest.mark.skipif(
    not _live,
    reason="JPL Horizons API not reachable from this environment",
)


# ---------------------------------------------------------------------------
# BodyData dataclass
# ---------------------------------------------------------------------------

class TestBodyData:

    def test_mass_property(self):
        bd = BodyData(mu=MU_SUN, radius=6.957e8)
        assert bd.mass == pytest.approx(MU_SUN / G, rel=1e-10)

    def test_frozen(self):
        bd = BodyData(mu=1.0, radius=1.0)
        with pytest.raises((AttributeError, TypeError)):
            bd.mu = 2.0  # type: ignore[misc]

    def test_j_coefficients_default_empty(self):
        bd = BodyData(mu=1.0, radius=1.0)
        assert bd.j_coefficients == ()


# ---------------------------------------------------------------------------
# BODY_DATA catalogue
# ---------------------------------------------------------------------------

class TestBodyDataCatalogue:

    def test_all_expected_keys_present(self):
        expected = {
            "sun", "mercury", "venus", "earth", "moon",
            "mars", "jupiter", "saturn", "uranus", "neptune",
            "pluto", "ceres",
        }
        assert expected.issubset(BODY_DATA.keys())

    def test_sun_mu_matches_constant(self):
        assert BODY_DATA["sun"].mu == pytest.approx(MU_SUN, rel=1e-6)

    def test_earth_mu_correct_order_of_magnitude(self):
        # Earth mu ≈ 3.986e14 m³/s²
        assert 3.9e14 < BODY_DATA["earth"].mu < 4.0e14

    def test_earth_j_coefficients(self):
        j = BODY_DATA["earth"].j_coefficients
        assert len(j) >= 1
        # J2 ≈ 1.08e-3
        assert j[0] == pytest.approx(1.08263e-3, rel=1e-3)

    def test_giant_planet_j2_larger_than_rocky(self):
        """Giant planets (Saturn, Jupiter) should have J2 >> rocky planets."""
        saturn_j2  = BODY_DATA["saturn"].j_coefficients[0]
        jupiter_j2 = BODY_DATA["jupiter"].j_coefficients[0]
        for name in ("mercury", "venus", "earth", "mars"):
            rocky_j2 = BODY_DATA[name].j_coefficients[0]
            assert rocky_j2 < jupiter_j2, f"{name}.J2 >= jupiter.J2"
            assert rocky_j2 < saturn_j2,  f"{name}.J2 >= saturn.J2"
        # Saturn is more oblate than Jupiter (lower density, faster rotation)
        assert saturn_j2 > jupiter_j2

    def test_all_radii_positive(self):
        for name, bd in BODY_DATA.items():
            assert bd.radius > 0, f"{name}: radius={bd.radius}"

    def test_all_mu_positive(self):
        for name, bd in BODY_DATA.items():
            assert bd.mu > 0, f"{name}: mu={bd.mu}"

    def test_sun_largest_mu(self):
        sun_mu = BODY_DATA["sun"].mu
        for name, bd in BODY_DATA.items():
            if name != "sun":
                assert bd.mu < sun_mu, f"{name}.mu >= Sun.mu"

    def test_sun_largest_radius(self):
        sun_r = BODY_DATA["sun"].radius
        for name, bd in BODY_DATA.items():
            if name != "sun":
                assert bd.radius < sun_r


# ---------------------------------------------------------------------------
# make_body
# ---------------------------------------------------------------------------

class TestMakeBody:

    def test_returns_body(self):
        b = make_body("earth")
        assert isinstance(b, Body)

    def test_mu_correct(self):
        b = make_body("sun")
        assert b.mu == pytest.approx(MU_SUN, rel=1e-6)

    def test_default_position_zero(self):
        b = make_body("earth")
        np.testing.assert_array_equal(b.position, np.zeros(3))

    def test_default_velocity_zero(self):
        b = make_body("mars")
        np.testing.assert_array_equal(b.velocity, np.zeros(3))

    def test_explicit_position(self):
        pos = np.array([AU, 0.0, 0.0])
        b = make_body("earth", position=pos)
        np.testing.assert_array_equal(b.position, pos)

    def test_radius_set(self):
        b = make_body("earth")
        assert b.radius == pytest.approx(BODY_DATA["earth"].radius)

    def test_j_coefficients_set(self):
        b = make_body("earth")
        assert b.j_coefficients == list(BODY_DATA["earth"].j_coefficients)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="nibiru"):
            make_body("nibiru")

    def test_case_insensitive(self):
        b = make_body("EARTH")
        assert b.mu == pytest.approx(BODY_DATA["earth"].mu, rel=1e-10)

    def test_j_coefficients_are_list_not_tuple(self):
        """make_body must return a mutable list, not the frozen tuple."""
        b = make_body("earth")
        assert isinstance(b.j_coefficients, list)


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

class TestConvenienceAliases:

    @pytest.mark.parametrize("func,key", [
        (sun,     "sun"),
        (mercury, "mercury"),
        (venus,   "venus"),
        (earth,   "earth"),
        (moon,    "moon"),
        (mars,    "mars"),
        (jupiter, "jupiter"),
        (saturn,  "saturn"),
        (uranus,  "uranus"),
        (neptune, "neptune"),
        (pluto,   "pluto"),
        (ceres,   "ceres"),
    ])
    def test_alias_mu(self, func, key):
        b = func()
        assert b.mu == pytest.approx(BODY_DATA[key].mu, rel=1e-9)

    def test_earth_with_velocity(self):
        v = np.array([0.0, 29784.69, 0.0])
        b = earth(position=np.array([AU, 0.0, 0.0]), velocity=v)
        np.testing.assert_array_equal(b.velocity, v)


# ---------------------------------------------------------------------------
# HORIZONS_IDS mapping
# ---------------------------------------------------------------------------

class TestHorizonsIDs:

    def test_all_planets_present(self):
        for name in ["sun", "mercury", "venus", "earth", "mars",
                     "jupiter", "saturn", "uranus", "neptune"]:
            assert name in HORIZONS_IDS

    def test_sun_id(self):
        assert HORIZONS_IDS["sun"] == "10"

    def test_earth_id(self):
        assert HORIZONS_IDS["earth"] == "399"   # geocenter, not EM barycenter

    def test_all_ids_are_strings(self):
        for name, hid in HORIZONS_IDS.items():
            assert isinstance(hid, str), f"{name}: ID is not a str"


# ---------------------------------------------------------------------------
# _next_day helper
# ---------------------------------------------------------------------------

class TestNextDay:

    def test_basic(self):
        assert _next_day("2000-01-01") == "2000-01-02"

    def test_month_rollover(self):
        assert _next_day("2000-01-31") == "2000-02-01"

    def test_year_rollover(self):
        assert _next_day("2000-12-31") == "2001-01-01"

    def test_ignores_time_component(self):
        # Only the first 10 chars (YYYY-MM-DD) are used
        assert _next_day("2024-06-15 12:00:00") == "2024-06-16"


# ---------------------------------------------------------------------------
# _parse_vectors
# ---------------------------------------------------------------------------

class TestParseVectors:

    def test_earth_position_in_meters(self):
        pos, vel = _parse_vectors(_EARTH_RESPONSE)
        # X = -2.622975507869803E+07 km → -2.622975507869803E+10 m
        assert pos[0] == pytest.approx(-2.622975507869803e7 * 1e3, rel=1e-10)

    def test_earth_velocity_in_mps(self):
        pos, vel = _parse_vectors(_EARTH_RESPONSE)
        # VX = -2.978452369399764E+01 km/s → -2.978452369399764E+04 m/s
        assert vel[0] == pytest.approx(-2.978452369399764e1 * 1e3, rel=1e-10)

    def test_earth_position_shape(self):
        pos, vel = _parse_vectors(_EARTH_RESPONSE)
        assert pos.shape == (3,)
        assert vel.shape == (3,)

    def test_sun_near_barycenter(self):
        """Sun's barycentric position is small (~7e8 m ≈ 1 solar radius)."""
        pos, vel = _parse_vectors(_SUN_RESPONSE)
        r = float(np.linalg.norm(pos))
        assert r < 2e9   # within ~3 solar radii

    def test_positive_y_component(self):
        """Earth Y is positive at J2000 (≈1.445e11 m)."""
        pos, _ = _parse_vectors(_EARTH_RESPONSE)
        assert pos[1] > 0

    def test_missing_soe_raises(self):
        with pytest.raises(HorizonsError, match="\\$\\$SOE"):
            _parse_vectors("some text without markers")

    def test_missing_xyz_raises(self):
        bad = "$$SOE\n2451545.0 = ...\n no position data here\n VX=0 VY=0 VZ=0\n$$EOE"
        with pytest.raises(HorizonsError, match="X/Y/Z"):
            _parse_vectors(bad)

    def test_missing_vel_raises(self):
        bad = "$$SOE\n2451545.0 = ...\n X = 1E+08 Y = 1E+08 Z = 1E+04\n no vel\n$$EOE"
        with pytest.raises(HorizonsError, match="VX/VY/VZ"):
            _parse_vectors(bad)

    def test_negative_values_parsed(self):
        """Negative X (no space between = and -) must parse correctly."""
        pos, _ = _parse_vectors(_EARTH_RESPONSE)
        assert pos[0] < 0   # Earth X is negative at J2000

    def test_jupiter_distance_from_sun(self):
        """Jupiter should be ~5 AU from the origin at J2000."""
        pos, _ = _parse_vectors(_JUPITER_RESPONSE)
        r = float(np.linalg.norm(pos))
        assert 4 * AU < r < 6 * AU


# ---------------------------------------------------------------------------
# fetch_body — mocked HTTP
# ---------------------------------------------------------------------------

class TestFetchBodyMocked:

    def test_returns_body(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth")
        assert isinstance(b, Body)

    def test_position_from_horizons(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth")
        # X = -2.622975507869803e7 km → -2.622975507869803e10 m
        assert b.position[0] == pytest.approx(-2.622975507869803e10, rel=1e-9)

    def test_velocity_from_horizons(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth")
        # VX = -29.784... km/s → -29784... m/s
        assert b.velocity[0] == pytest.approx(-29784.52369399764, rel=1e-9)

    def test_mu_from_catalogue(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth")
        assert b.mu == pytest.approx(BODY_DATA["earth"].mu, rel=1e-9)

    def test_radius_from_catalogue(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth")
        assert b.radius == pytest.approx(BODY_DATA["earth"].radius)

    def test_j_coefficients_from_catalogue(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth")
        assert b.j_coefficients == list(BODY_DATA["earth"].j_coefficients)

    def test_mu_override(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth", mu=1.0e14)
        assert b.mu == pytest.approx(1.0e14)

    def test_mass_override(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth", mass=6.0e24)
        assert b.mu == pytest.approx(6.0e24 * G, rel=1e-9)

    def test_radius_override(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth", radius=7.0e6)
        assert b.radius == pytest.approx(7.0e6)

    def test_case_insensitive(self):
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("Earth")
        assert isinstance(b, Body)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="nibiru"):
            fetch_body("nibiru")

    def test_explicit_horizons_id(self):
        """Passing horizons_id bypasses the HORIZONS_IDS table."""
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("earth", horizons_id="399", mu=BODY_DATA["earth"].mu)
        assert isinstance(b, Body)

    def test_unlisted_body_needs_mu(self):
        """Fetching an unlisted body without mu/mass raises HorizonsError."""
        with _mock_urlopen(_EARTH_RESPONSE):
            with pytest.raises(HorizonsError, match="physical data"):
                fetch_body("halley", horizons_id="900033")

    def test_unlisted_body_with_mu(self):
        """Fetching an unlisted body with explicit mu should succeed."""
        with _mock_urlopen(_EARTH_RESPONSE):
            b = fetch_body("halley", horizons_id="900033", mu=1.0e9)
        assert b.mu == pytest.approx(1.0e9)

    def test_http_error_raises_horizons_error(self):
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="", code=404, msg="Not Found", hdrs={}, fp=None,
            ),
        ):
            with pytest.raises(HorizonsError, match="HTTP 404"):
                fetch_body("earth")

    def test_url_error_raises_horizons_error(self):
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Name or service not known"),
        ):
            with pytest.raises(HorizonsError, match="reach"):
                fetch_body("earth")


# ---------------------------------------------------------------------------
# fetch_system — mocked HTTP
# ---------------------------------------------------------------------------

class TestFetchSystemMocked:

    def _responses(self, *texts):
        """Patch urlopen to return texts in sequence."""
        mocks = []
        for text in texts:
            m = MagicMock()
            m.read.return_value = text.encode()
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            mocks.append(m)
        return patch("urllib.request.urlopen", side_effect=mocks)

    def test_returns_nbody_system(self):
        with self._responses(_SUN_RESPONSE, _EARTH_RESPONSE):
            sys = fetch_system(["sun", "earth"])
        assert isinstance(sys, NBodySystem)

    def test_body_count(self):
        with self._responses(_SUN_RESPONSE, _EARTH_RESPONSE, _JUPITER_RESPONSE):
            sys = fetch_system(["sun", "earth", "jupiter"])
        assert sys.n == 3

    def test_body_names(self):
        with self._responses(_SUN_RESPONSE, _EARTH_RESPONSE):
            sys = fetch_system(["sun", "earth"])
        assert sys.get_body_by_name("Sun")   is not None
        assert sys.get_body_by_name("Earth") is not None

    def test_no_barycenter_translation_by_default(self):
        """
        Horizons vectors are already barycentric (CENTER=500@0).
        translate_to_barycenter defaults to False so Sun stays near origin.
        """
        with self._responses(_SUN_RESPONSE, _EARTH_RESPONSE):
            sys = fetch_system(["sun", "earth"])
        sun_body = sys.get_body_by_name("Sun")
        r_sun = float(np.linalg.norm(sun_body.position))
        # Sun should be within ~3 solar radii of origin (~2e9 m)
        assert r_sun < 3e9

    def test_translate_to_barycenter_option(self):
        with self._responses(_SUN_RESPONSE, _EARTH_RESPONSE):
            sys = fetch_system(["sun", "earth"], translate_to_barycenter=True)
        # After translation, mass-weighted position sum ≈ 0
        bary = sys.barycenter()
        assert float(np.linalg.norm(bary)) < 1e8   # within 100 000 km


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

class TestPackageExports:

    def test_solar_system_importable(self):
        import keppy.solar_system as ss
        assert hasattr(ss, "BODY_DATA")
        assert hasattr(ss, "make_body")
        assert hasattr(ss, "fetch_body")
        assert hasattr(ss, "fetch_system")

    def test_horizons_ids_exported(self):
        from keppy.solar_system import HORIZONS_IDS
        assert "earth" in HORIZONS_IDS

    def test_body_factories_exported(self):
        from keppy.solar_system import earth, jupiter
        assert callable(earth)
        assert callable(jupiter)


# ---------------------------------------------------------------------------
# Live network tests (skipped when Horizons is not reachable)
# ---------------------------------------------------------------------------

class TestHorizonsLive:

    @requires_live
    def test_fetch_earth_position_magnitude(self):
        """Earth's distance from the SSB at J2000 should be ≈ 1 AU."""
        b = fetch_body("earth")
        r = float(np.linalg.norm(b.position))
        assert 0.9 * AU < r < 1.1 * AU

    @requires_live
    def test_fetch_earth_speed(self):
        """Earth's orbital speed at J2000 should be ≈ 29.8 km/s."""
        b = fetch_body("earth")
        v = float(np.linalg.norm(b.velocity))
        assert 28e3 < v < 32e3

    @requires_live
    def test_fetch_sun_near_barycenter(self):
        """Sun should be within ≈ 2 solar radii of the SSB."""
        b = fetch_body("sun")
        r = float(np.linalg.norm(b.position))
        assert r < 2 * BODY_DATA["sun"].radius

    @requires_live
    def test_fetch_system_two_bodies(self):
        sys = fetch_system(["sun", "earth"])
        assert sys.n == 2
