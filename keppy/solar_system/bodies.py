"""
keppy.solar_system.bodies — Physical data and Body factory functions for
solar system objects.

Physical constants
------------------
All gravitational parameters (mu) are taken from the JPL DE440/DE441
planetary ephemeris documentation.  Radii are mean equatorial radii from
the IAU Working Group on Cartographic Coordinates and Rotational Elements
(2015 report).  Zonal harmonic coefficients (J2, J3, …) are from the
relevant spacecraft mission summaries.

Convention
----------
``j_coefficients[0] = J2``, ``[1] = J3``, etc.  Bodies with negligible
higher-order terms (e.g. Mercury) carry only J2.  Giant planets with
symmetric zonal harmonics include zeros at odd-n positions so that the
even terms align at their correct indices.

Usage
-----
    from keppy.solar_system.bodies import earth, make_body
    import numpy as np

    # Convenience factory — position and velocity default to zero
    e = earth()

    # With explicit state vectors (e.g., from Horizons)
    e = make_body("earth",
                  position=np.array([...]),
                  velocity=np.array([...]))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from keppy.body import Body
from keppy.constants import G


# ---------------------------------------------------------------------------
# BodyData — physical parameters only (no orbital state)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BodyData:
    """
    Physical parameters for a solar system body.

    Attributes
    ----------
    mu              : Gravitational parameter G·M (m³ s⁻²).
    radius          : Mean equatorial radius (meters).
    j_coefficients  : Zonal harmonic coefficients tuple (J2, J3, J4, …).
                      Empty tuple for spherically symmetric bodies.
    """
    mu:             float
    radius:         float
    j_coefficients: tuple[float, ...] = ()

    @property
    def mass(self) -> float:
        """Body mass in kg: mu / G."""
        return self.mu / G


# ---------------------------------------------------------------------------
# Physical data catalogue
# ---------------------------------------------------------------------------

BODY_DATA: dict[str, BodyData] = {
    # -----------------------------------------------------------------------
    # Sun
    # -----------------------------------------------------------------------
    "sun": BodyData(
        mu     = 1.32712440018e20,   # DE440, m³/s²
        radius = 6.957e8,            # IAU 2015, m
    ),

    # -----------------------------------------------------------------------
    # Rocky planets
    # -----------------------------------------------------------------------
    "mercury": BodyData(
        mu     = 2.2032e13,
        radius = 2.4397e6,
        j_coefficients = (6.0e-5,),  # J2 only
    ),
    "venus": BodyData(
        mu     = 3.24859e14,
        radius = 6.0518e6,
        j_coefficients = (4.458e-6,),
    ),
    "earth": BodyData(
        mu     = 3.986004418e14,
        radius = 6.3781366e6,
        # J2, J3, J4 from EGM2008
        j_coefficients = (1.08263e-3, -2.53266e-6, -1.61962e-6),
    ),
    "moon": BodyData(
        mu     = 4.9048695e12,
        radius = 1.7374e6,
        j_coefficients = (2.027e-4,),
    ),
    "mars": BodyData(
        mu     = 4.282837e13,
        radius = 3.3895e6,
        # J2, J3 from Mars Global Surveyor
        j_coefficients = (1.9606e-3, 3.145e-5),
    ),

    # -----------------------------------------------------------------------
    # Gas/ice giants
    # -----------------------------------------------------------------------
    "jupiter": BodyData(
        mu     = 1.26686534e17,
        radius = 7.1492e7,
        # J2, J3=0 (symmetric), J4 from Juno
        j_coefficients = (1.4697e-2, 0.0, -5.84e-4),
    ),
    "saturn": BodyData(
        mu     = 3.7931187e16,
        radius = 6.0268e7,
        # J2, J3=0, J4 from Cassini
        j_coefficients = (1.6298e-2, 0.0, -9.36e-4),
    ),
    "uranus": BodyData(
        mu     = 5.793951e15,
        radius = 2.5559e7,
        j_coefficients = (3.343e-3,),
    ),
    "neptune": BodyData(
        mu     = 6.836529e15,
        radius = 2.4764e7,
        j_coefficients = (3.411e-3,),
    ),

    # -----------------------------------------------------------------------
    # Dwarf planets / minor planets
    # -----------------------------------------------------------------------
    "pluto": BodyData(
        mu     = 8.71e11,
        radius = 1.1883e6,
    ),
    "ceres": BodyData(
        mu     = 6.26325e10,
        radius = 4.73e5,
    ),
}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_body(
    name: str,
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
) -> Body:
    """
    Construct a :class:`~keppy.Body` with physical parameters from the
    built-in catalogue.

    Parameters
    ----------
    name     : Body name (case-insensitive).  Must be a key in
               :data:`BODY_DATA`.
    position : Initial position (m), shape (3,).  Defaults to the origin.
    velocity : Initial velocity (m s⁻¹), shape (3,).  Defaults to zero.

    Returns
    -------
    Body
        ``body.mu`` and ``body.radius`` are set from the catalogue.
        ``body.j_coefficients`` is a list copy of the catalogue tuple.

    Raises
    ------
    KeyError
        If *name* is not recognised.
    """
    key = name.lower()
    if key not in BODY_DATA:
        supported = ", ".join(sorted(BODY_DATA))
        raise KeyError(
            f"Unknown body: {name!r}.  Supported names: {supported}."
        )
    bd  = BODY_DATA[key]
    pos = np.zeros(3) if position is None else np.asarray(position, dtype=float)
    vel = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)

    body = Body.from_mu(name.capitalize() if name == name.lower() else name,
                        bd.mu, pos, vel)
    body.radius         = bd.radius
    body.j_coefficients = list(bd.j_coefficients)
    return body


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def sun(     position=None, velocity=None) -> Body: return make_body("sun",     position, velocity)
def mercury( position=None, velocity=None) -> Body: return make_body("mercury", position, velocity)
def venus(   position=None, velocity=None) -> Body: return make_body("venus",   position, velocity)
def earth(   position=None, velocity=None) -> Body: return make_body("earth",   position, velocity)
def moon(    position=None, velocity=None) -> Body: return make_body("moon",    position, velocity)
def mars(    position=None, velocity=None) -> Body: return make_body("mars",    position, velocity)
def jupiter( position=None, velocity=None) -> Body: return make_body("jupiter", position, velocity)
def saturn(  position=None, velocity=None) -> Body: return make_body("saturn",  position, velocity)
def uranus(  position=None, velocity=None) -> Body: return make_body("uranus",  position, velocity)
def neptune( position=None, velocity=None) -> Body: return make_body("neptune", position, velocity)
def pluto(   position=None, velocity=None) -> Body: return make_body("pluto",   position, velocity)
def ceres(   position=None, velocity=None) -> Body: return make_body("ceres",   position, velocity)
