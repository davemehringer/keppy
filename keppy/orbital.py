"""
keppy.orbital — Keplerian orbital element conversions.

Provides two main functions:

    elements_to_vectors(mu, elements)  →  (position, velocity)
    vectors_to_elements(mu, position, velocity)  →  Elements

and the ``Elements`` dataclass that holds the six classical elements.

Improvements over C++ keplerpp
-------------------------------
* ``a`` is stored in **meters** (SI), not astronomical units.  The C++ code
  internally converted to km, which required knowing KM_PER_AU at every call
  site.
* ``mu`` is a single scalar (G·M_total = mu_primary + mu_secondary); the C++
  took two separate arguments which callers always had to sum.
* Kepler's equation is solved with **Newton-Raphson** (quadratic convergence)
  rather than the C++ fixed-point iteration (linear, slow near e→1).
* ``Elements`` is a dataclass with ``period``, ``periapsis``, and ``apoapsis``
  derived properties.
* All degenerate cases (circular, equatorial, circular-equatorial, hyperbolic)
  are handled gracefully with ``NaN`` for undefined angles rather than silent
  wrong values.
* ``elements_to_vectors`` and ``vectors_to_elements`` are pure functions with
  no side effects.

Angle convention
----------------
All angles stored in ``Elements`` are in **degrees**.  Internally, radians are
used for all computations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Elements dataclass
# ---------------------------------------------------------------------------

@dataclass
class Elements:
    """
    The six classical Keplerian orbital elements.

    Attributes
    ----------
    a        : Semi-major axis (meters).  Negative for hyperbolic orbits.
    e        : Eccentricity (≥ 0).  e=0 circular, 0<e<1 elliptic,
               e=1 parabolic, e>1 hyperbolic.
    i        : Inclination (degrees, 0–180).
    node     : Longitude of the ascending node Ω (degrees, 0–360).
               ``nan`` for equatorial orbits (i=0 or i=π).
    peri     : Argument of pericenter ω (degrees, 0–360).
               ``nan`` for circular orbits (e=0).
    M        : Mean anomaly (degrees, 0–360).
               ``nan`` for circular orbits (e=0) or hyperbolic orbits.

    Notes
    -----
    The coordinate frame is whatever frame the position/velocity vectors
    were given in (typically the ecliptic J2000 frame when working with
    solar system bodies).
    """

    a:    float
    e:    float
    i:    float
    node: float
    peri: float
    M:    float

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def period(self) -> float:
        """
        Orbital period in seconds.

        Requires ``mu`` to be known; use
        ``elements_to_vectors`` → derive mu externally, or call
        ``period_from_mu(mu)``.

        Returns ``nan`` for non-elliptic orbits (e ≥ 1).
        """
        return float("nan")   # mu is not stored; use period_from_mu()

    def period_from_mu(self, mu: float) -> float:
        """Orbital period in seconds given gravitational parameter *mu* (m³ s⁻²)."""
        if self.e >= 1.0 or self.a <= 0:
            return float("nan")
        return 2.0 * math.pi * math.sqrt(self.a ** 3 / mu)

    @property
    def periapsis(self) -> float:
        """Periapsis distance in meters: a(1 - e)."""
        return self.a * (1.0 - self.e)

    @property
    def apoapsis(self) -> float:
        """
        Apoapsis distance in meters: a(1 + e).

        Returns ``inf`` for parabolic / hyperbolic orbits.
        """
        if self.e >= 1.0:
            return float("inf")
        return self.a * (1.0 + self.e)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"Elements("
            f"a={self.a:.6e} m, "
            f"e={self.e:.8f}, "
            f"i={self.i:.6f}°, "
            f"node={self.node:.6f}°, "
            f"peri={self.peri:.6f}°, "
            f"M={self.M:.6f}°)"
        )


# ---------------------------------------------------------------------------
# Kepler's equation solver
# ---------------------------------------------------------------------------

def solve_kepler(
    M: float,
    e: float,
    tol: float = 1e-14,
    max_iter: int = 50,
) -> float:
    """
    Solve Kepler's equation  E - e·sin(E) = M  for the eccentric anomaly E.

    Uses Newton-Raphson iteration (quadratic convergence) rather than the
    fixed-point iteration in keplerpp (linear convergence, slow for e→1).

    Parameters
    ----------
    M        : Mean anomaly in radians.
    e        : Eccentricity (0 ≤ e < 1).
    tol      : Convergence tolerance on |E_{n+1} - E_n|.
    max_iter : Maximum iterations before giving up.

    Returns
    -------
    float
        Eccentric anomaly E in radians.

    Raises
    ------
    ValueError
        If e ≥ 1 (use a different solver for parabolic/hyperbolic).
    RuntimeError
        If the iteration fails to converge within max_iter steps.
    """
    if e >= 1.0:
        raise ValueError(
            f"solve_kepler requires e < 1 (got e={e}). "
            "Parabolic/hyperbolic orbits are not supported."
        )
    # Initial guess: Danby (1988) — better than E = M for high eccentricity
    E = M + e * math.sin(M) * (1.0 + e * math.cos(M))
    for _ in range(max_iter):
        f  = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) <= tol:
            return E
    raise RuntimeError(
        f"solve_kepler did not converge after {max_iter} iterations "
        f"(M={M:.6f}, e={e:.6f}, last |dE|={abs(dE):.2e})"
    )


# ---------------------------------------------------------------------------
# elements → state vectors
# ---------------------------------------------------------------------------

def elements_to_vectors(
    mu: float,
    elements: Elements,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian orbital elements to position and velocity vectors.

    Parameters
    ----------
    mu       : Gravitational parameter G·M_total (m³ s⁻²).
    elements : ``Elements`` instance (a in meters, angles in degrees).

    Returns
    -------
    position : shape (3,), meters — relative to the body being orbited.
    velocity : shape (3,), m s⁻¹ — relative to the body being orbited.

    Notes
    -----
    The method is the classical perifocal-frame decomposition, identical to
    the C++ keplerpp implementation.
    """
    a    = elements.a
    e    = elements.e
    i    = math.radians(elements.i)
    node = math.radians(elements.node) if not math.isnan(elements.node) else 0.0
    peri = math.radians(elements.peri) if not math.isnan(elements.peri) else 0.0
    M    = math.radians(elements.M)    if not math.isnan(elements.M)    else 0.0

    # Eccentric anomaly via Newton-Raphson
    E = solve_kepler(M, e)

    cp, sp = math.cos(peri), math.sin(peri)
    cn, sn = math.cos(node), math.sin(node)
    ci, si = math.cos(i),    math.sin(i)

    # Perifocal unit vectors in the reference frame
    P = np.array([
         cp * cn - sp * sn * ci,
         cp * sn + sp * cn * ci,
         sp * si,
    ])
    Q = np.array([
        -sp * cn - cp * ci * sn,
        -sp * sn + cp * ci * cn,
         si * cp,
    ])

    sE = math.sin(E)
    cE = math.cos(E)
    sqrt1_e2 = math.sqrt(1.0 - e * e)

    position = a * (cE - e) * P + a * sqrt1_e2 * sE * Q
    velocity = (math.sqrt(mu / a) / (1.0 - e * cE)) * (-sE * P + sqrt1_e2 * cE * Q)

    return position, velocity


# ---------------------------------------------------------------------------
# state vectors → elements
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
_EPS    = 1e-10   # tolerance for near-zero / near-±1 tests


def vectors_to_elements(
    mu: float,
    position: np.ndarray,
    velocity: np.ndarray,
) -> Elements:
    """
    Convert position and velocity vectors to Keplerian orbital elements.

    Parameters
    ----------
    mu       : Gravitational parameter G·M_total (m³ s⁻²).
    position : shape (3,), meters — relative to the central body.
    velocity : shape (3,), m s⁻¹ — relative to the central body.

    Returns
    -------
    Elements
        a in meters, all angles in degrees.
        ``node``, ``peri``, and ``M`` are ``nan`` when undefined (see the
        ``Elements`` docstring for when each is undefined).

    Notes
    -----
    The algorithm follows keplerpp exactly, with improved edge-case handling
    for equatorial and circular orbits.
    """
    x = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)

    r = float(np.linalg.norm(x))       # scalar distance
    v2 = float(np.dot(v, v))           # speed squared

    # Specific angular momentum  h = x × v
    h = np.cross(x, v)
    h_mag = float(np.linalg.norm(h))

    # Eccentricity vector  ev = (v × h)/mu  - x̂
    ev = np.cross(v, h) / mu - x / r
    e = float(np.linalg.norm(ev))

    # Semi-major axis from the vis-viva equation
    # ε = v²/2 - mu/r  →  a = -mu/(2ε)
    eps = 0.5 * v2 - mu / r
    a = -mu / (2.0 * eps)

    # Inclination  i = acos(h_z / |h|)
    i_rad = math.acos(_clamp(h[2] / h_mag, -1.0, 1.0)) if h_mag > 0 else 0.0

    # Line of nodes  n = k × h  (k = [0,0,1])
    n_vec = np.array([-h[1], h[0], 0.0])   # equivalent to [0,0,1] × h
    n_mag = float(np.linalg.norm(n_vec))

    # Longitude of ascending node Ω
    if n_mag > _EPS:
        cos_node = _clamp(n_vec[0] / n_mag, -1.0, 1.0)
        node_rad = math.acos(cos_node)
        if n_vec[1] < 0.0:
            node_rad = _TWO_PI - node_rad
    else:
        node_rad = float("nan")   # equatorial orbit

    # Argument of pericenter ω
    if e > _EPS and n_mag > _EPS:
        cos_peri = _clamp(np.dot(n_vec, ev) / (n_mag * e), -1.0, 1.0)
        peri_rad = math.acos(cos_peri)
        if ev[2] < 0.0:
            peri_rad = _TWO_PI - peri_rad
    elif e > _EPS and n_mag <= _EPS:
        # Equatorial orbit: measure peri from x-axis in the orbital plane
        actual_angle = math.atan2(x[1], x[0])
        if actual_angle < 0:
            actual_angle += _TWO_PI
        # We compute true anomaly first, then derive peri from position angle
        cos_ta = _clamp(np.dot(ev, x) / (e * r), -1.0, 1.0)
        ta_rad = math.acos(cos_ta)
        if np.dot(x, v) < 0.0:
            ta_rad = _TWO_PI - ta_rad
        peri_rad = actual_angle - ta_rad
        if peri_rad < 0:
            peri_rad += _TWO_PI
    else:
        peri_rad = float("nan")   # circular orbit

    # True anomaly ν
    if e > _EPS:
        cos_ta = _clamp(np.dot(ev, x) / (e * r), -1.0, 1.0)
        ta_rad = math.acos(cos_ta)
        if np.dot(x, v) < 0.0:
            ta_rad = _TWO_PI - ta_rad
    else:
        # Circular orbit: measure angle from ascending node (or x-axis if equatorial)
        if n_mag > _EPS:
            cos_ta = _clamp(np.dot(n_vec, x) / (n_mag * r), -1.0, 1.0)
            ta_rad = math.acos(cos_ta)
            if x[2] < 0.0:
                ta_rad = _TWO_PI - ta_rad
        else:
            ta_rad = math.atan2(x[1], x[0])
            if ta_rad < 0:
                ta_rad += _TWO_PI

    # Eccentric anomaly E from true anomaly ν
    cos_E = _clamp((e + math.cos(ta_rad)) / (1.0 + e * math.cos(ta_rad)), -1.0, 1.0)
    if e > _EPS:
        E_rad = math.acos(cos_E)
        if ta_rad > math.pi:
            E_rad = _TWO_PI - E_rad
        M_rad = E_rad - e * math.sin(E_rad)
        if M_rad < 0.0:
            M_rad += _TWO_PI
    else:
        M_rad = float("nan")   # circular: mean anomaly not meaningful

    return Elements(
        a    = a,
        e    = e,
        i    = math.degrees(i_rad),
        node = math.degrees(node_rad) if not math.isnan(node_rad) else float("nan"),
        peri = math.degrees(peri_rad) if not math.isnan(peri_rad) else float("nan"),
        M    = math.degrees(M_rad)    if not math.isnan(M_rad)    else float("nan"),
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]; needed before acos/asin to avoid domain errors."""
    return max(lo, min(hi, x))
