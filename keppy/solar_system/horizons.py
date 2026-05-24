"""
keppy.solar_system.horizons — JPL Horizons state vector fetcher.

Queries the JPL Horizons web API (https://ssd.jpl.nasa.gov/api/horizons.api)
for barycentric Cartesian state vectors in the ecliptic J2000 frame, then
combines them with physical parameters from :mod:`keppy.solar_system.bodies`
to return ready-to-use :class:`~keppy.Body` objects.

No third-party libraries required — only Python stdlib (``urllib``,
``datetime``, ``re``).

Usage
-----
    from keppy.solar_system.horizons import fetch_body, fetch_system

    # Single body at J2000 epoch
    earth = fetch_body("earth")

    # Multiple bodies → NBodySystem (8-planet solar system)
    sys = fetch_system(
        ["sun", "mercury", "venus", "earth", "mars",
         "jupiter", "saturn", "uranus", "neptune"],
        epoch="2024-01-01",
    )

Supported body names
--------------------
See :data:`HORIZONS_IDS` for the full name → Horizons-ID mapping.
Additional bodies can be queried by passing ``horizons_id`` explicitly.

Coordinate system
-----------------
All vectors are returned in the **solar system barycentric frame**,
ecliptic J2000 (ICRF-referenced), consistent with JPL planetary ephemeris
DE440.  Positions are in **meters**, velocities in **m s⁻¹**.

Epoch
-----
The default epoch ``"2000-01-01"`` corresponds to J2000.0
(2000-Jan-01 12:00 TDB).  Any ``"YYYY-MM-DD"`` date is accepted.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import numpy as np

from keppy.body import Body
from keppy.nbody_system import NBodySystem


# ---------------------------------------------------------------------------
# Horizons body ID table
# ---------------------------------------------------------------------------

#: Map of lower-case body names → JPL Horizons numeric IDs.
#: Planets use barycentric IDs (1–8) to capture satellite-system mass.
#: "earth" uses the geocenter (399) so the Moon can be treated separately.
HORIZONS_IDS: dict[str, str] = {
    "sun":     "10",
    "mercury": "1",      # Mercury system barycenter
    "venus":   "2",      # Venus system barycenter
    "earth":   "399",    # Earth geocenter
    "moon":    "301",    # Moon
    "mars":    "4",      # Mars system barycenter
    "jupiter": "5",      # Jupiter system barycenter
    "saturn":  "6",      # Saturn system barycenter
    "uranus":  "7",      # Uranus system barycenter
    "neptune": "8",      # Neptune system barycenter
    "pluto":   "9",      # Pluto system barycenter
    "ceres":   "2000001",
}

_HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
_KM_TO_M      = 1e3      # km → m  and  km/s → m/s

# Float pattern that matches all Horizons number formats, including:
#   " 1.496154165785768E+08"   (positive with leading space)
#   "-2.622975507869803E+07"   (negative, no space)
#   " 6.984209786391832E-04"   (small positive)
_FP = r"[+-]?\d+\.?\d*(?:[Ee][+-]?\d+)?"

_XYZ_RE = re.compile(
    rf"X\s*=\s*({_FP})\s+"
    rf"Y\s*=\s*({_FP})\s+"
    rf"Z\s*=\s*({_FP})"
)
_VEL_RE = re.compile(
    rf"VX=\s*({_FP})\s+"
    rf"VY=\s*({_FP})\s+"
    rf"VZ=\s*({_FP})"
)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class HorizonsError(RuntimeError):
    """
    Raised when the JPL Horizons API returns an error or an unexpected
    response format.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_day(epoch: str) -> str:
    """Return the date one day after *epoch* (``'YYYY-MM-DD'`` format)."""
    dt = datetime.strptime(epoch[:10], "%Y-%m-%d")
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


def _get(url: str, params: dict[str, str], timeout: int) -> str:
    """
    HTTP GET to *url* with *params*; return the response body as a string.

    Raises
    ------
    HorizonsError
        On HTTP errors from the Horizons server.
    """
    query    = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    req      = urllib.request.Request(
        full_url,
        headers={"User-Agent": "keppy/1.0 (https://github.com/davemehringer/keppy)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise HorizonsError(
            f"Horizons API returned HTTP {exc.code} {exc.reason} "
            f"for URL: {full_url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise HorizonsError(
            f"Cannot reach JPL Horizons ({exc.reason}).  "
            "Check network connectivity."
        ) from exc


def _parse_vectors(text: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the **first** position and velocity vectors from a Horizons
    VECTORS format response text.

    The response contains a ``$$SOE`` / ``$$EOE`` block with one record per
    requested epoch.  Each record consists of three lines::

        <JD> = <calendar date>
         X = ...  Y = ...  Z = ...
         VX= ...  VY= ...  VZ= ...

    Parameters
    ----------
    text : Full text returned by the Horizons API.

    Returns
    -------
    position : shape (3,), meters.
    velocity : shape (3,), m s⁻¹.

    Raises
    ------
    HorizonsError
        If the response does not contain ``$$SOE``/``$$EOE`` markers, or the
        X/Y/Z and VX/VY/VZ values cannot be parsed.
    """
    soe = text.find("$$SOE")
    eoe = text.find("$$EOE")
    if soe < 0 or eoe < 0:
        # Surface the first 400 chars of the response for debugging
        preview = text[:400].replace("\n", " | ")
        raise HorizonsError(
            "$$SOE / $$EOE markers not found in Horizons response.  "
            f"Response preview: {preview!r}"
        )

    block = text[soe + 5 : eoe]

    m_xyz = _XYZ_RE.search(block)
    m_vel = _VEL_RE.search(block)

    if m_xyz is None:
        raise HorizonsError(
            "Could not parse X/Y/Z position from Horizons response block."
        )
    if m_vel is None:
        raise HorizonsError(
            "Could not parse VX/VY/VZ velocity from Horizons response block."
        )

    position = np.array([float(m_xyz.group(i)) for i in range(1, 4)]) * _KM_TO_M
    velocity = np.array([float(m_vel.group(i)) for i in range(1, 4)]) * _KM_TO_M

    return position, velocity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_body(
    name: str,
    epoch: str = "2000-01-01",
    *,
    center:      str       = "500@0",
    horizons_id: str | None = None,
    mu:          float | None = None,
    mass:        float | None = None,
    radius:      float | None = None,
    timeout:     int       = 30,
) -> Body:
    """
    Fetch a solar system body's state vectors from JPL Horizons and return
    a :class:`~keppy.Body` with those vectors and the body's physical
    parameters.

    Parameters
    ----------
    name        : Body name (case-insensitive).  Must be a key in
                  :data:`HORIZONS_IDS` unless *horizons_id* is given.
    epoch       : UTC date in ``'YYYY-MM-DD'`` format.  Defaults to J2000.0
                  (2000-01-01).
    center      : Horizons observer location code.  Default ``'500@0'`` is
                  the solar system barycenter.
    horizons_id : Override the Horizons numeric ID (useful for bodies not in
                  :data:`HORIZONS_IDS`).
    mu          : Override the gravitational parameter (m³ s⁻²).
    mass        : Alternative to *mu* (kg); ignored if *mu* is given.
    radius      : Override the body radius (meters).
    timeout     : HTTP request timeout in seconds.

    Returns
    -------
    Body
        Position (m) and velocity (m s⁻¹) from Horizons in the ecliptic
        J2000 barycentric frame.  Physical parameters from
        :mod:`keppy.solar_system.bodies`, unless overridden.

    Raises
    ------
    KeyError
        If *name* is not in :data:`HORIZONS_IDS` and *horizons_id* is not
        given.
    HorizonsError
        On any HTTP or parse error from the Horizons API.
    """
    from keppy.solar_system.bodies import BODY_DATA
    from keppy.constants import G

    key = name.lower()

    # Resolve Horizons ID
    if horizons_id is None:
        if key not in HORIZONS_IDS:
            supported = ", ".join(sorted(HORIZONS_IDS))
            raise KeyError(
                f"Unknown body: {name!r}.  Supported names: {supported}.  "
                "Pass horizons_id='<ID>' to query an unlisted body."
            )
        horizons_id = HORIZONS_IDS[key]

    # Query Horizons
    params = {
        "format":      "text",
        "COMMAND":     horizons_id,
        "OBJ_DATA":    "NO",
        "MAKE_EPHEM":  "YES",
        "EPHEM_TYPE":  "VECTORS",
        "CENTER":      center,
        "START_TIME":  epoch,
        "STOP_TIME":   _next_day(epoch),
        "STEP_SIZE":   "1d",
        "OUT_UNITS":   "KM-S",
        "REF_PLANE":   "ECLIPTIC",
        "REF_SYSTEM":  "J2000",
        "VEC_TABLE":   "2",
    }
    text = _get(_HORIZONS_URL, params, timeout)
    position, velocity = _parse_vectors(text)

    # Resolve physical parameters
    bd = BODY_DATA.get(key)

    if mu is not None:
        body_mu = mu
    elif mass is not None:
        body_mu = mass * G
    elif bd is not None:
        body_mu = bd.mu
    else:
        raise HorizonsError(
            f"No physical data for {name!r} in the built-in catalogue.  "
            "Pass mu= or mass= explicitly."
        )

    body_radius = radius if radius is not None else (bd.radius if bd else 0.0)
    display_name = name.title() if name == name.lower() else name

    body = Body.from_mu(display_name, body_mu, position, velocity)
    body.radius = body_radius
    if bd is not None:
        body.j_coefficients = list(bd.j_coefficients)

    return body


def fetch_system(
    names: list[str],
    epoch: str = "2000-01-01",
    *,
    center:                str  = "500@0",
    translate_to_barycenter: bool = False,
    timeout:               int  = 30,
) -> NBodySystem:
    """
    Fetch multiple solar system bodies from JPL Horizons and return a
    ready-to-integrate :class:`~keppy.NBodySystem`.

    Parameters
    ----------
    names                   : List of body names (case-insensitive).
    epoch                   : UTC date ``'YYYY-MM-DD'``.  Defaults to J2000.
    center                  : Horizons observer location.  Default is the
                              solar system barycenter (``'500@0'``), which
                              matches the reference frame of DE440 and
                              should be used for whole-solar-system runs.
    translate_to_barycenter : Shift positions/velocities so the computed
                              system barycenter is at the origin.  Default
                              ``False`` because Horizons vectors with
                              ``center='500@0'`` are already barycentric.
    timeout                 : Per-body HTTP request timeout (seconds).

    Returns
    -------
    NBodySystem

    Raises
    ------
    KeyError, HorizonsError
        Propagated from :func:`fetch_body` for any body in *names*.
    """
    bodies = [
        fetch_body(name, epoch, center=center, timeout=timeout)
        for name in names
    ]
    return NBodySystem(bodies, translate_to_barycenter=translate_to_barycenter)
