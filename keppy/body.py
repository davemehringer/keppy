"""
Body — a single gravitating particle in an N-body simulation.

Improvements over the C++ keplerpp Body struct:
  * Uses numpy arrays for all vectors (position, velocity, acceleration).
    This avoids custom vector arithmetic and enables vectorised NumPy ops
    throughout the rest of the library.
  * Python Enum for Frame and Origin instead of plain C-style enums.
  * mass is a first-class attribute; mu is derived from it (or vice-versa)
    through a property pair so you can set whichever is more convenient.
  * j_coefficients is a plain list starting at index 0 = J2, 1 = J3, ...
    (the C++ code used raw pointers; here we keep it safe and iterable).
  * center_body is typed as Optional['Body'] — no raw pointer needed.
  * __repr__ / __str__ give readable output without a separate operator<<.
  * Frozen=False so the integrator can mutate position/velocity in-place.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from keppy.constants import G


# ---------------------------------------------------------------------------
# Reference frame and origin enumerations
# ---------------------------------------------------------------------------

class Frame(Enum):
    """Reference frame for position and velocity vectors."""
    ECLIPTIC = auto()
    EQUATORIAL = auto()          # body-equatorial frame (was ORIGIN_BDOY_EQUATORIAL)


class Origin(Enum):
    """Coordinate origin for position and velocity vectors."""
    CENTER_BODY = auto()         # relative to the body being orbited
    SYSTEM_BARYCENTER = auto()   # relative to the system's centre of mass


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------

@dataclass
class Body:
    """
    A single gravitating body.

    Units
    -----
    All distances    : metres  (m)
    All velocities   : m s⁻¹
    All accelerations: m s⁻²
    mass             : kg
    radius           : m
    mu               : m³ s⁻²  (= G * mass; set *either* mass or mu,
                                 the other is derived automatically)
    j_coefficients   : dimensionless  (J2 at index 0, J3 at index 1, …)

    Parameters
    ----------
    name        : Human-readable label.
    position    : 3-vector, metres.
    velocity    : 3-vector, m s⁻¹.
    mass        : Body mass in kg.  Pass exactly one of mass / mu.
    mu          : Gravitational parameter G·m in m³ s⁻².
    radius      : Mean equatorial radius in metres (0 if unknown).
    j_coefficients : Zonal harmonic coefficients [J2, J3, …].
    center_body : Body this body orbits (None for the primary / barycentre).
    frame       : Reference frame of the position/velocity vectors.
    origin      : Coordinate origin of the position/velocity vectors.
    rotation_matrix : 3×3 body-frame rotation matrix (or None).
    body_id     : Integer identifier (assigned automatically if omitted).
    """

    name: str
    position: np.ndarray                        # shape (3,), float64
    velocity: np.ndarray                        # shape (3,), float64

    # Exactly one of mass / _mu should be supplied at construction time.
    # Use the mass and mu properties after construction.
    _mass: float = field(default=0.0, repr=False)
    _mu: float = field(default=0.0, repr=False)

    radius: float = 0.0                         # m
    j_coefficients: list[float] = field(default_factory=list)
    center_body: Optional[Body] = field(default=None, repr=False)
    frame: Frame = Frame.ECLIPTIC
    origin: Origin = Origin.SYSTEM_BARYCENTER
    rotation_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    body_id: int = -1

    # Acceleration is computed by the integrator — not set by the user.
    acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float),
        repr=False,
    )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_mass(
        cls,
        name: str,
        mass: float,
        position: np.ndarray,
        velocity: np.ndarray,
        **kwargs,
    ) -> "Body":
        """Create a Body from a mass in kg."""
        b = cls(name=name, position=np.asarray(position, dtype=float),
                velocity=np.asarray(velocity, dtype=float), **kwargs)
        b._mass = float(mass)
        b._mu = G * mass
        return b

    @classmethod
    def from_mu(
        cls,
        name: str,
        mu: float,
        position: np.ndarray,
        velocity: np.ndarray,
        **kwargs,
    ) -> "Body":
        """Create a Body from a gravitational parameter mu = G·m (m³ s⁻²)."""
        b = cls(name=name, position=np.asarray(position, dtype=float),
                velocity=np.asarray(velocity, dtype=float), **kwargs)
        b._mu = float(mu)
        b._mass = mu / G
        return b

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mass(self) -> float:
        """Body mass in kg."""
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        self._mass = float(value)
        self._mu = G * value

    @property
    def mu(self) -> float:
        """Gravitational parameter G·m in m³ s⁻²."""
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        self._mu = float(value)
        self._mass = value / G

    @property
    def speed(self) -> float:
        """Magnitude of the velocity vector (m s⁻¹)."""
        return float(np.linalg.norm(self.velocity))

    @property
    def kinetic_energy(self) -> float:
        """Specific kinetic energy ½v² (J kg⁻¹)."""
        return 0.5 * float(np.dot(self.velocity, self.velocity))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Ensure vectors are proper float64 numpy arrays.
        object.__setattr__(self, "position",
                           np.asarray(self.position, dtype=float))
        object.__setattr__(self, "velocity",
                           np.asarray(self.velocity, dtype=float))
        if self.position.shape != (3,):
            raise ValueError(f"position must be a 3-vector, got {self.position.shape}")
        if self.velocity.shape != (3,):
            raise ValueError(f"velocity must be a 3-vector, got {self.velocity.shape}")

    def __repr__(self) -> str:
        return (
            f"Body(name={self.name!r}, id={self.body_id}, "
            f"mass={self._mass:.4e} kg, "
            f"pos={self.position}, vel={self.velocity})"
        )

    def __str__(self) -> str:
        lines = [
            f"Body : {self.name}  (id={self.body_id})",
            f"  mass   = {self._mass:.6e} kg",
            f"  mu     = {self._mu:.6e} m³/s²",
            f"  radius = {self.radius:.4e} m",
            f"  frame  = {self.frame.name},  origin = {self.origin.name}",
        ]
        if self.center_body is not None:
            lines.append(f"  orbits = {self.center_body.name}")
        if self.j_coefficients:
            jstr = ", ".join(
                f"J{i+2}={v:.4e}" for i, v in enumerate(self.j_coefficients)
            )
            lines.append(f"  J-coeff: {jstr}")
        lines += [
            f"  pos (m)    : {self.position}",
            f"  vel (m/s)  : {self.velocity}",
            f"  acc (m/s²) : {self.acceleration}",
        ]
        return "\n".join(lines)

    def copy(self) -> "Body":
        """Return a deep copy of this body (useful for state snapshots)."""
        return copy.deepcopy(self)
