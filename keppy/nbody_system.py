"""
NBodySystem — manages a collection of gravitating bodies.

Improvements over the C++ keplerpp NBodySystem:
  * Bodies are stored as a plain list; numpy is used for vectorised
    energy and momentum calculations instead of manual loops.
  * The barycenter translation is an explicit, named method rather than
    hidden inside the constructor — callers can choose when (or whether)
    to apply it.
  * total_energy(), angular_momentum(), and linear_momentum() are
    @property accessors that recompute on demand, so callers never hold
    a stale cached value.
  * time is stored in seconds internally, but helper properties expose
    it in days and years for convenience.
  * Body IDs are assigned automatically when a body without one is added.
  * Step delegation: NBodySystem owns the time counter and calls an
    external Integrator/AccelerationCalculator (to be implemented in a
    later step), keeping physics and bookkeeping separate.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from keppy.body import Body
from keppy.constants import DAY, YEAR, G


class NBodySystem:
    """
    A gravitationally interacting system of bodies.

    Parameters
    ----------
    bodies    : Initial collection of Body objects.
    time      : Start time in seconds (default 0).
    translate_to_barycenter : If True (default), shift all positions so
                              the system's centre of mass is at the origin,
                              and adjust velocities to zero the bulk momentum.

    Notes
    -----
    After construction the bodies are stored internally and can be
    accessed via the ``bodies`` property.  The integrator calls
    ``step(dt)`` which delegates to whatever AccelerationCalculator and
    Integrator are attached.
    """

    def __init__(
        self,
        bodies: Sequence[Body],
        time: float = 0.0,
        translate_to_barycenter: bool = True,
    ) -> None:
        self._bodies: list[Body] = []
        self._time: float = float(time)  # seconds

        for body in bodies:
            self.add_body(body)

        if translate_to_barycenter:
            self.translate_to_barycenter()

    # ------------------------------------------------------------------
    # Body management
    # ------------------------------------------------------------------

    def add_body(self, body: Body) -> None:
        """Add a body to the system, assigning an ID if needed."""
        if body.body_id == -1:
            body.body_id = len(self._bodies)
        elif any(b.body_id == body.body_id for b in self._bodies):
            raise ValueError(
                f"A body with id={body.body_id} already exists in the system."
            )
        self._bodies.append(body)

    def remove_body(self, body_id: int) -> Body:
        """Remove and return the body with the given id."""
        for i, b in enumerate(self._bodies):
            if b.body_id == body_id:
                return self._bodies.pop(i)
        raise KeyError(f"No body with id={body_id} in the system.")

    def get_body(self, body_id: int) -> Body:
        """Return the body with the given id."""
        for b in self._bodies:
            if b.body_id == body_id:
                return b
        raise KeyError(f"No body with id={body_id} in the system.")

    def get_body_by_name(self, name: str) -> Body:
        """Return the first body whose name matches (case-sensitive)."""
        for b in self._bodies:
            if b.name == name:
                return b
        raise KeyError(f"No body named {name!r} in the system.")

    @property
    def bodies(self) -> list[Body]:
        return self._bodies

    @property
    def n(self) -> int:
        """Number of bodies."""
        return len(self._bodies)

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        self._time = float(value)

    @property
    def time_days(self) -> float:
        """Current simulation time in Julian days."""
        return self._time / DAY

    @property
    def time_years(self) -> float:
        """Current simulation time in Julian years."""
        return self._time / YEAR

    # ------------------------------------------------------------------
    # Centre-of-mass helpers
    # ------------------------------------------------------------------

    @property
    def masses(self) -> np.ndarray:
        """1-D array of body masses (kg), ordered as stored."""
        return np.array([b.mass for b in self._bodies])

    @property
    def total_mass(self) -> float:
        """Total system mass in kg."""
        return float(self.masses.sum())

    def barycenter(self) -> np.ndarray:
        """
        Return the position of the system's centre of mass (m).

        Returns
        -------
        numpy.ndarray, shape (3,)
        """
        masses = self.masses
        positions = np.stack([b.position for b in self._bodies])  # (n, 3)
        return (masses[:, None] * positions).sum(axis=0) / masses.sum()

    def barycenter_velocity(self) -> np.ndarray:
        """
        Return the velocity of the system's centre of mass (m s⁻¹).

        A non-zero value means the system has bulk linear drift.

        Returns
        -------
        numpy.ndarray, shape (3,)
        """
        masses = self.masses
        velocities = np.stack([b.velocity for b in self._bodies])  # (n, 3)
        return (masses[:, None] * velocities).sum(axis=0) / masses.sum()

    def translate_to_barycenter(self) -> None:
        """
        Shift all positions so the centre of mass is at the origin and
        subtract any bulk velocity so the total linear momentum is zero.

        This is idempotent: calling it multiple times has no extra effect
        (up to floating-point rounding).
        """
        com = self.barycenter()
        com_vel = self.barycenter_velocity()
        for b in self._bodies:
            b.position -= com
            b.velocity -= com_vel

    # ------------------------------------------------------------------
    # Conserved quantities
    # ------------------------------------------------------------------

    @property
    def kinetic_energy(self) -> float:
        """
        Total kinetic energy of the system (J).

        KE = Σ ½ mᵢ |vᵢ|²
        """
        masses = self.masses
        v2 = np.array([np.dot(b.velocity, b.velocity) for b in self._bodies])
        return float(0.5 * (masses * v2).sum())

    @property
    def potential_energy(self) -> float:
        """
        Total gravitational potential energy of the system (J).

        PE = -Σᵢ<ⱼ G mᵢ mⱼ / rᵢⱼ
        """
        bodies = self._bodies
        pe = 0.0
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                r = np.linalg.norm(bodies[i].position - bodies[j].position)
                pe -= G * bodies[i].mass * bodies[j].mass / r
        return pe

    @property
    def total_energy(self) -> float:
        """Total mechanical energy KE + PE (J)."""
        return self.kinetic_energy + self.potential_energy

    @property
    def angular_momentum(self) -> np.ndarray:
        """
        Total angular momentum vector of the system (kg m² s⁻¹).

        L = Σ mᵢ (xᵢ × vᵢ)

        Returns
        -------
        numpy.ndarray, shape (3,)
        """
        L = np.zeros(3)
        for b in self._bodies:
            L += b.mass * np.cross(b.position, b.velocity)
        return L

    @property
    def linear_momentum(self) -> np.ndarray:
        """
        Total linear momentum vector of the system (kg m s⁻¹).

        p = Σ mᵢ vᵢ

        Returns
        -------
        numpy.ndarray, shape (3,)
        """
        masses = self.masses
        velocities = np.stack([b.velocity for b in self._bodies])
        return (masses[:, None] * velocities).sum(axis=0)

    # ------------------------------------------------------------------
    # State vector I/O  (mirrors C++ getXV / setXV)
    # ------------------------------------------------------------------

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return positions and velocities as stacked arrays.

        Returns
        -------
        positions  : shape (n, 3)  metres
        velocities : shape (n, 3)  m s⁻¹
        """
        positions = np.stack([b.position for b in self._bodies])
        velocities = np.stack([b.velocity for b in self._bodies])
        return positions, velocities

    def set_state(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """
        Overwrite all body positions and velocities from stacked arrays.

        Parameters
        ----------
        positions  : shape (n, 3)  metres
        velocities : shape (n, 3)  m s⁻¹
        """
        if positions.shape != (self.n, 3) or velocities.shape != (self.n, 3):
            raise ValueError(
                f"Expected arrays of shape ({self.n}, 3), "
                f"got {positions.shape} and {velocities.shape}."
            )
        for i, b in enumerate(self._bodies):
            b.position[:] = positions[i]
            b.velocity[:] = velocities[i]

    # ------------------------------------------------------------------
    # Simulation step  (integrator/accelerator are injected later)
    # ------------------------------------------------------------------

    def step(self, dt: float, integrator) -> None:
        """
        Advance the simulation by *dt* seconds using *integrator*.

        Parameters
        ----------
        dt         : Time step in seconds.
        integrator : An object with a ``step(system, dt)`` method
                     (to be provided by the Integrator module).
        """
        integrator.step(self, dt)
        self._time += dt

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NBodySystem(n={self.n}, "
            f"t={self.time_days:.4f} days, "
            f"E={self.total_energy:.6e} J)"
        )

    def __str__(self) -> str:
        lines = [
            "NBodySystem",
            f"  n_bodies : {self.n}",
            f"  time     : {self.time_days:.6f} days  "
            f"({self.time_years:.6f} yr)",
            f"  E_total  : {self.total_energy:.6e} J",
            f"  |L|      : {np.linalg.norm(self.angular_momentum):.6e} kg m²/s",
            f"  |p|      : {np.linalg.norm(self.linear_momentum):.6e} kg m/s",
            "",
            "  Bodies:",
        ]
        for b in self._bodies:
            lines.append(f"    [{b.body_id}] {b.name}  mass={b.mass:.4e} kg  "
                         f"pos={b.position}  vel={b.velocity}")
        return "\n".join(lines)
