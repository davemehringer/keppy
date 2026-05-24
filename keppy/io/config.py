"""
keppy.io.config — TOML-based simulation configuration reader/writer.

Provides dataclasses that describe a complete simulation setup, plus
functions to load from / save to TOML and to construct live objects from
the configuration.

TOML file format
----------------

    [simulation]
    t_end        = 3.156e7    # seconds (1 year)
    output_every = 1
    max_retries  = 10

    [integrator]
    type = "leapfrog"         # rk4 | rk45 | leapfrog | yoshida
    # RK45-only (ignored for other types):
    rtol   = 1e-09
    atol   = 1e-03
    min_dt = 1.0

    [timestep]
    type = "constant"         # constant | acceleration | scaled
    dt   = 86400.0            # initial / fixed step size in seconds

    # AccelerationTimeStepManager
    a_min           = 0.01
    a_max           = 0.10
    increase_factor = 2.0
    decrease_factor = 2.0
    min_dt          = 1.0
    # max_dt = 1e6            # omit for unlimited

    # ScaledTimeStepManager
    q_target      = 0.03
    exponent      = 0.3
    reject_factor = 3.0
    max_factor    = 5.0
    min_factor    = 0.1

    [[bodies]]
    name     = "Sun"
    mu       = 1.327124400180e20     # m³ s⁻²  (use mu OR mass, not both)
    position = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]

    [[bodies]]
    name     = "Earth"
    mass     = 5.972e24              # kg
    position = [1.495978707e11, 0.0, 0.0]
    velocity = [0.0, 29784.69, 0.0]

    # Bodies can also be specified via orbital elements:
    [[bodies]]
    name        = "Mars"
    mass        = 6.39e23
    center_body = "Sun"              # required when using elements
    elements.a    = 2.2794e11        # meters
    elements.e    = 0.0934
    elements.i    = 1.85             # degrees
    elements.node = 49.6
    elements.peri = 286.5
    elements.M    = 19.4

Notes
-----
* Exactly one of ``mass`` or ``mu`` must be given for each body.
* When ``elements`` is present the ``position`` / ``velocity`` keys are
  ignored; the state vectors are derived from the orbital elements relative
  to the named ``center_body``.
* ``center_body`` must appear earlier in the ``[[bodies]]`` list so its
  gravitational parameter is known.
* All angles in ``elements`` are in degrees; ``a`` is in meters.
* TOML does not support NaN or Infinity, so ``max_dt`` is omitted from the
  output when it is ``None`` (unlimited).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from keppy.body import Body
from keppy.nbody_system import NBodySystem
from keppy.orbital import Elements, elements_to_vectors


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BodyConfig:
    """
    Configuration for a single body.

    Attributes
    ----------
    name            : Display name.
    position        : Initial position, shape (3,), meters.
    velocity        : Initial velocity, shape (3,), m s⁻¹.
    mass            : Mass in kg.  Exactly one of ``mass`` / ``mu`` must be set.
    mu              : Gravitational parameter G·m in m³ s⁻².
    radius          : Body radius in meters (for J-perturbations).
    j_coefficients  : Zonal harmonic coefficients [J2, J3, …].
    body_id         : Pre-assigned body ID (``-1`` means auto-assign).
    """
    name:           str
    position:       np.ndarray              # shape (3,), meters
    velocity:       np.ndarray              # shape (3,), m/s
    mass:           float | None = None
    mu:             float | None = None
    radius:         float        = 0.0
    j_coefficients: list[float]  = field(default_factory=list)
    body_id:        int          = -1


@dataclass
class IntegratorConfig:
    """
    Integrator selection and parameters.

    Attributes
    ----------
    type   : ``"rk4"`` | ``"rk45"`` | ``"leapfrog"`` | ``"yoshida"``.
    rtol   : RK45 relative tolerance (ignored for other types).
    atol   : RK45 absolute tolerance (ignored for other types).
    min_dt : RK45 minimum step size (ignored for other types).
    """
    type:   str   = "leapfrog"
    rtol:   float = 1e-9
    atol:   float = 1e-3
    min_dt: float = 1.0


@dataclass
class TimestepConfig:
    """
    Timestep manager selection and parameters.

    Attributes
    ----------
    type            : ``"constant"`` | ``"acceleration"`` | ``"scaled"``.
    dt              : Initial (or fixed) step size in seconds.
    a_min           : ``AccelerationTimeStepManager`` lower threshold.
    a_max           : ``AccelerationTimeStepManager`` upper threshold.
    increase_factor : ``AccelerationTimeStepManager`` growth factor.
    decrease_factor : ``AccelerationTimeStepManager`` shrink factor.
    min_dt          : Hard lower bound on dt (all non-constant managers).
    max_dt          : Hard upper bound on dt (``None`` = unlimited).
    q_target        : ``ScaledTimeStepManager`` target quality metric.
    exponent        : ``ScaledTimeStepManager`` scaling aggressiveness.
    reject_factor   : ``ScaledTimeStepManager`` rejection threshold.
    max_factor      : ``ScaledTimeStepManager`` max growth per step.
    min_factor      : ``ScaledTimeStepManager`` max shrinkage per step.
    """
    type:            str         = "constant"
    dt:              float       = 86400.0
    # AccelerationTimeStepManager
    a_min:           float       = 0.01
    a_max:           float       = 0.10
    increase_factor: float       = 2.0
    decrease_factor: float       = 2.0
    min_dt:          float       = 1.0
    max_dt:          float | None = None
    # ScaledTimeStepManager
    q_target:        float       = 0.03
    exponent:        float       = 0.3
    reject_factor:   float       = 3.0
    max_factor:      float       = 5.0
    min_factor:      float       = 0.1


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.

    Attributes
    ----------
    bodies       : List of body configurations.
    t_end        : End time in seconds.
    integrator   : Integrator configuration.
    timestep     : Timestep manager configuration.
    output_every : Record a snapshot every N accepted steps.
    max_retries  : Maximum consecutive step retries before giving up.
    """
    bodies:       list[BodyConfig]
    t_end:        float
    integrator:   IntegratorConfig = field(default_factory=IntegratorConfig)
    timestep:     TimestepConfig   = field(default_factory=TimestepConfig)
    output_every: int              = 1
    max_retries:  int              = 10


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> SimulationConfig:
    """
    Load a simulation configuration from a TOML file.

    Parameters
    ----------
    path : Path to the ``.toml`` file.

    Returns
    -------
    SimulationConfig

    Raises
    ------
    ValueError
        If a body specifies ``elements`` without a ``center_body``, or if the
        ``center_body`` has not yet been defined earlier in the list.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    # [simulation]
    sim = data.get("simulation", {})
    t_end        = float(sim.get("t_end", 0.0))
    output_every = int(sim.get("output_every", 1))
    max_retries  = int(sim.get("max_retries", 10))

    # [integrator]
    id_ = data.get("integrator", {})
    integrator = IntegratorConfig(
        type   = str(id_.get("type",   "leapfrog")),
        rtol   = float(id_.get("rtol",   1e-9)),
        atol   = float(id_.get("atol",   1e-3)),
        min_dt = float(id_.get("min_dt", 1.0)),
    )

    # [timestep]
    td = data.get("timestep", {})
    timestep = TimestepConfig(
        type            = str(td.get("type",            "constant")),
        dt              = float(td.get("dt",              86400.0)),
        a_min           = float(td.get("a_min",           0.01)),
        a_max           = float(td.get("a_max",           0.10)),
        increase_factor = float(td.get("increase_factor", 2.0)),
        decrease_factor = float(td.get("decrease_factor", 2.0)),
        min_dt          = float(td.get("min_dt",          1.0)),
        max_dt          = float(td["max_dt"]) if "max_dt" in td else None,
        q_target        = float(td.get("q_target",        0.03)),
        exponent        = float(td.get("exponent",        0.3)),
        reject_factor   = float(td.get("reject_factor",   3.0)),
        max_factor      = float(td.get("max_factor",      5.0)),
        min_factor      = float(td.get("min_factor",      0.1)),
    )

    # [[bodies]]
    bodies: list[BodyConfig] = []
    mu_by_name: dict[str, float] = {}   # for center_body element conversions

    for bd in data.get("bodies", []):
        name   = str(bd["name"])
        mass   = float(bd["mass"])   if "mass"   in bd else None
        mu     = float(bd["mu"])     if "mu"     in bd else None
        radius = float(bd.get("radius", 0.0))
        j_coefficients = [float(x) for x in bd.get("j_coefficients", [])]
        body_id = int(bd.get("body_id", -1))

        # Register mu for center_body lookups
        if mu is not None:
            mu_by_name[name] = mu
        elif mass is not None:
            from keppy.constants import G
            mu_by_name[name] = mass * G

        # Resolve position and velocity
        if "elements" in bd:
            el_data     = bd["elements"]
            center_name = bd.get("center_body")
            if center_name is None:
                raise ValueError(
                    f"Body '{name}' has an 'elements' section but no 'center_body'."
                )
            if center_name not in mu_by_name:
                raise ValueError(
                    f"Body '{name}': center_body '{center_name}' not found or has no mu. "
                    "Ensure the center body is defined earlier in the [[bodies]] list."
                )
            el = Elements(
                a    = float(el_data["a"]),
                e    = float(el_data["e"]),
                i    = float(el_data["i"]),
                node = float(el_data.get("node", float("nan"))),
                peri = float(el_data.get("peri", float("nan"))),
                M    = float(el_data.get("M",    float("nan"))),
            )
            position, velocity = elements_to_vectors(mu_by_name[center_name], el)
        else:
            position = np.array([float(x) for x in bd["position"]])
            velocity = np.array([float(x) for x in bd["velocity"]])

        bodies.append(BodyConfig(
            name           = name,
            position       = position,
            velocity       = velocity,
            mass           = mass,
            mu             = mu,
            radius         = radius,
            j_coefficients = j_coefficients,
            body_id        = body_id,
        ))

    return SimulationConfig(
        bodies       = bodies,
        t_end        = t_end,
        integrator   = integrator,
        timestep     = timestep,
        output_every = output_every,
        max_retries  = max_retries,
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_config(config: SimulationConfig, path: str | Path) -> None:
    """
    Save a SimulationConfig to a TOML file.

    State vectors (position and velocity) are always written; orbital element
    notation is an input-only convenience and is not reproduced on output.

    Parameters
    ----------
    config : SimulationConfig to save.
    path   : Destination path (will be overwritten if it exists).
    """
    lines: list[str] = []

    def _f(x: float) -> str:
        """Format a float for TOML (handles scientific notation correctly)."""
        return repr(float(x))

    def _arr(arr) -> str:
        """Format a sequence of floats as a TOML array."""
        return "[" + ", ".join(_f(x) for x in arr) + "]"

    # [simulation]
    lines += [
        "[simulation]",
        f"t_end        = {_f(config.t_end)}",
        f"output_every = {config.output_every}",
        f"max_retries  = {config.max_retries}",
        "",
    ]

    # [integrator]
    ic = config.integrator
    lines += [
        "[integrator]",
        f'type   = "{ic.type}"',
        f"rtol   = {_f(ic.rtol)}",
        f"atol   = {_f(ic.atol)}",
        f"min_dt = {_f(ic.min_dt)}",
        "",
    ]

    # [timestep]
    tc = config.timestep
    lines += [
        "[timestep]",
        f'type            = "{tc.type}"',
        f"dt              = {_f(tc.dt)}",
        f"a_min           = {_f(tc.a_min)}",
        f"a_max           = {_f(tc.a_max)}",
        f"increase_factor = {_f(tc.increase_factor)}",
        f"decrease_factor = {_f(tc.decrease_factor)}",
        f"min_dt          = {_f(tc.min_dt)}",
        f"q_target        = {_f(tc.q_target)}",
        f"exponent        = {_f(tc.exponent)}",
        f"reject_factor   = {_f(tc.reject_factor)}",
        f"max_factor      = {_f(tc.max_factor)}",
        f"min_factor      = {_f(tc.min_factor)}",
    ]
    if tc.max_dt is not None:
        lines.append(f"max_dt          = {_f(tc.max_dt)}")
    lines.append("")

    # [[bodies]]
    for bc in config.bodies:
        lines.append("[[bodies]]")
        lines.append(f'name     = "{bc.name}"')
        if bc.mu is not None:
            lines.append(f"mu       = {_f(bc.mu)}")
        elif bc.mass is not None:
            lines.append(f"mass     = {_f(bc.mass)}")
        if bc.radius:
            lines.append(f"radius   = {_f(bc.radius)}")
        if bc.j_coefficients:
            lines.append(f"j_coefficients = {_arr(bc.j_coefficients)}")
        if bc.body_id >= 0:
            lines.append(f"body_id  = {bc.body_id}")
        lines.append(f"position = {_arr(bc.position)}")
        lines.append(f"velocity = {_arr(bc.velocity)}")
        lines.append("")

    Path(path).write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def config_to_system(
    config: SimulationConfig,
    *,
    translate_to_barycenter: bool = True,
) -> NBodySystem:
    """
    Construct an :class:`~keppy.NBodySystem` from a :class:`SimulationConfig`.

    Parameters
    ----------
    config                   : SimulationConfig to use.
    translate_to_barycenter  : If ``True`` (default), shift positions and
                               velocities so the system barycenter is at the
                               origin with zero net momentum.

    Raises
    ------
    ValueError
        If a body config has neither ``mass`` nor ``mu``.
    """
    bodies: list[Body] = []
    for bc in config.bodies:
        if bc.mu is not None:
            body = Body.from_mu(bc.name, bc.mu, bc.position.copy(), bc.velocity.copy())
        elif bc.mass is not None:
            body = Body.from_mass(bc.name, bc.mass, bc.position.copy(), bc.velocity.copy())
        else:
            raise ValueError(
                f"Body '{bc.name}' must specify either 'mass' or 'mu'."
            )
        body.radius          = bc.radius
        body.j_coefficients  = list(bc.j_coefficients)
        if bc.body_id >= 0:
            body.body_id = bc.body_id
        bodies.append(body)
    return NBodySystem(bodies, translate_to_barycenter=translate_to_barycenter)


def build_integrator(config: SimulationConfig):
    """
    Instantiate the integrator described by *config.integrator*.

    Returns
    -------
    An object satisfying the :class:`~keppy.integrator.Integrator` protocol.

    Raises
    ------
    ValueError
        If the integrator type string is not recognised.
    """
    from keppy.acceleration import PairwiseAccelerationCalculator
    from keppy.integrator import (
        RK4Integrator, RK45Integrator, LeapfrogIntegrator, YoshidaIntegrator,
    )

    calc = PairwiseAccelerationCalculator()
    ic   = config.integrator
    match ic.type:
        case "rk4":
            return RK4Integrator(calc)
        case "rk45":
            return RK45Integrator(calc, rtol=ic.rtol, atol=ic.atol, min_dt=ic.min_dt)
        case "leapfrog":
            return LeapfrogIntegrator(calc)
        case "yoshida":
            return YoshidaIntegrator(calc)
        case _:
            raise ValueError(
                f"Unknown integrator type: {ic.type!r}. "
                "Expected one of: rk4, rk45, leapfrog, yoshida."
            )


def build_timestep_manager(config: SimulationConfig):
    """
    Instantiate the timestep manager described by *config.timestep*.

    Returns
    -------
    An object satisfying the :class:`~keppy.timestep.TimeStepManager` protocol.

    Raises
    ------
    ValueError
        If the timestep manager type string is not recognised.
    """
    from keppy.timestep import (
        ConstantTimeStepManager,
        AccelerationTimeStepManager,
        ScaledTimeStepManager,
    )

    tc = config.timestep
    match tc.type:
        case "constant":
            return ConstantTimeStepManager(tc.dt)
        case "acceleration":
            return AccelerationTimeStepManager(
                dt_init         = tc.dt,
                a_min           = tc.a_min,
                a_max           = tc.a_max,
                increase_factor = tc.increase_factor,
                decrease_factor = tc.decrease_factor,
                min_dt          = tc.min_dt,
                max_dt          = tc.max_dt,
            )
        case "scaled":
            return ScaledTimeStepManager(
                dt_init       = tc.dt,
                q_target      = tc.q_target,
                exponent      = tc.exponent,
                reject_factor = tc.reject_factor,
                min_dt        = tc.min_dt,
                max_dt        = tc.max_dt,
                max_factor    = tc.max_factor,
                min_factor    = tc.min_factor,
            )
        case _:
            raise ValueError(
                f"Unknown timestep manager type: {tc.type!r}. "
                "Expected one of: constant, acceleration, scaled."
            )
