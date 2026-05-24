"""
keppy — command-line interface.

Sub-commands
------------
keppy list-bodies
    Print the names of all built-in solar system bodies.

keppy solar [options]
    Simulate a solar system configuration from named bodies, using
    built-in approximate initial conditions or live JPL Horizons data.

keppy run CONFIG [options]
    Load and execute a simulation from a TOML configuration file.

Duration / time-step format
---------------------------
All ``--duration`` and ``--dt`` arguments accept:

    1y      → one Julian year  (365.25 × 86 400 s)
    365d    → 365 days
    24h     → 24 hours
    3600s   → 3 600 seconds (explicit suffix)
    3600    → bare number, treated as seconds
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np

from keppy.constants import AU, MU_SUN
from keppy.nbody_system import NBodySystem
from keppy.acceleration import PairwiseAccelerationCalculator
from keppy.integrator import (
    RK4Integrator, RK45Integrator, LeapfrogIntegrator, YoshidaIntegrator,
)
from keppy.timestep import (
    ConstantTimeStepManager,
    AccelerationTimeStepManager,
    ScaledTimeStepManager,
    run as _run,
)
from keppy.solar_system import make_body, BODY_DATA
from keppy import io

_YEAR = 365.25 * 86_400.0   # seconds per Julian year


# ---------------------------------------------------------------------------
# Duration / time-step parser
# ---------------------------------------------------------------------------

def parse_duration(s: str) -> float:
    """Parse a duration string to seconds.

    Accepted formats
    ----------------
    ``Ny``  — N Julian years (365.25 days each)
    ``Nd``  — N days
    ``Nh``  — N hours
    ``Ns``  — N seconds (explicit suffix)
    ``N``   — N seconds (bare number)

    Parameters
    ----------
    s : Duration string, e.g. ``"1y"``, ``"30d"``, ``"86400"``.

    Returns
    -------
    float
        Number of seconds.

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    s = s.strip()
    try:
        if s.endswith('y'):
            return float(s[:-1]) * _YEAR
        if s.endswith('d'):
            return float(s[:-1]) * 86_400.0
        if s.endswith('h'):
            return float(s[:-1]) * 3_600.0
        if s.endswith('s'):
            return float(s[:-1])
        return float(s)
    except ValueError:
        raise ValueError(
            f"Cannot parse duration {s!r}. "
            "Use a number optionally followed by y/d/h/s."
        )


parse_timestep = parse_duration   # same grammar, separate name for clarity


# ---------------------------------------------------------------------------
# Known body names and built-in initial conditions
# ---------------------------------------------------------------------------

#: Body names accepted by ``keppy solar`` without ``--horizons``.
KNOWN_BODIES: list[str] = [
    "sun", "mercury", "venus", "earth", "moon",
    "mars", "jupiter", "saturn", "uranus", "neptune",
    "pluto", "ceres",
]

# Mean semi-major axes in AU (heliocentric, rounded to 3 dp).
_SEMI_MAJOR_AU: dict[str, float] = {
    "mercury": 0.387,
    "venus":   0.723,
    "earth":   1.000,
    "mars":    1.524,
    "jupiter": 5.203,
    "saturn":  9.537,
    "uranus":  19.19,
    "neptune": 30.07,
    "pluto":   39.48,
    "ceres":   2.767,
}

_MOON_DIST_M: float = 384_400_000.0   # mean Earth-Moon distance (m)


def _make_solar_body(name: str, earth_pos: np.ndarray | None = None) -> "Body":  # type: ignore[name-defined]
    """Return a Body with approximate circular-orbit initial conditions.

    All bodies (except the Sun) start on a circular orbit in the x-y plane.
    The Moon is placed relative to *earth_pos* (defaults to (AU, 0, 0)).

    Parameters
    ----------
    name      : Body name (case-insensitive).
    earth_pos : Current Earth position, used for the Moon.

    Returns
    -------
    Body

    Raises
    ------
    ValueError
        If *name* is not in :data:`KNOWN_BODIES`.
    """
    n = name.lower()

    if n == "sun":
        return make_body("sun", position=np.zeros(3), velocity=np.zeros(3))

    if n == "moon":
        ep = earth_pos if earth_pos is not None else np.array([AU, 0.0, 0.0])
        # Displace Moon in +y from Earth
        pos = ep + np.array([0.0, _MOON_DIST_M, 0.0])
        # Earth's heliocentric circular speed
        v_earth = np.sqrt(MU_SUN / np.linalg.norm(ep))
        # Moon's circular speed around Earth (radius vector is +y → vel is −x)
        v_moon = np.sqrt(BODY_DATA["earth"].mu / _MOON_DIST_M)
        vel = np.array([-v_moon, v_earth, 0.0])
        return make_body("moon", position=pos, velocity=vel)

    a_au = _SEMI_MAJOR_AU.get(n)
    if a_au is None:
        raise ValueError(
            f"No built-in orbital data for {name!r}. "
            "Run 'keppy list-bodies' to see supported names, "
            "or add --horizons to fetch from JPL Horizons."
        )
    r = a_au * AU
    pos = np.array([r, 0.0, 0.0])
    vel = np.array([0.0, np.sqrt(MU_SUN / r), 0.0])
    return make_body(n, position=pos, velocity=vel)


def _build_solar_system(
    names: list[str],
    *,
    use_horizons: bool,
    epoch: str,
) -> NBodySystem:
    """Build an NBodySystem from a list of body names.

    Parameters
    ----------
    names        : Body names (case-insensitive).
    use_horizons : Fetch state vectors from JPL Horizons when ``True``.
    epoch        : Epoch string ``"YYYY-MM-DD"`` for Horizons queries.

    Returns
    -------
    NBodySystem
    """
    if use_horizons:
        from keppy.solar_system.horizons import fetch_system
        return fetch_system(names, epoch=epoch)

    bodies = []
    earth_pos: np.ndarray | None = None
    for name in names:
        body = _make_solar_body(name, earth_pos=earth_pos)
        if name.lower() == "earth":
            earth_pos = body.position.copy()
        bodies.append(body)
    return NBodySystem(bodies)


# ---------------------------------------------------------------------------
# Integrator factory
# ---------------------------------------------------------------------------

def _make_integrator(name: str):
    """Instantiate an integrator by short name."""
    calc = PairwiseAccelerationCalculator()
    match name:
        case "rk4":
            return RK4Integrator(calc)
        case "rk45":
            return RK45Integrator(calc)
        case "leapfrog":
            return LeapfrogIntegrator(calc)
        case "yoshida":
            return YoshidaIntegrator(calc)
        case _:
            raise ValueError(f"Unknown integrator: {name!r}")


# ---------------------------------------------------------------------------
# Post-run output helpers
# ---------------------------------------------------------------------------

def _print_summary(records: list, body_names: list[str]) -> None:
    """Print a brief simulation result summary to stdout."""
    if not records:
        print("No records produced.")
        return

    n_steps       = len(records)
    t_start       = records[0].time
    t_end         = records[-1].time
    duration_days = (t_end - t_start) / 86_400.0

    print(f"\n{'─' * 54}")
    print(f"  Bodies     : {', '.join(body_names)}")
    print(f"  Steps      : {n_steps}")
    print(f"  Duration   : {duration_days:.2f} days")

    # Best-effort energy drift
    try:
        from keppy.viz.plot import _compute_total_energy
        mus_list = [
            BODY_DATA[n.lower()].mu
            for n in body_names
            if n.lower() in BODY_DATA
        ]
        if len(mus_list) == len(body_names):
            energies = _compute_total_energy(records, np.array(mus_list))
            e0 = energies[0]
            if abs(e0) > 0:
                drift = abs((energies[-1] - e0) / e0)
                print(f"  Energy drift: {drift:.2e}  (|ΔE/E₀|)")
    except Exception:
        pass   # diagnostic is best-effort; never block output

    print(f"{'─' * 54}\n")


def _handle_save(records: list, path: str) -> None:
    """Save trajectory to *path* and report."""
    io.save_trajectory(records, path)
    print(f"Trajectory saved → {path}")


def _handle_plot(
    records: list,
    body_names: list[str],
    *,
    unit: str,
    plane: str,
    trail_length: int | None,
) -> None:
    """Open an interactive OrbitViewer."""
    from keppy.viz.interactive import OrbitViewer
    import matplotlib.pyplot as plt

    viewer = OrbitViewer(
        records,
        body_names=body_names,
        unit=unit,
        plane=plane,
        trail_length=trail_length,
    )
    _anim = viewer.animate(interval=40)   # noqa: F841 — must stay in scope
    plt.show()


def _handle_plot_energy(records: list, body_names: list[str]) -> None:
    """Show the energy conservation diagnostic plot."""
    from keppy.viz.plot import plot_energy
    import matplotlib.pyplot as plt

    mus_list = [
        BODY_DATA[n.lower()].mu
        for n in body_names
        if n.lower() in BODY_DATA
    ]
    if len(mus_list) != len(body_names):
        print(
            "Warning: cannot compute energy — some bodies are not in "
            "BODY_DATA (try only standard solar system bodies).",
            file=sys.stderr,
        )
        return

    plot_energy(records, np.array(mus_list), relative=True)
    plt.show()


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_list_bodies(args) -> int:  # noqa: ARG001
    """Print all built-in solar system body names."""
    print("Built-in solar system bodies (usable with 'keppy solar --bodies'):")
    for name in KNOWN_BODIES:
        print(f"  {name}")
    return 0


def cmd_solar(args) -> int:
    """Run a solar system simulation from named bodies."""
    # Parse body names (accept comma-separated or space-separated)
    raw   = args.bodies or "sun,earth"
    names = [n.strip() for n in raw.replace(",", " ").split() if n.strip()]
    if not names:
        print("Error: --bodies must list at least one body.", file=sys.stderr)
        return 1

    # Validate against known names unless Horizons is handling them
    if not args.horizons:
        bad = [n for n in names if n.lower() not in KNOWN_BODIES]
        if bad:
            print(
                f"Error: unknown body name(s): {bad!r}. "
                "Run 'keppy list-bodies' for valid names, "
                "or add --horizons.",
                file=sys.stderr,
            )
            return 1

    # Build the N-body system
    try:
        system = _build_solar_system(
            names, use_horizons=args.horizons, epoch=args.epoch
        )
    except Exception as exc:
        print(f"Error building system: {exc}", file=sys.stderr)
        return 1

    # Parse duration and timestep
    try:
        t_end = parse_duration(args.duration)
        dt    = parse_timestep(args.dt)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Build integrator
    integrator = _make_integrator(args.integrator)

    # Build timestep manager
    adaptive = getattr(args, "adaptive", "none")
    if adaptive == "acceleration":
        tsm = AccelerationTimeStepManager(dt, a_min=0.01, a_max=0.1)
    elif adaptive == "scaled":
        tsm = ScaledTimeStepManager(dt)
    else:
        tsm = ConstantTimeStepManager(dt)

    # Run
    print(
        f"Simulating {', '.join(names)} "
        f"for {args.duration} "
        f"using {args.integrator} integrator …"
    )
    try:
        records = _run(
            system, integrator, tsm, t_end,
            output_every=args.output_every,
        )
    except Exception as exc:
        print(f"Simulation error: {exc}", file=sys.stderr)
        return 1

    _print_summary(records, names)

    if args.save:
        _handle_save(records, args.save)

    if args.plot:
        _handle_plot(
            records, names,
            unit=args.unit,
            plane=args.plane,
            trail_length=args.trail_length,
        )

    if args.plot_energy:
        _handle_plot_energy(records, names)

    return 0


_EXAMPLE_CONFIG = """\
# keppy simulation configuration — TOML format
# Run with:  keppy run my_sim.toml [--plot] [--save results.npz]

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
[simulation]
t_end        = 3.156e7    # end time in seconds (≈ 1 Julian year)
output_every = 1          # record a snapshot every N accepted steps
max_retries  = 10         # max consecutive step retries for adaptive methods

# ---------------------------------------------------------------------------
# Integration algorithm
# ---------------------------------------------------------------------------
[integrator]
type = "leapfrog"         # rk4 | rk45 | leapfrog | yoshida

# RK45-only options (ignored for other integrators):
# rtol   = 1e-09          # relative tolerance
# atol   = 1e-03          # absolute tolerance
# min_dt = 1.0            # minimum step size in seconds

# ---------------------------------------------------------------------------
# Time-step management
# ---------------------------------------------------------------------------
[timestep]
type = "constant"         # constant | acceleration | scaled
dt   = 86400.0            # initial / fixed step size in seconds (1 day)

# AccelerationTimeStepManager options (type = "acceleration"):
# a_min           = 0.01  # fractional acceleration change → increase step
# a_max           = 0.10  # fractional acceleration change → decrease step
# increase_factor = 2.0
# decrease_factor = 2.0
# min_dt          = 1.0
# max_dt          = 1e6   # omit for no upper limit

# ScaledTimeStepManager options (type = "scaled"):
# q_target      = 0.03    # target RMS fractional acceleration change
# exponent      = 0.3     # scaling aggressiveness (0 = no scaling)
# reject_factor = 3.0     # reject step if q > reject_factor * q_target
# max_factor    = 5.0     # maximum growth per step
# min_factor    = 0.1     # maximum shrinkage per step

# ---------------------------------------------------------------------------
# Bodies  (repeat [[bodies]] once per body)
# ---------------------------------------------------------------------------
[[bodies]]
name     = "Sun"
mu       = 1.327124400180e20   # m³ s⁻²  (use mu OR mass, not both)
position = [0.0, 0.0, 0.0]    # meters
velocity = [0.0, 0.0, 0.0]    # m s⁻¹

[[bodies]]
name     = "Earth"
mass     = 5.972e24            # kg
position = [1.495978707e11, 0.0, 0.0]
velocity = [0.0, 29784.69, 0.0]

# Bodies can also be specified via Keplerian elements
# (elements are converted to state vectors at load time):
# [[bodies]]
# name        = "Mars"
# mass        = 6.39e23
# center_body = "Sun"          # must appear earlier in the [[bodies]] list
# elements.a    = 2.2794e11    # semi-major axis in meters
# elements.e    = 0.0934       # eccentricity
# elements.i    = 1.85         # inclination in degrees
# elements.node = 49.6         # longitude of ascending node, degrees
# elements.peri = 286.5        # argument of periapsis, degrees
# elements.M    = 19.4         # mean anomaly at epoch, degrees
"""


def cmd_show_config(args) -> int:  # noqa: ARG001
    """Print an annotated example TOML config to stdout."""
    print(_EXAMPLE_CONFIG.rstrip())
    return 0


def cmd_run(args) -> int:
    """Load and execute a TOML simulation config file."""
    # --show-config prints the template and exits without needing a file
    if getattr(args, "show_config", False):
        return cmd_show_config(args)

    if not args.config:
        print(
            "Error: CONFIG argument is required unless --show-config is given.\n"
            "       Use 'keppy run --show-config' to print an example config.",
            file=sys.stderr,
        )
        return 1

    try:
        config = io.load_config(args.config)
    except FileNotFoundError:
        print(f"Error: config file not found: {args.config!r}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error loading {args.config!r}: {exc}", file=sys.stderr)
        return 1

    try:
        system     = io.config_to_system(config)
        integrator = io.build_integrator(config)
        tsm        = io.build_timestep_manager(config)
    except Exception as exc:
        print(f"Error building simulation: {exc}", file=sys.stderr)
        return 1

    print(f"Running simulation from {args.config!r} …")
    try:
        records = _run(
            system, integrator, tsm, config.t_end,
            output_every=config.output_every,
        )
    except Exception as exc:
        print(f"Simulation error: {exc}", file=sys.stderr)
        return 1

    body_names = [bc.name for bc in config.bodies]
    _print_summary(records, body_names)

    if args.save:
        _handle_save(records, args.save)

    if args.plot:
        _handle_plot(records, body_names, unit="au", plane="xy", trail_length=None)

    if args.plot_energy:
        _handle_plot_energy(records, body_names)

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="keppy",
        description=(
            "keppy — Python N-body orbital mechanics simulator.\n\n"
            "Sub-commands\n"
            "  list-bodies   Print built-in solar system body names.\n"
            "  solar         Simulate named solar system bodies.\n"
            "  run           Execute a TOML simulation config file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- list-bodies -------------------------------------------------------
    sub.add_parser(
        "list-bodies",
        help="Print built-in solar system body names.",
    )

    # ---- solar -------------------------------------------------------------
    p_s = sub.add_parser(
        "solar",
        help="Simulate named solar system bodies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_s.add_argument(
        "--bodies", metavar="NAME[,NAME…]", default="sun,earth",
        help="Comma-separated body names (e.g. sun,earth,mars).",
    )
    p_s.add_argument(
        "--duration", metavar="DUR", default="1y",
        help="Simulation duration. Suffixes: y, d, h, s, or bare seconds.",
    )
    p_s.add_argument(
        "--dt", metavar="DT", default="1d",
        help="Initial time step. Same suffix rules as --duration.",
    )
    p_s.add_argument(
        "--integrator",
        choices=["rk4", "rk45", "leapfrog", "yoshida"],
        default="leapfrog",
        help="Integration algorithm.",
    )
    p_s.add_argument(
        "--adaptive",
        choices=["none", "acceleration", "scaled"],
        default="none",
        metavar="SCHEME",
        help="Adaptive time-stepping scheme (none / acceleration / scaled).",
    )
    p_s.add_argument(
        "--output-every", type=int, default=1, metavar="N",
        help="Record a snapshot every N accepted steps.",
    )
    p_s.add_argument(
        "--horizons", action="store_true",
        help="Fetch state vectors from JPL Horizons (requires network).",
    )
    p_s.add_argument(
        "--epoch", metavar="YYYY-MM-DD", default="2000-01-01",
        help="Epoch for JPL Horizons state vectors.",
    )
    p_s.add_argument(
        "--unit", choices=["au", "km", "m"], default="au",
        help="Spatial unit for --plot.",
    )
    p_s.add_argument(
        "--plane", choices=["xy", "xz", "yz"], default="xy",
        help="Projection plane for --plot.",
    )
    p_s.add_argument(
        "--trail-length", type=int, default=None, metavar="N",
        help="Limit trail to last N frames in --plot (default: full history).",
    )
    p_s.add_argument(
        "--plot", action="store_true",
        help="Open the interactive OrbitViewer after simulation.",
    )
    p_s.add_argument(
        "--plot-energy", action="store_true",
        help="Show the energy conservation diagnostic after simulation.",
    )
    p_s.add_argument(
        "--save", metavar="PATH", default=None,
        help="Save the full trajectory to PATH (.npz binary; .npz is appended if omitted).",
    )

    # ---- run ---------------------------------------------------------------
    p_r = sub.add_parser(
        "run",
        help="Execute a TOML simulation config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_r.add_argument(
        "config", metavar="CONFIG", nargs="?", default=None,
        help="Path to a TOML simulation configuration file.",
    )
    p_r.add_argument(
        "--show-config", action="store_true",
        help="Print an annotated example config file and exit.",
    )
    p_r.add_argument(
        "--plot", action="store_true",
        help="Open the interactive OrbitViewer after simulation.",
    )
    p_r.add_argument(
        "--plot-energy", action="store_true",
        help="Show the energy conservation diagnostic after simulation.",
    )
    p_r.add_argument(
        "--save", metavar="PATH", default=None,
        help="Save the full trajectory to PATH (.npz binary; .npz is appended if omitted).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    """Parse *argv* (defaults to ``sys.argv[1:]``) and dispatch to sub-command.

    Exits with code 0 on success, non-zero on error.
    """
    parser = build_parser()
    args   = parser.parse_args(argv)

    dispatch = {
        "list-bodies": cmd_list_bodies,
        "solar":       cmd_solar,
        "run":         cmd_run,
    }
    handler = dispatch.get(args.command)
    if handler is None:        # unreachable with required=True, but defensive
        parser.print_help()
        sys.exit(1)
    sys.exit(handler(args))
