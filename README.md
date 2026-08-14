# keppy

A Python reimplementation of [keplerpp](https://github.com/davemehringer/keplerpp) —
a pairwise N-body gravitational simulation library originally written in C++.

## Design goals

* **Pure Python + NumPy** — no compiled extensions required; NumPy
  vectorization keeps inner loops fast.
* **Explicit units** — all quantities are in SI (meters, seconds, kilograms)
  unless a property name says otherwise (e.g. `time_days`).
* **Clean separation of concerns** — `Body` holds state, `NBodySystem`
  manages the collection, integrators and acceleration calculators are
  injected rather than hard-wired.
* **Testable** — every module ships with a `pytest` test suite (485+ tests).
* **Interactive** — animated orbit viewer with label toggles and dynamic
  center-body selection, matching keplerpp's interactive behavior.

---

## Installation

Install from this repository; the unrelated PyPI project named `keppy` does
not provide this library or its CLI:

```bash
git clone https://github.com/davemehringer/keppy.git
cd keppy
python -m pip install -e .              # core library (NumPy only)
python -m pip install -e ".[viz]"      # add matplotlib static plots
python -m pip install -e ".[vtk]"      # add the interactive 3-D VTK viewer
```

The editable install creates the `keppy` command used below. If you only need
the library from a checkout without installing it, use `python -m keppy` in
place of `keppy`.

---

## Quick start — Python API

```python
import numpy as np
from keppy import Body, NBodySystem, LeapfrogIntegrator, ConstantTimeStepManager, run
from keppy.constants import AU, MU_SUN, DAY

# Build bodies
sun   = Body.from_mu("Sun", MU_SUN, np.zeros(3), np.zeros(3))
earth = Body.from_mass("Earth", 5.972e24,
                        np.array([AU, 0.0, 0.0]),
                        np.array([0.0, (MU_SUN / AU) ** 0.5, 0.0]))

# Run for one year
from keppy.acceleration import PairwiseAccelerationCalculator
system = NBodySystem([sun, earth])
records = run(system,
              LeapfrogIntegrator(PairwiseAccelerationCalculator()),
              ConstantTimeStepManager(DAY),
              t_end=365 * DAY)

print(f"Steps: {len(records)}, final time: {records[-1].time / DAY:.1f} days")
```

### Interactive 3-D orbit viewer

```python
from keppy.viz.vtk_viewer import VTKOrbitViewer

viewer = VTKOrbitViewer(records, body_names=["Sun", "Earth"])
viewer.show()
```

Controls in the viewer window:
- **Left-drag** rotates the trackball camera freely; the mouse wheel zooms.
- Press **Space** to pause/resume playback before making a selection.
- **Right-click a body** to pause and centre the reference frame on it; press
  **0** to restore the absolute frame. Press **1–9** to centre the matching
  body in the displayed order without clicking.
- Press **n** to show/hide labels and **o** to show/hide trails.
- The renderer background is black by default.

### Solar system bodies

```python
from keppy.solar_system import make_body
from keppy.solar_system.horizons import fetch_system

# Built-in physical constants (mass, mu, radius, J coefficients):
sun     = make_body("Sun",     position=np.zeros(3), velocity=np.zeros(3))
jupiter = make_body("Jupiter", position=...,         velocity=...)

# Real state vectors from JPL Horizons (requires network):
system = fetch_system(["Sun", "Earth", "Mars"], epoch="2024-01-01")
```

### Save and load trajectories

```python
from keppy.io import save_trajectory, load_trajectory, save_trajectory_csv

save_trajectory(records, "orbit.npz")          # compact NumPy binary
records2 = load_trajectory("orbit.npz")

save_trajectory_csv(records, "orbit.csv",      # human-readable CSV
                    body_names=["Sun", "Earth"])
```

### Orbital elements

```python
from keppy.orbital import vectors_to_elements, elements_to_vectors
from keppy.constants import MU_SUN

el = vectors_to_elements(MU_SUN, position, velocity)
print(f"a = {el.a / AU:.3f} AU,  e = {el.e:.4f},  i = {el.i:.2f}°")

pos, vel = elements_to_vectors(MU_SUN, el)
```

---

## Quick start — CLI

```bash
# Simulate Earth and Mars for two years, show interactive viewer
keppy solar --bodies sun,earth,mars --duration 2y --plot

# All inner planets, real JPL positions at a specific date
keppy solar --bodies sun,mercury,venus,earth,mars \
            --horizons --epoch 2024-01-01 \
            --duration 5y --dt 1d \
            --plot --save inner_planets.npz

# Run from a TOML config file
keppy run my_sim.toml --plot --plot-energy --save results.npz

# Print all built-in body names
keppy list-bodies
```

### `keppy solar` options

| Option | Default | Description |
|---|---|---|
| `--bodies` | `sun,earth` | Comma-separated body names |
| `--duration` | `1y` | Simulation duration: `1y`, `365d`, `24h`, `3600s`, or bare seconds |
| `--dt` | `1d` | Initial time step (same format as `--duration`) |
| `--integrator` | `leapfrog` | `rk4`, `rk45`, `leapfrog`, or `yoshida` |
| `--adaptive` | `none` | Adaptive stepping: `acceleration` or `scaled` |
| `--output-every` | `1` | Record a snapshot every N steps |
| `--horizons` | off | Fetch state vectors from JPL Horizons |
| `--epoch` | `2000-01-01` | Epoch date for Horizons queries |
| `--unit` | `au` | Plot axis unit: `au`, `km`, or `m` |
| `--plane` | `xy` | Initial VTK camera plane: `xy`, `xz`, or `yz`; the camera can rotate freely afterwards |
| `--trail-length` | — | Limit trail to last N frames in `--plot` |
| `--plot` | off | Open the VTK interactive 3-D orbit viewer (requires `keppy[vtk]`) |
| `--plot-energy` | off | Show energy conservation diagnostic |
| `--save PATH` | — | Save trajectory to PATH (`.npz` binary) |

### After every simulation

```
──────────────────────────────────────────────────────
  Bodies     : sun, earth, mars
  Steps      : 730
  Duration   : 730.00 days
  Energy drift: 3.17e-09  (|ΔE/E₀|)
──────────────────────────────────────────────────────
```

---

## Configuration file format (`keppy run`)

Print a fully annotated template:

```bash
keppy run --show-config
```

Full format reference:

```toml
# keppy simulation configuration — TOML format

[simulation]
t_end        = 3.156e7    # end time in seconds (≈ 1 Julian year)
output_every = 1          # record a snapshot every N accepted steps
max_retries  = 10         # max consecutive step retries for adaptive methods

[integrator]
type = "leapfrog"         # rk4 | rk45 | leapfrog | yoshida

# RK45-only options (ignored for other integrators):
# rtol   = 1e-09
# atol   = 1e-03
# min_dt = 1.0

[timestep]
type = "constant"         # constant | acceleration | scaled
dt   = 86400.0            # initial / fixed step size in seconds (1 day)

# AccelerationTimeStepManager (type = "acceleration"):
# a_min           = 0.01  # fractional acceleration change → increase step
# a_max           = 0.10  # fractional acceleration change → decrease step
# increase_factor = 2.0
# decrease_factor = 2.0
# min_dt          = 1.0
# max_dt          = 1e6   # omit for no upper limit

# ScaledTimeStepManager (type = "scaled"):
# q_target      = 0.03
# exponent      = 0.3
# reject_factor = 3.0
# max_factor    = 5.0
# min_factor    = 0.1

# Repeat [[bodies]] once per body.
# Exactly one of 'mu' or 'mass' is required.
[[bodies]]
name     = "Sun"
mu       = 1.327124400180e20   # m³ s⁻²
position = [0.0, 0.0, 0.0]    # meters
velocity = [0.0, 0.0, 0.0]    # m s⁻¹

[[bodies]]
name     = "Earth"
mass     = 5.972e24            # kg
position = [1.495978707e11, 0.0, 0.0]
velocity = [0.0, 29784.69, 0.0]

# Bodies can also be specified via Keplerian elements:
# [[bodies]]
# name        = "Mars"
# mass        = 6.39e23
# center_body = "Sun"          # must appear earlier in [[bodies]]
# elements.a    = 2.2794e11    # semi-major axis, meters
# elements.e    = 0.0934       # eccentricity
# elements.i    = 1.85         # inclination, degrees
# elements.node = 49.6         # longitude of ascending node, degrees
# elements.peri = 286.5        # argument of periapsis, degrees
# elements.M    = 19.4         # mean anomaly at epoch, degrees
```

---

## Project layout

```
keppy/
├── keppy/
│   ├── __init__.py          # top-level exports
│   ├── __main__.py          # enables 'python -m keppy'
│   ├── cli.py               # command-line interface
│   ├── constants.py         # G, AU, DAY, MU_SUN, …
│   ├── body.py              # Body dataclass, Frame/Origin enums
│   ├── nbody_system.py      # NBodySystem — collection + conserved quantities
│   ├── acceleration.py      # PairwiseAccelerationCalculator (Protocol)
│   ├── integrator.py        # RK4, RK45, Leapfrog, Yoshida
│   ├── timestep.py          # TimeStepManager variants + run()
│   ├── orbital.py           # Elements, vectors ↔ elements, Kepler solver
│   ├── io/
│   │   ├── config.py        # TOML config load/save + factory helpers
│   │   └── trajectory.py    # .npz and CSV trajectory save/load
│   ├── solar_system/
│   │   ├── bodies.py        # BODY_DATA catalog + make_body() factory
│   │   └── horizons.py      # JPL Horizons state-vector fetcher
│   └── viz/
│       ├── plot.py          # plot_trajectory, plot_orbit, plot_system, plot_energy
│       └── interactive.py   # OrbitViewer (FuncAnimation + widgets)
└── tests/                   # pytest test suite (485+ tests)
```

---

## Integrators

| Name | Order | Symplectic | Notes |
|---|---|---|---|
| `leapfrog` | 2 | ✅ | Velocity Verlet (KDK form); best for long runs |
| `yoshida` | 4 | ✅ | Yoshida 1990; higher accuracy at similar cost |
| `rk4` | 4 | ❌ | Classic Runge-Kutta; secular energy drift |
| `rk45` | 4(5) | ❌ | Dormand-Prince adaptive; use with `--adaptive scaled` |

Symplectic integrators conserve a modified Hamiltonian, so total energy
oscillates rather than drifts — essential for long solar system simulations.

---

## License

GPL-3.0-or-later (same as keplerpp).
