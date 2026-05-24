# keppy — Design Notes

This document records the design decisions made during the Python reimplementation
of [keplerpp](https://github.com/davemehringer/keplerpp), and the reasoning behind
each one.  It is intended as a living document: each new module adds a section here
before or during implementation.

---

## General principles

### Units: SI throughout
All internal quantities use SI units (meters, kilograms, seconds) unless a name
explicitly says otherwise (e.g. `time_days`, `time_years`).  The C++ code mixed
astronomical units in places, relying on implicit conversions that were easy to
get wrong.  Explicit SI units make every formula checkable against a textbook.

### American English spellings
All identifiers, docstrings, and comments use American English
(e.g. "center", "meter", "barycenter", "vectorized").

### NumPy arrays for all vectors
Every 3-vector (position, velocity, acceleration) is a `numpy.ndarray` of
shape `(3,)` with dtype `float64`.  The C++ code used a custom `Vector` type
with overloaded operators.  NumPy arrays give the same arithmetic convenience
with no extra code, and they compose naturally with the vectorized pair-loop in
the acceleration calculator.

### Explicit over implicit
Behaviors that the C++ code performed silently inside constructors (barycenter
translation, unit conversion) are exposed as named methods with documented
parameters so callers know exactly what is happening.

### Injected dependencies, not hard-wired ones
The integrator and acceleration calculator are passed into `NBodySystem.step()`
rather than selected at construction time.  This keeps each class focused on one
responsibility and makes the combination easy to test in isolation.

---

## `keppy/constants.py`

All physical constants are module-level `float` values with inline comments
giving units and sources.  Grouping them in one place avoids magic numbers
scattered through the codebase and makes it trivial to update a value (e.g.
if a more precise measurement of G is published).

---

## `keppy/body.py` — `Body`, `Frame`, `Origin`

### `@dataclass` instead of a plain struct
The C++ `Body` was a struct with public fields.  A Python `@dataclass` gives
the same direct field access but also auto-generates `__init__`, enforces
type annotations, and integrates cleanly with `copy.deepcopy`.

### `mass` and `mu` as a synchronized property pair
The C++ code stored only `mu` (= G × mass).  Mass was never a first-class
value, so callers had to divide by G themselves.  In keppy both are available
as properties: setting one immediately updates the other, so there is no risk
of them getting out of sync.

```python
body.mass = 5.972e24     # mu is updated automatically
body.mu   = 1.327e20     # mass is updated automatically
```

### `from_mass()` and `from_mu()` class methods
The `@dataclass`-generated `__init__` cannot cleanly handle the mass/mu duality
(only one should be supplied at construction time).  Named class methods make
the intent explicit at the call site and avoid an overloaded constructor.

### `Frame` and `Origin` as Python `Enum`
The C++ code used plain C-style enums, which are unscoped integers.  Python
`Enum` values are namespaced, printable, and cannot be accidentally compared
to raw integers.

### `j_coefficients` as `list[float]`
The C++ code stored zonal harmonic coefficients behind a raw pointer (`body.j`).
A plain Python list is safe, iterable, and requires no manual memory management.
The convention `j_coefficients[0] = J2, [1] = J3, …` is documented in the
docstring.

### `center_body` as `Optional[Body]`
The C++ code stored a raw pointer to the central body.  A Python optional
reference carries no ownership ambiguity and is garbage-collected automatically.

### `acceleration` initialized to zeros, not set by the user
Acceleration is a derived quantity computed by the `AccelerationCalculator`.
Initializing it to zero makes it safe to read before the first integration step
without raising an error, while the docstring makes clear it is not an input.

### `Body.copy()` for snapshots
The integrator needs to save and restore body state (e.g. for adaptive stepping
or output).  A named `copy()` method using `copy.deepcopy` is more readable than
requiring callers to import `copy` themselves.

### `speed` and `kinetic_energy` as properties
These are frequently needed during simulation diagnostics.  Exposing them as
properties avoids repeated inline math at call sites and makes them easy to
mock in tests.

---

## `keppy/nbody_system.py` — `NBodySystem`

### `translate_to_barycenter()` is explicit and optional
The C++ constructor silently translated all positions to the barycenter.  In
keppy this is a named method with a boolean parameter (`translate_to_barycenter`,
default `True`) so callers who provide pre-centered initial conditions can opt
out, and the operation is visible in code review.

### Conserved quantities as `@property` accessors
`total_energy`, `angular_momentum`, and `linear_momentum` are properties that
recompute on demand.  The C++ methods could return stale values if bodies were
mutated between calls.  On-demand computation is always correct; the performance
cost is negligible for typical n < 1000.

### `kinetic_energy` and `potential_energy` use NumPy vectorization
`kinetic_energy` stacks all velocity vectors and computes ½ Σ mᵢ|vᵢ|² in one
`np.einsum`.  `potential_energy` still uses a Python double loop (O(n²) pairs)
because the loop body is a single scalar division — acceptable for n < 1000 and
clearer than a broadcasting expression that creates an n × n distance matrix
purely for the energy sum.

### `time` stored in seconds; `time_days` / `time_years` as derived properties
Storing one canonical unit internally prevents conversion errors.  Named
properties for days and years are convenient for I/O and display without
polluting the core time representation.

### Auto-assigned body IDs
If a `Body` is constructed without a `body_id` (the default is `-1`), the
system assigns one when the body is added.  This removes a common source of
bookkeeping bugs in the C++ code where IDs had to be set manually.

### `get_state()` / `set_state()` for stacked arrays
Integrators work most efficiently with flat NumPy arrays of shape `(n, 3)`.
These two methods are the bridge between the list-of-Body representation and
the array representation the integrators consume.

### `step(dt, integrator)` takes an injected integrator
`NBodySystem` owns the simulation time and the body list.  It does not own the
integration algorithm.  Passing the integrator into `step()` makes it trivial
to swap algorithms (RK4 → symplectic → adaptive) without touching the system
class.

---

## `keppy/acceleration.py` — `AccelerationCalculator`, `PairwiseAccelerationCalculator`

### `AccelerationCalculator` is a `typing.Protocol`
The C++ base class forced inheritance via a pure virtual `compute()` method.
A Python `Protocol` achieves the same interface contract with duck typing: any
object that has a `compute(positions, mus)` method satisfies the protocol,
including plain functions wrapped in a small class.  No `super().__init__()`
boilerplate is needed.

### `DistanceCalculator` is not a separate class
In keplerpp, `DistanceCalculator` was a class whose sole job was to pre-populate
distance, squared-distance, and displacement matrices before each acceleration
computation.  In keppy those matrices are computed in two NumPy lines:

```python
diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (n, n, 3)
d2   = np.einsum("ijk,ijk->ij", diff, diff)                        # (n, n)
```

Eliminating the class removes an allocation and an extra method call on every
integration step, and keeps the data flow local to `compute()`.

### Fully vectorized Newtonian pair loop
The C++ `_compute()` method iterated over all pairs with a manual `i/j` double
loop.  In keppy the entire O(n²) sum is expressed as:

```python
acc = (mus[np.newaxis, :, np.newaxis] * diff) / d3[:, :, np.newaxis]
result = acc.sum(axis=1)
```

This replaces n² Python iterations with a single C-speed NumPy reduction,
giving roughly two orders of magnitude speedup for typical system sizes.

### `_zonal_harmonic_acceleration()` is a standalone pure function
In keplerpp, `doJContrib` was an `inline` method on the `AccelerationCalculator`
base class.  Moving it to a module-level function with no side effects makes it:
- Directly unit-testable without constructing a full system.
- Reusable by any future calculator (e.g. a GPU-based one).
- Easier to read: all inputs and outputs are explicit in the signature.

### J-perturbation gating conditions
The correction is applied only when all three conditions hold:
1. The perturbing body has `j_coefficients` set and a non-zero `radius`.
2. The orbiting body's `center_body` points at the perturber (prevents
   spurious cross-body perturbations in multi-planet systems).
3. The distance is within `j_radius_factor × perturber.radius` (default 100×,
   matching keplerpp).  Beyond this distance the harmonic corrections are
   negligible relative to floating-point noise.

### J2–J10 coefficients indexed from J2
`j_coefficients[0]` is J2, `[1]` is J3, etc.  This matches the physical
convention (there is no J0 or J1 term in the zonal expansion) and avoids the
off-by-one indexing in the C++ code where `body.j->operator[](2)` was J2.

---

## `keppy/integrator.py` — Integrators

### `Integrator` is a Protocol, not an ABC
The C++ base class used pure virtual inheritance.  A Python `Protocol` gives the
same contract with duck typing: any object with `step(system, dt) -> float`
satisfies it, including thin wrappers around third-party solvers.

### `step()` returns the actual dt taken
All integrators return the actual time step from `step()`.  `NBodySystem.step()`
uses the returned value to advance its internal clock.  This makes adaptive
integrators (RK45) transparent: the caller sees exactly how far time advanced
without needing to query internal state.

### `NBodySystem.step()` updated to use the return value
In step 1 the method returned `None` and always added the requested `dt` to the
clock.  In step 3 it returns `float` (the actual step taken) so adaptive methods
work correctly.

### RK4 — classical fixed-step
Four force evaluations per step, O(dt⁴) global error.  Not symplectic: energy
error grows secularly over long integrations.  Useful as a reference/validation
integrator.

### RK45 (Dormand-Prince) — adaptive
Uses the standard Dormand-Prince Butcher tableau.  On each call, retries with a
smaller step if the RMS scaled error exceeds 1.0; otherwise the requested dt is
accepted.  Suitable when accuracy matters more than computational cost or when
forces vary on multiple timescales.

Note: the adaptive tolerance controls LOCAL error per step.  GLOBAL orbit-closure
accuracy requires either a tight tolerance or a fine step suggestion.  With
rtol=1e-9 and a coarse (1-day) suggested step, each step is individually
accurate but global phase drift can still be large — this is expected behavior
for local-error controllers and is documented in the test comments.

### Leapfrog (KDK Velocity Verlet) — symplectic 2nd-order
The Kick-Drift-Kick form requires only one force evaluation per step after
initialization (the end-of-step acceleration is cached and reused as the
start-of-step acceleration for the next step).  Being symplectic, it preserves
a shadow Hamiltonian exactly: energy error oscillates but never drifts
secularly, making it far superior to RK4 for long orbital integrations despite
being only 2nd-order.

The `reset()` method clears the cached acceleration for cases where body state
is modified externally between steps.

### Yoshida — symplectic 4th-order
Uses the standard Yoshida (1990) triple-jump coefficients (three force
evaluations per step) to achieve 4th-order accuracy while remaining symplectic.
It is the recommended default for high-accuracy long-run simulations.

### Choosing an integrator
| Use case | Recommended |
|----------|-------------|
| Short integrations, validation | RK4Integrator |
| Variable time scales, need accuracy | RK45Integrator |
| Long orbital integrations | LeapfrogIntegrator (safe default) |
| Long + high accuracy | YoshidaIntegrator |

### Orbit-closure test design
A naive "run for one year, check Earth returns to [AU, 0, 0]" test fails with
dt = 1 day because the integration error at that step size is ~2.58 × 10⁹ m
(4th-order methods) or ~2.48 × 10⁹ m (leapfrog) — dominated by the large step
size, not the integrator order.  The tests use:
- n = 3650 steps (~2.4 h per step) for orbit-closure tests: RK4 gives ~0.2 m
  error, Yoshida gives ~7 m error.
- Leapfrog is 2nd-order and needs ~36 000 steps for km-level closure; instead
  we test orbital-radius preservation (the symplectic invariant that leapfrog
  does protect well at any step size).

---

## `keppy/timestep.py` — TimeStepManagers

### `TimeStepManager` is a Protocol
Consistent with the rest of the library.  Any object exposing a `dt` property
and an `update(acc_before, acc_after) -> ChangeType` method satisfies the
interface without forced inheritance.

### `update()` returns `ChangeType` and mutates `dt` atomically
The C++ design returned an enum from `modify()` and required a separate
`getDeltaT()` call to retrieve the new step.  In keppy, `update()` does both
in one call: the caller gets the verdict and `tsm.dt` is already updated.

### `ConstantTimeStepManager`
Trivially satisfies the protocol with `update()` always returning `NO_CHANGE`.
Useful for using `run()` with a fixed step size without special-casing.

### `AccelerationTimeStepManager` — faithful to C++ keplerpp
Computes the per-body fractional acceleration change
`q_i = max_component(|Δa_i| / |a_i|)` and applies binary factor-of-2 scaling
(configurable) with three outcomes: DECREASE / NO_CHANGE / INCREASE.

Improvement: `min_dt` and `max_dt` guard rails prevent pathological step sizes;
the C++ version had neither.

### `ScaledTimeStepManager` — smooth continuous scaling
Instead of a binary jump, scales dt continuously:
`dt_new = dt * (q_target / q) ** exponent`.
This eliminates oscillation near the threshold boundary that the binary
approach can exhibit.  Uses RMS fractional change over all components and all
bodies for a single scalar quality metric `q`.

### `run()` helper
Encapsulates the retry loop (save state → step → evaluate → rollback if
DECREASE → retry) so callers do not need to reimplement it.  Returns a list of
`StepRecord` snapshots for downstream analysis or plotting.

The C++ retry logic was embedded inside the integrator `_step()` methods,
coupling integration algorithm to step-size control.  In keppy these concerns
are fully separated: any integrator works with any TimeStepManager via `run()`.

### `TargetedAccelerationTimeStepManager` — not yet implemented
The C++ version included a per-body independent stepping scheme (block-step
method) where different bodies could use different step sizes.  This is a
significant feature, left for a future step.

---

## `keppy/orbital.py` — Orbital Element Conversions

### `Elements` as a dataclass with NaN for undefined angles

The six classical Keplerian elements (a, e, i, Ω, ω, M) map cleanly to a
Python dataclass.  Rather than using sentinel values (0, -1, or raising an
exception), undefined angles are stored as `float("nan")`:

| Situation | Undefined quantity |
|-----------|-------------------|
| Equatorial orbit (i = 0 or 180°) | `node` (Ω) — no ascending node |
| Circular orbit (e = 0) | `peri` (ω) and `M` — no pericenter |

This matches the behavior of most reference implementations and avoids silent
wrong values: any downstream computation that uses a nan will produce a nan,
making the error visible immediately.

### `a` in meters (SI)
The C++ code stored `a` in astronomical units, requiring KM_PER_AU at every
call site.  Storing `a` in SI meters makes the vis-viva equation and the
period formula dimensionally self-consistent without any conversion.

### Single `mu` parameter
The C++ `keplerpp` took two separate arguments `mu_primary` and `mu_secondary`
(G × each mass) which callers always summed.  A single `mu = G × M_total` is
cleaner and matches every standard orbital mechanics reference.

### Newton-Raphson for Kepler's equation
`solve_kepler(M, e)` uses Newton-Raphson with a Danby (1988) initial guess
instead of the C++ fixed-point iteration.  Newton-Raphson converges
quadratically (each step roughly doubles the correct digits), whereas
fixed-point converges linearly.  For e → 1 the fixed-point method can require
hundreds of iterations; Newton-Raphson converges in 4–6 steps for any e < 1.

### Perifocal frame decomposition for `elements_to_vectors`
The perifocal frame (P, Q) unit vectors are formed from the rotation matrix
that maps the orbital plane into the reference frame:

```
P = R(Ω) · R(i) · R(ω) · [1, 0, 0]^T
Q = R(Ω) · R(i) · R(ω) · [0, 1, 0]^T
```

Position and velocity in the perifocal frame then use the standard elliptic
expressions:

```
r    = a(cos E - e) P + a sqrt(1-e²) sin E  Q
r_dot = sqrt(mu/a) / (1 - e cos E)  ×  (-sin E  P + sqrt(1-e²) cos E  Q)
```

This is identical to the C++ implementation — the math is the same, just
expressed with NumPy arrays instead of component-wise multiplications.

### Edge-case handling in `vectors_to_elements`
All degenerate cases are handled explicitly:

* **Equatorial** (|h_z| / |h| ≈ 1, so `n_mag` ≈ 0): `node = nan`.
* **Circular** (|e_vec| ≈ 0): `peri = nan`, `M = nan`.  The true anomaly for
  display purposes is measured from the ascending node (or from the x-axis for
  a circular equatorial orbit) but is not stored as `M` (which has no meaning
  when there is no pericenter).
* **Circular equatorial**: both `node = nan` and `peri = nan`.

The C++ code returned `0.0` for all these cases, which could be silently wrong
in downstream computations.

### `_clamp(x, lo, hi)` guard before `acos`/`asin`
Floating-point arithmetic can produce values like 1.0000000000000002 due to
rounding, which causes `math.acos` to raise a domain error.  The `_clamp`
helper clips arguments to `[-1, 1]` before every inverse trig call, exactly as
the C++ code used `std::max(-1.0, std::min(1.0, x))`.

### `period` property returns `nan`; `period_from_mu(mu)` is a method
`Elements` does not store `mu`, so the period cannot be computed without it.
The `period` property returns `nan` to signal "not available" rather than
raising an error.  `period_from_mu(mu)` is the explicit method to call when
`mu` is known.  This avoids the temptation to store `mu` inside `Elements`
(which would make it a mix of orbital state and physical parameter).

---

## `keppy/io/` — Configuration and Trajectory I/O

### TOML for config, NumPy `.npz` for trajectories

The C++ code used XML for configuration.  keppy uses **TOML** (stdlib
`tomllib` in Python 3.11+) for config and **NumPy `.npz`** for trajectory
output.

TOML is chosen over JSON because it supports inline comments (critical for
documenting simulation parameters), has clean syntax for tables and arrays,
and is now part of the Python standard library.  YAML would be even more
readable but requires a third-party dependency (`PyYAML`).

NumPy `.npz` is chosen for trajectories over CSV because:
- Files are typically 10–100× smaller (compressed binary vs. ASCII).
- Exact floating-point round-trip: no precision loss from decimal formatting.
- Zero additional dependencies.
- `np.load()` gives back named arrays directly; no parsing step.

A CSV exporter is also provided as a convenience for quick inspection in a
spreadsheet or plotting tool.

### `SimulationConfig` as a dataclass tree

The full simulation specification decomposes cleanly into nested dataclasses:

```
SimulationConfig
  ├── bodies: list[BodyConfig]
  ├── integrator: IntegratorConfig
  └── timestep: TimestepConfig
```

Each dataclass carries defaults for every optional field.  This means a
minimal config only needs `bodies` and `t_end`; everything else has a
sensible default.  The C++ code spread configuration across constructor
arguments, separate setter methods, and XML attributes, making it hard to
see what was actually set.

### Orbital elements as an input-only convenience

TOML config files can specify a body's initial conditions using Keplerian
orbital elements rather than state vectors:

```toml
[[bodies]]
name        = "Mars"
mass        = 6.39e23
center_body = "Sun"
elements.a  = 2.2794e11
elements.e  = 0.0934
...
```

`load_config` converts elements → state vectors immediately using
`elements_to_vectors`.  The `BodyConfig` dataclass always holds state
vectors (position and velocity).  This keeps the rest of the code
element-agnostic and `save_config` always writes state vectors — there is
no ambiguity about which representation is canonical.

### `build_integrator` and `build_timestep_manager` as factories

Rather than encoding integrator selection inside `NBodySystem` or
`config_to_system`, two standalone factory functions read the config and
return the appropriate concrete object.  This keeps each function small,
testable in isolation, and easy to extend: adding a new integrator type
means adding one `case` branch in `build_integrator`.

### TOML writer without third-party dependencies

`tomllib` (stdlib) is read-only.  Rather than adding `tomli_w` as a
dependency, `save_config` uses a minimal hand-rolled writer.  The config
structure is a predictable hierarchy of scalars, float arrays, and string
scalars — no arbitrary nesting.  Python's `repr(float)` produces valid TOML
float literals for all finite values; `max_dt = None` is simply omitted
(TOML has no null/NaN literal).

---

## `keppy/solar_system/` — Built-in Body Data and JPL Horizons Fetcher

### `bodies.py` — Physical data only, no orbital state

`BodyData` is a frozen dataclass holding `mu`, `radius`, and
`j_coefficients` for each body.  It deliberately has **no** position or
velocity: those are either provided by the caller or fetched from Horizons.
This separation of concerns means the physical data catalogue never goes
stale (masses and radii change far more slowly than orbital positions).

The `BODY_DATA` dictionary covers the Sun, eight planets, the Moon, Pluto,
and Ceres.  `make_body(name, position, velocity)` combines catalogue data
with caller-supplied state vectors into a `Body`.  One-liner aliases
(`sun()`, `earth()`, …) default position and velocity to zero, useful when
state vectors will be assigned later.

### `horizons.py` — JPL Horizons web API

The [JPL Horizons API](https://ssd.jpl.nasa.gov/api/horizons.api) is
queried with `urllib.request` (stdlib, no extra dependencies) for
Cartesian state vectors in the ecliptic J2000 barycentric frame.
Output units are km and km/s; both are converted to SI (m, m/s) before
being stored in the `Body`.

`HORIZONS_IDS` maps lower-case body names to Horizons numeric IDs.
Planets use system-barycentric IDs (1–8) so that satellite masses are
automatically included.  Earth uses geocenter (399) rather than the
Earth-Moon barycenter (3) so that the Moon can be queried and modelled
separately at full accuracy.

#### Response parsing
The Horizons text response is parsed with two compiled regular expressions
(`_XYZ_RE`, `_VEL_RE`) that extract values between the `$$SOE` and `$$EOE`
markers.  The patterns handle both the positive-value format
(`X = 1.496E+08`) and the negative-value format (`X =-2.622E+07`, no space
between `=` and `-`) that Horizons uses.

#### `fetch_body` design
`fetch_body(name, epoch, ...)` accepts optional keyword overrides
(`horizons_id`, `mu`, `mass`, `radius`) so that bodies not in the built-in
catalogue (minor planets, spacecraft targets, etc.) can still be queried by
passing a Horizons ID and explicit physical parameters.  If neither the
catalogue nor the caller supplies a gravitational parameter, a `HorizonsError`
is raised immediately with a helpful message.

#### `fetch_system` and barycentric coordinates
`fetch_system` makes one HTTP request per body and returns a `NBodySystem`.
`translate_to_barycenter` defaults to `False` because Horizons vectors with
`CENTER=500@0` are already in the solar system barycentric frame; applying
a second barycenter translation would introduce small numerical errors.

#### Network test strategy
Live tests (those that actually call the Horizons API) are decorated with
`@requires_live` and skipped when the API is not reachable.  All other
tests use `unittest.mock.patch` to replace `urllib.request.urlopen` with
a pre-recorded response string, so the test suite is fully deterministic in
offline environments.

### Saturn J2 > Jupiter J2
Saturn's zonal oblateness coefficient J2 (0.01630) is larger than
Jupiter's (0.01470).  This is physically correct: Saturn has lower mean
density (0.687 g/cm³ vs Jupiter's 1.33 g/cm³) and a similar rotation
period, so its equatorial bulge is proportionally larger.

---

## `keppy/viz/` — Matplotlib Visualization

### Optional dependency

matplotlib is listed under `[project.optional-dependencies] viz` in
`pyproject.toml` rather than as a hard dependency.  The viz package raises
`ModuleNotFoundError` with an install hint at import time rather than
silently doing nothing, so users know exactly what to install.

### Four focused functions

| Function | Input | Output |
|---|---|---|
| `plot_trajectory` | `list[StepRecord]` | Body paths over time |
| `plot_orbit` | `Elements` + mu | Orbital ellipse sampled at N anomalies |
| `plot_system` | `NBodySystem` | Snapshot of current body positions |
| `plot_energy` | `list[StepRecord]` + mus | Total energy (or relative error) vs time |

### Axes are returned, not shown

All functions return the `matplotlib.axes.Axes` they drew on.  They never
call `plt.show()` — that is the caller's responsibility.  This follows the
matplotlib object-oriented idiom and makes embedding plots in larger figures
easy: call any function with an existing `ax=` argument to draw onto it.

### `unit` and `time_unit` parameters

Spatial axes accept `unit="au"`, `"km"`, or `"m"`.  Time axes accept
`time_unit="s"`, `"h"`, `"day"`, or `"year"`.  All stored data stays in
SI; the conversion factor is applied only to the plotted values.  This
avoids unit-conversion bugs in the physics code while giving readable plots.

### 2-D and 3-D support

A `plane` parameter selects the projection: `"xy"`, `"xz"`, `"yz"`, or
`"3d"`.  For `"3d"` a `matplotlib` `Axes3D` is created (or required if an
existing axes is passed).  The same API works for all four projections.

### `plot_orbit` sampling strategy

The orbit is sampled by iterating `M` from 0° to 360° in `n_points` steps
and calling `elements_to_vectors` at each value.  This correctly handles
all inclinations and non-zero Ω and ω without any special-case 2-D
geometry, at the cost of one `solve_kepler` call per point.  For a typical
`n_points=360` this takes < 1 ms.

### `plot_energy` diagnostic

`_compute_total_energy` recomputes KE + PE from scratch at each step using
the stored positions and velocities and the caller-supplied `mus` array.
This is independent of the integrator — no internal integrator state is
inspected.  The `relative=True` default plots `(E − E₀) / |E₀|`, which
immediately reveals whether the integrator conserves energy and at what
level (Leapfrog: oscillates near 10⁻⁶; RK4: drifts secularly).

---

## `keppy/viz/interactive.py` — `OrbitViewer`

### Goal

The C++ keplerpp application includes an interactive orbit viewer that lets
the user toggle body name labels on and off and dynamically change which
body is at the center of the reference frame while the simulation plays.
`OrbitViewer` reproduces both features as a matplotlib figure with live
controls alongside a `FuncAnimation` playback loop.

### Layout: orbit axes + two widget panels

The figure is split horizontally: the main orbit axes occupy the left 65%
of the figure, and two stacked widget panels occupy the right 30%:

* **"Show labels" (top)** — a `CheckButtons` widget with one toggle per
  body.  Clicking a button calls `_on_label_toggle(label)`, which flips
  `_show_labels[idx]` and calls `set_visible()` on the corresponding
  `Text` artist.
* **"Center on" (bottom)** — a `RadioButtons` widget offering
  `"(absolute)"` plus one option per body.  Clicking an option calls
  `_on_center_change(label)`, which updates `_center_idx` and recomputes
  axis limits.

Using `fig.add_axes([left, bottom, width, height])` with explicit
coordinates gives precise control over placement without depending on
`GridSpec` or `subplot_mosaic`, which don't play well with widget axes.

### Reference-frame transformation

All positions are pre-computed once at construction and stored in
`_pos_all` with shape `(n_steps, n_bodies, 3)`.  `_relative_positions()`
returns the same array unchanged when `_center_idx == -1`, or subtracts the
center body's column when a body is selected:

```python
center = pos[:, center_idx : center_idx + 1, :]   # broadcast-safe slice
return pos - center
```

This is applied lazily on every `_update_frame` call rather than eagerly
re-storing a transformed copy, so switching the center body is instant and
does not allocate extra memory proportional to trajectory length.

### `blit=False` for widget compatibility

`FuncAnimation` is created with `blit=False`.  Using `blit=True` would
require the widget axes to be re-drawn explicitly on every tick, which is
awkward.  With `blit=False` matplotlib redraws the whole figure on every
frame.  For typical trajectory lengths (< 10 000 steps) this is fast
enough; if performance is needed a future improvement could use
`blit=True` with manual background restoration.

### Trail control

`trail_length=None` (default) shows the full history up to the current
frame.  An integer value limits the trail to the last `trail_length` frames
via `_trail_slice(frame_idx)`:

```python
slice(max(0, frame_idx - trail_length), frame_idx + 1)
```

The same slice is applied after the reference-frame transformation, so
trails are always shown in the currently selected frame.

### Axis limits on center change

When the center body changes, `_update_axis_limits()` recomputes limits
from the full transformed trajectory, excluding the center body itself
(which is always at the origin).  An 8% margin is added on each side.
This ensures all non-center bodies remain visible regardless of their
orbital scale.

### `animate()` returns `FuncAnimation`

The public `animate()` method returns the `FuncAnimation` object rather
than calling `plt.show()` internally.  This follows the matplotlib idiom
of keeping lifecycle decisions with the caller: the user can embed the
viewer inside a larger GUI, save it with `anim.save()`, or call
`plt.show()` at their discretion.  A convenience `show()` wrapper handles
the common case.

---

## Roadmap

| Step | Module | Status |
|------|--------|--------|
| 1 | `body.py`, `nbody_system.py` | ✅ Done |
| 2 | `acceleration.py` | ✅ Done |
| 3 | `integrator.py` — RK4, adaptive RK, symplectic | ✅ Done |
| 4 | `timestep.py` — adaptive step sizing | ✅ Done |
| 5 | `orbital.py` — state vectors ↔ Keplerian elements | ✅ Done |
| 6 | `io/` — config reader/writer, trajectory output | ✅ Done |
| 7 | `solar_system/` — built-in bodies, JPL Horizons fetcher | ✅ Done |
| 8 | `viz/` — matplotlib static + interactive visualization | ✅ Done |
