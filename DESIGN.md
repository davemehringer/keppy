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

## Roadmap

| Step | Module | Status |
|------|--------|--------|
| 1 | `body.py`, `nbody_system.py` | ✅ Done |
| 2 | `acceleration.py` | ✅ Done |
| 3 | `integrator.py` — RK4, adaptive RK, symplectic | ✅ Done |
| 4 | `timestep.py` — adaptive step sizing | ✅ Done |
| 5 | `orbital.py` — state vectors ↔ Keplerian elements | ✅ Done |
| 6 | `io/` — config reader/writer, trajectory output | ⬜ |
| 7 | `solar_system/` — built-in bodies, JPL Horizons fetcher | ⬜ |
| 8 | Visualization — matplotlib / plotly | ⬜ |
