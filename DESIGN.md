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

## Roadmap

| Step | Module | Status |
|------|--------|--------|
| 1 | `body.py`, `nbody_system.py` | ✅ Done |
| 2 | `acceleration.py` | ✅ Done |
| 3 | `integrator.py` — RK4, adaptive RK, symplectic | ⬜ Next |
| 4 | `timestep.py` — adaptive step sizing | ⬜ |
| 5 | `orbital.py` — state vectors ↔ Keplerian elements | ⬜ |
| 6 | `io/` — config reader/writer, trajectory output | ⬜ |
| 7 | `solar_system/` — built-in bodies, JPL Horizons fetcher | ⬜ |
| 8 | Visualization — matplotlib / plotly | ⬜ |
