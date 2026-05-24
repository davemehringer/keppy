# keppy

A Python reimplementation of [keplerpp](https://github.com/davemehringer/keplerpp) —
a pair-wise N-body gravitational simulation library.

## Design goals

* **Pure Python + NumPy** — no compiled extensions required, but NumPy
  vectorisation keeps inner loops fast.
* **Explicit units** — all quantities are in SI (meters, seconds, kilograms)
  unless a property name says otherwise (e.g. `time_days`).
* **Clean separation of concerns** — `Body` holds state, `NBodySystem`
  manages the collection and conserved quantities, integrators and
  acceleration calculators are injected rather than hard-wired.
* **Testable** — every module ships with a `pytest` test suite.

## Quick start

```python
import numpy as np
from keppy import Body, NBodySystem
from keppy.constants import AU, G, MU_SUN

# Sun at origin
sun = Body.from_mu("Sun", MU_SUN,
                   position=np.zeros(3),
                   velocity=np.zeros(3))

# Earth in a circular orbit at 1 AU
v_circ = (MU_SUN / AU) ** 0.5
earth = Body.from_mass("Earth", 5.972e24,
                        position=np.array([AU, 0.0, 0.0]),
                        velocity=np.array([0.0, v_circ, 0.0]))

sys = NBodySystem([sun, earth])
print(sys)
```

## Project layout

```
keppy/
├── keppy/
│   ├── __init__.py
│   ├── constants.py       # G, AU, DAY, YEAR, …
│   ├── body.py            # Body dataclass, Frame/Origin enums
│   └── nbody_system.py    # NBodySystem
├── tests/
│   ├── test_body.py
│   └── test_nbody_system.py
└── pyproject.toml
```

## Roadmap

- [ ] `AccelerationCalculator` — pairwise Newtonian gravity (+ J₂ perturbations)
- [ ] `Integrator` — RK4, adaptive RK, symplectic (McLachlan)
- [ ] `TimeStepManager` — adaptive step sizing
- [ ] `orbital.py` — state vectors ↔ Keplerian elements
- [ ] `io/` — XML config reader/writer, trajectory output
- [ ] `solar_system/` — built-in body data, JPL Horizons fetcher
- [ ] 3-D visualisation via matplotlib / plotly

## Licence

GPL-3.0-or-later (same as keplerpp).
