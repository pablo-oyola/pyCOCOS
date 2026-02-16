# pyCOCOS

`pyCOCOS` is a standalone Python package for tokamak equilibrium handling and
magnetic coordinate workflows extracted from `pynova`.

## Features

- g-EQDSK read/write and equilibrium construction
- Magnetic field handling on R-z grids
- Magnetic coordinates (Boozer framework and registry architecture)
- COCOS convention detection and transformations

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from pycocos import EQDSK

eq = EQDSK("equilibrium.geqdsk")
mag_coords = eq.compute_coordinates(coordinate_system="boozer")
coords = mag_coords(R=2.0, z=0.0)
```

## Package Layout

- `pycocos.core`: equilibrium and magnetic coordinate classes
- `pycocos.io`: EQDSK and COCOS utilities
- `pycocos.coordinates`: Jacobians, registry, field-line and coordinate solvers
- `pycocos.utils`: performance helpers (numba utilities)

