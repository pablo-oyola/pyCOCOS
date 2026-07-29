# pyCOCOS

`pyCOCOS` is a standalone Python package for tokamak equilibrium handling and
magnetic coordinate workflows extracted from `pynova`.

## Features

- g-EQDSK read/write and equilibrium construction
- Magnetic field handling on R-z grids
- Magnetic coordinates (Boozer, PEST, Equal-Arc, Hamada)
- Flux-constrained surfaces for general up-down-asymmetric equilibria
- One Fourier/radial-spline coordinate map for transforms, derivatives, metrics,
  and the signed physical Jacobian
- COCOS convention detection and transformations

## Installation

```bash
pip install -e .
```

Documentation build extras:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/_build/html
```

GUI launcher:

```bash
pycocos-gui
```

The GUI supports:

- loading an EQDSK equilibrium file
- plotting selectable 1D and 2D variables
- computing magnetic coordinates for one or more coordinate systems
- overlaying computed magnetic-coordinate grids on top of a 2D plot

## CI/CD (GitHub Actions)

- `CI` workflow (`.github/workflows/ci.yml`)
  - Runs test matrix on Python 3.10/3.11/3.12
  - Verifies docs build with `sphinx-build -W`
- `Deploy Docs` workflow (`.github/workflows/docs-deploy.yml`)
  - Builds Sphinx docs on pushes to `main`/`master`
  - Publishes docs to GitHub Pages

To enable documentation publishing, set in GitHub repository settings:

- `Settings -> Pages -> Source`: **GitHub Actions**

## Quick Start

```python
from pycocos import EQDSK

eq = EQDSK("equilibrium.geqdsk", cocos_in=1, cocos_internal=1)
mag_coords = eq.compute_coordinates(coordinate_system="boozer")
coords = mag_coords(R=2.0, z=0.0)
```

See [CONVENTIONS.md](CONVENTIONS.md) for the canonical COCOS, Boozer-gauge,
current-coefficient, and metric contracts.

## Jacobian Formula Mapping

`pycocos.coordinates.jacobian_builders` implements the PDF Jacobian family:

- General family: `J = R^i / |grad(psi)|^j / B^k`
- `boozer`: `J = h/B^2` with `h = I + qF`
- `pest`: `J ~ R^2` (per-surface normalized)
- `equal_arc`: `J ~ R/|grad(psi)|` (per-surface normalized)
- `hamada`: `J ~ constant` on each flux surface (per-surface normalized)

Implementation details:

- `|grad(psi)|` is computed from axisymmetric identity `|grad(psi)| = R*Bpol`
- Per-surface normalization follows the `theta`-span construction from Eq. 8.99/8.100 logic
- Hot loops (power-law Jacobian assembly, Boozer `h/B^2`, normalization integrals) run in
  `numba` kernels in `pycocos.coordinates.jacobian_numba_kernels`
- Registry callables use the single interface `jacobian(context) -> J(theta)`
- Custom non-Boozer Jacobian shapes are validated and normalized once to a
  monotonic `0 -> 2*pi` poloidal coordinate

## Coordinate Construction

Each retained contour is Fourier-filtered without imposing parity, projected
back to its requested physical poloidal flux, and checked for orientation and
strict nesting. Up-down-asymmetric sine and cosine content is retained.

The inverse map `(R, z, nu)(psi, theta)` is represented by one periodic Fourier
expansion and one radial spline family. The radial fit uses signed
sqrt-normalized flux for regularity near the magnetic axis, while all exported
derivatives are transformed back to physical `psi`. Forward/inverse transforms,
covariant and contravariant metrics, and the Jacobian are all evaluated from
this same map. The fitted annulus is exposed as
`coords.inside_coordinate_domain`; equilibrium `dPsi/dR` and `dPsi/dz` remain
defined on the complete finite R-z grid, as does the pure flux metric
`g^{psi psi}`. Magnetic-angle derivatives and mixed/angular metrics are masked
outside the fitted annulus.

## Numba Runtime Notes

- `pycocos` treats Numba as mandatory for Jacobian hot paths; there is no silent
  fallback to Python loops.
- A runtime guard checks Numba availability and JIT readiness and fails fast with
  a clear error if the environment is unsafe.
- In this environment, `pytest` with the `pytest-pyvista` plugin can trigger a
  low-level `numba/llvmlite` segmentation fault during first JIT compile.
  Project tests disable this plugin by default (`-p no:pyvista`).

## Package Layout

- `pycocos.core`: equilibrium and magnetic coordinate classes
- `pycocos.io`: EQDSK and COCOS utilities
- `pycocos.coordinates`: Jacobians, registry, field-line and coordinate solvers
- `pycocos.utils`: performance helpers (numba utilities)
