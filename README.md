# pyCOCOS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23811057.svg)](https://doi.org/10.5281/zenodo.23811057)

<!-- docs-intro-start -->
Tokamak equilibria and magnetic coordinates in Python: g-EQDSK input and
output in any COCOS convention, the magnetic field on the (R, z) grid, and
flux-surface coordinates (Boozer, PEST, equal-arc, Hamada) with their metric
and Jacobian.  pyCOCOS is the equilibrium layer of
[superNOVA](https://github.com/pablo-oyola/superNOVA), and works on its own.

- **COCOS handled once.**  A file is read in its own convention, detected or
  given, and converted on load to one internal convention (COCOS 1 by
  default).  Writing converts only the output copy.
- **One coordinate map.**  Each coordinate system is a single Fourier ×
  radial-spline representation of (R, z, ν)(ψ, θ).  The forward and inverse
  transforms, the metric tensors and the signed Jacobian all come from that
  one map.
- **General shapes.**  The flux surfaces are traced without assuming up-down
  symmetry, and nesting and orientation are checked.
- **Registered Jacobians.**  The built-in systems are entries in a registry,
  and your own Jacobian can be added the same way.
<!-- docs-intro-end -->

Documentation: <https://pablo-oyola.github.io/pyCOCOS/>

<!-- docs-guide-start -->
## Installation

pyCOCOS needs Python 3.10 or later.  Numba is a hard dependency: the
Jacobian kernels have no pure-Python fallback.

```bash
pip install git+https://github.com/pablo-oyola/pyCOCOS.git
```

From a clone, for development:

```bash
git clone git@github.com:pablo-oyola/pyCOCOS.git
cd pyCOCOS
pip install -e ".[dev]"        # tests; ".[docs]" for the documentation tools
```

## Quick start

Any g-EQDSK file will do.  The repository ships a generator for an analytic
Solov'ev equilibrium, so the examples need no experimental data:

```bash
python examples/solovev_equilibrium.py     # writes examples/solovev.geqdsk
```

```python
from pycocos import EQDSK

# A g-EQDSK file does not record its COCOS: give it, or let pyCOCOS detect it
# from the signs in the file plus the toroidal orientation and flux units.
eq = EQDSK("examples/solovev.geqdsk", cocos_in=1)
eq.summary()                      # grid, axis, flux and field ranges

# Boozer coordinates on 65 flux surfaces x 128 poloidal points.
mag = eq.compute_coordinates("boozer", lpsi=65, ltheta=128)

point = mag(R=3.5, z=0.3)                     # -> psi, theta, nu
back = mag.transform_inverse(psi=0.5, thetamag=0.0, psi_is_norm=True)
q = mag.deriv["q"]                            # q from the traced surfaces
```

The point (R, z) = (3.5, 0.3) m maps to ψ = 0.6210 Wb/rad and θ = 6.069 rad.
The ψ_N = 0.5 surface crosses θ = 0 at R = 3.735 m.  The q profile matches the
equilibrium's exact one to about 1 part in 10⁵.  The construction takes a few
seconds.

The [examples](https://github.com/pablo-oyola/pyCOCOS/tree/master/examples)
cover the rest:

| Script | What it shows |
|---|---|
| `solovev_equilibrium.py` | writing an analytic equilibrium with an exact q, as a g-EQDSK |
| `basic_usage.py` | loading, COCOS detection, the field and the profiles |
| `boozer_coordinates.py` | Boozer coordinates, q against the exact profile, both transforms, a plot of the grid |
| `eqdsk_io.py` | converting a file to COCOS 11 and reading it back |

## Coordinate systems

`compute_coordinates(coordinate_system=...)` builds any registered system.
They differ in the Jacobian J(ψ, θ):

| Name | Jacobian |
|---|---|
| `boozer` | J = h/B², with h = I + qF |
| `pest` | J ∝ R² |
| `equal_arc` | J ∝ R/\|∇ψ\| |
| `hamada` | J constant on each surface |

`list_coordinate_systems()` lists them.  `register_coordinate_system(...)`
adds a new one: a function `jacobian(context) -> J(theta)` on one surface.
Give it a `cache_version` before enabling persistent checkpoints.  Large
builds can be staged with `materialize_rz=False` and cached with
`checkpoint_dir=...`.  The tolerance budget is chosen with
`coordinate_accuracy="standard"` (the default) or `"strict"`.  The
[numerics page](https://pablo-oyola.github.io/pyCOCOS/numerics_implementation.html)
describes all three.

## Converting COCOS

```python
gdata = eq.to_geqdsk(cocos_out=11)              # a converted copy, as a dict
eq.save("solovev_cocos11.geqdsk", cocos_out=11) # refuses to overwrite a file
EQDSK("solovev_cocos11.geqdsk", cocos_in=11)    # reads back to the same arrays
```

## GUI

```bash
pycocos-gui
```

The GUI is a Tk window.  It loads a g-EQDSK, plots any 1-D profile or 2-D
field, computes one or more coordinate systems, and draws their grids over
the 2-D plots.  On Debian/Ubuntu, Tk comes from `sudo apt install
python3-tk`.

## Package layout

| Module | Contents |
|---|---|
| `pycocos.io` | `EQDSK` reader/writer, COCOS tables, detection and transforms |
| `pycocos.core` | `Equilibrium` (field, flux, profiles on the grid) and `MagneticCoordinates` |
| `pycocos.coordinates` | surface tracing, the coordinate map, Jacobian registry and kernels, accuracy budgets, checkpoints |
| `pycocos.utils` | Numba interpolation helpers |
| `pycocos.gui` | the Tk viewer behind `pycocos-gui` |

## Tests

```bash
pytest
```

The `pytest-pyvista` plugin can crash numba/llvmlite on the first JIT
compile, so `pyproject.toml` turns it off (`-p no:pyvista`).
<!-- docs-guide-end -->

## Conventions

[CONVENTIONS.md](CONVENTIONS.md) gives the COCOS, Boozer-gauge,
current-coefficient and metric conventions the code follows.

## Citing

If you use pyCOCOS in published work, please cite it.  The DOI
[10.5281/zenodo.23811057](https://doi.org/10.5281/zenodo.23811057) always
resolves to the latest release, and each release also has its own DOI on
Zenodo.  GitHub's "Cite this repository" button reads
[CITATION.cff](CITATION.cff).

```bibtex
@software{oyola_pycocos,
  author    = {Oyola, Pablo},
  title     = {pyCOCOS},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23811057},
  url       = {https://doi.org/10.5281/zenodo.23811057}
}
```

## License

MIT. See [LICENSE](LICENSE).
