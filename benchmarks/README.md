# Coordinate-mapping benchmark

`benchmark_coordinate_mapping.py` exercises the public
`EQDSK.compute_coordinates()` workflow on a supplied g-EQDSK and writes one
machine-readable JSON document. It records:

- coordinate-construction wall time (EQDSK load time is reported separately),
- peak Python-managed allocation measured by `tracemalloc`,
- surface, bridge, coordinate-map, and final-coordinate diagnostics that are
  available in the active checkout,
- summaries of key coordinate and physics arrays, and
- pointwise output deltas between requested variants, including wrapped
  angular deltas for `theta` and `nu`.

The benchmark uses the source tree adjacent to this directory, even when a
different pyCOCOS checkout is installed in editable mode.

## Examples

Compare the standard and strict accuracy profiles using automatic surface
quadrature:

```bash
python benchmarks/benchmark_coordinate_mapping.py equilibrium.geqdsk \
  --cocos-in 1 --comparison accuracy \
  --output accuracy-benchmark.json
```

Compare automatic quadrature with the historical explicit 7200-point surface
grid:

```bash
python benchmarks/benchmark_coordinate_mapping.py equilibrium.geqdsk \
  --cocos-in 1 --comparison theta \
  --lpsi 256 --ltheta 256 \
  --output theta-benchmark.json
```

Use `--comparison all` for the full standard/strict by automatic/legacy
matrix. The default `lpsi=33` and `ltheta=65` are deliberately modest so a
first run is not accidentally a production-sized job. Pass production sizes
explicitly when collecting final evidence.

Use ``--map-only`` to measure the surface/bridge/spectral-map stage without
allocating the full Cartesian derivative datasets. This is the relevant mode
for workflows that consume ``MagneticCoordinateMapProduct`` directly or defer
R-z materialization until a later phase.

The strict projected-flux bridge can intentionally reject an oversized sparse
repair when `--projected-bridge-repair-strategy bounded` is active. Such a run
is preserved as structured failure data in the JSON. Use the `allow` strategy
only after reviewing the estimated system size and available memory.

`tracemalloc` observes Python-managed allocations. Some native allocations
made by NumPy, SciPy, BLAS, or compiled kernels may not be included, so its
peak is useful for relative comparisons but is not a process RSS measurement.

Run `python benchmarks/benchmark_coordinate_mapping.py --help` for all grid,
COCOS, symmetry, repetition, and output options.
