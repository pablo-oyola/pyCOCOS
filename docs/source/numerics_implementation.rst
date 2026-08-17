Numerics Implementation
=======================

This page describes how the coordinate physics is executed numerically.

Solver Pipeline
---------------

For each requested flux surface, ``compute_magnetic_coordinates`` performs:

1. **Field-line integration**  
   A closed poloidal trajectory is traced using
   ``pycocos.coordinates.field_lines.integrate_pol_field_line``.
2. **Periodic resampling on geometric angle**  
   Surface quantities are interpolated onto a workload-scaled periodic
   :math:`\theta_{geom}` grid. The automatic size is the next power of two of
   ``max(512, 4*ltheta, 8*spectral_max_mode)``; callers can request an explicit
   legacy or convergence-check resolution with ``n_theta_geom``.
3. **Derived geometric arrays**  
   Build :math:`B`, :math:`B_{pol}`, :math:`dR`, :math:`dZ`, :math:`dl_p`,
   and profile scalars :math:`I`, :math:`F`, :math:`q`.
4. **Jacobian assembly**  
   Construct per-surface context and evaluate Jacobian via the registry-selected
   callable.
5. **Magnetic-angle integration**  
   Integrate :math:`\theta_{mag}` and :math:`\nu` tables and interpolate to the
   output grid.
6. **Inverse mapping tables**  
   Build :math:`R(\psi,\theta_{mag})` and :math:`z(\psi,\theta_{mag})`.

Surface filtering and flux projection are batched over all retained contours.
Newton projection and both radial/angle inverse solves update only active,
unconverged points. A fused vector-valued radial spline shares Fourier phase
and coefficient work across :math:`R`, :math:`z`, and :math:`\nu`.

Accuracy Budgets
----------------

``CoordinateAccuracy`` separates approximation/solver tolerances from strict
geometric and algebraic contracts. The standard profile uses normalized flux
tolerances ``1e-5`` (projected public-grid bridge), ``1e-7`` (surface
projection), and ``1e-7`` (constrained spectral map), with an angular update
tolerance of ``1e-8`` radians. The strict profile retains the former
``1e-8``, ``1e-10``, ``1e-10``, and ``5e-11`` values. Explicit per-field
overrides remain available.

The projected-grid bridge takes a no-solve fast path when its measured label
and reflection residuals meet the selected budget. Oversized sparse repairs
are diagnosed and rejected before matrix assembly unless the caller explicitly
selects ``projected_bridge_repair_strategy="allow"``.

Staged Materialization and Checkpoints
--------------------------------------

``materialize_rz=False`` returns a ``MagneticCoordinateMapProduct`` after
surface construction and spectral fitting. It supports map values,
differentials, and angle inversion immediately; ``materialize_rz()`` builds
the traditional Cartesian datasets later without retracing. On full results,
``build_metric_cache=False`` defers derived Lamé/metric arrays.
The map-only ``jacobian`` is the determinant of that fitted map;
``target_jacobian`` retains the independently constructed table used to audit
spectral-fit residuals.

Full R-z materialization requests only mapped values and direct coordinate
tangents from each 20,000-point chunk. Reciprocal bases and metric tensors are
not allocated before the physical-flux correction; the corrected sparse
axisymmetric basis is then inverted analytically.

Supplying ``checkpoint_dir`` persists the traced/fitted construction state in
a content-addressed directory. Keys include canonical source-array hashes,
the normalized build configuration, and an algorithm version. JSON manifests,
the primitive-only NPZ payload, and per-array hashes are verified on load;
pickle is never enabled.

Numba Hot Paths
---------------

Heavy loops are implemented in ``pycocos.coordinates.jacobian_numba_kernels``:

- ``compute_grad_psi_abs``
- ``build_boozer_jacobian``
- ``build_power_law_jacobian``
- ``compute_theta_span``
- ``apply_scalar_scale``

All kernels are ``@njit`` and operate on contiguous numeric arrays.

For spectral-map batches above 100,000 points,
``coordinate_map_numba.axisymmetric_differential_kernel`` fuses the exact
axisymmetric direct/inverse basis, Jacobian, and reciprocal metric algebra in
one parallel compiled pass. Smaller batches retain vectorized NumPy because
dispatch and parallel overhead outweigh the kernel gain there.

Stability and Safeguards
------------------------

``pyCOCOS`` includes numerical guards to avoid silent corruption:

- shape checks between :math:`J(\theta)` and :math:`B(\theta)`
- finite-value checks on Jacobians
- small-denominator protection via epsilon thresholds
- Boozer identity residual check :math:`J B^2 \leftrightarrow I+qF`

Numba Runtime Policy
--------------------

``pyCOCOS`` treats Numba as mandatory for Jacobian hot paths.
If Numba cannot compile safely, execution fails fast with a clear error.

The runtime guard lives in ``pycocos.coordinates.numba_runtime`` and checks:

- Numba import/JIT probe success
- known crash-prone runtime combinations under pytest (notably
  ``pytest-pyvista`` plugin auto-loading)

This policy avoids silently falling back to pure-Python loops on heavy paths.

Verification Strategy
---------------------

The test suite covers both physics behavior and numerical execution:

- coordinate-property tests for Boozer/PEST/Hamada/Equal-Arc Jacobians
- kernel equivalence checks against small Python references
- JIT compilation smoke checks and coarse regression guards on hot-path speed
