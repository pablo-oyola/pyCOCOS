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
   Surface quantities are interpolated onto a dense periodic
   :math:`\theta_{geom}` grid.
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

Numba Hot Paths
---------------

Heavy loops are implemented in ``pycocos.coordinates.jacobian_numba_kernels``:

- ``compute_grad_psi_abs``
- ``build_boozer_jacobian``
- ``build_power_law_jacobian``
- ``compute_theta_span``
- ``apply_scalar_scale``

All kernels are ``@njit`` and operate on contiguous numeric arrays.

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
