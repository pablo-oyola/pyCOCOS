Physics Implementation
======================

This page maps the implemented coordinate physics to the formulas used by
``pyCOCOS``.

Per-Surface Inputs
------------------

Each flux surface is represented by a geometric context containing:

- :math:`R(\theta)` major radius
- :math:`B(\theta)` magnetic field magnitude
- :math:`B_{pol}(\theta)` poloidal field magnitude
- :math:`dl_p(\theta)` poloidal arc-length increments
- scalar profiles :math:`I(\psi)`, :math:`F(\psi)`, :math:`q(\psi)`

The context is assembled in ``pycocos.coordinates.compute_coordinates`` and
passed to the Jacobian builder layer.

Jacobian Family
---------------

The implemented Jacobian family is:

.. math::

   J(\psi,\theta) = \frac{R(\theta)^i}{\left|\nabla\psi\right|^j B(\theta)^k}

for coordinate-specific integer exponents :math:`(i,j,k)`.

Axisymmetric identity used in the implementation:

.. math::

   \left|\nabla\psi\right| = R B_{pol}

Coordinate-Specific Branches
----------------------------

Boozer
^^^^^^

.. math::

   J_{Boozer} = \frac{h(\psi)}{B^2}, \qquad h(\psi)=I+qF

This branch is implemented directly and validated with the diagnostic residual:

.. math::

   \epsilon_h = \max_{\theta} \left|J B^2 - (I+qF)\right|

PEST
^^^^

Uses the :math:`R^2` family branch:

.. math::

   (i,j,k)=(2,0,0)

followed by per-surface normalization.

Equal-Arc
^^^^^^^^^

Uses:

.. math::

   (i,j,k)=(1,1,0)

which gives :math:`J \propto R/|\nabla\psi|`, then per-surface normalization.

Hamada
^^^^^^

Uses:

.. math::

   (i,j,k)=(0,0,0)

which yields a theta-independent Jacobian after normalization on each surface.

Per-Surface Normalization
-------------------------

For non-Boozer branches, ``pyCOCOS`` applies a surface normalization so the
constructed poloidal angle spans :math:`2\pi`, following the Eq. 8.99/8.100
style logic:

.. math::

   d\theta = \frac{R}{|J|\,|\nabla\psi|}\,dl_p

The raw span is integrated over the surface and used to scale :math:`J`.

Code Mapping
------------

- Context assembly: ``pycocos.coordinates.compute_coordinates``
- Physics branches + normalization orchestration:
  ``pycocos.coordinates.jacobian_builders``
- Heavy numerical kernels:
  ``pycocos.coordinates.jacobian_numba_kernels``
