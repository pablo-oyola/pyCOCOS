Overview
========

Scope
-----

``pyCOCOS`` focuses on:

- Equilibrium reading and magnetic field reconstruction
- COCOS convention handling and conversion
- Magnetic coordinate generation (Boozer, PEST, Equal-Arc, Hamada)
- Jacobian-driven coordinate transformations

Package Layout
--------------

- ``pycocos.core``: equilibrium model and magnetic-coordinate container classes
- ``pycocos.io``: EQDSK I/O and COCOS utilities
- ``pycocos.coordinates``: field-line tracing, Jacobian builders, coordinate solvers
- ``pycocos.utils``: low-level Numba interpolation helpers

Build Documentation
-------------------

Install the docs dependencies and build HTML pages:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/_build/html

The rendered site is written to ``docs/_build/html/index.html``.
