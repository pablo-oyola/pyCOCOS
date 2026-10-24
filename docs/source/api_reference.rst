API Reference
=============

The top-level package re-exports the objects most scripts need:
``EQDSK``, ``Equilibrium``, ``MagneticCoordinates``,
``compute_magnetic_coordinates``, ``CoordinateAccuracy``,
``MagneticCoordinateMapProduct`` and the registry helpers
``get_jacobian_function``, ``register_coordinate_system`` and
``list_coordinate_systems``.  They are documented below in the modules that
define them.

I/O and COCOS
-------------

.. automodule:: pycocos.io.eqdsk
   :members:
   :undoc-members:

.. automodule:: pycocos.io.cocos
   :members:
   :undoc-members:

Coordinates
-----------

.. automodule:: pycocos.coordinates.compute_coordinates
   :members:
   :undoc-members:

.. automodule:: pycocos.coordinates.jacobian_builders
   :members:
   :undoc-members:

.. automodule:: pycocos.coordinates.jacobian_numba_kernels
   :members:
   :undoc-members:

.. automodule:: pycocos.coordinates.jacobians
   :members:
   :undoc-members:

.. automodule:: pycocos.coordinates.registry
   :members:
   :undoc-members:

.. automodule:: pycocos.coordinates.accuracy
   :members:
   :undoc-members:

.. automodule:: pycocos.coordinates.product
   :members:
   :undoc-members:

.. automodule:: pycocos.coordinates.checkpoint
   :members:
   :undoc-members:

.. automodule:: pycocos.coordinates.field_lines
   :members:
   :undoc-members:

Core
----

.. automodule:: pycocos.core.equilibrium
   :members:
   :undoc-members:
   :exclude-members: getFSprop

.. automodule:: pycocos.core.magnetic_coordinates
   :members:
   :undoc-members:
