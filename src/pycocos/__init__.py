"""
pycocos: standalone tokamak equilibrium and coordinate tools.

Public surface
--------------

* Equilibrium and coordinate classes (PascalCase, lazy-loaded):

  - ``Equilibrium`` -- equilibrium wrapper around an EQDSK + flux/field data.
  - ``MagneticCoordinates`` -- magnetic-coordinate frame attached to an
    equilibrium (Boozer, PEST, Hamada, equal-arc).
  - ``EQDSK`` -- g-EQDSK reader/writer with COCOS handling.

  Lowercase aliases (``equilibrium``, ``magnetic_coordinates``, ``eqdsk``)
  pointing at the same factory classes are also available for callers that
  prefer the module-style name.

* Coordinate registry helpers:

  - ``get_jacobian_function`` -- look up a Jacobian builder by name.
  - ``register_coordinate_system`` -- register a new coordinate system.
  - ``list_coordinate_systems`` -- enumerate registered systems.

* Pipeline helpers exposed at the top level for convenience:

  - ``compute_magnetic_coordinates`` -- trace flux surfaces and assemble
    ``MagneticCoordinates``.

Both the PascalCase classes and the lowercase aliases delegate to the same
underlying objects; pick whichever style matches your codebase.
"""

__version__ = "0.1.0"

from .coordinates import (
    get_jacobian_function,
    register_coordinate_system,
    list_coordinate_systems,
)

__all__ = [
    "Equilibrium",
    "EQDSK",
    "MagneticCoordinates",
    "equilibrium",
    "magnetic_coordinates",
    "eqdsk",
    "compute_magnetic_coordinates",
    "get_jacobian_function",
    "register_coordinate_system",
    "list_coordinate_systems",
]


def __getattr__(name):
    """
    Lazy-load heavy modules so lightweight subpackages can be imported without
    importing the full equilibrium stack.
    """
    if name in ("Equilibrium", "equilibrium"):
        from .core.equilibrium import equilibrium as _equilibrium

        return _equilibrium
    if name in ("MagneticCoordinates", "magnetic_coordinates"):
        from .core.magnetic_coordinates import magnetic_coordinates as _magnetic_coordinates

        return _magnetic_coordinates
    if name in ("EQDSK", "eqdsk"):
        from .io.eqdsk import eqdsk as _eqdsk

        return _eqdsk
    if name == "compute_magnetic_coordinates":
        from .coordinates.compute_coordinates import compute_magnetic_coordinates as _cmc

        return _cmc
    raise AttributeError(f"module 'pycocos' has no attribute {name!r}")
