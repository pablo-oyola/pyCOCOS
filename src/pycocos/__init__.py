"""
pycocos: standalone tokamak equilibrium and coordinate tools.
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
    "get_jacobian_function",
    "register_coordinate_system",
    "list_coordinate_systems",
]


def __getattr__(name):
    """
    Lazy-load heavy modules so lightweight subpackages can be imported without
    importing the full equilibrium stack.
    """
    if name == "Equilibrium":
        from .core.equilibrium import equilibrium as _equilibrium

        return _equilibrium
    if name == "MagneticCoordinates":
        from .core.magnetic_coordinates import magnetic_coordinates as _magnetic_coordinates

        return _magnetic_coordinates
    if name == "EQDSK":
        from .io.eqdsk import eqdsk as _eqdsk

        return _eqdsk
    raise AttributeError(f"module 'pycocos' has no attribute '{name}'")

