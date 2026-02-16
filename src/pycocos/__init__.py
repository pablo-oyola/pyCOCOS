"""
pycocos: standalone tokamak equilibrium and coordinate tools.
"""

__version__ = "0.1.0"

from .core import equilibrium as Equilibrium
from .core import magnetic_coordinates as MagneticCoordinates
from .io import EQDSK
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

