"""
Coordinate system implementations for pycocos.
"""

from . import jacobians
from . import registry
from . import field_lines
from . import compute_coordinates

from .registry import (
    get_jacobian_function,
    register_coordinate_system,
    list_coordinate_systems,
    JACOBIAN_REGISTRY,
)

from .field_lines import (
    get_field_line,
    integrate_pol_field_line,
)

from .compute_coordinates import compute_magnetic_coordinates

__all__ = [
    "jacobians",
    "registry",
    "field_lines",
    "compute_coordinates",
    "get_jacobian_function",
    "register_coordinate_system",
    "list_coordinate_systems",
    "JACOBIAN_REGISTRY",
    "get_field_line",
    "integrate_pol_field_line",
    "compute_magnetic_coordinates",
]

