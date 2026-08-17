"""
Coordinate system implementations for pycocos.
"""

from . import jacobians
from . import registry
from . import field_lines
from . import compute_coordinates
from . import jacobian_builders
from . import jacobian_numba_kernels
from . import numba_runtime

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
from .accuracy import (
    CoordinateAccuracy,
    CoordinateAccuracyProfile,
    resolve_coordinate_accuracy,
)
from .checkpoint import (
    CheckpointIntegrityError,
    CheckpointMismatchError,
    CoordinateCheckpointError,
    LoadedCoordinateCheckpoint,
    coordinate_checkpoint_key,
    coordinate_checkpoint_path,
    load_coordinate_checkpoint,
    write_coordinate_checkpoint,
)
from .product import MagneticCoordinateMapProduct

__all__ = [
    "jacobians",
    "registry",
    "field_lines",
    "compute_coordinates",
    "jacobian_builders",
    "jacobian_numba_kernels",
    "numba_runtime",
    "get_jacobian_function",
    "register_coordinate_system",
    "list_coordinate_systems",
    "JACOBIAN_REGISTRY",
    "get_field_line",
    "integrate_pol_field_line",
    "compute_magnetic_coordinates",
    "CoordinateAccuracy",
    "CoordinateAccuracyProfile",
    "resolve_coordinate_accuracy",
    "MagneticCoordinateMapProduct",
    "CoordinateCheckpointError",
    "CheckpointIntegrityError",
    "CheckpointMismatchError",
    "LoadedCoordinateCheckpoint",
    "coordinate_checkpoint_key",
    "coordinate_checkpoint_path",
    "load_coordinate_checkpoint",
    "write_coordinate_checkpoint",
]
