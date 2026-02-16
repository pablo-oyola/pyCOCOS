"""
Registry for coordinate system names and their Jacobian functions.

This module provides a registry mapping coordinate system names to their
corresponding Jacobian computation functions.
"""

from typing import Dict, Callable, Any
from .jacobians import (
    compute_boozer_jacobian,
    compute_hamada_jacobian,
    compute_pest_jacobian,
    compute_equal_arc_jacobian,
)

# Registry mapping coordinate system names to Jacobian functions
JACOBIAN_REGISTRY: Dict[str, Callable] = {
    'boozer': compute_boozer_jacobian,
    'hamada': compute_hamada_jacobian,
    'pest': compute_pest_jacobian,
    'equal_arc': compute_equal_arc_jacobian,
}


def get_jacobian_function(name: str) -> Callable:
    """
    Get the Jacobian computation function for a coordinate system.

    Parameters
    ----------
    name : str
        Name of the coordinate system ('boozer', 'hamada', 'pest', 'equal_arc')

    Returns
    -------
    Callable
        Jacobian computation function

    Raises
    ------
    ValueError
        If the coordinate system name is not recognized
    """
    name_lower = name.lower()
    if name_lower not in JACOBIAN_REGISTRY:
        available = ', '.join(JACOBIAN_REGISTRY.keys())
        raise ValueError(
            f"Unknown coordinate system '{name}'. "
            f"Available systems: {available}"
        )
    return JACOBIAN_REGISTRY[name_lower]


def register_coordinate_system(name: str, jacobian_func: Callable) -> None:
    """
    Register a new coordinate system.

    Parameters
    ----------
    name : str
        Name of the coordinate system
    jacobian_func : Callable
        Function that computes the Jacobian J(psi, theta)
        Signature: jacobian_func(I, F, q, B, **kwargs) -> J
    """
    JACOBIAN_REGISTRY[name.lower()] = jacobian_func


def list_coordinate_systems() -> list:
    """
    List all available coordinate systems.

    Returns
    -------
    list
        List of coordinate system names
    """
    return list(JACOBIAN_REGISTRY.keys())
