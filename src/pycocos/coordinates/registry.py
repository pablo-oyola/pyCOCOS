"""
Registry for coordinate-system Jacobian callables.
"""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable, Dict, Mapping, TypeAlias

from .jacobians import (
    compute_boozer_jacobian,
    compute_equal_arc_jacobian,
    compute_hamada_jacobian,
    compute_pest_jacobian,
)


JacobianCallable: TypeAlias = Callable[[Mapping[str, Any]], object]


def _require_context_callable(func: Callable) -> JacobianCallable:
    """Require the sole supported Jacobian API: ``func(context)``."""
    if not callable(func):
        raise TypeError("Jacobian callable must be callable.")

    try:
        parameters = tuple(signature(func).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Jacobian callable must have an inspectable signature "
            "of exactly func(context)."
        ) from exc

    if (
        len(parameters) != 1
        or parameters[0].kind
        not in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
    ):
        raise TypeError(
            "Jacobian callable must have exactly one positional parameter: "
            "func(context)."
        )

    return func


JACOBIAN_REGISTRY: Dict[str, JacobianCallable] = {
    "boozer": compute_boozer_jacobian,
    "hamada": compute_hamada_jacobian,
    "pest": compute_pest_jacobian,
    "equal_arc": compute_equal_arc_jacobian,
}


def get_jacobian_function(name: str) -> JacobianCallable:
    """Get a context-based Jacobian function for a coordinate system."""
    name_lower = name.lower()
    if name_lower not in JACOBIAN_REGISTRY:
        available = ", ".join(JACOBIAN_REGISTRY.keys())
        raise ValueError(
            f"Unknown coordinate system '{name}'. "
            f"Available systems: {available}"
        )
    return _require_context_callable(JACOBIAN_REGISTRY[name_lower])


def register_coordinate_system(name: str, jacobian_func: Callable) -> None:
    """
    Register a context-only coordinate-system Jacobian callable.

    The callback must have exactly the signature
    ``jacobian_func(context) -> J``. Jacobian output validation and
    normalization are performed by the coordinate assembly layer.
    """
    JACOBIAN_REGISTRY[name.lower()] = _require_context_callable(jacobian_func)


def list_coordinate_systems() -> list:
    """List all available coordinate systems."""
    return list(JACOBIAN_REGISTRY.keys())
