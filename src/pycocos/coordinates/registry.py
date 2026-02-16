"""
Registry for coordinate-system Jacobian callables.
"""

from __future__ import annotations

from inspect import signature
from typing import Any, Callable, Dict, Mapping

import numpy as np

from .jacobians import (
    compute_boozer_jacobian,
    compute_equal_arc_jacobian,
    compute_hamada_jacobian,
    compute_pest_jacobian,
)


def _is_context_callable(func: Callable) -> bool:
    return bool(getattr(func, "_pycocos_context_api", False))


def _wrap_legacy_callable(func: Callable) -> Callable:
    """
    Adapt a legacy jacobian callable with signature (I, F, q, B, **kwargs)
    into context API:
        func(context) -> J
    """

    def context_adapter(context: Mapping[str, Any]) -> np.ndarray:
        I = np.array([float(context["I"])], dtype=np.float64)
        F = np.array([float(context["F"])], dtype=np.float64)
        q = np.array([float(context["q"])], dtype=np.float64)
        B = np.ascontiguousarray(np.asarray(context["B"], dtype=np.float64))
        out = func(I, F, q, B)
        return np.asarray(out, dtype=np.float64)

    context_adapter._pycocos_context_api = True  # type: ignore[attr-defined]
    context_adapter.__name__ = f"{func.__name__}_context_adapter"
    return context_adapter


def _ensure_context_callable(func: Callable) -> Callable:
    if _is_context_callable(func):
        return func

    # Heuristic for legacy callables.
    params = list(signature(func).parameters.values())
    if len(params) >= 4:
        return _wrap_legacy_callable(func)

    raise TypeError(
        "Jacobian callable must be context-based (func(context)) or legacy "
        "signature (func(I, F, q, B, ...))."
    )


JACOBIAN_REGISTRY: Dict[str, Callable] = {
    "boozer": compute_boozer_jacobian,
    "hamada": compute_hamada_jacobian,
    "pest": compute_pest_jacobian,
    "equal_arc": compute_equal_arc_jacobian,
}


def get_jacobian_function(name: str) -> Callable:
    """
    Get a context-based Jacobian function for a coordinate system.
    """
    name_lower = name.lower()
    if name_lower not in JACOBIAN_REGISTRY:
        available = ", ".join(JACOBIAN_REGISTRY.keys())
        raise ValueError(
            f"Unknown coordinate system '{name}'. "
            f"Available systems: {available}"
        )
    return _ensure_context_callable(JACOBIAN_REGISTRY[name_lower])


def register_coordinate_system(name: str, jacobian_func: Callable) -> None:
    """
    Register a coordinate system Jacobian callable.

    Accepted signatures:
    - New API: ``jacobian_func(context) -> J``
    - Legacy API: ``jacobian_func(I, F, q, B, **kwargs) -> J``
    """
    JACOBIAN_REGISTRY[name.lower()] = _ensure_context_callable(jacobian_func)


def list_coordinate_systems() -> list:
    """
    List all available coordinate systems.
    """
    return list(JACOBIAN_REGISTRY.keys())

