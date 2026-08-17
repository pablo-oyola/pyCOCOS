"""
Context-only Jacobian functions for magnetic coordinate systems.

Every public callable in this module follows one API:

    ``jacobian_func(context) -> J(theta)``

Power-family functions return their unnormalized surface shape. The coordinate
assembly layer owns the single normalization to a ``2*pi`` poloidal span.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .jacobian_builders import (
    build_boozer_jacobian_from_context,
    build_coordinate_jacobian,
)


def _require_context(context: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        raise TypeError(
            "Jacobian functions require one context mapping: func(context)."
        )
    return context


def compute_boozer_jacobian(context: Mapping[str, Any]) -> np.ndarray:
    """Compute the physical Boozer Jacobian ``(I + q*F) / B**2``."""
    return build_boozer_jacobian_from_context(_require_context(context))


def compute_hamada_jacobian(context: Mapping[str, Any]) -> np.ndarray:
    """Compute the raw Hamada Jacobian shape from a surface context."""
    return build_coordinate_jacobian(_require_context(context))


def compute_pest_jacobian(context: Mapping[str, Any]) -> np.ndarray:
    """Compute the raw PEST Jacobian shape from a surface context."""
    return build_coordinate_jacobian(_require_context(context))


def compute_equal_arc_jacobian(context: Mapping[str, Any]) -> np.ndarray:
    """Compute the raw equal-arc Jacobian shape from a surface context."""
    return build_coordinate_jacobian(_require_context(context))
