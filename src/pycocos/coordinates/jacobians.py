"""
Jacobian computation functions for magnetic coordinate systems.

Primary API (used by pycocos internals) is context-based:
    jacobian_func(context_dict) -> J(theta)

Legacy Boozer API is kept for compatibility:
    compute_boozer_jacobian(I, F, q, B) -> J
"""

from __future__ import annotations

from typing import Any, Mapping, Union

import numpy as np

from .jacobian_builders import build_coordinate_jacobian, build_boozer_jacobian_from_context


EPS = 1.0e-14


def _is_context_payload(arg: Any) -> bool:
    return isinstance(arg, Mapping) and "B" in arg and "coordinate_system" in arg


def _broadcast_profile_to_b(profile: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Broadcast scalar/1D/2D profile to B shape for legacy Boozer API.
    """
    if profile.ndim == 0:
        return np.full_like(B, float(profile))

    if profile.ndim == 1:
        if B.ndim == 2 and profile.size == B.shape[0]:
            return profile[:, np.newaxis]
        if profile.size == B.size:
            return profile.reshape(B.shape)
        if profile.size > 0:
            return np.full_like(B, profile[0])
        return np.zeros_like(B)

    if profile.shape == B.shape:
        return profile
    return np.broadcast_to(profile, B.shape)


def _legacy_boozer_jacobian(
    I: Union[np.ndarray, float],
    F: Union[np.ndarray, float],
    q: Union[np.ndarray, float],
    B: np.ndarray,
) -> np.ndarray:
    """
    Backward-compatible Boozer implementation:
      J = (I + qF) / B^2
    """
    I_arr = np.asarray(I, dtype=np.float64)
    F_arr = np.asarray(F, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)

    I_b = _broadcast_profile_to_b(I_arr, B_arr)
    F_b = _broadcast_profile_to_b(F_arr, B_arr)
    q_b = _broadcast_profile_to_b(q_arr, B_arr)

    B2 = B_arr * B_arr
    B2 = np.where(B2 < EPS, EPS, B2)
    return (I_b + q_b * F_b) / B2


def compute_boozer_jacobian(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Compute Boozer coordinate Jacobian.

    Supported signatures
    --------------------
    - Context API: ``compute_boozer_jacobian(context)``
      where context contains at least:
      ``coordinate_system``, ``R``, ``B``, ``Bpol``, ``dlp``, ``I``, ``F``, ``q``.
    - Legacy API: ``compute_boozer_jacobian(I, F, q, B)``
    """
    if len(args) == 1 and _is_context_payload(args[0]):
        return build_boozer_jacobian_from_context(args[0])

    if len(args) >= 4:
        return _legacy_boozer_jacobian(args[0], args[1], args[2], args[3])

    raise TypeError(
        "compute_boozer_jacobian expects either a context payload or "
        "legacy arguments (I, F, q, B)."
    )


def compute_hamada_jacobian(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Compute Hamada coordinate Jacobian from context.

    Requires context-based API because Hamada construction depends on geometric
    quantities beyond (I, F, q, B).
    """
    if len(args) == 1 and _is_context_payload(args[0]):
        return build_coordinate_jacobian(args[0])

    raise TypeError(
        "compute_hamada_jacobian requires a context payload. "
        "Use get_jacobian_function('hamada') inside compute_coordinates."
    )


def compute_pest_jacobian(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Compute PEST coordinate Jacobian from context.
    """
    if len(args) == 1 and _is_context_payload(args[0]):
        return build_coordinate_jacobian(args[0])

    raise TypeError(
        "compute_pest_jacobian requires a context payload. "
        "Use get_jacobian_function('pest') inside compute_coordinates."
    )


def compute_equal_arc_jacobian(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Compute equal-arc coordinate Jacobian from context.
    """
    if len(args) == 1 and _is_context_payload(args[0]):
        return build_coordinate_jacobian(args[0])

    raise TypeError(
        "compute_equal_arc_jacobian requires a context payload. "
        "Use get_jacobian_function('equal_arc') inside compute_coordinates."
    )


# Mark built-ins as context-aware for registry adapters.
compute_boozer_jacobian._pycocos_context_api = True  # type: ignore[attr-defined]
compute_hamada_jacobian._pycocos_context_api = True  # type: ignore[attr-defined]
compute_pest_jacobian._pycocos_context_api = True  # type: ignore[attr-defined]
compute_equal_arc_jacobian._pycocos_context_api = True  # type: ignore[attr-defined]

