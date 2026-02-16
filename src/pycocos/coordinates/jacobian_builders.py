"""
Jacobian builders for magnetic coordinate systems.

This module keeps Python orchestration and data validation separate from Numba
kernels used for heavy per-surface numerical work.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from .jacobian_numba_kernels import (
    EPS,
    apply_scalar_scale,
    build_boozer_jacobian,
    build_power_law_jacobian,
    compute_grad_psi_abs,
    compute_theta_span,
)
from .numba_runtime import ensure_numba_runtime_ready


JacobianContext = Dict[str, Any]

# J = R^i / |grad(psi)|^j / B^k (PDF Eq. 8.115 family)
JACOBIAN_EXPONENTS = {
    "pest": (2, 0, 0),
    "equal_arc": (1, 1, 0),
    "hamada": (0, 0, 0),
}


def _as_float64_1d(array_like: Any, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(array_like, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f"Context entry '{name}' must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"Context entry '{name}' cannot be empty")
    return arr


def make_jacobian_context(
    *,
    coordinate_system: str,
    R: np.ndarray,
    B: np.ndarray,
    Bpol: np.ndarray,
    dlp: np.ndarray,
    I: float,
    F: float,
    q: float,
) -> JacobianContext:
    """
    Build a Numba-friendly Jacobian context for a single flux surface.
    """
    rvals = _as_float64_1d(R, "R")
    bvals = _as_float64_1d(B, "B")
    bpol_vals = _as_float64_1d(Bpol, "Bpol")
    dlp_vals = _as_float64_1d(dlp, "dlp")

    npts = bvals.size
    if rvals.size != npts or bpol_vals.size != npts or dlp_vals.size != npts:
        raise ValueError(
            "Context arrays R/B/Bpol/dlp must have matching lengths, got "
            f"{rvals.size}/{bvals.size}/{bpol_vals.size}/{dlp_vals.size}"
        )

    return {
        "coordinate_system": coordinate_system.lower(),
        "R": rvals,
        "B": bvals,
        "Bpol": bpol_vals,
        "dlp": dlp_vals,
        "I": float(I),
        "F": float(F),
        "q": float(q),
    }


def _normalize_jacobian_for_two_pi_span(context: Mapping[str, Any], jacobian: np.ndarray) -> np.ndarray:
    """
    Normalize Jacobian so implied poloidal-angle span is 2*pi.

    Follows Eq. 8.99/8.100 logic through:
      ``dtheta = R / (|J| |grad(psi)|) dlp``
    """
    grad_psi = compute_grad_psi_abs(context["R"], context["Bpol"])
    span = compute_theta_span(context["R"], jacobian, grad_psi, context["dlp"])
    if not np.isfinite(span) or span < EPS:
        return jacobian

    # If theta_raw span is span, then scaling by s=2*pi/span implies J_new = J_old / s.
    scale = span / (2.0 * np.pi)
    return apply_scalar_scale(jacobian, scale)


def build_boozer_jacobian_from_context(context: Mapping[str, Any]) -> np.ndarray:
    """
    Boozer Jacobian:
      J = h(psi) / B^2, with h = I + qF
    """
    ensure_numba_runtime_ready()
    h_val = float(context["I"]) + float(context["q"]) * float(context["F"])
    return build_boozer_jacobian(context["B"], h_val)


def boozer_consistency_residual(context: Mapping[str, Any], jacobian: np.ndarray) -> float:
    """
    Diagnostic residual for Boozer relation:
        ``residual = max_theta |J*B^2 - (I + qF)|``
    """
    h_val = float(context["I"]) + float(context["q"]) * float(context["F"])
    b2 = np.asarray(context["B"], dtype=np.float64) ** 2
    residual = np.max(np.abs(np.asarray(jacobian, dtype=np.float64) * b2 - h_val))
    return float(residual)


def build_power_family_jacobian_from_context(
    context: Mapping[str, Any],
    *,
    i_power: int,
    j_power: int,
    k_power: int,
) -> np.ndarray:
    """
    Build + normalize Jacobian from family
    ``J = R^i / |grad(psi)|^j / B^k``.
    """
    ensure_numba_runtime_ready()
    grad_psi = compute_grad_psi_abs(context["R"], context["Bpol"])
    jac_raw = build_power_law_jacobian(
        context["R"],
        grad_psi,
        context["B"],
        i_power=i_power,
        j_power=j_power,
        k_power=k_power,
        prefactor=1.0,
    )
    return _normalize_jacobian_for_two_pi_span(context, jac_raw)


def build_coordinate_jacobian(context: Mapping[str, Any]) -> np.ndarray:
    """
    Main builder for registered coordinate systems.
    """
    coord = str(context["coordinate_system"]).lower()
    if coord == "boozer":
        return build_boozer_jacobian_from_context(context)

    if coord not in JACOBIAN_EXPONENTS:
        available = ", ".join(["boozer", *sorted(JACOBIAN_EXPONENTS.keys())])
        raise ValueError(
            f"Unknown coordinate system '{coord}' for Jacobian builder. "
            f"Available systems: {available}"
        )

    i_power, j_power, k_power = JACOBIAN_EXPONENTS[coord]
    return build_power_family_jacobian_from_context(
        context,
        i_power=i_power,
        j_power=j_power,
        k_power=k_power,
    )

