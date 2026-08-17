"""
Jacobian builders and validation for magnetic coordinate systems.

Built-in power-family builders return only their raw surface shape. A single
explicit helper, :func:`normalize_jacobian_to_two_pi`, owns normalization of
that shape to a closed ``2*pi`` poloidal angle.
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

# Raw shapes J = R^i / |grad(psi)|^j / B^k (PDF Eq. 8.115 family).
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
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Context entry '{name}' must contain only finite values")
    return arr


def _as_finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Context entry '{name}' must be finite")
    return result


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
    """Build a validated Jacobian context for one flux surface."""
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
        "coordinate_system": str(coordinate_system).lower(),
        "R": rvals,
        "B": bvals,
        "Bpol": bpol_vals,
        "dlp": dlp_vals,
        "I": _as_finite_scalar(I, "I"),
        "F": _as_finite_scalar(F, "F"),
        "q": _as_finite_scalar(q, "q"),
    }


def validate_jacobian(
    context: Mapping[str, Any],
    jacobian: np.ndarray,
) -> np.ndarray:
    """
    Validate a per-surface Jacobian without changing its magnitude or sign.

    A valid Jacobian is a finite one-dimensional array with exactly the surface
    shape, is bounded away from zero, and has a single orientation everywhere.
    """
    bvals = _as_float64_1d(context["B"], "B")
    values = np.ascontiguousarray(np.asarray(jacobian, dtype=np.float64))

    if values.ndim != 1 or values.shape != bvals.shape:
        raise ValueError(
            "Jacobian must be a 1D array matching context['B']; "
            f"got shape {values.shape}, expected {bvals.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Jacobian must contain only finite values")
    if np.any(np.abs(values) <= EPS):
        raise ValueError(
            f"Jacobian must be nonzero with abs(J) > {EPS:.1e} everywhere"
        )

    has_positive = bool(np.any(values > 0.0))
    has_negative = bool(np.any(values < 0.0))
    if has_positive and has_negative:
        raise ValueError("Jacobian must have one sign over the complete surface")

    return values


def normalize_jacobian_to_two_pi(
    context: Mapping[str, Any],
    jacobian: np.ndarray,
) -> np.ndarray:
    """
    Normalize a raw Jacobian shape to one ``2*pi`` poloidal circuit.

    The angle increment is

    ``dtheta = R / (|J| |grad(psi)|) dlp``.

    Multiplying an input shape by any positive surface constant therefore
    yields the same normalized Jacobian. The sign is retained as the coordinate
    orientation. Boozer's physical ``(I + q*F)/B**2`` Jacobian is already
    normalized analytically and should not be passed through this helper.
    """
    values = validate_jacobian(context, jacobian)
    rvals = _as_float64_1d(context["R"], "R")
    bpol_vals = _as_float64_1d(context["Bpol"], "Bpol")
    dlp_vals = _as_float64_1d(context["dlp"], "dlp")

    expected_shape = values.shape
    for name, array in (
        ("R", rvals),
        ("Bpol", bpol_vals),
        ("dlp", dlp_vals),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"Context entry '{name}' has shape {array.shape}, "
                f"expected {expected_shape}"
            )

    if np.any(rvals <= EPS):
        raise ValueError("Context entry 'R' must be positive on the surface")
    if np.any(np.abs(bpol_vals) <= EPS):
        raise ValueError(
            "Context entry 'Bpol' must be nonzero on the surface "
            "to normalize a Jacobian"
        )
    if np.any(dlp_vals <= 0.0):
        raise ValueError("Context entry 'dlp' must be positive on the surface")

    ensure_numba_runtime_ready()
    grad_psi = compute_grad_psi_abs(rvals, bpol_vals)
    span = compute_theta_span(rvals, values, grad_psi, dlp_vals)
    if not np.isfinite(span) or span <= EPS:
        raise ValueError(
            "Jacobian shape does not define a finite, nonzero poloidal-angle span"
        )

    normalized = apply_scalar_scale(values, span / (2.0 * np.pi))
    return validate_jacobian(context, normalized)


def build_boozer_jacobian_from_context(context: Mapping[str, Any]) -> np.ndarray:
    """
    Build the physical Boozer Jacobian ``J = (I + q*F) / B**2``.

    Its surface factor is fixed by the equilibrium profiles, so no numerical
    span normalization is applied.
    """
    ensure_numba_runtime_ready()
    h_val = (
        _as_finite_scalar(context["I"], "I")
        + _as_finite_scalar(context["q"], "q")
        * _as_finite_scalar(context["F"], "F")
    )
    bvals = _as_float64_1d(context["B"], "B")
    return validate_jacobian(context, build_boozer_jacobian(bvals, h_val))


def boozer_consistency_residual(
    context: Mapping[str, Any],
    jacobian: np.ndarray,
) -> float:
    """
    Return ``max_theta |J*B**2 - (I + q*F)|`` for a Boozer Jacobian.
    """
    values = validate_jacobian(context, jacobian)
    h_val = (
        _as_finite_scalar(context["I"], "I")
        + _as_finite_scalar(context["q"], "q")
        * _as_finite_scalar(context["F"], "F")
    )
    b2 = _as_float64_1d(context["B"], "B") ** 2
    return float(np.max(np.abs(values * b2 - h_val)))


def build_power_family_jacobian_from_context(
    context: Mapping[str, Any],
    *,
    i_power: int,
    j_power: int,
    k_power: int,
) -> np.ndarray:
    """
    Build the raw shape ``R^i / |grad(psi)|^j / B^k``.

    This function deliberately does not normalize the shape. Coordinate
    assembly must call :func:`normalize_jacobian_to_two_pi` exactly once.
    """
    ensure_numba_runtime_ready()
    rvals = _as_float64_1d(context["R"], "R")
    bpol_vals = _as_float64_1d(context["Bpol"], "Bpol")
    bvals = _as_float64_1d(context["B"], "B")
    if rvals.shape != bvals.shape or bpol_vals.shape != bvals.shape:
        raise ValueError("Context arrays R/B/Bpol must have matching shapes")

    grad_psi = compute_grad_psi_abs(rvals, bpol_vals)
    raw = build_power_law_jacobian(
        rvals,
        grad_psi,
        bvals,
        i_power=i_power,
        j_power=j_power,
        k_power=k_power,
        prefactor=1.0,
    )
    return validate_jacobian(context, raw)


def build_coordinate_jacobian(context: Mapping[str, Any]) -> np.ndarray:
    """Build a physical Boozer Jacobian or a raw power-family shape."""
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
