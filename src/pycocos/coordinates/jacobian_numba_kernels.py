"""
Numba kernels for Jacobian construction hot paths.

These functions are intentionally array-centric and object-free so they can be
used inside performance-critical loops.
"""

import numba as nb
import numpy as np


EPS = 1.0e-14


@nb.njit(nogil=True)
def compute_grad_psi_abs(R: np.ndarray, bpol: np.ndarray) -> np.ndarray:
    """
    Compute |grad(psi)| using axisymmetric identity:
        |grad(psi)| = R * Bpol
    """
    out = np.empty(R.size, dtype=np.float64)
    for i in range(R.size):
        val = abs(R[i] * bpol[i])
        if val < EPS:
            val = EPS
        out[i] = val
    return out


@nb.njit(nogil=True)
def build_boozer_jacobian(B: np.ndarray, h: float) -> np.ndarray:
    """
    Build Boozer Jacobian: J = h / B^2.
    """
    out = np.empty(B.size, dtype=np.float64)
    for i in range(B.size):
        b2 = B[i] * B[i]
        if b2 < EPS:
            b2 = EPS
        out[i] = h / b2
    return out


@nb.njit(nogil=True)
def build_power_law_jacobian(
    R: np.ndarray,
    grad_psi: np.ndarray,
    B: np.ndarray,
    i_power: int,
    j_power: int,
    k_power: int,
    prefactor: float = 1.0,
) -> np.ndarray:
    """
    Build Jacobian from power-law family:
        J = prefactor * R^i / |grad(psi)|^j / B^k
    """
    out = np.empty(B.size, dtype=np.float64)
    for idx in range(B.size):
        value = prefactor
        if i_power != 0:
            value *= R[idx] ** i_power

        if j_power != 0:
            gp = grad_psi[idx]
            if gp < EPS:
                gp = EPS
            value /= gp ** j_power

        if k_power != 0:
            babs = abs(B[idx])
            if babs < EPS:
                babs = EPS
            value /= babs ** k_power

        out[idx] = value
    return out


@nb.njit(nogil=True)
def compute_theta_span(
    R: np.ndarray,
    jacobian: np.ndarray,
    grad_psi: np.ndarray,
    dlp: np.ndarray,
) -> float:
    """
    Compute unnormalized poloidal-angle span implied by Jacobian:
        span = integral ( R / (|J| * |grad(psi)|) ) dlp
    """
    span = 0.0
    for i in range(jacobian.size):
        denom = abs(jacobian[i]) * grad_psi[i]
        if denom < EPS:
            denom = EPS
        span += (R[i] / denom) * dlp[i]
    return span


@nb.njit(nogil=True)
def apply_scalar_scale(values: np.ndarray, scale: float) -> np.ndarray:
    """
    Multiply a vector by a scalar and return a new array.
    """
    out = np.empty(values.size, dtype=np.float64)
    for i in range(values.size):
        out[i] = values[i] * scale
    return out

