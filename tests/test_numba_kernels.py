from time import perf_counter

import numpy as np

from pycocos.coordinates import jacobian_numba_kernels as kernels


def _python_grad_psi(R: np.ndarray, bpol: np.ndarray) -> np.ndarray:
    out = np.abs(R * bpol)
    out[out < kernels.EPS] = kernels.EPS
    return out


def _python_power_law(
    R: np.ndarray,
    grad_psi: np.ndarray,
    B: np.ndarray,
    i_power: int,
    j_power: int,
    k_power: int,
) -> np.ndarray:
    out = np.ones_like(B, dtype=np.float64)
    if i_power:
        out = out * (R**i_power)
    if j_power:
        out = out / (grad_psi**j_power)
    if k_power:
        out = out / (np.abs(B) ** k_power)
    return out


def test_numba_jacobian_kernels_match_python_reference():
    n = 4096
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R = 1.7 + 0.2 * np.cos(theta)
    B = 2.3 + 0.3 * np.sin(theta)
    Bpol = 0.65 + 0.05 * np.cos(3.0 * theta)
    dlp = np.full(n, 2.0 * np.pi / n)

    grad_numba = kernels.compute_grad_psi_abs(R, Bpol)
    grad_py = _python_grad_psi(R, Bpol)
    assert np.allclose(grad_numba, grad_py)

    jac_numba = kernels.build_power_law_jacobian(R, grad_numba, B, 1, 1, 0, 1.0)
    jac_py = _python_power_law(R, grad_py, B, 1, 1, 0)
    assert np.allclose(jac_numba, jac_py)

    span_numba = kernels.compute_theta_span(R, jac_numba, grad_numba, dlp)
    span_py = np.sum(R / (np.abs(jac_py) * grad_py) * dlp)
    assert np.isfinite(span_numba)
    assert np.allclose(span_numba, span_py)

    scaled = kernels.apply_scalar_scale(jac_numba, span_numba / (2.0 * np.pi))
    assert np.allclose(scaled, jac_numba * (span_numba / (2.0 * np.pi)))

    # Smoke-check that njit kernels were actually compiled.
    assert kernels.compute_grad_psi_abs.signatures
    assert kernels.build_power_law_jacobian.signatures
    assert kernels.compute_theta_span.signatures


def test_numba_hot_path_regression_guard():
    """
    Coarse regression guard: repeated kernel calls should not become dramatically
    slower than the first measured call.
    """
    n = 250_000
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R = 1.9 + 0.3 * np.cos(theta)
    B = 2.0 + 0.2 * np.sin(theta)
    Bpol = 0.55 + 0.08 * np.cos(2.0 * theta)
    grad = kernels.compute_grad_psi_abs(R, Bpol)

    t0 = perf_counter()
    _ = kernels.build_power_law_jacobian(R, grad, B, 2, 0, 0, 1.0)
    t1 = perf_counter() - t0

    t0 = perf_counter()
    _ = kernels.build_power_law_jacobian(R, grad, B, 2, 0, 0, 1.0)
    t2 = perf_counter() - t0

    assert t2 <= 2.0 * t1

