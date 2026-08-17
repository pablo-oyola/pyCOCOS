"""Compiled object-free kernels for magnetic-coordinate map materialization."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def axisymmetric_differential_kernel(
    R: np.ndarray,
    R_psi: np.ndarray,
    R_theta: np.ndarray,
    z_psi: np.ndarray,
    z_theta: np.ndarray,
    nu_psi: np.ndarray,
    nu_theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build bases, reciprocal metrics, and Jacobian in one fused loop.

    Inputs are flat contiguous ``float64`` arrays.  Keeping this boundary free
    of SciPy and xarray objects makes it suitable both for Numba today and a
    future C++/OpenMP implementation without changing the Python map API.
    """
    point_count = R.size
    direct = np.empty((point_count, 3, 3), dtype=np.float64)
    inverse = np.empty((point_count, 3, 3), dtype=np.float64)
    metric_covariant = np.empty((point_count, 3, 3), dtype=np.float64)
    metric_contravariant = np.empty((point_count, 3, 3), dtype=np.float64)
    jacobian = np.empty(point_count, dtype=np.float64)

    for index in prange(point_count):
        radius = R[index]
        radius_psi = R_psi[index]
        radius_theta = R_theta[index]
        height_psi = z_psi[index]
        height_theta = z_theta[index]
        gauge_psi = nu_psi[index]
        gauge_theta = nu_theta[index]

        direct[index, 0, 0] = radius_psi
        direct[index, 0, 1] = radius_theta
        direct[index, 0, 2] = 0.0
        direct[index, 1, 0] = -radius * gauge_psi
        direct[index, 1, 1] = -radius * gauge_theta
        direct[index, 1, 2] = radius
        direct[index, 2, 0] = height_psi
        direct[index, 2, 1] = height_theta
        direct[index, 2, 2] = 0.0

        poloidal_determinant = (
            radius_psi * height_theta - radius_theta * height_psi
        )
        jacobian[index] = -radius * poloidal_determinant
        if poloidal_determinant == 0.0 or radius == 0.0:
            reciprocal_poloidal = np.nan
            reciprocal_radius = np.nan
        else:
            reciprocal_poloidal = 1.0 / poloidal_determinant
            reciprocal_radius = 1.0 / radius

        gradient_psi_R = height_theta * reciprocal_poloidal
        gradient_psi_z = -radius_theta * reciprocal_poloidal
        gradient_theta_R = -height_psi * reciprocal_poloidal
        gradient_theta_z = radius_psi * reciprocal_poloidal
        gradient_zeta_R = (
            gauge_psi * height_theta - gauge_theta * height_psi
        ) * reciprocal_poloidal
        gradient_zeta_phi = reciprocal_radius
        gradient_zeta_z = (
            gauge_theta * radius_psi - gauge_psi * radius_theta
        ) * reciprocal_poloidal

        inverse[index, 0, 0] = gradient_psi_R
        inverse[index, 0, 1] = 0.0
        inverse[index, 0, 2] = gradient_psi_z
        inverse[index, 1, 0] = gradient_theta_R
        inverse[index, 1, 1] = 0.0
        inverse[index, 1, 2] = gradient_theta_z
        inverse[index, 2, 0] = gradient_zeta_R
        inverse[index, 2, 1] = gradient_zeta_phi
        inverse[index, 2, 2] = gradient_zeta_z

        radius_squared = radius * radius
        covariant_00 = (
            radius_psi * radius_psi
            + height_psi * height_psi
            + radius_squared * gauge_psi * gauge_psi
        )
        covariant_01 = (
            radius_psi * radius_theta
            + height_psi * height_theta
            + radius_squared * gauge_psi * gauge_theta
        )
        covariant_02 = -radius_squared * gauge_psi
        covariant_11 = (
            radius_theta * radius_theta
            + height_theta * height_theta
            + radius_squared * gauge_theta * gauge_theta
        )
        covariant_12 = -radius_squared * gauge_theta
        metric_covariant[index, 0, 0] = covariant_00
        metric_covariant[index, 0, 1] = covariant_01
        metric_covariant[index, 0, 2] = covariant_02
        metric_covariant[index, 1, 0] = covariant_01
        metric_covariant[index, 1, 1] = covariant_11
        metric_covariant[index, 1, 2] = covariant_12
        metric_covariant[index, 2, 0] = covariant_02
        metric_covariant[index, 2, 1] = covariant_12
        metric_covariant[index, 2, 2] = radius_squared

        contravariant_00 = (
            gradient_psi_R * gradient_psi_R
            + gradient_psi_z * gradient_psi_z
        )
        contravariant_01 = (
            gradient_psi_R * gradient_theta_R
            + gradient_psi_z * gradient_theta_z
        )
        contravariant_02 = (
            gradient_psi_R * gradient_zeta_R
            + gradient_psi_z * gradient_zeta_z
        )
        contravariant_11 = (
            gradient_theta_R * gradient_theta_R
            + gradient_theta_z * gradient_theta_z
        )
        contravariant_12 = (
            gradient_theta_R * gradient_zeta_R
            + gradient_theta_z * gradient_zeta_z
        )
        contravariant_22 = (
            gradient_zeta_R * gradient_zeta_R
            + gradient_zeta_phi * gradient_zeta_phi
            + gradient_zeta_z * gradient_zeta_z
        )
        metric_contravariant[index, 0, 0] = contravariant_00
        metric_contravariant[index, 0, 1] = contravariant_01
        metric_contravariant[index, 0, 2] = contravariant_02
        metric_contravariant[index, 1, 0] = contravariant_01
        metric_contravariant[index, 1, 1] = contravariant_11
        metric_contravariant[index, 1, 2] = contravariant_12
        metric_contravariant[index, 2, 0] = contravariant_02
        metric_contravariant[index, 2, 1] = contravariant_12
        metric_contravariant[index, 2, 2] = contravariant_22

    return (
        direct,
        inverse,
        metric_covariant,
        metric_contravariant,
        jacobian,
    )
