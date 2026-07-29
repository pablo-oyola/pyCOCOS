"""Axisymmetric magnetic-coordinate construction.

Flux surfaces are traced independently, optionally projected back onto the
requested physical poloidal flux, and then reparameterized by a registered
Jacobian.  No radial smoothing or symmetry projection is performed.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from .field_lines import integrate_pol_field_line
from .jacobian_builders import (
    boozer_consistency_residual,
    make_jacobian_context,
    normalize_jacobian_to_two_pi,
    validate_jacobian,
)
from .jacobians import compute_boozer_jacobian
from .surfaces import (
    build_flux_constrained_surfaces,
    canonicalize_contour_samples,
    resample_closed_contour_by_arclength,
)


_TWO_PI = 2.0 * np.pi
_DEFAULT_SPECTRAL_MAX_FOURIER_MODE = 16
_THETA_GEOM_POINTS = 7200


def _validate_spectral_max_mode(spectral_max_mode: int) -> int:
    if isinstance(spectral_max_mode, bool) or not isinstance(
        spectral_max_mode,
        (int, np.integer),
    ):
        raise TypeError("spectral_max_mode must be an integer.")
    result = int(spectral_max_mode)
    if result < 1:
        raise ValueError("spectral_max_mode must be >= 1.")
    return result


def _resample_trace_values(
    Rline: np.ndarray,
    zline: np.ndarray,
    values: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """Resample a traced periodic field on the contour arclength grid."""
    radial = np.asarray(Rline, dtype=np.float64)
    vertical = np.asarray(zline, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    scale = max(1.0, float(np.ptp(radial)), float(np.ptp(vertical)))
    if np.hypot(
        radial[-1] - radial[0],
        vertical[-1] - vertical[0],
    ) <= 1.0e-10 * scale:
        radial = radial[:-1]
        vertical = vertical[:-1]
        data = data[:-1]
    closed_R = np.append(radial, radial[0])
    closed_z = np.append(vertical, vertical[0])
    closed_values = np.append(data, data[0])
    arclength = np.concatenate(
        ([0.0], np.cumsum(np.hypot(np.diff(closed_R), np.diff(closed_z))))
    )
    target = np.linspace(0.0, arclength[-1], target_size, endpoint=False)
    return np.interp(target, arclength, closed_values)


def _trace_flux_surfaces(
    *,
    Rgrid: np.ndarray,
    zgrid: np.ndarray,
    br: np.ndarray,
    bz: np.ndarray,
    bphi: np.ndarray,
    R_at_psi: np.ndarray,
    zaxis: float,
    ntheta: int,
    integration_sign: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace and arclength-resample full closed poloidal field lines."""
    npsi = len(R_at_psi)
    outputs = [
        np.empty((npsi, ntheta), dtype=np.float64)
        for _ in range(5)
    ]
    for index, seed_R in enumerate(R_at_psi):
        traced = integrate_pol_field_line(
            Rgrid,
            zgrid,
            br,
            bz,
            bphi,
            float(seed_R),
            float(zaxis),
            integration_sign=float(integration_sign),
        )
        Rline, zline, brline, bzline, bphiline, count = traced
        if count < 8:
            raise ValueError(
                f"Flux-surface tracing returned only {count} points at index {index}."
            )
        Rline = np.asarray(Rline[:count], dtype=np.float64)
        zline = np.asarray(zline[:count], dtype=np.float64)
        brline = np.asarray(brline[:count], dtype=np.float64)
        bzline = np.asarray(bzline[:count], dtype=np.float64)
        bphiline = np.asarray(bphiline[:count], dtype=np.float64)
        R_surface, z_surface = resample_closed_contour_by_arclength(
            Rline,
            zline,
            ntheta,
        )
        br_surface = _resample_trace_values(
            Rline,
            zline,
            brline,
            ntheta,
        )
        bz_surface = _resample_trace_values(
            Rline,
            zline,
            bzline,
            ntheta,
        )
        bphi_surface = _resample_trace_values(
            Rline,
            zline,
            bphiline,
            ntheta,
        )
        (
            R_surface,
            z_surface,
            br_surface,
            bz_surface,
            bphi_surface,
        ) = canonicalize_contour_samples(
            R_surface,
            z_surface,
            br_surface,
            bz_surface,
            bphi_surface,
            gauge_z=zaxis,
        )
        outputs[0][index] = R_surface
        outputs[1][index] = z_surface
        outputs[2][index] = br_surface
        outputs[3][index] = bz_surface
        outputs[4][index] = bphi_surface
    return tuple(outputs)  # type: ignore[return-value]


def _evaluate_field_interpolator(
    interpolator,
    R: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    if interpolator is None:
        raise ValueError("A field interpolator is required for this surface.")
    if hasattr(interpolator, "ev"):
        return np.asarray(interpolator.ev(R, z), dtype=np.float64)
    points = np.column_stack((R, z))
    return np.asarray(interpolator(points), dtype=np.float64)


def _sample_surface_fields(
    R: np.ndarray,
    z: np.ndarray,
    br_surface: Optional[np.ndarray],
    bz_surface: Optional[np.ndarray],
    bphi_surface: Optional[np.ndarray],
    br_interp,
    bz_interp,
    bphi_interp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if br_surface is not None and bz_surface is not None and bphi_surface is not None:
        return (
            np.asarray(br_surface, dtype=np.float64),
            np.asarray(bz_surface, dtype=np.float64),
            np.asarray(bphi_surface, dtype=np.float64),
        )
    return (
        _evaluate_field_interpolator(br_interp, R, z),
        _evaluate_field_interpolator(bz_interp, R, z),
        _evaluate_field_interpolator(bphi_interp, R, z),
    )


def _compute_surface_coordinate_row(
    R: np.ndarray,
    z: np.ndarray,
    br_surface: Optional[np.ndarray],
    bz_surface: Optional[np.ndarray],
    bphi_surface: Optional[np.ndarray],
    thetageom: np.ndarray,
    theta_eval: np.ndarray,
    thgeogrid: np.ndarray,
    thmaggrid: np.ndarray,
    coordinate_system: str,
    jacobian_func: Callable,
    br_interp,
    bz_interp,
    bphi_interp,
) -> Tuple[
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Construct one magnetic-coordinate row from a closed surface."""
    del theta_eval  # Retained in the public helper signature for direct callers.
    radial = np.asarray(R, dtype=np.float64)
    vertical = np.asarray(z, dtype=np.float64)
    if radial.ndim != 1 or vertical.shape != radial.shape:
        raise ValueError("R and z must be matching one-dimensional surface arrays.")
    if thetageom.size != radial.size + 1:
        raise ValueError("thetageom must contain one periodic endpoint.")

    br_vals, bz_vals, bphi_vals = _sample_surface_fields(
        radial,
        vertical,
        br_surface,
        bz_surface,
        bphi_surface,
        br_interp,
        bz_interp,
        bphi_interp,
    )
    if any(values.shape != radial.shape for values in (br_vals, bz_vals, bphi_vals)):
        raise ValueError("Surface magnetic-field arrays must match the geometry.")

    dR = np.roll(radial, -1) - radial
    dz = np.roll(vertical, -1) - vertical
    dlp = np.hypot(dR, dz)
    if np.any(dlp <= 0.0):
        raise ValueError("Flux surface contains a zero-length segment.")

    Bpol = np.hypot(br_vals, bz_vals)
    if np.any(Bpol <= 1.0e-14):
        raise ValueError("Poloidal magnetic field vanishes on a retained surface.")
    B = np.sqrt(Bpol**2 + bphi_vals**2)

    field_aligned_increment = (dR * br_vals + dz * bz_vals) / Bpol
    orientation = float(np.sign(np.sum(field_aligned_increment)))
    if orientation == 0.0:
        raise ValueError("Unable to determine the poloidal surface orientation.")
    signed_dlp = orientation * dlp

    Iprof = float(np.sum(Bpol * signed_dlp) / _TWO_PI)
    vertex_weight = 0.5 * (np.roll(dlp, 1) + dlp)
    Fprof = float(
        np.sum(radial * bphi_vals * vertex_weight)
        / np.sum(vertex_weight)
    )
    qprof = float(
        np.sum(
            signed_dlp * Fprof / (radial**2 * Bpol)
        )
        / _TWO_PI
    )

    context = make_jacobian_context(
        coordinate_system=coordinate_system,
        R=radial,
        B=B,
        Bpol=Bpol,
        dlp=dlp,
        I=Iprof,
        F=Fprof,
        q=qprof,
    )
    raw_jacobian = np.asarray(jacobian_func(context), dtype=np.float64)
    if coordinate_system.lower() == "boozer":
        jacobian = validate_jacobian(context, raw_jacobian)
        residual = boozer_consistency_residual(context, jacobian)
        reference = max(1.0, abs(Iprof + qprof * Fprof))
        if residual > 1.0e-10 * reference:
            raise ValueError(
                "Boozer Jacobian consistency check failed: "
                f"residual={residual:.3e}."
            )
        if np.sign(jacobian[0]) != orientation:
            raise ValueError(
                "Boozer Jacobian sign is inconsistent with surface orientation."
            )
    else:
        jacobian = normalize_jacobian_to_two_pi(context, raw_jacobian)
        if np.sign(jacobian[0]) != orientation:
            jacobian = -jacobian

    theta_increment = signed_dlp / (jacobian * Bpol)
    if np.any(theta_increment <= 0.0):
        raise ValueError("Magnetic angle is not monotonic around the surface.")
    theta_closed = np.concatenate(([0.0], np.cumsum(theta_increment)))
    theta_span = float(theta_closed[-1])
    if not np.isclose(theta_span, _TWO_PI, rtol=2.0e-8, atol=2.0e-10):
        raise ValueError(
            "Jacobian does not close the poloidal angle at 2*pi: "
            f"span={theta_span:.16g}."
        )
    # Remove only accumulated floating-point closure error.
    theta_closed *= _TWO_PI / theta_span
    theta_closed[-1] = _TWO_PI

    toroidal_integrand = signed_dlp / (radial**2 * Bpol)
    toroidal_primitive = np.concatenate(
        ([0.0], np.cumsum(toroidal_integrand))
    )
    nu_closed = -Fprof * toroidal_primitive + qprof * theta_closed
    nu_closed[-1] = nu_closed[0]

    surface_parameter = np.linspace(
        0.0,
        _TWO_PI,
        radial.size + 1,
    )
    theta_direct = np.interp(thgeogrid, surface_parameter, theta_closed)

    closed_R = np.append(radial, radial[0])
    closed_z = np.append(vertical, vertical[0])
    R_inverse = np.interp(thmaggrid, theta_closed, closed_R)
    z_inverse = np.interp(thmaggrid, theta_closed, closed_z)
    closed_jacobian = np.append(jacobian, jacobian[0])
    jacobian_direct = np.interp(
        thgeogrid,
        surface_parameter,
        closed_jacobian,
    )
    nu_direct = np.interp(thgeogrid, surface_parameter, nu_closed)
    nu_direct[-1] = nu_direct[0]

    return (
        qprof,
        Fprof,
        Iprof,
        theta_direct,
        nu_direct,
        jacobian_direct,
        R_inverse,
        z_inverse,
    )


def compute_magnetic_coordinates(
    Rgrid: np.ndarray,
    zgrid: np.ndarray,
    br: np.ndarray,
    bz: np.ndarray,
    bphi: np.ndarray,
    raxis: float,
    zaxis: float,
    psigrid: np.ndarray,
    ltheta: int = 256,
    phiclockwise: bool = True,
    jacobian_func: Optional[Callable] = None,
    R_at_psi: Optional[np.ndarray] = None,
    coordinate_system: str = "boozer",
    rho_at_psi: Optional[np.ndarray] = None,
    spectral_max_mode: int = _DEFAULT_SPECTRAL_MAX_FOURIER_MODE,
    n_theta_geom: Optional[int] = None,
    psi_field: Optional[np.ndarray] = None,
    flux_scale: Optional[float] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute magnetic coordinates while preserving the historical tuple API.

    ``rho_at_psi`` remains accepted as a public radial-label input, but all
    interpolation and differentiation now use the supplied physical
    ``psigrid``.  When ``psi_field`` is supplied by :class:`Equilibrium`, each
    filtered contour is projected back to its requested physical flux.
    """
    del rho_at_psi
    radial_grid = np.asarray(Rgrid, dtype=np.float64)
    vertical_grid = np.asarray(zgrid, dtype=np.float64)
    Br = np.asarray(br, dtype=np.float64)
    Bz = np.asarray(bz, dtype=np.float64)
    Bphi = np.asarray(bphi, dtype=np.float64)
    radial_flux = np.asarray(psigrid, dtype=np.float64)
    expected_shape = (radial_grid.size, vertical_grid.size)
    for name, values in (("br", Br), ("bz", Bz), ("bphi", Bphi)):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match {expected_shape}."
            )
    if radial_flux.ndim != 1 or radial_flux.size < 2:
        raise ValueError("psigrid must contain at least two physical fluxes.")
    if ltheta < 4:
        raise ValueError("ltheta must be at least four.")
    spectral_max_mode = _validate_spectral_max_mode(spectral_max_mode)
    if jacobian_func is None:
        jacobian_func = compute_boozer_jacobian

    if n_theta_geom is None:
        n_theta_geom = _THETA_GEOM_POINTS
    n_theta_geom = int(n_theta_geom)
    minimum_theta = max(4 * int(ltheta), 2 * spectral_max_mode + 4, 64)
    if n_theta_geom < minimum_theta:
        raise ValueError(
            f"n_theta_geom must be at least {minimum_theta}; got {n_theta_geom}."
        )

    if R_at_psi is None:
        R_at_psi = np.linspace(raxis, radial_grid.max(), radial_flux.size)
    seeds = np.asarray(R_at_psi, dtype=np.float64)
    if seeds.shape != radial_flux.shape:
        raise ValueError("R_at_psi must match psigrid exactly.")

    # B_pol vanishes at the magnetic axis by definition, so its sign cannot
    # determine a contour-tracing orientation there.  Sample the first
    # retained outboard seed instead, where the surface is non-degenerate.
    bz_seed = float(
        RegularGridInterpolator(
            (radial_grid, vertical_grid),
            Bz,
            bounds_error=False,
            fill_value=None,
        )((seeds[0], zaxis))
    )
    integration_sign = np.sign(bz_seed)
    if not phiclockwise:
        integration_sign *= -1.0
    if integration_sign == 0.0:
        raise ValueError(
            "Cannot determine field-line orientation at the first retained "
            "outboard flux-surface seed."
        )

    raw = _trace_flux_surfaces(
        Rgrid=radial_grid,
        zgrid=vertical_grid,
        br=Br,
        bz=Bz,
        bphi=Bphi,
        R_at_psi=seeds,
        zaxis=zaxis,
        ntheta=n_theta_geom,
        integration_sign=integration_sign,
    )
    raw_R, raw_z, raw_Br, raw_Bz, raw_Bphi = raw

    if psi_field is not None:
        if flux_scale is None:
            flux_scale = float(np.ptp(radial_flux))
        surfaces = build_flux_constrained_surfaces(
            Rgrid=radial_grid,
            zgrid=vertical_grid,
            psi_field=np.asarray(psi_field, dtype=np.float64),
            psigrid=radial_flux,
            R_raw=raw_R,
            z_raw=raw_z,
            ntheta=n_theta_geom,
            spectral_max_mode=spectral_max_mode,
            flux_scale=float(flux_scale),
            validate_nesting=True,
            gauge_z=zaxis,
        )
        surface_R = surfaces.R
        surface_z = surfaces.z
        surface_Br = surface_Bz = surface_Bphi = None
    else:
        surface_R = raw_R
        surface_z = raw_z
        surface_Br = raw_Br
        surface_Bz = raw_Bz
        surface_Bphi = raw_Bphi

    spline_orders = (
        min(3, radial_grid.size - 1),
        min(3, vertical_grid.size - 1),
    )
    field_splines = tuple(
        RectBivariateSpline(
            radial_grid,
            vertical_grid,
            values,
            kx=spline_orders[0],
            ky=spline_orders[1],
            s=0.0,
        )
        for values in (Br, Bz, Bphi)
    )

    theta_surface = np.linspace(0.0, _TWO_PI, n_theta_geom, endpoint=False)
    theta_surface_closed = np.linspace(0.0, _TWO_PI, n_theta_geom + 1)
    theta_output = np.linspace(0.0, _TWO_PI, ltheta)
    outputs = [
        np.empty(radial_flux.size, dtype=np.float64)
        for _ in range(3)
    ] + [
        np.empty((radial_flux.size, ltheta), dtype=np.float64)
        for _ in range(5)
    ]

    for index in range(radial_flux.size):
        row = _compute_surface_coordinate_row(
            R=surface_R[index],
            z=surface_z[index],
            br_surface=(
                None if surface_Br is None else surface_Br[index]
            ),
            bz_surface=(
                None if surface_Bz is None else surface_Bz[index]
            ),
            bphi_surface=(
                None if surface_Bphi is None else surface_Bphi[index]
            ),
            thetageom=theta_surface_closed,
            theta_eval=theta_surface,
            thgeogrid=theta_output,
            thmaggrid=theta_output,
            coordinate_system=coordinate_system,
            jacobian_func=jacobian_func,
            br_interp=field_splines[0],
            bz_interp=field_splines[1],
            bphi_interp=field_splines[2],
        )
        for output, value in zip(outputs, row):
            output[index] = value

    return tuple(outputs)  # type: ignore[return-value]
