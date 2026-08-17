"""Axisymmetric magnetic-coordinate construction.

Flux surfaces are traced independently, optionally projected back onto the
requested physical poloidal flux, and then reparameterized by a registered
Jacobian. Optional up-down projection symmetrizes the equilibrium flux and
magnetic field before retracing and rebuilding every coordinate quantity.
"""

from __future__ import annotations

from typing import Any, Callable, MutableMapping, Optional, Tuple

import numpy as np
from scipy.interpolate import (
    CubicSpline,
    PchipInterpolator,
    RectBivariateSpline,
    RegularGridInterpolator,
)

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
_DEFAULT_TRACE_STEP = 1.0e-3
_MAX_TRACE_REFINEMENTS = 8
_ANGLE_CLOSURE_RTOL = 1.0e-5


def _centered_segment_integrals(
    values: np.ndarray,
    segment_lengths: np.ndarray,
) -> np.ndarray:
    """Integrate periodic vertex data over each following contour segment."""
    vertex_values = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(segment_lengths, dtype=np.float64)
    if vertex_values.ndim != 1 or lengths.shape != vertex_values.shape:
        raise ValueError(
            "periodic segment quadrature requires matching one-dimensional arrays"
        )
    return 0.5 * (vertex_values + np.roll(vertex_values, -1)) * lengths


def _normalize_magnetic_angle_closure(theta_closed: np.ndarray) -> np.ndarray:
    """Remove small quadrature drift while rejecting a non-closing angle."""
    values = np.asarray(theta_closed, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("closed magnetic angle must be a finite vector.")
    theta_span = float(values[-1])
    if not np.isclose(
        theta_span,
        _TWO_PI,
        rtol=_ANGLE_CLOSURE_RTOL,
        atol=2.0e-10,
    ):
        raise ValueError(
            "Jacobian does not close the poloidal angle at 2*pi: "
            f"span={theta_span:.16g}."
        )
    normalized = values * (_TWO_PI / theta_span)
    normalized[-1] = _TWO_PI
    return normalized


def _up_down_surface_projection(
    R: np.ndarray,
    z: np.ndarray,
    *,
    zaxis: float,
    applied: bool,
    tolerance: Optional[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Audit and optionally project endpoint-exclusive surface geometry."""
    radial = np.asarray(R, dtype=np.float64)
    vertical = np.asarray(z, dtype=np.float64)
    if radial.ndim != 2 or vertical.shape != radial.shape:
        raise ValueError(
            "up-down projection requires matching (surface, angle) geometry"
        )
    reflection = np.concatenate(
        ([0], np.arange(radial.shape[1] - 1, 0, -1))
    )
    reflected_R = radial[:, reflection]
    vertical_offset = vertical - float(zaxis)
    reflected_z = vertical_offset[:, reflection]
    geometry_scale = np.maximum(
        np.maximum(
            np.ptp(radial, axis=1),
            np.ptp(vertical, axis=1),
        ),
        np.finfo(np.float64).tiny,
    )
    R_residual = (
        np.max(np.abs(radial - reflected_R), axis=1) / geometry_scale
    )
    z_residual = (
        np.max(np.abs(vertical_offset + reflected_z), axis=1)
        / geometry_scale
    )
    geometry_residual = np.maximum(R_residual, z_residual)
    projected_R = 0.5 * (radial + reflected_R)
    projected_z_offset = 0.5 * (vertical_offset - reflected_z)
    projected_z = projected_z_offset + float(zaxis)
    field_changes = {
        "R": (
            np.max(np.abs(projected_R - radial), axis=1) / geometry_scale
        ),
        "z": (
            np.max(np.abs(projected_z - vertical), axis=1) / geometry_scale
        ),
    }
    if applied:
        if tolerance is None:
            raise ValueError(
                "symmetry_tolerance is required when up-down projection is enabled"
            )
        if float(np.max(geometry_residual)) > tolerance:
            raise ValueError(
                "flux surfaces are not sufficiently up-down symmetric for "
                "explicit projection: "
                f"residual={float(np.max(geometry_residual)):.3e}, "
                f"tolerance={tolerance:.3e}"
            )
    audit: dict[str, Any] = {
        "applied": bool(applied),
        "tolerance": tolerance,
        "geometry_residual": geometry_residual,
        "field_residuals": {
            "R": R_residual,
            "z": z_residual,
        },
        "field_relative_changes": field_changes,
    }
    if applied:
        return projected_R, projected_z, audit, reflection
    return radial, vertical, audit, reflection


def _project_periodic_field(
    values: np.ndarray,
    *,
    reflection: np.ndarray,
    parity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one endpoint-exclusive surface field onto a parity sector."""
    data = np.asarray(values, dtype=np.float64)
    reflected = data[:, reflection]
    scale = np.maximum(
        np.max(np.abs(data), axis=1),
        np.finfo(np.float64).tiny,
    )
    residual = (
        np.max(np.abs(data - parity * reflected), axis=1) / scale
    )
    projected = 0.5 * (data + parity * reflected)
    relative_change = (
        np.max(np.abs(projected - data), axis=1) / scale
    )
    return projected, residual, relative_change


def _symmetrize_equilibrium_grid(
    *,
    Rgrid: np.ndarray,
    zgrid: np.ndarray,
    zaxis: float,
    Br: np.ndarray,
    Bz: np.ndarray,
    Bphi: np.ndarray,
    psi_field: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return one internally consistent up-down projected equilibrium grid."""
    radial_grid = np.asarray(Rgrid, dtype=np.float64)
    vertical_grid = np.asarray(zgrid, dtype=np.float64)
    reflected_z = 2.0 * float(zaxis) - vertical_grid
    RR, ZZ_reflected = np.meshgrid(
        radial_grid,
        reflected_z,
        indexing="ij",
    )
    # Only the intersection of the original and reflected rectangular domains
    # is physically constrained. Keep exterior-only columns untouched; fitted
    # closed surfaces must remain inside the common domain.
    common_columns = (
        (reflected_z >= vertical_grid[0])
        & (reflected_z <= vertical_grid[-1])
    )

    def project(values: np.ndarray, parity: float) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        spline = RectBivariateSpline(
            radial_grid,
            vertical_grid,
            data,
            kx=min(3, radial_grid.size - 1),
            ky=min(3, vertical_grid.size - 1),
            s=0.0,
        )
        reflected = spline.ev(
            RR.ravel(),
            ZZ_reflected.ravel(),
        ).reshape(data.shape)
        output = data.copy()
        output[:, common_columns] = 0.5 * (
            data[:, common_columns]
            + parity * reflected[:, common_columns]
        )
        return output

    projected_psi = (
        None if psi_field is None else project(psi_field, 1.0)
    )
    return (
        project(Br, -1.0),
        project(Bz, 1.0),
        project(Bphi, 1.0),
        projected_psi,
    )


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
    minimum_points: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace and arclength-resample full closed poloidal field lines."""
    minimum_points = int(minimum_points)
    if minimum_points < 8:
        raise ValueError("minimum_points must be at least eight.")
    npsi = len(R_at_psi)
    outputs = [
        np.empty((npsi, ntheta), dtype=np.float64)
        for _ in range(5)
    ]
    for index, seed_R in enumerate(R_at_psi):
        trace_step = _DEFAULT_TRACE_STEP
        for refinement in range(_MAX_TRACE_REFINEMENTS + 1):
            traced = integrate_pol_field_line(
                Rgrid,
                zgrid,
                br,
                bz,
                bphi,
                float(seed_R),
                float(zaxis),
                tol=trace_step,
                integration_sign=float(integration_sign),
            )
            Rline, zline, brline, bzline, bphiline, count = traced
            if count >= minimum_points:
                break
            trace_step *= 0.5
        else:
            raise ValueError(
                "Flux-surface tracing remained under-resolved at index "
                f"{index}: {count} points after {_MAX_TRACE_REFINEMENTS} "
                f"refinements; required at least {minimum_points}."
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

    br_segment = 0.5 * (br_vals + np.roll(br_vals, -1))
    bz_segment = 0.5 * (bz_vals + np.roll(bz_vals, -1))
    bpol_segment = np.hypot(br_segment, bz_segment)
    field_aligned_increment = (dR * br_segment + dz * bz_segment) / bpol_segment
    orientation = float(np.sign(np.sum(field_aligned_increment)))
    if orientation == 0.0:
        raise ValueError("Unable to determine the poloidal surface orientation.")
    signed_dlp = orientation * dlp

    Iprof = float(
        np.sum(_centered_segment_integrals(Bpol, signed_dlp))
        / _TWO_PI
    )
    vertex_weight = 0.5 * (np.roll(dlp, 1) + dlp)
    Fprof = float(
        np.sum(radial * bphi_vals * vertex_weight)
        / np.sum(vertex_weight)
    )
    toroidal_rate = 1.0 / (radial**2 * Bpol)
    qprof = float(
        Fprof
        * np.sum(_centered_segment_integrals(toroidal_rate, signed_dlp))
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

    theta_rate = 1.0 / (jacobian * Bpol)
    theta_increment = _centered_segment_integrals(
        theta_rate,
        signed_dlp,
    )
    if np.any(theta_increment <= 0.0):
        raise ValueError("Magnetic angle is not monotonic around the surface.")
    theta_closed = np.concatenate(([0.0], np.cumsum(theta_increment)))
    # Remove only accumulated floating-point closure error.
    theta_closed = _normalize_magnetic_angle_closure(theta_closed)

    toroidal_primitive = np.concatenate(
        (
            [0.0],
            np.cumsum(
                _centered_segment_integrals(
                    toroidal_rate,
                    signed_dlp,
                )
            ),
        )
    )
    nu_closed = -Fprof * toroidal_primitive + qprof * theta_closed
    nu_closed[-1] = nu_closed[0]

    surface_parameter = np.linspace(
        0.0,
        _TWO_PI,
        radial.size + 1,
    )
    theta_correction = theta_closed - surface_parameter
    theta_correction[-1] = theta_correction[0]
    theta_direct = thgeogrid + CubicSpline(
        surface_parameter,
        theta_correction,
        bc_type="periodic",
    )(thgeogrid)
    theta_direct[0] = 0.0
    theta_direct[-1] = _TWO_PI
    if np.any(np.diff(theta_direct) <= 0.0):
        # The source primitive is strictly increasing, but an unconstrained
        # periodic cubic can overshoot for a strongly varying yet valid
        # registered Jacobian. Preserve the periodic cubic for smooth maps and
        # fall back to a shape-preserving lifted-angle interpolation only when
        # that overshoot occurs.
        theta_direct = PchipInterpolator(
            surface_parameter,
            theta_closed,
        )(thgeogrid)
        theta_direct[0] = 0.0
        theta_direct[-1] = _TWO_PI
        if np.any(np.diff(theta_direct) <= 0.0):
            raise ValueError(
                "Magnetic-angle interpolation is not monotonic."
            )

    closed_R = np.append(radial, radial[0])
    closed_z = np.append(vertical, vertical[0])
    R_inverse = CubicSpline(
        theta_closed,
        closed_R,
        bc_type="periodic",
    )(thmaggrid)
    z_inverse = CubicSpline(
        theta_closed,
        closed_z,
        bc_type="periodic",
    )(thmaggrid)
    closed_jacobian = np.append(jacobian, jacobian[0])
    jacobian_sign = float(np.sign(closed_jacobian[0]))
    log_abs_jacobian = np.log(np.abs(closed_jacobian))
    with np.errstate(over="ignore", invalid="ignore"):
        jacobian_direct = jacobian_sign * np.exp(CubicSpline(
            surface_parameter,
            log_abs_jacobian,
            bc_type="periodic",
        )(thgeogrid))
    if (
        np.any(~np.isfinite(jacobian_direct))
        or np.any(np.sign(jacobian_direct) != jacobian_sign)
    ):
        jacobian_direct = jacobian_sign * np.exp(PchipInterpolator(
            surface_parameter,
            log_abs_jacobian,
        )(thgeogrid))
    if (
        np.any(~np.isfinite(jacobian_direct))
        or np.any(np.sign(jacobian_direct) != jacobian_sign)
    ):
        raise ValueError(
            "Sign-preserving Jacobian interpolation produced invalid values."
        )
    nu_direct = CubicSpline(
        surface_parameter,
        nu_closed,
        bc_type="periodic",
    )(thgeogrid)
    R_inverse[-1] = R_inverse[0]
    z_inverse[-1] = z_inverse[0]
    jacobian_direct[-1] = jacobian_direct[0]
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
    enforce_up_down_symmetry: bool = False,
    symmetry_tolerance: Optional[float] = None,
    diagnostics: Optional[MutableMapping[str, Any]] = None,
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
    if not isinstance(enforce_up_down_symmetry, (bool, np.bool_)):
        raise TypeError("enforce_up_down_symmetry must be boolean.")
    if symmetry_tolerance is not None:
        symmetry_tolerance = float(symmetry_tolerance)
        if not np.isfinite(symmetry_tolerance) or symmetry_tolerance <= 0.0:
            raise ValueError("symmetry_tolerance must be finite and positive.")
    if enforce_up_down_symmetry and symmetry_tolerance is None:
        raise ValueError(
            "symmetry_tolerance is required when up-down projection is enabled"
        )
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

    minimum_trace_points = max(64, 2 * spectral_max_mode + 4)
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
        minimum_points=minimum_trace_points,
    )
    raw_R, raw_z, raw_Br, raw_Bz, raw_Bphi = raw

    input_psi_field = (
        None
        if psi_field is None
        else np.asarray(psi_field, dtype=np.float64)
    )
    if input_psi_field is not None:
        if flux_scale is None:
            flux_scale = float(np.ptp(radial_flux))
        surfaces = build_flux_constrained_surfaces(
            Rgrid=radial_grid,
            zgrid=vertical_grid,
            psi_field=input_psi_field,
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

    (
        _,
        _,
        symmetry_audit,
        source_reflection,
    ) = _up_down_surface_projection(
        surface_R,
        surface_z,
        zaxis=zaxis,
        applied=bool(enforce_up_down_symmetry),
        tolerance=symmetry_tolerance,
    )
    source_field_splines = tuple(
        RectBivariateSpline(
            radial_grid,
            vertical_grid,
            values,
            kx=min(3, radial_grid.size - 1),
            ky=min(3, vertical_grid.size - 1),
            s=0.0,
        )
        for values in (Br, Bz, Bphi)
    )
    source_field_parity = {
        "Br": -1.0,
        "Bz": 1.0,
        "Bphi": 1.0,
    }
    for name, spline in zip(source_field_parity, source_field_splines):
        source_values = spline.ev(
            surface_R.ravel(),
            surface_z.ravel(),
        ).reshape(surface_R.shape)
        _, residual, relative_change = _project_periodic_field(
            source_values,
            reflection=source_reflection,
            parity=source_field_parity[name],
        )
        symmetry_audit["field_residuals"][name] = residual
        symmetry_audit["field_relative_changes"][name] = relative_change
    if enforce_up_down_symmetry:
        source_field_residual = max(
            float(np.max(np.asarray(values, dtype=np.float64)))
            for name, values in symmetry_audit["field_residuals"].items()
            if name in source_field_parity
        )
        if source_field_residual > float(symmetry_tolerance):
            raise ValueError(
                "magnetic field is not sufficiently up-down symmetric for "
                "explicit projection: "
                f"residual={source_field_residual:.3e}, "
                f"tolerance={float(symmetry_tolerance):.3e}"
            )

    working_Br = Br
    working_Bz = Bz
    working_Bphi = Bphi
    working_psi_field = input_psi_field
    if enforce_up_down_symmetry:
        (
            working_Br,
            working_Bz,
            working_Bphi,
            working_psi_field,
        ) = _symmetrize_equilibrium_grid(
            Rgrid=radial_grid,
            zgrid=vertical_grid,
            zaxis=zaxis,
            Br=Br,
            Bz=Bz,
            Bphi=Bphi,
            psi_field=input_psi_field,
        )
        projected_bz_seed = float(
            RegularGridInterpolator(
                (radial_grid, vertical_grid),
                working_Bz,
                bounds_error=False,
                fill_value=None,
            )((seeds[0], zaxis))
        )
        projected_integration_sign = np.sign(projected_bz_seed)
        if not phiclockwise:
            projected_integration_sign *= -1.0
        if projected_integration_sign == 0.0:
            raise ValueError(
                "Cannot determine projected field-line orientation at the "
                "first retained outboard flux-surface seed."
            )
        projected_raw = _trace_flux_surfaces(
            Rgrid=radial_grid,
            zgrid=vertical_grid,
            br=working_Br,
            bz=working_Bz,
            bphi=working_Bphi,
            R_at_psi=seeds,
            zaxis=zaxis,
            ntheta=n_theta_geom,
            integration_sign=projected_integration_sign,
            minimum_points=minimum_trace_points,
        )
        projected_raw_R, projected_raw_z, *_ = projected_raw
        if working_psi_field is not None:
            projected_surfaces = build_flux_constrained_surfaces(
                Rgrid=radial_grid,
                zgrid=vertical_grid,
                psi_field=working_psi_field,
                psigrid=radial_flux,
                R_raw=projected_raw_R,
                z_raw=projected_raw_z,
                ntheta=n_theta_geom,
                spectral_max_mode=spectral_max_mode,
                flux_scale=float(flux_scale),
                validate_nesting=True,
                gauge_z=zaxis,
                reflection_z=zaxis,
            )
            projected_surface_R = projected_surfaces.R
            projected_surface_z = projected_surfaces.z
        else:
            projected_surface_R = projected_raw_R
            projected_surface_z = projected_raw_z
        (
            surface_R,
            surface_z,
            final_symmetry_audit,
            reflection,
        ) = _up_down_surface_projection(
            projected_surface_R,
            projected_surface_z,
            zaxis=zaxis,
            applied=True,
            tolerance=symmetry_tolerance,
        )
        symmetry_audit["projected_geometry_residual"] = (
            final_symmetry_audit["geometry_residual"]
        )
        if working_psi_field is not None:
            projected_flux_residual = np.asarray(
                projected_surfaces.normalized_flux_residual,
                dtype=np.float64,
            )
            symmetry_audit["projected_flux_residual"] = (
                projected_flux_residual
            )
            if float(np.max(projected_flux_residual)) > 1.0e-8:
                raise ValueError(
                    "up-down projected surfaces no longer match their "
                    "projected physical-flux labels: "
                    f"residual={float(np.max(projected_flux_residual)):.3e}"
                )

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
        for values in (working_Br, working_Bz, working_Bphi)
    )

    if enforce_up_down_symmetry:
        sampled_fields = [
            spline.ev(surface_R.ravel(), surface_z.ravel()).reshape(
                surface_R.shape
            )
            for spline in field_splines
        ]
        parity_by_name = {
            "Br": -1.0,
            "Bz": 1.0,
            "Bphi": 1.0,
        }
        projected_fields: list[np.ndarray] = []
        symmetry_audit["projected_field_residuals"] = {}
        symmetry_audit["projected_field_relative_changes"] = {}
        for name, values in zip(parity_by_name, sampled_fields):
            projected, residual, relative_change = _project_periodic_field(
                values,
                reflection=reflection,
                parity=parity_by_name[name],
            )
            projected_fields.append(projected)
            symmetry_audit["projected_field_residuals"][name] = residual
            symmetry_audit["projected_field_relative_changes"][name] = (
                relative_change
            )
        surface_Br, surface_Bz, surface_Bphi = projected_fields
    if diagnostics is not None:
        diagnostics["up_down_symmetry"] = symmetry_audit
        if working_psi_field is not None:
            diagnostics["coordinate_psi_field"] = working_psi_field

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
