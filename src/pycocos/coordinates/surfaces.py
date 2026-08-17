"""Flux-constrained construction of general axisymmetric surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .accuracy import CoordinateAccuracy


_TWO_PI = 2.0 * np.pi
_DEFAULT_PROJECTION_TOLERANCE = (
    CoordinateAccuracy.standard().surface_flux_tolerance
)


@dataclass(frozen=True)
class FluxSurfaceSet:
    """Endpoint-exclusive, axis-to-boundary ordered flux surfaces."""

    psi: np.ndarray
    theta: np.ndarray
    R: np.ndarray
    z: np.ndarray
    normalized_flux_residual: np.ndarray
    signed_area: np.ndarray


def _strip_closed_endpoint(R: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    radial = np.asarray(R, dtype=np.float64).reshape(-1)
    vertical = np.asarray(z, dtype=np.float64).reshape(-1)
    if radial.size != vertical.size or radial.size < 8:
        raise ValueError("A closed contour requires matching R/z arrays with >= 8 points.")
    if not np.all(np.isfinite(radial)) or not np.all(np.isfinite(vertical)):
        raise ValueError("Flux-surface contours must contain only finite points.")
    scale = max(1.0, float(np.ptp(radial)), float(np.ptp(vertical)))
    if np.hypot(radial[-1] - radial[0], vertical[-1] - vertical[0]) <= 1.0e-10 * scale:
        radial = radial[:-1]
        vertical = vertical[:-1]
    if radial.size < 8:
        raise ValueError("Too few distinct points remain after removing contour closure.")
    return radial, vertical


def resample_closed_contour_by_arclength(
    R: np.ndarray,
    z: np.ndarray,
    ntheta: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a general closed contour without a geometric-angle assumption."""
    radial, vertical = _strip_closed_endpoint(R, z)
    if int(ntheta) != ntheta or ntheta < 8:
        raise ValueError("ntheta must be an integer >= 8.")
    ntheta = int(ntheta)

    closed_R = np.append(radial, radial[0])
    closed_z = np.append(vertical, vertical[0])
    segment = np.hypot(np.diff(closed_R), np.diff(closed_z))
    positive = segment > 10.0 * np.finfo(np.float64).eps
    if np.count_nonzero(positive) < 8:
        raise ValueError("Flux-surface contour contains too few distinct segments.")
    if not np.all(positive):
        keep = np.append(positive, True)
        closed_R = closed_R[keep]
        closed_z = closed_z[keep]
        segment = np.hypot(np.diff(closed_R), np.diff(closed_z))

    arclength = np.concatenate(([0.0], np.cumsum(segment)))
    perimeter = float(arclength[-1])
    if not np.isfinite(perimeter) or perimeter <= 0.0:
        raise ValueError("Flux-surface contour has a non-positive perimeter.")
    sample = np.linspace(0.0, perimeter, ntheta, endpoint=False)
    return (
        np.interp(sample, arclength, closed_R),
        np.interp(sample, arclength, closed_z),
    )


def _fourier_filter(values: np.ndarray, max_mode: int) -> np.ndarray:
    """Filter one or more contours in one batched real FFT."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim < 1 or data.shape[-1] < 2:
        raise ValueError("Periodic contour data must have at least two samples.")
    coefficients = np.fft.rfft(data, axis=-1)
    retained = min(int(max_mode), coefficients.shape[-1] - 1)
    coefficients[..., retained + 1 :] = 0.0
    return np.fft.irfft(coefficients, n=data.shape[-1], axis=-1)


def _pair_contour_about_midplane(
    R: np.ndarray,
    z: np.ndarray,
    *,
    reflection_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project periodic contour samples onto exact up-down parity."""
    radial = np.asarray(R, dtype=np.float64)
    vertical = np.asarray(z, dtype=np.float64)
    if radial.ndim < 1 or vertical.shape != radial.shape:
        raise ValueError("Reflection pairing requires matching contour arrays.")
    vertical_offset = vertical - float(reflection_z)
    reflection = np.concatenate(
        ([0], np.arange(radial.shape[-1] - 1, 0, -1))
    )
    paired_R = 0.5 * (radial + radial[..., reflection])
    paired_z_offset = 0.5 * (
        vertical_offset - vertical_offset[..., reflection]
    )
    return paired_R, paired_z_offset + float(reflection_z)


def canonicalize_contour_samples(
    R: np.ndarray,
    z: np.ndarray,
    *fields: np.ndarray,
    gauge_z: float | None = None,
) -> Tuple[np.ndarray, ...]:
    """Set theta=0 at a stable outboard gauge and shift all sampled fields."""
    radial, vertical = _strip_closed_endpoint(R, z)
    if radial.size != vertical.size:
        raise ValueError("R and z must have matching periodic samples.")
    sampled_fields = tuple(
        np.asarray(field, dtype=np.float64).reshape(-1)
        for field in fields
    )
    if any(field.size != radial.size for field in sampled_fields):
        raise ValueError("Every periodic field must match the contour size.")

    modes = np.rint(
        np.fft.fftfreq(radial.size, d=1.0 / radial.size)
    ).astype(int)
    if gauge_z is None:
        maximum = int(np.argmax(radial))
        previous_value = radial[(maximum - 1) % radial.size]
        maximum_value = radial[maximum]
        next_value = radial[(maximum + 1) % radial.size]
        denominator = previous_value - 2.0 * maximum_value + next_value
        if abs(denominator) <= np.finfo(np.float64).eps * max(
            1.0,
            abs(maximum_value),
        ):
            subgrid_offset = 0.0
        else:
            subgrid_offset = 0.5 * (
                previous_value - next_value
            ) / denominator
            subgrid_offset = float(np.clip(subgrid_offset, -0.5, 0.5))
        shift = _TWO_PI * (maximum + subgrid_offset) / radial.size
    else:
        vertical_offset = vertical - float(gauge_z)
        next_offset = np.roll(vertical_offset, -1)
        crossings = np.flatnonzero(
            (vertical_offset == 0.0)
            | (vertical_offset * next_offset < 0.0)
        )
        if crossings.size == 0:
            raise ValueError(
                "Flux surface does not cross the requested horizontal "
                "angular-gauge line."
            )
        fractions = -vertical_offset[crossings] / (
            next_offset[crossings] - vertical_offset[crossings]
        )
        crossing_R = radial[crossings] + fractions * (
            np.roll(radial, -1)[crossings] - radial[crossings]
        )
        selected = int(np.argmax(crossing_R))
        shift = _TWO_PI * (
            crossings[selected] + fractions[selected]
        ) / radial.size

        coefficients_z = np.fft.fft(vertical)
        if vertical.size % 2 == 0:
            coefficients_z[vertical.size // 2] = 0.0
        coefficients_z /= vertical.size
        for _ in range(8):
            spectral_phase = np.exp(1j * modes * shift)
            residual = float(
                np.real(np.sum(coefficients_z * spectral_phase))
                - float(gauge_z)
            )
            derivative = float(
                np.real(
                    np.sum(
                        1j * modes * coefficients_z * spectral_phase
                    )
                )
            )
            if abs(derivative) <= np.finfo(np.float64).tiny:
                break
            update = residual / derivative
            shift -= update
            if abs(update) <= 1.0e-13:
                break
        shift = float(np.mod(shift, _TWO_PI))

    phase = np.exp(1j * modes * shift)

    def shifted(values: np.ndarray) -> np.ndarray:
        coefficients = np.fft.fft(values)
        if values.size % 2 == 0:
            # A lone Nyquist coefficient has no real-valued fractional phase
            # continuation; discard this unresolved grid-scale component.
            coefficients[values.size // 2] = 0.0
        return np.asarray(
            np.real(np.fft.ifft(coefficients * phase)),
            dtype=np.float64,
        )

    return tuple(
        shifted(values)
        for values in (radial, vertical, *sampled_fields)
    )


def canonicalize_contour_origin(
    R: np.ndarray,
    z: np.ndarray,
    *,
    gauge_z: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set theta=0 at the continuously resolved outboard-most point."""
    radial, vertical = canonicalize_contour_samples(
        R,
        z,
        gauge_z=gauge_z,
    )
    return radial, vertical


def _signed_polygon_area(R: np.ndarray, z: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(R * np.roll(z, -1) - np.roll(R, -1) * z)
    )


def _point_in_polygon(
    point_R: np.ndarray,
    point_z: np.ndarray,
    polygon_R: np.ndarray,
    polygon_z: np.ndarray,
) -> np.ndarray:
    """Vectorized even-odd containment test."""
    x = np.asarray(point_R, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(point_z, dtype=np.float64).reshape(-1, 1)
    x0 = polygon_R[None, :]
    y0 = polygon_z[None, :]
    x1 = np.roll(polygon_R, -1)[None, :]
    y1 = np.roll(polygon_z, -1)[None, :]
    crosses_y = (y0 > y) != (y1 > y)
    denominator = y1 - y0
    safe_denominator = np.where(
        np.abs(denominator) > np.finfo(np.float64).eps,
        denominator,
        np.inf,
    )
    crossing_x = (x1 - x0) * (y - y0) / safe_denominator + x0
    return np.count_nonzero(crosses_y & (x < crossing_x), axis=1) % 2 == 1


def project_contour_to_flux(
    *,
    R: np.ndarray,
    z: np.ndarray,
    target_psi: float,
    psi_value: Callable[[np.ndarray, np.ndarray], np.ndarray],
    psi_gradient: Callable[
        [np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ],
    flux_scale: float,
    tolerance: float = _DEFAULT_PROJECTION_TOLERANCE,
    max_iterations: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Newton-project contour points along ``grad(psi)``."""
    radial = np.asarray(R, dtype=np.float64)
    vertical = np.asarray(z, dtype=np.float64)
    if radial.ndim != 1 or vertical.shape != radial.shape:
        raise ValueError("A projected contour must be matching one-dimensional arrays.")
    projected_R, projected_z, residual = project_contours_to_flux(
        R=radial[None, :],
        z=vertical[None, :],
        target_psi=np.asarray([target_psi], dtype=np.float64),
        psi_value=psi_value,
        psi_gradient=psi_gradient,
        flux_scale=flux_scale,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return projected_R[0], projected_z[0], float(residual[0])


def project_contours_to_flux(
    *,
    R: np.ndarray,
    z: np.ndarray,
    target_psi: np.ndarray,
    psi_value: Callable[[np.ndarray, np.ndarray], np.ndarray],
    psi_gradient: Callable[
        [np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ],
    flux_scale: float,
    tolerance: float,
    max_iterations: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a batch of contours, updating only unconverged vertices."""
    radial = np.asarray(R, dtype=np.float64).copy()
    vertical = np.asarray(z, dtype=np.float64).copy()
    targets = np.asarray(target_psi, dtype=np.float64).reshape(-1)
    if radial.ndim != 2 or vertical.shape != radial.shape:
        raise ValueError("Projected contours must have shape (surface, angle).")
    if targets.shape != (radial.shape[0],):
        raise ValueError("target_psi must contain one value per contour.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Projection tolerance must be finite and positive.")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")

    scale = max(abs(float(flux_scale)), np.finfo(np.float64).tiny)
    target_grid = np.broadcast_to(targets[:, None], radial.shape)
    active = np.ones(radial.shape, dtype=bool)
    error = np.empty(radial.shape, dtype=np.float64)

    for _ in range(int(max_iterations)):
        active_R = radial[active]
        active_z = vertical[active]
        active_error = (
            np.asarray(psi_value(active_R, active_z), dtype=np.float64)
            - target_grid[active]
        )
        if not np.all(np.isfinite(active_error)):
            raise ValueError(
                "Cannot project a contour with non-finite psi values."
            )
        error[active] = active_error
        active = np.abs(error) / scale > tolerance
        if not np.any(active):
            residual = np.max(np.abs(error), axis=1) / scale
            return radial, vertical, residual
        dpsi_dR, dpsi_dz = psi_gradient(radial[active], vertical[active])
        dpsi_dR = np.asarray(dpsi_dR, dtype=np.float64)
        dpsi_dz = np.asarray(dpsi_dz, dtype=np.float64)
        gradient_squared = dpsi_dR**2 + dpsi_dz**2
        if np.any(~np.isfinite(gradient_squared)) or np.any(
            gradient_squared <= np.finfo(np.float64).tiny
        ):
            raise ValueError("Cannot project a contour where |grad(psi)| vanishes.")
        radial[active] -= error[active] * dpsi_dR / gradient_squared
        vertical[active] -= error[active] * dpsi_dz / gradient_squared
        if not np.all(np.isfinite(radial)) or not np.all(np.isfinite(vertical)):
            raise ValueError("Flux-surface projection left the finite coordinate domain.")

    error = (
        np.asarray(psi_value(radial, vertical), dtype=np.float64)
        - target_grid
    )
    residual = np.max(np.abs(error), axis=1) / scale
    raise ValueError(
        "Flux-surface projection did not converge for all contours: "
        f"maximum normalized residual={float(np.max(residual)):.3e}, "
        f"tolerance={tolerance:.3e}."
    )


def build_flux_constrained_surfaces(
    *,
    Rgrid: np.ndarray,
    zgrid: np.ndarray,
    psi_field: np.ndarray,
    psigrid: np.ndarray,
    R_raw: np.ndarray,
    z_raw: np.ndarray,
    ntheta: int,
    spectral_max_mode: int,
    flux_scale: float,
    projection_tolerance: float = _DEFAULT_PROJECTION_TOLERANCE,
    validate_nesting: bool = True,
    gauge_z: float | None = None,
    reflection_z: float | None = None,
) -> FluxSurfaceSet:
    """Filter, project, and validate axis-to-boundary ordered surfaces.

    When ``reflection_z`` is supplied, flux values and gradients are evaluated
    as reflection pairs and every construction step ends in the corresponding
    geometric parity sector.  This keeps the final contour both exactly
    up-down paired and on its physical flux label; averaging two independently
    projected contours after construction does not preserve that label.
    """
    R_axis = np.asarray(Rgrid, dtype=np.float64)
    z_axis = np.asarray(zgrid, dtype=np.float64)
    psi_values = np.asarray(psi_field, dtype=np.float64)
    radial_flux = np.asarray(psigrid, dtype=np.float64)
    raw_R = np.asarray(R_raw, dtype=np.float64)
    raw_z = np.asarray(z_raw, dtype=np.float64)
    if psi_values.shape != (R_axis.size, z_axis.size):
        raise ValueError("psi_field shape must match (Rgrid, zgrid).")
    if raw_R.shape != raw_z.shape or raw_R.shape[0] != radial_flux.size:
        raise ValueError("Raw surface arrays must have shape (npsi, npoints).")
    if spectral_max_mode < 1 or spectral_max_mode >= ntheta // 2:
        raise ValueError(
            "spectral_max_mode must satisfy 1 <= mode < ntheta/2."
        )
    projection_tolerance = float(projection_tolerance)
    if not np.isfinite(projection_tolerance) or projection_tolerance <= 0.0:
        raise ValueError("projection_tolerance must be finite and positive.")

    spline = RectBivariateSpline(
        R_axis,
        z_axis,
        psi_values,
        kx=min(3, R_axis.size - 1),
        ky=min(3, z_axis.size - 1),
        s=0.0,
    )

    if reflection_z is not None:
        reflection_z = float(reflection_z)
        if not np.isfinite(reflection_z):
            raise ValueError("reflection_z must be finite.")

    def psi_value(R: np.ndarray, z: np.ndarray) -> np.ndarray:
        values = spline.ev(R, z)
        if reflection_z is None:
            return values
        reflected_z = 2.0 * reflection_z - np.asarray(z, dtype=np.float64)
        return 0.5 * (values + spline.ev(R, reflected_z))

    def psi_gradient(
        R: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dpsi_dR = spline.ev(R, z, dx=1)
        dpsi_dz = spline.ev(R, z, dy=1)
        if reflection_z is None:
            return dpsi_dR, dpsi_dz
        reflected_z = 2.0 * reflection_z - np.asarray(z, dtype=np.float64)
        return (
            0.5 * (dpsi_dR + spline.ev(R, reflected_z, dx=1)),
            0.5 * (dpsi_dz - spline.ev(R, reflected_z, dy=1)),
        )

    def project_surfaces(
        radial: np.ndarray,
        vertical: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if reflection_z is not None:
            radial, vertical = _pair_contour_about_midplane(
                radial,
                vertical,
                reflection_z=reflection_z,
            )
        radial, vertical, residual = project_contours_to_flux(
            R=radial,
            z=vertical,
            target_psi=radial_flux,
            psi_value=psi_value,
            psi_gradient=psi_gradient,
            flux_scale=flux_scale,
            tolerance=projection_tolerance,
        )
        if reflection_z is not None:
            radial, vertical = _pair_contour_about_midplane(
                radial,
                vertical,
                reflection_z=reflection_z,
            )
            residual = (
                np.max(
                    np.abs(
                        psi_value(radial, vertical)
                        - radial_flux[:, None]
                    ),
                    axis=1,
                )
                / max(abs(float(flux_scale)), np.finfo(np.float64).tiny)
            )
            if float(np.max(residual)) > projection_tolerance:
                raise ValueError(
                    "Reflection-paired flux-surface projection did not "
                    "converge: "
                    f"maximum normalized residual={float(np.max(residual)):.3e}, "
                    f"tolerance={projection_tolerance:.3e}."
                )
        return radial, vertical, residual

    surfaces_R = np.empty((radial_flux.size, ntheta), dtype=np.float64)
    surfaces_z = np.empty_like(surfaces_R)
    areas = np.empty(radial_flux.size, dtype=np.float64)
    theta = np.linspace(0.0, _TWO_PI, ntheta, endpoint=False)

    for index in range(radial_flux.size):
        surfaces_R[index], surfaces_z[index] = resample_closed_contour_by_arclength(
            raw_R[index],
            raw_z[index],
            ntheta,
        )

    # Filter every surface in one FFT batch. Alternating filtering and exact-
    # flux projection removes tracing noise without relabeling a surface.
    for _ in range(2):
        surfaces_R = _fourier_filter(surfaces_R, spectral_max_mode)
        surfaces_z = _fourier_filter(surfaces_z, spectral_max_mode)
        surfaces_R, surfaces_z, _ = project_surfaces(
            surfaces_R,
            surfaces_z,
        )

    # The second filter pass already ends in an exact-flux projection. The old
    # immediate third projection repeated the same Newton residual evaluation
    # without changing the contour.
    for index in range(radial_flux.size):
        surfaces_R[index], surfaces_z[index] = canonicalize_contour_origin(
            surfaces_R[index],
            surfaces_z[index],
            gauge_z=gauge_z,
        )
    surfaces_R, surfaces_z, residuals = project_surfaces(
        surfaces_R,
        surfaces_z,
    )

    for index in range(radial_flux.size):
        area = _signed_polygon_area(surfaces_R[index], surfaces_z[index])
        scale = max(
            float(np.ptp(surfaces_R[index])),
            float(np.ptp(surfaces_z[index])),
            1.0,
        )
        if abs(area) <= 1.0e-12 * scale**2:
            raise ValueError(f"Flux surface {index} has a degenerate signed area.")
        areas[index] = area

    orientation = np.sign(areas)
    if np.any(orientation != orientation[0]):
        raise ValueError("Flux-surface orientation changes across the radial domain.")

    if validate_nesting and radial_flux.size > 1:
        stride = max(1, ntheta // 128)
        for index in range(radial_flux.size - 1):
            inside = _point_in_polygon(
                surfaces_R[index, ::stride],
                surfaces_z[index, ::stride],
                surfaces_R[index + 1],
                surfaces_z[index + 1],
            )
            if not np.all(inside):
                raise ValueError(
                    f"Flux surfaces {index} and {index + 1} are not strictly nested."
                )

    return FluxSurfaceSet(
        psi=radial_flux.copy(),
        theta=theta,
        R=surfaces_R,
        z=surfaces_z,
        normalized_flux_residual=residuals,
        signed_area=areas,
    )
