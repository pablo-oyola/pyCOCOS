"""Differentiable axisymmetric magnetic-coordinate maps.

The authoritative representation is Fourier in the periodic magnetic angle
and interpolating in one smooth radial parameter.  Equilibrium-built maps use
signed sqrt-normalized flux internally for near-axis regularity and apply the
exact chain rule back to physical poloidal flux. Values and every first
derivative therefore come from one set of coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline, make_interp_spline
from scipy.optimize import brentq

try:
    from numba import config as _numba_config

    from .coordinate_map_numba import axisymmetric_differential_kernel
except ImportError:  # pragma: no cover - NumPy fallback supports minimal installs.
    _numba_config = None
    axisymmetric_differential_kernel = None


_TWO_PI = 2.0 * np.pi
# The vectorized NumPy algebra is faster for ordinary interactive batches;
# amortize parallel dispatch and JIT costs only for materialization-scale grids.
_COMPILED_DIFFERENTIAL_MINIMUM_POINTS = 100_000


def _endpoint_exclusive_periodic_data(
    theta: np.ndarray,
    values: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return a uniform endpoint-exclusive periodic grid and matching data."""
    angle = np.asarray(theta, dtype=np.float64)
    if angle.ndim != 1 or angle.size < 4:
        raise ValueError("theta must be one-dimensional with at least four points.")
    if not np.all(np.isfinite(angle)) or np.any(np.diff(angle) <= 0.0):
        raise ValueError("theta must be finite and strictly increasing.")

    arrays = {
        name: np.asarray(array, dtype=np.float64)
        for name, array in values.items()
    }
    for name, array in arrays.items():
        if array.ndim != 2 or array.shape[1] != angle.size:
            raise ValueError(
                f"{name} must have shape (npsi, ntheta); got {array.shape}."
            )

    span = angle[-1] - angle[0]
    closed = np.isclose(span, _TWO_PI, rtol=1.0e-10, atol=1.0e-12)
    if closed:
        for name, array in arrays.items():
            scale = max(1.0, float(np.nanmax(np.abs(array))))
            if not np.allclose(
                array[:, -1],
                array[:, 0],
                rtol=1.0e-9,
                atol=1.0e-11 * scale,
            ):
                raise ValueError(
                    f"{name} does not close periodically at theta=2*pi."
                )
        angle = angle[:-1]
        arrays = {name: array[:, :-1] for name, array in arrays.items()}

    if angle.size < 4:
        raise ValueError("At least four endpoint-exclusive theta points are required.")
    steps = np.diff(angle)
    expected_step = _TWO_PI / angle.size
    if not np.allclose(
        steps,
        expected_step,
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        raise ValueError("theta must be a uniform periodic grid.")

    origin = float(angle[0])
    canonical = origin + expected_step * np.arange(angle.size)
    return canonical, arrays


@dataclass(frozen=True)
class CoordinateTangents:
    """Mapped values and the direct cylindrical coordinate basis only.

    ``direct[..., component, coordinate]`` uses orthonormal cylindrical
    components ``(e_R, e_phi, e_z)`` and coordinate columns
    ``(psi, theta, zeta)``.  No inverse, metric tensor, or determinant is
    constructed.
    """

    values: Mapping[str, np.ndarray]
    direct: np.ndarray


@dataclass(frozen=True)
class CoordinateDifferentials:
    """Coordinate values, basis matrices, metrics, and Jacobian."""

    values: Mapping[str, np.ndarray]
    direct: np.ndarray
    inverse: np.ndarray
    metric_covariant: np.ndarray
    metric_contravariant: np.ndarray
    jacobian: np.ndarray


class SpectralCoordinateMap:
    """Fourier--spline representation of ``(R, z, phi)(psi, theta, zeta)``.

    The axisymmetric toroidal gauge is

    ``zeta = phi + nu(psi, theta)``,

    hence ``phi = zeta - nu`` when constructing the inverse-coordinate basis.
    """

    field_names = ("R", "z", "nu")
    state_format_version = 1

    def __init__(
        self,
        *,
        psi: np.ndarray,
        theta: np.ndarray,
        R: np.ndarray,
        z: np.ndarray,
        nu: Optional[np.ndarray] = None,
        max_mode: Optional[int] = None,
        psi_axis: Optional[float] = None,
        psi_boundary: Optional[float] = None,
        R_axis: Optional[float] = None,
        z_axis: Optional[float] = None,
        enforce_up_down_symmetry: bool = False,
        symmetry_tolerance: Optional[float] = None,
        flux_constraint_R: Optional[np.ndarray] = None,
        flux_constraint_z: Optional[np.ndarray] = None,
        flux_constraint_psi: Optional[np.ndarray] = None,
        flux_constraint_tolerance: float = 1.0e-10,
        flux_constraint_max_iterations: int = 12,
        flux_constraint_minimum_radial_derivative: float = 1.0e-8,
    ) -> None:
        radial = np.asarray(psi, dtype=np.float64)
        if radial.ndim != 1 or radial.size < 2:
            raise ValueError("psi must be one-dimensional with at least two points.")
        if not np.all(np.isfinite(radial)) or np.any(np.diff(radial) <= 0.0):
            raise ValueError("psi must be finite and strictly increasing.")

        if nu is None:
            nu = np.zeros_like(R, dtype=np.float64)
        angle, arrays = _endpoint_exclusive_periodic_data(
            theta,
            {"R": R, "z": z, "nu": nu},
        )
        for name, array in arrays.items():
            if array.shape[0] != radial.size:
                raise ValueError(
                    f"{name} radial size {array.shape[0]} does not match "
                    f"psi size {radial.size}."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values.")

        # Keep the normalized constructor inputs, rather than SciPy spline
        # objects, as the durable representation used by ``to_state``.  This
        # is intentionally captured before an optional symmetry projection so
        # restoring the map also restores the same projection audit.
        self._state_input_fields = {
            name: np.array(array, dtype=np.float64, copy=True)
            for name, array in arrays.items()
        }

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
        if (R_axis is None) != (z_axis is None):
            raise ValueError(
                "R_axis and z_axis must either both be supplied or both be omitted."
            )
        self._state_R_axis = None if R_axis is None else float(R_axis)
        self._state_z_axis = None if z_axis is None else float(z_axis)
        self._state_enforce_up_down_symmetry = bool(enforce_up_down_symmetry)
        self._state_symmetry_tolerance = symmetry_tolerance
        if R_axis is None:
            center_R = np.mean(arrays["R"], axis=1)
            center_z = np.mean(arrays["z"], axis=1)
        else:
            center_R = np.full(radial.size, float(R_axis))
            center_z = np.full(radial.size, float(z_axis))

        reflection_indices = np.concatenate(
            ([0], np.arange(angle.size - 1, 0, -1))
        )
        centered_fields = {
            "R": arrays["R"] - center_R[:, None],
            "z": arrays["z"] - center_z[:, None],
            "nu": arrays["nu"],
        }
        expected_parity = {"R": 1.0, "z": -1.0, "nu": -1.0}
        field_residuals: Dict[str, np.ndarray] = {}
        field_changes: Dict[str, np.ndarray] = {}
        projected_fields: Dict[str, np.ndarray] = {}
        geometry_scale = np.maximum(
            np.maximum(
                np.ptp(arrays["R"], axis=1),
                np.ptp(arrays["z"], axis=1),
            ),
            np.finfo(np.float64).tiny,
        )
        for name, values in centered_fields.items():
            reflected = np.take(values, reflection_indices, axis=1)
            parity = expected_parity[name]
            scale = (
                geometry_scale
                if name in {"R", "z"}
                else np.maximum(
                    np.max(np.abs(values), axis=1),
                    np.finfo(np.float64).tiny,
                )
            )
            field_residuals[name] = (
                np.max(np.abs(values - parity * reflected), axis=1) / scale
            )
            projected = 0.5 * (values + parity * reflected)
            projected_fields[name] = projected
            field_changes[name] = (
                np.max(np.abs(projected - values), axis=1) / scale
            )

        geometry_residual = np.maximum(
            field_residuals["R"],
            field_residuals["z"],
        )
        if (
            enforce_up_down_symmetry
            and symmetry_tolerance is not None
            and float(np.max(geometry_residual)) > symmetry_tolerance
        ):
            raise ValueError(
                "coordinate map is not sufficiently up-down symmetric for "
                "explicit projection: "
                f"residual={float(np.max(geometry_residual)):.3e}, "
                f"tolerance={symmetry_tolerance:.3e}"
            )
        if enforce_up_down_symmetry:
            arrays["R"] = projected_fields["R"] + center_R[:, None]
            arrays["z"] = projected_fields["z"] + center_z[:, None]
            arrays["nu"] = projected_fields["nu"]

        self.up_down_symmetry_audit = {
            "applied": bool(enforce_up_down_symmetry),
            "tolerance": symmetry_tolerance,
            "geometry_residual": geometry_residual,
            "field_residuals": field_residuals,
            "field_relative_changes": field_changes,
        }

        ntheta = angle.size
        if max_mode is None:
            retained_mode = ntheta // 2
        else:
            if isinstance(max_mode, bool) or int(max_mode) != max_mode:
                raise TypeError("max_mode must be an integer or None.")
            retained_mode = int(max_mode)
            if retained_mode < 0:
                raise ValueError("max_mode must be non-negative.")
            retained_mode = min(retained_mode, ntheta // 2)

        modes = np.rint(np.fft.fftfreq(ntheta, d=1.0 / ntheta)).astype(int)
        retained = np.abs(modes) <= retained_mode
        self.psi = radial
        self.theta = angle
        self.theta_origin = float(angle[0])
        self.modes = modes[retained]
        self.max_mode = retained_mode
        signed_area = 0.5 * np.sum(
            arrays["R"] * np.roll(arrays["z"], -1, axis=1)
            - np.roll(arrays["R"], -1, axis=1) * arrays["z"],
            axis=1,
        )
        orientation = float(np.sign(np.median(signed_area)))
        if orientation == 0.0:
            raise ValueError(
                "Cannot determine the poloidal orientation of the coordinate map."
            )
        self.angular_orientation = orientation
        origin_phase = np.arctan2(
            arrays["z"][:, 0] - center_z,
            arrays["R"][:, 0] - center_R,
        )
        self.geometric_theta_origin = float(
            np.angle(np.mean(np.exp(1j * origin_phase)))
        )
        if (psi_axis is None) != (psi_boundary is None):
            raise ValueError(
                "psi_axis and psi_boundary must either both be supplied or "
                "both be omitted."
            )
        self.psi_axis = None if psi_axis is None else float(psi_axis)
        self.psi_boundary = (
            None if psi_boundary is None else float(psi_boundary)
        )
        if self.psi_axis is None:
            radial_parameter = radial
        else:
            if not np.isfinite(self.psi_axis) or not np.isfinite(
                self.psi_boundary
            ):
                raise ValueError("Axis and boundary fluxes must be finite.")
            if np.isclose(self.psi_axis, self.psi_boundary):
                raise ValueError("Axis and boundary fluxes must be distinct.")
            radial_parameter, _ = self._radial_parameter(radial)
            if np.any(np.diff(radial_parameter) <= 0.0):
                raise ValueError(
                    "The signed sqrt-normalized-flux parameter must increase "
                    "with the supplied physical psi grid."
                )
        radial_degree = 5 if radial.size >= 6 else (3 if radial.size >= 4 else 1)
        self.radial_degree = radial_degree
        self._splines: Dict[str, object] = {}
        spectral_coefficients = []

        for name, array in arrays.items():
            coefficients = np.fft.fft(array, axis=1) / ntheta
            coefficients = coefficients[:, retained]
            spectral_coefficients.append(coefficients)
            self._splines[name] = make_interp_spline(
                radial_parameter,
                coefficients,
                k=radial_degree,
                axis=0,
            )
        # Internal coordinate operations almost always need R, z, and nu
        # together.  A single vector-valued spline lets those operations share
        # radial interval lookup and basis evaluation; the individual splines
        # above retain the inexpensive public single-field path.
        self._field_spline = make_interp_spline(
            radial_parameter,
            np.stack(spectral_coefficients, axis=1),
            k=radial_degree,
            axis=0,
        )

        constraint_arrays = (
            flux_constraint_R,
            flux_constraint_z,
            flux_constraint_psi,
        )
        if any(value is not None for value in constraint_arrays) and not all(
            value is not None for value in constraint_arrays
        ):
            raise ValueError(
                "flux_constraint_R, flux_constraint_z, and "
                "flux_constraint_psi must be supplied together."
            )
        if (
            not np.isfinite(flux_constraint_tolerance)
            or float(flux_constraint_tolerance) <= 0.0
        ):
            raise ValueError("flux_constraint_tolerance must be finite and positive.")
        if (
            isinstance(flux_constraint_max_iterations, bool)
            or int(flux_constraint_max_iterations) != flux_constraint_max_iterations
            or int(flux_constraint_max_iterations) < 1
        ):
            raise ValueError("flux_constraint_max_iterations must be a positive integer.")
        if (
            not np.isfinite(flux_constraint_minimum_radial_derivative)
            or float(flux_constraint_minimum_radial_derivative) <= 0.0
        ):
            raise ValueError(
                "flux_constraint_minimum_radial_derivative must be finite and positive."
            )

        self.flux_constraint_tolerance = float(flux_constraint_tolerance)
        self.flux_constraint_max_iterations = int(
            flux_constraint_max_iterations
        )
        self.flux_constraint_minimum_radial_derivative = float(
            flux_constraint_minimum_radial_derivative
        )
        self.flux_constraint_R: Optional[np.ndarray] = None
        self.flux_constraint_z: Optional[np.ndarray] = None
        self.flux_constraint_psi: Optional[np.ndarray] = None
        self._flux_constraint_spline: Optional[RectBivariateSpline] = None
        self._flux_constraint_cache: Optional[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                Dict[str, float | int],
            ]
        ] = None
        if flux_constraint_R is None:
            self.flux_constraint_scale = 1.0
            self.flux_constraint_audit = {"applied": False}
        else:
            constraint_R = np.asarray(flux_constraint_R, dtype=np.float64)
            constraint_z = np.asarray(flux_constraint_z, dtype=np.float64)
            constraint_psi = np.asarray(flux_constraint_psi, dtype=np.float64)
            if (
                constraint_R.ndim != 1
                or constraint_z.ndim != 1
                or constraint_R.size < 4
                or constraint_z.size < 4
                or constraint_psi.shape
                != (constraint_R.size, constraint_z.size)
                or not np.all(np.isfinite(constraint_R))
                or not np.all(np.isfinite(constraint_z))
                or not np.all(np.isfinite(constraint_psi))
                or np.any(np.diff(constraint_R) <= 0.0)
                or np.any(np.diff(constraint_z) <= 0.0)
            ):
                raise ValueError(
                    "flux-constraint grids must be finite, increasing, and "
                    "match the two-dimensional psi table."
                )
            self.flux_constraint_R = constraint_R.copy()
            self.flux_constraint_z = constraint_z.copy()
            self.flux_constraint_psi = constraint_psi.copy()
            self._flux_constraint_spline = RectBivariateSpline(
                self.flux_constraint_R,
                self.flux_constraint_z,
                self.flux_constraint_psi,
                kx=min(3, self.flux_constraint_R.size - 1),
                ky=min(3, self.flux_constraint_z.size - 1),
                s=0.0,
            )
            if self.psi_axis is not None:
                constraint_scale = abs(
                    float(self.psi_boundary - self.psi_axis)
                )
            else:
                constraint_scale = float(np.ptp(self.flux_constraint_psi))
            self.flux_constraint_scale = max(
                constraint_scale,
                np.finfo(np.float64).tiny,
            )
            validation_psi, validation_theta = np.meshgrid(
                self.psi,
                self.theta,
                indexing="ij",
            )
            _, validation = self._solve_flux_constraint(
                validation_psi,
                validation_theta,
            )
            self.flux_constraint_audit = {
                "applied": True,
                "tolerance": self.flux_constraint_tolerance,
                "max_iterations": self.flux_constraint_max_iterations,
                "minimum_radial_derivative": (
                    self.flux_constraint_minimum_radial_derivative
                ),
                "validation_iterations": validation["iterations"],
                "validation_normalized_residual": validation[
                    "normalized_residual"
                ],
                "validation_minimum_abs_F_sigma": validation[
                    "minimum_abs_F_sigma"
                ],
                "validation_bounded_root_fallback_count": validation[
                    "bounded_root_fallback_count"
                ],
                "validation_bounded_root_fallback_iterations": validation[
                    "bounded_root_fallback_iterations"
                ],
                "validation_newton_point_count": validation[
                    "newton_point_count"
                ],
                "validation_newton_active_evaluations": validation[
                    "newton_active_evaluations"
                ],
                "validation_newton_full_batch_equivalent_evaluations": validation[
                    "newton_full_batch_equivalent_evaluations"
                ],
                "validation_newton_evaluation_rounds": validation[
                    "newton_evaluation_rounds"
                ],
                "radial_grid_size": int(self.flux_constraint_R.size),
                "vertical_grid_size": int(self.flux_constraint_z.size),
            }

    def to_state(self) -> Dict[str, np.ndarray | np.generic]:
        """Return a deterministic, non-pickle serialization state.

        Every value is a NumPy scalar or numeric array, so the returned
        mapping can be passed directly to ``numpy.savez`` and restored with
        ``allow_pickle=False``.  SciPy spline objects and transient Newton
        caches are deliberately excluded; ``from_state`` rebuilds them from
        the normalized constructor data.
        """
        has_flux_normalization = self.psi_axis is not None
        has_fixed_axis = self._state_R_axis is not None
        has_symmetry_tolerance = self._state_symmetry_tolerance is not None
        has_flux_constraint = self._flux_constraint_spline is not None
        return {
            "state_format_version": np.int64(self.state_format_version),
            "psi": np.array(self.psi, dtype=np.float64, copy=True),
            "theta": np.array(self.theta, dtype=np.float64, copy=True),
            "R": self._state_input_fields["R"].copy(),
            "z": self._state_input_fields["z"].copy(),
            "nu": self._state_input_fields["nu"].copy(),
            "max_mode": np.int64(self.max_mode),
            "has_flux_normalization": np.bool_(has_flux_normalization),
            "psi_axis": np.float64(
                self.psi_axis if has_flux_normalization else np.nan
            ),
            "psi_boundary": np.float64(
                self.psi_boundary if has_flux_normalization else np.nan
            ),
            "has_fixed_axis": np.bool_(has_fixed_axis),
            "R_axis": np.float64(
                self._state_R_axis if has_fixed_axis else np.nan
            ),
            "z_axis": np.float64(
                self._state_z_axis if has_fixed_axis else np.nan
            ),
            "enforce_up_down_symmetry": np.bool_(
                self._state_enforce_up_down_symmetry
            ),
            "has_symmetry_tolerance": np.bool_(has_symmetry_tolerance),
            "symmetry_tolerance": np.float64(
                self._state_symmetry_tolerance
                if has_symmetry_tolerance
                else np.nan
            ),
            "has_flux_constraint": np.bool_(has_flux_constraint),
            "flux_constraint_R": (
                self.flux_constraint_R.copy()
                if has_flux_constraint
                else np.empty(0, dtype=np.float64)
            ),
            "flux_constraint_z": (
                self.flux_constraint_z.copy()
                if has_flux_constraint
                else np.empty(0, dtype=np.float64)
            ),
            "flux_constraint_psi": (
                self.flux_constraint_psi.copy()
                if has_flux_constraint
                else np.empty((0, 0), dtype=np.float64)
            ),
            "flux_constraint_tolerance": np.float64(
                self.flux_constraint_tolerance
            ),
            "flux_constraint_max_iterations": np.int64(
                self.flux_constraint_max_iterations
            ),
            "flux_constraint_minimum_radial_derivative": np.float64(
                self.flux_constraint_minimum_radial_derivative
            ),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, object],
    ) -> "SpectralCoordinateMap":
        """Rebuild a coordinate map from :meth:`to_state` output."""
        required = {
            "state_format_version",
            "psi",
            "theta",
            "R",
            "z",
            "nu",
            "max_mode",
            "has_flux_normalization",
            "psi_axis",
            "psi_boundary",
            "has_fixed_axis",
            "R_axis",
            "z_axis",
            "enforce_up_down_symmetry",
            "has_symmetry_tolerance",
            "symmetry_tolerance",
            "has_flux_constraint",
            "flux_constraint_R",
            "flux_constraint_z",
            "flux_constraint_psi",
            "flux_constraint_tolerance",
            "flux_constraint_max_iterations",
            "flux_constraint_minimum_radial_derivative",
        }
        missing = sorted(required.difference(state.keys()))
        if missing:
            raise ValueError(
                "Coordinate-map state is missing required entries: "
                + ", ".join(missing)
            )

        def scalar(name: str):
            value = np.asarray(state[name])
            if value.ndim != 0:
                raise ValueError(
                    f"Coordinate-map state entry '{name}' must be scalar."
                )
            return value.item()

        version = int(scalar("state_format_version"))
        if version != cls.state_format_version:
            raise ValueError(
                "Unsupported coordinate-map state format version "
                f"{version}; expected {cls.state_format_version}."
            )
        has_flux_normalization = bool(scalar("has_flux_normalization"))
        has_fixed_axis = bool(scalar("has_fixed_axis"))
        has_symmetry_tolerance = bool(scalar("has_symmetry_tolerance"))
        has_flux_constraint = bool(scalar("has_flux_constraint"))

        constraint_R = (
            np.asarray(state["flux_constraint_R"], dtype=np.float64).copy()
            if has_flux_constraint
            else None
        )
        constraint_z = (
            np.asarray(state["flux_constraint_z"], dtype=np.float64).copy()
            if has_flux_constraint
            else None
        )
        constraint_psi = (
            np.asarray(state["flux_constraint_psi"], dtype=np.float64).copy()
            if has_flux_constraint
            else None
        )
        return cls(
            psi=np.asarray(state["psi"], dtype=np.float64).copy(),
            theta=np.asarray(state["theta"], dtype=np.float64).copy(),
            R=np.asarray(state["R"], dtype=np.float64).copy(),
            z=np.asarray(state["z"], dtype=np.float64).copy(),
            nu=np.asarray(state["nu"], dtype=np.float64).copy(),
            max_mode=int(scalar("max_mode")),
            psi_axis=(
                float(scalar("psi_axis")) if has_flux_normalization else None
            ),
            psi_boundary=(
                float(scalar("psi_boundary"))
                if has_flux_normalization
                else None
            ),
            R_axis=float(scalar("R_axis")) if has_fixed_axis else None,
            z_axis=float(scalar("z_axis")) if has_fixed_axis else None,
            enforce_up_down_symmetry=bool(
                scalar("enforce_up_down_symmetry")
            ),
            symmetry_tolerance=(
                float(scalar("symmetry_tolerance"))
                if has_symmetry_tolerance
                else None
            ),
            flux_constraint_R=constraint_R,
            flux_constraint_z=constraint_z,
            flux_constraint_psi=constraint_psi,
            flux_constraint_tolerance=float(
                scalar("flux_constraint_tolerance")
            ),
            flux_constraint_max_iterations=int(
                scalar("flux_constraint_max_iterations")
            ),
            flux_constraint_minimum_radial_derivative=float(
                scalar("flux_constraint_minimum_radial_derivative")
            ),
        )

    def _radial_parameter(
        self,
        psi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the smooth fit coordinate and its physical-flux derivative."""
        physical_flux = np.asarray(psi, dtype=np.float64)
        if self.psi_axis is None:
            return physical_flux, np.ones_like(physical_flux)

        span = float(self.psi_boundary - self.psi_axis)
        normalized = (physical_flux - self.psi_axis) / span
        tolerance = 100.0 * np.finfo(np.float64).eps
        if np.any(normalized < -tolerance):
            raise ValueError(
                "Physical psi lies beyond the magnetic axis, where the "
                "sqrt-normalized radial coordinate is undefined."
            )
        rho = np.sqrt(np.maximum(normalized, 0.0))
        signed_rho = np.sign(span) * rho
        derivative = np.full_like(rho, np.inf)
        np.divide(
            1.0,
            2.0 * abs(span) * rho,
            out=derivative,
            where=rho > 0.0,
        )
        return signed_rho, derivative

    def _base_evaluate(
        self,
        field: str,
        psi: np.ndarray,
        theta: np.ndarray,
        *,
        dpsi: int = 0,
        dtheta: int = 0,
    ) -> np.ndarray:
        """Evaluate the underlying Fourier--radial spline map."""
        if field not in self._splines:
            valid = ", ".join(self.field_names)
            raise ValueError(f"Unknown mapped field '{field}'. Expected one of {valid}.")
        if dpsi < 0 or dtheta < 0:
            raise ValueError("Derivative orders must be non-negative.")

        radial, angle = np.broadcast_arrays(
            np.asarray(psi, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
        )
        flat_psi = radial.ravel()
        flat_theta = angle.ravel()
        radial_parameter, parameter_psi = self._radial_parameter(flat_psi)
        if dpsi > 1 and self.psi_axis is not None:
            raise ValueError(
                "Only first physical-flux derivatives are implemented for "
                "the sqrt-normalized radial fit coordinate."
            )
        if dpsi > 0 and np.any(~np.isfinite(parameter_psi)):
            raise ValueError(
                "Physical-flux derivatives of the coordinate map are "
                "singular at the magnetic axis."
            )
        coefficients = self._splines[field](radial_parameter, nu=dpsi)
        if dpsi == 1:
            coefficients = coefficients * parameter_psi[:, None]
        angular_factor = (1j * self.modes) ** dtheta
        phase = np.exp(
            1j
            * np.outer(
                flat_theta - self.theta_origin,
                self.modes,
            )
        )
        result = np.sum(coefficients * angular_factor[None, :] * phase, axis=1)
        result = np.real_if_close(result, tol=1000)
        if np.iscomplexobj(result):
            imaginary_scale = float(np.max(np.abs(np.imag(result))))
            real_scale = max(1.0, float(np.max(np.abs(np.real(result)))))
            if imaginary_scale > 1.0e-10 * real_scale:
                raise ValueError(
                    f"Spectral evaluation of {field} acquired a non-real component."
                )
            result = np.real(result)
        return np.asarray(result, dtype=np.float64).reshape(radial.shape)

    def _base_field_bundle(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
        *,
        derivative_orders: tuple[tuple[int, int], ...] = ((0, 0),),
    ) -> Dict[tuple[int, int], Dict[str, np.ndarray]]:
        """Evaluate all mapped fields while sharing spline/Fourier work.

        ``derivative_orders`` contains ``(dpsi, dtheta)`` pairs.  The return
        value is indexed first by that pair and then by ``R``, ``z``, or
        ``nu``.  Coordinate assembly uses this fused path; ``evaluate`` keeps
        the individual-field path so isolated public evaluations do not pay
        for fields they did not request.
        """
        if not derivative_orders:
            return {}
        unique_orders = tuple(dict.fromkeys(derivative_orders))
        for dpsi, dtheta in unique_orders:
            if dpsi < 0 or dtheta < 0:
                raise ValueError("Derivative orders must be non-negative.")
            if dpsi > 1 and self.psi_axis is not None:
                raise ValueError(
                    "Only first physical-flux derivatives are implemented for "
                    "the sqrt-normalized radial fit coordinate."
                )

        radial, angle = np.broadcast_arrays(
            np.asarray(psi, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
        )
        flat_psi = radial.ravel()
        flat_theta = angle.ravel()
        radial_parameter, parameter_psi = self._radial_parameter(flat_psi)
        if any(order[0] > 0 for order in unique_orders) and np.any(
            ~np.isfinite(parameter_psi)
        ):
            raise ValueError(
                "Physical-flux derivatives of the coordinate map are "
                "singular at the magnetic axis."
            )

        phase = np.exp(
            1j
            * np.outer(
                flat_theta - self.theta_origin,
                self.modes,
            )
        )
        coefficient_cache: Dict[int, np.ndarray] = {}
        output: Dict[tuple[int, int], Dict[str, np.ndarray]] = {}
        for dpsi, dtheta in unique_orders:
            coefficients = coefficient_cache.get(dpsi)
            if coefficients is None:
                coefficients = self._field_spline(radial_parameter, nu=dpsi)
                if dpsi == 1:
                    coefficients = coefficients * parameter_psi[:, None, None]
                coefficient_cache[dpsi] = coefficients
            angular_phase = phase * ((1j * self.modes) ** dtheta)[None, :]
            result = np.einsum(
                "nfm,nm->nf",
                coefficients,
                angular_phase,
                optimize=True,
            )
            result = np.real_if_close(result, tol=1000)
            if np.iscomplexobj(result):
                imaginary_scale = np.max(np.abs(np.imag(result)), axis=0)
                real_scale = np.maximum(
                    1.0,
                    np.max(np.abs(np.real(result)), axis=0),
                )
                bad = imaginary_scale > 1.0e-10 * real_scale
                if np.any(bad):
                    fields = ", ".join(
                        name
                        for name, invalid in zip(self.field_names, bad)
                        if invalid
                    )
                    raise ValueError(
                        "Fused spectral evaluation acquired a non-real "
                        f"component for {fields}."
                    )
                result = np.real(result)
            real_result = np.asarray(result, dtype=np.float64)
            output[(dpsi, dtheta)] = {
                name: real_result[:, field_index].reshape(radial.shape)
                for field_index, name in enumerate(self.field_names)
            }
        return output

    def _physical_flux_geometry(
        self,
        R: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate authoritative flux and its gradient on mapped geometry."""
        if self._flux_constraint_spline is None:
            raise RuntimeError("No authoritative flux constraint is configured.")
        radial_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            float(np.max(np.abs(self.flux_constraint_R))),
        )
        vertical_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            float(np.max(np.abs(self.flux_constraint_z))),
        )
        if (
            np.min(R) < self.flux_constraint_R[0] - radial_tolerance
            or np.max(R) > self.flux_constraint_R[-1] + radial_tolerance
            or np.min(z) < self.flux_constraint_z[0] - vertical_tolerance
            or np.max(z) > self.flux_constraint_z[-1] + vertical_tolerance
        ):
            raise ValueError(
                "Flux-constrained coordinate-map geometry leaves its "
                "authoritative R-z grid."
            )
        clipped_R = np.clip(
            R,
            self.flux_constraint_R[0],
            self.flux_constraint_R[-1],
        )
        clipped_z = np.clip(
            z,
            self.flux_constraint_z[0],
            self.flux_constraint_z[-1],
        )
        physical_psi = self._flux_constraint_spline.ev(
            clipped_R.ravel(),
            clipped_z.ravel(),
        ).reshape(R.shape)
        physical_psi_R = self._flux_constraint_spline.ev(
            clipped_R.ravel(),
            clipped_z.ravel(),
            dx=1,
        ).reshape(R.shape)
        physical_psi_z = self._flux_constraint_spline.ev(
            clipped_R.ravel(),
            clipped_z.ravel(),
            dy=1,
        ).reshape(R.shape)
        return clipped_R, clipped_z, physical_psi, physical_psi_R, physical_psi_z

    def _constraint_geometry(
        self,
        sigma: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return base geometry and the physical-flux gradient on it."""
        if self._flux_constraint_spline is None:
            raise RuntimeError("No authoritative flux constraint is configured.")
        base = self._base_field_bundle(sigma, theta)[(0, 0)]
        return self._physical_flux_geometry(base["R"], base["z"])

    def _solve_flux_constraint(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, float | int]]:
        """Relabel the base radial parameter onto authoritative physical psi."""
        target, angle = np.broadcast_arrays(
            np.asarray(psi, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
        )
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(angle)):
            raise ValueError("Flux-constrained coordinate samples must be finite.")
        cached = self._flux_constraint_cache
        if (
            cached is not None
            and target.shape == cached[0].shape
            and angle.shape == cached[1].shape
            and np.array_equal(target, cached[0])
            and np.array_equal(angle, cached[1])
        ):
            self.last_flux_constraint_solve_audit = dict(
                cached[3],
                cache_hit=1,
            )
            return cached[2], dict(cached[3])

        flat_target = target.reshape(-1)
        flat_angle = angle.reshape(-1)
        flat_sigma = target.copy().reshape(-1)
        flat_error = np.full(flat_sigma.shape, np.inf, dtype=np.float64)
        flat_F_sigma = np.full(flat_sigma.shape, np.nan, dtype=np.float64)
        active = np.ones(flat_sigma.shape, dtype=bool)
        tolerance_absolute = (
            self.flux_constraint_tolerance * self.flux_constraint_scale
        )
        normalized_residual = np.inf
        minimum_abs_F_sigma = np.inf
        iterations = 0
        converged = False
        newton_active_evaluations = 0
        newton_evaluation_rounds = 0
        newton_peak_active_points = int(flat_sigma.size)

        def evaluate_active(
            flat_indices: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            nonlocal newton_active_evaluations, newton_evaluation_rounds
            active_sigma = flat_sigma[flat_indices]
            active_angle = flat_angle[flat_indices]
            (
                _,
                _,
                physical_psi,
                physical_psi_R,
                physical_psi_z,
            ) = self._constraint_geometry(active_sigma, active_angle)
            radial_bundle = self._base_field_bundle(
                active_sigma,
                active_angle,
                derivative_orders=((1, 0),),
            )[(1, 0)]
            F_sigma_active = (
                physical_psi_R * radial_bundle["R"]
                + physical_psi_z * radial_bundle["z"]
            )
            if not np.all(np.isfinite(F_sigma_active)):
                raise ValueError(
                    "Flux-constrained radial relabelling produced a non-finite "
                    "physical-flux derivative."
                )
            minimum_active = float(np.min(np.abs(F_sigma_active)))
            if (
                minimum_active
                <= self.flux_constraint_minimum_radial_derivative
            ):
                raise ValueError(
                    "Flux-constrained radial relabelling is singular: "
                    f"min|F_sigma|={minimum_active:.3e}."
                )
            newton_active_evaluations += int(flat_indices.size)
            newton_evaluation_rounds += 1
            return physical_psi - flat_target[flat_indices], F_sigma_active

        for iterations in range(1, self.flux_constraint_max_iterations + 1):
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                converged = True
                break
            error_active, F_sigma_active = evaluate_active(active_indices)
            flat_error[active_indices] = error_active
            flat_F_sigma[active_indices] = F_sigma_active
            converged_active = np.abs(error_active) <= tolerance_absolute
            active[active_indices[converged_active]] = False
            remaining_indices = active_indices[~converged_active]
            if remaining_indices.size == 0:
                converged = True
                break
            update_active = (
                error_active[~converged_active]
                / F_sigma_active[~converged_active]
            )
            if not np.all(np.isfinite(update_active)):
                raise ValueError(
                    "Flux-constrained radial relabelling produced a non-finite "
                    "Newton update."
                )
            flat_sigma[remaining_indices] -= update_active

        sigma = flat_sigma.reshape(target.shape)
        error = flat_error.reshape(target.shape)
        F_sigma = flat_F_sigma.reshape(target.shape)
        minimum_abs_F_sigma = float(np.min(np.abs(flat_F_sigma)))
        normalized_residual = float(
            np.max(np.abs(flat_error)) / self.flux_constraint_scale
        )
        bounded_root_fallback_count = 0
        bounded_root_fallback_iterations = 0
        if not converged:
            # The fully vectorized Newton solve is normally both the fastest
            # and most accurate path.  Very occasionally, a point close to a
            # support endpoint oscillates at the interpolation roundoff floor
            # even though its one-dimensional physical-flux map remains
            # regular and monotone.  Resolve only those points with a bracketed
            # scalar solve; the same authoritative flux spline and unchanged
            # residual tolerance remain the acceptance criterion.
            failed = active.reshape(target.shape)
            radial_span = max(
                float(np.ptp(self.psi)),
                self.flux_constraint_scale,
                np.finfo(np.float64).tiny,
            )

            for flat_index in np.flatnonzero(failed.reshape(-1)):
                target_value = float(flat_target[flat_index])
                angle_value = float(flat_angle[flat_index])
                center = float(flat_sigma[flat_index])
                center_error = float(flat_error[flat_index])
                center_derivative = float(flat_F_sigma[flat_index])

                def scalar_residual(radial_value: float) -> float:
                    _, _, flux_value, _, _ = self._constraint_geometry(
                        np.asarray(radial_value, dtype=np.float64),
                        np.asarray(angle_value, dtype=np.float64),
                    )
                    return float(np.asarray(flux_value)) - target_value

                width = max(
                    2.0 * abs(center_error / center_derivative),
                    64.0 * np.finfo(np.float64).eps * radial_span,
                )
                bracket = None
                for _ in range(24):
                    lower = center - width
                    upper = center + width
                    lower_error = scalar_residual(lower)
                    upper_error = scalar_residual(upper)
                    if lower_error == 0.0:
                        bracket = (lower, lower)
                        break
                    if upper_error == 0.0:
                        bracket = (upper, upper)
                        break
                    if np.signbit(lower_error) != np.signbit(upper_error):
                        bracket = (lower, upper)
                        break
                    width *= 2.0
                if bracket is None:
                    raise ValueError(
                        "Flux-constrained radial relabelling could not bracket "
                        "a regular scalar root after vector Newton stalled: "
                        f"normalized residual="
                        f"{abs(center_error) / self.flux_constraint_scale:.3e}."
                    )
                if bracket[0] == bracket[1]:
                    root = bracket[0]
                    root_iterations = 0
                else:
                    root, result = brentq(
                        scalar_residual,
                        bracket[0],
                        bracket[1],
                        xtol=max(
                            0.05
                            * self.flux_constraint_tolerance
                            * self.flux_constraint_scale,
                            np.finfo(np.float64).tiny,
                        ),
                        rtol=8.0 * np.finfo(np.float64).eps,
                        maxiter=100,
                        full_output=True,
                        disp=False,
                    )
                    if not result.converged:
                        raise ValueError(
                            "Flux-constrained bounded scalar root did not converge."
                        )
                    root_iterations = int(result.iterations)
                flat_sigma[flat_index] = root
                bounded_root_fallback_count += 1
                bounded_root_fallback_iterations += root_iterations

            failed_indices = np.flatnonzero(failed.reshape(-1))
            final_error, final_F_sigma = evaluate_active(failed_indices)
            flat_error[failed_indices] = final_error
            flat_F_sigma[failed_indices] = final_F_sigma
            active[failed_indices] = np.abs(final_error) > tolerance_absolute
            sigma = flat_sigma.reshape(target.shape)
            error = flat_error.reshape(target.shape)
            F_sigma = flat_F_sigma.reshape(target.shape)
            minimum_abs_F_sigma = float(np.min(np.abs(flat_F_sigma)))
            if (
                minimum_abs_F_sigma
                <= self.flux_constraint_minimum_radial_derivative
            ):
                raise ValueError(
                    "Flux-constrained bounded roots are singular: "
                    f"min|F_sigma|={minimum_abs_F_sigma:.3e}."
                )
            normalized_residual = float(
                np.max(np.abs(flat_error)) / self.flux_constraint_scale
            )
            if normalized_residual > self.flux_constraint_tolerance:
                raise ValueError(
                    "Flux-constrained radial relabelling did not converge: "
                    f"normalized residual={normalized_residual:.3e}, "
                    f"tolerance={self.flux_constraint_tolerance:.3e}, "
                    f"Newton iterations={self.flux_constraint_max_iterations}, "
                    f"bounded roots={bounded_root_fallback_count}."
                )
        audit: Dict[str, float | int] = {
            "iterations": int(iterations),
            "normalized_residual": normalized_residual,
            "minimum_abs_F_sigma": minimum_abs_F_sigma,
            "bounded_root_fallback_count": bounded_root_fallback_count,
            "bounded_root_fallback_iterations": (
                bounded_root_fallback_iterations
            ),
            "newton_point_count": int(flat_sigma.size),
            "newton_active_evaluations": int(newton_active_evaluations),
            "newton_full_batch_equivalent_evaluations": int(
                flat_sigma.size * max(newton_evaluation_rounds, 1)
            ),
            "newton_evaluation_rounds": int(newton_evaluation_rounds),
            "newton_peak_active_points": newton_peak_active_points,
            "newton_final_active_points": int(np.count_nonzero(active)),
        }
        cache_target = target.copy()
        cache_angle = angle.copy()
        cache_sigma = sigma.copy()
        cache_target.setflags(write=False)
        cache_angle.setflags(write=False)
        cache_sigma.setflags(write=False)
        self._flux_constraint_cache = (
            cache_target,
            cache_angle,
            cache_sigma,
            dict(audit),
        )
        self.last_flux_constraint_solve_audit = dict(audit, cache_hit=0)
        return cache_sigma, audit

    def _constraint_chain_rule(
        self,
        sigma: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return d(sigma)/d(psi) and d(sigma)/d(theta)|psi."""
        bundle = self._base_field_bundle(
            sigma,
            theta,
            derivative_orders=((0, 0), (1, 0), (0, 1)),
        )
        return self._constraint_chain_rule_from_base(
            bundle[(0, 0)],
            bundle[(1, 0)],
            bundle[(0, 1)],
        )

    def _constraint_chain_rule_from_base(
        self,
        values: Mapping[str, np.ndarray],
        sigma_derivatives: Mapping[str, np.ndarray],
        theta_derivatives: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the exact flux-constraint chain rule to a fused base bundle."""
        (
            _,
            _,
            _,
            physical_psi_R,
            physical_psi_z,
        ) = self._physical_flux_geometry(values["R"], values["z"])
        R_sigma = sigma_derivatives["R"]
        z_sigma = sigma_derivatives["z"]
        R_theta = theta_derivatives["R"]
        z_theta = theta_derivatives["z"]
        F_sigma = physical_psi_R * R_sigma + physical_psi_z * z_sigma
        minimum = float(np.min(np.abs(F_sigma)))
        if (
            not np.all(np.isfinite(F_sigma))
            or minimum <= self.flux_constraint_minimum_radial_derivative
        ):
            raise ValueError(
                "Flux-constrained coordinate derivatives are singular: "
                f"min|F_sigma|={minimum:.3e}."
            )
        F_theta = physical_psi_R * R_theta + physical_psi_z * z_theta
        return 1.0 / F_sigma, -F_theta / F_sigma

    def evaluate(
        self,
        field: str,
        psi: np.ndarray,
        theta: np.ndarray,
        *,
        dpsi: int = 0,
        dtheta: int = 0,
    ) -> np.ndarray:
        """Evaluate a mapped field or its exact first constrained derivative."""
        if self._flux_constraint_spline is None:
            return self._base_evaluate(
                field,
                psi,
                theta,
                dpsi=dpsi,
                dtheta=dtheta,
            )
        if (dpsi, dtheta) not in ((0, 0), (1, 0), (0, 1)):
            raise ValueError(
                "Flux-constrained maps implement values and exact first "
                "physical-psi or theta derivatives only."
            )
        radial, angle = np.broadcast_arrays(
            np.asarray(psi, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
        )
        if dpsi == 0 and dtheta == 0:
            sigma, _ = self._solve_flux_constraint(radial, angle)
            return self._base_evaluate(field, sigma, angle)
        _, psi_derivatives, theta_derivatives = (
            self._constrained_field_bundle(radial, angle)
        )
        if dpsi == 1:
            return psi_derivatives[field]
        return theta_derivatives[field]

    def values(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate ``R``, ``z``, and ``nu``."""
        if self._flux_constraint_spline is not None:
            radial, angle = np.broadcast_arrays(
                np.asarray(psi, dtype=np.float64),
                np.asarray(theta, dtype=np.float64),
            )
            sigma, _ = self._solve_flux_constraint(radial, angle)
            return self._base_field_bundle(sigma, angle)[(0, 0)]
        return self._base_field_bundle(psi, theta)[(0, 0)]

    def _constrained_field_bundle(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
    ]:
        """Evaluate all constrained fields and their first derivatives once."""
        radial, angle = np.broadcast_arrays(
            np.asarray(psi, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
        )
        sigma, _ = self._solve_flux_constraint(radial, angle)
        base = self._base_field_bundle(
            sigma,
            angle,
            derivative_orders=((0, 0), (1, 0), (0, 1)),
        )
        values = base[(0, 0)]
        sigma_derivatives = base[(1, 0)]
        base_theta_derivatives = base[(0, 1)]
        sigma_psi, sigma_theta = self._constraint_chain_rule_from_base(
            values,
            sigma_derivatives,
            base_theta_derivatives,
        )
        psi_derivatives: Dict[str, np.ndarray] = {}
        theta_derivatives: Dict[str, np.ndarray] = {}
        for name in self.field_names:
            field_sigma = sigma_derivatives[name]
            field_theta = base_theta_derivatives[name]
            psi_derivatives[name] = field_sigma * sigma_psi
            theta_derivatives[name] = (
                field_theta + field_sigma * sigma_theta
            )
        return values, psi_derivatives, theta_derivatives

    def _first_field_derivatives(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
    ]:
        """Evaluate values and first map derivatives through one shared path."""
        if self._flux_constraint_spline is not None:
            return self._constrained_field_bundle(psi, theta)
        base = self._base_field_bundle(
            psi,
            theta,
            derivative_orders=((0, 0), (1, 0), (0, 1)),
        )
        return base[(0, 0)], base[(1, 0)], base[(0, 1)]

    def direct_differentials(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> CoordinateTangents:
        """Evaluate mapped values and coordinate tangents, but no reciprocals.

        This is the low-allocation path for callers that will modify or invert
        the direct basis themselves.  It is numerically identical to the
        ``values`` and ``direct`` members returned by :meth:`differentials`.
        """
        values, psi_derivatives, theta_derivatives = (
            self._first_field_derivatives(psi, theta)
        )
        R = values["R"]
        direct = np.zeros(R.shape + (3, 3), dtype=np.float64)
        # Rows are orthonormal cylindrical components (e_R, e_phi, e_z);
        # columns are the coordinate tangents (psi, theta, zeta).
        direct[..., 0, 0] = psi_derivatives["R"]
        direct[..., 0, 1] = theta_derivatives["R"]
        direct[..., 1, 0] = -R * psi_derivatives["nu"]
        direct[..., 1, 1] = -R * theta_derivatives["nu"]
        direct[..., 1, 2] = R
        direct[..., 2, 0] = psi_derivatives["z"]
        direct[..., 2, 1] = theta_derivatives["z"]
        return CoordinateTangents(values=values, direct=direct)

    def differentials(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> CoordinateDifferentials:
        """Evaluate the coordinate basis and reciprocal metric tensors."""
        values, psi_derivatives, theta_derivatives = (
            self._first_field_derivatives(psi, theta)
        )
        R = values["R"]
        R_psi = psi_derivatives["R"]
        R_theta = theta_derivatives["R"]
        z_psi = psi_derivatives["z"]
        z_theta = theta_derivatives["z"]
        nu_psi = psi_derivatives["nu"]
        nu_theta = theta_derivatives["nu"]

        shape = R.shape
        use_compiled_differentials = (
            axisymmetric_differential_kernel is not None
            and _numba_config is not None
            and not _numba_config.DISABLE_JIT
            and R.size >= _COMPILED_DIFFERENTIAL_MINIMUM_POINTS
        )
        if use_compiled_differentials:
            compiled = axisymmetric_differential_kernel(
                np.ascontiguousarray(R.reshape(-1)),
                np.ascontiguousarray(R_psi.reshape(-1)),
                np.ascontiguousarray(R_theta.reshape(-1)),
                np.ascontiguousarray(z_psi.reshape(-1)),
                np.ascontiguousarray(z_theta.reshape(-1)),
                np.ascontiguousarray(nu_psi.reshape(-1)),
                np.ascontiguousarray(nu_theta.reshape(-1)),
            )
            direct = compiled[0].reshape(shape + (3, 3))
            inverse = compiled[1].reshape(shape + (3, 3))
            metric_covariant = compiled[2].reshape(shape + (3, 3))
            metric_contravariant = compiled[3].reshape(shape + (3, 3))
            jacobian = compiled[4].reshape(shape)
            if np.any(~np.isfinite(jacobian)) or np.any(
                np.isclose(jacobian, 0.0)
            ):
                raise ValueError(
                    "Coordinate map has a singular or non-finite Jacobian."
                )
            self.last_differential_backend = "numba"
            return CoordinateDifferentials(
                values=values,
                direct=direct,
                inverse=inverse,
                metric_covariant=metric_covariant,
                metric_contravariant=metric_contravariant,
                jacobian=jacobian,
            )

        self.last_differential_backend = "numpy"
        direct = np.zeros(shape + (3, 3), dtype=np.float64)
        # Rows are orthonormal cylindrical components (e_R, e_phi, e_z);
        # columns are the coordinate tangents (psi, theta, zeta).
        direct[..., 0, 0] = R_psi
        direct[..., 0, 1] = R_theta
        direct[..., 1, 0] = -R * nu_psi
        direct[..., 1, 1] = -R * nu_theta
        direct[..., 1, 2] = R
        direct[..., 2, 0] = z_psi
        direct[..., 2, 1] = z_theta

        # For an axisymmetric map the direct matrix has the exact sparsity
        #
        #   [[R_psi,       R_theta,       0],
        #    [-R nu_psi,  -R nu_theta,    R],
        #    [z_psi,       z_theta,       0]].
        #
        # Exploit it explicitly instead of launching one dense 3x3 determinant
        # and inverse per sample.  Besides being faster, these expressions
        # ensure the inverse and both metrics come from the same poloidal
        # determinant rather than independent factorizations.
        poloidal_determinant = R_psi * z_theta - R_theta * z_psi
        jacobian = -R * poloidal_determinant
        if np.any(~np.isfinite(jacobian)) or np.any(np.isclose(jacobian, 0.0)):
            raise ValueError("Coordinate map has a singular or non-finite Jacobian.")

        reciprocal_poloidal = 1.0 / poloidal_determinant
        inverse = np.zeros_like(direct)
        inverse[..., 0, 0] = z_theta * reciprocal_poloidal
        inverse[..., 0, 2] = -R_theta * reciprocal_poloidal
        inverse[..., 1, 0] = -z_psi * reciprocal_poloidal
        inverse[..., 1, 2] = R_psi * reciprocal_poloidal
        inverse[..., 2, 0] = (
            nu_psi * z_theta - nu_theta * z_psi
        ) * reciprocal_poloidal
        inverse[..., 2, 1] = 1.0 / R
        inverse[..., 2, 2] = (
            nu_theta * R_psi - nu_psi * R_theta
        ) * reciprocal_poloidal

        R_squared = R * R
        metric_covariant = np.empty_like(direct)
        metric_covariant[..., 0, 0] = (
            R_psi * R_psi + z_psi * z_psi + R_squared * nu_psi * nu_psi
        )
        metric_covariant[..., 0, 1] = (
            R_psi * R_theta
            + z_psi * z_theta
            + R_squared * nu_psi * nu_theta
        )
        metric_covariant[..., 1, 0] = metric_covariant[..., 0, 1]
        metric_covariant[..., 0, 2] = -R_squared * nu_psi
        metric_covariant[..., 2, 0] = metric_covariant[..., 0, 2]
        metric_covariant[..., 1, 1] = (
            R_theta * R_theta
            + z_theta * z_theta
            + R_squared * nu_theta * nu_theta
        )
        metric_covariant[..., 1, 2] = -R_squared * nu_theta
        metric_covariant[..., 2, 1] = metric_covariant[..., 1, 2]
        metric_covariant[..., 2, 2] = R_squared

        metric_contravariant = np.empty_like(direct)
        gradient_psi_R = inverse[..., 0, 0]
        gradient_psi_z = inverse[..., 0, 2]
        gradient_theta_R = inverse[..., 1, 0]
        gradient_theta_z = inverse[..., 1, 2]
        gradient_zeta_R = inverse[..., 2, 0]
        gradient_zeta_phi = inverse[..., 2, 1]
        gradient_zeta_z = inverse[..., 2, 2]
        metric_contravariant[..., 0, 0] = (
            gradient_psi_R**2 + gradient_psi_z**2
        )
        metric_contravariant[..., 0, 1] = (
            gradient_psi_R * gradient_theta_R
            + gradient_psi_z * gradient_theta_z
        )
        metric_contravariant[..., 1, 0] = metric_contravariant[..., 0, 1]
        metric_contravariant[..., 0, 2] = (
            gradient_psi_R * gradient_zeta_R
            + gradient_psi_z * gradient_zeta_z
        )
        metric_contravariant[..., 2, 0] = metric_contravariant[..., 0, 2]
        metric_contravariant[..., 1, 1] = (
            gradient_theta_R**2 + gradient_theta_z**2
        )
        metric_contravariant[..., 1, 2] = (
            gradient_theta_R * gradient_zeta_R
            + gradient_theta_z * gradient_zeta_z
        )
        metric_contravariant[..., 2, 1] = metric_contravariant[..., 1, 2]
        metric_contravariant[..., 2, 2] = (
            gradient_zeta_R**2
            + gradient_zeta_phi**2
            + gradient_zeta_z**2
        )
        return CoordinateDifferentials(
            values=values,
            direct=direct,
            inverse=inverse,
            metric_covariant=metric_covariant,
            metric_contravariant=metric_contravariant,
            jacobian=jacobian,
        )

    def solve_theta(
        self,
        *,
        psi: np.ndarray,
        R: np.ndarray,
        z: np.ndarray,
        initial_theta: np.ndarray,
        tolerance: float = 1.0e-11,
        max_iterations: int = 20,
    ) -> np.ndarray:
        """Find the closest mapped angle on each supplied physical flux surface."""
        radial, target_R, target_z, angle = np.broadcast_arrays(
            np.asarray(psi, dtype=np.float64),
            np.asarray(R, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            np.asarray(initial_theta, dtype=np.float64),
        )
        # Callers seed this solve with the ordinary counter-clockwise
        # geometrical angle.  Magnetic theta can run in either direction,
        # depending on field/COCOS orientation, while still increasing from
        # zero to 2*pi.  Reflect the seed for clockwise maps before Newton
        # iteration so it starts on the correct side of the surface.
        angle = self.theta_origin + self.angular_orientation * (
            angle - self.geometric_theta_origin
        )
        angle = np.mod(angle - self.theta_origin, _TWO_PI) + self.theta_origin
        maximum_step = 0.25 * np.pi
        theta_newton_active_evaluations = 0
        theta_newton_full_batch_equivalent_evaluations = 0
        theta_newton_evaluation_rounds = 0
        theta_newton_passes = 0
        coarse_reseed_count = 0
        bounded_root_fallback_count = 0

        def newton_pass(
            current: np.ndarray,
            iteration_limit: int = max_iterations,
            initial_converged: Optional[np.ndarray] = None,
            initial_update: Optional[np.ndarray] = None,
        ):
            nonlocal theta_newton_active_evaluations
            nonlocal theta_newton_full_batch_equivalent_evaluations
            nonlocal theta_newton_evaluation_rounds, theta_newton_passes
            current_shape = current.shape
            current_flat = np.array(current, copy=True).reshape(-1)
            if initial_converged is None:
                converged_flat = np.zeros(current_flat.shape, dtype=bool)
            else:
                converged_flat = np.asarray(
                    initial_converged,
                    dtype=bool,
                ).copy().reshape(-1)
            if initial_update is None:
                update_flat = np.full(
                    current_flat.shape,
                    np.inf,
                    dtype=np.float64,
                )
            else:
                update_flat = np.asarray(
                    initial_update,
                    dtype=np.float64,
                ).copy().reshape(-1)
            radial_flat = radial.reshape(-1)
            target_R_flat = target_R.reshape(-1)
            target_z_flat = target_z.reshape(-1)
            pass_point_count = int(np.count_nonzero(~converged_flat))
            theta_newton_passes += 1
            for _ in range(iteration_limit):
                active_indices = np.flatnonzero(~converged_flat)
                if active_indices.size == 0:
                    break
                active_radial = radial_flat[active_indices]
                active_angle = current_flat[active_indices]
                if self._flux_constraint_spline is None:
                    base = self._base_field_bundle(
                        active_radial,
                        active_angle,
                        derivative_orders=((0, 0), (0, 1), (0, 2)),
                    )
                    mapped_R = base[(0, 0)]["R"]
                    mapped_z = base[(0, 0)]["z"]
                    R_theta = base[(0, 1)]["R"]
                    z_theta = base[(0, 1)]["z"]
                else:
                    (
                        constrained_values,
                        _,
                        constrained_theta_derivatives,
                    ) = self._constrained_field_bundle(
                        active_radial,
                        active_angle,
                    )
                    mapped_R = constrained_values["R"]
                    mapped_z = constrained_values["z"]
                    R_theta = constrained_theta_derivatives["R"]
                    z_theta = constrained_theta_derivatives["z"]

                delta_R = mapped_R - target_R_flat[active_indices]
                delta_z = mapped_z - target_z_flat[active_indices]
                residual = delta_R * R_theta + delta_z * z_theta
                if self._flux_constraint_spline is None:
                    R_theta2 = base[(0, 2)]["R"]
                    z_theta2 = base[(0, 2)]["z"]
                    derivative = (
                        R_theta**2
                        + z_theta**2
                        + delta_R * R_theta2
                        + delta_z * z_theta2
                    )
                else:
                    # The constrained map supplies exact first derivatives.
                    # A positive Gauss--Newton curvature keeps angle inversion
                    # on that same map without inventing an inconsistent
                    # second-derivative approximation.
                    derivative = R_theta**2 + z_theta**2
                safe = (
                    np.isfinite(residual)
                    & np.isfinite(derivative)
                    & (derivative > 1.0e-15)
                )
                active_update = np.full(
                    active_indices.shape,
                    np.inf,
                    dtype=np.float64,
                )
                active_update[safe] = residual[safe] / derivative[safe]
                step = np.zeros(active_indices.shape, dtype=np.float64)
                step[safe] = np.clip(
                    active_update[safe],
                    -maximum_step,
                    maximum_step,
                )
                current_flat[active_indices] = np.mod(
                    active_angle - step - self.theta_origin,
                    _TWO_PI,
                ) + self.theta_origin
                update_flat[active_indices] = active_update
                converged_flat[active_indices] = safe & (
                    np.abs(active_update) <= tolerance
                )
                theta_newton_active_evaluations += int(active_indices.size)
                theta_newton_full_batch_equivalent_evaluations += pass_point_count
                theta_newton_evaluation_rounds += 1
                if np.all(converged_flat):
                    break
            return (
                current_flat.reshape(current_shape),
                converged_flat.reshape(current_shape),
                update_flat.reshape(current_shape),
            )

        angle, converged, update = newton_pass(angle)

        if not np.all(converged):
            # A local Newton solve can jump to the opposite side of a shaped
            # surface when the geometrical seed is imperfect. Re-seed only
            # failed points from a bounded full-turn distance scan, then
            # retain the same high-accuracy Newton convergence criterion.
            failed = np.flatnonzero(~converged.ravel())
            coarse_reseed_count = int(failed.size)
            coarse_theta = self.theta_origin + np.linspace(
                0.0,
                _TWO_PI,
                64,
                endpoint=False,
            )
            radial_failed = radial.ravel()[failed]
            target_R_failed = target_R.ravel()[failed]
            target_z_failed = target_z.ravel()[failed]
            if self._flux_constraint_spline is None:
                coarse_values = self._base_field_bundle(
                    radial_failed[:, None],
                    coarse_theta[None, :],
                )[(0, 0)]
                mapped_R = coarse_values["R"]
                mapped_z = coarse_values["z"]
            else:
                # Newton above follows the exact-flux constrained geometry.
                # Seed its retry from that same map; choosing a basin on the
                # latent band-limited map can put a shaped projected surface
                # on the opposite side of its physical point.
                constrained_values, _, _ = self._constrained_field_bundle(
                    radial_failed[:, None],
                    coarse_theta[None, :],
                )
                mapped_R = constrained_values["R"]
                mapped_z = constrained_values["z"]
            distance_squared = (
                (mapped_R - target_R_failed[:, None]) ** 2
                + (mapped_z - target_z_failed[:, None]) ** 2
            )
            best = np.argmin(distance_squared, axis=1)
            angle_flat = angle.ravel()
            angle_flat[failed] = coarse_theta[best]
            angle = angle_flat.reshape(angle.shape)
            angle, converged, update = newton_pass(
                angle,
                initial_converged=converged,
                initial_update=update,
            )

        if not np.all(converged):
            # Gauss--Newton may still miss a narrow basin on strongly shaped
            # surfaces because Cartesian audit points lie near, rather than
            # exactly on, the truncated coordinate contour.  Solve the actual
            # closest-point stationarity equation, (X-X_target).X_theta=0,
            # for only the remaining failures.  This uses the same exact
            # constrained geometry and the same final angular-update gate.
            failed = np.flatnonzero(~converged.ravel())
            coarse_count = 256
            coarse_step = _TWO_PI / coarse_count
            coarse_theta = self.theta_origin + np.arange(coarse_count) * coarse_step
            radial_failed = radial.ravel()[failed]
            target_R_failed = target_R.ravel()[failed]
            target_z_failed = target_z.ravel()[failed]
            if self._flux_constraint_spline is None:
                coarse_values = self._base_field_bundle(
                    radial_failed[:, None],
                    coarse_theta[None, :],
                )[(0, 0)]
                mapped_R = coarse_values["R"]
                mapped_z = coarse_values["z"]
            else:
                constrained_values, _, _ = self._constrained_field_bundle(
                    radial_failed[:, None], coarse_theta[None, :]
                )
                mapped_R = constrained_values["R"]
                mapped_z = constrained_values["z"]
            distance_squared = (
                (mapped_R - target_R_failed[:, None]) ** 2
                + (mapped_z - target_z_failed[:, None]) ** 2
            )
            best = np.argmin(distance_squared, axis=1)
            angle_flat = angle.ravel()
            converged_flat = converged.ravel()
            update_flat = update.ravel()
            for local_index, flat_index in enumerate(failed):
                radial_value = float(radial_failed[local_index])
                target_R_value = float(target_R_failed[local_index])
                target_z_value = float(target_z_failed[local_index])
                center = float(coarse_theta[best[local_index]])

                def scalar_stationarity(angle_value: float) -> tuple[float, float]:
                    if self._flux_constraint_spline is None:
                        scalar_bundle = self._base_field_bundle(
                            np.asarray(radial_value),
                            np.asarray(angle_value),
                            derivative_orders=((0, 0), (0, 1)),
                        )
                        mapped_R_value = scalar_bundle[(0, 0)]["R"]
                        mapped_z_value = scalar_bundle[(0, 0)]["z"]
                        R_theta_value = scalar_bundle[(0, 1)]["R"]
                        z_theta_value = scalar_bundle[(0, 1)]["z"]
                    else:
                        constrained_value, _, constrained_theta = (
                            self._constrained_field_bundle(
                            np.asarray(radial_value),
                            np.asarray(angle_value),
                        )
                        )
                        mapped_R_value = constrained_value["R"]
                        mapped_z_value = constrained_value["z"]
                        R_theta_value = constrained_theta["R"]
                        z_theta_value = constrained_theta["z"]
                    residual_value = float(
                        (mapped_R_value - target_R_value) * R_theta_value
                        + (mapped_z_value - target_z_value) * z_theta_value
                    )
                    curvature_value = float(
                        R_theta_value**2 + z_theta_value**2
                    )
                    return residual_value, curvature_value

                bracket = None
                for multiplier in (1.0, 2.0, 4.0, 8.0):
                    lower = center - multiplier * coarse_step
                    upper = center + multiplier * coarse_step
                    lower_residual, _ = scalar_stationarity(lower)
                    upper_residual, _ = scalar_stationarity(upper)
                    if lower_residual == 0.0:
                        bracket = (lower, lower)
                        break
                    if upper_residual == 0.0:
                        bracket = (upper, upper)
                        break
                    if np.signbit(lower_residual) != np.signbit(upper_residual):
                        bracket = (lower, upper)
                        break
                if bracket is None:
                    raise ValueError(
                        "Magnetic-angle closest-point stationarity root could "
                        "not be bracketed."
                    )
                if bracket[0] == bracket[1]:
                    root = bracket[0]
                else:
                    root = brentq(
                        lambda value: scalar_stationarity(value)[0],
                        bracket[0],
                        bracket[1],
                        xtol=4.0 * np.finfo(np.float64).eps,
                        rtol=8.0 * np.finfo(np.float64).eps,
                        maxiter=100,
                    )
                root_residual, root_curvature = scalar_stationarity(root)
                root_safe = (
                    np.isfinite(root_residual)
                    and np.isfinite(root_curvature)
                    and root_curvature > 1.0e-15
                )
                root_update = (
                    root_residual / root_curvature if root_safe else np.inf
                )
                angle_flat[flat_index] = float(root)
                update_flat[flat_index] = root_update
                converged_flat[flat_index] = (
                    root_safe and abs(root_update) <= tolerance
                )
                bounded_root_fallback_count += 1
            angle = np.mod(
                angle_flat.reshape(angle.shape) - self.theta_origin,
                _TWO_PI,
            ) + self.theta_origin
            converged = converged_flat.reshape(converged.shape)
            update = update_flat.reshape(update.shape)

        self.theta_inversion_audit = {
            "point_count": int(radial.size),
            "tolerance": float(tolerance),
            "max_iterations": int(max_iterations),
            "newton_passes": int(theta_newton_passes),
            "newton_evaluation_rounds": int(theta_newton_evaluation_rounds),
            "newton_active_evaluations": int(theta_newton_active_evaluations),
            "newton_full_batch_equivalent_evaluations": int(
                theta_newton_full_batch_equivalent_evaluations
            ),
            "coarse_reseed_count": int(coarse_reseed_count),
            "bounded_root_fallback_count": int(
                bounded_root_fallback_count
            ),
            "failed_count": int(np.count_nonzero(~converged)),
        }
        if not np.all(converged):
            worst = float(np.max(np.abs(update[~converged])))
            raise ValueError(
                "Magnetic-angle inversion did not converge after bounded "
                f"closest-point fallback; failed={int(np.count_nonzero(~converged))}, "
                f"worst update={worst:.3e}."
            )
        return angle
