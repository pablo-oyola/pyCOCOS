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
from scipy.interpolate import make_interp_spline


_TWO_PI = 2.0 * np.pi


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

        for name, array in arrays.items():
            coefficients = np.fft.fft(array, axis=1) / ntheta
            coefficients = coefficients[:, retained]
            self._splines[name] = make_interp_spline(
                radial_parameter,
                coefficients,
                k=radial_degree,
                axis=0,
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

    def evaluate(
        self,
        field: str,
        psi: np.ndarray,
        theta: np.ndarray,
        *,
        dpsi: int = 0,
        dtheta: int = 0,
    ) -> np.ndarray:
        """Evaluate one mapped field or a mixed derivative."""
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

    def values(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate ``R``, ``z``, and ``nu``."""
        return {
            name: self.evaluate(name, psi, theta)
            for name in self.field_names
        }

    def differentials(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> CoordinateDifferentials:
        """Evaluate the coordinate basis and reciprocal metric tensors."""
        values = self.values(psi, theta)
        R = values["R"]
        R_psi = self.evaluate("R", psi, theta, dpsi=1)
        R_theta = self.evaluate("R", psi, theta, dtheta=1)
        z_psi = self.evaluate("z", psi, theta, dpsi=1)
        z_theta = self.evaluate("z", psi, theta, dtheta=1)
        nu_psi = self.evaluate("nu", psi, theta, dpsi=1)
        nu_theta = self.evaluate("nu", psi, theta, dtheta=1)

        shape = R.shape
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

        jacobian = np.linalg.det(direct)
        if np.any(~np.isfinite(jacobian)) or np.any(np.isclose(jacobian, 0.0)):
            raise ValueError("Coordinate map has a singular or non-finite Jacobian.")
        inverse = np.linalg.inv(direct)
        metric_covariant = np.einsum("...ai,...aj->...ij", direct, direct)
        metric_contravariant = np.einsum(
            "...ia,...ja->...ij",
            inverse,
            inverse,
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
        converged = np.zeros(angle.shape, dtype=bool)

        for _ in range(max_iterations):
            mapped_R = self.evaluate("R", radial, angle)
            mapped_z = self.evaluate("z", radial, angle)
            R_theta = self.evaluate("R", radial, angle, dtheta=1)
            z_theta = self.evaluate("z", radial, angle, dtheta=1)
            R_theta2 = self.evaluate("R", radial, angle, dtheta=2)
            z_theta2 = self.evaluate("z", radial, angle, dtheta=2)

            delta_R = mapped_R - target_R
            delta_z = mapped_z - target_z
            residual = delta_R * R_theta + delta_z * z_theta
            derivative = (
                R_theta**2
                + z_theta**2
                + delta_R * R_theta2
                + delta_z * z_theta2
            )
            safe = np.abs(derivative) > 1.0e-15
            update = np.zeros_like(angle)
            update[safe] = residual[safe] / derivative[safe]
            angle = np.mod(
                angle - update - self.theta_origin,
                _TWO_PI,
            ) + self.theta_origin
            converged = np.abs(update) <= tolerance
            if np.all(converged):
                break

        if not np.all(converged):
            worst = float(np.max(np.abs(update[~converged])))
            raise ValueError(
                "Magnetic-angle inversion did not converge; "
                f"worst update={worst:.3e}."
            )
        return angle
