"""
Library to handle the magnetic equilibrium and compute magnetic coordinates
related to tokamaks.
"""

import numpy as np
import xarray as xr
from typing import Union, Optional, Tuple, Dict, Any, Literal
from findiff import FinDiff
from skimage import measure
from scipy.interpolate import (
    BSpline,
    CubicSpline,
    PchipInterpolator,
    RectBivariateSpline,
)
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.constants import mu_0

# Importing the internal utils.
from ..coordinates.registry import get_jacobian_function
from ..coordinates.compute_coordinates import compute_magnetic_coordinates
from ..coordinates.coordinate_map import SpectralCoordinateMap
from .magnetic_coordinates import magnetic_coordinates as MagneticCoordinates


_PROJECTED_FLUX_LABEL_TOLERANCE = 1.0e-8


def _reflection_paired_coordinate_psi_grid(
    *,
    tracing_R: np.ndarray,
    tracing_z: np.ndarray,
    tracing_psi: np.ndarray,
    public_R: np.ndarray,
    reflection_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent the paired projected flux on its complete vertical knot set.

    Reflection about an off-grid magnetic axis shifts every vertical spline
    knot.  Sampling the reflected average back onto only the original knots
    therefore cannot, in general, represent the field that generated the
    projected surfaces.  Retain the union of the tracing knots and their
    reflections over the common domain.  The radial knot set is unchanged by
    an up-down reflection, so the equilibrium's public radial grid remains
    sufficient.
    """
    radial = np.asarray(tracing_R, dtype=np.float64)
    vertical = np.asarray(tracing_z, dtype=np.float64)
    values = np.asarray(tracing_psi, dtype=np.float64)
    output_radial = np.asarray(public_R, dtype=np.float64)
    midplane = float(reflection_z)
    if (
        radial.ndim != 1
        or vertical.ndim != 1
        or output_radial.ndim != 1
        or values.shape != (radial.size, vertical.size)
        or radial.size < 4
        or vertical.size < 4
        or output_radial.size < 4
        or not np.all(np.isfinite(radial))
        or not np.all(np.isfinite(vertical))
        or not np.all(np.isfinite(output_radial))
        or not np.all(np.isfinite(values))
        or not np.isfinite(midplane)
        or np.any(np.diff(radial) <= 0.0)
        or np.any(np.diff(vertical) <= 0.0)
        or np.any(np.diff(output_radial) <= 0.0)
    ):
        raise ValueError("projected coordinate psi grids must be finite and increasing.")

    reflected_vertical = 2.0 * midplane - vertical
    common_lower = max(vertical[0], float(np.min(reflected_vertical)))
    common_upper = min(vertical[-1], float(np.max(reflected_vertical)))
    scale = max(1.0, abs(common_lower), abs(common_upper))
    bound_tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if common_upper <= common_lower + bound_tolerance:
        raise ValueError("projected coordinate psi has no reflected vertical domain.")
    candidates = np.concatenate(
        (
            vertical[
                (vertical >= common_lower - bound_tolerance)
                & (vertical <= common_upper + bound_tolerance)
            ],
            reflected_vertical[
                (reflected_vertical >= common_lower - bound_tolerance)
                & (reflected_vertical <= common_upper + bound_tolerance)
            ],
        )
    )
    output_vertical = np.unique(np.clip(candidates, common_lower, common_upper))
    # Remove only roundoff duplicates, then pair the surviving knots exactly.
    keep = np.concatenate(
        (
            np.asarray([True]),
            np.diff(output_vertical) > bound_tolerance,
        )
    )
    output_vertical = output_vertical[keep]
    output_vertical = 0.5 * (
        output_vertical + 2.0 * midplane - output_vertical[::-1]
    )
    if output_vertical.size < 4 or np.any(np.diff(output_vertical) <= 0.0):
        raise ValueError("projected coordinate psi knot union is not increasing.")

    source = RectBivariateSpline(
        radial,
        vertical,
        values,
        kx=min(3, radial.size - 1),
        ky=min(3, vertical.size - 1),
        s=0.0,
    )
    RR, ZZ = np.meshgrid(output_radial, output_vertical, indexing="ij")
    paired_values = 0.5 * (
        source.ev(RR.ravel(), ZZ.ravel())
        + source.ev(RR.ravel(), (2.0 * midplane - ZZ).ravel())
    ).reshape(RR.shape)
    return output_radial.copy(), output_vertical, paired_values


def _fit_projected_coordinate_psi_bridge(
    *,
    Rgrid: np.ndarray,
    zgrid: np.ndarray,
    psi_field: np.ndarray,
    surface_R: np.ndarray,
    surface_z: np.ndarray,
    surface_psi: np.ndarray,
    flux_scale: float,
    reflection_z: Optional[float] = None,
    tolerance: float = _PROJECTED_FLUX_LABEL_TOLERANCE,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Constrain the public projected-psi spline to its qualified surfaces.

    A direct downsample-and-refit can move the mapped surfaces off their
    physical flux labels.  Correct the tensor-product spline coefficients by
    the minimum scaled least-squares update that restores those labels.  The
    projected caller first supplies the complete reflection-knot union, for
    which this routine normally becomes an independent no-op verification.
    """
    radial_grid = np.asarray(Rgrid, dtype=np.float64)
    vertical_grid = np.asarray(zgrid, dtype=np.float64)
    values = np.asarray(psi_field, dtype=np.float64)
    mapped_R = np.asarray(surface_R, dtype=np.float64)
    mapped_z = np.asarray(surface_z, dtype=np.float64)
    targets = np.asarray(surface_psi, dtype=np.float64)
    scale = max(abs(float(flux_scale)), np.finfo(np.float64).tiny)
    tolerance = float(tolerance)
    if values.shape != (radial_grid.size, vertical_grid.size):
        raise ValueError(
            "projected coordinate psi field must match its public R-z grid."
        )
    if (
        mapped_R.shape != mapped_z.shape
        or mapped_R.ndim != 2
        or mapped_R.shape[0] != targets.size
    ):
        raise ValueError(
            "projected inverse geometry must have shape (surface, angle)."
        )
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(mapped_R))
        and np.all(np.isfinite(mapped_z))
        and np.all(np.isfinite(targets))
        and np.isfinite(tolerance)
        and tolerance > 0.0
    ):
        raise ValueError("projected coordinate psi bridge inputs must be finite.")
    if reflection_z is not None:
        reflection_z = float(reflection_z)
        if not np.isfinite(reflection_z):
            raise ValueError("projected coordinate psi reflection_z must be finite.")

    spline = RectBivariateSpline(
        radial_grid,
        vertical_grid,
        values,
        kx=min(3, radial_grid.size - 1),
        ky=min(3, vertical_grid.size - 1),
        s=0.0,
    )

    def surface_residual(candidate: RectBivariateSpline) -> np.ndarray:
        sampled = candidate.ev(
            mapped_R.ravel(),
            mapped_z.ravel(),
        ).reshape(mapped_R.shape)
        return np.max(
            np.abs(sampled - targets[:, None]),
            axis=1,
        ) / scale

    initial_residual = surface_residual(spline)
    symmetry_R = np.empty(0, dtype=np.float64)
    symmetry_z = np.empty(0, dtype=np.float64)
    reflected_symmetry_z = np.empty(0, dtype=np.float64)
    initial_symmetry_residual = 0.0
    if reflection_z is not None:
        grid_R, grid_z = np.meshgrid(
            radial_grid,
            vertical_grid,
            indexing="ij",
        )
        reflected_grid_z = 2.0 * reflection_z - grid_z
        in_reflected_domain = (
            (reflected_grid_z >= vertical_grid[0])
            & (reflected_grid_z <= vertical_grid[-1])
        )
        grid_flux = spline.ev(grid_R.ravel(), grid_z.ravel()).reshape(
            grid_R.shape
        )
        reflected_grid_flux = spline.ev(
            grid_R.ravel(),
            reflected_grid_z.ravel(),
        ).reshape(grid_R.shape)
        target_min = float(np.min(targets))
        target_max = float(np.max(targets))
        symmetry_domain = (
            in_reflected_domain
            & (grid_flux >= target_min)
            & (grid_flux <= target_max)
            & (reflected_grid_flux >= target_min)
            & (reflected_grid_flux <= target_max)
        )
        symmetry_R = grid_R[symmetry_domain]
        symmetry_z = grid_z[symmetry_domain]
        reflected_symmetry_z = reflected_grid_z[symmetry_domain]
        if symmetry_R.size:
            initial_symmetry_residual = float(
                np.max(
                    np.abs(
                        grid_flux[symmetry_domain]
                        - reflected_grid_flux[symmetry_domain]
                    )
                ) / scale
            )
    if (
        float(np.max(initial_residual)) <= tolerance
        and initial_symmetry_residual <= tolerance
    ):
        return values.copy(), {
            "initial_residual": initial_residual,
            "final_residual": initial_residual.copy(),
            "initial_symmetry_residual": initial_symmetry_residual,
            "final_symmetry_residual": initial_symmetry_residual,
            "relative_grid_correction": 0.0,
            "solver_stop_code": 0,
            "solver_iterations": 0,
        }

    # The coordinate tables close periodically, so omit their duplicate last
    # angular point from the constraint system while retaining it in the
    # independent final residual check above and below.
    constraint_R = mapped_R
    constraint_z = mapped_z
    geometry_scale = max(
        1.0,
        float(np.ptp(mapped_R)),
        float(np.ptp(mapped_z)),
    )
    if (
        mapped_R.shape[1] > 1
        and np.allclose(
            mapped_R[:, -1],
            mapped_R[:, 0],
            rtol=0.0,
            atol=1.0e-12 * geometry_scale,
        )
        and np.allclose(
            mapped_z[:, -1],
            mapped_z[:, 0],
            rtol=0.0,
            atol=1.0e-12 * geometry_scale,
        )
    ):
        constraint_R = mapped_R[:, :-1]
        constraint_z = mapped_z[:, :-1]

    flat_R = constraint_R.ravel()
    flat_z = constraint_z.ravel()
    target_vector = np.repeat(targets, constraint_R.shape[1])
    radial_tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(radial_grid))),
    )
    vertical_tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(vertical_grid))),
    )
    if (
        np.min(flat_R) < radial_grid[0] - radial_tolerance
        or np.max(flat_R) > radial_grid[-1] + radial_tolerance
        or np.min(flat_z) < vertical_grid[0] - vertical_tolerance
        or np.max(flat_z) > vertical_grid[-1] + vertical_tolerance
    ):
        raise ValueError(
            "projected inverse geometry leaves the public coordinate psi grid."
        )
    flat_R = np.clip(flat_R, radial_grid[0], radial_grid[-1])
    flat_z = np.clip(flat_z, vertical_grid[0], vertical_grid[-1])

    knots_R, knots_z = spline.get_knots()
    degree_R, degree_z = spline.degrees
    width_R = degree_R + 1
    width_z = degree_z + 1
    coefficient_shape = (
        knots_R.size - degree_R - 1,
        knots_z.size - degree_z - 1,
    )

    def tensor_design(
        sample_R: np.ndarray,
        sample_z: np.ndarray,
    ) -> csr_matrix:
        sample_radial = np.asarray(sample_R, dtype=np.float64).ravel()
        sample_vertical = np.asarray(sample_z, dtype=np.float64).ravel()
        basis_R = BSpline.design_matrix(
            sample_radial,
            knots_R,
            degree_R,
        ).tocsr()
        basis_z = BSpline.design_matrix(
            sample_vertical,
            knots_z,
            degree_z,
        ).tocsr()
        if not (
            np.all(np.diff(basis_R.indptr) == width_R)
            and np.all(np.diff(basis_z.indptr) == width_z)
        ):
            raise RuntimeError("Unexpected tensor-product spline basis layout.")
        indices_R = basis_R.indices.reshape(sample_radial.size, width_R)
        indices_z = basis_z.indices.reshape(sample_vertical.size, width_z)
        data_R = basis_R.data.reshape(sample_radial.size, width_R)
        data_z = basis_z.data.reshape(sample_vertical.size, width_z)
        columns = (
            indices_R[:, :, None] * coefficient_shape[1]
            + indices_z[:, None, :]
        ).reshape(-1)
        data = (data_R[:, :, None] * data_z[:, None, :]).reshape(-1)
        rows = np.repeat(
            np.arange(sample_radial.size, dtype=np.int64),
            width_R * width_z,
        )
        return csr_matrix(
            (data, (rows, columns)),
            shape=(sample_radial.size, int(np.prod(coefficient_shape))),
        )

    design = tensor_design(flat_R, flat_z)

    coefficients = np.asarray(spline.get_coeffs(), dtype=np.float64)
    label_right_hand_side = (
        target_vector - design @ coefficients
    ) / scale
    # Flux labels are the hard physical contract. Reflection rows regularize
    # the otherwise underdetermined public-grid correction without allowing
    # it to amplify the interpolation-limited field asymmetry between fitted
    # surfaces.
    label_weight = 1000.0 if symmetry_R.size else 1.0
    design_blocks = [label_weight * design]
    right_hand_side_blocks = [label_weight * label_right_hand_side]
    if symmetry_R.size:
        symmetry_design = tensor_design(symmetry_R, symmetry_z) - tensor_design(
            symmetry_R,
            reflected_symmetry_z,
        )
        design_blocks.append(symmetry_design)
        right_hand_side_blocks.append(
            -(symmetry_design @ coefficients) / scale
        )
    combined_design = vstack(design_blocks, format="csr")
    right_hand_side = np.concatenate(right_hand_side_blocks)
    solution = lsqr(
        combined_design,
        right_hand_side,
        atol=1.0e-14,
        btol=1.0e-14,
        iter_lim=4000,
    )
    corrected_coefficients = coefficients + scale * solution[0]

    grid_basis_R = BSpline.design_matrix(
        radial_grid,
        knots_R,
        degree_R,
    )
    grid_basis_z = BSpline.design_matrix(
        vertical_grid,
        knots_z,
        degree_z,
    )
    corrected_values = np.asarray(
        grid_basis_R
        @ corrected_coefficients.reshape(coefficient_shape)
        @ grid_basis_z.T,
        dtype=np.float64,
    )
    corrected_spline = RectBivariateSpline(
        radial_grid,
        vertical_grid,
        corrected_values,
        kx=degree_R,
        ky=degree_z,
        s=0.0,
    )
    final_residual = surface_residual(corrected_spline)
    maximum_final_residual = float(np.max(final_residual))
    if maximum_final_residual > tolerance:
        raise ValueError(
            "public projected coordinate psi bridge no longer matches its "
            "physical-flux labels: "
            f"residual={maximum_final_residual:.3e}"
        )
    final_symmetry_residual = 0.0
    if symmetry_R.size:
        final_symmetry_residual = float(
            np.max(
                np.abs(
                    corrected_spline.ev(symmetry_R, symmetry_z)
                    - corrected_spline.ev(
                        symmetry_R,
                        reflected_symmetry_z,
                    )
                )
            ) / scale
        )
        symmetry_limit = max(tolerance, initial_symmetry_residual)
        if final_symmetry_residual > symmetry_limit * (1.0 + 1.0e-8):
            raise ValueError(
                "public projected coordinate psi bridge amplified its "
                "reflection asymmetry in the fitted annulus: "
                f"initial={initial_symmetry_residual:.3e}, "
                f"final={final_symmetry_residual:.3e}"
            )
    return corrected_values, {
        "initial_residual": initial_residual,
        "final_residual": final_residual,
        "initial_symmetry_residual": initial_symmetry_residual,
        "final_symmetry_residual": final_symmetry_residual,
        "relative_grid_correction": float(
            np.max(np.abs(corrected_values - values)) / scale
        ),
        "solver_stop_code": int(solution[1]),
        "solver_iterations": int(solution[2]),
    }


def _extend_radial_support(
    core: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    guard_surfaces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add hidden same-spacing guard nodes without changing the core grid."""
    values = np.asarray(core, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("radial core grid must be finite and strictly increasing.")
    if isinstance(guard_surfaces, bool) or not isinstance(
        guard_surfaces,
        (int, np.integer),
    ):
        raise TypeError("radial_guard_surfaces must be an integer.")
    guard_count = int(guard_surfaces)
    if guard_count < 0:
        raise ValueError("radial_guard_surfaces must be non-negative.")
    if not (
        np.isfinite(lower_bound)
        and np.isfinite(upper_bound)
        and lower_bound < upper_bound
        and lower_bound <= values[0] < values[-1] <= upper_bound
    ):
        raise ValueError("radial support bounds must contain the core grid.")

    left_step = values[1] - values[0]
    right_step = values[-1] - values[-2]
    left = values[0] - left_step * np.arange(guard_count, 0, -1)
    right = values[-1] + right_step * np.arange(1, guard_count + 1)
    left = left[left > lower_bound]
    right = right[right < upper_bound]
    support = np.concatenate((left, values, right))
    core_indices = left.size + np.arange(values.size, dtype=np.int64)
    if not np.array_equal(support[core_indices], values):
        raise RuntimeError("radial support construction changed the requested core.")
    return support, core_indices


def _normalized_resolvable_axis_flux(
    psi_at_axis: float,
    *,
    psi_axis: float,
    psi_boundary: float,
) -> float:
    """Return the non-negative normalized flux represented at the axis."""
    values = np.asarray(
        [psi_at_axis, psi_axis, psi_boundary],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("axis and boundary flux values must be finite.")
    span = float(psi_boundary - psi_axis)
    if abs(span) <= np.finfo(np.float64).tiny:
        raise ValueError("axis and boundary flux values must be distinct.")
    normalized = float((psi_at_axis - psi_axis) / span)
    if normalized > 1.0:
        raise ValueError(
            "the interpolated axis flux lies beyond the boundary flux."
        )
    return float(np.clip(normalized, 0.0, 1.0))


def _outboard_midplane_seeds(
    R: np.ndarray,
    psi: np.ndarray,
    target_rho: np.ndarray,
    *,
    psi_axis: float,
    psi_boundary: float,
) -> np.ndarray:
    """Invert the monotone outboard flux branch without cubic overshoot."""
    radial = np.asarray(R, dtype=np.float64)
    flux = np.asarray(psi, dtype=np.float64)
    targets = np.asarray(target_rho, dtype=np.float64)
    if (
        radial.ndim != 1
        or flux.shape != radial.shape
        or targets.ndim != 1
        or radial.size < 2
        or not np.all(np.isfinite(radial))
        or not np.all(np.isfinite(flux))
        or not np.all(np.isfinite(targets))
        or np.any(np.diff(radial) <= 0.0)
        or np.any(np.diff(targets) <= 0.0)
    ):
        raise ValueError(
            "outboard seed inversion requires finite increasing radial grids."
        )
    span = float(psi_boundary - psi_axis)
    if not np.isfinite(span) or abs(span) <= np.finfo(np.float64).tiny:
        raise ValueError("axis and boundary flux values must be distinct.")

    normalized = (flux - float(psi_axis)) / span
    rho = np.sqrt(np.clip(normalized, 0.0, None))
    monotone_rho = np.maximum.accumulate(rho)
    scale = max(1.0, float(np.max(np.abs(monotone_rho))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    keep = np.concatenate(
        ([True], np.diff(monotone_rho) > tolerance)
    )
    rho_nodes = monotone_rho[keep]
    radial_nodes = radial[keep]
    if rho_nodes.size < 2:
        raise ValueError("outboard midplane flux does not form an invertible branch.")
    if (
        targets[0] < rho_nodes[0] - tolerance
        or targets[-1] > rho_nodes[-1] + tolerance
    ):
        raise ValueError(
            "requested radial labels leave the resolvable outboard flux branch: "
            f"requested=[{targets[0]:.6g}, {targets[-1]:.6g}], "
            f"available=[{rho_nodes[0]:.6g}, {rho_nodes[-1]:.6g}]."
        )

    seeds = np.asarray(
        PchipInterpolator(
            rho_nodes,
            radial_nodes,
            extrapolate=False,
        )(targets),
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(seeds))
        or np.any(np.diff(seeds) <= 0.0)
        or seeds[0] < radial[0] - tolerance
        or seeds[-1] > radial[-1] + tolerance
    ):
        raise ValueError("outboard midplane seed inversion produced invalid seeds.")
    return seeds


def _inverse_toroidal_derivatives(
    dpsi_dR,
    dpsi_dZ,
    dtheta_dR,
    dtheta_dZ,
    dnu_dR,
    dnu_dZ,
):
    r"""Return inverse toroidal derivatives for ``zeta = phi + nu``.

    The denominator here is the direct two-dimensional determinant

    ``D_RZ = det(partial(psi, theta) / partial(R, Z))``.

    It is distinct from the signed physical coordinate Jacobian

    ``J = [grad(psi) . (grad(theta) x grad(zeta))]**-1 = -R / D_RZ``.

    In particular, ``phi = zeta - nu(psi, theta)`` makes
    ``partial(phi)/partial(zeta)`` exactly one.
    """
    direct_det = dpsi_dR * dtheta_dZ - dpsi_dZ * dtheta_dR
    with np.errstate(divide="ignore", invalid="ignore"):
        dphi_dpsi = (
            dtheta_dR * dnu_dZ - dtheta_dZ * dnu_dR
        ) / direct_det
        dphi_dtheta = (
            dpsi_dZ * dnu_dR - dpsi_dR * dnu_dZ
        ) / direct_det
    dphi_dzeta = np.ones_like(direct_det, dtype=np.float64)
    return direct_det, dphi_dpsi, dphi_dtheta, dphi_dzeta


def _require_matplotlib_pyplot():
    """
    Import matplotlib pyplot lazily for plotting helpers.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for equilibrium plotting helpers. "
            "Install pyCOCOS with plotting extras: pip install 'pyCOCOS[plot]'."
        ) from exc
    return plt

# ----------------------------------------------------------------------------
# AUXILIAR FUNCTIONS TO COMPUTE MAGNETIC FIELD PROPERTIES.
# ----------------------------------------------------------------------------
def getFSprop(R: float, z: float):
    """
    For a given flux surface described by the corresponding (R, z) the
    following properties are computed:
        - Outermost radius (Raus)
        - Innermost radius (Rin)
        - Upper z-value (Zup)
        - Lower z-value (Zdown)
        - Geometrical radius (Rgeo)
        - Minor radius (ageo)
        - Upper triangularity (delta_u)
        - Lower triangularity (delta_d)
        - Average triangularity (delta)
        - Elongation (kappa)
    """

    R = np.atleast_1d(R)
    z = np.atleast_1d(z)

    # Checking the consistency of the inputs.
    if R.size != z.size:
        raise ValueError('R and z must have same size (%d != %d)'%(R.size,
                                                                   z.size))

    # Computing the properties.
    output = { 'Raus': R.max(),
               'Rin':  R.min(),
               'Zup':  z.max(),
               'Zdown': z.min()
             }

    output['Rgeo'] = (output['Raus'] + output['Rin'])/2.0
    output['ageo'] = (output['Raus'] - output['Rin'])/2.0

    # To compute the upper/lower triangularity we need to get major radius at
    # which we have the Zup and the Zdown, respectively.
    Rup   = R[z.argmax()]
    Rdown = R[z.argmin()]

    output['delta_u'] = (output['Rgeo'] - Rup)/output['ageo']
    output['delta_d'] = (output['Rgeo'] - Rdown)/output['ageo']
    output['delta']   = (output['delta_u'] + output['delta_d'])/2.0
    output['kappa']   = (output['Zup'] - output['Zdown'])/output['ageo']


    return output

def get_currents(R: float, z: float, br: float, bz: float, bphi: float):
    """
    Compute the currents flowing in the plasma for a given flux surface.

    :param R: Major radius.
    :param z: Vertical coordinate.
    :param br: Radial magnetic field.
    :param bz: Vertical magnetic field.
    :param bphi: Toroidal magnetic field.
    :return: Dictionary with the computed current values.
    """
    currents = dict()
    # We will use the Ampere law to determine the current.
    dr = R[1] - R[0]
    dz = z[1] - z[0]
    d_dr = FinDiff(0, dr, 1, acc=4)
    d_dz = FinDiff(1, dz, 1, acc=4)

    # The currents are the curl of the magnetic field.
    jr = - d_dz(bphi) / mu_0
    jz = + d_dr(R[:, None] * bphi) / (mu_0 * R[:, None])
    jphi = (1.0 / mu_0) * (- d_dr(bz) + d_dz(br))

    currents['jr'] = jr
    currents['jz'] = jz
    currents['jphi'] = jphi

    return currents


_CURVATURE_EPS = 1.0e-14

def _curvature_axisymmetric_findiff(R_1d: np.ndarray, dR: float, dZ: float,
                                    b_R: np.ndarray, b_phi: np.ndarray, b_Z: np.ndarray):
    """
    Compute kappa = (b dot grad)b in cylindrical coordinates for axisymmetric
    fields (d/dphi = 0) using FinDiff derivatives.
    """
    d_dR = FinDiff(0, dR, 1, acc=4)
    d_dZ = FinDiff(1, dZ, 1, acc=4)

    dbR_dR = d_dR(b_R)
    dbR_dZ = d_dZ(b_R)
    dbphi_dR = d_dR(b_phi)
    dbphi_dZ = d_dZ(b_phi)
    dbZ_dR = d_dR(b_Z)
    dbZ_dZ = d_dZ(b_Z)

    safe_R = R_1d[:, None].copy()
    safe_R[np.abs(safe_R) < _CURVATURE_EPS] = _CURVATURE_EPS

    b_dot_grad_bR = b_R * dbR_dR + b_Z * dbR_dZ
    b_dot_grad_bphi = b_R * dbphi_dR + b_Z * dbphi_dZ
    b_dot_grad_bZ = b_R * dbZ_dR + b_Z * dbZ_dZ

    kappa_R = b_dot_grad_bR - (b_phi * b_phi / safe_R)
    kappa_phi = b_dot_grad_bphi + (b_R * b_phi / safe_R)
    kappa_Z = b_dot_grad_bZ

    return kappa_R, kappa_phi, kappa_Z

# ----------------------------------------------------------------------------
# MAIN CLASS FUNCTION.
# ----------------------------------------------------------------------------
class equilibrium:
    """
    Container for magnetic equilibrium data and coordinate transformations.

    This class stores tokamak equilibrium data including magnetic field
    components, flux surfaces, and provides methods to compute magnetic
    coordinates for various coordinate systems.

    Parameters
    ----------
    rgrid : np.ndarray
        Radial grid points (major radius R)
    zgrid : np.ndarray
        Vertical grid points (height z)
    br : np.ndarray
        Radial component of magnetic field (2D: R x z)
    bz : np.ndarray
        Vertical component of magnetic field (2D: R x z)
    bphi : np.ndarray
        Toroidal component of magnetic field (2D: R x z)
    psi : np.ndarray
        Poloidal magnetic flux (2D: R x z)
    Raxis : float
        Radial position of magnetic axis
    zaxis : float
        Vertical position of magnetic axis
    psi_edge : float
        Poloidal flux at plasma boundary
    psi_ax : float
        Poloidal flux at magnetic axis
    phiclockwise : bool, optional
        Orientation of arrays supplied directly to this generic constructor.
        Default is False, the canonical internal COCOS-1 orientation.
    flux_normalization : {"Wb", "Wb/rad"}, optional
        Poloidal-flux normalization of the supplied arrays. Default is
        ``"Wb/rad"``.
    R_boundary, z_boundary : np.ndarray, optional
        Paired coordinates of a supplied plasma boundary. When both are
        provided they define the LCFS used by the equilibrium; otherwise the
        LCFS is reconstructed from the ``rhopol=1`` contour.

    Attributes
    ----------
    Rgrid : xr.DataArray
        Radial grid
    zgrid : xr.DataArray
        Vertical grid
    Bdata : xr.Dataset
        Magnetic field data (Br, Bz, Bphi, Babs, Bpol)
    fluxdata : xr.Dataset
        Flux surface data (psipol, rhopol)
    boundary : xr.Dataset
        Plasma boundary data
    Jdata : xr.Dataset
        Current density data

    Examples
    --------
    >>> from pycocos import Equilibrium
    >>> eq = Equilibrium(rgrid, zgrid, br, bz, bphi, psi, Raxis, zaxis, psi_edge, psi_ax)
    >>> mag_coords = eq.compute_coordinates(coordinate_system='boozer')

    """

    def __init__(
        self,
        rgrid: np.ndarray,
        zgrid: np.ndarray,
        br: np.ndarray,
        bz: np.ndarray,
        bphi: np.ndarray,
        psi: np.ndarray,
        Raxis: float,
        zaxis: float,
        psi_edge: float,
        psi_ax: float,
        phiclockwise: bool = False,
        flux_normalization: Literal["Wb", "Wb/rad"] = "Wb/rad",
        R_boundary: Optional[np.ndarray] = None,
        z_boundary: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize a generic equilibrium.

        Parameters
        ----------
        rgrid : np.ndarray
            Radial grid points (major radius R)
        zgrid : np.ndarray
            Vertical grid points (height z)
        br : np.ndarray
            Radial component of magnetic field (2D: R x z)
        bz : np.ndarray
            Vertical component of magnetic field (2D: R x z)
        bphi : np.ndarray
            Toroidal component of magnetic field (2D: R x z)
        psi : np.ndarray
            Poloidal magnetic flux (2D: R x z)
        Raxis : float
            Radial position of magnetic axis
        zaxis : float
            Vertical position of magnetic axis
        psi_edge : float
            Poloidal flux at plasma boundary
        psi_ax : float
            Poloidal flux at magnetic axis
        phiclockwise : bool, optional
            Toroidal-angle orientation of the supplied arrays. Default is
            False.
        flux_normalization : {"Wb", "Wb/rad"}, optional
            Poloidal-flux normalization of the supplied arrays. Default is
            ``"Wb/rad"``.
        R_boundary, z_boundary : np.ndarray, optional
            Paired coordinates of a supplied plasma boundary. Both arrays
            must be finite, one-dimensional, have the same length, and span
            the magnetic axis in both coordinate ranges. If omitted, the
            boundary is reconstructed from the ``rhopol=1`` contour.

        Raises
        ------
        ValueError
            If grid dimensions don't match field arrays or axis is outside domain
        """
        rgrid = np.atleast_1d(rgrid)
        zgrid = np.atleast_1d(zgrid)
        if not isinstance(phiclockwise, (bool, np.bool_)):
            raise TypeError("phiclockwise must be a boolean.")
        if flux_normalization not in ("Wb", "Wb/rad"):
            raise ValueError("flux_normalization must be either 'Wb' or 'Wb/rad'")

        self.phiclockwise = bool(phiclockwise)
        self.flux_normalization = flux_normalization
        
        # Store axis values for later use in _build_structured_data
        self._Raxis_init = Raxis
        self._zaxis_init = zaxis
        self._psi_ax_init = psi_ax
        self._psi_edge_init = psi_edge

        # Checking size consistency.
        self.nr = len(rgrid)
        self.nz = len(zgrid)

        if br.ndim > 2:
            raise ValueError(f'Dimension of Br is {br.ndim}, instead of 2!')

        if bz.ndim > 2:
            raise ValueError(f'Dimension of Bz is {bz.ndim}, instead of 2!')

        if bphi.ndim > 2:
            raise ValueError(f'Dimension of Bphi is {bphi.ndim}, instead of 2!')

        if psi.ndim > 2:
            raise ValueError(f'Dimension of Psi is {bphi.ndim}, instead of 2!')

        if br.shape[0] != self.nr:
            raise ValueError(f'First dimension of Br must be {self.nr}')

        if bz.shape[0] != self.nr:
            raise ValueError(f'First dimension of Bz must be {self.nr}')

        if bphi.shape[0] != self.nr:
            raise ValueError(f'First dimension of Bphi must be {self.nr}')

        if psi.shape[0] != self.nr:
            raise ValueError(f'First dimension of Psi must be {self.nr}')

        if br.shape[1] != self.nz:
            raise ValueError(f'Second dimension of Br must be {self.nz}')

        if bz.shape[1] != self.nz:
            raise ValueError(f'Second dimension of Bz must be {self.nz}')

        if bphi.shape[1] != self.nz:
            raise ValueError(f'Second dimension of Bphi must be {self.nz}')

        if psi.shape[1] != self.nz:
            raise ValueError(f'Second dimension of Psi must be {self.nz}')

        # Checking that the magnetic axis.
        if (Raxis < rgrid.min()) or (Raxis > rgrid.max()):
            raise ValueError(f'Magnetic axis must be within the domain: {Raxis}.')

        if (zaxis < zgrid.min()) or (zaxis > zgrid.max()):
            raise ValueError(f'Magnetic axis must be within the domain: {zaxis}.')

        # Storing the data internally.
        self.Rgrid = xr.DataArray(rgrid, dims=('R',),
                                  attrs={'name': 'R',
                                         'desc': 'Major radius',
                                         'short_name': 'Major radius',
                                         'units': 'm'})
        self.zgrid = xr.DataArray(zgrid, dims=('z',),
                                  attrs={'name': 'z',
                                         'desc': 'Height',
                                         'short_name': 'Height',
                                         'units': 'm'})


        # We create a dataset to store all the magnetic-field related 2D data.
        self.Bdata = xr.Dataset()


        self.Bdata['Br'] = xr.DataArray(br, coords=(self.Rgrid, self.zgrid),
                                            attrs={'name': 'Br',
                                                'desc': 'Radial magnetic field',
                                                'short_name': '$B_R$',
                                                'units': 'T'})
        self.Bdata['Bz'] = xr.DataArray(bz, coords=(self.Rgrid, self.zgrid),
                                            attrs={'name': 'Bz',
                                                'desc': 'Vertical magnetic field',
                                                'short_name': '$B_z$',
                                                'units': 'T'})
        self.Bdata['Bphi'] = xr.DataArray(bphi, coords=(self.Rgrid, self.zgrid),
                                                attrs={'name': 'Bphi',
                                                        'desc': 'Toroidal magnetic field',
                                                        'short_name': '$B_\\varphi$',
                                                        'units': 'T'})
        self.Bdata['Babs'] = np.sqrt(self.Bdata.Br**2 + self.Bdata.Bz**2 + self.Bdata.Bphi**2)
        self.Bdata.Babs.attrs['name'] = 'Babs'
        self.Bdata.Babs.attrs['units'] = self.Bdata.Br.attrs['units']
        self.Bdata.Babs.attrs['desc'] = 'Magnetic field strenght'
        self.Bdata.Babs.attrs['short_name'] = '$B_{abs}$'

        self.Bdata['Bpol'] = np.sqrt(self.Bdata.Br**2 + self.Bdata.Bz**2)# * np.sign(self.Bdata.Bz)
        self.Bdata.Bpol.attrs['name'] = 'Bpol'
        self.Bdata.Bpol.attrs['units'] = self.Bdata.Br.attrs['units']
        self.Bdata.Bpol.attrs['desc'] = 'Poloidal magnetic field'
        self.Bdata.Bpol.attrs['short_name'] = '$B_{pol}$'


        # Building the magnetic coordinates.
        self.fluxdata = xr.Dataset()

        self.fluxdata['psipol'] = xr.DataArray(psi, coords=(self.Rgrid, self.zgrid),
                                                attrs={'name': 'Psi',
                                                        'desc': 'Poloidal magnetic flux',
                                                        'short_name': '$\\Psi$',
                                                        'units': self.flux_normalization})

        psimax = psi_edge - psi_ax
        flux_tolerance = max(1.0e-10, 1.0e-3 * abs(psimax))

        # Checking consistency of the axis flux.
        psiax_intrp = self.fluxdata.psipol.interp(R=Raxis, z=zaxis, method='cubic')
        if np.abs(psiax_intrp - psi_ax) > flux_tolerance:
            raise ValueError('The specified magnetic axis is not ' +
                             'consistent with the input flux: ' +
                             '%f (evaluated) vs %f (input)' % (psiax_intrp, psi_ax))

        # With the psipol, we get the magnetic axis and the separatrix values
        # for the flux to finally get rhopol.
        self.fluxdata['rhopol'] = np.sqrt((self.fluxdata.psipol - psi_ax) / psimax)
        self.fluxdata.rhopol.attrs['units'] = ''
        self.fluxdata.rhopol.attrs['desc'] = 'Radial magnetic coordinate'
        self.fluxdata.rhopol.attrs['name'] = 'rhopol'
        self.fluxdata.rhopol.attrs['short_name'] = '$\\rho_{pol}$'

        supplied_boundary = R_boundary is not None or z_boundary is not None
        if supplied_boundary:
            if R_boundary is None or z_boundary is None:
                raise ValueError(
                    "R_boundary and z_boundary must be provided together."
                )
            R = np.array(R_boundary, dtype=np.float64, copy=True)
            z = np.array(z_boundary, dtype=np.float64, copy=True)
            if (
                R.ndim != 1
                or z.ndim != 1
                or R.shape != z.shape
                or R.size < 3
                or not np.all(np.isfinite(R))
                or not np.all(np.isfinite(z))
            ):
                raise ValueError(
                    "The supplied plasma boundary must contain matching finite "
                    "one-dimensional R and z arrays with at least three points."
                )
            if (
                np.min(R) >= Raxis
                or np.max(R) <= Raxis
                or np.min(z) >= zaxis
                or np.max(z) <= zaxis
            ):
                raise ValueError(
                    "The supplied plasma boundary must span the magnetic axis "
                    "within its R-z bounds."
                )
            if (
                np.min(R) < rgrid.min()
                or np.max(R) > rgrid.max()
                or np.min(z) < zgrid.min()
                or np.max(z) > zgrid.max()
            ):
                raise ValueError(
                    "The supplied plasma boundary must lie inside the "
                    "equilibrium grid."
                )
            boundary_source = "supplied"
        else:
            # Generic equilibria without an explicit LCFS retain the contour
            # reconstruction fallback.
            R, z = self.rhopol2rz((1.0,))
            if len(R) == 1:
                R = R[0]
                z = z[0]
            boundary_source = "rhopol_contour"

        self._boundary = xr.Dataset()
        self._boundary['R'] = xr.DataArray(R, dims=('idx',),
                                           attrs={'name': 'R',
                                                  'desc': 'LCFS Radii',
                                                  'short_name': 'R',
                                                  'units': 'm'})
        self._boundary['z'] = xr.DataArray(z, dims=('idx',),
                                           attrs={'name': 'z',
                                                  'desc': 'LCFS Heights',
                                                  'short_name': 'z',
                                                  'units': 'm'})

        # Checking consistency of the flux at the LCFS.
        psiedge_intrp = self.fluxdata.psipol.interp(R=self._boundary.R[0],
                                                    z=self._boundary.z[0], method='cubic')
        if np.abs(psiedge_intrp - psi_edge) > flux_tolerance:
            raise ValueError('The specified separatrix flux is not consistent with the equilibrium.')

        self._boundary.attrs['psi_bdy'] = psi_edge
        self._boundary.attrs['psi_ax'] = psi_ax
        self._boundary.attrs['psimax'] = psimax
        self._boundary.attrs['source'] = boundary_source

        # Getting the radius of the separatrix at the geometrical midplane.
        rgrid_max = float(self.Rgrid.values[-1])
        nr_fine = int((rgrid_max - float(Raxis)) / 1.0e-3)
        rgrid_fine = np.linspace(float(Raxis), rgrid_max, nr_fine)

        # Sometimes life is hard and the coils are close to our plasma
        # and there may be places outside the confined region where 
        # psipol < psi_edge, and the corresponding rhopol is an imaginary
        # number. We will use in that case linear interpolation instead
        # of cubic.
        if np.any(np.isnan(self.fluxdata.rhopol.values)):
            method = 'linear'
        else:
            method = 'cubic'

        rhop1d = self.fluxdata.rhopol.interp(R=rgrid_fine,
                                             z=zaxis,
                                             method=method).values
        idx = np.abs(rhop1d[rhop1d <= 1.0] - 1.0).argmin()
        Raus = rgrid_fine[idx]

        # We add now this variables to fluxdata
        self.fluxdata.attrs['Raxis'] = Raxis
        self.fluxdata.attrs['zaxis'] = zaxis
        self.fluxdata.attrs['Raus']  = Raus
        self.fluxdata.attrs['aminor'] = Raus - Raxis


        # Getting the magnetic field at the axis.
        self.Bdata['Baxis'] = self.Bdata.Babs.interp(R=Raxis, z=zaxis, method='cubic')
        self.Bdata.Baxis.attrs['name'] = 'Baxis'
        self.Bdata.Baxis.attrs['units'] = self.Bdata.Br.attrs['units']
        self.Bdata.Baxis.attrs['desc'] = 'Magnetic field at the axis'
        self.Bdata.Baxis.attrs['short_name'] = '$B_{ax}$'

        # Getting the current density.
        jdata = get_currents(self.Bdata.R.values, self.Bdata.z.values,
                             self.Bdata.Br.values, self.Bdata.Bz.values,
                             self.Bdata.Bphi.values)
        self.Jdata = xr.Dataset()
        self.Jdata['Jr'] = xr.DataArray(jdata['jr'], dims = ('R', 'z'),
                                        coords={'R': self.Bdata.R, 'z': self.Bdata.z},
                                        attrs={'name': 'Jr',
                                               'desc': 'Radial current density',
                                               'short_name': '$J_R$',
                                               'units': 'A/m$^2$'})
        self.Jdata['Jz'] = xr.DataArray(jdata['jz'], dims = ('R', 'z'),
                                        attrs={'name': 'Jz',
                                               'desc': 'Vertical current density',
                                               'short_name': '$J_Z$',
                                               'units': 'A/m$^2$'})
        self.Jdata['Jphi'] = xr.DataArray(jdata['jphi'], dims = ('R', 'z'),
                                           attrs={'name': 'Jphi',
                                                  'desc': 'Toroidal current density',
                                                  'short_name': '$J_{phi}$',
                                                  'units': 'A/m$^2$'})
        
        # We now evaluate the current at the axis.
        Jraxis = self.Jdata.Jr.interp(R=Raxis, z=zaxis, method='cubic').values
        Jzaxis = self.Jdata.Jz.interp(R=Raxis, z=zaxis, method='cubic').values
        Jphiaxis = self.Jdata.Jphi.interp(R=Raxis, z=zaxis, method='cubic').values
        Jaxis = np.sqrt(Jraxis**2 + Jzaxis**2 + Jphiaxis**2) * np.sign(Jphiaxis)
        self.Jdata.attrs['Jaxis'] = Jaxis

        # Curvature is computed on demand by make_curvature/compute_curvature_vector.
        self.Kdata = xr.Dataset()

        # Create structured data organization (Option A: sub-Datasets as views)
        # These provide convenient access while keeping backward compatibility
        self._build_structured_data()

        # Creating the plotting lists (kept for backward compatibility)
        self.plot_1d_names = dict()
        self.plot_2d_names = dict()

        # Adding the variables for the plotting.
        for ivar in self.Bdata.data_vars:
            if self.Bdata[ivar].values.ndim == 0:
                continue
            self.add_var(ivar, self.Bdata[ivar])

        for ivar in self.fluxdata.data_vars:
            if self.fluxdata[ivar].values.ndim == 0:
                continue
            self.add_var(ivar, self.fluxdata[ivar])
        
        for ivar in self.Jdata.data_vars:
            if self.Jdata[ivar].values.ndim == 0:
                continue
            self.add_var(ivar, self.Jdata[ivar])

    def _build_structured_data(self) -> None:
        """
        Build structured data organization as xr.Dataset views.
        
        Creates convenient sub-Datasets (grid, field, flux, profiles, geometry)
        that provide views/subsets of the underlying data for easier access.
        """
        # Get axis values from stored attributes
        Raxis_val = self._Raxis_init
        zaxis_val = self._zaxis_init
        psi_ax_val = self._psi_ax_init
        psi_bdy_val = self._psi_edge_init
        psimax_val = psi_bdy_val - psi_ax_val
        
        # Grid: R and z coordinates
        self._grid = xr.Dataset({
            'R': self.Rgrid,
            'z': self.zgrid,
        })
        
        # Field: magnetic field components
        self._field = xr.Dataset({
            'Br': self.Bdata.Br,
            'Bz': self.Bdata.Bz,
            'Bphi': self.Bdata.Bphi,
            'B': self.Bdata.Babs,  # Total magnetic field
            'Bpol': self.Bdata.Bpol,  # Poloidal magnetic field
        })
        
        # Flux: flux surfaces
        self._flux = xr.Dataset({
            'psi': self.fluxdata.psipol,
            'rho': self.fluxdata.rhopol,
        })
        
        # Geometry: axis and boundary
        self._geometry = xr.Dataset({
            'R_axis': xr.DataArray(Raxis_val, attrs={'name': 'R_axis', 'units': 'm',
                                                     'desc': 'Radial position of magnetic axis',
                                                     'short_name': '$R_{axis}$'}),
            'z_axis': xr.DataArray(zaxis_val, attrs={'name': 'z_axis', 'units': 'm',
                                                     'desc': 'Vertical position of magnetic axis',
                                                     'short_name': '$z_{axis}$'}),
            'R_boundary': self._boundary.R,
            'z_boundary': self._boundary.z,
        })
        self._geometry.attrs.update({
            'psi_ax': psi_ax_val,
            'psi_bdy': psi_bdy_val,
            'psimax': psimax_val,
        })
        
        # Profiles: 1D profiles (initially empty, populated by EQDSK or user)
        self._profiles = xr.Dataset()
        # Curvature: computed on demand by compute_curvature_vector/make_curvature
        if hasattr(self, "Kdata") and len(self.Kdata.data_vars) > 0:
            self._curvature = self.Kdata
        else:
            self._curvature = xr.Dataset()
        
        # Initialize profiles dict for tracking
        self._profiles_dict = {}

    @property
    def grid(self) -> xr.Dataset:
        """Grid coordinates (R, z)."""
        return self._grid
    
    @property
    def field(self) -> xr.Dataset:
        """Magnetic field components (Br, Bz, Bphi, B, Bpol)."""
        return self._field
    
    @property
    def flux(self) -> xr.Dataset:
        """Flux surfaces (psi, rho)."""
        return self._flux
    
    @property
    def geometry(self) -> xr.Dataset:
        """Geometric properties (axis, boundary)."""
        return self._geometry
    
    @property
    def profiles(self) -> xr.Dataset:
        """1D profiles (q, pres, fpol, etc.)."""
        return self._profiles

    @property
    def curvature(self) -> xr.Dataset:
        """Curvature vector components (kappa_R, kappa_phi, kappa_z, kappa_abs)."""
        return self._curvature
    
    # Direct property access for common quantities
    @property
    def R(self) -> xr.DataArray:
        """Radial grid coordinates."""
        return self.Rgrid
    
    @property
    def z(self) -> xr.DataArray:
        """Vertical grid coordinates."""
        return self.zgrid
    
    @property
    def Br(self) -> xr.DataArray:
        """Radial magnetic field component."""
        return self.Bdata.Br
    
    @property
    def Bz(self) -> xr.DataArray:
        """Vertical magnetic field component."""
        return self.Bdata.Bz
    
    @property
    def Bphi(self) -> xr.DataArray:
        """Toroidal magnetic field component."""
        return self.Bdata.Bphi
    
    @property
    def B(self) -> xr.DataArray:
        """Total magnetic field magnitude."""
        return self.Bdata.Babs
    
    @property
    def Bpol(self) -> xr.DataArray:
        """Poloidal magnetic field magnitude."""
        return self.Bdata.Bpol
    
    @property
    def psi(self) -> xr.DataArray:
        """Poloidal magnetic flux."""
        return self.fluxdata.psipol
    
    @property
    def rho(self) -> xr.DataArray:
        """Normalized poloidal flux coordinate."""
        return self.fluxdata.rhopol
    
    @property
    def R_axis(self) -> float:
        """Radial position of magnetic axis."""
        return float(self.geometry.R_axis.values)
    
    @property
    def z_axis(self) -> float:
        """Vertical position of magnetic axis."""
        return float(self.geometry.z_axis.values)
    
    @property
    def boundary(self) -> xr.Dataset:
        """Plasma boundary (LCFS) coordinates."""
        return self._boundary
    
    @property
    def axis(self) -> xr.Dataset:
        """Magnetic axis position."""
        return xr.Dataset({
            'R': self.geometry.R_axis,
            'z': self.geometry.z_axis,
        })
    
    # Profile properties (may be None if not loaded)
    @property
    def q(self) -> Optional[xr.DataArray]:
        """Safety factor profile (if available)."""
        return self._profiles.get('q', None)
    
    @property
    def pres(self) -> Optional[xr.DataArray]:
        """Pressure profile (if available)."""
        return self._profiles.get('pres', None)
    
    @property
    def fpol(self) -> Optional[xr.DataArray]:
        """F(psi) = R*B_phi profile (if available)."""
        return self._profiles.get('fpol', None)

    def add_var(
        self,
        varname: str,
        var: xr.DataArray,
        add_to_class: bool = False
    ) -> None:
        """
        Add a variable to the database of the class.

        The class keeps a database of variables that can be used to store
        any information. This method allows adding a new variable to the
        database and eases handling of plotting routines.

        Parameters
        ----------
        varname : str
            Name of the variable to add
        var : xr.DataArray
            Variable as DataArray with proper metadata (name, desc, units, short_name)
        add_to_class : bool, optional
            If True, also add as class attribute. Default is False

        Raises
        ------
        ValueError
            If variable name already exists or metadata is missing

        Examples
        --------
        >>> eq.add_var('custom_field', my_dataarray)
        """
        # Checking if the variable name already exists.
        if (varname in self.plot_1d_names) or (varname in self.plot_2d_names):
            raise ValueError('Variable name already exists in the database.')

        # Checking that the variable contains all the metadata.
        if not hasattr(var, 'attrs'):
            raise ValueError('Variable does not have the metadata.')

        minimalmetadata = ['name', 'desc', 'short_name', 'units']
        for ikey in minimalmetadata:
            if ikey not in var.attrs:
                raise ValueError('Variable does not have the metadata (%s)' % ikey)

        # Checking the size of the input variable.
        ndim = var.ndim
        if ndim == 1:
            self.plot_1d_names[varname] = var
        elif ndim == 2:
            self.plot_2d_names[varname] = var
        else:
            raise NotImplementedError('Only 1D and 2D variables are supported.')

        if add_to_class:
            self.__dict__[varname] = var

    def rhopol2rz(
        self,
        rhopol: Union[float, np.ndarray],
        return_all: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform rhopol into (R, z) trajectories.

        Parameters
        ----------
        rhopol : float or np.ndarray
            Value(s) of the rho poloidal to transform into (R, z) contours
        return_all : bool, optional
            If True, return all contour segments. Default is False

        Returns
        -------
        R : np.ndarray
            Radial coordinates of the contour(s)
        z : np.ndarray
            Vertical coordinates of the contour(s)

        Examples
        --------
        >>> R, z = eq.rhopol2rz(0.5)  # Get contour at rho=0.5
        >>> R, z = eq.rhopol2rz([0.3, 0.5, 0.7])  # Get multiple contours
        """
        rhopol = np.atleast_1d(rhopol)
        nrho = rhopol.size  # Number of contours to evaluate.

        # We generate the empty output.
        R = np.empty((nrho,), dtype=object)
        z = np.empty((nrho,), dtype=object)

        Rmin = self.Rgrid.values.min()
        dr   = self.Rgrid.values[1] - self.Rgrid.values[0]
        zmin = self.zgrid.values.min()
        dz   = self.zgrid.values[1] - self.zgrid.values[0]

        # Looking for the contours.
        for ii in range(nrho):
            aux = measure.find_contours(self.fluxdata.rhopol.values, rhopol[ii],
                                        fully_connected='high')

            idx = 0
            if len(aux) == 0:
                continue
            if return_all:
                R[ii] = [(Rmin + dr*aux[idx][:, 0]) for idx in range(len(aux))]
                z[ii] = [(zmin + dz*aux[idx][:, 1]) for idx in range(len(aux))]
            else:
                if len(aux) > 1:
                    # We have here to choose among different options, since
                    # several flux surfaces legs have been found.
                    # We take as a measure of the closedness of the flux surface
                    # the fact that the last and first point must the closest to
                    # each other possible.
                    dl = np.zeros((len(aux),), dtype=float)
                    for jj in range(len(aux)):
                        drr = (aux[jj][-1, 0] - aux[jj][0, 0])
                        dzz = (aux[jj][-1, 1] - aux[jj][0, 1])

                        dl[jj] = np.sqrt(drr**2 + dzz**2)

                    idx = dl.argmin()
                R[ii] = aux[idx][:, 0]
                z[ii] = aux[idx][:, 1]

                # Transforming the indices into variables R and z.
                R[ii] = Rmin + dr*R[ii]
                z[ii] = zmin + dz*z[ii]

        return R, z

    def __call__(
        self,
        R: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        grid: bool = False
    ) -> xr.Dataset:
        """
        Evaluate magnetic field at given position(s).

        This provides a callable interface similar to magnetic_coordinates:
        `eq(R, z)` returns magnetic field components.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s)
        z : float or np.ndarray
            Vertical coordinate(s)
        grid : bool, optional
            If True, create a grid from R and z. Default is False

        Returns
        -------
        xr.Dataset
            Dataset containing Br, Bz, Bphi, B, Bpol at the requested points

        Examples
        --------
        >>> B = eq(2.0, 0.0)  # Single point
        >>> B = eq([1.5, 2.0], [0.0, 0.1])  # Multiple points
        >>> B = eq(R_grid, z_grid, grid=True)  # Grid evaluation
        """
        return self.getB(R, z, grid=grid)

    def getB(
        self,
        Rin: Union[float, np.ndarray],
        zin: Union[float, np.ndarray],
        grid: bool = False
    ) -> xr.Dataset:
        """
        Evaluate the magnetic field at the input position(s).

        Parameters
        ----------
        Rin : float or np.ndarray
            Radial coordinate(s) where to evaluate B
        zin : float or np.ndarray
            Vertical coordinate(s) where to evaluate B
        grid : bool, optional
            If True, create a grid from Rin and zin. Default is False

        Returns
        -------
        xr.Dataset
            Dataset containing Br, Bz, Bphi, B, Bpol at the requested points

        Examples
        --------
        >>> B = eq.getB(2.0, 0.0)  # Evaluate at single point
        >>> B = eq.getB([1.5, 2.0, 2.5], [0.0, 0.1, 0.0])  # Evaluate at multiple points
        """
        # Creating the output dataset.
        output = xr.Dataset()

        Rin = np.atleast_1d(Rin)
        zin = np.atleast_1d(zin)

        output['R'] = xr.DataArray(Rin, attrs={'name': 'R',
                                               'units': 'm',
                                               'desc': 'Major radius'},
                                   dims=('R') if not grid else None)

        output['z'] = xr.DataArray(zin, attrs={'name': 'z',
                                               'units': 'm',
                                               'desc': 'Height'},
                                   dims=('z') if not grid else None)

        # Interpolate magnetic field components
        if grid:
            for ikey in ('Br', 'Bz', 'Bphi'):
                output[ikey] = self.Bdata[ikey].interp(R=Rin, z=zin,
                                                        method='cubic')
        else:
            for ikey in ('Br', 'Bz', 'Bphi'):
                intrp = RectBivariateSpline(self.Rgrid.values, self.zgrid.values,
                                            self.Bdata[ikey].values)
                tmp = intrp(Rin, zin, grid=False)
                dims = ('point',) if tmp.ndim == 1 else None
                output[ikey] = xr.DataArray(tmp, dims=dims)
                output[ikey].attrs.update(self.Bdata[ikey].attrs)

        # Compute derived quantities
        output['B'] = np.sqrt(output['Br']**2 + output['Bz']**2 + output['Bphi']**2)
        output['B'].attrs.update({
            'name': 'B',
            'units': 'T',
            'desc': 'Total magnetic field magnitude',
            'short_name': '$B$'
        })
        
        output['Bpol'] = np.sqrt(output['Br']**2 + output['Bz']**2)
        output['Bpol'].attrs.update({
            'name': 'Bpol',
            'units': 'T',
            'desc': 'Poloidal magnetic field magnitude',
            'short_name': '$B_{pol}$'
        })

        return output

    def flux_surface(
        self,
        rho: Union[float, np.ndarray]
    ) -> xr.Dataset:
        """
        Get flux surface contour(s) at given rho value(s).

        Parameters
        ----------
        rho : float or np.ndarray
            Normalized poloidal flux coordinate value(s)

        Returns
        -------
        xr.Dataset
            Dataset with R and z coordinates of the flux surface(s)

        Examples
        --------
        >>> surface = eq.flux_surface(0.5)  # Single flux surface
        >>> surfaces = eq.flux_surface([0.3, 0.5, 0.7])  # Multiple surfaces
        """
        R, z = self.rhopol2rz(rho, return_all=False)
        
        # Handle single vs multiple surfaces
        if isinstance(R, np.ndarray) and R.dtype == object:
            # Multiple surfaces
            datasets = []
            for i in range(len(R)):
                if R[i] is not None and len(R[i]) > 0:
                    datasets.append(xr.Dataset({
                        'R': xr.DataArray(R[i], dims=('idx',),
                                         attrs={'name': 'R', 'units': 'm',
                                                'desc': 'Radial coordinate',
                                                'short_name': 'R'}),
                        'z': xr.DataArray(z[i], dims=('idx',),
                                         attrs={'name': 'z', 'units': 'm',
                                                'desc': 'Vertical coordinate',
                                                'short_name': 'z'}),
                    }))
            if len(datasets) == 1:
                return datasets[0]
            # Return as list or concatenate - for now return first if single value
            return datasets[0] if len(datasets) > 0 else xr.Dataset()
        else:
            # Single surface
            if R is not None and len(R) > 0:
                return xr.Dataset({
                    'R': xr.DataArray(R, dims=('idx',),
                                     attrs={'name': 'R', 'units': 'm',
                                            'desc': 'Radial coordinate',
                                            'short_name': 'R'}),
                    'z': xr.DataArray(z, dims=('idx',),
                                     attrs={'name': 'z', 'units': 'm',
                                            'desc': 'Vertical coordinate',
                                            'short_name': 'z'}),
                })
            else:
                return xr.Dataset()

    def interpolate(
        self,
        R: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        variables: Optional[list] = None,
        grid: bool = False
    ) -> xr.Dataset:
        """
        Generic interpolation of equilibrium variables onto (R, z) points.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s)
        z : float or np.ndarray
            Vertical coordinate(s)
        variables : list of str, optional
            Variables to interpolate. If None, interpolates all 2D variables
            (Br, Bz, Bphi, psi, rho). Default is None
        grid : bool, optional
            If True, create a grid from R and z. Default is False

        Returns
        -------
        xr.Dataset
            Dataset containing interpolated variables

        Examples
        --------
        >>> data = eq.interpolate(2.0, 0.0)  # Single point
        >>> data = eq.interpolate([1.5, 2.0], [0.0, 0.1], variables=['psi', 'Br'])
        """
        if variables is None:
            variables = ['Br', 'Bz', 'Bphi', 'psi', 'rho']
        
        output = xr.Dataset()
        R_arr = np.atleast_1d(R)
        z_arr = np.atleast_1d(z)
        
        # Map variable names to their datasets
        var_map = {
            'Br': ('Bdata', 'Br'),
            'Bz': ('Bdata', 'Bz'),
            'Bphi': ('Bdata', 'Bphi'),
            'B': ('Bdata', 'Babs'),
            'Bpol': ('Bdata', 'Bpol'),
            'psi': ('fluxdata', 'psipol'),
            'rho': ('fluxdata', 'rhopol'),
        }
        
        for var_name in variables:
            if var_name in var_map:
                ds_name, var_key = var_map[var_name]
                source_ds = getattr(self, ds_name)
                source_var = source_ds[var_key]
                
                if grid:
                    output[var_name] = source_var.interp(R=R_arr, z=z_arr, method='cubic')
                else:
                    intrp = RectBivariateSpline(self.Rgrid.values, self.zgrid.values,
                                                source_var.values)
                    tmp = intrp(R_arr, z_arr, grid=False)
                    dims = ('point',) if tmp.ndim == 1 else None
                    output[var_name] = xr.DataArray(tmp, dims=dims)
                    output[var_name].attrs.update(source_var.attrs)
        
        return output

    def at_rho(
        self,
        rho: Union[float, np.ndarray]
    ) -> xr.Dataset:
        """
        Get flux-surface quantities at given rho value(s).

        Parameters
        ----------
        rho : float or np.ndarray
            Normalized poloidal flux coordinate value(s)

        Returns
        -------
        xr.Dataset
            Dataset containing flux-surface averaged or profile values
            (q, pres, fpol if available, plus geometric properties)

        Examples
        --------
        >>> fs_data = eq.at_rho(0.5)  # Get quantities at rho=0.5
        """
        rho_arr = np.atleast_1d(rho)
        output = xr.Dataset()
        
        # Interpolate profiles if available
        if len(self._profiles) > 0:
            # Profiles are typically functions of psi_n or rho
            # For now, interpolate from profiles if they exist
            for var_name in self._profiles.data_vars:
                profile_var = self._profiles[var_name]
                if profile_var.ndim == 1:
                    # Interpolate profile to requested rho values
                    # Assuming profile is on rho coordinate
                    if 'rhop' in profile_var.coords or len(profile_var.coords) == 1:
                        coord_name = list(profile_var.coords.keys())[0]
                        output[var_name] = profile_var.interp({coord_name: rho_arr},
                                                               method='linear',
                                                               kwargs={'fill_value': 'extrapolate'})
        
        # Add geometric properties
        output['rho'] = xr.DataArray(rho_arr, dims=('rho',),
                                    attrs={'name': 'rho', 'units': '',
                                           'desc': 'Normalized poloidal flux',
                                           'short_name': r'$\rho$'})
        
        return output

    def compute_curvature_vector(
        self,
        cache: bool = True,
    ) -> xr.Dataset:
        """
        Compute magnetic-field-line curvature in cylindrical coordinates.

        Computes ``kappa = (b dot nabla)b`` where ``b = B / |B|`` and returns
        the cylindrical components ``(kappa_R, kappa_phi, kappa_z)`` on the
        native ``(R, z)`` grid.

        Notes
        -----
        - This routine assumes axisymmetric equilibrium data
          (i.e. ``d/dphi = 0``).
        - 4th-order finite-difference stencils are used in ``R`` and ``z``;
          therefore, at least 5 points are required in each direction.

        Parameters
        ----------
        use_numba : bool, optional
            If True, use a Numba kernel for the hot loops. If False, use a
            FinDiff-based NumPy implementation. Default is True.
        cache : bool, optional
            If True, store the result internally in ``self.Kdata``,
            ``self.curvaturedata`` (legacy alias), and ``self._curvature``.
            Default is True.

        Returns
        -------
        xr.Dataset
            Dataset with ``kappa_R``, ``kappa_phi``, ``kappa_z`` and
            ``kappa_abs``.
        """
        if self.nr < 5 or self.nz < 5:
            raise ValueError("Curvature computation requires at least 5 grid points in both R and z.")

        R_vals = np.asarray(self.Rgrid.values, dtype=np.float64)
        z_vals = np.asarray(self.zgrid.values, dtype=np.float64)

        dR = float(R_vals[1] - R_vals[0])
        dZ = float(z_vals[1] - z_vals[0])

        if not np.allclose(np.diff(R_vals), dR, rtol=1.0e-8, atol=1.0e-12):
            raise ValueError("R grid must be uniformly spaced for 4th-order finite differences.")
        if not np.allclose(np.diff(z_vals), dZ, rtol=1.0e-8, atol=1.0e-12):
            raise ValueError("z grid must be uniformly spaced for 4th-order finite differences.")

        Babs = np.asarray(self.Bdata.Babs.values, dtype=np.float64)
        Babs_safe = np.where(np.abs(Babs) < _CURVATURE_EPS, _CURVATURE_EPS, Babs)

        b_R = np.asarray(self.Bdata.Br.values, dtype=np.float64) / Babs_safe
        b_phi = np.asarray(self.Bdata.Bphi.values, dtype=np.float64) / Babs_safe
        b_Z = np.asarray(self.Bdata.Bz.values, dtype=np.float64) / Babs_safe

        kappa_R, kappa_phi, kappa_z = _curvature_axisymmetric_findiff(
            R_vals, dR, dZ, b_R, b_phi, b_Z
        )

        curvature = xr.Dataset()
        curvature["kappa_R"] = xr.DataArray(
            kappa_R,
            coords=(self.Rgrid, self.zgrid),
            attrs={
                "name": "kappa_R",
                "units": "1/m",
                "desc": "Radial component of field-line curvature vector",
                "short_name": r"$\kappa_R$",
            },
        )
        curvature["kappa_phi"] = xr.DataArray(
            kappa_phi,
            coords=(self.Rgrid, self.zgrid),
            attrs={
                "name": "kappa_phi",
                "units": "1/m",
                "desc": "Toroidal component of field-line curvature vector",
                "short_name": r"$\kappa_\phi$",
            },
        )
        curvature["kappa_z"] = xr.DataArray(
            kappa_z,
            coords=(self.Rgrid, self.zgrid),
            attrs={
                "name": "kappa_z",
                "units": "1/m",
                "desc": "Vertical component of field-line curvature vector",
                "short_name": r"$\kappa_z$",
            },
        )
        curvature["kappa_abs"] = np.sqrt(
            curvature.kappa_R**2 + curvature.kappa_phi**2 + curvature.kappa_z**2
        )
        curvature.kappa_abs.attrs.update({
            "name": "kappa_abs",
            "units": "1/m",
            "desc": "Magnitude of field-line curvature vector",
            "short_name": r"$|\kappa|$",
        })

        if cache:
            self.Kdata = curvature
            # Backward-compatible alias used by early integrations.
            self.curvaturedata = self.Kdata
            self._curvature = self.Kdata

            # Register/update plottable names without failing on recompute.
            for var_name, var in self.Kdata.data_vars.items():
                if var.ndim != 2:
                    continue
                if var_name in self.plot_2d_names:
                    self.plot_2d_names[var_name] = var
                elif var_name in self.plot_1d_names:
                    self.plot_1d_names[var_name] = var
                else:
                    self.add_var(var_name, var)

        return curvature

    def make_curvature(self) -> xr.Dataset:
        """
        Compute curvature vectors and store them internally.

        This convenience wrapper always updates the internal storage
        (``Kdata``/``curvature``) and returns the computed dataset.
        """
        return self.compute_curvature_vector(cache=True)

    def summary(self) -> None:
        """
        Print a summary of equilibrium properties.

        Examples
        --------
        >>> eq.summary()
        """
        print("=" * 60)
        print("Equilibrium Summary")
        print("=" * 60)
        print(f"\nGrid:")
        print(f"  R: [{self.Rgrid.min().values:.3f}, {self.Rgrid.max().values:.3f}] m ({self.nr} points)")
        print(f"  z: [{self.zgrid.min().values:.3f}, {self.zgrid.max().values:.3f}] m ({self.nz} points)")
        
        print(f"\nMagnetic Axis:")
        print(f"  R_axis = {self.R_axis:.3f} m")
        print(f"  z_axis = {self.z_axis:.3f} m")
        print(f"  B_axis = {float(self.Bdata.Baxis.values):.3f} T")
        
        print(f"\nFlux Surfaces:")
        print(f"  psi_ax = {self.geometry.attrs.get('psi_ax', 'N/A'):.3e} {self.flux_normalization}")
        print(f"  psi_bdy = {self.geometry.attrs.get('psi_bdy', 'N/A'):.3e} {self.flux_normalization}")
        print(f"  psimax = {self.geometry.attrs.get('psimax', 'N/A'):.3e} {self.flux_normalization}")
        
        if len(self._profiles) > 0:
            print(f"\nProfiles available:")
            for var_name in self._profiles.data_vars:
                print(f"  - {var_name}")
        else:
            print(f"\nProfiles: None loaded")
        
        print(f"\nMagnetic Field Components:")
        print(f"  Br:   [{self.Br.min().values:.3f}, {self.Br.max().values:.3f}] T")
        print(f"  Bz:   [{self.Bz.min().values:.3f}, {self.Bz.max().values:.3f}] T")
        print(f"  Bphi: [{self.Bphi.min().values:.3f}, {self.Bphi.max().values:.3f}] T")
        print(f"  B:    [{self.B.min().values:.3f}, {self.B.max().values:.3f}] T")
        
        print("=" * 60)

    def plot(self, name: Optional[str] = None, ax=None, put_labels: bool=True,
             line=None, **kwargs):
        """
        Plot equilibrium variables.

        If no name is provided, lists available plottable variables.
        Otherwise, plots the specified variable (2D or 1D).

        Parameters
        ----------
        name : str, optional
            Variable name to plot. If None, lists available variables.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        put_labels : bool, optional
            If True, add axis labels and title. Default is True
        line : matplotlib Line2D or Image, optional
            Existing plot object to update. Default is None
        **kwargs
            Additional arguments passed to matplotlib plotting functions

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object
        plot_obj : matplotlib Line2D or Image
            Plot object (line or image)

        Examples
        --------
        >>> eq.plot()  # List available variables
        >>> ax, im = eq.plot('psi')  # Plot 2D variable
        >>> ax, line = eq.plot('q')  # Plot 1D profile
        """
        if name is None:
            # List available plottable variables
            plottable_2d = []
            plottable_1d = []
            
            # From structured datasets
            for var_name in self.field.data_vars:
                if self.field[var_name].ndim == 2:
                    plottable_2d.append(f"field.{var_name}")
            for var_name in self.flux.data_vars:
                if self.flux[var_name].ndim == 2:
                    plottable_2d.append(f"flux.{var_name}")
            for var_name in self.profiles.data_vars:
                if self.profiles[var_name].ndim == 1:
                    plottable_1d.append(f"profiles.{var_name}")
            for var_name in self.curvature.data_vars:
                if self.curvature[var_name].ndim == 2:
                    plottable_2d.append(f"curvature.{var_name}")
            
            # From legacy plot dicts (backward compatibility)
            plottable_2d.extend(self.plot_2d_names.keys())
            plottable_1d.extend(self.plot_1d_names.keys())
            
            print("Plottable 2D variables:")
            for v in sorted(set(plottable_2d)):
                print(f"  - {v}")
            print("\nPlottable 1D variables:")
            for v in sorted(set(plottable_1d)):
                print(f"  - {v}")
            return None, None
        
        # Resolve variable name from structured datasets
        resolved_var = self._resolve_plot_variable(name)
        
        if resolved_var is None:
            # Fall back to legacy plot dicts
            if name in self.plot_2d_names:
                return self.plot2d(name=name, ax=ax, put_labels=put_labels,
                                   image=line, **kwargs)
            elif name in self.plot_1d_names:
                return self.plot1d(name=name, ax=ax, put_labels=put_labels,
                                   line=line, **kwargs)
            else:
                raise ValueError(f'Cannot plot {name}: variable not found. '
                               f'Use eq.plot() to list available variables.')
        
        # Plot resolved variable
        var, is_2d = resolved_var
        if is_2d:
            return self.plot2d_var(var, name=name, ax=ax, put_labels=put_labels,
                                  image=line, **kwargs)
        else:
            return self.plot1d_var(var, name=name, ax=ax, put_labels=put_labels,
                                 line=line, **kwargs)
    
    def _resolve_plot_variable(self, name: str) -> Optional[Tuple[xr.DataArray, bool]]:
        """
        Resolve variable name to xr.DataArray from structured datasets.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'psi', 'Br', 'q', 'field.Br', 'profiles.q')

        Returns
        -------
        tuple or None
            (DataArray, is_2d) if found, None otherwise
        """
        # Try direct name first
        if name in self.field.data_vars:
            var = self.field[name]
            return (var, var.ndim == 2)
        if name in self.flux.data_vars:
            var = self.flux[name]
            return (var, var.ndim == 2)
        if name in self.profiles.data_vars:
            var = self.profiles[name]
            return (var, var.ndim == 2)
        if name in self.curvature.data_vars:
            var = self.curvature[name]
            return (var, var.ndim == 2)
        
        # Try prefixed names (field.Br, flux.psi, profiles.q)
        if '.' in name:
            prefix, var_name = name.split('.', 1)
            if prefix == 'field' and var_name in self.field.data_vars:
                var = self.field[var_name]
                return (var, var.ndim == 2)
            if prefix == 'flux' and var_name in self.flux.data_vars:
                var = self.flux[var_name]
                return (var, var.ndim == 2)
            if prefix == 'profiles' and var_name in self.profiles.data_vars:
                var = self.profiles[var_name]
                return (var, var.ndim == 2)
            if prefix == 'curvature' and var_name in self.curvature.data_vars:
                var = self.curvature[var_name]
                return (var, var.ndim == 2)
        
        return None
    
    def plot1d_var(self, var: xr.DataArray, name: str, ax=None,
                   put_labels: bool=True, line=None, **kwargs):
        """
        Plot a 1D variable using xarray's plot functionality.

        Parameters
        ----------
        var : xr.DataArray
            Variable to plot (1D)
        name : str
            Variable name (for labeling)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        put_labels : bool, optional
            If True, add labels
        line : matplotlib Line2D, optional
            Existing line to update
        **kwargs
            Arguments passed to xarray's plot

        Returns
        -------
        ax : matplotlib.axes.Axes
        line : matplotlib Line2D
        """
        plt = _require_matplotlib_pyplot()
        x = var[list(var.coords.keys())[0]]
        
        if line is not None:
            line.set_xdata(x.values)
            line.set_ydata(var.values)
            ax_was_none = False
        else:
            ax_was_none = ax is None
            if ax_was_none:
                fig, ax = plt.subplots(1)
            
            # Use xarray's plot if available, otherwise fall back
            try:
                line = var.plot(ax=ax, **kwargs)
                if isinstance(line, list):
                    line = line[0]
            except:
                line, = ax.plot(x.values, var.values, **kwargs)
        
        if ax_was_none and put_labels:
            fig = line.axes.figure
            x_label = x.attrs.get('short_name', x.name)
            x_units = x.attrs.get('units', '')
            y_label = var.attrs.get('short_name', var.name)
            y_units = var.attrs.get('units', '')
            ax.set_xlabel(f'{x_label} [{x_units}]' if x_units else x_label)
            ax.set_ylabel(f'{y_label} [{y_units}]' if y_units else y_label)
            ax.grid('both')
            fig.tight_layout()
        
        return ax, line
    
    def plot2d_var(self, var: xr.DataArray, name: str, ax=None,
                   put_labels: bool=True, image=None, **kwargs):
        """
        Plot a 2D variable using xarray's plot functionality.

        Parameters
        ----------
        var : xr.DataArray
            Variable to plot (2D)
        name : str
            Variable name (for labeling)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        put_labels : bool, optional
            If True, add labels and overlay boundary/axis
        image : matplotlib Image, optional
            Existing image to update
        **kwargs
            Arguments passed to xarray's plot

        Returns
        -------
        ax : matplotlib.axes.Axes
        image : matplotlib Image
        """
        plt = _require_matplotlib_pyplot()
        x = var[list(var.coords.keys())[0]]
        y = var[list(var.coords.keys())[1]]
        plot_data = np.asarray(var.values).T
        extent = [x.min().values, x.max().values, y.min().values, y.max().values]
        
        if image is not None:
            # Update existing image
            image.set_data(plot_data)
            image.set_extent(extent)
            ax_was_none = False
        else:
            ax_was_none = ax is None
            if ax_was_none:
                fig, ax = plt.subplots(1)

            image = ax.imshow(plot_data, origin='lower', extent=extent, **kwargs)
            
            # Overlay boundary and axis
            if put_labels:
                ax.plot(self.geometry.R_boundary.values,
                       self.geometry.z_boundary.values, 'w-', linewidth=1.5)
                ax.plot(float(self.geometry.R_axis.values),
                       float(self.geometry.z_axis.values), 'wx', markersize=10, markeredgewidth=2)
        
        if ax_was_none and put_labels:
            fig = image.axes.figure
            x_label = x.attrs.get('short_name', x.name)
            x_units = x.attrs.get('units', '')
            y_label = y.attrs.get('short_name', y.name)
            y_units = y.attrs.get('units', '')
            ax.set_xlabel(f'{x_label} [{x_units}]' if x_units else x_label)
            ax.set_ylabel(f'{y_label} [{y_units}]' if y_units else y_label)
            
            # Add colorbar
            if hasattr(image, 'colorbar') and image.colorbar is None:
                cbar = fig.colorbar(mappable=image, ax=ax)
                z_label = var.attrs.get('short_name', var.name)
                z_units = var.attrs.get('units', '')
                cbar.set_label(f'{z_label} [{z_units}]' if z_units else z_label)
            elif not hasattr(image, 'colorbar'):
                cbar = fig.colorbar(mappable=image, ax=ax)
                z_label = var.attrs.get('short_name', var.name)
                z_units = var.attrs.get('units', '')
                cbar.set_label(f'{z_label} [{z_units}]' if z_units else z_label)
            
            ax.set_aspect('equal')
            fig.tight_layout()
        
        return ax, image

    def plot1d(self, name: str, ax=None, put_labels: bool=True,
               line=None, **kwargs):
        """
        Plot a 1D variable (explicit 1D entry point).

        This is a convenience wrapper around plot() for 1D variables.
        It maintains backward compatibility with the old API.

        Parameters
        ----------
        name : str
            Variable name to plot
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        put_labels : bool, optional
            If True, add labels. Default is True
        line : matplotlib Line2D, optional
            Existing line to update
        **kwargs
            Arguments passed to matplotlib plot

        Returns
        -------
        ax : matplotlib.axes.Axes
        line : matplotlib Line2D
        """
        # Try to resolve from structured datasets first
        resolved = self._resolve_plot_variable(name)
        if resolved is not None:
            var, is_2d = resolved
            if not is_2d:
                return self.plot1d_var(var, name=name, ax=ax,
                                      put_labels=put_labels, line=line, **kwargs)
        
        # Fall back to legacy plot dicts
        if name not in self.plot_1d_names:
            raise ValueError(f'Cannot plot {name}. Variable not found. '
                           f'Use eq.plot() to list available variables.')

        var = self.plot_1d_names[name]
        return self.plot1d_var(var, name=name, ax=ax,
                              put_labels=put_labels, line=line, **kwargs)

    def plot2d(self, name: str, ax=None, put_labels: bool=True,
               image=None, **kwargs):
        """
        Plot a 2D variable (explicit 2D entry point).

        This is a convenience wrapper around plot() for 2D variables.
        It maintains backward compatibility with the old API.

        Parameters
        ----------
        name : str
            Variable name to plot
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        put_labels : bool, optional
            If True, add labels and overlay boundary/axis. Default is True
        image : matplotlib Image, optional
            Existing image to update
        **kwargs
            Arguments passed to matplotlib imshow/contour

        Returns
        -------
        ax : matplotlib.axes.Axes
        image : matplotlib Image
        """
        # Try to resolve from structured datasets first
        resolved = self._resolve_plot_variable(name)
        if resolved is not None:
            var, is_2d = resolved
            if is_2d:
                return self.plot2d_var(var, name=name, ax=ax,
                                      put_labels=put_labels, image=image, **kwargs)
        
        # Fall back to legacy plot dicts
        if name not in self.plot_2d_names:
            raise ValueError(f'Cannot plot {name}. Variable not found. '
                           f'Use eq.plot() to list available variables.')

        var = self.plot_2d_names[name]
        return self.plot2d_var(var, name=name, ax=ax,
                              put_labels=put_labels, image=image, **kwargs)

    def compute_coordinates(self, coordinate_system: str='boozer',
                           lpsi: int=201, ltheta: int=256,
                           dr_hr: float=1.0e-3, dz_hz: float=1.0e-3,
                           padding: float=0.05, ntht_pad: int=5,
                           rhopol_min: Optional[float]=None,
                           rhopol_max: Optional[float]=None,
                           spectral_max_mode: int=16,
                           radial_guard_surfaces: int=3,
                           enforce_up_down_symmetry: bool=False,
                           symmetry_tolerance: Optional[float]=None):
        """
        Compute magnetic coordinates for the specified coordinate system.

        This is a generic method that works with any coordinate system
        registered in the Jacobian registry. The main difference between
        coordinate systems is the choice of Jacobian.

        Parameters
        ----------
        coordinate_system : str, optional
            Name of the coordinate system ('boozer', 'hamada', 'pest', etc.)
            Default is 'boozer'
        lpsi : int, optional
            Number of points along the radial direction. Default is 201
        ltheta : int, optional
            Number of points along the poloidal direction. Default is 256
        dr_hr : float, optional
            Requested radial step for the tracing grid. Refinement is capped
            at four samples per native equilibrium cell. Default is 1.0e-3.
        dz_hz : float, optional
            Requested vertical step for the tracing grid. Refinement is capped
            at four samples per native equilibrium cell. Default is 1.0e-3.
        padding : float, optional
            Padding for the coordinate grid. Default is 0.05
        ntht_pad : int, optional
            Number of padding points for theta. Default is 5
        rhopol_min : float, optional
            Minimum normalized poloidal radius to include, in [0, 1].
            If provided (with or without ``rhopol_max``), this overrides
            symmetric ``padding`` behavior.
        rhopol_max : float, optional
            Maximum normalized poloidal radius to include, in [0, 1].
            If provided (with or without ``rhopol_min``), this overrides
            symmetric ``padding`` behavior.
        spectral_max_mode : int, optional
            Maximum retained poloidal Fourier mode in the spectral surface
            reconstruction. Default is 16.
        radial_guard_surfaces : int, optional
            Number of hidden same-spacing radial support surfaces added on
            each physically available side of the requested grid. The returned
            ``psi0`` grid still contains exactly ``lpsi`` points. Default is 3.
        enforce_up_down_symmetry : bool, optional
            Explicitly project the common ``R(psi,theta)``, ``z(psi,theta)``,
            and ``nu(psi,theta)`` map onto up-down parity before derivatives
            and metrics are constructed. General equilibria remain unchanged
            by default.
        symmetry_tolerance : float, optional
            Maximum relative R/Z asymmetry accepted when explicit projection
            is requested. A finite positive value is required whenever
            ``enforce_up_down_symmetry=True``.

        Returns
        -------
        MagneticCoordinates
            Magnetic coordinates object containing the transformation

        Notes
        -----
        Available coordinate systems are driven by the Jacobian registry
        (e.g. ``boozer``, ``pest``, ``equal_arc``, ``hamada``). Any registered
        system can be computed by passing its name.
        """
        # Get the Jacobian function for this coordinate system
        jacobian_func = get_jacobian_function(coordinate_system)
        
        # Build fine grid for flux surface contours
        rmin = float(self.Rgrid.values[0])
        rmax = float(self.Rgrid.values[-1])
        zmin = float(self.zgrid.values[0])
        zmax = float(self.zgrid.values[-1])

        if dr_hr <= 0.0 or dz_hz <= 0.0:
            raise ValueError("dr_hr and dz_hz must be positive.")
        requested_nr = int(np.ceil((rmax - rmin) / dr_hr)) + 1
        requested_nz = int(np.ceil((zmax - zmin) / dz_hz)) + 1
        # Cubic refinement does not create independent equilibrium
        # information.  Cap the tracing grid at four samples per native cell
        # to avoid workstation-scale O(N_R*N_Z) memory growth when callers
        # retain historical 1 mm settings on a multi-metre domain.
        nr_fine = max(
            int(self.nr),
            min(requested_nr, 4 * (int(self.nr) - 1) + 1),
        )
        nz_fine = max(
            int(self.nz),
            min(requested_nz, 4 * (int(self.nz) - 1) + 1),
        )

        R_fine = np.linspace(rmin, rmax, nr_fine)
        z_fine = np.linspace(zmin, zmax, nz_fine)

        # Evaluate on fine grid (using new structured access)
        psip = self.flux.psi.interp(R=R_fine, z=z_fine, method='cubic').values
        br_fine = self.field.Br.interp(R=R_fine, z=z_fine, method='cubic').values
        bz_fine = self.field.Bz.interp(R=R_fine, z=z_fine, method='cubic').values
        bphi_fine = self.field.Bphi.interp(R=R_fine, z=z_fine, method='cubic').values

        # Generate psi grid (using geometry attributes)
        psi_axis = float(self.geometry.attrs.get('psi_ax', self._psi_ax_init))
        psi_edge = float(self.geometry.attrs.get('psi_bdy', self._psi_edge_init))
        R_axis_val = float(self.geometry.R_axis.values)
        z_axis_val = float(self.geometry.z_axis.values)
        psi_at_axis = np.asarray(
            RectBivariateSpline(
                R_fine,
                z_fine,
                psip,
                kx=min(3, R_fine.size - 1),
                ky=min(3, z_fine.size - 1),
                s=0.0,
            ).ev(R_axis_val, z_axis_val),
            dtype=np.float64,
        ).item()
        normalized_axis_floor = _normalized_resolvable_axis_flux(
            psi_at_axis,
            psi_axis=psi_axis,
            psi_boundary=psi_edge,
        )
        if rhopol_min is not None or rhopol_max is not None:
            rho_min = padding if rhopol_min is None else float(rhopol_min)
            rho_max = 1.0 - padding if rhopol_max is None else float(rhopol_max)
            if not (0.0 <= rho_min < rho_max <= 1.0):
                raise ValueError("rhopol_min/rhopol_max must satisfy 0 <= min < max <= 1.")
            eps = 1.0e-8
            rho_min = max(rho_min, eps)
            rho_max = min(rho_max, 1.0 - eps)
            if rho_max <= rho_min:
                raise ValueError("rhopol_min/rhopol_max window is too narrow after edge protection.")
            # Distribute the fitted surfaces uniformly in rho while retaining
            # physical poloidal flux as the interpolation coordinate.  A
            # uniform-psi mesh severely under-resolves R(psi) ~ sqrt(psi)
            # near the magnetic axis and corrupts radial map derivatives.
            rho_grid = np.linspace(rho_min, rho_max, lpsi)
            rho_axis_floor = float(np.sqrt(normalized_axis_floor))
            if rho_axis_floor >= rho_grid[0]:
                raise ValueError(
                    "The requested inner rhopol surface lies inside the flux "
                    "range resolved by the interpolated equilibrium field: "
                    f"requested={rho_grid[0]:.6g}, "
                    f"resolvable>{rho_axis_floor:.6g}."
                )
            support_rho, core_indices = _extend_radial_support(
                rho_grid,
                lower_bound=rho_axis_floor,
                upper_bound=1.0,
                guard_surfaces=radial_guard_surfaces,
            )
            core_psigrid_requested = (
                psi_axis + rho_grid**2 * (psi_edge - psi_axis)
            )
            psigrid = psi_axis + support_rho**2 * (psi_edge - psi_axis)
            support_family = "rhopol"
        else:
            psi_span = psi_edge - psi_axis
            normalized_core = np.linspace(padding, 1.0 - padding, lpsi)
            normalized_support, core_indices = _extend_radial_support(
                normalized_core,
                lower_bound=normalized_axis_floor,
                upper_bound=1.0,
                guard_surfaces=radial_guard_surfaces,
            )
            core_psigrid_requested = (
                psi_axis + normalized_core * psi_span
            )
            psigrid = psi_axis + normalized_support * psi_span
            support_rho = np.sqrt(np.clip(normalized_support, 0.0, 1.0))
            support_family = "psi_n"

        # Transform psigrid to radial positions at midplane (using geometry)
        R_bdy_max = float(self.geometry.R_boundary.max().values)
        Rgrid_mid = np.linspace(R_axis_val, R_bdy_max, 1000)
        psi_on_Rgrid = self.flux.psi.interp(R=Rgrid_mid,
                                            z=z_axis_val,
                                            method='cubic')
        frr0 = _outboard_midplane_seeds(
            Rgrid_mid,
            np.asarray(psi_on_Rgrid, dtype=np.float64),
            support_rho,
            psi_axis=psi_axis,
            psi_boundary=psi_edge,
        )

        psi_span = psi_edge - psi_axis
        if abs(psi_span) < 1.0e-14:
            rho_at_psi = np.zeros_like(psigrid, dtype=float)
        else:
            rho_norm = (psigrid - psi_axis) / psi_span
            rho_at_psi = np.sqrt(np.clip(rho_norm, 0.0, None))

        # Compute coordinates using generic function
        # Pass frr0 (radial positions at midplane) corresponding to psigrid
        coordinate_construction_diagnostics: Dict[str, Any] = {}
        out = compute_magnetic_coordinates(
            Rgrid=R_fine, zgrid=z_fine,
            br=br_fine, bz=bz_fine, bphi=bphi_fine,
            raxis=R_axis_val,
            zaxis=z_axis_val,
            psigrid=psigrid,
            ltheta=ltheta,
            phiclockwise=self.phiclockwise,
            jacobian_func=jacobian_func,
            R_at_psi=frr0,
            rho_at_psi=rho_at_psi,
            coordinate_system=coordinate_system,
            spectral_max_mode=spectral_max_mode,
            psi_field=psip,
            flux_scale=abs(psi_edge - psi_axis),
            enforce_up_down_symmetry=enforce_up_down_symmetry,
            symmetry_tolerance=symmetry_tolerance,
            diagnostics=coordinate_construction_diagnostics,
        )

        qprof, Fprof, Iprof, thtable, nutable, jac, Rtransform, ztransform = out
        coordinate_psi_field = coordinate_construction_diagnostics.pop(
            "coordinate_psi_field",
            None,
        )
        coordinate_Rgrid = np.asarray(self.Rgrid, dtype=np.float64)
        coordinate_zgrid = np.asarray(self.zgrid, dtype=np.float64)
        if coordinate_psi_field is not None:
            # Coordinate construction operates on the refined tracing grid.
            # General equilibria retain the original public R-z grid.  An
            # explicitly projected equilibrium instead retains the reflected
            # vertical knot union needed to represent its authoritative paired
            # flux evaluator without relabelling the fitted surfaces.
            coordinate_psi_values = np.asarray(
                coordinate_psi_field,
                dtype=np.float64,
            )
            expected_tracing_shape = (R_fine.size, z_fine.size)
            if coordinate_psi_values.shape != expected_tracing_shape:
                raise ValueError(
                    "projected coordinate psi field has shape "
                    f"{coordinate_psi_values.shape}; expected tracing-grid "
                    f"shape {expected_tracing_shape}."
                )
            projection_audit = coordinate_construction_diagnostics.get(
                "up_down_symmetry"
            )
            if (
                isinstance(projection_audit, dict)
                and bool(projection_audit.get("applied", False))
            ):
                (
                    coordinate_Rgrid,
                    coordinate_zgrid,
                    coordinate_psi_field,
                ) = _reflection_paired_coordinate_psi_grid(
                    tracing_R=R_fine,
                    tracing_z=z_fine,
                    tracing_psi=coordinate_psi_values,
                    public_R=coordinate_Rgrid,
                    reflection_z=z_axis_val,
                )
                coordinate_psi_field, bridge_audit = (
                    _fit_projected_coordinate_psi_bridge(
                        Rgrid=coordinate_Rgrid,
                        zgrid=coordinate_zgrid,
                        psi_field=coordinate_psi_field,
                        surface_R=Rtransform,
                        surface_z=ztransform,
                        surface_psi=psigrid,
                        flux_scale=abs(psi_edge - psi_axis),
                        reflection_z=z_axis_val,
                    )
                )
                projection_audit[
                    "projected_bridge_flux_initial_residual"
                ] = bridge_audit["initial_residual"]
                projection_audit["projected_bridge_flux_residual"] = (
                    bridge_audit["final_residual"]
                )
                projection_audit[
                    "projected_bridge_relative_grid_correction"
                ] = bridge_audit["relative_grid_correction"]
                projection_audit[
                    "projected_bridge_symmetry_initial_residual"
                ] = bridge_audit["initial_symmetry_residual"]
                projection_audit[
                    "projected_bridge_symmetry_residual"
                ] = bridge_audit["final_symmetry_residual"]
                projection_audit[
                    "projected_bridge_solver_stop_code"
                ] = bridge_audit["solver_stop_code"]
                projection_audit[
                    "projected_bridge_solver_iterations"
                ] = bridge_audit["solver_iterations"]
                projection_audit[
                    "projected_bridge_radial_grid_size"
                ] = int(coordinate_Rgrid.size)
                projection_audit[
                    "projected_bridge_vertical_grid_size"
                ] = int(coordinate_zgrid.size)
            else:
                coordinate_psi_field = RectBivariateSpline(
                    R_fine,
                    z_fine,
                    coordinate_psi_values,
                    kx=min(3, R_fine.size - 1),
                    ky=min(3, z_fine.size - 1),
                    s=0.0,
                )(coordinate_Rgrid, coordinate_zgrid)

        # Surface tracing and near-axis regularization use axis-to-boundary
        # order. Sort every radial output together only at the storage
        # boundary because scipy's spline constructors require increasing
        # physical psi. The physical axis-to-boundary direction is retained
        # separately in psi0 metadata.
        psi_order = np.argsort(psigrid)
        psigrid = np.asarray(psigrid)[psi_order]
        qprof = np.asarray(qprof)[psi_order]
        Fprof = np.asarray(Fprof)[psi_order]
        Iprof = np.asarray(Iprof)[psi_order]
        thtable = np.asarray(thtable)[psi_order, :]
        nutable = np.asarray(nutable)[psi_order, :]
        jac = np.asarray(jac)[psi_order, :]
        Rtransform = np.asarray(Rtransform)[psi_order, :]
        ztransform = np.asarray(ztransform)[psi_order, :]
        symmetry_projection_audit = coordinate_construction_diagnostics.get(
            "up_down_symmetry"
        )
        if isinstance(symmetry_projection_audit, dict):
            for name, values in tuple(symmetry_projection_audit.items()):
                if (
                    isinstance(values, np.ndarray)
                    and values.shape == (psi_order.size,)
                ):
                    symmetry_projection_audit[name] = values[psi_order]
                elif isinstance(values, dict):
                    for field_name, field_values in tuple(values.items()):
                        field_array = np.asarray(field_values)
                        if field_array.shape == (psi_order.size,):
                            values[field_name] = field_array[psi_order]
        if np.any(np.diff(psigrid) <= 0.0):
            raise ValueError(
                "Computed magnetic-coordinate psi grid must be strictly increasing."
            )
        sorted_core = np.sort(
            np.asarray(core_psigrid_requested, dtype=np.float64)
        )
        core_indices = np.searchsorted(psigrid, sorted_core)
        if (
            core_indices.shape != (lpsi,)
            or np.any(core_indices >= psigrid.size)
            or not np.allclose(
                psigrid[core_indices],
                sorted_core,
                rtol=0.0,
                atol=10.0 * np.finfo(np.float64).eps
                * max(1.0, float(np.max(np.abs(psigrid)))),
            )
        ):
            raise ValueError(
                "Hidden radial support construction lost a requested core surface."
            )
        if abs(psi_span) < 1.0e-14:
            final_support_rho = np.zeros_like(psigrid, dtype=np.float64)
            final_core_rho = np.zeros_like(sorted_core, dtype=np.float64)
        else:
            final_support_rho = np.sqrt(
                np.clip((psigrid - psi_axis) / psi_span, 0.0, None)
            )
            final_core_rho = np.sqrt(
                np.clip((sorted_core - psi_axis) / psi_span, 0.0, None)
            )
        radial_scale = max(
            1.0,
            float(np.max(np.abs(final_support_rho))),
        )
        radial_tolerance = 10.0 * np.finfo(np.float64).eps * radial_scale
        inner_guard_surfaces = int(
            np.count_nonzero(
                final_support_rho
                < float(np.min(final_core_rho)) - radial_tolerance
            )
        )
        outer_guard_surfaces = int(
            np.count_nonzero(
                final_support_rho
                > float(np.max(final_core_rho)) + radial_tolerance
            )
        )

        # Continue with post-processing
        return self._build_magnetic_coordinates_dataset(
            psigrid, thtable, nutable, jac, Rtransform, ztransform,
            coordinate_Rgrid,
            coordinate_zgrid,
            qprof, Fprof, Iprof, ntht_pad, coordinate_system,
            spectral_max_mode=spectral_max_mode,
            core_indices=core_indices,
            radial_support_metadata={
                "family": support_family,
                "requested_guard_surfaces": int(radial_guard_surfaces),
                "support_nsurface": int(psigrid.size),
                "core_nsurface": int(lpsi),
                "inner_guard_surfaces": inner_guard_surfaces,
                "outer_guard_surfaces": outer_guard_surfaces,
                "support_rhopol_min": float(np.min(final_support_rho)),
                "support_rhopol_max": float(np.max(final_support_rho)),
            },
            symmetry_projection_audit=coordinate_construction_diagnostics.get(
                "up_down_symmetry"
            ),
            coordinate_psi_field=coordinate_psi_field,
        )
    
    def _build_magnetic_coordinates_dataset(
        self,
        psigrid: np.ndarray,
        thtable: np.ndarray,
        nutable: np.ndarray,
        jac: np.ndarray,
        Rtransform: np.ndarray,
        ztransform: np.ndarray,
        R_fine: np.ndarray,
        z_fine: np.ndarray,
        qprof: np.ndarray,
        Fprof: np.ndarray,
        Iprof: np.ndarray,
        ntht_pad: int,
        coordinate_system: str = 'boozer',
        spectral_max_mode: Optional[int] = None,
        core_indices: Optional[np.ndarray] = None,
        radial_support_metadata: Optional[Dict[str, Any]] = None,
        symmetry_projection_audit: Optional[Dict[str, Any]] = None,
        coordinate_psi_field: Optional[np.ndarray] = None,
    ) -> MagneticCoordinates:
        """
        Build the MagneticCoordinates dataset from computed coordinate arrays.

        Parameters
        ----------
        psigrid : np.ndarray
            Poloidal flux grid
        thtable : np.ndarray
            Magnetic poloidal angle table
        nutable : np.ndarray
            Axisymmetric toroidal gauge-shift table ``nu(psi, theta)`` in
            ``zeta = phi + nu``
        jac : np.ndarray
            Jacobian table
        Rtransform : np.ndarray
            Inverse transformation R(psi, theta)
        ztransform : np.ndarray
            Inverse transformation z(psi, theta)
        R_fine : np.ndarray
            Fine radial grid
        z_fine : np.ndarray
            Fine vertical grid
        qprof : np.ndarray
            Safety factor profile
        Fprof : np.ndarray
            F(psi) profile
        Iprof : np.ndarray
            Covariant Boozer magnetic-field coefficient ``I = B_Theta``
        ntht_pad : int
            Number of padding points for theta
        spectral_max_mode : int, optional
            Maximum Fourier mode retained by the common coordinate map.

        Returns
        -------
        MagneticCoordinates
            Magnetic coordinates object
        """
        support_psi = np.asarray(psigrid, dtype=np.float64)
        support_theta_table = np.asarray(thtable, dtype=np.float64)
        support_nu_table = np.asarray(nutable, dtype=np.float64)
        support_jacobian_table = np.asarray(jac, dtype=np.float64)
        support_R = np.asarray(Rtransform, dtype=np.float64)
        support_z = np.asarray(ztransform, dtype=np.float64)
        support_q = np.asarray(qprof, dtype=np.float64)
        support_F = np.asarray(Fprof, dtype=np.float64)
        support_I = np.asarray(Iprof, dtype=np.float64)
        ltheta = support_theta_table.shape[1]
        support_nsurface = support_psi.size
        expected_surface_shape = (support_nsurface, ltheta)
        for name, values in (
            ("theta", support_theta_table),
            ("nu", support_nu_table),
            ("jacobian", support_jacobian_table),
            ("R", support_R),
            ("z", support_z),
        ):
            if values.shape != expected_surface_shape:
                raise ValueError(
                    f"{name} support table has shape {values.shape}; "
                    f"expected {expected_surface_shape}."
                )
        for name, values in (
            ("q", support_q),
            ("F", support_F),
            ("I", support_I),
        ):
            if values.shape != (support_nsurface,):
                raise ValueError(
                    f"{name} support profile has shape {values.shape}; "
                    f"expected {(support_nsurface,)}."
                )
        if core_indices is None:
            core_selection = np.arange(support_nsurface, dtype=np.int64)
        else:
            core_selection = np.asarray(core_indices, dtype=np.int64)
            if (
                core_selection.ndim != 1
                or core_selection.size < 2
                or np.any(np.diff(core_selection) <= 0)
                or core_selection[0] < 0
                or core_selection[-1] >= support_nsurface
            ):
                raise ValueError(
                    "core_indices must select an increasing interior radial grid."
                )
        projected_coordinate_field = bool(
            coordinate_psi_field is not None
            and isinstance(symmetry_projection_audit, dict)
            and symmetry_projection_audit.get("applied", False)
        )
        flux_constraint_options: Dict[str, Any] = {}
        if projected_coordinate_field:
            flux_constraint_options = {
                "flux_constraint_R": np.asarray(R_fine, dtype=np.float64),
                "flux_constraint_z": np.asarray(z_fine, dtype=np.float64),
                "flux_constraint_psi": np.asarray(
                    coordinate_psi_field,
                    dtype=np.float64,
                ),
                "flux_constraint_tolerance": 1.0e-10,
                # Projected high-resolution equilibria can require more than
                # twelve Newton steps near the private support guards.  Keep
                # the strict 1e-10 constructor residual and allow the same
                # iteration budget used by the downstream angle inversion.
                "flux_constraint_max_iterations": 30,
            }

        # Convert direct surface-parameter tables to the same uniform magnetic
        # angle used by Rtransform/ztransform. Periodic cubic interpolation
        # avoids a directional seam bias in nu and the target Jacobian.
        magnetic_theta = np.linspace(0.0, 2.0*np.pi, ltheta)
        support_nu_magnetic = np.empty_like(
            support_nu_table,
            dtype=np.float64,
        )
        support_jacobian_magnetic = np.empty_like(
            support_jacobian_table,
            dtype=np.float64,
        )
        for radial_index in range(support_nsurface):
            theta_row = support_theta_table[radial_index]
            if np.any(np.diff(theta_row) <= 0.0):
                raise ValueError(
                    "Direct magnetic-angle table must be strictly increasing "
                    f"on surface {radial_index}."
                )
            nu_row = support_nu_table[radial_index].copy()
            jacobian_row = support_jacobian_table[radial_index].copy()
            nu_row[-1] = nu_row[0]
            jacobian_row[-1] = jacobian_row[0]
            support_nu_magnetic[radial_index] = CubicSpline(
                theta_row,
                nu_row,
                bc_type="periodic",
            )(magnetic_theta)
            support_jacobian_magnetic[radial_index] = CubicSpline(
                theta_row,
                jacobian_row,
                bc_type="periodic",
            )(magnetic_theta)
            support_nu_magnetic[radial_index, -1] = (
                support_nu_magnetic[radial_index, 0]
            )
            support_jacobian_magnetic[radial_index, -1] = (
                support_jacobian_magnetic[radial_index, 0]
            )

        R_axis_val = float(self.geometry.R_axis.values)
        z_axis_val = float(self.geometry.z_axis.values)
        coordinate_map = SpectralCoordinateMap(
            psi=support_psi,
            theta=magnetic_theta,
            R=support_R,
            z=support_z,
            nu=support_nu_magnetic,
            psi_axis=float(
                self.geometry.attrs.get('psi_ax', self._psi_ax_init)
            ),
            psi_boundary=float(
                self.geometry.attrs.get('psi_bdy', self._psi_edge_init)
            ),
            R_axis=R_axis_val,
            z_axis=z_axis_val,
            max_mode=spectral_max_mode,
            **flux_constraint_options,
        )

        # Everything public remains on the exact requested core grid. The
        # private coordinate map above retains hidden support on both sides.
        psigrid = support_psi[core_selection]
        thtable = support_theta_table[core_selection]
        nutable = support_nu_table[core_selection]
        jac = support_jacobian_table[core_selection]
        Rtransform = support_R[core_selection]
        ztransform = support_z[core_selection]
        qprof = support_q[core_selection]
        Fprof = support_F[core_selection]
        Iprof = support_I[core_selection]
        nu_magnetic = support_nu_magnetic[core_selection]
        jacobian_magnetic = support_jacobian_magnetic[core_selection]

        # Build coordinate grids (using geometry for axis)
        grr, gzz = np.meshgrid(R_fine, z_fine, indexing='ij')
        thetageom = np.arctan2(gzz - z_axis_val,
                               grr - R_axis_val)
        thetageom = np.mod(thetageom + 2*np.pi, 2*np.pi)
        if coordinate_psi_field is None:
            psirz = self.flux.psi.interp(
                R=R_fine,
                z=z_fine,
                method='cubic',
            )
        else:
            coordinate_psi_values = np.asarray(
                coordinate_psi_field,
                dtype=np.float64,
            )
            expected_psi_shape = (R_fine.size, z_fine.size)
            if coordinate_psi_values.shape != expected_psi_shape:
                raise ValueError(
                    "coordinate_psi_field has shape "
                    f"{coordinate_psi_values.shape}; expected "
                    f"{expected_psi_shape}."
                )
            psirz = xr.DataArray(
                coordinate_psi_values,
                dims=("R", "z"),
                coords={"R": R_fine, "z": z_fine},
            )

        # Add padding to theta grid.
        thetagrid = np.linspace(0, 2*np.pi, ltheta)
        dtheta = thetagrid[1] - thetagrid[0]
        thetagrid = np.linspace(-ntht_pad*dtheta,
                                2*np.pi + ntht_pad*dtheta,
                                ltheta + 2*ntht_pad)
        thtable_padded = thtable.copy()
        thtable_padded[:, -1] = 2*np.pi
        thtable_padded[:, 0] = 0.0
        leftside = thtable_padded[:, -(ntht_pad + 1):-1] - 2*np.pi
        rightside = thtable_padded[:, 1:ntht_pad + 1] + 2*np.pi
        thtable_padded = np.concatenate(
            (leftside, thtable_padded, rightside),
            axis=1,
        )
        thtable_da = xr.DataArray(
            thtable_padded,
            coords={'psi0': psigrid, 'thetageom': thetagrid},
            dims=('psi0', 'thetageom'),
        )

        leftside = nutable[:, -(ntht_pad + 1):-1]
        rightside = nutable[:, 1:ntht_pad + 1]
        nutable_padded = np.concatenate(
            (leftside, nutable, rightside),
            axis=1,
        )
        nutable_da = xr.DataArray(
            nutable_padded,
            coords={'psi0': psigrid, 'thetageom': thetagrid},
            dims=('psi0', 'thetageom'),
        )
        leftside = Rtransform[:, -(ntht_pad + 1):-1]
        rightside = Rtransform[:, 1:ntht_pad + 1]
        Rtransform_padded = np.concatenate(
            (leftside, Rtransform, rightside),
            axis=1,
        )
        leftside = ztransform[:, -(ntht_pad + 1):-1]
        rightside = ztransform[:, 1:ntht_pad + 1]
        ztransform_padded = np.concatenate(
            (leftside, ztransform, rightside),
            axis=1,
        )

        # The equilibrium flux remains authoritative outside the fitted
        # magnetic-coordinate annulus. Inside it, all coordinate derivatives
        # and both metric tensors are obtained by algebraically inverting the
        # one Fourier--spline map.
        psirz_vals = np.asarray(psirz, dtype=np.float64)
        # The spectral map is fitted on the full radial support grid, including
        # the private guard surfaces.  Use that same support interval when
        # constructing its R-Z differential fields.  Restricting this mask to
        # the public core grid removes the interpolation stencil exactly where
        # the guard surfaces are intended to protect the first/last core
        # surfaces.
        fit_min = float(np.min(support_psi))
        fit_max = float(np.max(support_psi))
        coordinate_domain = (
            np.isfinite(psirz_vals)
            & (psirz_vals >= fit_min)
            & (psirz_vals <= fit_max)
        )
        shape_rz = psirz_vals.shape
        theta_Rz = np.full(shape_rz, np.nan, dtype=np.float64)
        nu_Rz = np.full(shape_rz, np.nan, dtype=np.float64)
        jac_Rz = np.full(shape_rz, np.nan, dtype=np.float64)

        derivative_names = (
            'dR_dpsi', 'dR_dtheta', 'dR_dzeta',
            'dz_dpsi', 'dz_dtheta', 'dz_dzeta',
            'dphi_dpsi', 'dphi_dtheta', 'dphi_dzeta',
            'dPsi_dr', 'dPsi_dz', 'dPsi_dphi',
            'dTheta_dr', 'dTheta_dz', 'dTheta_dphi',
            'dzeta_dr', 'dzeta_dz', 'dzeta_dphi',
            'direct_det_Rz',
        )
        map_derivatives = {
            name: np.full(shape_rz, np.nan, dtype=np.float64)
            for name in derivative_names
        }

        if projected_coordinate_field:
            physical_psi_spline = RectBivariateSpline(
                R_fine,
                z_fine,
                psirz_vals,
                kx=min(3, R_fine.size - 1),
                ky=min(3, z_fine.size - 1),
                s=0.0,
            )
            equilibrium_dPsi_dr = np.asarray(
                physical_psi_spline(R_fine, z_fine, dx=1, dy=0),
                dtype=np.float64,
            )
            equilibrium_dPsi_dz = np.asarray(
                physical_psi_spline(R_fine, z_fine, dx=0, dy=1),
                dtype=np.float64,
            )
        else:
            d_dr = FinDiff(0, R_fine[1] - R_fine[0], 1, acc=4)
            d_dz = FinDiff(1, z_fine[1] - z_fine[0], 1, acc=4)
            equilibrium_dPsi_dr = np.asarray(d_dr(psirz), dtype=np.float64)
            equilibrium_dPsi_dz = np.asarray(d_dz(psirz), dtype=np.float64)

        flat_indices = np.flatnonzero(coordinate_domain.ravel())
        flat_R = grr.ravel()
        flat_z = gzz.ravel()
        flat_psi = psirz_vals.ravel()
        initial_theta = thetageom.ravel()
        chunk_size = 20_000
        for chunk_start in range(0, flat_indices.size, chunk_size):
            chunk_indices = flat_indices[chunk_start:chunk_start + chunk_size]
            psi_chunk = flat_psi[chunk_indices]
            R_chunk = flat_R[chunk_indices]
            z_chunk = flat_z[chunk_indices]
            theta_chunk = coordinate_map.solve_theta(
                psi=psi_chunk,
                R=R_chunk,
                z=z_chunk,
                initial_theta=initial_theta[chunk_indices],
                tolerance=5.0e-11,
                max_iterations=30,
            )
            differential = coordinate_map.differentials(
                psi_chunk,
                theta_chunk,
            )
            direct = differential.direct.copy()
            map_radius = np.asarray(
                differential.values['R'],
                dtype=np.float64,
            )
            # ``solve_theta`` locates the closest point on the fitted surface;
            # between radial knots that point can differ slightly from the
            # physical R-Z grid point carrying the equilibrium psi value.
            # Cylindrical orthonormal toroidal components must use the actual
            # evaluation radius so the public tangent/gradient matrices and
            # their determinant remain exactly reciprocal.
            direct[:, 1, :] *= (R_chunk / map_radius)[:, None]
            direct[:, 1, 2] = R_chunk
            grad_R = equilibrium_dPsi_dr.ravel()[chunk_indices]
            grad_z = equilibrium_dPsi_dz.ravel()[chunk_indices]
            grad_squared = grad_R**2 + grad_z**2
            if np.any(grad_squared <= np.finfo(np.float64).tiny):
                raise ValueError(
                    "Equilibrium poloidal-flux gradient vanishes in the "
                    "retained magnetic-coordinate annulus."
                )
            # Enforce the exact physical-flux differential identities on the
            # fitted map: grad(psi).x_psi=1 and grad(psi).x_theta=0.
            for column, target_dot in ((0, 1.0), (1, 0.0)):
                current_dot = (
                    grad_R * direct[:, 0, column]
                    + grad_z * direct[:, 2, column]
                )
                correction = (target_dot - current_dot) / grad_squared
                direct[:, 0, column] += correction * grad_R
                direct[:, 2, column] += correction * grad_z

            jacobian_chunk = np.linalg.det(direct)
            if np.any(~np.isfinite(jacobian_chunk)) or np.any(
                np.isclose(jacobian_chunk, 0.0)
            ):
                raise ValueError(
                    "Flux-constrained coordinate differential is singular."
                )
            inverse = np.linalg.inv(direct)

            theta_Rz.ravel()[chunk_indices] = theta_chunk
            nu_Rz.ravel()[chunk_indices] = differential.values['nu']
            jac_Rz.ravel()[chunk_indices] = jacobian_chunk

            map_derivatives['dR_dpsi'].ravel()[chunk_indices] = direct[:, 0, 0]
            map_derivatives['dR_dtheta'].ravel()[chunk_indices] = direct[:, 0, 1]
            map_derivatives['dR_dzeta'].ravel()[chunk_indices] = direct[:, 0, 2]
            map_derivatives['dz_dpsi'].ravel()[chunk_indices] = direct[:, 2, 0]
            map_derivatives['dz_dtheta'].ravel()[chunk_indices] = direct[:, 2, 1]
            map_derivatives['dz_dzeta'].ravel()[chunk_indices] = direct[:, 2, 2]
            map_derivatives['dphi_dpsi'].ravel()[chunk_indices] = (
                direct[:, 1, 0] / R_chunk
            )
            map_derivatives['dphi_dtheta'].ravel()[chunk_indices] = (
                direct[:, 1, 1] / R_chunk
            )
            map_derivatives['dphi_dzeta'].ravel()[chunk_indices] = (
                direct[:, 1, 2] / R_chunk
            )
            map_derivatives['dPsi_dr'].ravel()[chunk_indices] = inverse[:, 0, 0]
            map_derivatives['dPsi_dphi'].ravel()[chunk_indices] = (
                R_chunk * inverse[:, 0, 1]
            )
            map_derivatives['dPsi_dz'].ravel()[chunk_indices] = inverse[:, 0, 2]
            map_derivatives['dTheta_dr'].ravel()[chunk_indices] = inverse[:, 1, 0]
            map_derivatives['dTheta_dphi'].ravel()[chunk_indices] = (
                R_chunk * inverse[:, 1, 1]
            )
            map_derivatives['dTheta_dz'].ravel()[chunk_indices] = inverse[:, 1, 2]
            map_derivatives['dzeta_dr'].ravel()[chunk_indices] = inverse[:, 2, 0]
            map_derivatives['dzeta_dphi'].ravel()[chunk_indices] = (
                R_chunk * inverse[:, 2, 1]
            )
            map_derivatives['dzeta_dz'].ravel()[chunk_indices] = inverse[:, 2, 2]
            map_derivatives['direct_det_Rz'].ravel()[chunk_indices] = (
                inverse[:, 0, 0] * inverse[:, 1, 2]
                - inverse[:, 0, 2] * inverse[:, 1, 0]
            )

        thtable_Rz = xr.DataArray(
            theta_Rz,
            coords=(R_fine, z_fine),
            dims=('R', 'z'),
        )
        nutable_Rz = xr.DataArray(
            nu_Rz,
            coords=(R_fine, z_fine),
            dims=('R', 'z'),
        )

        # Preserve finite equilibrium gradients on the complete R-Z grid, as
        # required by the public EQDSK field interface. Within the coordinate
        # annulus these are replaced by the reciprocal map derivatives.
        map_derivatives['dPsi_dr'][~coordinate_domain] = (
            equilibrium_dPsi_dr[~coordinate_domain]
        )
        map_derivatives['dPsi_dz'][~coordinate_domain] = (
            equilibrium_dPsi_dz[~coordinate_domain]
        )
        map_derivatives['dPsi_dphi'][~coordinate_domain] = 0.0

        dR_dpsi = map_derivatives['dR_dpsi']
        dR_dtheta = map_derivatives['dR_dtheta']
        dR_dzeta = map_derivatives['dR_dzeta']
        dz_dpsi = map_derivatives['dz_dpsi']
        dz_dtheta = map_derivatives['dz_dtheta']
        dz_dzeta = map_derivatives['dz_dzeta']
        dphi_dpsi = map_derivatives['dphi_dpsi']
        dphi_dtheta = map_derivatives['dphi_dtheta']
        dphi_dzeta = map_derivatives['dphi_dzeta']
        dPsi_dr = map_derivatives['dPsi_dr']
        dPsi_dz = map_derivatives['dPsi_dz']
        dPsi_dphi = map_derivatives['dPsi_dphi']
        dTheta_dr = map_derivatives['dTheta_dr']
        dTheta_dz = map_derivatives['dTheta_dz']
        dTheta_dphi = map_derivatives['dTheta_dphi']
        dzeta_dr = map_derivatives['dzeta_dr']
        dzeta_dz = map_derivatives['dzeta_dz']
        dzeta_dphi = map_derivatives['dzeta_dphi']
        direct_det_Rz = map_derivatives['direct_det_Rz']

        if self.flux_normalization == "Wb/rad":
            jacobian_units = "m**3/Wb"
            position_per_flux_units = "m*rad/Wb"
            angle_per_flux_units = "rad**2/Wb"
            flux_per_length_units = "Wb/(rad*m)"
            flux_per_angle_units = "Wb/rad**2"
            direct_det_units = "Wb/m**2"
        else:
            jacobian_units = "m**3/(Wb*rad)"
            position_per_flux_units = "m/Wb"
            angle_per_flux_units = "rad/Wb"
            flux_per_length_units = "Wb/m"
            flux_per_angle_units = "Wb/rad"
            direct_det_units = "Wb*rad/m**2"

        support_metadata = dict(radial_support_metadata or {})
        support_metadata.update({
            "support_nsurface": int(support_nsurface),
            "core_nsurface": int(core_selection.size),
            "support_psi_min": float(support_psi[0]),
            "support_psi_max": float(support_psi[-1]),
            "core_psi_min": float(psigrid[0]),
            "core_psi_max": float(psigrid[-1]),
        })
        map_symmetry_audit = coordinate_map.up_down_symmetry_audit
        if symmetry_projection_audit is None:
            symmetry_audit = map_symmetry_audit
        else:
            symmetry_audit = dict(symmetry_projection_audit)
            symmetry_audit["coordinate_map_geometry_residual"] = np.asarray(
                map_symmetry_audit["geometry_residual"],
                dtype=np.float64,
            )
            symmetry_audit["coordinate_map_field_residuals"] = (
                map_symmetry_audit["field_residuals"]
            )
            # The map has already been constructed from projected physical
            # contours and fields. Preserve the source audit on the map so
            # downstream bridge users see what was changed, not merely the
            # round-off-level residual after projection.
        symmetry_audit["flux_constraint"] = dict(
            coordinate_map.flux_constraint_audit
        )
        coordinate_map.up_down_symmetry_audit = symmetry_audit
        symmetry_geometry_max = float(
            np.max(symmetry_audit["geometry_residual"])
        )

        # Build coordinate dataset
        magcoords = xr.Dataset(attrs={
            "radial_support_family": str(
                support_metadata.get("family", "unspecified")
            ),
            "radial_guard_surfaces_requested": int(
                support_metadata.get("requested_guard_surfaces", 0)
            ),
            "radial_support_nsurface": int(
                support_metadata["support_nsurface"]
            ),
            "radial_core_nsurface": int(
                support_metadata["core_nsurface"]
            ),
            "radial_inner_guard_surfaces": int(
                support_metadata.get("inner_guard_surfaces", 0)
            ),
            "radial_outer_guard_surfaces": int(
                support_metadata.get("outer_guard_surfaces", 0)
            ),
            "radial_support_psi_min": float(
                support_metadata["support_psi_min"]
            ),
            "radial_support_psi_max": float(
                support_metadata["support_psi_max"]
            ),
            "radial_core_psi_min": float(support_metadata["core_psi_min"]),
            "radial_core_psi_max": float(support_metadata["core_psi_max"]),
            "up_down_symmetry_projection_applied": int(
                symmetry_audit["applied"]
            ),
            "up_down_symmetry_projected_equilibrium": int(
                coordinate_psi_field is not None
                and symmetry_audit["applied"]
            ),
            "up_down_symmetry_input_geometry_residual": (
                symmetry_geometry_max
            ),
            "up_down_symmetry_projected_flux_residual": float(
                np.max(
                    np.asarray(
                        symmetry_audit.get(
                            "projected_flux_residual",
                            np.asarray([0.0]),
                        ),
                        dtype=np.float64,
                    )
                )
            ),
            "up_down_symmetry_projected_bridge_flux_residual": float(
                np.max(
                    np.asarray(
                        symmetry_audit.get(
                            "projected_bridge_flux_residual",
                            np.asarray([0.0]),
                        ),
                        dtype=np.float64,
                    )
                )
            ),
            "up_down_symmetry_flux_constraint_residual": float(
                coordinate_map.flux_constraint_audit.get(
                    "validation_normalized_residual",
                    0.0,
                )
            ),
            "up_down_symmetry_flux_constraint_iterations": int(
                coordinate_map.flux_constraint_audit.get(
                    "validation_iterations",
                    0,
                )
            ),
            "up_down_symmetry_flux_constraint_min_abs_F_sigma": float(
                coordinate_map.flux_constraint_audit.get(
                    "validation_minimum_abs_F_sigma",
                    np.nan,
                )
            ),
        })
        psi_attrs = {
            'name': 'psi',
            'units': self.flux_normalization,
            'desc': 'Poloidal flux',
            'short_name': '$\\Psi$',
        }
        if projected_coordinate_field:
            psi_attrs.update({
                'interpolation_order_R': 3,
                'interpolation_order_z': 3,
                'projected_reflection_knot_union': 1,
            })
        magcoords['psi'] = xr.DataArray(
            psirz,
            dims=('R', 'z'),
            coords={'R': R_fine, 'z': z_fine},
            attrs=psi_attrs,
        )
        magcoords['theta'] = thtable_da
        magcoords['theta'].attrs = {'name': 'theta', 'units': 'rad',
                                    'desc': 'Magnetic poloidal angle',
                                    'short_name': '$\\Theta*$'}
        magcoords['nu'] = nutable_da
        magcoords['nu'].attrs = {
            'name': 'nu',
            'units': 'rad',
            'desc': 'Toroidal gauge shift nu(psi, theta) in zeta = phi + nu',
            'short_name': '$\\nu$',
            'gauge_relation': 'zeta = phi + nu',
        }

        magcoords.R.attrs = {'name': 'R', 'units': 'm', 'desc': 'Major radius',
                             'short_name': 'R'}
        magcoords.z.attrs = {'name': 'z', 'units': 'm', 'desc': 'Height',
                             'short_name': 'z'}
        magcoords.thetageom.attrs = {'name': 'thetageom', 'units': 'rad',
                                     'desc': 'Geometrical poloidal angle',
                                     'short_name': '$\\Theta_{geom}$'}

        # Build derivatives dataset
        magdevs = xr.Dataset()
        magdevs['jacobian'] = xr.DataArray(jac_Rz, dims=('R', 'z'),
                                           coords={'R': R_fine, 'z': z_fine},
                                           attrs={'name': 'jacobian', 'units': jacobian_units,
                                                  'desc': ('Signed physical Jacobian '
                                                           '[grad(psi) . (grad(theta) x grad(zeta))]**-1'),
                                                  'short_name': '$\\mathcal{J}$'})
        
        # Add all derivative arrays with proper attributes
        full_grid_flux_derivatives = frozenset({
            'dPsi_dr',
            'dPsi_dz',
            'dPsi_dphi',
        })
        derivatives = {
            'dR_dpsi': (dR_dpsi, position_per_flux_units, 'Partial derivative of R with respect to poloidal flux'),
            'dR_dtheta': (dR_dtheta, 'm/rad', 'Partial derivative of R with respect to magnetic poloidal angle'),
            'dR_dzeta': (dR_dzeta, 'm/rad', 'Partial derivative of R with respect to magnetic toroidal angle'),
            'dz_dpsi': (dz_dpsi, position_per_flux_units, 'Partial derivative of z with respect to poloidal flux'),
            'dz_dtheta': (dz_dtheta, 'm/rad', 'Partial derivative of z with respect to magnetic poloidal angle'),
            'dz_dzeta': (dz_dzeta, 'm/rad', 'Partial derivative of z with respect to magnetic toroidal angle'),
            'dphi_dpsi': (dphi_dpsi, angle_per_flux_units, 'Partial derivative of phi with respect to poloidal flux'),
            'dphi_dtheta': (dphi_dtheta, 'rad/rad', 'Partial derivative of phi with respect to magnetic poloidal angle'),
            'dphi_dzeta': (dphi_dzeta, 'rad/rad', 'Partial derivative of phi with respect to magnetic toroidal angle'),
            'dPsi_dr': (dPsi_dr, flux_per_length_units, 'Partial derivative of poloidal flux with respect to R'),
            'dPsi_dz': (dPsi_dz, flux_per_length_units, 'Partial derivative of poloidal flux with respect to z'),
            'dPsi_dphi': (dPsi_dphi, flux_per_angle_units, 'Partial derivative of poloidal flux with respect to phi'),
            'dTheta_dr': (dTheta_dr, 'rad/m', 'Partial derivative of magnetic poloidal angle with respect to R'),
            'dTheta_dz': (dTheta_dz, 'rad/m', 'Partial derivative of magnetic poloidal angle with respect to z'),
            'dTheta_dphi': (dTheta_dphi, 'rad/rad', 'Partial derivative of magnetic poloidal angle with respect to phi'),
            'dzeta_dr': (dzeta_dr, 'rad/m', 'Partial derivative of magnetic toroidal angle with respect to R'),
            'dzeta_dz': (dzeta_dz, 'rad/m', 'Partial derivative of magnetic toroidal angle with respect to z'),
            'dzeta_dphi': (dzeta_dphi, 'rad/rad', 'Partial derivative of magnetic toroidal angle with respect to phi'),
            'direct_det_Rz': (direct_det_Rz, direct_det_units,
                              'Direct determinant det(partial(psi, theta)/partial(R, Z)); physical J = -R/direct_det_Rz'),
        }
        
        for name, (data, units, desc) in derivatives.items():
            short_name = name.replace("_", " / \\partial ")
            magdevs[name] = xr.DataArray(data, dims=('R', 'z'),
                                        coords={'R': R_fine, 'z': z_fine},
                                        attrs={'name': name, 'units': units,
                                               'desc': desc,
                                               'short_name': f'$\\partial {short_name}$'})

        for name in full_grid_flux_derivatives:
            magdevs[name].attrs.update({
                'validity_domain': 'finite_equilibrium_RZ_grid',
            })

        magdevs.R.attrs = {'name': 'R', 'units': 'm', 'desc': 'Major radius',
                           'short_name': 'R'}
        magdevs.z.attrs = {'name': 'z', 'units': 'm', 'desc': 'Height',
                           'short_name': 'z'}
        
        # Solver-facing Boozer convention: I=B_Theta and h=J*B^2=I+qF.
        I_values = np.asarray(Iprof, dtype=float)

        magdevs['q'] = xr.DataArray(qprof, dims=('psi0',),
                                    coords={'psi0': psigrid},
                                    attrs={'name': 'q', 'units': '',
                                           'desc': 'Safety factor',
                                           'short_name': '$q$'})
        magdevs['F'] = xr.DataArray(Fprof, dims=('psi0',),
                                    coords={'psi0': psigrid},
                                    attrs={'name': 'F', 'units': 'T*m',
                                           'desc': 'F(psi) function in GS equation = RB_T', 
                                             'short_name': '$F$'})
        magdevs['I'] = xr.DataArray(I_values, dims=('psi0',),
                                    coords={'psi0': psigrid},
                                    attrs={'name': 'I', 'units': 'T*m',
                                           'desc': ('Covariant Boozer magnetic-field '
                                                    'coefficient I = B_Theta'),
                                           'short_name': r'$I=B_\Theta$'})
        magdevs['h'] = magdevs.q * magdevs.F + magdevs.I
        magdevs['h'].attrs = {
            'name': 'h',
            'desc': 'Signed Boozer Jacobian factor h = J*B**2 = I + qF',
            'units': 'T*m',
            'short_name': '$h$',
        }

        # Add inverse transformation
        magcoords['R_inv'] = xr.DataArray(Rtransform_padded,
                                          dims=('psi0', 'theta_star'),
                                          coords={'psi0': psigrid,
                                                  'theta_star': thetagrid},
                                          attrs={'name': 'R_inv',
                                                 'desc': 'R = R(psi, theta*)',
                                                 'units': 'm',
                                                 'short_name': '$R(\\Psi, \\Theta^*)$'})
        
        magcoords['z_inv'] = xr.DataArray(ztransform_padded,
                                          dims=('psi0', 'theta_star'),
                                          coords={'psi0': psigrid,
                                                  'theta_star': thetagrid},
                                          attrs={'name': 'z_inv',
                                                 'desc': 'z = z(psi, theta*)',
                                                 'units': 'm',
                                                 'short_name': '$z(\\Psi, \\Theta^*)$'})

        magcoords.theta_star.attrs = {'name': 'theta_star', 'units': 'rad',
                                      'desc': 'Magnetic poloidal angle',
                                      'short_name': '$\\Theta^*$'}
        psi_axis = float(self.geometry.attrs.get('psi_ax', self._psi_ax_init))
        psi_boundary = float(
            self.geometry.attrs.get('psi_bdy', self._psi_edge_init)
        )
        magcoords.psi0.attrs = {
            'name': 'psi0',
            'units': self.flux_normalization,
            'desc': 'Strictly increasing physical poloidal-flux spline coordinate',
            'short_name': '$\\Psi_0$',
            'psi_axis': psi_axis,
            'psi_boundary': psi_boundary,
            'normalization': (
                'psi_N = (psi - psi_axis) / (psi_boundary - psi_axis)'
            ),
        }
        magdevs.psi0.attrs = magcoords.psi0.attrs.copy()

        # Build the LCFS mask from finite physical psi rather than interpolating
        # rho, whose source grid legitimately contains NaNs outside the LCFS.
        psi_span = psi_boundary - psi_axis
        if abs(psi_span) < 1.0e-14:
            raise ValueError("Axis and boundary poloidal flux must be distinct.")
        rho_squared = (np.asarray(psirz, dtype=float) - psi_axis) / psi_span
        inside_lcfs = (
            np.isfinite(rho_squared)
            & (rho_squared >= -1.0e-12)
            & (rho_squared <= 1.0 + 1.0e-12)
        )
        inside_lcfs_da = xr.DataArray(
            inside_lcfs,
            dims=('R', 'z'),
            coords={'R': R_fine, 'z': z_fine},
            attrs={
                'name': 'inside_lcfs',
                'units': '',
                'desc': 'True where normalized physical poloidal flux is in [0, 1]',
            },
        )
        magcoords['inside_lcfs'] = inside_lcfs_da
        coordinate_domain_da = xr.DataArray(
            coordinate_domain & inside_lcfs,
            dims=('R', 'z'),
            coords={'R': R_fine, 'z': z_fine},
            attrs={
                'name': 'inside_coordinate_domain',
                'units': '',
                'desc': (
                    'True where the common magnetic-coordinate map is fitted'
                ),
            },
        )
        magcoords['inside_coordinate_domain'] = coordinate_domain_da
        for name, field in list(magdevs.data_vars.items()):
            if (
                field.dims == ('R', 'z')
                and name not in full_grid_flux_derivatives
            ):
                attrs = field.attrs.copy()
                magdevs[name] = field.where(coordinate_domain_da)
                magdevs[name].attrs = attrs

        # Store computed coordinates for easy access
        mag_coords_obj = MagneticCoordinates(magcoords, magdevs,
                                              Raxis=R_axis_val,
                                              zaxis=z_axis_val,
                                              pad=ntht_pad)
        # Private numerical engine used by the unchanged public transform
        # methods. It is intentionally not serialized as an xarray object.
        mag_coords_obj._coordinate_map = coordinate_map

        node_psi, node_theta = np.meshgrid(
            psigrid,
            magnetic_theta,
            indexing='ij',
        )
        node_differential = coordinate_map.differentials(
            node_psi,
            node_theta,
        )
        target_scale = np.maximum(
            np.max(np.abs(jacobian_magnetic), axis=1),
            np.finfo(np.float64).tiny,
        )
        jacobian_relative_residual = np.max(
            np.abs(
                node_differential.jacobian
                - jacobian_magnetic
            ),
            axis=1,
        ) / target_scale
        mag_coords_obj._coordinate_diagnostics = {
            'jacobian_relative_residual': jacobian_relative_residual,
            'radial_support': support_metadata,
            'up_down_symmetry': symmetry_audit,
        }
        mag_coords_obj.deriv['jacobian'].attrs.update({
            'construction': 'determinant of the common Fourier-spline map',
            'target_jacobian_max_relative_residual': float(
                np.max(jacobian_relative_residual)
            ),
        })
        
        # Cache the result for easy access
        if not hasattr(self, '_magnetic_coordinates_cache'):
            self._magnetic_coordinates_cache = {}
        coord_sys_lower = coordinate_system.lower()
        self._magnetic_coordinates_cache[coord_sys_lower] = mag_coords_obj
        
        # Also set as main attribute if this is the first/only coordinate system
        self.magnetic_coordinates = mag_coords_obj
        self.coord_sys = coordinate_system.lower()
        
        return mag_coords_obj
    
    @property
    def coords(self) -> Dict[str, MagneticCoordinates]:
        """
        Dictionary of computed magnetic coordinate systems.
        
        Returns
        -------
        dict
            Dictionary mapping coordinate system names to MagneticCoordinates objects
            
        Examples
        --------
        >>> eq.compute_coordinates('boozer')
        >>> eq.compute_coordinates('hamada')
        >>> eq.coords['boozer']  # Access Boozer coordinates
        >>> eq.coords['hamada']  # Access Hamada coordinates
        """
        if not hasattr(self, '_magnetic_coordinates_cache'):
            self._magnetic_coordinates_cache = {}
        return self._magnetic_coordinates_cache
