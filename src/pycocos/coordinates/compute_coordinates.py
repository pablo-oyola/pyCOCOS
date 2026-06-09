"""
Generic coordinate computation using Jacobian-based architecture.

This module provides a generic function to compute magnetic coordinates
for any coordinate system by using the appropriate Jacobian function.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
from .field_lines import integrate_pol_field_line
from .jacobians import compute_boozer_jacobian
from .jacobian_builders import boozer_consistency_residual, make_jacobian_context

_SURFACE_RECONSTRUCTION_MODE = "spectral"
_DEFAULT_SPECTRAL_MAX_FOURIER_MODE = 16
_MIN_SPECTRAL_SURFACES = 5
_THETA_GEOM_POINTS = 7200


def _normalize_rho_labels(
    rho_at_psi: Optional[np.ndarray],
    npsi: int,
) -> np.ndarray:
    """
    Return a finite radial label for the spectral smoothing stage.
    """
    if npsi <= 1:
        return np.zeros((npsi,), dtype=np.float64)

    if rho_at_psi is None:
        return np.linspace(0.0, 1.0, npsi, dtype=np.float64)

    rho = np.asarray(rho_at_psi, dtype=np.float64).reshape(-1)
    if rho.size != npsi or not np.all(np.isfinite(rho)):
        return np.linspace(0.0, 1.0, npsi, dtype=np.float64)

    if np.ptp(rho) < 1.0e-14:
        return np.linspace(0.0, 1.0, npsi, dtype=np.float64)

    return rho


def _validate_spectral_max_mode(spectral_max_mode: int) -> int:
    """
    Validate the retained maximum poloidal Fourier mode.
    """
    if isinstance(spectral_max_mode, bool) or not isinstance(
        spectral_max_mode, (int, np.integer)
    ):
        raise TypeError("spectral_max_mode must be an integer.")

    spectral_max_mode = int(spectral_max_mode)
    if spectral_max_mode < 0:
        raise ValueError("spectral_max_mode must be >= 0.")

    return spectral_max_mode


def _collapse_duplicate_samples(
    samples: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average values that share the same radial label.
    """
    unique_samples, inverse = np.unique(samples, return_inverse=True)
    if unique_samples.size == samples.size:
        return samples, values

    collapsed = np.zeros((unique_samples.size,) + values.shape[1:], dtype=values.dtype)
    counts = np.zeros(unique_samples.size, dtype=np.int64)
    for idx, target in enumerate(inverse):
        collapsed[target] += values[idx]
        counts[target] += 1

    reshape = (counts.size,) + (1,) * (values.ndim - 1)
    collapsed /= counts.reshape(reshape)
    return unique_samples, collapsed


def _trace_flux_surfaces(
    Rgrid: np.ndarray,
    zgrid: np.ndarray,
    br: np.ndarray,
    bz: np.ndarray,
    bphi: np.ndarray,
    R_at_psi: np.ndarray,
    zaxis: float,
    raxis: float,
    thetageom: np.ndarray,
    integration_sign: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trace raw flux surfaces and resample them onto a uniform geometrical theta grid.
    """
    npsi = len(R_at_psi)
    ntheta = thetageom.size - 1
    R_raw = np.zeros((npsi, ntheta), dtype=np.float64)
    z_raw = np.zeros((npsi, ntheta), dtype=np.float64)
    br_raw = np.zeros((npsi, ntheta), dtype=np.float64)
    bz_raw = np.zeros((npsi, ntheta), dtype=np.float64)
    bphi_raw = np.zeros((npsi, ntheta), dtype=np.float64)

    for ii, ir in enumerate(R_at_psi):
        Rline, zline, brline, bzline, bphiline, iend = integrate_pol_field_line(
            Rgrid,
            zgrid,
            br,
            bz,
            bphi,
            ir,
            zaxis,
            integration_sign=integration_sign,
        )

        Rline = np.asarray(Rline[:iend], dtype=np.float64)
        zline = np.asarray(zline[:iend], dtype=np.float64)
        brline = np.asarray(brline[:iend], dtype=np.float64)
        bzline = np.asarray(bzline[:iend], dtype=np.float64)
        bphiline = np.asarray(bphiline[:iend], dtype=np.float64)

        thetaval = np.mod(np.arctan2(zline - zaxis, Rline - raxis), 2.0 * np.pi)
        R_full = np.interp(thetageom, thetaval, Rline, period=2.0 * np.pi)
        z_full = np.interp(thetageom, thetaval, zline, period=2.0 * np.pi)
        br_full = np.interp(thetageom, thetaval, brline, period=2.0 * np.pi)
        bz_full = np.interp(thetageom, thetaval, bzline, period=2.0 * np.pi)
        bphi_full = np.interp(thetageom, thetaval, bphiline, period=2.0 * np.pi)

        R_raw[ii, :] = R_full[:-1]
        z_raw[ii, :] = z_full[:-1]
        br_raw[ii, :] = br_full[:-1]
        bz_raw[ii, :] = bz_full[:-1]
        bphi_raw[ii, :] = bphi_full[:-1]

    return R_raw, z_raw, br_raw, bz_raw, bphi_raw


def _smooth_fourier_coefficients(
    coefficients: np.ndarray,
    rho_at_psi: np.ndarray,
) -> np.ndarray:
    """
    Smooth Fourier coefficients independently along the radial direction.
    """
    sort_idx = np.argsort(rho_at_psi)
    rho_sorted = np.asarray(rho_at_psi[sort_idx], dtype=np.float64)
    coeff_sorted = np.asarray(coefficients[sort_idx], dtype=np.complex128)
    rho_unique, coeff_unique = _collapse_duplicate_samples(rho_sorted, coeff_sorted)

    if rho_unique.size < _MIN_SPECTRAL_SURFACES:
        raise ValueError(
            "Too few unique surfaces for spectral smoothing "
            f"({rho_unique.size} < {_MIN_SPECTRAL_SURFACES})."
        )

    spline_order = min(3, rho_unique.size - 1)
    coeff_smooth = np.zeros((rho_at_psi.size, coeff_unique.shape[1]), dtype=np.complex128)

    for mode in range(coeff_unique.shape[1]):
        coeff_mode = coeff_unique[:, mode]
        real_part = np.asarray(coeff_mode.real, dtype=np.float64)
        imag_part = np.asarray(coeff_mode.imag, dtype=np.float64)

        smooth_real = 1.0e-4 * rho_unique.size * float(np.var(real_part))
        smooth_imag = 1.0e-4 * rho_unique.size * float(np.var(imag_part))

        real_spline = UnivariateSpline(rho_unique, real_part, k=spline_order, s=smooth_real)
        imag_spline = UnivariateSpline(rho_unique, imag_part, k=spline_order, s=smooth_imag)
        coeff_smooth[:, mode] = real_spline(rho_at_psi) + 1j * imag_spline(rho_at_psi)

    return coeff_smooth


def _build_spectral_surfaces(
    R_raw: np.ndarray,
    z_raw: np.ndarray,
    rho_at_psi: np.ndarray,
    spectral_max_mode: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build globally smoothed spectral reconstructions of the traced surfaces.
    """
    npsi, ntheta = R_raw.shape
    if npsi < _MIN_SPECTRAL_SURFACES:
        raise ValueError(
            "Too few surfaces for spectral reconstruction "
            f"({npsi} < {_MIN_SPECTRAL_SURFACES})."
        )

    R_coeff = np.fft.rfft(R_raw, axis=1)
    z_coeff = np.fft.rfft(z_raw, axis=1)
    max_mode = min(spectral_max_mode, R_coeff.shape[1] - 1)

    R_trunc = np.zeros_like(R_coeff)
    z_trunc = np.zeros_like(z_coeff)
    R_trunc[:, : max_mode + 1] = _smooth_fourier_coefficients(
        R_coeff[:, : max_mode + 1],
        rho_at_psi,
    )
    z_trunc[:, : max_mode + 1] = _smooth_fourier_coefficients(
        z_coeff[:, : max_mode + 1],
        rho_at_psi,
    )

    R_fit = np.fft.irfft(R_trunc, n=ntheta, axis=1)
    z_fit = np.fft.irfft(z_trunc, n=ntheta, axis=1)
    return np.asarray(R_fit, dtype=np.float64), np.asarray(z_fit, dtype=np.float64)


def _sample_surface_fields(
    R: np.ndarray,
    z: np.ndarray,
    br_surface: Optional[np.ndarray],
    bz_surface: Optional[np.ndarray],
    bphi_surface: Optional[np.ndarray],
    br_interp: RegularGridInterpolator,
    bz_interp: RegularGridInterpolator,
    bphi_interp: RegularGridInterpolator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return magnetic-field samples along a flux surface.
    """
    if br_surface is not None and bz_surface is not None and bphi_surface is not None:
        return (
            np.asarray(br_surface, dtype=np.float64),
            np.asarray(bz_surface, dtype=np.float64),
            np.asarray(bphi_surface, dtype=np.float64),
        )

    points = np.column_stack((R, z))
    return (
        np.asarray(br_interp(points), dtype=np.float64),
        np.asarray(bz_interp(points), dtype=np.float64),
        np.asarray(bphi_interp(points), dtype=np.float64),
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
    br_interp: RegularGridInterpolator,
    bz_interp: RegularGridInterpolator,
    bphi_interp: RegularGridInterpolator,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the coordinate tables for one surface.
    """
    br_vals, bz_vals, bphi_vals = _sample_surface_fields(
        R,
        z,
        br_surface,
        bz_surface,
        bphi_surface,
        br_interp,
        bz_interp,
        bphi_interp,
    )

    R_full = np.empty(thetageom.size, dtype=np.float64)
    z_full = np.empty(thetageom.size, dtype=np.float64)
    R_full[:-1] = R
    R_full[-1] = R[0]
    z_full[:-1] = z
    z_full[-1] = z[0]

    dR = np.diff(R_full)
    dZ = np.diff(z_full)
    dlp = np.sqrt(dR**2 + dZ**2)

    bnorm = np.sqrt(br_vals**2 + bz_vals**2 + bphi_vals**2)
    bpol = np.sqrt(br_vals**2 + bz_vals**2)
    bpol_safe = np.where(bpol > 1.0e-14, bpol, 1.0e-14)
    ds = (dR * br_vals + dZ * bz_vals) / bpol_safe
    dlbpol = dR * br_vals + dZ * bz_vals

    Iprof = np.sum(dlbpol) / (2.0 * np.pi)
    # F = R*B_phi is a flux function (constant on each flux surface in an
    # axisymmetric equilibrium). Sampling at a single tracing point made the
    # surface-integrated q profile sensitive to local tracing noise; use the
    # arc-weighted average over the closed poloidal loop instead, which is
    # robust to local sample-spacing variations and converges to the exact
    # flux function as the surface tracing refines.
    dlp_at_vertex = 0.5 * (np.roll(dlp, 1) + dlp)
    arc_total = float(np.sum(dlp_at_vertex))
    if arc_total > 0.0:
        Fprof = float(np.sum(R * bphi_vals * dlp_at_vertex) / arc_total)
    else:
        Fprof = float(R[0] * bphi_vals[0])
    qprof = np.sum(ds * Fprof / (R**2 * bpol_safe)) / (2.0 * np.pi)

    jac_context = make_jacobian_context(
        coordinate_system=coordinate_system,
        R=R,
        B=bnorm,
        Bpol=bpol_safe,
        dlp=dlp,
        I=Iprof,
        F=Fprof,
        q=qprof,
    )
    jac = np.asarray(jacobian_func(jac_context), dtype=np.float64)

    if jac.ndim > 1:
        jac = jac.flatten()

    if len(jac) != len(bnorm):
        if jac.size == bnorm.size:
            jac = jac.reshape(bnorm.shape)
        else:
            raise ValueError(
                "Jacobian shape mismatch for coordinate system "
                f"'{coordinate_system}': got {jac.shape}, expected {bnorm.shape}"
            )

    if not np.all(np.isfinite(jac)):
        raise ValueError(
            f"Jacobian contains non-finite values for coordinate system '{coordinate_system}'"
        )

    if coordinate_system.lower() == "boozer":
        residual = boozer_consistency_residual(jac_context, jac)
        h_ref = abs(jac_context["I"] + jac_context["q"] * jac_context["F"])
        tol = 1.0e-8 * max(1.0, h_ref)
        if residual > tol:
            raise ValueError(
                f"Boozer Jacobian consistency check failed: residual={residual:.3e}"
            )

    jac_safe = jac.copy()
    small = np.abs(jac_safe) < 1.0e-14
    jac_safe[small] = np.where(jac_safe[small] < 0.0, -1.0e-14, 1.0e-14)

    jacobian_row = np.interp(thgeogrid, theta_eval, jac_safe, period=2.0 * np.pi)

    btheta = np.append(0.0, np.cumsum(ds / (jac_safe * bpol_safe)))
    btheta *= 2.0 * np.pi / btheta[-1]
    thtable_row = np.interp(thgeogrid, thetageom, btheta, period=2.0 * np.pi)

    nu = (
        -Fprof * np.append(0.0, np.cumsum(ds / (R**2 * bpol_safe)))
        + qprof * btheta
    )
    nutable_row = np.interp(thgeogrid, thetageom, nu, period=2.0 * np.pi)

    theta_geom_tmp = np.interp(thmaggrid, thtable_row, thgeogrid, period=2.0 * np.pi)
    Rtransform_row = np.interp(theta_geom_tmp, theta_eval, R, period=2.0 * np.pi)
    ztransform_row = np.interp(theta_geom_tmp, theta_eval, z, period=2.0 * np.pi)

    return (
        qprof,
        Fprof,
        Iprof,
        thtable_row,
        nutable_row,
        jacobian_row,
        Rtransform_row,
        ztransform_row,
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute magnetic coordinates using a generic Jacobian function.

    Parameters
    ----------
    Rgrid : np.ndarray
        Radial grid where (br, bz, bphi) are defined.
    zgrid : np.ndarray
        Vertical grid where (br, bz, bphi) are defined.
    br : np.ndarray
        Radial component of the magnetic field.
    bz : np.ndarray
        Vertical component of the magnetic field.
    bphi : np.ndarray
        Toroidal component of the magnetic field.
    raxis : float
        Radial position of the magnetic axis.
    zaxis : float
        Vertical position of the magnetic axis.
    psigrid : np.ndarray
        Poloidal flux grid where coordinates are defined.
    ltheta : int, optional
        Number of points in the poloidal direction. Default is 256.
    phiclockwise : bool, optional
        Whether toroidal angle increases clockwise. Default is True.
    jacobian_func : Callable, optional
        Function to compute Jacobian: ``jacobian_func(context) -> J``. If
        ``None``, uses Boozer Jacobian.
    R_at_psi : np.ndarray, optional
        Radial positions corresponding to psigrid at midplane. If None, will
        be computed from psigrid.
    coordinate_system : str, optional
        Name of coordinate system, used for Jacobian context construction.
        Default is ``'boozer'``.
    rho_at_psi : np.ndarray, optional
        Normalized radial labels associated with ``psigrid``. Used internally
        for spectral surface reconstruction.
    spectral_max_mode : int, optional
        Maximum retained poloidal Fourier mode in the spectral surface
        reconstruction. Default is 16.
    n_theta_geom : int, optional
        Number of points used to discretize the geometric poloidal angle when
        tracing each flux surface. The default ``None`` selects the
        module-level ``_THETA_GEOM_POINTS`` (currently 7200) which is dense
        enough for production runs at any reasonable ``ltheta``. Lower it
        when sweeping many surfaces in a diagnostic; raise it when a
        particular shape requires extra angular resolution. Must be at least
        ``max(4 * ltheta, 64)`` to keep the spectral surface reconstruction
        well-resolved.

    Returns
    -------
    qprof : np.ndarray
        Safety factor profile.
    Fprof : np.ndarray
        ``F(psi) = R*B_phi`` profile.
    Iprof : np.ndarray
        Toroidal current profile.
    thtable : np.ndarray
        Magnetic poloidal angle table (psi x theta).
    nutable : np.ndarray
        Magnetic toroidal angle table (psi x theta).
    jacobian : np.ndarray
        Jacobian table (psi x theta).
    Rtransform : np.ndarray
        Inverse transformation ``R(psi, theta)``.
    ztransform : np.ndarray
        Inverse transformation ``z(psi, theta)``.
    """
    if jacobian_func is None:
        jacobian_func = compute_boozer_jacobian
    spectral_max_mode = _validate_spectral_max_mode(spectral_max_mode)

    if n_theta_geom is None:
        n_theta_geom = _THETA_GEOM_POINTS
    else:
        n_theta_geom = int(n_theta_geom)
        min_required = max(4 * int(ltheta), 64)
        if n_theta_geom < min_required:
            raise ValueError(
                f"n_theta_geom must be at least max(4*ltheta, 64) = {min_required}, "
                f"got {n_theta_geom}."
            )

    # Generate theta grids
    thetageom = np.linspace(0, 2.0 * np.pi, n_theta_geom)
    theta_eval = thetageom[:-1]
    thgeogrid = np.linspace(0, 2.0 * np.pi, ltheta)
    thmaggrid = np.linspace(0, 2.0 * np.pi, ltheta)

    # Define output arrays
    npsi = len(psigrid)
    qprof = np.zeros(npsi)
    Fprof = np.zeros(npsi)
    Iprof = np.zeros(npsi)

    # Storing the magnetic coordinates
    thtable = np.zeros((npsi, ltheta))
    nutable = np.zeros((npsi, ltheta))
    jacobian = np.zeros((npsi, ltheta))
    Rtransform = np.zeros((npsi, ltheta))
    ztransform = np.zeros((npsi, ltheta))

    # Find appropriate direction of integration
    bzsep = RegularGridInterpolator((Rgrid, zgrid), bz, bounds_error=False,
                                    fill_value=None)((raxis, zaxis))
    if phiclockwise:
        integration_sign = np.sign(bzsep)
    else:
        integration_sign = -1 * np.sign(bzsep)

    # Convert psigrid to radial positions at midplane
    if R_at_psi is None:
        # Default: linear spacing (should be provided by caller)
        R_at_psi = np.linspace(raxis, Rgrid.max(), npsi)
    
    # Ensure R_at_psi matches psigrid length
    if len(R_at_psi) != npsi:
        R_at_psi = np.linspace(raxis, Rgrid.max(), npsi)
    rho_labels = _normalize_rho_labels(rho_at_psi, npsi)
    (
        R_raw,
        z_raw,
        br_raw,
        bz_raw,
        bphi_raw,
    ) = _trace_flux_surfaces(
        Rgrid=Rgrid,
        zgrid=zgrid,
        br=br,
        bz=bz,
        bphi=bphi,
        R_at_psi=np.asarray(R_at_psi, dtype=np.float64),
        zaxis=zaxis,
        raxis=raxis,
        thetageom=thetageom,
        integration_sign=integration_sign,
    )

    surface_mode = _SURFACE_RECONSTRUCTION_MODE.strip().lower()
    if surface_mode not in {"raw", "spectral"}:
        raise ValueError(
            "Invalid internal surface reconstruction mode "
            f"'{_SURFACE_RECONSTRUCTION_MODE}'."
        )

    R_surfaces = R_raw
    z_surfaces = z_raw
    br_surfaces = br_raw
    bz_surfaces = bz_raw
    bphi_surfaces = bphi_raw

    if surface_mode == "spectral":
        try:
            R_surfaces, z_surfaces = _build_spectral_surfaces(
                R_raw=R_raw,
                z_raw=z_raw,
                rho_at_psi=rho_labels,
                spectral_max_mode=spectral_max_mode,
            )
            br_surfaces = None
            bz_surfaces = None
            bphi_surfaces = None
        except ValueError:
            R_surfaces = R_raw
            z_surfaces = z_raw
            br_surfaces = br_raw
            bz_surfaces = bz_raw
            bphi_surfaces = bphi_raw

    br_interp = RegularGridInterpolator((Rgrid, zgrid), br, bounds_error=False, fill_value=None)
    bz_interp = RegularGridInterpolator((Rgrid, zgrid), bz, bounds_error=False, fill_value=None)
    bphi_interp = RegularGridInterpolator((Rgrid, zgrid), bphi, bounds_error=False, fill_value=None)

    for ii in range(npsi):
        row = _compute_surface_coordinate_row(
            R=R_surfaces[ii, :],
            z=z_surfaces[ii, :],
            br_surface=None if br_surfaces is None else br_surfaces[ii, :],
            bz_surface=None if bz_surfaces is None else bz_surfaces[ii, :],
            bphi_surface=None if bphi_surfaces is None else bphi_surfaces[ii, :],
            thetageom=thetageom,
            theta_eval=theta_eval,
            thgeogrid=thgeogrid,
            thmaggrid=thmaggrid,
            coordinate_system=coordinate_system,
            jacobian_func=jacobian_func,
            br_interp=br_interp,
            bz_interp=bz_interp,
            bphi_interp=bphi_interp,
        )

        (
            qprof[ii],
            Fprof[ii],
            Iprof[ii],
            thtable[ii, :],
            nutable[ii, :],
            jacobian[ii, :],
            Rtransform[ii, :],
            ztransform[ii, :],
        ) = row

    if npsi > 1:
        qprof[0] = qprof[1]
        Fprof[0] = Fprof[1]
        Iprof[0] = Iprof[1]
        jacobian[0, :] = jacobian[1, :]
        thtable[0, :] = thtable[1, :]
        nutable[0, :] = nutable[1, :]
        Rtransform[0, :] = Rtransform[1, :]
        ztransform[0, :] = ztransform[1, :]
        
    return qprof, Fprof, Iprof, thtable, nutable, jacobian, Rtransform, ztransform



def compute_magnetic_coordinates2(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute magnetic coordinates using a generic Jacobian function.

    Parameters
    ----------
    Rgrid : np.ndarray
        Radial grid where (br, bz, bphi) are defined
    zgrid : np.ndarray
        Vertical grid where (br, bz, bphi) are defined
    br : np.ndarray
        Radial component of the magnetic field
    bz : np.ndarray
        Vertical component of the magnetic field
    bphi : np.ndarray
        Toroidal component of the magnetic field
    raxis : float
        Radial position of the magnetic axis
    zaxis : float
        Vertical position of the magnetic axis
    psigrid : np.ndarray
        Poloidal flux grid where coordinates are defined
    ltheta : int, optional
        Number of points in the poloidal direction. Default is 256
    phiclockwise : bool, optional
        Whether toroidal angle increases clockwise. Default is True
    jacobian_func : Callable, optional
        Function to compute Jacobian: jacobian_func(context) -> J
        If None, uses Boozer Jacobian
    R_at_psi : np.ndarray, optional
        Radial positions corresponding to psigrid at midplane.
        If None, will be computed from psigrid
    coordinate_system : str, optional
        Name of coordinate system, used for Jacobian context construction.
        Default is 'boozer'
    Returns
    -------
    qprof : np.ndarray
        Safety factor profile
    Fprof : np.ndarray
        F(psi) = R*B_phi profile
    Iprof : np.ndarray
        Toroidal current profile
    thtable : np.ndarray
        Magnetic poloidal angle table (psi x theta)
    nutable : np.ndarray
        Magnetic toroidal angle table (psi x theta)
    jacobian : np.ndarray
        Jacobian table (psi x theta)
    Rtransform : np.ndarray
        Inverse transformation R(psi, theta)
    ztransform : np.ndarray
        Inverse transformation z(psi, theta)
    """
    if jacobian_func is None:
        jacobian_func = compute_boozer_jacobian

    # Generate theta grids
    thetageom = np.linspace(0, 2*np.pi, 7200)
    thgeogrid = np.linspace(0, 2*np.pi, ltheta)
    thmaggrid = np.linspace(0, 2*np.pi, ltheta)

    # Define output arrays
    npsi = len(psigrid)
    qprof = np.zeros(npsi)
    Fprof = np.zeros(npsi)
    Iprof = np.zeros(npsi)

    # Storing the magnetic coordinates
    thtable = np.zeros((npsi, ltheta))
    nutable = np.zeros((npsi, ltheta))
    jacobian = np.zeros((npsi, ltheta))
    Rtransform = np.zeros((npsi, ltheta))
    ztransform = np.zeros((npsi, ltheta))

    # Find appropriate direction of integration
    bzsep = RegularGridInterpolator((Rgrid, zgrid), bz, bounds_error=False,
                                    fill_value=None)((raxis, zaxis))
    if phiclockwise:
        integration_sign = np.sign(bzsep)
    else:
        integration_sign = -1 * np.sign(bzsep)

    # Convert psigrid to radial positions at midplane
    if R_at_psi is None:
        # Default: linear spacing (should be provided by caller)
        R_at_psi = np.linspace(raxis, Rgrid.max(), npsi)
    
    # Ensure R_at_psi matches psigrid length
    if len(R_at_psi) != npsi:
        R_at_psi = np.linspace(raxis, Rgrid.max(), npsi)

    for ii in range(npsi):
        ir = R_at_psi[ii]
        
        # Get the flux surface by integrating field line
        Rline, zline, brline, bzline, bphiline, iend = \
            integrate_pol_field_line(Rgrid, zgrid, br, bz, bphi,
                                     ir, zaxis, integration_sign=integration_sign)
        
        Rline = Rline[:iend]
        zline = zline[:iend]
        brline = brline[:iend]
        bzline = bzline[:iend]
        bphiline = bphiline[:iend]
        
        # Evaluate flux surface over theta grid
        thetaval = np.fmod(np.arctan2(zline - zaxis, Rline - raxis), 2*np.pi)
        R_full = np.interp(thetageom, thetaval, Rline, period=2*np.pi)
        z_full = np.interp(thetageom, thetaval, zline, period=2*np.pi)
        br_interp = np.interp(thetageom, thetaval, brline, period=2*np.pi)[:-1]
        bz_interp = np.interp(thetageom, thetaval, bzline, period=2*np.pi)[:-1]
        bphi_interp = np.interp(thetageom, thetaval, bphiline, period=2*np.pi)[:-1]

        dR = np.diff(R_full)
        dZ = np.diff(z_full)
        R = R_full[:-1]
        z = z_full[:-1]
        dlp = np.sqrt(dR**2 + dZ**2)

        bnorm = np.sqrt(br_interp**2 + bz_interp**2 + bphi_interp**2)
        bpol = np.sqrt(br_interp**2 + bz_interp**2)
        bpol_safe = np.where(bpol > 1.0e-14, bpol, 1.0e-14)
        ds = (dR * br_interp + dZ * bz_interp) / bpol_safe
        dlbpol = dR * br_interp + dZ * bz_interp

        # Compute profiles.
        # Iprof follows the Boozer-Jacobian convention used below:
        #   J * B^2 = Iprof + q*F
        Iprof[ii] = np.sum(dlbpol) / (2*np.pi)
        Fprof[ii] = R[0] * bphi_interp[0]
        qprof[ii] = np.sum(ds * Fprof[ii] / (R**2 * bpol_safe)) / (2*np.pi)

        jac_context = make_jacobian_context(
            coordinate_system=coordinate_system,
            R=R,
            B=bnorm,
            Bpol=bpol_safe,
            dlp=dlp,
            I=Iprof[ii],
            F=Fprof[ii],
            q=qprof[ii],
        )
        jac = np.asarray(jacobian_func(jac_context), dtype=np.float64)

        if jac.ndim > 1:
            jac = jac.flatten()

        if len(jac) != len(bnorm):
            if jac.size == bnorm.size:
                jac = jac.reshape(bnorm.shape)
            else:
                raise ValueError(
                    "Jacobian shape mismatch for coordinate system "
                    f"'{coordinate_system}': got {jac.shape}, expected {bnorm.shape}"
                )

        if not np.all(np.isfinite(jac)):
            raise ValueError(
                f"Jacobian contains non-finite values for coordinate system '{coordinate_system}'"
            )

        if coordinate_system.lower() == "boozer":
            residual = boozer_consistency_residual(jac_context, jac)
            h_ref = abs(jac_context["I"] + jac_context["q"] * jac_context["F"])
            tol = 1.0e-8 * max(1.0, h_ref)
            if residual > tol:
                raise ValueError(
                    f"Boozer Jacobian consistency check failed: residual={residual:.3e}"
                )

        jac_safe = jac.copy()
        small = np.abs(jac_safe) < 1.0e-14
        jac_safe[small] = np.where(jac_safe[small] < 0.0, -1.0e-14, 1.0e-14)

        jacobian[ii, :] = np.interp(thgeogrid, thetageom[:-1], jac_safe, period=2*np.pi)
        
        # Compute magnetic poloidal angle
        btheta = np.append(0, np.cumsum(ds / (jac_safe * bpol_safe)))
        
        # Normalize to remove numerical error
        a = 2*np.pi / btheta[-1]
        btheta *= a
        thtable[ii, :] = np.interp(thgeogrid, thetageom, btheta, period=2*np.pi)

        # Compute magnetic toroidal coordinate
        nu = (-Fprof[ii] * np.append(0, np.cumsum(ds / (R**2 * bpol_safe))) +
              qprof[ii] * btheta)
        nutable[ii, :] = np.interp(thgeogrid, thetageom, nu, period=2*np.pi)
        
        # Handle edge cases
        if ii == 0:
            qprof[0] = qprof[1] if npsi > 1 else qprof[0]
            Fprof[0] = Fprof[1] if npsi > 1 else Fprof[0]
            Iprof[0] = Iprof[1] if npsi > 1 else Iprof[0]
            jacobian[0, :] = jacobian[1, :] if npsi > 1 else jacobian[0, :]
            thtable[0, :] = thtable[1, :] if npsi > 1 else thtable[0, :]
            nutable[0, :] = nutable[1, :] if npsi > 1 else nutable[0, :]

        # Build inverse transformation
        theta_geom_tmp = np.interp(thmaggrid, thtable[ii, :], thgeogrid, 
                                   period=2*np.pi)
        Rtransform[ii, :] = np.interp(theta_geom_tmp, thetageom[:-1], R, period=2*np.pi)
        ztransform[ii, :] = np.interp(theta_geom_tmp, thetageom[:-1], z, period=2*np.pi)
        
    return qprof, Fprof, Iprof, thtable, nutable, jacobian, Rtransform, ztransform
