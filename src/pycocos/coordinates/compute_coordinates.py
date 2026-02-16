"""
Generic coordinate computation using Jacobian-based architecture.

This module provides a generic function to compute magnetic coordinates
for any coordinate system by using the appropriate Jacobian function.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy.interpolate import RegularGridInterpolator
from .field_lines import integrate_pol_field_line
from .jacobians import compute_boozer_jacobian
from .jacobian_builders import boozer_consistency_residual, make_jacobian_context


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

        # Compute profiles
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
