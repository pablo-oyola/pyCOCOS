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
    R_at_psi: Optional[np.ndarray] = None
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
        Function to compute Jacobian: jacobian_func(I, F, q, B) -> J
        If None, uses Boozer Jacobian
    R_at_psi : np.ndarray, optional
        Radial positions corresponding to psigrid at midplane.
        If None, will be computed from psigrid

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
        R = np.interp(thetageom, thetaval, Rline, period=2*np.pi)
        z = np.interp(thetageom, thetaval, zline, period=2*np.pi)
        br_interp = np.interp(thetageom, thetaval, brline, period=2*np.pi)[:-1]
        bz_interp = np.interp(thetageom, thetaval, bzline, period=2*np.pi)[:-1]
        bphi_interp = np.interp(thetageom, thetaval, bphiline, period=2*np.pi)[:-1]

        bnorm = np.sqrt(br_interp**2 + bz_interp**2 + bphi_interp**2)
        bpol = np.sqrt(br_interp**2 + bz_interp**2)
        ds = (np.diff(R) * br_interp + np.diff(z) * bz_interp) / bpol
        dlbpol = (np.diff(R) * br_interp + np.diff(z) * bz_interp)
        R = R[:-1]
        z = z[:-1]

        # Compute profiles
        Iprof[ii] = np.sum(dlbpol) / (2*np.pi)
        Fprof[ii] = R[0] * bphi_interp[0]
        qprof[ii] = np.sum(ds * Fprof[ii] / (R**2 * bpol)) / (2*np.pi)

        # Compute Jacobian using the provided function
        # For each flux surface, we have scalar profiles (I, F, q)
        # and a 1D B array along theta. The Jacobian function should
        # handle this by broadcasting the scalars.
        I_scalar = np.array([Iprof[ii]])
        F_scalar = np.array([Fprof[ii]])
        q_scalar = np.array([qprof[ii]])
        B_1d = bnorm  # Already 1D along theta
        
        # Call Jacobian function - it should broadcast scalars to match B
        jac = jacobian_func(I_scalar, F_scalar, q_scalar, B_1d)
        
        # Jacobian should be 1D matching bnorm length
        if jac.ndim > 1:
            jac = jac.flatten()
        # Ensure correct length
        if len(jac) != len(bnorm):
            # If shape mismatch, try to fix it
            if jac.size == bnorm.size:
                jac = jac.reshape(bnorm.shape)
            else:
                # Fallback: use scalar value
                jac_val = jac[0] if jac.size > 0 else 1.0
                jac = np.full_like(bnorm, jac_val)
        
        jacobian[ii, :] = np.interp(thgeogrid, thetageom[:-1], jac, period=2*np.pi)
        
        # Compute magnetic poloidal angle
        btheta = np.append(0, np.cumsum(ds / (jac * bpol)))
        
        # Normalize to remove numerical error
        a = 2*np.pi / btheta[-1]
        btheta *= a
        thtable[ii, :] = np.interp(thgeogrid, thetageom, btheta, period=2*np.pi)

        # Compute magnetic toroidal coordinate
        nu = (-Fprof[ii] * np.append(0, np.cumsum(ds / (R**2 * bpol))) +
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
