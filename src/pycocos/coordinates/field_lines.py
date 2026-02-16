"""
Field line integration functions for magnetic coordinate computation.

These functions integrate field lines to compute flux surfaces and profiles
needed for coordinate transformations.
"""

import numpy as np
from numba import njit
from typing import Tuple
from ..utils.numba_utils import interp2d_fast
from scipy.interpolate import RegularGridInterpolator


@njit(nogil=True, cache=True)
def get_field_line(
    grr: np.ndarray,
    gzz: np.ndarray,
    br: np.ndarray,
    bz: np.ndarray,
    bphi: np.ndarray,
    R: float,
    zaxis: float,
    tol: float = 1.0e-3,
    Nmax: int = 100000,
    integration_sign: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute the field line projected on the poloidal plane.

    Parameters
    ----------
    grr : np.ndarray
        Radial grid where (br, bz) are defined
    gzz : np.ndarray
        Vertical grid where (br, bz) are defined
    br : np.ndarray
        Radial component of the magnetic field
    bz : np.ndarray
        Vertical component of the magnetic field
    bphi : np.ndarray
        Toroidal component of the magnetic field
    R : float
        Initial radial position of the field line
    zaxis : float
        Vertical position of the magnetic axis
    tol : float, optional
        Tolerance for the integration. Default is 1.0e-3
    Nmax : int, optional
        Maximum number of iterations. Default is 100000
    integration_sign : int, optional
        Sign for integration direction. Default is 1

    Returns
    -------
    Routput : np.ndarray
        Radial positions along field line
    zoutput : np.ndarray
        Vertical positions along field line
    phiout : np.ndarray
        Toroidal angles along field line
    brout : np.ndarray
        Radial magnetic field along field line
    bzout : np.ndarray
        Vertical magnetic field along field line
    bphiout : np.ndarray
        Toroidal magnetic field along field line
    iend : int
        Number of points computed
    """
    # Packing up the input magnetic field
    Bpacked = (br, bz, bphi)

    # Getting the grid properties for the interpolation
    rmin = grr.min()
    zmin = gzz.min()
    dr = grr[1] - grr[0]
    dz = gzz[1] - gzz[0]
    nR = len(grr)
    nz = len(gzz)

    # Defining the RK4 coefficients
    c0 = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
    c1 = np.array([0.50, 0.50, 1.0, 0.0])
    ds = tol

    # Defining the temporal values
    Rnow = R
    znow = zaxis
    phinow = 0.0

    # Storage for the output
    Routput = np.zeros(Nmax)
    zoutput = np.zeros(Nmax)
    phiout = np.zeros(Nmax)
    brout = np.zeros(Nmax)
    bzout = np.zeros(Nmax)
    bphiout = np.zeros(Nmax)

    done = False
    for ii in range(Nmax):
        r1 = Rnow
        r2 = Rnow
        z1 = znow
        z2 = znow
        phi1 = phinow
        phi2 = phinow
        
        # The RK4 loop
        for irk in range(4):
            binterp = interp2d_fast(rmin, zmin, dr, dz,
                                    Bpacked, Rnow, znow,
                                    nR, nz)

            bnorm = np.sqrt(binterp[0]**2 + binterp[1]**2 + binterp[2]**2)
            dRval = binterp[0] / bnorm * ds
            dzval = binterp[1] / bnorm * ds
            dphival = binterp[2] / bnorm * ds / Rnow

            if irk != 3:
                r1 += c0[irk] * dRval
                z1 += c0[irk] * dzval
                phi1 += c0[irk] * dphival

                Rnow = r2 + c1[irk] * dRval
                znow = z2 + c1[irk] * dzval
                phinow = phi2 + c1[irk] * dphival
            else:
                Rnow = r1 + c0[irk] * dRval
                znow = z1 + c0[irk] * dzval
                phinow = phi1 + c0[irk] * dphival

            # Checking whether we have crossed the axis
            if ((znow - zaxis) * (z2 - zaxis) < 0.0) and (znow < zaxis):
                # Using linear interpolation, we get the R value at the axis
                Rnow = (zaxis - z2) * (Rnow - r2) / (znow - z2) + r2
                znow = zaxis

                binterp = interp2d_fast(rmin, zmin, dr, dz,
                                        Bpacked, Rnow, znow,
                                        nR, nz)
                Routput[ii] = Rnow
                zoutput[ii] = znow
                phiout[ii] = phinow
                brout[ii] = binterp[0]
                bzout[ii] = binterp[1]
                bphiout[ii] = binterp[2]
                done = True
                break

            Routput[ii] = Rnow
            zoutput[ii] = znow
            phiout[ii] = phinow
            brout[ii] = binterp[0]
            bzout[ii] = binterp[1]
            bphiout[ii] = binterp[2]

        if done:
            break
    
    if not done:
        print("WARNING: The field line did not reach the axis.")

    return Routput, zoutput, phiout, brout, bzout, bphiout, ii


@njit(nogil=True, cache=True)
def integrate_pol_field_line(
    grr: np.ndarray,
    gzz: np.ndarray,
    br: np.ndarray,
    bz: np.ndarray,
    bphi: np.ndarray,
    R: float,
    zaxis: float,
    tol: float = 1.0e-3,
    Nmax: int = 100000,
    integration_sign: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute the field line projected on the poloidal plane (poloidal only).

    Parameters
    ----------
    grr : np.ndarray
        Radial grid where (br, bz) are defined
    gzz : np.ndarray
        Vertical grid where (br, bz) are defined
    br : np.ndarray
        Radial component of the magnetic field
    bz : np.ndarray
        Vertical component of the magnetic field
    bphi : np.ndarray
        Toroidal component of the magnetic field
    R : float
        Initial radial position of the field line
    zaxis : float
        Vertical position of the magnetic axis
    tol : float, optional
        Tolerance for the integration. Default is 1.0e-3
    Nmax : int, optional
        Maximum number of iterations. Default is 100000
    integration_sign : int, optional
        Sign for integration direction. Default is 1

    Returns
    -------
    Routput : np.ndarray
        Radial positions along field line
    zoutput : np.ndarray
        Vertical positions along field line
    brout : np.ndarray
        Radial magnetic field along field line
    bzout : np.ndarray
        Vertical magnetic field along field line
    bphiout : np.ndarray
        Toroidal magnetic field along field line
    iend : int
        Number of points computed
    """
    # Packing up the input magnetic field
    Bpacked = (br, bz, bphi)

    # Getting the grid properties for the interpolation
    rmin = grr.min()
    zmin = gzz.min()
    dr = grr[1] - grr[0]
    dz = gzz[1] - gzz[0]
    nR = len(grr)
    nz = len(gzz)

    # Defining the RK4 coefficients
    c0 = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
    c1 = np.array([0.50, 0.50, 1.0, 0.0])
    ds = tol * integration_sign
    
    # Defining the temporal values
    Rnow = R
    znow = zaxis

    # Storage for the output
    Routput = np.zeros(Nmax)
    zoutput = np.zeros(Nmax)
    brout = np.zeros(Nmax)
    bzout = np.zeros(Nmax)
    bphiout = np.zeros(Nmax)

    done = False
    for ii in range(Nmax):
        r1 = Rnow
        r2 = Rnow
        z1 = znow
        z2 = znow
        
        # The RK4 loop
        for irk in range(4):
            binterp = interp2d_fast(rmin, zmin, dr, dz,
                                    Bpacked, Rnow, znow,
                                    nR, nz)

            bpol = np.sqrt(binterp[0]**2 + binterp[1]**2)
            dRval = binterp[0] / bpol * ds
            dzval = binterp[1] / bpol * ds

            if irk != 3:
                r1 += c0[irk] * dRval
                z1 += c0[irk] * dzval

                Rnow = r2 + c1[irk] * dRval
                znow = z2 + c1[irk] * dzval
            else:
                Rnow = r1 + c0[irk] * dRval
                znow = z1 + c0[irk] * dzval

            # Checking whether we have crossed the axis
            if ((znow - zaxis) * (z2 - zaxis) < 0.0) and (znow < zaxis):
                # Using linear interpolation, we get the R value at the axis
                Rnow = (zaxis - z2) * (Rnow - r2) / (znow - z2) + r2
                znow = zaxis

                binterp = interp2d_fast(rmin, zmin, dr, dz,
                                        Bpacked, Rnow, znow,
                                        nR, nz)
                Routput[ii] = Rnow
                zoutput[ii] = znow
                brout[ii] = binterp[0]
                bzout[ii] = binterp[1]
                bphiout[ii] = binterp[2]
                done = True
                break

            Routput[ii] = Rnow
            zoutput[ii] = znow
            brout[ii] = binterp[0]
            bzout[ii] = binterp[1]
            bphiout[ii] = binterp[2]

        if done:
            break
    
    if not done:
        print("WARNING: The field line did not reach the axis.")

    return Routput, zoutput, brout, bzout, bphiout, ii
