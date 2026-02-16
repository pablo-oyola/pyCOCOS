"""Functions for checking and transforming equilibrium COCOS.

COCOS (COordinate COnventionS) is a standard for describing conventions
used in tokamak equilibrium codes.

Reference: O. Sauter et al, Comp. Phys. Comm., 184 (2013) 293-302
https://www.sciencedirect.com/science/article/pii/S0010465512002962
"""
import numpy as np
import copy
from typing import Any, Optional, Dict, Tuple

class COCOS:
    """
    Class to model COCOS (COordinate COnventionS) conventions.

    Parameters
    ----------
    cocos : int
        COCOS ID number (1-18)
    exp_Bp : int
        Exponent for 2π in poloidal flux (0 or 1)
    sigma_Bp : int
        Sign of psi gradient (+1 or -1)
    sigma_RpZ : int
        Handedness of (R, phi, Z) coordinate system (+1 or -1)
    sigma_rhotp : int
        Handedness of (rho, theta, phi) coordinate system (+1 or -1)
    sign_q_pos : int
        Sign of q with positive Ip and B0 (+1 or -1)
    sign_pprime_pos : int
        Sign of dp/dpsi with positive Ip and B0 (+1 or -1)

    Attributes
    ----------
    cocos : int
        COCOS ID number
    exp_Bp : int
        Exponent for 2π in poloidal flux
    sigma_Bp : int
        Sign of psi gradient
    sigma_RpZ : int
        Handedness of (R, phi, Z)
    sigma_rhotp : int
        Handedness of (rho, theta, phi)
    sign_q_pos : int
        Sign of q
    sign_pprime_pos : int
        Sign of dp/dpsi
    """
    def __init__(
        self,
        cocos: int,
        exp_Bp: int,
        sigma_Bp: int,
        sigma_RpZ: int,
        sigma_rhotp: int,
        sign_q_pos: int,
        sign_pprime_pos: int
    ) -> None:
        
        # # Checking all the inputs of this class are either +1 or -1 (
        # # except for cocos which is an integer)
        if not all([i in [-1, 1] for i in [sigma_Bp, sigma_RpZ, 
                                           sigma_rhotp, sign_q_pos, sign_pprime_pos]]):
            raise ValueError("All inputs must be either +1 or -1")

        self.cocos = cocos
        self.exp_Bp = exp_Bp
        self.sigma_Bp = sigma_Bp
        self.sigma_RpZ = sigma_RpZ
        self.sigma_rhotp = sigma_rhotp
        self.sign_q_pos = sign_q_pos
        self.sign_pprime_pos = sign_pprime_pos

def cocos(cocos_in: int) -> COCOS:
    """
    Create COCOS object for the given COCOS ID number.

    Parameters
    ----------
    cocos_in : int
        The COCOS identification number (1-18)

    Returns
    -------
    COCOS
        COCOS object with the specified convention

    Raises
    ------
    ValueError
        If cocos_in is outside the accepted range (1-18)

    Examples
    --------
    >>> cc = cocos(11)  # ITER/Boozer convention
    >>> cc = cocos(3)   # EFIT convention
    """
    exp_Bp = 1 if cocos_in >= 11 else 0

    if cocos_in in (1, 11):
        # ITER, Boozer are COCOS=11
        # Also used in TRANSP and ORBIT.
        return COCOS(cocos_in, exp_Bp, 1, 1, 1, 1, -1)
    elif cocos_in in (2, 12):
        # CHEASE, ONETWO, Hinton-Hazeltine, LION is COCOS=2
        return COCOS(cocos_in, exp_Bp, 1, -1, 1, 1, -1)
    elif cocos_in in (3, 13):
        # Freidberg, CAXE, KINX, EFIT are COCOS=3
        # EU-ITM up to end of 2011 is COCOS=13
        return COCOS(cocos_in, exp_Bp, -1, 1, -1, -1, 1)
    elif cocos_in in (4, 14):
        return COCOS(cocos_in, exp_Bp, -1, -1, -1, -1, 1)
    elif cocos_in in (5, 15):
        return COCOS(cocos_in, exp_Bp, 1, 1, -1, -1, -1)
    elif cocos_in in (6, 16):
        return COCOS(cocos_in, exp_Bp, 1, -1, -1, -1, -1)
    elif cocos_in in (7, 17):
        return COCOS(cocos_in, exp_Bp, -1, 1, 1, 1, 1)
    elif cocos_in in (8, 18):
        return COCOS(cocos_in, exp_Bp, -1, -1, 1, 1, 1)
    else:
        raise ValueError(f"COCOS = {cocos_in} does not exist")

def assign(
    q: float,
    ip: float,
    b0: float,
    psiaxis: float,
    psibndr: float,
    phiclockwise: bool = False,
    weberperrad: bool = True
) -> int:
    """
    Automatically determine COCOS convention from equilibrium parameters.

    Parameters
    ----------
    q : float
        Safety factor (at any point, sign included)
    ip : float
        Plasma current (sign included)
    b0 : float
        Toroidal field (sign included)
    psiaxis : float
        Poloidal flux at the magnetic axis
    psibndr : float
        Poloidal flux at the boundary
    phiclockwise : bool, optional
        If True, toroidal angle increases clockwise when viewed from above.
        Default is False
    weberperrad : bool, optional
        If True, poloidal flux is in Wb/rad (divided by 2π).
        True for COCOS ID 1-8, False for COCOS ID 11-18.
        Default is True

    Returns
    -------
    int
        The COCOS number (1-18) corresponding to this equilibrium

    Raises
    ------
    ValueError
        If correct COCOS could not be determined

    Examples
    --------
    >>> cocos_id = assign(q=2.0, ip=1e6, b0=2.5, psiaxis=0.0, psibndr=1.0)
    """
    sign_q  = np.sign(q)
    sign_ip = np.sign(ip)
    sign_b0 = np.sign(b0)
    cocos = set([1, 2, 3, 4, 5, 6, 7, 8])

    sigma_bp = np.sign(psibndr-psiaxis)/sign_ip
    if sigma_bp > 0:
        for i in [3,4,7,8]:
            cocos.discard(i)
    else:
        for i in [1,2,5,6]:
            cocos.discard(i)

    sigma_rhothetaphi = sign_q/(sign_ip*sign_b0)
    if sigma_rhothetaphi < 0:
        for i in [1,2,7,8]:
            cocos.discard(i)
    else:
        for i in [3,4,5,6]:
            cocos.discard(i)

    if phiclockwise:
        for i in [1,3,5,7]:
            cocos.discard(i)
    else:
        for i in [2,4,6,8]:
            cocos.discard(i)


    if len(cocos) > 1:
        raise ValueError("Could not determine COCOS")
    cocos = cocos.pop()
    if not weberperrad: # COCOS ID 11-18 are NOT divided by 2*pi.
        cocos += 10
    return cocos

def transform_cocos(
    cc_in: COCOS,
    cc_out: COCOS,
    sigma_Ip: Optional[Tuple[int, int]] = None,
    sigma_B0: Optional[Tuple[int, int]] = None,
    ld: Tuple[int, int] = (1, 1),
    lB: Tuple[int, int] = (1, 1),
    exp_mu0: Tuple[int, int] = (0, 0)
) -> Dict[str, float]:
    """
    Compute multiplicative factors to transform between COCOS conventions.

    These equations are based on O. Sauter et al, Comp. Phys. Comm., 184 (2013).

    Parameters
    ----------
    cc_in : COCOS
        Input COCOS convention
    cc_out : COCOS
        Output COCOS convention
    sigma_Ip : tuple of int, optional
        (Input, Output) current sign. If None, inferred from coordinate systems
    sigma_B0 : tuple of int, optional
        (Input, Output) toroidal field sign. If None, inferred from coordinate systems
    ld : tuple of int, optional
        (Input, Output) length scale factor. Default is (1, 1)
    lB : tuple of int, optional
        (Input, Output) magnetic field scale factor. Default is (1, 1)
    exp_mu0 : tuple of int, optional
        (Input, Output) μ₀ exponent (0 or 1). Default is (0, 0)

    Returns
    -------
    dict
        Dictionary of multiplicative factors for transforming quantities:
        - 'R', 'Z': length scales
        - 'PRES': pressure
        - 'PSI': poloidal flux
        - 'TOR': toroidal flux
        - 'PPRIME': pressure gradient
        - 'FFPRIME': F*F' term
        - 'B': magnetic field
        - 'F': F(psi) function
        - 'I': current
        - 'J': current density
        - 'Q': safety factor

    Examples
    --------
    >>> cc1 = cocos(1)
    >>> cc3 = cocos(3)
    >>> factors = transform_cocos(cc1, cc3)
    >>> psi_new = psi_old * factors['PSI']
    """

    ld_eff = ld[1] / ld[0]
    lB_eff = lB[1] / lB[0]
    exp_mu0_eff = exp_mu0[1] - exp_mu0[0]

    sigma_RpZ_eff = cc_in.sigma_RpZ * cc_out.sigma_RpZ

    if sigma_Ip is None:
        sigma_Ip_eff = cc_in.sigma_RpZ * cc_out.sigma_RpZ
    else:
        sigma_Ip_eff = sigma_Ip[0] * sigma_Ip[1]

    if sigma_B0 is None:
        sigma_B0_eff = cc_in.sigma_RpZ * cc_out.sigma_RpZ
    else:
        sigma_B0_eff = sigma_B0[0] * sigma_B0[1]

    sigma_Bp_eff = cc_in.sigma_Bp * cc_out.sigma_Bp
    exp_Bp_eff = cc_out.exp_Bp - cc_in.exp_Bp
    sigma_rhotp_eff = cc_in.sigma_rhotp * cc_out.sigma_rhotp

    mu0 = 4 * np.pi * 1e-7

    transforms = {}
    transforms["R"] = ld_eff
    transforms["Z"] = ld_eff
    transforms["PRES"] = (lB_eff ** 2) / (mu0 ** exp_mu0_eff)
    transforms["PSI"] = lB_eff * (ld_eff ** 2) * sigma_Ip_eff * sigma_Bp_eff \
        * ((2 * np.pi) ** exp_Bp_eff) * (ld_eff ** 2) * lB_eff
    transforms["TOR"] = lB_eff * (ld_eff ** 2) * sigma_B0_eff
    transforms["PPRIME"] = (lB_eff / ((ld_eff ** 2) * (mu0 ** exp_mu0_eff))) \
        * sigma_Ip_eff * sigma_Bp_eff / ((2 * np.pi) ** exp_Bp_eff)
    transforms["FFPRIME"] = lB_eff * sigma_Ip_eff * sigma_Bp_eff \
        / ((2 * np.pi) ** exp_Bp_eff)
    transforms["B"] = lB_eff * sigma_B0_eff
    transforms["F"] = sigma_B0_eff * ld_eff * lB_eff
    transforms["I"] = sigma_Ip_eff * ld_eff * lB_eff / (mu0 ** exp_mu0_eff)
    transforms["J"] = sigma_Ip_eff * lB_eff / ((mu0 ** exp_mu0_eff) * ld_eff)
    transforms["Q"] = sigma_Ip_eff * sigma_B0_eff * sigma_rhotp_eff

    return transforms

def fromCocosNtoCocosM(
    eqd: Dict[str, Any],
    cocos_m: int,
    cocos_n: Optional[int] = None,
    phiclockwise: Optional[bool] = None,
    weberperrad: bool = True
) -> Dict[str, Any]:
    """
    Transform equilibrium dictionary from one COCOS to another.

    Parameters
    ----------
    eqd : dict
        Dictionary from reading the EQDSK file
    cocos_m : int
        Target COCOS convention (1-18)
    cocos_n : int, optional
        Input COCOS convention. If None, will be auto-detected
    phiclockwise : bool, optional
        Whether toroidal angle increases clockwise. Used for auto-detection
    weberperrad : bool, optional
        Whether flux is in Wb/rad. Used for auto-detection. Default is True

    Returns
    -------
    dict
        Equilibrium data converted to cocos_m

    Examples
    --------
    >>> eqd_cocos1 = fromCocosNtoCocosM(eqd_data, cocos_m=1, cocos_n=3)
    """
    if not cocos_n: # If None, determine from G-EQDSK data
        cocos_n = assign(eqd["qpsi"][0], eqd["cpasma"], eqd["bcentr"],
                         eqd["simagx"], eqd["sibdry"], phiclockwise,
                         weberperrad)

    transform_dict = transform_cocos(cocos(cocos_n), cocos(cocos_m))

    # Define output
    eqdout = copy.deepcopy(eqd)
    # eqdout["nx"]    = eqd["nx"]    # For clarity (this is not altered)
    # eqdout["ny"]    = eqd["ny"]    # -||-
    # eqdout["nbdry"] = eqd["nbdry"] # -||-
    # eqdout["nlim"]  = eqd["nlim"]  # -||-
    eqdout["rdim"]    = eqd["rdim"]    * transform_dict["R"]
    eqdout["zdim"]    = eqd["zdim"]    * transform_dict["Z"]
    eqdout["rcentr"]  = eqd["rcentr"]  * transform_dict["R"]
    eqdout["rleft"]   = eqd["rleft"]   * transform_dict["R"]
    eqdout["zmid"]    = eqd["zmid"]    * transform_dict["Z"]
    eqdout["rmagx"]   = eqd["rmagx"]   * transform_dict["R"]
    eqdout["zmagx"]   = eqd["zmagx"]   * transform_dict["Z"]
    eqdout["simagx"]  = eqd["simagx"]  * transform_dict["PSI"]
    eqdout["sibdry"]  = eqd["sibdry"]  * transform_dict["PSI"]
    eqdout["bcentr"]  = eqd["bcentr"]  * transform_dict["B"]
    eqdout["cpasma"]  = eqd["cpasma"]  * transform_dict["I"]
    eqdout["fpol"]    = eqd["fpol"]    * transform_dict["F"]
    eqdout["pres"]    = eqd["pres"]    * transform_dict["PRES"]
    eqdout["ffprime"] = eqd["ffprime"] * transform_dict["FFPRIME"]
    eqdout["pprime"]  = eqd["pprime"]  * transform_dict["PPRIME"]
    eqdout["psi"]     = eqd["psi"]     * transform_dict["PSI"]
    eqdout["qpsi"]    = eqd["qpsi"]    * transform_dict["Q"]
    eqdout["rbdry"]   = eqd["rbdry"]   * transform_dict["R"]
    eqdout["zbdry"]   = eqd["zbdry"]   * transform_dict["Z"]
    eqdout["rlim"]    = eqd["rlim"]    * transform_dict["R"]
    eqdout["zlim"]    = eqd["zlim"]    * transform_dict["Z"]

    return eqdout
