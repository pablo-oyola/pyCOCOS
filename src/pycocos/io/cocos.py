"""Functions for checking and transforming equilibrium COCOS.

COCOS (COordinate COnventionS) is a standard for describing conventions
used in tokamak equilibrium codes.

Reference: O. Sauter et al, Comp. Phys. Comm., 184 (2013) 293-302
https://www.sciencedirect.com/science/article/pii/S0010465512002962
"""
import copy
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np


FluxNormalization = Literal["Wb", "Wb/rad"]
_VALID_COCOS_IDS = tuple(range(1, 9)) + tuple(range(11, 19))


@dataclass(frozen=True)
class COCOSResolution:
    """Candidate COCOS conventions consistent with the supplied information.

    A plain axisymmetric EQDSK does not encode every fact needed to identify a
    unique COCOS convention.  In particular, callers may omit toroidal-angle
    orientation and poloidal-flux normalization here without forcing an
    assumption.  :attr:`candidates` then retains every consistent convention.
    """

    candidates: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("At least one COCOS candidate is required")
        if self.candidates != tuple(sorted(set(self.candidates))):
            raise ValueError("COCOS candidates must be sorted and unique")
        invalid = tuple(
            candidate
            for candidate in self.candidates
            if candidate not in _VALID_COCOS_IDS
        )
        if invalid:
            raise ValueError(f"Invalid COCOS candidates: {invalid}")

    @property
    def cocos(self) -> Optional[int]:
        """Return the unique COCOS ID, or ``None`` when still ambiguous."""
        return self.candidates[0] if self.is_unique else None

    @property
    def is_unique(self) -> bool:
        """Whether the available information selects one COCOS convention."""
        return len(self.candidates) == 1

    def require_unique(self) -> int:
        """Return the unique COCOS ID or raise with the remaining candidates."""
        if self.cocos is None:
            raise ValueError(
                f"COCOS is ambiguous; candidates are {self.candidates}"
            )
        return self.cocos

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
        signs = (
            sigma_Bp,
            sigma_RpZ,
            sigma_rhotp,
            sign_q_pos,
            sign_pprime_pos,
        )
        if not all(sign in (-1, 1) for sign in signs):
            raise ValueError("All inputs must be either +1 or -1")

        self.cocos = cocos
        self.exp_Bp = exp_Bp
        self.sigma_Bp = sigma_Bp
        self.sigma_RpZ = sigma_RpZ
        self.sigma_rhotp = sigma_rhotp
        self.sign_q_pos = sign_q_pos
        self.sign_pprime_pos = sign_pprime_pos

    @property
    def phiclockwise(self) -> bool:
        """Whether the COCOS toroidal angle increases clockwise from +Z."""
        return self.sigma_RpZ < 0

    @property
    def flux_normalization(self) -> FluxNormalization:
        """Poloidal-flux normalization encoded by this COCOS convention."""
        return "Wb" if self.exp_Bp else "Wb/rad"


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
    if isinstance(cocos_in, (bool, np.bool_)) or not isinstance(
        cocos_in,
        (int, np.integer),
    ):
        raise TypeError("cocos_in must be an integer COCOS ID")
    cocos_in = int(cocos_in)
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


def _finite_nonzero_sign(value: float, name: str) -> int:
    """Return the sign of a finite, nonzero scalar convention quantity."""
    array = np.asarray(value)
    if array.ndim != 0:
        raise TypeError(f"{name} must be a scalar")
    try:
        scalar = float(array)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(scalar) or scalar == 0.0:
        raise ValueError(f"{name} must be finite and nonzero")
    return 1 if scalar > 0.0 else -1


def identify_cocos(
    q: float,
    ip: float,
    b0: float,
    psiaxis: float,
    psibndr: float,
    phiclockwise: Optional[bool] = None,
    flux_normalization: Optional[FluxNormalization] = None,
) -> COCOSResolution:
    """Identify every COCOS convention consistent with the supplied facts.

    If toroidal-angle orientation or flux normalization is omitted, the
    returned resolution remains ambiguous rather than silently guessing.
    """
    if phiclockwise is not None and not isinstance(
        phiclockwise, (bool, np.bool_)
    ):
        raise TypeError("phiclockwise must be a boolean or None")

    if flux_normalization not in (None, "Wb", "Wb/rad"):
        raise ValueError("flux_normalization must be either 'Wb' or 'Wb/rad'")

    sign_q = _finite_nonzero_sign(q, "q")
    sign_ip = _finite_nonzero_sign(ip, "ip")
    sign_b0 = _finite_nonzero_sign(b0, "b0")

    try:
        psi_difference = float(np.asarray(psibndr)) - float(np.asarray(psiaxis))
    except (TypeError, ValueError) as exc:
        raise TypeError("psiaxis and psibndr must be real scalars") from exc
    psi_difference_sign = _finite_nonzero_sign(
        psi_difference,
        "psibndr - psiaxis",
    )

    sigma_bp = psi_difference_sign * sign_ip
    sigma_rhothetaphi = sign_q * sign_ip * sign_b0

    base_candidates = []
    for cocos_id in range(1, 9):
        descriptor = cocos(cocos_id)
        if descriptor.sigma_Bp != sigma_bp:
            continue
        if descriptor.sigma_rhotp != sigma_rhothetaphi:
            continue
        if (
            phiclockwise is not None
            and descriptor.phiclockwise != bool(phiclockwise)
        ):
            continue
        base_candidates.append(cocos_id)

    if flux_normalization == "Wb/rad":
        candidates = base_candidates
    elif flux_normalization == "Wb":
        candidates = [candidate + 10 for candidate in base_candidates]
    else:
        candidates = base_candidates + [
            candidate + 10 for candidate in base_candidates
        ]

    return COCOSResolution(tuple(sorted(candidates)))


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
        * ((2 * np.pi) ** exp_Bp_eff)
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
    cocos_n: int,
) -> Dict[str, Any]:
    """
    Transform equilibrium dictionary from one COCOS to another.

    Parameters
    ----------
    eqd : dict
        Dictionary from reading the EQDSK file
    cocos_m : int
        Target COCOS convention (1-18)
    cocos_n : int
        Input COCOS convention (1-18)

    Returns
    -------
    dict
        Equilibrium data converted to cocos_m

    Examples
    --------
    >>> eqd_cocos1 = fromCocosNtoCocosM(eqd_data, cocos_m=1, cocos_n=3)
    """
    transform_dict = transform_cocos(cocos(cocos_n), cocos(cocos_m))

    eqdout = copy.deepcopy(eqd)
    field_transforms = {
        "rdim": "R",
        "zdim": "Z",
        "rcentr": "R",
        "rleft": "R",
        "zmid": "Z",
        "rmagx": "R",
        "zmagx": "Z",
        "simagx": "PSI",
        "sibdry": "PSI",
        "bcentr": "B",
        "cpasma": "I",
        "fpol": "F",
        "pres": "PRES",
        "ffprime": "FFPRIME",
        "pprime": "PPRIME",
        "psi": "PSI",
        "qpsi": "Q",
        "rbdry": "R",
        "zbdry": "Z",
        "rlim": "R",
        "zlim": "Z",
    }
    for field, transform in field_transforms.items():
        if field in eqd:
            eqdout[field] = eqd[field] * transform_dict[transform]

    return eqdout
