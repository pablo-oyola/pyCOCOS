"""
Example: write an analytic Solov'ev equilibrium as a g-EQDSK file.

The other examples read the file this one writes, so they run without any
experimental data. The flux function is

    psi(R, z) = A * (R**2 z**2 + kappa**2 / 4 * (R**2 - R0**2)**2),

an exact Grad-Shafranov solution with constant p' and FF' = 0. It is written
in COCOS 1: psi in Wb/rad increasing outwards, B = grad(phi) x grad(psi) +
F grad(phi), with positive plasma current and toroidal field. Its safety
factor follows from the flux-surface volume integral and is exact to
quadrature, which makes it a reference for the q that pyCOCOS computes.

Run it directly to write ``solovev.geqdsk`` next to this script.
"""

from pathlib import Path

import numpy as np
from freeqdsk import geqdsk

MU0 = 4.0e-7 * np.pi

# Geometry and field: a medium-size, elongated tokamak.
R0, A_MINOR, KAPPA = 3.0, 1.0, 1.6     # m, m, elongation
B0, Q0 = 2.5, 1.1                      # T on axis, on-axis safety factor
A = B0 / (2.0 * R0**2 * KAPPA * Q0)    # psi coefficient, T/m^2
PSI_BOUNDARY = A * KAPPA**2 / 4.0 * ((R0 + A_MINOR)**2 - R0**2)**2
PPRIME = -2.0 * A * (1.0 + KAPPA**2) / MU0
F0 = B0 * R0

# Gauss-Legendre nodes on theta in (-pi/2, pi/2): the surface integrals below
# are smooth in theta after u = R**2 = R0**2 + w sin(theta).
_X, _W = np.polynomial.legendre.leggauss(64)
_THETA, _WEIGHT = 0.5 * np.pi * _X, 0.5 * np.pi * _W


def psi_rz(R, z):
    return A * (R**2 * z**2 + KAPPA**2 / 4.0 * (R**2 - R0**2)**2)


def safety_factor(psi):
    """Exact q(psi) = F / (2 pi) dV/dpsi, with V = integral of dR dz / R."""
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    s = np.sqrt(np.maximum(psi, 1.0e-300) / A)
    w = 2.0 * s / KAPPA
    sin, cos2 = np.sin(_THETA), np.cos(_THETA)**2
    u = R0**2 + w[:, None] * sin[None, :]
    G = (cos2 * u**-1.5) @ _WEIGHT
    dG = -1.5 * (cos2 * sin * u**-2.5) @ _WEIGHT
    dV_ds = 4.0 * s / KAPPA * G + 2.0 * s**2 / KAPPA * dG * 2.0 / KAPPA
    return F0 / (2.0 * np.pi) * dV_ds / (2.0 * A * s)


def plasma_current():
    """Ip = integral of j_phi dA = -p' * integral of R dA over the plasma."""
    s = np.sqrt(PSI_BOUNDARY / A)
    w = 2.0 * s / KAPPA
    u = R0**2 + w * np.sin(_THETA)
    integral_R_dA = s * w * (np.cos(_THETA)**2 * u**-0.5) @ _WEIGHT
    return -PPRIME * integral_R_dA


def boundary(npoints=181):
    """The last closed surface, psi = PSI_BOUNDARY, as a closed (R, z) loop."""
    u_out, u_in = (R0 + A_MINOR)**2, 2.0 * R0**2 - (R0 + A_MINOR)**2
    u = 0.5 * (u_in + u_out) - 0.5 * (u_out - u_in) * np.cos(np.linspace(0.0, np.pi, npoints))
    z = np.sqrt(np.maximum(PSI_BOUNDARY / A - KAPPA**2 / 4.0 * (u - R0**2)**2, 0.0) / u)
    R = np.sqrt(u)
    return np.concatenate([R, R[-2::-1]]), np.concatenate([z, -z[-2::-1]])


def write_solovev(path, nx=129, ny=129):
    """Write the equilibrium to ``path`` and return the path."""
    path = Path(path)
    R = np.linspace(1.0, 4.4, nx)
    z = np.linspace(-2.5, 2.5, ny)
    psi_1d = np.linspace(0.0, PSI_BOUNDARY, nx)
    rbdry, zbdry = boundary()
    data = {
        "nx": nx, "ny": ny,
        "rdim": R[-1] - R[0], "zdim": z[-1] - z[0], "rleft": R[0], "zmid": 0.0,
        "rcentr": R0, "bcentr": B0,
        "rmagx": R0, "zmagx": 0.0, "simagx": 0.0, "sibdry": PSI_BOUNDARY,
        "cpasma": plasma_current(),
        "fpol": np.full(nx, F0),
        "pres": PPRIME * (psi_1d - PSI_BOUNDARY),
        "ffprime": np.zeros(nx),
        "pprime": np.full(nx, PPRIME),
        "qpsi": safety_factor(psi_1d),
        "psi": psi_rz(R[:, None], z[None, :]),
        "rbdry": rbdry, "zbdry": zbdry,
    }
    with open(path, "w") as handle:
        geqdsk.write(data, handle, label="SOLOVEV")
    return path


if __name__ == "__main__":
    out = write_solovev(Path(__file__).with_name("solovev.geqdsk"))
    print(f"Wrote {out.name}: Ip = {plasma_current() / 1e6:.3f} MA, "
          f"q from {safety_factor(1e-12)[0]:.3f} to {safety_factor(PSI_BOUNDARY)[0]:.3f}")
