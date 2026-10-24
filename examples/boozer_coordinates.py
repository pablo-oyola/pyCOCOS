"""
Example: compute Boozer coordinates and transform between frames.

The Solov'ev equilibrium has an exact safety factor, so the q that pyCOCOS
derives from the traced flux surfaces can be checked against it.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycocos import EQDSK
from solovev_equilibrium import PSI_BOUNDARY, safety_factor, write_solovev

path = write_solovev(Path(__file__).with_name("solovev.geqdsk"))
eq = EQDSK(str(path), cocos_in=1)

# Boozer coordinates, J = (I + qF) / B^2.  Other registered systems:
# "pest", "equal_arc", "hamada".
mag = eq.compute_coordinates("boozer", lpsi=65, ltheta=128)

# The q profile that comes out of the construction, against the exact one.
q = mag.deriv["q"]
q_exact = safety_factor(q.psi0.values)
print(f"max relative q error: {np.max(np.abs(q.values - q_exact) / q_exact):.1e}")

# Cylindrical -> magnetic at one point...
point = mag(R=3.5, z=0.3)
print(f"(R, z) = (3.5, 0.3) m -> psi = {float(point.psi[0]):.4f} Wb/rad, "
      f"theta = {float(point.theta[0]):.4f} rad, nu = {float(point.nu[0]):.4f} rad")

# ...and magnetic -> cylindrical, here on the psi_N = 0.5 surface at theta = 0.
back = mag.transform_inverse(psi=0.5, thetamag=0.0, psi_is_norm=True)
print(f"(psi_N, theta) = (0.5, 0) -> R = {float(back.R_inv[0]):.4f} m, "
      f"z = {float(back.z_inv[0]):.1e} m")

# The Boozer grid over the plasma, and q.
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.5), width_ratios=(1, 1.3))
# theta_star is padded past [0, 2 pi) for periodic interpolation; draw one turn.
theta_star = mag.coords["theta_star"].values
one_turn = (theta_star >= 0.0) & (theta_star < 2.0 * np.pi)
R_inv = mag.coords["R_inv"].values[:, one_turn]
z_inv = mag.coords["z_inv"].values[:, one_turn]
for i in range(0, R_inv.shape[0], 8):
    left.plot(np.r_[R_inv[i], R_inv[i, 0]], np.r_[z_inv[i], z_inv[i, 0]],
              color="#2B5EA6", lw=0.8)
for j in range(0, R_inv.shape[1], 8):
    left.plot(R_inv[:, j], z_inv[:, j], color="#9C5D14", lw=0.6)
left.plot(eq.geometry["R_boundary"], eq.geometry["z_boundary"], "k", lw=1.2)
left.set(xlabel="R [m]", ylabel="z [m]", aspect="equal", title="Boozer grid")

psi_n = q.psi0.values / PSI_BOUNDARY
right.plot(psi_n, q_exact, "k", lw=2.5, alpha=0.3, label="exact")
right.plot(psi_n, q.values, "C3--", lw=1.2, label="pyCOCOS")
right.set(xlabel=r"$\psi_N$", ylabel="q", title="Safety factor")
right.legend()
fig.tight_layout()
fig.savefig(Path(__file__).with_name("boozer_coordinates.png"), dpi=110)
