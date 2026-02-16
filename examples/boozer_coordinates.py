"""
Example: Computing Boozer coordinates.
"""

from pycocos import EQDSK

# Load equilibrium
eq = EQDSK("equilibrium.geqdsk")

# Compute Boozer coordinates
mag_coords = eq.compute_coordinates(coordinate_system="boozer", lpsi=201, ltheta=256)

# Transform from cylindrical to magnetic coordinates
coords = mag_coords(R=2.0, z=0.0, grid=False)

print(f"Psi: {coords.psi.values}")
print(f"Theta: {coords.theta.values}")
print(f"Nu: {coords.nu.values}")

# Transform from magnetic to cylindrical coordinates
cyl_coords = mag_coords.transform_inverse(psi=0.5, thetamag=0.0)

print(f"R: {cyl_coords.R_inv.values}")
print(f"z: {cyl_coords.z_inv.values}")

