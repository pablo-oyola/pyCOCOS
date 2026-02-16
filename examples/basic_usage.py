"""
Basic usage example for pycocos.
"""

from pycocos import EQDSK

# Load equilibrium from file
eq = EQDSK("equilibrium.geqdsk")

# Access magnetic field components
Br = eq.Bdata.Br
Bz = eq.Bdata.Bz
Bphi = eq.Bdata.Bphi

# Access flux surfaces
psi = eq.fluxdata.psipol
rho = eq.fluxdata.rhopol

# Get magnetic axis
R_axis = eq.fluxdata.Raxis
z_axis = eq.fluxdata.zaxis

print(f"Magnetic axis: R={R_axis:.3f} m, z={z_axis:.3f} m")

