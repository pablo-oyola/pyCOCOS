"""
Example: load a g-EQDSK file and look at the equilibrium.

Uses the analytic Solov'ev file that solovev_equilibrium.py writes.
"""

from pathlib import Path

from pycocos import EQDSK
from solovev_equilibrium import write_solovev

path = write_solovev(Path(__file__).with_name("solovev.geqdsk"))

# A g-EQDSK file does not record its COCOS. Pass it when you know it...
eq = EQDSK(str(path), cocos_in=1)

# ...or let pyCOCOS detect it. The signs of psi, q and Ip alone leave four
# candidates (1, 2, 11, 12); the toroidal-angle orientation and the flux
# normalization settle it. Either way the file is converted once, on load.
detected = EQDSK(str(path), phiclockwise_in=False, flux_normalization="Wb/rad")
print(f"Detected COCOS {detected.cocos_input}, stored as COCOS {detected.cocos_internal}")

# Grid, axis, flux and field at a glance.
eq.summary()

# Field components on the (R, z) grid, and the 1-D Grad-Shafranov profiles.
print(f"Field components: {list(eq.Bdata.data_vars)}")
q = eq.gs_profs["q"]
print(f"q(axis) = {float(q[0]):.3f}, q(edge) = {float(q[-1]):.3f}")
