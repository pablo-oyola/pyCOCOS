"""
Example: convert a g-EQDSK file between COCOS conventions.

The Solov'ev file is COCOS 1 (psi in Wb/rad). Writing it as COCOS 11 scales
psi by 2 pi (Wb) and keeps every sign; reading it back as COCOS 11 recovers
the same internal arrays.
"""

from pathlib import Path

import numpy as np

from pycocos import EQDSK
from solovev_equilibrium import write_solovev

here = Path(__file__).parent
eq = EQDSK(str(write_solovev(here / "solovev.geqdsk")), cocos_in=1)

# A non-mutating g-EQDSK view of the stored arrays, in any COCOS.
gdata = eq.to_geqdsk(cocos_out=11)
print(f"psi at the boundary: COCOS 1 {eq.to_geqdsk()['sibdry']:.4f} Wb/rad, "
      f"COCOS 11 {gdata['sibdry']:.4f} Wb")

# save() converts only the copy it writes, and refuses to overwrite a file.
out = here / "solovev_cocos11.geqdsk"
out.unlink(missing_ok=True)
eq.save(str(out), cocos_out=11)

reloaded = EQDSK(str(out), cocos_in=11)
same = np.allclose(reloaded.psi, eq.psi) and np.allclose(reloaded.q, eq.q)
print(f"Round trip through COCOS 11 recovers psi and q: {same}")
