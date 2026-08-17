"""
Example: Reading and writing EQDSK files.
"""

from pycocos import EQDSK

# Read EQDSK file (auto-detects COCOS)
eq = EQDSK(
    "input.geqdsk",
    cocos_internal=1,
    phiclockwise_in=True,
    flux_normalization="Wb/rad",
)

# Build a non-mutating g-EQDSK view in the stored internal convention.
gdata = eq.to_geqdsk()

print(f"Prepared {len(gdata)} g-EQDSK fields")

# Convert only the output copy and write it. The file itself does not record
# enough metadata to recover its COCOS, so pass cocos_in=3 when reloading it.
eq.save("output.geqdsk", cocos_out=3)
