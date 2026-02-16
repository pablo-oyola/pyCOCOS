"""
Example: Reading and writing EQDSK files.
"""

from pycocos import EQDSK

# Read EQDSK file (auto-detects COCOS)
eq = EQDSK("input.geqdsk", cocos=1, phiclockwise=True)

# Modify equilibrium if needed
# ...

# Write to new file
gdata = eq.to_geqdsk("output.geqdsk", cocos=1)

print("EQDSK file written successfully")

