"""
Numba utilities required by the pycocos equilibrium stack.

Only the interpolation primitives needed by magnetic field-line integration
are kept in this standalone package.
"""

from typing import Tuple

import numba as nb
import numpy as np


@nb.njit(nogil=True, cache=True)
def interp2d_fast(
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    fields: Tuple[np.ndarray, ...],
    xq: float,
    yq: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Fast 2D bilinear interpolation within a regular grid.

    Parameters
    ----------
    xmin, ymin : float
        Minimum grid coordinates.
    dx, dy : float
        Uniform grid spacing.
    fields : tuple[np.ndarray, ...]
        Tuple of 2D arrays to interpolate.
    xq, yq : float
        Query point.
    nx, ny : int
        Grid lengths for x and y.

    Returns
    -------
    np.ndarray
        Interpolated value for each field.
    """
    ia = int(min(nx - 2, max(0, np.floor((xq - xmin) / dx))))
    ja = int(min(ny - 2, max(0, np.floor((yq - ymin) / dy))))

    ia1 = ia + 1
    ja1 = ja + 1

    xgrid_ia = xmin + dx * ia
    ygrid_ja = ymin + dy * ja

    ax1 = min(1.0, max(0.0, (xq - xgrid_ia) / dx))
    ax = 1.0 - ax1
    ay1 = min(1.0, max(0.0, (yq - ygrid_ja) / dy))
    ay = 1.0 - ay1

    a00 = ax * ay
    a10 = ax1 * ay
    a01 = ax * ay1
    a11 = ax1 * ay1

    output = np.zeros((len(fields),))
    for ifield in range(len(fields)):
        output[ifield] = (
            fields[ifield][ia, ja] * a00
            + fields[ifield][ia1, ja] * a10
            + fields[ifield][ia, ja1] * a01
            + fields[ifield][ia1, ja1] * a11
        )
    return output

