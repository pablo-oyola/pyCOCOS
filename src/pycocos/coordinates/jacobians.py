"""
Jacobian computation functions for different magnetic coordinate systems.

The main difference between coordinate systems is the choice of Jacobian.
Each function computes the Jacobian J(psi, theta) for a specific coordinate system.
"""

import numpy as np
from typing import Tuple, Any, Union


def compute_boozer_jacobian(
    I: Union[np.ndarray, float],
    F: Union[np.ndarray, float],
    q: Union[np.ndarray, float],
    B: np.ndarray
) -> np.ndarray:
    """
    Compute the Boozer coordinate Jacobian.

    The Boozer Jacobian is defined as:
        J = (I + q*F) / B^2

    where:
        I: Toroidal current profile
        F: F(psi) = R*B_phi function
        q: Safety factor profile
        B: Magnetic field magnitude

    Parameters
    ----------
    I : np.ndarray or float
        Toroidal current profile. Can be:
        - Scalar: constant value for a flux surface
        - 1D array: profile along psi
        - 2D array: (psi x theta) matching B shape
    F : np.ndarray or float
        F(psi) function profile (same shape options as I)
    q : np.ndarray or float
        Safety factor profile (same shape options as I)
    B : np.ndarray
        Magnetic field magnitude. Can be:
        - 1D array: along theta for a single flux surface
        - 2D array: (psi x theta)

    Returns
    -------
    np.ndarray
        Jacobian J matching B's shape

    Notes
    -----
    The function automatically handles broadcasting of scalar or 1D profiles
    to match the B array shape.
    """
    # Ensure inputs are arrays
    I = np.asarray(I)
    F = np.asarray(F)
    q = np.asarray(q)
    B = np.asarray(B)
    
    # Handle broadcasting based on input shapes
    # If I, F, q are scalars or 1D, broadcast to match B
    if I.ndim == 0:
        # Scalar - broadcast to all elements
        I_2d = np.full_like(B, I.item())
    elif I.ndim == 1:
        # 1D array - broadcast along appropriate axis
        if len(I) == B.shape[0] and B.ndim == 2:
            # Profile along psi, broadcast to theta
            I_2d = I[:, np.newaxis]
        elif len(I) == B.size:
            # Flattened array matching B size
            I_2d = I.reshape(B.shape)
        else:
            # Single value - broadcast
            I_2d = np.full_like(B, I[0] if len(I) > 0 else 0.0)
    else:
        # Already 2D or higher - try to broadcast
        I_2d = np.broadcast_to(I, B.shape) if I.shape != B.shape else I
    
    # Same for F and q
    if F.ndim == 0:
        F_2d = np.full_like(B, F.item())
    elif F.ndim == 1:
        if len(F) == B.shape[0] and B.ndim == 2:
            F_2d = F[:, np.newaxis]
        elif len(F) == B.size:
            F_2d = F.reshape(B.shape)
        else:
            F_2d = np.full_like(B, F[0] if len(F) > 0 else 0.0)
    else:
        F_2d = np.broadcast_to(F, B.shape) if F.shape != B.shape else F
    
    if q.ndim == 0:
        q_2d = np.full_like(B, q.item())
    elif q.ndim == 1:
        if len(q) == B.shape[0] and B.ndim == 2:
            q_2d = q[:, np.newaxis]
        elif len(q) == B.size:
            q_2d = q.reshape(B.shape)
        else:
            q_2d = np.full_like(B, q[0] if len(q) > 0 else 0.0)
    else:
        q_2d = np.broadcast_to(q, B.shape) if q.shape != B.shape else q
    
    # Compute Jacobian: J = (I + q*F) / B^2
    J = (I_2d + q_2d * F_2d) / (B**2)
    
    return J


def compute_hamada_jacobian(
    I: np.ndarray,
    F: np.ndarray,
    q: np.ndarray,
    B: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Compute the Hamada coordinate Jacobian.

    The Hamada Jacobian is constant on flux surfaces:
        J = constant(psi)

    This is a placeholder for future implementation.

    Parameters
    ----------
    I : np.ndarray
        Toroidal current profile
    F : np.ndarray
        F(psi) function profile
    q : np.ndarray
        Safety factor profile
    B : np.ndarray
        Magnetic field magnitude
    **kwargs
        Additional parameters

    Returns
    -------
    np.ndarray
        Jacobian J(psi, theta)

    Raises
    ------
    NotImplementedError
        Hamada coordinates not yet implemented
    """
    raise NotImplementedError("Hamada coordinates not yet implemented")


def compute_pest_jacobian(
    I: np.ndarray,
    F: np.ndarray,
    q: np.ndarray,
    B: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Compute the PEST coordinate Jacobian.

    PEST (Poloidal Equal Spacing) coordinates use a different Jacobian.
    This is a placeholder for future implementation.

    Parameters
    ----------
    I : np.ndarray
        Toroidal current profile
    F : np.ndarray
        F(psi) function profile
    q : np.ndarray
        Safety factor profile
    B : np.ndarray
        Magnetic field magnitude
    **kwargs
        Additional parameters

    Returns
    -------
    np.ndarray
        Jacobian J(psi, theta)

    Raises
    ------
    NotImplementedError
        PEST coordinates not yet implemented
    """
    raise NotImplementedError("PEST coordinates not yet implemented")


def compute_equal_arc_jacobian(
    I: np.ndarray,
    F: np.ndarray,
    q: np.ndarray,
    B: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Compute the equal-arc coordinate Jacobian.

    Equal-arc coordinates use arc length along field lines.
    This is a placeholder for future implementation.

    Parameters
    ----------
    I : np.ndarray
        Toroidal current profile
    F : np.ndarray
        F(psi) function profile
    q : np.ndarray
        Safety factor profile
    B : np.ndarray
        Magnetic field magnitude
    **kwargs
        Additional parameters

    Returns
    -------
    np.ndarray
        Jacobian J(psi, theta)

    Raises
    ------
    NotImplementedError
        Equal-arc coordinates not yet implemented
    """
    raise NotImplementedError("Equal-arc coordinates not yet implemented")
