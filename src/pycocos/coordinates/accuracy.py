"""Accuracy profiles for magnetic-coordinate construction.

The tolerances collected here control approximation and nonlinear-solver
stopping criteria.  They do not relax topological checks, coordinate
orientation, or algebraic differential identities.
"""

from dataclasses import dataclass, replace
from typing import Literal, Optional, Union

import numpy as np


CoordinateAccuracyProfile = Literal["standard", "strict"]


@dataclass(frozen=True)
class CoordinateAccuracy:
    """Numerical accuracy budget for magnetic-coordinate construction.

    All flux tolerances are normalized by the axis-to-boundary flux span.
    ``theta_tolerance`` is an angular Newton-update tolerance in radians.

    The standard profile avoids spending workstation-scale resources to
    correct interpolation residuals that are already below one part per
    hundred thousand.  The strict profile retains the previous pyCOCOS stopping
    thresholds for reference calculations and regression comparisons.
    """

    bridge_flux_tolerance: float = 1.0e-5
    surface_flux_tolerance: float = 1.0e-7
    constraint_flux_tolerance: float = 1.0e-7
    theta_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in (
            "bridge_flux_tolerance",
            "surface_flux_tolerance",
            "constraint_flux_tolerance",
            "theta_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)

    @classmethod
    def standard(cls) -> "CoordinateAccuracy":
        """Return the default, performance-oriented coordinate budget."""
        return cls()

    @classmethod
    def strict(cls) -> "CoordinateAccuracy":
        """Return the legacy high-accuracy coordinate budget."""
        return cls(
            bridge_flux_tolerance=1.0e-8,
            surface_flux_tolerance=1.0e-10,
            constraint_flux_tolerance=1.0e-10,
            theta_tolerance=5.0e-11,
        )

    def with_overrides(
        self,
        *,
        bridge_flux_tolerance: Optional[float] = None,
        surface_flux_tolerance: Optional[float] = None,
        constraint_flux_tolerance: Optional[float] = None,
        theta_tolerance: Optional[float] = None,
    ) -> "CoordinateAccuracy":
        """Return a copy with explicitly supplied tolerances replaced."""
        overrides = {
            name: value
            for name, value in (
                ("bridge_flux_tolerance", bridge_flux_tolerance),
                ("surface_flux_tolerance", surface_flux_tolerance),
                ("constraint_flux_tolerance", constraint_flux_tolerance),
                ("theta_tolerance", theta_tolerance),
            )
            if value is not None
        }
        return replace(self, **overrides)


def resolve_coordinate_accuracy(
    profile: Optional[
        Union[CoordinateAccuracy, CoordinateAccuracyProfile]
    ] = None,
    *,
    bridge_flux_tolerance: Optional[float] = None,
    surface_flux_tolerance: Optional[float] = None,
    constraint_flux_tolerance: Optional[float] = None,
    theta_tolerance: Optional[float] = None,
) -> CoordinateAccuracy:
    """Resolve a named/object profile and apply explicit field overrides."""
    if profile is None or profile == "standard":
        resolved = CoordinateAccuracy.standard()
    elif profile == "strict":
        resolved = CoordinateAccuracy.strict()
    elif isinstance(profile, CoordinateAccuracy):
        resolved = profile
    else:
        raise ValueError(
            "coordinate_accuracy must be None, 'standard', 'strict', or a "
            "CoordinateAccuracy instance."
        )
    return resolved.with_overrides(
        bridge_flux_tolerance=bridge_flux_tolerance,
        surface_flux_tolerance=surface_flux_tolerance,
        constraint_flux_tolerance=constraint_flux_tolerance,
        theta_tolerance=theta_tolerance,
    )
