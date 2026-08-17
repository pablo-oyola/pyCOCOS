"""Lightweight products from magnetic-coordinate construction.

The spectral coordinate map and its fitted surface tables are useful on their
own.  Building every direct/inverse derivative and metric coefficient on the
full equilibrium R-z mesh can be substantially more expensive and consume
many times the storage of the map.  This module provides an explicit map-only
result whose full R-z representation is materialized only when requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from .accuracy import CoordinateAccuracy
from .coordinate_map import CoordinateDifferentials, SpectralCoordinateMap


@dataclass
class MagneticCoordinateMapProduct:
    """A fitted magnetic-coordinate map without full R-z materialization.

    Instances are returned by
    :meth:`pycocos.Equilibrium.compute_coordinates` when
    ``materialize_rz=False``.  Surface profiles and fitted tables are retained
    as primitive NumPy arrays, while :attr:`coordinate_map` provides fast
    forward-map, differential, and angle-inversion operations.  Call
    :meth:`materialize_rz` to construct the traditional
    ``MagneticCoordinates`` object without retracing flux surfaces.

    :attr:`jacobian` is the determinant of the fitted spectral map on the
    product nodes, consistent with :meth:`differentials`.  The independently
    requested/interpolated construction table is retained as
    :attr:`target_jacobian` for fit-quality audits.
    """

    coordinate_map: SpectralCoordinateMap
    psi: np.ndarray
    theta: np.ndarray
    R: np.ndarray
    z: np.ndarray
    nu: np.ndarray
    jacobian: np.ndarray
    target_jacobian: np.ndarray
    q: np.ndarray
    F: np.ndarray
    I: np.ndarray
    coordinate_system: str
    accuracy: CoordinateAccuracy
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    _materializer: Optional[Callable[[bool], Any]] = field(
        default=None,
        repr=False,
        compare=False,
    )
    _materialized_result: Optional[Any] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        surface_count = np.asarray(self.psi).size
        theta_count = np.asarray(self.theta).size
        expected = (surface_count, theta_count)
        for name in ("R", "z", "nu", "jacobian", "target_jacobian"):
            values = np.asarray(getattr(self, name))
            if values.shape != expected:
                raise ValueError(
                    f"{name} has shape {values.shape}; expected {expected}."
                )
            setattr(self, name, values)
        for name in ("q", "F", "I"):
            values = np.asarray(getattr(self, name))
            if values.shape != (surface_count,):
                raise ValueError(
                    f"{name} has shape {values.shape}; expected "
                    f"{(surface_count,)}."
                )
            setattr(self, name, values)
        self.psi = np.asarray(self.psi)
        self.theta = np.asarray(self.theta)
        self.coordinate_system = str(self.coordinate_system).lower()
        self.diagnostics = dict(self.diagnostics)

    @property
    def rz_materialized(self) -> bool:
        """Whether full R-z coordinate and derivative arrays are present."""

        return self._materialized_result is not None

    def values(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate ``R``, ``z``, and ``nu`` on the fitted spectral map."""

        return self.coordinate_map.values(psi, theta)

    def differentials(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
    ) -> CoordinateDifferentials:
        """Evaluate the fitted coordinate basis and reciprocal tensors."""

        return self.coordinate_map.differentials(psi, theta)

    def solve_theta(
        self,
        *,
        psi: np.ndarray,
        R: np.ndarray,
        z: np.ndarray,
        initial_theta: np.ndarray,
        tolerance: Optional[float] = None,
        max_iterations: int = 30,
    ) -> np.ndarray:
        """Invert the fitted surface geometry for magnetic poloidal angle."""

        selected_tolerance = (
            self.accuracy.theta_tolerance
            if tolerance is None
            else float(tolerance)
        )
        return self.coordinate_map.solve_theta(
            psi=psi,
            R=R,
            z=z,
            initial_theta=initial_theta,
            tolerance=selected_tolerance,
            max_iterations=max_iterations,
        )

    def materialize_rz(self, *, build_metric_cache: bool = False) -> Any:
        """Build the traditional full R-z coordinate object once requested.

        Surface tracing, filtering, and the spectral fit are reused.  Only the
        Cartesian-grid inversion/derivative pass and optional derived metric
        caches are constructed.
        """

        if self._materializer is None:
            raise RuntimeError(
                "This coordinate-map product has no attached R-z materializer."
            )
        if not isinstance(build_metric_cache, (bool, np.bool_)):
            raise TypeError("build_metric_cache must be boolean.")
        if self._materialized_result is None:
            self._materialized_result = self._materializer(
                bool(build_metric_cache)
            )
            checkpoint = getattr(self, "_coordinate_checkpoint", None)
            if checkpoint is not None:
                self._materialized_result._coordinate_checkpoint = dict(
                    checkpoint
                )
        elif build_metric_cache and not self._materialized_result.metric_cache_built:
            # Match eager-construction semantics: requesting the derived cache
            # builds Lamé factors as well as both metric tensors, without
            # repeating the R-z map.
            _ = self._materialized_result.lame_mag
            _ = self._materialized_result.metric_covariant
        return self._materialized_result


__all__ = ["MagneticCoordinateMapProduct"]
