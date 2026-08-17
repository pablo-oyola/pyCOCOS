#!/usr/bin/env python3
"""Benchmark pyCOCOS coordinate construction on a real EQDSK equilibrium.

The command measures the public ``EQDSK.compute_coordinates`` workflow and
emits strict JSON.  It can compare the standard and strict accuracy profiles,
the automatic surface quadrature and the historical 7200-point quadrature, or
the full two-by-two matrix.  Timing covers coordinate construction only; EQDSK
loading is measured separately.  ``tracemalloc`` reports Python-managed
allocations and therefore does not include every NumPy/SciPy native allocation.

Examples
--------
Run the inexpensive accuracy comparison with modest benchmark grids::

    python benchmarks/benchmark_coordinate_mapping.py equilibrium.geqdsk \
        --cocos-in 1 --comparison accuracy

Compare automatic and historical surface quadrature at production resolution::

    python benchmarks/benchmark_coordinate_mapping.py equilibrium.geqdsk \
        --cocos-in 1 --comparison theta --lpsi 256 --ltheta 256 \
        --output coordinate-benchmark.json
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if _SOURCE_ROOT.is_dir():
    # Prefer the checkout containing this benchmark over an unrelated editable
    # install.  This is particularly important when comparing worktrees.
    sys.path.insert(0, str(_SOURCE_ROOT))


_DEFAULT_DELTA_FIELDS = (
    "coords:R_inv",
    "coords:z_inv",
    "coords:psi",
    "coords:theta",
    "coords:nu",
    "deriv:jacobian",
    "deriv:q",
    "deriv:F",
    "deriv:I",
    "deriv:h",
    "deriv:dPsi_dr",
    "deriv:dPsi_dz",
    "deriv:dTheta_dr",
    "deriv:dTheta_dz",
    "deriv:dzeta_dr",
    "deriv:dzeta_dz",
    "deriv:direct_det_Rz",
)
_ANGULAR_FIELDS = frozenset(("coords:theta", "coords:nu"))


@dataclass(frozen=True)
class Variant:
    """One coordinate-construction configuration."""

    accuracy: str
    theta_mode: str
    n_theta_geom: int | None

    @property
    def key(self) -> str:
        return f"{self.accuracy}__{self.theta_mode}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "theta_mode": self.theta_mode,
            "n_theta_geom": self.n_theta_geom,
        }


@dataclass
class FieldSnapshot:
    """Small description plus values retained for pairwise deltas."""

    dims: tuple[str, ...]
    values: np.ndarray


def _parse_optional_integer(value: str) -> int | None:
    if value.strip().lower() in {"auto", "none"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer or 'auto', got {value!r}"
        ) from exc


def _parse_optional_boolean(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"auto", "none"}:
        return None
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected auto, true, or false, got {value!r}"
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _finite_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark pyCOCOS coordinate mapping and write timing, Python "
            "allocation peaks, diagnostics, and numerical deltas as JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("eqdsk", type=Path, help="input g-EQDSK file")
    parser.add_argument(
        "--comparison",
        choices=("accuracy", "theta", "all"),
        default="accuracy",
        help=(
            "accuracy: standard-auto vs strict-auto; theta: standard-auto "
            "vs standard-legacy; all: the complete two-by-two matrix"
        ),
    )
    parser.add_argument(
        "--legacy-n-theta-geom",
        type=_positive_integer,
        default=7200,
        help="explicit surface quadrature used by legacy variants",
    )
    parser.add_argument(
        "--coordinate-system",
        default="boozer",
        help="registered pyCOCOS magnetic coordinate system",
    )
    parser.add_argument(
        "--cocos-in",
        type=_parse_optional_integer,
        default=1,
        help="input COCOS number, or 'auto' for inference",
    )
    parser.add_argument(
        "--cocos-internal",
        type=int,
        default=1,
        help="COCOS convention used by pyCOCOS internally",
    )
    parser.add_argument(
        "--phiclockwise-in",
        type=_parse_optional_boolean,
        default=None,
        metavar="{auto,true,false}",
        help="input toroidal-angle orientation used during COCOS inference",
    )
    parser.add_argument(
        "--flux-normalization",
        choices=("Wb", "Wb/rad"),
        default=None,
        help="input flux normalization used during COCOS inference",
    )

    grid = parser.add_argument_group("coordinate grid")
    grid.add_argument(
        "--lpsi",
        type=_positive_integer,
        default=33,
        help="returned radial surface count (modest benchmark default)",
    )
    grid.add_argument(
        "--ltheta",
        type=_positive_integer,
        default=65,
        help="returned poloidal angle count (modest benchmark default)",
    )
    grid.add_argument("--dr-hr", type=_finite_positive, default=1.0e-3)
    grid.add_argument("--dz-hz", type=_finite_positive, default=1.0e-3)
    grid.add_argument("--padding", type=_finite_float, default=0.05)
    grid.add_argument("--ntht-pad", type=int, default=5)
    grid.add_argument("--rhopol-min", type=_finite_float, default=None)
    grid.add_argument("--rhopol-max", type=_finite_float, default=None)
    grid.add_argument(
        "--spectral-max-mode", type=_positive_integer, default=16
    )
    grid.add_argument(
        "--radial-guard-surfaces", type=int, default=3
    )

    symmetry = parser.add_argument_group("optional symmetry projection")
    symmetry.add_argument(
        "--enforce-up-down-symmetry",
        action="store_true",
        help="enable the projected-equilibrium/bridge path",
    )
    symmetry.add_argument(
        "--symmetry-tolerance",
        type=_finite_positive,
        default=None,
        help="required when symmetry projection is enabled",
    )
    symmetry.add_argument(
        "--projected-bridge-repair-strategy",
        choices=("bounded", "allow"),
        default="bounded",
        help="whether an oversized projected-flux bridge repair is rejected",
    )

    measurement = parser.add_argument_group("measurement and output")
    measurement.add_argument(
        "--repeat",
        type=_positive_integer,
        default=1,
        help="fresh-EQDSK repetitions per variant",
    )
    measurement.add_argument(
        "--delta-field",
        action="append",
        default=None,
        metavar="{coords,deriv}:NAME",
        help=(
            "field retained for pairwise numerical deltas; repeat the option "
            "to select multiple fields"
        ),
    )
    measurement.add_argument(
        "--defer-metrics",
        action="store_true",
        help=(
            "request lazy metric materialization when supported by the active "
            "pyCOCOS checkout"
        ),
    )
    measurement.add_argument(
        "--map-only",
        action="store_true",
        help=(
            "stop after the fitted spectral map and surface profiles, without "
            "materializing full R-z coordinate/derivative arrays"
        ),
    )
    measurement.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop launching variants after the first failed variant",
    )
    measurement.add_argument(
        "--allow-failures",
        action="store_true",
        help="return exit status zero even when a benchmark variant fails",
    )
    measurement.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress messages on stderr",
    )
    measurement.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path; stdout is used when omitted",
    )
    measurement.add_argument(
        "--indent", type=int, default=2, help="JSON indentation"
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.ltheta < 4:
        parser.error("--ltheta must be at least 4")
    if args.lpsi < 2:
        parser.error("--lpsi must be at least 2")
    if args.ntht_pad < 0:
        parser.error("--ntht-pad must be nonnegative")
    if args.radial_guard_surfaces < 0:
        parser.error("--radial-guard-surfaces must be nonnegative")
    if args.enforce_up_down_symmetry and args.symmetry_tolerance is None:
        parser.error(
            "--symmetry-tolerance is required with "
            "--enforce-up-down-symmetry"
        )
    if (
        args.rhopol_min is not None
        and not 0.0 <= args.rhopol_min <= 1.0
    ):
        parser.error("--rhopol-min must lie in [0, 1]")
    if (
        args.rhopol_max is not None
        and not 0.0 <= args.rhopol_max <= 1.0
    ):
        parser.error("--rhopol-max must lie in [0, 1]")
    if (
        args.rhopol_min is not None
        and args.rhopol_max is not None
        and args.rhopol_min >= args.rhopol_max
    ):
        parser.error("--rhopol-min must be smaller than --rhopol-max")

    fields = args.delta_field or _DEFAULT_DELTA_FIELDS
    invalid = [
        field
        for field in fields
        if ":" not in field or field.split(":", 1)[0] not in {"coords", "deriv"}
    ]
    if invalid:
        parser.error(
            "--delta-field values must use coords:NAME or deriv:NAME; "
            f"invalid values: {invalid}"
        )


def _variant_plan(args: argparse.Namespace) -> list[Variant]:
    automatic_standard = Variant("standard", "automatic", None)
    automatic_strict = Variant("strict", "automatic", None)
    legacy_standard = Variant(
        "standard", "legacy", args.legacy_n_theta_geom
    )
    legacy_strict = Variant("strict", "legacy", args.legacy_n_theta_geom)
    if args.comparison == "accuracy":
        return [automatic_standard, automatic_strict]
    if args.comparison == "theta":
        return [automatic_standard, legacy_standard]
    return [
        automatic_standard,
        automatic_strict,
        legacy_standard,
        legacy_strict,
    ]


def _comparison_plan(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.comparison == "accuracy":
        return [
            (
                "accuracy_at_automatic_theta",
                "standard__automatic",
                "strict__automatic",
            )
        ]
    if args.comparison == "theta":
        return [
            (
                "theta_at_standard_accuracy",
                "standard__automatic",
                "standard__legacy",
            )
        ]
    return [
        (
            "accuracy_at_automatic_theta",
            "standard__automatic",
            "strict__automatic",
        ),
        (
            "accuracy_at_legacy_theta",
            "standard__legacy",
            "strict__legacy",
        ),
        (
            "theta_at_standard_accuracy",
            "standard__automatic",
            "standard__legacy",
        ),
        (
            "theta_at_strict_accuracy",
            "strict__automatic",
            "strict__legacy",
        ),
    ]


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {
            "real": _safe_scalar(number.real),
            "imag": _safe_scalar(number.imag),
        }
    if isinstance(value, (str, type(None))):
        return value
    return None


def _array_summary(values: Any, *, include_values_limit: int = 16) -> dict[str, Any]:
    array = np.asarray(values)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if array.size == 0:
        return summary
    if np.issubdtype(array.dtype, np.bool_):
        summary.update(
            {
                "true_count": int(np.count_nonzero(array)),
                "false_count": int(array.size - np.count_nonzero(array)),
            }
        )
    elif np.issubdtype(array.dtype, np.number):
        magnitudes = np.abs(array) if np.iscomplexobj(array) else array
        finite = np.isfinite(magnitudes)
        finite_values = magnitudes[finite]
        summary["finite_count"] = int(finite_values.size)
        summary["nonfinite_count"] = int(array.size - finite_values.size)
        if finite_values.size:
            summary.update(
                {
                    "minimum": _safe_scalar(np.min(finite_values)),
                    "maximum": _safe_scalar(np.max(finite_values)),
                    "mean": _safe_scalar(np.mean(finite_values)),
                    "rms": _safe_scalar(
                        np.sqrt(np.mean(np.square(finite_values)))
                    ),
                }
            )
    if array.size <= include_values_limit:
        summary["values"] = _jsonable(array.tolist())
    return summary


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    """Convert diagnostics to strict JSON without dumping large arrays."""
    scalar = _safe_scalar(value)
    if scalar is not None or value is None:
        return scalar
    if depth >= 10:
        return repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if isinstance(value, (list, tuple)):
        if len(value) <= 32:
            return [_jsonable(item, depth=depth + 1) for item in value]
        try:
            return _array_summary(np.asarray(value))
        except (TypeError, ValueError):
            return {
                "length": len(value),
                "head": [
                    _jsonable(item, depth=depth + 1) for item in value[:8]
                ],
            }
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item(), depth=depth + 1)
        except (TypeError, ValueError):
            pass
    return repr(value)


def _dataset_summary(dataset: Any) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    for name, data_array in dataset.data_vars.items():
        variables[str(name)] = {
            "dims": list(data_array.dims),
            "shape": list(data_array.shape),
            "dtype": str(data_array.dtype),
            "units": data_array.attrs.get("units"),
        }
    return {
        "dimensions": {str(name): int(size) for name, size in dataset.sizes.items()},
        "attributes": _jsonable(dataset.attrs),
        "variables": variables,
    }


def _physics_summary(result: Any) -> dict[str, Any]:
    if not hasattr(result, "coords"):
        summary = {
            f"map:{name}": _array_summary(getattr(result, name))
            for name in (
                "psi", "theta", "R", "z", "nu", "jacobian",
                "target_jacobian", "q", "F", "I",
            )
        }
        summary["map:h"] = _array_summary(result.I + result.q * result.F)
        return summary
    summary: dict[str, Any] = {}
    for name in ("R_inv", "z_inv", "psi", "theta", "nu"):
        if name in result.coords:
            summary[f"coords:{name}"] = _array_summary(result.coords[name])
    for name in ("jacobian", "q", "F", "I", "h", "direct_det_Rz"):
        if name in result.deriv:
            summary[f"deriv:{name}"] = _array_summary(result.deriv[name])

    if "inside_lcfs" in result.coords:
        inside = np.asarray(result.coords["inside_lcfs"], dtype=bool)
        summary["inside_lcfs_fraction"] = float(np.mean(inside))
    if "inside_coordinate_domain" in result.coords:
        inside = np.asarray(result.coords["inside_coordinate_domain"], dtype=bool)
        summary["inside_coordinate_domain_fraction"] = float(np.mean(inside))
    if "jacobian" in result.deriv:
        jacobian = np.asarray(result.deriv["jacobian"], dtype=np.float64)
        finite = np.isfinite(jacobian)
        if np.any(finite):
            valid = jacobian[finite]
            summary["jacobian_sign_counts"] = {
                "negative": int(np.count_nonzero(valid < 0.0)),
                "zero": int(np.count_nonzero(valid == 0.0)),
                "positive": int(np.count_nonzero(valid > 0.0)),
            }
    return summary


def _diagnostic_summary(result: Any, construction: Mapping[str, Any]) -> dict[str, Any]:
    coordinate_map = getattr(
        result,
        "_coordinate_map",
        getattr(result, "coordinate_map", None),
    )
    map_diagnostics: dict[str, Any] = {}
    if coordinate_map is not None:
        for name in (
            "flux_constraint_audit",
            "last_flux_constraint_solve_audit",
            "theta_inversion_audit",
            "up_down_symmetry_audit",
        ):
            if hasattr(coordinate_map, name):
                map_diagnostics[name] = _jsonable(
                    getattr(coordinate_map, name)
                )
    return {
        "construction": _jsonable(construction),
        "coordinate": _jsonable(
            getattr(
                result,
                "_coordinate_diagnostics",
                getattr(result, "diagnostics", {}),
            )
        ),
        "coordinate_map": map_diagnostics,
    }


def _take_snapshot(
    result: Any,
    selected_fields: Sequence[str],
) -> dict[str, FieldSnapshot]:
    snapshot: dict[str, FieldSnapshot] = {}
    if not hasattr(result, "coords"):
        product_fields = {
            "coords:R_inv": (("psi", "theta"), result.R),
            "coords:z_inv": (("psi", "theta"), result.z),
            "coords:nu": (("psi", "theta"), result.nu),
            "deriv:jacobian": (("psi", "theta"), result.jacobian),
            "deriv:q": (("psi",), result.q),
            "deriv:F": (("psi",), result.F),
            "deriv:I": (("psi",), result.I),
            "deriv:h": (("psi",), result.I + result.q * result.F),
        }
        for name in selected_fields:
            if name in product_fields:
                dims, values = product_fields[name]
                snapshot[name] = FieldSnapshot(
                    dims=dims,
                    values=np.array(values, copy=True),
                )
        snapshot["coords:@psi0"] = FieldSnapshot(
            dims=("psi",),
            values=np.array(result.psi, copy=True),
        )
        snapshot["coords:@theta_star"] = FieldSnapshot(
            dims=("theta",),
            values=np.array(result.theta, copy=True),
        )
        return snapshot
    for qualified_name in selected_fields:
        dataset_name, field_name = qualified_name.split(":", 1)
        dataset = getattr(result, dataset_name)
        if field_name not in dataset:
            continue
        data_array = dataset[field_name]
        snapshot[qualified_name] = FieldSnapshot(
            dims=tuple(str(dim) for dim in data_array.dims),
            values=np.array(data_array.values, copy=True),
        )

    # Coordinate axes are outputs too and can expose apparently equal fields
    # living on different physical grids.
    for dataset_name in ("coords", "deriv"):
        dataset = getattr(result, dataset_name)
        for axis_name, coordinate in dataset.coords.items():
            if coordinate.ndim != 1:
                continue
            key = f"{dataset_name}:@{axis_name}"
            if key in snapshot:
                continue
            snapshot[key] = FieldSnapshot(
                dims=tuple(str(dim) for dim in coordinate.dims),
                values=np.array(coordinate.values, copy=True),
            )
    return snapshot


def _field_delta(
    reference: FieldSnapshot,
    candidate: FieldSnapshot,
    *,
    angular: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "reference_dims": list(reference.dims),
        "candidate_dims": list(candidate.dims),
        "reference_shape": list(reference.values.shape),
        "candidate_shape": list(candidate.values.shape),
        "angular_wrapping_applied": bool(angular),
    }
    if (
        reference.dims != candidate.dims
        or reference.values.shape != candidate.values.shape
    ):
        report["status"] = "incompatible_shape_or_dimensions"
        return report

    reference_values = reference.values
    candidate_values = candidate.values
    if (
        np.issubdtype(reference_values.dtype, np.bool_)
        or np.issubdtype(candidate_values.dtype, np.bool_)
    ):
        mismatches = np.not_equal(reference_values, candidate_values)
        report.update(
            {
                "status": "ok",
                "mismatch_count": int(np.count_nonzero(mismatches)),
                "mismatch_fraction": float(np.mean(mismatches)),
            }
        )
        return report

    try:
        reference_numeric = np.asarray(reference_values, dtype=np.float64)
        candidate_numeric = np.asarray(candidate_values, dtype=np.float64)
    except (TypeError, ValueError):
        report["status"] = "non_numeric"
        return report

    reference_finite = np.isfinite(reference_numeric)
    candidate_finite = np.isfinite(candidate_numeric)
    overlap = reference_finite & candidate_finite
    report.update(
        {
            "reference_finite_count": int(np.count_nonzero(reference_finite)),
            "candidate_finite_count": int(np.count_nonzero(candidate_finite)),
            "finite_overlap_count": int(np.count_nonzero(overlap)),
            "finite_mask_mismatch_count": int(
                np.count_nonzero(reference_finite ^ candidate_finite)
            ),
        }
    )
    if not np.any(overlap):
        report["status"] = "no_finite_overlap"
        return report

    delta = candidate_numeric[overlap] - reference_numeric[overlap]
    if angular:
        delta = np.arctan2(np.sin(delta), np.cos(delta))
    absolute_delta = np.abs(delta)
    reference_scale = max(
        float(np.max(np.abs(reference_numeric[reference_finite]))),
        np.finfo(np.float64).tiny,
    )
    report.update(
        {
            "status": "ok",
            "max_abs": float(np.max(absolute_delta)),
            "mean_abs": float(np.mean(absolute_delta)),
            "rms": float(np.sqrt(np.mean(np.square(delta)))),
            "reference_global_scale": reference_scale,
            "max_abs_over_reference_global_scale": float(
                np.max(absolute_delta) / reference_scale
            ),
        }
    )
    return report


def _compare_snapshots(
    reference: Mapping[str, FieldSnapshot],
    candidate: Mapping[str, FieldSnapshot],
) -> dict[str, Any]:
    common = sorted(reference.keys() & candidate.keys())
    fields = {
        name: _field_delta(
            reference[name],
            candidate[name],
            angular=name in _ANGULAR_FIELDS,
        )
        for name in common
    }
    numeric_fields = {
        name: field
        for name, field in fields.items()
        if field.get("status") == "ok" and "max_abs" in field
    }
    mismatch_fields = {
        name: field
        for name, field in fields.items()
        if field.get("status") == "ok" and "mismatch_count" in field
    }
    summary: dict[str, Any] = {
        "compared_field_count": len(fields),
        "numeric_field_count": len(numeric_fields),
        "finite_mask_mismatch_count": int(
            sum(
                int(field.get("finite_mask_mismatch_count", 0))
                for field in numeric_fields.values()
            )
        ),
        "boolean_mismatch_count": int(
            sum(int(field["mismatch_count"]) for field in mismatch_fields.values())
        ),
    }
    if numeric_fields:
        largest_absolute_name = max(
            numeric_fields, key=lambda name: numeric_fields[name]["max_abs"]
        )
        largest_relative_name = max(
            numeric_fields,
            key=lambda name: numeric_fields[name][
                "max_abs_over_reference_global_scale"
            ],
        )
        summary.update(
            {
                "largest_max_abs_field": largest_absolute_name,
                "largest_max_abs": numeric_fields[largest_absolute_name][
                    "max_abs"
                ],
                "largest_scaled_delta_field": largest_relative_name,
                "largest_max_abs_over_reference_global_scale": (
                    numeric_fields[largest_relative_name][
                        "max_abs_over_reference_global_scale"
                    ]
                ),
            }
        )
    return {
        "status": "ok",
        "summary": summary,
        "fields": fields,
        "missing_from_reference": sorted(candidate.keys() - reference.keys()),
        "missing_from_candidate": sorted(reference.keys() - candidate.keys()),
    }


def _performance_comparison(
    reference_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare median construction time and Python allocation peak."""
    if (
        reference_report.get("status") != "ok"
        or candidate_report.get("status") != "ok"
    ):
        return {"status": "unavailable_due_to_failed_variant"}
    reference = reference_report["aggregate"]
    candidate = candidate_report["aggregate"]
    reference_seconds = float(reference["coordinate_seconds_median"])
    candidate_seconds = float(candidate["coordinate_seconds_median"])
    reference_peak = float(reference["tracemalloc_peak_bytes_median"])
    candidate_peak = float(candidate["tracemalloc_peak_bytes_median"])
    return {
        "status": "ok",
        "reference_coordinate_seconds_median": reference_seconds,
        "candidate_coordinate_seconds_median": candidate_seconds,
        "candidate_over_reference_wall_time": (
            candidate_seconds / reference_seconds
            if reference_seconds > 0.0
            else None
        ),
        "reference_over_candidate_speedup": (
            reference_seconds / candidate_seconds
            if candidate_seconds > 0.0
            else None
        ),
        "reference_tracemalloc_peak_bytes_median": int(reference_peak),
        "candidate_tracemalloc_peak_bytes_median": int(candidate_peak),
        "candidate_over_reference_tracemalloc_peak": (
            candidate_peak / reference_peak if reference_peak > 0.0 else None
        ),
    }


@contextlib.contextmanager
def _capture_coordinate_construction(
    n_theta_geom: int | None,
) -> Iterator[MutableMapping[str, Any]]:
    """Capture transient diagnostics and select the surface quadrature.

    ``Equilibrium.compute_coordinates`` currently owns the public workflow but
    its low-level construction diagnostics are intentionally local.  This
    benchmark wraps the module-local callable for the duration of one serial
    invocation.  The original is restored even when construction raises.
    """
    equilibrium_module = importlib.import_module("pycocos.core.equilibrium")
    original = equilibrium_module.compute_magnetic_coordinates
    capture: MutableMapping[str, Any] = {"calls": []}

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs["n_theta_geom"] = n_theta_geom
        diagnostics = kwargs.get("diagnostics")
        call: dict[str, Any] = {
            "requested_n_theta_geom": n_theta_geom,
            "diagnostics": {},
        }
        capture["calls"].append(call)
        try:
            return original(*args, **kwargs)
        finally:
            if isinstance(diagnostics, Mapping):
                # Keep references to nested audit dictionaries so bridge
                # updates made immediately after the low-level call remain
                # visible.  Exclude the potentially large projected R-z field.
                call["diagnostics"] = {
                    key: value
                    for key, value in diagnostics.items()
                    if key != "coordinate_psi_field"
                }

    equilibrium_module.compute_magnetic_coordinates = wrapped
    try:
        yield capture
    finally:
        equilibrium_module.compute_magnetic_coordinates = original


def _loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cocos_in": args.cocos_in,
        "cocos_internal": args.cocos_internal,
        "phiclockwise_in": args.phiclockwise_in,
        "flux_normalization": args.flux_normalization,
    }


def _coordinate_kwargs(
    args: argparse.Namespace,
    variant: Variant,
    compute_method: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "coordinate_system": args.coordinate_system,
        "lpsi": args.lpsi,
        "ltheta": args.ltheta,
        "dr_hr": args.dr_hr,
        "dz_hz": args.dz_hz,
        "padding": args.padding,
        "ntht_pad": args.ntht_pad,
        "rhopol_min": args.rhopol_min,
        "rhopol_max": args.rhopol_max,
        "spectral_max_mode": args.spectral_max_mode,
        "radial_guard_surfaces": args.radial_guard_surfaces,
        "enforce_up_down_symmetry": args.enforce_up_down_symmetry,
        "symmetry_tolerance": args.symmetry_tolerance,
        "coordinate_accuracy": variant.accuracy,
        "projected_bridge_repair_strategy": (
            args.projected_bridge_repair_strategy
        ),
    }
    signature = inspect.signature(compute_method)
    if "n_theta_geom" in signature.parameters:
        # Keep the public build configuration and any content-addressed
        # provenance consistent with the quadrature actually benchmarked.
        # The temporary low-level wrapper below remains as a compatibility
        # fallback for older checkouts that do not expose this argument.
        kwargs["n_theta_geom"] = variant.n_theta_geom
    if args.defer_metrics:
        if "build_metric_cache" not in signature.parameters:
            raise RuntimeError(
                "--defer-metrics was requested, but this checkout does not "
                "expose build_metric_cache on EQDSK.compute_coordinates"
            )
        kwargs["build_metric_cache"] = False
    if args.map_only:
        if "materialize_rz" not in signature.parameters:
            raise RuntimeError(
                "--map-only was requested, but this checkout does not expose "
                "materialize_rz on EQDSK.compute_coordinates"
            )
        kwargs["materialize_rz"] = False
    return kwargs


def _exception_summary(error: BaseException) -> dict[str, Any]:
    output = {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }
    if hasattr(error, "diagnostics"):
        output["diagnostics"] = _jsonable(error.diagnostics)
    return output


def _run_variant(
    args: argparse.Namespace,
    variant: Variant,
) -> tuple[dict[str, Any], Any | None]:
    from pycocos import EQDSK

    repeat_records: list[dict[str, Any]] = []
    retained_result: Any | None = None
    retained_construction: Mapping[str, Any] = {}
    for repeat_index in range(args.repeat):
        if retained_result is not None:
            del retained_result
            retained_result = None
            gc.collect()

        load_start = time.perf_counter()
        try:
            equilibrium = EQDSK(args.eqdsk, **_loader_kwargs(args))
        except Exception as error:  # benchmark must preserve a JSON result
            repeat_records.append(
                {
                    "repeat": repeat_index + 1,
                    "status": "load_failed",
                    "eqdsk_load_seconds": time.perf_counter() - load_start,
                    "error": _exception_summary(error),
                }
            )
            break
        load_seconds = time.perf_counter() - load_start

        try:
            compute_kwargs = _coordinate_kwargs(
                args, variant, equilibrium.compute_coordinates
            )
        except Exception as error:
            repeat_records.append(
                {
                    "repeat": repeat_index + 1,
                    "status": "configuration_failed",
                    "eqdsk_load_seconds": load_seconds,
                    "error": _exception_summary(error),
                }
            )
            break

        gc.collect()
        construction_capture: Mapping[str, Any] = {}
        tracemalloc.start()
        construction_start = time.perf_counter()
        try:
            with _capture_coordinate_construction(
                variant.n_theta_geom
            ) as construction_capture:
                retained_result = equilibrium.compute_coordinates(
                    **compute_kwargs
                )
            status = "ok"
            error_summary = None
        except Exception as error:  # keep other variants measurable
            status = "compute_failed"
            error_summary = _exception_summary(error)
        construction_seconds = time.perf_counter() - construction_start
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        retained_construction = construction_capture

        record: dict[str, Any] = {
            "repeat": repeat_index + 1,
            "status": status,
            "eqdsk_load_seconds": load_seconds,
            "coordinate_seconds": construction_seconds,
            "tracemalloc_current_bytes": int(traced_current),
            "tracemalloc_peak_bytes": int(traced_peak),
        }
        if error_summary is not None:
            record["error"] = error_summary
        repeat_records.append(record)
        if status != "ok":
            retained_result = None
            break

    successful = [record for record in repeat_records if record["status"] == "ok"]
    variant_report: dict[str, Any] = {
        "variant": variant.as_dict(),
        "status": "ok" if len(successful) == args.repeat else "failed",
        "repetitions": repeat_records,
    }
    if successful and retained_result is not None:
        coordinate_seconds = [
            float(record["coordinate_seconds"]) for record in successful
        ]
        traced_peaks = [
            int(record["tracemalloc_peak_bytes"]) for record in successful
        ]
        variant_report["aggregate"] = {
            "successful_repetitions": len(successful),
            "coordinate_seconds_min": min(coordinate_seconds),
            "coordinate_seconds_median": statistics.median(coordinate_seconds),
            "coordinate_seconds_max": max(coordinate_seconds),
            "tracemalloc_peak_bytes_min": min(traced_peaks),
            "tracemalloc_peak_bytes_median": statistics.median(traced_peaks),
            "tracemalloc_peak_bytes_max": max(traced_peaks),
            "tracemalloc_scope": (
                "Python-managed allocations during coordinate construction; "
                "native NumPy/SciPy allocations may be incomplete"
            ),
        }
        if hasattr(retained_result, "coords"):
            variant_report["outputs"] = {
                "coords": _dataset_summary(retained_result.coords),
                "deriv": _dataset_summary(retained_result.deriv),
                "physics": _physics_summary(retained_result),
            }
        else:
            variant_report["outputs"] = {
                "map_product": {
                    "coordinate_system": retained_result.coordinate_system,
                    "rz_materialized": retained_result.rz_materialized,
                },
                "physics": _physics_summary(retained_result),
            }
        variant_report["diagnostics"] = _diagnostic_summary(
            retained_result, retained_construction
        )
    return variant_report, retained_result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"repository_root": str(_REPOSITORY_ROOT)}
    for key, arguments in (
        ("commit", ("rev-parse", "HEAD")),
        ("branch", ("branch", "--show-current")),
    ):
        try:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=_REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            metadata[key] = None
        else:
            metadata[key] = completed.stdout.strip() or None
    try:
        dirty = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        metadata["dirty"] = None
    else:
        metadata["dirty"] = bool(dirty.strip())
    return metadata


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "comparison": args.comparison,
        "coordinate_system": args.coordinate_system,
        "legacy_n_theta_geom": args.legacy_n_theta_geom,
        "repeat": args.repeat,
        "defer_metrics": args.defer_metrics,
        "map_only": args.map_only,
        "eqdsk_loader": _loader_kwargs(args),
        "coordinate_grid": {
            "lpsi": args.lpsi,
            "ltheta": args.ltheta,
            "dr_hr": args.dr_hr,
            "dz_hz": args.dz_hz,
            "padding": args.padding,
            "ntht_pad": args.ntht_pad,
            "rhopol_min": args.rhopol_min,
            "rhopol_max": args.rhopol_max,
            "spectral_max_mode": args.spectral_max_mode,
            "radial_guard_surfaces": args.radial_guard_surfaces,
        },
        "symmetry": {
            "enforce_up_down_symmetry": args.enforce_up_down_symmetry,
            "symmetry_tolerance": args.symmetry_tolerance,
            "projected_bridge_repair_strategy": (
                args.projected_bridge_repair_strategy
            ),
        },
        "delta_fields": list(args.delta_field or _DEFAULT_DELTA_FIELDS),
    }


def _emit(report: Mapping[str, Any], args: argparse.Namespace) -> None:
    serialized = json.dumps(
        report,
        indent=args.indent,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output is None:
        print(serialized)
        return
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.eqdsk = args.eqdsk.expanduser().resolve()
    if not args.eqdsk.is_file():
        parser.error(f"EQDSK file does not exist: {args.eqdsk}")

    selected_fields = tuple(args.delta_field or _DEFAULT_DELTA_FIELDS)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": {
            "path": str(args.eqdsk),
            "size_bytes": args.eqdsk.stat().st_size,
            "sha256": _sha256(args.eqdsk),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "pycocos": _package_version("pycocos"),
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            "git": _git_metadata(),
        },
        "configuration": _configuration(args),
        "variants": {},
        "comparisons": {},
    }

    snapshots: dict[str, dict[str, FieldSnapshot]] = {}
    any_failure = False
    for variant in _variant_plan(args):
        if not args.quiet:
            print(f"[pyCOCOS benchmark] running {variant.key}", file=sys.stderr)
        variant_report, result = _run_variant(args, variant)
        report["variants"][variant.key] = variant_report
        if variant_report["status"] != "ok":
            any_failure = True
            if args.fail_fast:
                break
            continue
        assert result is not None
        snapshots[variant.key] = _take_snapshot(result, selected_fields)
        del result
        gc.collect()

    for name, reference_key, candidate_key in _comparison_plan(args):
        comparison: dict[str, Any] = {
            "reference": reference_key,
            "candidate": candidate_key,
            "performance": _performance_comparison(
                report["variants"].get(reference_key, {}),
                report["variants"].get(candidate_key, {}),
            ),
        }
        if reference_key not in snapshots or candidate_key not in snapshots:
            comparison["status"] = "unavailable_due_to_failed_variant"
        else:
            comparison.update(
                _compare_snapshots(
                    snapshots[reference_key], snapshots[candidate_key]
                )
            )
        report["comparisons"][name] = comparison

    report["status"] = "failed" if any_failure else "ok"
    _emit(report, args)
    return 0 if (not any_failure or args.allow_failures) else 2


if __name__ == "__main__":
    raise SystemExit(main())
