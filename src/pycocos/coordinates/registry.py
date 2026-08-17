"""
Registry for coordinate-system Jacobian callables.
"""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable, Dict, Mapping, TypeAlias

from .jacobians import (
    compute_boozer_jacobian,
    compute_equal_arc_jacobian,
    compute_hamada_jacobian,
    compute_pest_jacobian,
)


JacobianCallable: TypeAlias = Callable[[Mapping[str, Any]], object]


def _require_context_callable(func: Callable) -> JacobianCallable:
    """Require the sole supported Jacobian API: ``func(context)``."""
    if not callable(func):
        raise TypeError("Jacobian callable must be callable.")

    try:
        parameters = tuple(signature(func).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Jacobian callable must have an inspectable signature "
            "of exactly func(context)."
        ) from exc

    if (
        len(parameters) != 1
        or parameters[0].kind
        not in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
    ):
        raise TypeError(
            "Jacobian callable must have exactly one positional parameter: "
            "func(context)."
        )

    return func


JACOBIAN_REGISTRY: Dict[str, JacobianCallable] = {
    "boozer": compute_boozer_jacobian,
    "hamada": compute_hamada_jacobian,
    "pest": compute_pest_jacobian,
    "equal_arc": compute_equal_arc_jacobian,
}
_BUILTIN_CACHE_VERSION = "builtin-v1"
_JACOBIAN_PERSISTENT_CACHE_VERSIONS: Dict[str, str | None] = {
    name: _BUILTIN_CACHE_VERSION for name in JACOBIAN_REGISTRY
}
_JACOBIAN_RUNTIME_CACHE_TOKENS: Dict[str, str] = {
    name: f"{name}:{_BUILTIN_CACHE_VERSION}" for name in JACOBIAN_REGISTRY
}
_JACOBIAN_TOKEN_CALLABLES: Dict[str, JacobianCallable] = dict(
    JACOBIAN_REGISTRY
)
_REGISTRATION_GENERATION = 0


def get_jacobian_function(name: str) -> JacobianCallable:
    """Get a context-based Jacobian function for a coordinate system."""
    name_lower = name.lower()
    if name_lower not in JACOBIAN_REGISTRY:
        available = ", ".join(JACOBIAN_REGISTRY.keys())
        raise ValueError(
            f"Unknown coordinate system '{name}'. "
            f"Available systems: {available}"
        )
    return _require_context_callable(JACOBIAN_REGISTRY[name_lower])


def register_coordinate_system(
    name: str,
    jacobian_func: Callable,
    *,
    cache_version: str | None = None,
) -> None:
    """
    Register a context-only coordinate-system Jacobian callable.

    The callback must have exactly the signature
    ``jacobian_func(context) -> J``. Jacobian output validation and
    normalization are performed by the coordinate assembly layer.

    ``cache_version`` is required only when persistent coordinate checkpoints
    will be used. It is an application-owned token that must change whenever
    the callable, its closure, or relevant module-global state changes.
    Registrations without a version remain safe for in-process reuse, but are
    deliberately ineligible for cross-process checkpoint reuse.
    """
    global _REGISTRATION_GENERATION

    normalized_name = name.lower()
    if cache_version is not None:
        if not isinstance(cache_version, str) or not cache_version.strip():
            raise ValueError("cache_version must be a nonempty string or None.")
        persistent_version = cache_version.strip()
    else:
        persistent_version = None
    validated = _require_context_callable(jacobian_func)
    _REGISTRATION_GENERATION += 1
    JACOBIAN_REGISTRY[normalized_name] = validated
    _JACOBIAN_PERSISTENT_CACHE_VERSIONS[normalized_name] = persistent_version
    _JACOBIAN_RUNTIME_CACHE_TOKENS[normalized_name] = (
        f"registration-{_REGISTRATION_GENERATION}"
    )
    _JACOBIAN_TOKEN_CALLABLES[normalized_name] = validated


def _get_jacobian_cache_identity(name: str) -> tuple[str, str | None]:
    """Return runtime and persistent cache identities for a registration."""

    normalized_name = name.lower()
    func = get_jacobian_function(normalized_name)
    if _JACOBIAN_TOKEN_CALLABLES.get(normalized_name) is not func:
        # Direct mutation of an existing public registry entry bypasses the
        # versioned registration API. It is safe for this process only and
        # must never inherit the replaced callable's persistent identity.
        return f"direct-registry-entry-{id(func)}", None
    runtime_token = _JACOBIAN_RUNTIME_CACHE_TOKENS.get(normalized_name)
    if runtime_token is None:
        # Direct mutation of the public registry is supported for historical
        # compatibility, but it cannot establish a cross-process identity.
        runtime_token = f"direct-registry-entry-{id(func)}"
    return (
        runtime_token,
        _JACOBIAN_PERSISTENT_CACHE_VERSIONS.get(normalized_name),
    )


def list_coordinate_systems() -> list:
    """List all available coordinate systems."""
    return list(JACOBIAN_REGISTRY.keys())
