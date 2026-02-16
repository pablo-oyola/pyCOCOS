"""
Runtime guards for Numba execution in pycocos coordinate kernels.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

import numpy as np


_NUMBA_RUNTIME_READY = False
_NUMBA_RUNTIME_LOCK = threading.Lock()


def _running_under_pytest() -> bool:
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)


def _pyvista_pytest_plugin_loaded() -> bool:
    return "pytest_pyvista.pytest_pyvista" in sys.modules


def _known_crash_prone_pytest_combo() -> bool:
    """
    Detect known crash-prone environment:
      pytest + pytest-pyvista plugin + numba/llvmlite JIT compile.
    """
    return _running_under_pytest() and _pyvista_pytest_plugin_loaded()


def ensure_numba_runtime_ready(*, force: bool = False) -> None:
    """
    Fail-fast guard for Numba JIT runtime.

    Policy:
      - If Numba is unavailable or cannot compile a small probe kernel, raise
        RuntimeError with a clear message.
      - If a known crash-prone pytest plugin combination is detected, raise
        RuntimeError before triggering JIT compilation.
    """
    global _NUMBA_RUNTIME_READY

    if _NUMBA_RUNTIME_READY and not force:
        return

    with _NUMBA_RUNTIME_LOCK:
        if _NUMBA_RUNTIME_READY and not force:
            return

        if _known_crash_prone_pytest_combo() and os.environ.get("PYCOCOS_ALLOW_UNSAFE_NUMBA", "0") != "1":
            raise RuntimeError(
                "Detected pytest + pytest-pyvista plugin, a known crash-prone "
                "combination for numba/llvmlite JIT in this environment. "
                "Run tests with '-p no:pyvista' or set "
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. "
                "To bypass this guard (unsafe), set PYCOCOS_ALLOW_UNSAFE_NUMBA=1."
            )

        try:
            import numba as nb
        except Exception as exc:
            raise RuntimeError(
                "Numba is required for pycocos Jacobian hot paths but could not be imported."
            ) from exc

        try:
            @nb.njit
            def _probe(x: np.ndarray) -> np.ndarray:
                return x + 1.0

            _probe(np.ones(2, dtype=np.float64))
        except Exception as exc:
            raise RuntimeError(
                "Numba JIT probe failed. pycocos does not fall back to Python loops "
                "for heavy Jacobian kernels; fix numba/llvmlite environment first."
            ) from exc

        _NUMBA_RUNTIME_READY = True


def _reset_numba_runtime_guard_for_tests() -> None:
    """
    Test helper to reset one-time runtime guard state.
    """
    global _NUMBA_RUNTIME_READY
    with _NUMBA_RUNTIME_LOCK:
        _NUMBA_RUNTIME_READY = False

