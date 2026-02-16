import sys
import types

import pytest

from pycocos.coordinates import numba_runtime


def test_numba_runtime_guard_blocks_known_pytest_pyvista_combo(monkeypatch):
    numba_runtime._reset_numba_runtime_guard_for_tests()
    fake_plugin = types.ModuleType("pytest_pyvista.pytest_pyvista")
    monkeypatch.setitem(sys.modules, "pytest_pyvista.pytest_pyvista", fake_plugin)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake::test")
    monkeypatch.delenv("PYCOCOS_ALLOW_UNSAFE_NUMBA", raising=False)

    with pytest.raises(RuntimeError, match="pytest-pyvista"):
        numba_runtime.ensure_numba_runtime_ready(force=True)


def test_numba_runtime_guard_can_be_overridden(monkeypatch):
    numba_runtime._reset_numba_runtime_guard_for_tests()
    fake_plugin = types.ModuleType("pytest_pyvista.pytest_pyvista")
    monkeypatch.setitem(sys.modules, "pytest_pyvista.pytest_pyvista", fake_plugin)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake::test")
    monkeypatch.setenv("PYCOCOS_ALLOW_UNSAFE_NUMBA", "1")

    # Should not raise when explicit unsafe override is requested.
    numba_runtime.ensure_numba_runtime_ready(force=True)

