from types import SimpleNamespace

import numpy as np
import xarray as xr

from pycocos.gui.app import list_plot_variables, resolve_plot_variable, sample_indices


def _make_fake_eq():
    r = xr.DataArray(np.linspace(1.0, 2.0, 4), dims=("R",), attrs={"units": "m"})
    z = xr.DataArray(np.linspace(-0.5, 0.5, 5), dims=("z",), attrs={"units": "m"})
    rho = xr.DataArray(np.linspace(0.0, 1.0, 6), dims=("rhop",), attrs={"units": ""})

    field = xr.Dataset(
        {
            "Br": xr.DataArray(np.zeros((4, 5)), dims=("R", "z"), coords={"R": r, "z": z}),
            "Bz": xr.DataArray(np.ones((4, 5)), dims=("R", "z"), coords={"R": r, "z": z}),
        }
    )
    flux = xr.Dataset(
        {
            "psi": xr.DataArray(np.zeros((4, 5)), dims=("R", "z"), coords={"R": r, "z": z}),
        }
    )
    profiles = xr.Dataset(
        {
            "q": xr.DataArray(np.linspace(1.0, 2.0, 6), dims=("rhop",), coords={"rhop": rho}),
        }
    )
    legacy_1d = {"legacy_q": profiles["q"]}
    legacy_2d = {"legacy_psi": flux["psi"]}

    fake = SimpleNamespace(
        field=field,
        flux=flux,
        profiles=profiles,
        plot_1d_names=legacy_1d,
        plot_2d_names=legacy_2d,
    )

    def _resolve(name):
        if name == "field.Br":
            return field["Br"], True
        if name == "profiles.q":
            return profiles["q"], False
        return None

    fake._resolve_plot_variable = _resolve  # noqa: SLF001
    return fake


def test_list_plot_variables_collects_structured_and_legacy():
    eq = _make_fake_eq()
    out = list_plot_variables(eq)
    assert "field.Br" in out["2d"]
    assert "flux.psi" in out["2d"]
    assert "legacy_psi" in out["2d"]
    assert "profiles.q" in out["1d"]
    assert "legacy_q" in out["1d"]


def test_resolve_plot_variable_falls_back_to_legacy():
    eq = _make_fake_eq()
    var, is_2d = resolve_plot_variable(eq, "legacy_psi")
    assert is_2d
    assert var.ndim == 2


def test_sample_indices_has_bounds_and_minimum():
    idx = sample_indices(20, 5)
    assert idx[0] == 0
    assert idx[-1] == 19
    assert len(idx) == 5

    idx_small = sample_indices(1, 10)
    assert np.array_equal(idx_small, np.array([0]))
