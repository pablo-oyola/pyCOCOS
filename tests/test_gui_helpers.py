from types import SimpleNamespace

import numpy as np
import xarray as xr

from pycocos.gui.app import (
    EquilibriumGuiApp,
    list_plot_variables,
    resolve_plot_variable,
    sample_indices,
)


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


def test_set_or_update_colorbar_reuses_existing_axis():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)
    app.figure = fig
    app.ax = ax
    app._active_colorbar = None

    var = xr.DataArray(
        np.arange(16, dtype=float).reshape(4, 4),
        dims=("R", "z"),
        attrs={"name": "B", "short_name": "B", "units": "T"},
    )

    image1 = ax.imshow(var.values)
    app._set_or_update_colorbar(image1, var)  # noqa: SLF001 - direct regression coverage
    axes_count_after_first = len(fig.axes)
    assert app._active_colorbar is not None

    ax.clear()
    image2 = ax.imshow(var.values * 2.0)
    app._set_or_update_colorbar(image2, var)  # noqa: SLF001 - direct regression coverage
    axes_count_after_second = len(fig.axes)

    assert axes_count_after_second == axes_count_after_first
    plt.close(fig)


def test_set_plot_aspect_switches_between_2d_and_1d():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)
    app.ax = ax

    app._set_plot_aspect(is_2d=True)  # noqa: SLF001 - direct regression coverage
    assert ax.get_aspect() == 1.0

    app._set_plot_aspect(is_2d=False)  # noqa: SLF001 - direct regression coverage
    assert ax.get_aspect() == "auto"
    plt.close(fig)


def test_get_colormap_returns_colormap_instance():
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)
    cmap = app._get_colormap("tab10")  # noqa: SLF001 - direct regression coverage
    rgba = cmap(0)
    assert hasattr(cmap, "__call__")
    assert len(rgba) == 4


def test_get_overlay_grids_uses_transform_inverse_when_available():
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)

    psi0 = xr.DataArray(np.linspace(0.1, 0.9, 8), dims=("psi0",))
    theta_star = xr.DataArray(np.linspace(-0.1, 2.0 * np.pi + 0.1, 20), dims=("theta_star",))
    r_inv = xr.DataArray(np.zeros((8, 20)), dims=("psi0", "theta_star"), coords={"psi0": psi0, "theta_star": theta_star})
    z_inv = xr.DataArray(np.zeros((8, 20)), dims=("psi0", "theta_star"), coords={"psi0": psi0, "theta_star": theta_star})

    class _FakeCoords:
        def __init__(self):
            self.coords = xr.Dataset({"R_inv": r_inv, "z_inv": z_inv})
            self.nthtpad = 2

        def transform_inverse(self, psi, thetamag, grid=True):
            rr = np.tile(np.linspace(1.2, 1.8, thetamag.size), (psi.size, 1))
            zz = np.tile(np.linspace(-0.2, 0.2, thetamag.size), (psi.size, 1))
            return xr.Dataset(
                {
                    "R_inv": xr.DataArray(rr, dims=("psi", "thetamag")),
                    "z_inv": xr.DataArray(zz, dims=("psi", "thetamag")),
                }
            )

    coords = _FakeCoords()
    r_plot, z_plot = app._get_overlay_grids(coords, n_surf=4)  # noqa: SLF001 - direct regression coverage
    assert r_plot.shape[0] == 4
    assert z_plot.shape == r_plot.shape
    assert np.isfinite(r_plot).all()
    assert np.isfinite(z_plot).all()


def test_get_overlay_grids_falls_back_to_cached_tables():
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)

    psi0 = xr.DataArray(np.linspace(0.1, 0.9, 6), dims=("psi0",))
    theta_star = xr.DataArray(np.linspace(-0.2, 2.0 * np.pi + 0.2, 14), dims=("theta_star",))
    rr = np.tile(np.linspace(1.1, 1.9, 14), (6, 1))
    zz = np.tile(np.linspace(-0.3, 0.3, 14), (6, 1))
    r_inv = xr.DataArray(rr, dims=("psi0", "theta_star"), coords={"psi0": psi0, "theta_star": theta_star})
    z_inv = xr.DataArray(zz, dims=("psi0", "theta_star"), coords={"psi0": psi0, "theta_star": theta_star})

    class _FakeCoords:
        def __init__(self):
            self.coords = xr.Dataset({"R_inv": r_inv, "z_inv": z_inv})
            self.nthtpad = 2

        def transform_inverse(self, psi, thetamag, grid=True):
            raise RuntimeError("force fallback")

    coords = _FakeCoords()
    r_plot, z_plot = app._get_overlay_grids(coords, n_surf=3)  # noqa: SLF001 - direct regression coverage
    assert r_plot.shape[0] == 3
    # Fallback strips pad from 14 -> 10
    assert r_plot.shape[1] == 10
    assert z_plot.shape == r_plot.shape


def test_overlay_domain_mask_filters_outliers_but_keeps_finite_fallback():
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)
    app.eq = SimpleNamespace(
        Rgrid=xr.DataArray(np.linspace(1.0, 2.0, 5), dims=("R",)),
        zgrid=xr.DataArray(np.linspace(-0.5, 0.5, 5), dims=("z",)),
    )

    r_line = np.array([1.2, 1.5, 80.0])
    z_line = np.array([0.0, 0.1, 80.0])
    mask = app._overlay_domain_mask(r_line, z_line)  # noqa: SLF001 - direct regression coverage
    assert np.array_equal(mask, np.array([True, True, False]))

    # If every point is outside the expected domain, helper falls back to finite points.
    r_far = np.array([70.0, 71.0, np.nan])
    z_far = np.array([80.0, 81.0, np.nan])
    fallback_mask = app._overlay_domain_mask(r_far, z_far)  # noqa: SLF001 - direct regression coverage
    assert np.array_equal(fallback_mask, np.array([True, True, False]))


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def test_get_rhopol_window_valid_and_clamped():
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)
    app.rhopol_min_var = _FakeVar("0.0")
    app.rhopol_max_var = _FakeVar("1.0")

    rho_min, rho_max = app._get_rhopol_window()  # noqa: SLF001 - direct regression coverage
    assert rho_min > 0.0
    assert rho_max < 1.0
    assert rho_min < rho_max


def test_get_rhopol_window_rejects_invalid_window():
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)
    app.rhopol_min_var = _FakeVar("0.8")
    app.rhopol_max_var = _FakeVar("0.2")

    try:
        app._get_rhopol_window()  # noqa: SLF001 - direct regression coverage
    except ValueError as exc:
        assert "min < max" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid rhopol window.")


def test_coord_settings_match_uses_cached_values():
    app = EquilibriumGuiApp.__new__(EquilibriumGuiApp)
    app._computed_coord_settings = {"boozer": (128, 256, 0.1, 0.9)}

    assert app._coord_settings_match("boozer", 128, 256, 0.1, 0.9)  # noqa: SLF001
    assert not app._coord_settings_match("boozer", 129, 256, 0.1, 0.9)  # noqa: SLF001
    assert not app._coord_settings_match("pest", 128, 256, 0.1, 0.9)  # noqa: SLF001
