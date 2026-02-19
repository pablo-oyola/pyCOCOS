import numpy as np
import xarray as xr
from types import SimpleNamespace
import importlib

from pycocos.io import eqdsk as eqdsk_mod

equilibrium_mod = importlib.import_module("pycocos.core.equilibrium")


def test_read_eqdsk_smoke(monkeypatch, tmp_path):
    nx = 4
    ny = 3
    fake_geqdsk = {
        "comment": "fake equilibrium",
        "bcentr": 2.0,
        "cpasma": 1.0e6,
        "nx": nx,
        "ny": ny,
        "rbdry": np.array([1.2, 1.5, 1.8]),
        "zbdry": np.array([0.0, 0.2, 0.0]),
        "rmagx": 1.4,
        "zmagx": 0.0,
        "psi": np.linspace(0.0, 1.0, nx * ny).reshape(nx, ny),
        "simagx": 0.0,
        "sibdry": 1.0,
        "fpol": np.linspace(2.0, 1.6, nx),
        "pres": np.linspace(2e3, 0.0, nx),
        "ffprime": np.zeros(nx),
        "pprime": np.zeros(nx),
        "qpsi": np.linspace(1.0, 2.0, nx),
        "rleft": 1.0,
        "rdim": 1.0,
        "zmid": 0.0,
        "zdim": 1.0,
    }

    monkeypatch.setattr(eqdsk_mod.freeqdsk.geqdsk, "read", lambda _f: fake_geqdsk)
    monkeypatch.setattr(eqdsk_mod, "assign_cocos", lambda *args, **kwargs: 1)
    monkeypatch.setattr(eqdsk_mod, "fromCocosNtoCocosM", lambda data, _: data)

    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")

    out = eqdsk_mod.read_eqdsk(str(file_path), cocos=1)
    assert out["lr"] == nx
    assert out["lz"] == ny
    assert out["Rgrid"].shape == (nx,)
    assert out["zgrid"].shape == (ny,)
    assert out["fpolrz"].shape == (nx, ny)
    assert out["cocos_in"] == 1
    assert out["cocos_out"] == 1


def _make_fake_eq_instance(monkeypatch, tmp_path):
    nx = 64
    ny = 64
    raxis = 1.5
    zaxis = 0.0
    rgrid = np.linspace(1.0, 2.0, nx)
    zgrid = np.linspace(-0.5, 0.5, ny)
    rr, zz = np.meshgrid(rgrid, zgrid, indexing="ij")

    # Circular-ish flux surfaces around axis to ensure a valid LCFS contour.
    psi = (rr - raxis) ** 2 + (zz - zaxis) ** 2
    psi_ax = 0.0
    psi_bdy = 0.16

    fake_gdata = {
        "Rgrid": rgrid,
        "zgrid": zgrid,
        "Raxis": raxis,
        "zaxis": zaxis,
        "psi": psi,
        "psi_ax": psi_ax,
        "psi_bdy": psi_bdy,
        "psimax": psi_bdy - psi_ax,
        "lr": nx,
        "lz": ny,
        "fpol": np.linspace(2.0, 1.8, nx),
        "prs": np.linspace(2.0e3, 0.0, nx),
        "ffp": np.zeros(nx),
        "pprime": np.zeros(nx),
        "q": np.linspace(1.0, 2.0, nx),
    }
    fake_bfield = {
        "br": np.zeros((nx, ny)),
        "bz": np.zeros((nx, ny)),
        "bphi": np.full((nx, ny), 2.0),
        "babs": np.full((nx, ny), 2.0),
        "btht": np.zeros((nx, ny)),
    }

    monkeypatch.setattr(eqdsk_mod, "read_eqdsk", lambda *args, **kwargs: fake_gdata)
    monkeypatch.setattr(eqdsk_mod, "eqdsk2magnetic", lambda *_args, **_kwargs: fake_bfield)
    monkeypatch.setattr(eqdsk_mod, "assign_cocos", lambda *args, **kwargs: 1)
    monkeypatch.setattr(eqdsk_mod, "fromCocosNtoCocosM", lambda data, _: data)
    monkeypatch.setattr(
        eqdsk_mod.freeqdsk.geqdsk,
        "read",
        lambda _f: {"qpsi": [1.0], "cpasma": 1.0e6, "bcentr": 2.0, "simagx": 0.0, "sibdry": 1.0},
    )

    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")

    return eqdsk_mod.eqdsk(str(file_path))


def test_eqdsk_class_init_boundary_property_regression(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    assert "R" in eq.boundary
    assert "z" in eq.boundary
    assert eq.boundary.R.size > 10


def test_build_magnetic_coordinates_dataset_has_expected_coordinate_names(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)

    npsi = 12
    ltheta = 32
    ntht_pad = 3
    theta = np.linspace(0.0, 2.0 * np.pi, ltheta)
    psi0 = float(eq.geometry.attrs["psi_ax"]) + 0.01
    psi1 = float(eq.geometry.attrs["psi_bdy"]) - 0.01
    psigrid = np.linspace(psi0, psi1, npsi)

    thtable = np.tile(theta, (npsi, 1))
    nutable = np.zeros((npsi, ltheta))
    jac = np.ones((npsi, ltheta))
    Rtransform = np.tile(1.5 + 0.1 * np.cos(theta), (npsi, 1))
    ztransform = np.tile(0.0 + 0.1 * np.sin(theta), (npsi, 1))
    qprof = np.linspace(1.0, 2.0, npsi)
    Fprof = np.linspace(2.0, 1.8, npsi)
    Iprof = np.linspace(1.0e6, 1.1e6, npsi)

    mag = eq._build_magnetic_coordinates_dataset(  # noqa: SLF001 - regression coverage for builder output
        psigrid=psigrid,
        thtable=thtable,
        nutable=nutable,
        jac=jac,
        Rtransform=Rtransform,
        ztransform=ztransform,
        R_fine=eq.Rgrid.values,
        z_fine=eq.zgrid.values,
        qprof=qprof,
        Fprof=Fprof,
        Iprof=Iprof,
        ntht_pad=ntht_pad,
        coordinate_system="boozer",
    )

    assert "psi0" in mag.coords.coords
    assert "thetageom" in mag.coords.coords
    assert mag.coords["theta"].dims == ("psi0", "thetageom")
    assert mag.coords["nu"].dims == ("psi0", "thetageom")


def test_plot2d_var_transposes_data_for_rz_layout(monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    raw = np.arange(eq.Rgrid.size * eq.zgrid.size, dtype=float).reshape(eq.Rgrid.size, eq.zgrid.size)
    var = xr.DataArray(
        raw,
        dims=("R", "z"),
        coords={"R": eq.Rgrid, "z": eq.zgrid},
        attrs={"name": "orientation_test", "short_name": "orientation_test", "units": ""},
    )

    fig, ax = plt.subplots()
    _, image = eq.plot2d_var(var, name="orientation_test", ax=ax, put_labels=False)
    plotted = np.asarray(image.get_array())
    assert plotted.shape == raw.T.shape
    assert np.array_equal(plotted, raw.T)
    plt.close(fig)


def test_resolve_plot_variable_marks_profiles_as_1d(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    var, is_2d = eq._resolve_plot_variable("profiles.q")  # noqa: SLF001 - regression coverage for resolver
    assert var.ndim == 1
    assert not is_2d


def test_plot_profiles_1d_returns_line_artist(monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    fig, ax = plt.subplots()
    _, line = eq.plot("profiles.q", ax=ax, put_labels=True)
    assert len(ax.lines) >= 1
    assert hasattr(line, "get_xdata")
    assert len(line.get_xdata()) == eq.profiles["q"].size
    plt.close(fig)


def test_compute_coordinates_rhopol_window_maps_to_expected_psi(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    captured = {}

    def _fake_compute_magnetic_coordinates(*args, **kwargs):
        psigrid = np.asarray(kwargs["psigrid"], dtype=float)
        ltheta = int(kwargs["ltheta"])
        npsi = psigrid.size
        captured["psigrid"] = psigrid.copy()
        qprof = np.ones(npsi)
        Fprof = np.ones(npsi)
        Iprof = np.ones(npsi)
        thtable = np.zeros((npsi, ltheta))
        nutable = np.zeros((npsi, ltheta))
        jac = np.ones((npsi, ltheta))
        Rtransform = np.tile(np.linspace(1.2, 1.8, ltheta), (npsi, 1))
        ztransform = np.tile(np.linspace(-0.2, 0.2, ltheta), (npsi, 1))
        return qprof, Fprof, Iprof, thtable, nutable, jac, Rtransform, ztransform

    monkeypatch.setattr(equilibrium_mod, "compute_magnetic_coordinates", _fake_compute_magnetic_coordinates)
    monkeypatch.setattr(
        eq,
        "_build_magnetic_coordinates_dataset",
        lambda *args, **kwargs: SimpleNamespace(dummy=True),
    )

    out = eq.compute_coordinates(
        coordinate_system="boozer",
        lpsi=9,
        ltheta=24,
        rhopol_min=0.2,
        rhopol_max=0.8,
    )

    assert getattr(out, "dummy", False)
    psi_axis = float(eq.geometry.attrs["psi_ax"])
    psi_edge = float(eq.geometry.attrs["psi_bdy"])
    expected_psi_start = psi_axis + (0.2**2) * (psi_edge - psi_axis)
    expected_psi_end = psi_axis + (0.8**2) * (psi_edge - psi_axis)
    assert np.isclose(captured["psigrid"][0], expected_psi_start)
    assert np.isclose(captured["psigrid"][-1], expected_psi_end)


def test_compute_curvature_vector_toroidal_field_limit(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)

    curvature = eq.make_curvature(use_numba=False)

    expected_kappa_r = -1.0 / eq.Rgrid.values[:, None]
    assert np.allclose(curvature["kappa_R"].values, expected_kappa_r, rtol=1.0e-7, atol=1.0e-9)
    assert np.allclose(curvature["kappa_phi"].values, 0.0, atol=1.0e-10)
    assert np.allclose(curvature["kappa_z"].values, 0.0, atol=1.0e-10)
    assert hasattr(eq, "Kdata")
    assert "kappa_R" in eq.Kdata
    assert "kappa_R" in eq.curvature
    assert hasattr(eq, "curvaturedata")


def test_compute_curvature_vector_numba_matches_findiff(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)

    reference = eq.compute_curvature_vector(use_numba=False, cache=False)
    numba_out = eq.compute_curvature_vector(use_numba=True, cache=False)

    assert np.allclose(numba_out["kappa_R"].values, reference["kappa_R"].values, atol=1.0e-9)
    assert np.allclose(numba_out["kappa_phi"].values, reference["kappa_phi"].values, atol=1.0e-9)
    assert np.allclose(numba_out["kappa_z"].values, reference["kappa_z"].values, atol=1.0e-9)


def test_curvature_variables_are_resolvable_for_plotting(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    eq.make_curvature(use_numba=False)

    resolved = eq._resolve_plot_variable("curvature.kappa_R")  # noqa: SLF001 - resolver regression coverage
    assert resolved is not None
    var, is_2d = resolved
    assert var.ndim == 2
    assert is_2d

