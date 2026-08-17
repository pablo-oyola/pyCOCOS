import inspect
import warnings
import numpy as np
import pytest
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

    conversion_calls = []

    def _convert(data, cocos_m, cocos_n):
        conversion_calls.append((cocos_n, cocos_m))
        return data

    monkeypatch.setattr(eqdsk_mod.freeqdsk.geqdsk, "read", lambda _f: fake_geqdsk)
    monkeypatch.setattr(
        eqdsk_mod,
        "identify_cocos",
        lambda *args, **kwargs: pytest.fail("explicit cocos_in must bypass detection"),
    )
    monkeypatch.setattr(eqdsk_mod, "fromCocosNtoCocosM", _convert)

    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")

    out = eqdsk_mod.read_eqdsk(str(file_path), cocos_in=1)
    assert out["lr"] == nx
    assert out["lz"] == ny
    assert out["Rgrid"].shape == (nx,)
    assert out["zgrid"].shape == (ny,)
    assert out["fpolrz"].shape == (nx, ny)
    assert out["cocos_input"] == 1
    assert out["cocos_internal"] == 1
    assert out["phiclockwise_input"] is False
    assert out["phiclockwise_internal"] is False
    assert out["flux_normalization_input"] == "Wb/rad"
    assert out["flux_normalization_internal"] == "Wb/rad"
    assert conversion_calls == [(1, 1)]


def test_read_eqdsk_detects_once_and_passes_explicit_source(monkeypatch, tmp_path):
    nx = 4
    ny = 3
    raw = {
        "comment": "fake equilibrium",
        "bcentr": 2.0,
        "cpasma": 1.0e6,
        "nx": nx,
        "ny": ny,
        "rbdry": np.array([1.2, 1.5, 1.8]),
        "zbdry": np.array([0.0, 0.2, 0.0]),
        "rmagx": 1.4,
        "zmagx": 0.0,
        "psi": np.linspace(0.0, 2.0 * np.pi, nx * ny).reshape(nx, ny),
        "simagx": 0.0,
        "sibdry": 2.0 * np.pi,
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
    detection_calls = []
    conversion_calls = []

    def _identify(*args, **kwargs):
        detection_calls.append((args, kwargs))
        return SimpleNamespace(
            is_unique=True,
            candidates=(11,),
            require_unique=lambda: 11,
        )

    def _convert(data, cocos_m, cocos_n):
        conversion_calls.append((cocos_n, cocos_m))
        return data

    monkeypatch.setattr(eqdsk_mod.freeqdsk.geqdsk, "read", lambda _f: raw)
    monkeypatch.setattr(eqdsk_mod, "identify_cocos", _identify)
    monkeypatch.setattr(eqdsk_mod, "fromCocosNtoCocosM", _convert)

    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")
    out = eqdsk_mod.read_eqdsk(
        str(file_path),
        cocos_internal=12,
        phiclockwise_in=False,
        flux_normalization="Wb",
    )

    assert len(detection_calls) == 1
    assert detection_calls[0][1]["phiclockwise"] is False
    assert detection_calls[0][1]["flux_normalization"] == "Wb"
    assert conversion_calls == [(11, 12)]
    assert out["cocos_input"] == 11
    assert out["cocos_internal"] == 12
    assert out["phiclockwise_internal"] is True
    assert out["flux_normalization_internal"] == "Wb"


def test_read_eqdsk_converts_explicit_cocos_11_to_internal_cocos_1(
    monkeypatch,
    tmp_path,
):
    nx = 4
    ny = 3
    cocos_1_raw = {
        "comment": "fake equilibrium",
        "bcentr": 2.0,
        "cpasma": 1.0e6,
        "nx": nx,
        "ny": ny,
        "rbdry": np.array([1.2, 1.5, 1.8]),
        "zbdry": np.array([0.0, 0.2, 0.0]),
        "rlim": np.array([1.0, 2.0]),
        "zlim": np.array([-0.5, 0.5]),
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
        "rcentr": 1.5,
        "zmid": 0.0,
        "zdim": 1.0,
    }
    cocos_11_raw = eqdsk_mod.fromCocosNtoCocosM(
        cocos_1_raw,
        cocos_m=11,
        cocos_n=1,
    )
    monkeypatch.setattr(
        eqdsk_mod.freeqdsk.geqdsk,
        "read",
        lambda _f: cocos_11_raw,
    )
    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")

    out = eqdsk_mod.read_eqdsk(str(file_path), cocos_in=11)

    np.testing.assert_allclose(out["psi"], cocos_1_raw["psi"])
    np.testing.assert_allclose(out["q"], cocos_1_raw["qpsi"])
    assert out["psi_ax"] == pytest.approx(cocos_1_raw["simagx"])
    assert out["psi_bdy"] == pytest.approx(cocos_1_raw["sibdry"])
    assert out["cocos_input"] == 11
    assert out["cocos_internal"] == 1
    assert out["flux_normalization_input"] == "Wb"
    assert out["flux_normalization_internal"] == "Wb/rad"

    expected_12 = eqdsk_mod.fromCocosNtoCocosM(
        cocos_11_raw,
        cocos_m=12,
        cocos_n=11,
    )
    out_12 = eqdsk_mod.read_eqdsk(
        str(file_path),
        cocos_in=11,
        cocos_internal=12,
    )
    np.testing.assert_allclose(out_12["psi"], expected_12["psi"])
    np.testing.assert_allclose(out_12["q"], expected_12["qpsi"])
    assert out_12["cocos_internal"] == 12
    assert out_12["phiclockwise_internal"] is True
    assert out_12["flux_normalization_internal"] == "Wb"


def test_read_eqdsk_rejects_ambiguous_flux_normalization(monkeypatch, tmp_path):
    raw = {
        "qpsi": np.array([1.0]),
        "cpasma": 1.0e6,
        "bcentr": 2.0,
        "simagx": 0.0,
        "sibdry": 1.0,
    }
    monkeypatch.setattr(eqdsk_mod.freeqdsk.geqdsk, "read", lambda _f: raw)
    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"candidates are \(1, 11\)"):
        eqdsk_mod.read_eqdsk(str(file_path), phiclockwise_in=False)


def test_read_eqdsk_rejects_input_metadata_conflicting_with_explicit_cocos(
    monkeypatch,
    tmp_path,
):
    raw = {
        "qpsi": np.array([1.0]),
        "cpasma": 1.0e6,
        "bcentr": 2.0,
        "simagx": 0.0,
        "sibdry": 1.0,
    }
    monkeypatch.setattr(eqdsk_mod.freeqdsk.geqdsk, "read", lambda _f: raw)
    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires phiclockwise_in=False"):
        eqdsk_mod.read_eqdsk(
            str(file_path),
            cocos_in=1,
            phiclockwise_in=True,
        )
    with pytest.raises(ValueError, match="requires flux_normalization='Wb/rad'"):
        eqdsk_mod.read_eqdsk(
            str(file_path),
            cocos_in=1,
            flux_normalization="Wb",
        )


def test_detection_uses_first_finite_nonzero_q_and_rejects_mixed_signs():
    raw = {
        "qpsi": np.array([0.0, np.nan, 1.5, 2.0]),
        "cpasma": 1.0e6,
        "bcentr": 2.0,
        "simagx": 0.0,
        "sibdry": 1.0,
    }
    convention = eqdsk_mod._resolve_loader_input_convention(
        raw,
        cocos_in=None,
        cocos_internal=1,
        phiclockwise_in=False,
        flux_normalization="Wb/rad",
    )
    assert convention["cocos_input"] == 1

    raw["qpsi"] = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match="mixed nonzero signs"):
        eqdsk_mod._resolve_loader_input_convention(
            raw,
            cocos_in=None,
            cocos_internal=1,
            phiclockwise_in=False,
            flux_normalization="Wb/rad",
        )


def test_loader_rejects_invalid_internal_cocos_before_conversion():
    raw = {
        "qpsi": np.array([1.0]),
        "cpasma": 1.0e6,
        "bcentr": 2.0,
        "simagx": 0.0,
        "sibdry": 1.0,
    }
    with pytest.raises(ValueError, match="COCOS = 9 does not exist"):
        eqdsk_mod._resolve_loader_input_convention(
            raw,
            cocos_in=1,
            cocos_internal=9,
            phiclockwise_in=None,
            flux_normalization=None,
        )


def test_loader_api_has_no_legacy_arguments_or_keyword_separator():
    expected = (
        "filename",
        "cocos_in",
        "cocos_internal",
        "phiclockwise_in",
        "flux_normalization",
    )
    read_signature = inspect.signature(eqdsk_mod.read_eqdsk)
    init_signature = inspect.signature(eqdsk_mod.eqdsk.__init__)
    load_signature = inspect.signature(eqdsk_mod.eqdsk.load)
    assert tuple(read_signature.parameters) == expected
    assert tuple(init_signature.parameters) == (
        "self",
        "fn",
        *expected[1:],
    )
    assert tuple(load_signature.parameters) == expected
    for signature in (read_signature, init_signature, load_signature):
        assert all(
            parameter.kind is not inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )
    assert not hasattr(eqdsk_mod, "read_eqdsk_2")


def test_eqdsk2magnetic_applies_full_flux_two_pi_normalization():
    rgrid = np.linspace(1.0, 2.0, 9)
    zgrid = np.linspace(-0.4, 0.4, 9)
    rr, zz = np.meshgrid(rgrid, zgrid, indexing="ij")
    psi_per_radian = rr**2 + 3.0 * zz
    fpolrz = np.full_like(psi_per_radian, 2.0)
    common = {
        "Rgrid": rgrid,
        "zgrid": zgrid,
        "fpolrz": fpolrz,
        "q": np.ones(rgrid.size),
    }

    cocos_1 = eqdsk_mod.eqdsk2magnetic(
        {**common, "psi": psi_per_radian, "cocos_internal": 1}
    )
    cocos_11 = eqdsk_mod.eqdsk2magnetic(
        {**common, "psi": 2.0 * np.pi * psi_per_radian, "cocos_internal": 11}
    )

    expected_br = np.broadcast_to(3.0 / rgrid[:, None], psi_per_radian.shape)
    expected_bz = np.full_like(psi_per_radian, -2.0)
    np.testing.assert_allclose(cocos_1["br"], expected_br, atol=1e-12)
    np.testing.assert_allclose(cocos_1["bz"], expected_bz, atol=1e-12)
    np.testing.assert_allclose(cocos_11["br"], cocos_1["br"], atol=1e-12)
    np.testing.assert_allclose(cocos_11["bz"], cocos_1["bz"], atol=1e-12)


def _make_fake_eq_instance(
    monkeypatch,
    tmp_path,
    cocos_input=1,
    cocos_internal=1,
    include_boundary=True,
):
    nx = 64
    ny = 64
    raxis = 1.5
    zaxis = 0.0
    rgrid = np.linspace(1.0, 2.0, nx)
    zgrid = np.linspace(-0.5, 0.5, ny)
    rr, zz = np.meshgrid(rgrid, zgrid, indexing="ij")

    # Circular-ish flux surfaces around axis to ensure a valid LCFS contour.
    psi_cocos_1 = (rr - raxis) ** 2 + (zz - zaxis) ** 2
    psi_ax = 0.0

    input_descriptor = eqdsk_mod.get_cocos(cocos_input)
    internal_descriptor = eqdsk_mod.get_cocos(cocos_internal)
    psi_scale = (
        internal_descriptor.sigma_RpZ
        * internal_descriptor.sigma_Bp
        * (2.0 * np.pi) ** internal_descriptor.exp_Bp
    )
    derivative_scale = (
        internal_descriptor.sigma_RpZ
        * internal_descriptor.sigma_Bp
        / (2.0 * np.pi) ** internal_descriptor.exp_Bp
    )
    psi = psi_scale * psi_cocos_1
    psi_bdy = psi_scale * 0.16
    theta = np.linspace(0.0, 2.0 * np.pi, 65)[:-1]
    fake_gdata = {
        "Rgrid": rgrid,
        "zgrid": zgrid,
        "Raxis": raxis,
        "zaxis": zaxis,
        "rcentr": 1.5,
        "Bcenter": 2.0 * internal_descriptor.sigma_RpZ,
        "Ip": 1.0e6 * internal_descriptor.sigma_RpZ,
        "psi": psi,
        "psi_ax": psi_ax,
        "psi_bdy": psi_bdy,
        "psimax": psi_bdy - psi_ax,
        "lr": nx,
        "lz": ny,
        "fpol": (
            internal_descriptor.sigma_RpZ
            * np.linspace(2.0, 1.8, nx)
        ),
        "prs": np.linspace(2.0e3, 0.0, nx),
        "ffp": derivative_scale * np.linspace(-0.2, -0.1, nx),
        "pprime": derivative_scale * np.linspace(-2.0e3, -1.0e3, nx),
        "q": internal_descriptor.sigma_rhotp * np.linspace(1.0, 2.0, nx),
        "r_bdy": raxis + 0.4 * np.cos(theta),
        "z_bdy": zaxis + 0.4 * np.sin(theta),
        "rlim": raxis + 0.48 * np.cos(theta),
        "zlim": zaxis + 0.48 * np.sin(theta),
        "cocos_input": cocos_input,
        "cocos_internal": cocos_internal,
        "phiclockwise_input": input_descriptor.phiclockwise,
        "phiclockwise_internal": internal_descriptor.phiclockwise,
        "flux_normalization_input": input_descriptor.flux_normalization,
        "flux_normalization_internal": internal_descriptor.flux_normalization,
    }
    if not include_boundary:
        fake_gdata.pop("r_bdy")
        fake_gdata.pop("z_bdy")
    fake_bfield = {
        "br": np.zeros((nx, ny)),
        "bz": np.zeros((nx, ny)),
        "bphi": np.full((nx, ny), 2.0 * internal_descriptor.sigma_RpZ),
        "babs": np.full((nx, ny), 2.0),
        "btht": np.zeros((nx, ny)),
    }

    monkeypatch.setattr(eqdsk_mod, "read_eqdsk", lambda *args, **kwargs: fake_gdata)
    monkeypatch.setattr(eqdsk_mod, "eqdsk2magnetic", lambda *_args, **_kwargs: fake_bfield)

    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")

    return eqdsk_mod.eqdsk(
        str(file_path),
        cocos_in=cocos_input,
        cocos_internal=cocos_internal,
    )


def test_eqdsk_class_init_boundary_property_regression(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    assert "R" in eq.boundary
    assert "z" in eq.boundary
    assert eq.boundary.attrs["source"] == "supplied"
    np.testing.assert_array_equal(eq.boundary.R, eq._gdata["r_bdy"])
    np.testing.assert_array_equal(eq.boundary.z, eq._gdata["z_bdy"])
    np.testing.assert_array_equal(eq.geometry.R_boundary, eq._gdata["r_bdy"])
    np.testing.assert_array_equal(eq.geometry.z_boundary, eq._gdata["z_bdy"])


def test_eqdsk_supplied_boundary_bypasses_private_flux_contour_selection(
    monkeypatch,
    tmp_path,
):
    def _unexpected_contour_reconstruction(*_args, **_kwargs):
        raise AssertionError(
            "EQDSK construction must not replace its supplied plasma boundary."
        )

    monkeypatch.setattr(
        equilibrium_mod.equilibrium,
        "rhopol2rz",
        _unexpected_contour_reconstruction,
    )

    eq = _make_fake_eq_instance(monkeypatch, tmp_path)

    assert eq.boundary.attrs["source"] == "supplied"
    assert float(eq.geometry.R_boundary.max()) > float(eq.geometry.R_axis)


def test_eqdsk_without_supplied_boundary_retains_flux_contour_fallback(
    monkeypatch,
    tmp_path,
):
    eq = _make_fake_eq_instance(
        monkeypatch,
        tmp_path,
        include_boundary=False,
    )

    assert eq.boundary.attrs["source"] == "rhopol_contour"
    assert float(eq.geometry.R_boundary.max()) > float(eq.geometry.R_axis)


def test_eqdsk_class_preserves_input_and_internal_convention_metadata(
    monkeypatch,
    tmp_path,
):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    assert eq.cocos_input == 1
    assert eq.cocos_internal == 1
    assert eq.phiclockwise_input is False
    assert eq.phiclockwise_internal is False
    assert eq.phiclockwise is False
    assert eq.flux_normalization_input == "Wb/rad"
    assert eq.flux_normalization_internal == "Wb/rad"

    out = eq.to_dict()
    assert out["cocos_input"] == 1
    assert out["cocos_internal"] == 1
    assert out["phiclockwise_internal"] is False


def test_eqdsk_output_api_is_clean_and_positional_or_keyword():
    to_geqdsk_signature = inspect.signature(eqdsk_mod.eqdsk.to_geqdsk)
    save_signature = inspect.signature(eqdsk_mod.eqdsk.save)

    assert tuple(to_geqdsk_signature.parameters) == ("self", "cocos_out")
    assert tuple(save_signature.parameters) == ("self", "filename", "cocos_out")
    for signature in (to_geqdsk_signature, save_signature):
        assert all(
            parameter.kind is not inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )


def test_to_geqdsk_defaults_to_internal_cocos_and_returns_an_independent_view(
    monkeypatch,
    tmp_path,
):
    eq = _make_fake_eq_instance(
        monkeypatch,
        tmp_path,
        cocos_internal=12,
    )
    cache_before = dict(getattr(eq, "_magnetic_coordinates_cache", {}))
    internal_psi = eq._gdata["psi"].copy()

    output = eq.to_geqdsk()

    required = {
        "nx",
        "ny",
        "rdim",
        "zdim",
        "rcentr",
        "rleft",
        "zmid",
        "rmagx",
        "zmagx",
        "simagx",
        "sibdry",
        "bcentr",
        "cpasma",
        "fpol",
        "pres",
        "ffprime",
        "pprime",
        "psi",
        "qpsi",
    }
    assert required <= output.keys()
    assert output["psi"].shape == (output["nx"], output["ny"])
    assert output["qpsi"].shape == (output["nx"],)
    np.testing.assert_allclose(output["psi"], internal_psi)
    np.testing.assert_allclose(output["qpsi"], eq._gdata["q"])
    assert dict(getattr(eq, "_magnetic_coordinates_cache", {})) == cache_before

    output["psi"][0, 0] += 1.0
    np.testing.assert_array_equal(eq._gdata["psi"], internal_psi)

    snapshot = eq.to_dict()
    assert snapshot["psi"].ndim == 2
    snapshot["psi"][0, 0] += 1.0
    np.testing.assert_array_equal(eq._gdata["psi"], internal_psi)


@pytest.mark.parametrize(
    ("cocos_internal", "cocos_out"),
    ((1, 12), (12, 3)),
)
def test_to_geqdsk_converts_once_from_internal_to_requested_cocos(
    monkeypatch,
    tmp_path,
    cocos_internal,
    cocos_out,
):
    eq = _make_fake_eq_instance(
        monkeypatch,
        tmp_path,
        cocos_internal=cocos_internal,
    )
    internal = eq._internal_geqdsk_data()
    expected = eqdsk_mod.fromCocosNtoCocosM(
        internal,
        cocos_m=cocos_out,
        cocos_n=cocos_internal,
    )

    output = eq.to_geqdsk(cocos_out=cocos_out)

    for field in (
        "simagx",
        "sibdry",
        "bcentr",
        "cpasma",
        "fpol",
        "pres",
        "ffprime",
        "pprime",
        "psi",
        "qpsi",
    ):
        np.testing.assert_allclose(output[field], expected[field])
    assert eq.cocos_internal == cocos_internal


def test_to_geqdsk_allows_absent_optional_boundaries(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)
    for field in ("r_bdy", "z_bdy", "rlim", "zlim"):
        del eq._gdata[field]

    output = eq.to_geqdsk(cocos_out=11)

    for field in ("rbdry", "zbdry", "rlim", "zlim"):
        assert field not in output


def test_save_writes_requested_cocos_and_refuses_overwrite(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path, cocos_internal=12)
    output_path = tmp_path / "output.geqdsk"
    expected = eq.to_geqdsk(cocos_out=3)

    assert eq.save(output_path, cocos_out=3) is None
    with output_path.open("r") as stream:
        written = eqdsk_mod.freeqdsk.geqdsk.read(stream)

    for field in (
        "simagx",
        "sibdry",
        "bcentr",
        "cpasma",
        "fpol",
        "pres",
        "ffprime",
        "pprime",
        "psi",
        "qpsi",
    ):
        np.testing.assert_allclose(written[field], expected[field], rtol=2e-8)

    with pytest.raises(FileExistsError):
        eq.save(output_path, cocos_out=3)

    invalid_path = tmp_path / "invalid.geqdsk"
    with pytest.raises(ValueError, match="COCOS = 9 does not exist"):
        eq.save(invalid_path, cocos_out=9)
    assert not invalid_path.exists()


def test_save_and_reload_round_trip_through_a_third_cocos(monkeypatch, tmp_path):
    read_eqdsk = eqdsk_mod.read_eqdsk
    eqdsk2magnetic = eqdsk_mod.eqdsk2magnetic
    eq = _make_fake_eq_instance(monkeypatch, tmp_path, cocos_internal=12)
    output_path = tmp_path / "round_trip.geqdsk"
    eq.save(output_path, cocos_out=3)

    monkeypatch.setattr(eqdsk_mod, "read_eqdsk", read_eqdsk)
    monkeypatch.setattr(eqdsk_mod, "eqdsk2magnetic", eqdsk2magnetic)
    reloaded = eqdsk_mod.eqdsk(
        str(output_path),
        cocos_in=3,
        cocos_internal=12,
    )

    for field in (
        "psi",
        "psi_ax",
        "psi_bdy",
        "q",
        "fpol",
        "prs",
        "ffp",
        "pprime",
    ):
        np.testing.assert_allclose(
            reloaded._gdata[field],
            eq._gdata[field],
            rtol=3e-8,
            atol=1e-8,
        )
    assert reloaded.cocos_input == 3
    assert reloaded.cocos_internal == 12


def test_generic_equilibrium_defaults_to_internal_cocos1_phi_orientation(
    monkeypatch,
    tmp_path,
):
    loaded = _make_fake_eq_instance(monkeypatch, tmp_path)
    generic = equilibrium_mod.equilibrium(
        rgrid=loaded.Rgrid.values,
        zgrid=loaded.zgrid.values,
        br=loaded.field.Br.values,
        bz=loaded.field.Bz.values,
        bphi=loaded.field.Bphi.values,
        psi=loaded.flux.psi.values,
        Raxis=float(loaded.geometry.R_axis),
        zaxis=float(loaded.geometry.z_axis),
        psi_edge=float(loaded.geometry.attrs["psi_bdy"]),
        psi_ax=float(loaded.geometry.attrs["psi_ax"]),
    )

    assert generic.phiclockwise is False
    assert generic.flux_normalization == "Wb/rad"


def test_generic_equilibrium_accepts_clockwise_full_weber_arrays(
    monkeypatch,
    tmp_path,
):
    loaded = _make_fake_eq_instance(monkeypatch, tmp_path)
    generic = equilibrium_mod.equilibrium(
        rgrid=loaded.Rgrid.values,
        zgrid=loaded.zgrid.values,
        br=loaded.field.Br.values,
        bz=loaded.field.Bz.values,
        bphi=loaded.field.Bphi.values,
        psi=loaded.flux.psi.values,
        Raxis=float(loaded.geometry.R_axis),
        zaxis=float(loaded.geometry.z_axis),
        psi_edge=float(loaded.geometry.attrs["psi_bdy"]),
        psi_ax=float(loaded.geometry.attrs["psi_ax"]),
        phiclockwise=True,
        flux_normalization="Wb",
    )
    assert generic.phiclockwise is True
    assert generic.flux_normalization == "Wb"
    assert generic.fluxdata["psipol"].attrs["units"] == "Wb"


def test_eqdsk_class_does_not_forward_clockwise_input_after_conversion(
    monkeypatch,
    tmp_path,
):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path, cocos_input=2)
    assert eq.cocos_input == 2
    assert eq.cocos_internal == 1
    assert eq.phiclockwise_input is True
    assert eq.phiclockwise_internal is False
    assert eq.phiclockwise is False

    forwarded = {}

    class _CoordinateCallCaptured(Exception):
        pass

    def _capture_coordinate_call(**kwargs):
        forwarded.update(kwargs)
        raise _CoordinateCallCaptured

    monkeypatch.setattr(
        equilibrium_mod,
        "compute_magnetic_coordinates",
        _capture_coordinate_call,
    )
    with pytest.raises(_CoordinateCallCaptured):
        eq.compute_coordinates(
            lpsi=8,
            ltheta=16,
            dr_hr=0.05,
            dz_hz=0.05,
        )

    assert forwarded["phiclockwise"] is False


def test_eqdsk_class_supports_clockwise_full_weber_internal_cocos(
    monkeypatch,
    tmp_path,
):
    eq = _make_fake_eq_instance(
        monkeypatch,
        tmp_path,
        cocos_input=1,
        cocos_internal=12,
    )
    assert eq.cocos_input == 1
    assert eq.cocos_internal == 12
    assert eq.phiclockwise_input is False
    assert eq.phiclockwise_internal is True
    assert eq.phiclockwise is True
    assert eq.flux_normalization_internal == "Wb"
    assert eq.fluxdata["psipol"].attrs["units"] == "Wb"
    assert eq.profiles["psi"].attrs["units"] == "Wb"


def test_cocos_info_uses_explicit_input_and_internal_fields(
    monkeypatch,
    tmp_path,
):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path, cocos_input=12)
    info = eq.cocos_info

    assert info["cocos_input"] == 12
    assert info["cocos_internal"] == 1
    assert info["cocos_internal_obj"].cocos == 1
    assert info["cocos_input_obj"].cocos == 12
    assert info["exp_Bp_input"] == 1
    assert info["sigma_RpZ_input"] == -1
    assert info["exp_Bp_internal"] == 0
    assert info["sigma_RpZ_internal"] == 1


@pytest.mark.parametrize(
    (
        "cocos_internal",
        "flux_units",
        "position_per_flux_units",
        "angle_per_flux_units",
        "flux_per_length_units",
        "flux_per_angle_units",
        "jacobian_units",
        "direct_det_units",
    ),
    (
        (
            1,
            "Wb/rad",
            "m*rad/Wb",
            "rad**2/Wb",
            "Wb/(rad*m)",
            "Wb/rad**2",
            "m**3/Wb",
            "Wb/m**2",
        ),
        (
            12,
            "Wb",
            "m/Wb",
            "rad/Wb",
            "Wb/m",
            "Wb/rad",
            "m**3/(Wb*rad)",
            "Wb*rad/m**2",
        ),
    ),
    ids=("cocos_1", "cocos_12"),
)
def test_build_magnetic_coordinates_dataset_has_expected_coordinate_names(
    monkeypatch,
    tmp_path,
    cocos_internal,
    flux_units,
    position_per_flux_units,
    angle_per_flux_units,
    flux_per_length_units,
    flux_per_angle_units,
    jacobian_units,
    direct_det_units,
):
    eq = _make_fake_eq_instance(
        monkeypatch,
        tmp_path,
        cocos_input=cocos_internal,
        cocos_internal=cocos_internal,
    )

    npsi = 12
    ltheta = 32
    ntht_pad = 3
    theta = np.linspace(0.0, 2.0 * np.pi, ltheta)
    psi_axis = float(eq.geometry.attrs["psi_ax"])
    psi_boundary = float(eq.geometry.attrs["psi_bdy"])
    # The spline coordinate stays ascending even when physical psi decreases
    # from axis to boundary, as it does for COCOS 12.
    psi_axis_order = psi_axis + np.linspace(0.05, 0.95, npsi) * (
        psi_boundary - psi_axis
    )
    psi_order = np.argsort(psi_axis_order)
    psigrid = psi_axis_order[psi_order]

    thtable = np.tile(theta, (npsi, 1))
    nutable = np.zeros((npsi, ltheta))
    jac = np.ones((npsi, ltheta))
    psi_scale = (psi_boundary - psi_axis) / 0.16
    surface_radius = np.sqrt((psigrid - psi_axis) / psi_scale)
    Rtransform = 1.5 + surface_radius[:, None] * np.cos(theta)[None, :]
    ztransform = surface_radius[:, None] * np.sin(theta)[None, :]
    qprof = np.linspace(1.0, 2.0, npsi)[psi_order]
    Fprof = np.linspace(2.0, 1.8, npsi)[psi_order]
    Iprof = np.linspace(1.0e6, 1.1e6, npsi)[psi_order]

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
    assert "nu_shift" not in mag.coords
    assert mag.coords["nu"].attrs["name"] == "nu"
    assert mag.coords["nu"].attrs["gauge_relation"] == "zeta = phi + nu"
    assert eq.fluxdata["psipol"].attrs["units"] == flux_units
    assert mag.coords["psi"].attrs["units"] == flux_units
    assert mag.coords["psi0"].attrs["units"] == flux_units
    assert mag.deriv["psi0"].attrs["units"] == flux_units
    assert mag.deriv["dR_dpsi"].attrs["units"] == position_per_flux_units
    assert mag.deriv["dphi_dpsi"].attrs["units"] == angle_per_flux_units
    assert mag.deriv["dPsi_dr"].attrs["units"] == flux_per_length_units
    assert mag.deriv["dPsi_dphi"].attrs["units"] == flux_per_angle_units
    # The theta derivative contributes one radian, so the signed physical
    # three-coordinate Jacobian and direct R-Z determinant retain these units.
    assert mag.deriv["jacobian"].attrs["units"] == jacobian_units
    assert mag.deriv["direct_det_Rz"].attrs["units"] == direct_det_units
    assert "inside_lcfs" in mag.coords
    inside = mag.coords["inside_lcfs"].values
    assert inside.dtype == np.bool_
    assert np.any(inside)
    assert np.any(~inside)
    fitted = mag.coords["inside_coordinate_domain"].values
    assert np.any(fitted)
    assert np.any(inside & ~fitted)

    rr, zz = np.meshgrid(eq.Rgrid.values, eq.zgrid.values, indexing="ij")
    expected_dpsi_dr = 2.0 * psi_scale * (rr - 1.5)
    expected_dpsi_dz = 2.0 * psi_scale * zz

    for name in ("dPsi_dr", "dPsi_dz", "dPsi_dphi"):
        assert np.all(np.isfinite(mag.deriv[name].values))
        assert (
            mag.deriv[name].attrs["validity_domain"]
            == "finite_equilibrium_RZ_grid"
        )
    np.testing.assert_allclose(
        mag.deriv["dPsi_dr"].values,
        expected_dpsi_dr,
        rtol=3.0e-4,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        mag.deriv["dPsi_dz"].values,
        expected_dpsi_dz,
        rtol=3.0e-4,
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(mag.deriv["dPsi_dphi"].values, 0.0)

    np.testing.assert_array_equal(mag.deriv["dphi_dzeta"].values[fitted], 1.0)
    assert np.all(np.isnan(mag.deriv["dphi_dzeta"].values[~fitted]))
    assert np.all(np.isnan(mag.deriv["jacobian"].values[~fitted]))
    assert np.all(np.isnan(mag.deriv["direct_det_Rz"].values[~fitted]))
    assert np.all(np.isnan(mag.deriv["dTheta_dr"].values[~fitted]))
    assert np.all(np.isnan(mag.deriv["dR_dpsi"].values[~fitted]))

    gpsi_psi = mag.metric("psi", "psi", tensor="contravariant")
    expected_gpsi_psi = (
        mag.deriv["dPsi_dr"] ** 2
        + (mag.deriv["dPsi_dphi"] / mag.deriv.R) ** 2
        + mag.deriv["dPsi_dz"] ** 2
    )
    assert np.all(np.isfinite(gpsi_psi.values))
    np.testing.assert_allclose(gpsi_psi, expected_gpsi_psi)
    gpsi_theta = mag.metric("psi", "theta", tensor="contravariant")
    assert np.all(np.isnan(gpsi_theta.values[~fitted]))

    outside_index = tuple(np.argwhere(~inside)[0])
    R_outside = np.array([mag.coords.R.values[outside_index[0]]])
    z_outside = np.array([mag.coords.z.values[outside_index[1]]])
    exterior_gradient = mag._transform_deriv(  # noqa: SLF001
        R_outside,
        z_outside,
        only="dPsi_dr",
    )
    np.testing.assert_allclose(
        exterior_gradient["dPsi_dr"].values,
        expected_dpsi_dr[outside_index],
        rtol=3.0e-4,
        atol=1.0e-10,
    )

    assert "direct_det_Rz" in mag.deriv
    assert mag.coords["psi0"].attrs["psi_axis"] == pytest.approx(
        eq.geometry.attrs["psi_ax"]
    )
    assert mag.coords["psi0"].attrs["psi_boundary"] == pytest.approx(
        eq.geometry.attrs["psi_bdy"]
    )


def test_build_magnetic_coordinates_dataset_boozer_current_convention(monkeypatch, tmp_path):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)

    npsi = 8
    ltheta = 24
    ntht_pad = 2
    theta = np.linspace(0.0, 2.0 * np.pi, ltheta)
    psi0 = float(eq.geometry.attrs["psi_ax"]) + 0.01
    psi1 = float(eq.geometry.attrs["psi_bdy"]) - 0.01
    psigrid = np.linspace(psi0, psi1, npsi)

    thtable = np.tile(theta, (npsi, 1))
    nutable = np.zeros((npsi, ltheta))
    jac = np.ones((npsi, ltheta))
    psi_scale = (
        float(eq.geometry.attrs["psi_bdy"])
        - float(eq.geometry.attrs["psi_ax"])
    ) / 0.16
    surface_radius = np.sqrt(
        (psigrid - float(eq.geometry.attrs["psi_ax"])) / psi_scale
    )
    Rtransform = 1.5 + surface_radius[:, None] * np.cos(theta)[None, :]
    ztransform = surface_radius[:, None] * np.sin(theta)[None, :]
    qprof = np.linspace(1.0, 2.0, npsi)
    Fprof = np.linspace(2.0, 1.8, npsi)
    Iprof = np.linspace(0.3, 0.5, npsi)

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

    # I is the one canonical covariant Boozer coefficient B_Theta.
    np.testing.assert_allclose(mag.deriv["I"].values, Iprof, rtol=1e-12, atol=1e-12)
    assert "I_boozer" not in mag.deriv
    assert "I_legacy_2pi" not in mag.deriv
    assert not hasattr(eq, "boozer_profs")
    # h must stay consistent with the Boozer Jacobian relation.
    np.testing.assert_allclose(
        mag.deriv["h"].values,
        qprof * Fprof + Iprof,
        rtol=1e-12,
        atol=1e-12,
    )
    assert mag.deriv["h"].attrs["units"] == "T*m"
    assert "J*B**2 = I + qF" in mag.deriv["h"].attrs["desc"]


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

    real_spline = equilibrium_mod.RectBivariateSpline

    class _LengthOneScalarSpline:
        def __init__(self, *args, **kwargs):
            self._spline = real_spline(*args, **kwargs)

        def ev(self, *args, **kwargs):
            value = np.atleast_1d(self._spline.ev(*args, **kwargs))
            captured["axis_spline_shape"] = value.shape
            return value

        def __call__(self, *args, **kwargs):
            return self._spline(*args, **kwargs)

    monkeypatch.setattr(
        equilibrium_mod,
        "RectBivariateSpline",
        _LengthOneScalarSpline,
    )

    def _fake_compute_magnetic_coordinates(*args, **kwargs):
        psigrid = np.asarray(kwargs["psigrid"], dtype=float)
        ltheta = int(kwargs["ltheta"])
        npsi = psigrid.size
        captured["psigrid"] = psigrid.copy()
        tracing_R = np.asarray(kwargs["Rgrid"], dtype=float)
        tracing_z = np.asarray(kwargs["zgrid"], dtype=float)
        kwargs["diagnostics"]["coordinate_psi_field"] = (
            tracing_R[:, None] + 2.0 * tracing_z[None, :]
        )
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
    def _capture_builder(*args, **kwargs):
        captured["builder_psigrid"] = np.asarray(args[0], dtype=float).copy()
        captured["core_indices"] = np.asarray(
            kwargs["core_indices"],
            dtype=int,
        ).copy()
        captured["support_metadata"] = dict(
            kwargs["radial_support_metadata"]
        )
        captured["coordinate_psi_field"] = np.asarray(
            kwargs["coordinate_psi_field"],
            dtype=float,
        ).copy()
        return SimpleNamespace(dummy=True)

    monkeypatch.setattr(eq, "_build_magnetic_coordinates_dataset", _capture_builder)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Conversion of an array with ndim > 0 to a scalar",
            category=DeprecationWarning,
        )
        out = eq.compute_coordinates(
            coordinate_system="boozer",
            lpsi=9,
            ltheta=24,
            rhopol_min=0.2,
            rhopol_max=0.8,
        )

    assert getattr(out, "dummy", False)
    assert captured["axis_spline_shape"] == (1,)
    psi_axis = float(eq.geometry.attrs["psi_ax"])
    psi_edge = float(eq.geometry.attrs["psi_bdy"])
    expected_psi_start = psi_axis + (0.2**2) * (psi_edge - psi_axis)
    expected_psi_end = psi_axis + (0.8**2) * (psi_edge - psi_axis)
    core_psi = captured["builder_psigrid"][captured["core_indices"]]
    assert np.isclose(core_psi[0], expected_psi_start)
    assert np.isclose(core_psi[-1], expected_psi_end)
    assert captured["psigrid"].size > 9
    assert captured["support_metadata"]["core_nsurface"] == 9
    assert captured["support_metadata"]["support_nsurface"] == (
        captured["psigrid"].size
    )
    assert captured["coordinate_psi_field"].shape == (
        eq.Rgrid.size,
        eq.zgrid.size,
    )
    np.testing.assert_allclose(
        captured["coordinate_psi_field"],
        np.asarray(eq.Rgrid)[:, None] + 2.0 * np.asarray(eq.zgrid)[None, :],
        rtol=0.0,
        atol=2.0e-14,
    )


def test_radial_support_preserves_endpoint_inclusive_core():
    core = np.linspace(0.0, 1.0, 9)
    support, core_indices = equilibrium_mod._extend_radial_support(  # noqa: SLF001
        core,
        lower_bound=0.0,
        upper_bound=1.0,
        guard_surfaces=3,
    )

    np.testing.assert_array_equal(support, core)
    np.testing.assert_array_equal(core_indices, np.arange(core.size))


@pytest.mark.parametrize("orientation", [1.0, -1.0])
def test_radial_support_skips_flux_below_interpolated_axis_floor(orientation):
    normalized_floor = equilibrium_mod._normalized_resolvable_axis_flux(  # noqa: SLF001
        orientation * 4.6565e-6,
        psi_axis=0.0,
        psi_boundary=orientation,
    )
    assert normalized_floor == pytest.approx(4.6565e-6)

    core = np.linspace(0.005, 0.995, 511)
    support, core_indices = equilibrium_mod._extend_radial_support(  # noqa: SLF001
        core,
        lower_bound=np.sqrt(normalized_floor),
        upper_bound=1.0,
        guard_surfaces=3,
    )

    assert support[0] > np.sqrt(normalized_floor)
    assert support[0] < core[0]
    np.testing.assert_array_equal(support[core_indices], core)


@pytest.mark.parametrize("orientation", [1.0, -1.0])
def test_outboard_midplane_seed_inverse_is_bounded_and_monotone(orientation):
    radial = np.linspace(1.7, 2.0, 7)
    normalized_flux = np.array(
        [4.0e-6, 0.03, 0.025, 0.18, 0.42, 0.72, 1.05]
    )
    targets = np.array([0.003, 0.1, 0.5, 0.9])

    seeds = equilibrium_mod._outboard_midplane_seeds(  # noqa: SLF001
        radial,
        orientation * normalized_flux,
        targets,
        psi_axis=0.0,
        psi_boundary=orientation,
    )

    assert np.all(np.isfinite(seeds))
    assert np.all(np.diff(seeds) > 0.0)
    assert seeds[0] >= radial[0]
    assert seeds[-1] <= radial[-1]


def test_compute_coordinates_keeps_descending_physical_flux_spline_sorted(
    monkeypatch,
    tmp_path,
):
    eq = _make_fake_eq_instance(monkeypatch, tmp_path)

    descending_psi = -np.asarray(eq.flux.psi.values)
    eq._flux["psi"] = xr.DataArray(  # noqa: SLF001 - synthetic COCOS-1 regression
        descending_psi,
        dims=("R", "z"),
        coords={"R": eq.Rgrid.values, "z": eq.zgrid.values},
        attrs=eq.flux.psi.attrs.copy(),
    )
    eq.fluxdata["psipol"] = eq._flux["psi"]  # noqa: SLF001
    eq.geometry.attrs["psi_ax"] = 0.0
    eq.geometry.attrs["psi_bdy"] = -0.16
    eq._psi_ax_init = 0.0  # noqa: SLF001
    eq._psi_edge_init = -0.16  # noqa: SLF001

    captured = {}

    def _fake_compute_magnetic_coordinates(*args, **kwargs):
        psigrid = np.asarray(kwargs["psigrid"], dtype=float)
        ltheta = int(kwargs["ltheta"])
        npsi = psigrid.size
        captured["kernel_psigrid"] = psigrid.copy()
        theta = np.linspace(0.0, 2.0 * np.pi, ltheta)
        table = np.tile(theta, (npsi, 1))
        zeros = np.zeros_like(table)
        ones = np.ones_like(table)
        return (
            psigrid.copy(),
            np.ones(npsi),
            np.ones(npsi),
            table,
            zeros,
            ones,
            np.ones_like(table),
            np.zeros_like(table),
        )

    def _capture_builder(*args, **kwargs):
        captured["builder_psigrid"] = np.asarray(args[0], dtype=float).copy()
        captured["builder_qprof"] = np.asarray(args[8], dtype=float).copy()
        captured["core_indices"] = np.asarray(
            kwargs["core_indices"],
            dtype=int,
        ).copy()
        return SimpleNamespace(dummy=True)

    monkeypatch.setattr(
        equilibrium_mod,
        "compute_magnetic_coordinates",
        _fake_compute_magnetic_coordinates,
    )
    monkeypatch.setattr(eq, "_build_magnetic_coordinates_dataset", _capture_builder)

    out = eq.compute_coordinates(
        coordinate_system="boozer",
        lpsi=9,
        ltheta=24,
        rhopol_min=0.2,
        rhopol_max=0.8,
    )

    assert getattr(out, "dummy", False)
    # The kernel must see the physically meaningful axis-to-boundary order;
    # for this equilibrium that means decreasing psi. Storage is sorted only
    # after tracing so spline constructors receive an increasing coordinate.
    assert np.all(np.diff(captured["kernel_psigrid"]) < 0.0)
    assert np.all(np.diff(captured["builder_psigrid"]) > 0.0)
    np.testing.assert_array_equal(
        captured["builder_psigrid"],
        captured["kernel_psigrid"][::-1],
    )
    np.testing.assert_array_equal(
        captured["builder_qprof"],
        captured["builder_psigrid"],
    )
    expected_core = -0.16 * np.linspace(0.2, 0.8, 9) ** 2
    np.testing.assert_allclose(
        captured["builder_psigrid"][captured["core_indices"]],
        np.sort(expected_core),
        rtol=0.0,
        atol=1.0e-15,
    )


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
