"""Behavioral contract shared by pyCOCOS, superNOVA, and the NOVA bridge."""

import numpy as np
import xarray as xr

from test_eqdsk_reader import _make_fake_eq_instance


def _build_public_coordinates(monkeypatch, tmp_path):
    """Build a small public MagneticCoordinates product without tracing lines."""
    equilibrium = _make_fake_eq_instance(monkeypatch, tmp_path)

    npsi = 8
    ltheta = 24
    ntht_pad = 2
    theta = np.linspace(0.0, 2.0 * np.pi, ltheta)
    psi_axis = float(equilibrium.geometry.attrs["psi_ax"])
    psi_boundary = float(equilibrium.geometry.attrs["psi_bdy"])
    psigrid = np.linspace(psi_axis + 0.01, psi_boundary - 0.01, npsi)

    theta_table = np.tile(theta, (npsi, 1))
    nu_table = np.zeros((npsi, ltheta))
    jacobian = np.ones((npsi, ltheta))
    psi_scale = (psi_boundary - psi_axis) / 0.16
    surface_radius = np.sqrt((psigrid - psi_axis) / psi_scale)
    R_inverse = (
        float(equilibrium.geometry.R_axis)
        + surface_radius[:, None] * np.cos(theta)[None, :]
    )
    z_inverse = (
        float(equilibrium.geometry.z_axis)
        + surface_radius[:, None] * np.sin(theta)[None, :]
    )
    q = np.linspace(1.0, 2.0, npsi)
    F = np.linspace(2.0, 1.8, npsi)
    I = np.linspace(0.3, 0.5, npsi)

    magnetic = equilibrium._build_magnetic_coordinates_dataset(  # noqa: SLF001
        psigrid=psigrid,
        thtable=theta_table,
        nutable=nu_table,
        jac=jacobian,
        Rtransform=R_inverse,
        ztransform=z_inverse,
        R_fine=equilibrium.Rgrid.values,
        z_fine=equilibrium.zgrid.values,
        qprof=q,
        Fprof=F,
        Iprof=I,
        ntht_pad=ntht_pad,
        coordinate_system="boozer",
    )
    return equilibrium, magnetic


def test_supernova_required_coordinate_and_derivative_schema(monkeypatch, tmp_path):
    equilibrium, magnetic = _build_public_coordinates(monkeypatch, tmp_path)

    assert {"psi", "theta", "nu", "R_inv", "z_inv"} <= set(
        magnetic.coords.data_vars
    )
    assert magnetic.coords["psi0"].dims == ("psi0",)
    assert magnetic.coords["theta"].dims == ("psi0", "thetageom")
    assert magnetic.coords["nu"].dims == ("psi0", "thetageom")
    assert magnetic.coords["R_inv"].dims == ("psi0", "theta_star")
    assert magnetic.coords["z_inv"].dims == ("psi0", "theta_star")
    assert np.all(np.diff(magnetic.coords["psi0"].values) > 0.0)
    assert magnetic.coords["psi0"].attrs["psi_axis"] == float(
        equilibrium.geometry.attrs["psi_ax"]
    )
    assert magnetic.coords["psi0"].attrs["psi_boundary"] == float(
        equilibrium.geometry.attrs["psi_bdy"]
    )
    assert magnetic.coords["nu"].attrs["gauge_relation"] == "zeta = phi + nu"

    required_profiles = {"q", "F", "I", "h"}
    required_rz_fields = {
        "jacobian",
        "dPsi_dr",
        "dPsi_dz",
        "dPsi_dphi",
    }
    assert required_profiles | required_rz_fields <= set(magnetic.deriv.data_vars)
    for name in required_profiles:
        assert magnetic.deriv[name].dims == ("psi0",)
    for name in required_rz_fields:
        assert magnetic.deriv[name].dims == ("R", "z")
    xr.testing.assert_allclose(
        magnetic.deriv["h"],
        magnetic.deriv["I"] + magnetic.deriv["q"] * magnetic.deriv["F"],
    )


def test_scalar_forward_transform_returns_canonical_coordinate_names(
    monkeypatch,
    tmp_path,
):
    _, magnetic = _build_public_coordinates(monkeypatch, tmp_path)

    transformed = magnetic(R=1.7, z=0.0, grid=False, fill_nan=False)

    assert {"psi", "theta", "nu"} <= set(transformed.data_vars)
    assert {"R", "z"} <= set(transformed.coords)
    for name in ("psi", "theta", "nu"):
        assert transformed[name].size == 1
        assert np.all(np.isfinite(transformed[name].values))
    assert transformed["nu"].attrs["name"] == "nu"


def test_inverse_transform_preserves_public_names_and_grid_dimensions(
    monkeypatch,
    tmp_path,
):
    _, magnetic = _build_public_coordinates(monkeypatch, tmp_path)
    psi = magnetic.coords["psi0"].values[[1, 3, 5]]
    theta = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)

    inverse = magnetic.transform_inverse(
        psi=psi,
        thetamag=theta,
        grid=True,
        psi_is_norm=False,
    )

    assert set(inverse.data_vars) == {"R_inv", "z_inv"}
    assert inverse["R_inv"].dims == ("psi", "thetamag")
    assert inverse["z_inv"].dims == ("psi", "thetamag")
    assert inverse["R_inv"].shape == (psi.size, theta.size)
    assert inverse["z_inv"].shape == (psi.size, theta.size)
    assert np.all(np.isfinite(inverse["R_inv"].values))
    assert np.all(np.isfinite(inverse["z_inv"].values))


def test_cyl2mag_uses_physical_axis_boundary_flux_normalization(
    monkeypatch,
    tmp_path,
):
    _, magnetic = _build_public_coordinates(monkeypatch, tmp_path)
    R = magnetic.coords["R"].values
    z = magnetic.coords["z"].values
    field = xr.DataArray(
        np.ones((R.size, z.size)),
        dims=("R", "z"),
        coords={"R": R, "z": z},
        attrs={"name": "unit_field", "units": "1"},
    )

    transformed = magnetic.cyl2mag_scalar(field, return_psi_norm=True)

    psi_axis = float(magnetic.coords["psi0"].attrs["psi_axis"])
    psi_boundary = float(magnetic.coords["psi0"].attrs["psi_boundary"])
    expected = (
        magnetic.coords["psi0"].values - psi_axis
    ) / (psi_boundary - psi_axis)
    np.testing.assert_allclose(transformed.coords["psi"].values, expected)
    assert transformed.coords["psi"].attrs["name"] == "psi_norm"
    assert transformed.dims == ("psi", "theta", "zeta")


def test_metric_and_jacobian_accessors_keep_rz_contract(monkeypatch, tmp_path):
    _, magnetic = _build_public_coordinates(monkeypatch, tmp_path)

    for tensor in ("covariant", "contravariant"):
        for first in ("psi", "theta", "zeta"):
            for second in ("psi", "theta", "zeta"):
                component = magnetic.metric(first, second, tensor=tensor)
                reflected = magnetic.metric(second, first, tensor=tensor)
                assert component.dims == ("R", "z")
                np.testing.assert_allclose(
                    component.values,
                    reflected.values,
                    equal_nan=True,
                )

    jacobian = magnetic.jacobian(return_in="Rzphi")
    inverse = magnetic.jacobian(return_in="Rzphi", inverse=True)
    assert jacobian.dims == ("R", "z")
    assert inverse.dims == ("R", "z")
    finite = np.isfinite(jacobian.values) & np.isfinite(inverse.values)
    assert np.any(finite)
    np.testing.assert_allclose(
        jacobian.values[finite] * inverse.values[finite],
        1.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
