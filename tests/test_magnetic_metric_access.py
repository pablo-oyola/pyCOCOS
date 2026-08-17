import numpy as np
import xarray as xr
import pytest

from test_magnetic_scalar_transform import _build_synthetic_magnetic_coordinates
from pycocos.core.equilibrium import _inverse_toroidal_derivatives
from pycocos.core.magnetic_coordinates import magnetic_coordinates


def _add_constant_jacobian(mag, value: float = 2.0):
    template = mag.deriv["dR_dpsi"]
    mag.deriv["jacobian"] = xr.DataArray(
        np.full(template.shape, value, dtype=np.float64),
        dims=template.dims,
        coords=template.coords,
        attrs={
            "name": "jacobian",
            "units": "",
            "desc": "Jacobian of the transformation",
            "short_name": "$\\mathcal{J}$",
        },
    )


def test_metric_covariant_access_returns_rzphi_dataarray():
    mag = _build_synthetic_magnetic_coordinates()
    gij = mag.metric("psi", "theta", tensor="covariant")

    assert gij.dims == ("R", "z")
    assert gij.attrs["name"] == "g_psi_theta"
    assert np.all(np.isfinite(gij.values))


def test_metric_cache_can_be_built_lazily():
    complete = _build_synthetic_magnetic_coordinates()
    lazy = magnetic_coordinates(
        complete.coords,
        complete.deriv,
        Raxis=complete.Raxis,
        zaxis=complete.zaxis,
        pad=complete.nthtpad,
        build_metric_cache=False,
    )

    assert not lazy.metric_cache_built
    metric = lazy.metric_covariant
    assert lazy.metric_cache_built
    assert "g_psi_psi" in metric
    assert "h_psi" in lazy.lame_mag


def test_metric_covariant_and_contravariant_are_inverse():
    mag = _build_synthetic_magnetic_coordinates()
    names = ("psi", "theta", "zeta")
    ridx, zidx = 7, 9

    g_cov = np.array(
        [[mag.metric(i, j).values[ridx, zidx] for j in names] for i in names],
        dtype=np.float64,
    )
    g_contra = np.array(
        [
            [
                mag.metric(i, j, tensor="contravariant").values[ridx, zidx]
                for j in names
            ]
            for i in names
        ],
        dtype=np.float64,
    )

    eye = g_cov @ g_contra
    assert np.allclose(eye, np.eye(3), rtol=0.0, atol=1.0e-10)


def test_nonorthogonal_shifted_metric_tensors_are_exact_inverses():
    base = _build_synthetic_magnetic_coordinates()
    deriv = base.deriv.copy(deep=True)

    # Rows are d(psi, theta, zeta)/d(R, phi, Z). The last row includes a
    # non-trivial axisymmetric toroidal shift while d(zeta)/d(phi)=1.
    direct = np.array(
        [
            [1.7, 0.0, 0.35],
            [-0.25, 0.0, 1.2],
            [0.4, 1.0, -0.3],
        ]
    )
    inverse = np.linalg.inv(direct)
    shape = deriv["dR_dpsi"].shape

    direct_names = (
        ("dPsi_dr", "dPsi_dphi", "dPsi_dz"),
        ("dTheta_dr", "dTheta_dphi", "dTheta_dz"),
        ("dzeta_dr", "dzeta_dphi", "dzeta_dz"),
    )
    inverse_names = (
        ("dR_dpsi", "dR_dtheta", "dR_dzeta"),
        ("dphi_dpsi", "dphi_dtheta", "dphi_dzeta"),
        ("dz_dpsi", "dz_dtheta", "dz_dzeta"),
    )
    for row, names in enumerate(direct_names):
        for column, name in enumerate(names):
            deriv[name].values[:] = np.full(shape, direct[row, column])
    for row, names in enumerate(inverse_names):
        for column, name in enumerate(names):
            deriv[name].values[:] = np.full(shape, inverse[row, column])

    mag = magnetic_coordinates(
        coords=base.coords,
        deriv=deriv,
        Raxis=base.Raxis,
        zaxis=base.zaxis,
        pad=base.nthtpad,
    )
    names = ("psi", "theta", "zeta")
    for ridx, zidx in ((3, 5), (17, 19)):
        g_cov = np.array(
            [
                [mag.metric(i, j).values[ridx, zidx] for j in names]
                for i in names
            ]
        )
        g_contra = np.array(
            [
                [
                    mag.metric(i, j, tensor="contravariant").values[
                        ridx, zidx
                    ]
                    for j in names
                ]
                for i in names
            ]
        )
        np.testing.assert_allclose(
            g_cov @ g_contra, np.eye(3), rtol=0.0, atol=1.0e-12
        )


def test_cylindrical_metric_uses_correct_angular_scale_factors():
    mag = _build_synthetic_magnetic_coordinates()
    radius = np.broadcast_to(
        mag.coords.R.values[:, None], mag.deriv.dR_dpsi.shape
    )

    g_zeta_zeta = mag.metric("zeta", "zeta", tensor="covariant")
    g_contra_zeta_zeta = mag.metric(
        "zeta", "zeta", tensor="contravariant"
    )

    np.testing.assert_allclose(g_zeta_zeta, radius**2)
    np.testing.assert_allclose(g_contra_zeta_zeta, 1.0 / radius**2)


@pytest.mark.parametrize(
    "tensor",
    ("legacy_covariant", "legacy_contravariant"),
)
def test_removed_metric_selectors_are_rejected(tensor):
    mag = _build_synthetic_magnetic_coordinates()

    with pytest.raises(ValueError, match="covariant.*contravariant"):
        mag.metric("zeta", "zeta", tensor=tensor)


def test_nu_is_not_a_metric_index_alias_for_zeta():
    mag = _build_synthetic_magnetic_coordinates()

    with pytest.raises(ValueError, match="Invalid metric index 'nu'"):
        mag.metric("nu", "nu")


def test_metric_reports_missing_direct_derivatives():
    complete = _build_synthetic_magnetic_coordinates()
    deriv = complete.deriv.drop_vars("dPsi_dr")
    incomplete = magnetic_coordinates(
        coords=complete.coords,
        deriv=deriv,
        Raxis=complete.Raxis,
        zaxis=complete.zaxis,
        pad=complete.nthtpad,
    )

    with pytest.raises(ValueError, match="dPsi_dr"):
        incomplete.metric("psi", "psi", tensor="contravariant")


def test_inverse_toroidal_derivatives_use_direct_two_dimensional_determinant():
    dpsi_dR = np.array([2.0, 1.5])
    dpsi_dZ = np.array([0.5, -0.25])
    dtheta_dR = np.array([-0.25, 0.4])
    dtheta_dZ = np.array([1.5, 1.2])
    dnu_dR = np.array([0.3, -0.1])
    dnu_dZ = np.array([-0.4, 0.2])

    direct_det, dphi_dpsi, dphi_dtheta, dphi_dzeta = (
        _inverse_toroidal_derivatives(
            dpsi_dR,
            dpsi_dZ,
            dtheta_dR,
            dtheta_dZ,
            dnu_dR,
            dnu_dZ,
        )
    )
    expected_det = dpsi_dR * dtheta_dZ - dpsi_dZ * dtheta_dR

    np.testing.assert_allclose(direct_det, expected_det)
    np.testing.assert_allclose(
        dphi_dpsi,
        (dtheta_dR * dnu_dZ - dtheta_dZ * dnu_dR) / expected_det,
    )
    np.testing.assert_allclose(
        dphi_dtheta,
        (dpsi_dZ * dnu_dR - dpsi_dR * dnu_dZ) / expected_det,
    )
    np.testing.assert_array_equal(dphi_dzeta, np.ones_like(expected_det))

    radius = np.array([1.7, 2.1])
    grad_psi = np.stack((dpsi_dR, np.zeros(2), dpsi_dZ), axis=-1)
    grad_theta = np.stack((dtheta_dR, np.zeros(2), dtheta_dZ), axis=-1)
    grad_zeta = np.stack((dnu_dR, 1.0 / radius, dnu_dZ), axis=-1)
    triple_product = np.einsum(
        "ij,ij->i", grad_psi, np.cross(grad_theta, grad_zeta)
    )
    np.testing.assert_allclose(1.0 / triple_product, -radius / direct_det)


def test_manufactured_pointwise_boozer_identities():
    """The signed-J identities hold with cylindrical physical components."""
    mag = _build_synthetic_magnetic_coordinates()
    radius = np.broadcast_to(
        mag.coords.R.values[:, None], mag.deriv.dR_dpsi.shape
    )

    grad_psi = np.stack(
        (
            mag.deriv.dPsi_dr.values,
            mag.deriv.dPsi_dphi.values / radius,
            mag.deriv.dPsi_dz.values,
        ),
        axis=-1,
    )
    grad_theta = np.stack(
        (
            mag.deriv.dTheta_dr.values,
            mag.deriv.dTheta_dphi.values / radius,
            mag.deriv.dTheta_dz.values,
        ),
        axis=-1,
    )
    grad_zeta = np.stack(
        (
            mag.deriv.dzeta_dr.values,
            mag.deriv.dzeta_dphi.values / radius,
            mag.deriv.dzeta_dz.values,
        ),
        axis=-1,
    )

    inverse_jacobian = np.einsum(
        "...i,...i->...", grad_psi, np.cross(grad_theta, grad_zeta)
    )
    jacobian = 1.0 / inverse_jacobian
    tangent_theta = jacobian[..., None] * np.cross(grad_zeta, grad_psi)
    tangent_zeta = jacobian[..., None] * np.cross(grad_psi, grad_theta)

    q = np.broadcast_to(
        0.8 + 0.1 * mag.coords.z.values[None, :], jacobian.shape
    )
    magnetic_field = (
        tangent_theta + q[..., None] * tangent_zeta
    ) / jacobian[..., None]
    I = np.einsum("...i,...i->...", magnetic_field, tangent_theta)
    g = np.einsum("...i,...i->...", magnetic_field, tangent_zeta)
    h = I + q * g
    b_cross_grad_psi = np.cross(magnetic_field, grad_psi)

    np.testing.assert_allclose(
        jacobian
        * np.einsum("...i,...i->...", magnetic_field, grad_theta),
        np.ones_like(jacobian),
    )
    np.testing.assert_allclose(
        jacobian
        * np.einsum("...i,...i->...", magnetic_field, grad_zeta),
        q,
    )
    np.testing.assert_allclose(
        jacobian * np.einsum("...i,...i->...", magnetic_field, magnetic_field),
        h,
    )
    np.testing.assert_allclose(
        jacobian
        * np.einsum("...i,...i->...", b_cross_grad_psi, grad_theta),
        g,
    )
    np.testing.assert_allclose(
        jacobian
        * np.einsum("...i,...i->...", b_cross_grad_psi, grad_zeta),
        -I,
    )


def test_metric_is_symmetric_for_all_index_pairs():
    mag = _build_synthetic_magnetic_coordinates()
    names = ("psi", "theta", "zeta")

    for i in names:
        for j in names:
            gij = mag.metric(i, j, tensor="covariant")
            gji = mag.metric(j, i, tensor="covariant")
            assert np.allclose(gij.values, gji.values, rtol=0.0, atol=1.0e-12)


def test_metric_can_be_returned_in_magnetic_coordinates():
    mag = _build_synthetic_magnetic_coordinates()
    gij_mag = mag.metric(
        "psi",
        "psi",
        tensor="covariant",
        return_in="magnetic_coordinates",
        return_psi_norm=True,
    )

    assert gij_mag.dims == ("psi", "theta", "zeta")
    assert np.isclose(gij_mag.psi.values.min(), 0.0)
    assert np.isclose(gij_mag.psi.values.max(), 1.0)


def test_jacobian_accessor_returns_direct_and_inverse_forms():
    mag = _build_synthetic_magnetic_coordinates()
    _add_constant_jacobian(mag, value=2.0)

    jac = mag.jacobian(return_in="Rzphi")
    jac_inv = mag.jacobian(return_in="Rzphi", inverse=True)
    jac_mag = mag.jacobian(return_in="magnetic_coordinates")

    assert jac.dims == ("R", "z")
    assert np.allclose(jac.values, 2.0, rtol=0.0, atol=0.0)
    assert np.allclose(jac_inv.values, 0.5, rtol=0.0, atol=0.0)
    assert jac_mag.dims == ("psi", "theta", "zeta")


def test_jacobian_accessor_raises_when_missing():
    mag = _build_synthetic_magnetic_coordinates()

    with pytest.raises(ValueError):
        mag.jacobian()
