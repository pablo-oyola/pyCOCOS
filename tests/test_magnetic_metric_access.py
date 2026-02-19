import numpy as np
import xarray as xr
import pytest

from test_magnetic_scalar_transform import _build_synthetic_magnetic_coordinates


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

    assert gij_mag.dims == ("psi", "theta", "nu")
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
    assert jac_mag.dims == ("psi", "theta", "nu")


def test_jacobian_accessor_raises_when_missing():
    mag = _build_synthetic_magnetic_coordinates()

    with pytest.raises(ValueError):
        mag.jacobian()
