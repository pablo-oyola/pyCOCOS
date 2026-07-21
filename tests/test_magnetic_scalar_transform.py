import numpy as np
import xarray as xr
import pytest

from pycocos.core.magnetic_coordinates import magnetic_coordinates


def _attrs(name: str, units: str, desc: str, short_name: str):
    return {
        'name': name,
        'units': units,
        'desc': desc,
        'short_name': short_name,
    }


def _build_synthetic_magnetic_coordinates(
    nu_value: float = 0.0,
) -> magnetic_coordinates:
    R = np.linspace(1.0, 2.0, 24)
    z = np.linspace(-0.6, 0.6, 25)
    psi0 = np.linspace(1.05, 1.95, 12)
    thetageom = np.linspace(0.0, 2.0 * np.pi, 64)
    theta_star = np.linspace(0.0, 2.0 * np.pi, 64)

    RR, ZZ = np.meshgrid(R, z, indexing='ij')

    psi_2d = RR.copy()
    theta_table = np.tile(thetageom, (psi0.size, 1))
    nu_table = np.full_like(theta_table, nu_value)

    R_inv = np.tile(psi0[:, None], (1, theta_star.size))
    z_inv = np.zeros_like(R_inv)

    coords = xr.Dataset(
        {
            'psi': xr.DataArray(
                psi_2d,
                dims=('R', 'z'),
                coords={'R': R, 'z': z},
                attrs=_attrs('psi', 'Wb', 'Poloidal flux', '$\\Psi$'),
            ),
            'theta': xr.DataArray(
                theta_table,
                dims=('psi0', 'thetageom'),
                coords={'psi0': psi0, 'thetageom': thetageom},
                attrs=_attrs('theta', 'rad', 'Magnetic poloidal angle', '$\\Theta^*$'),
            ),
            'nu': xr.DataArray(
                nu_table,
                dims=('psi0', 'thetageom'),
                coords={'psi0': psi0, 'thetageom': thetageom},
                attrs={
                    **_attrs(
                        'nu',
                        'rad',
                        'Toroidal gauge shift',
                        '$\\nu$',
                    ),
                    'gauge_relation': 'zeta = phi + nu',
                },
            ),
            'R_inv': xr.DataArray(
                R_inv,
                dims=('psi0', 'theta_star'),
                coords={'psi0': psi0, 'theta_star': theta_star},
                attrs=_attrs('R_inv', 'm', 'R = R(psi, theta*)', '$R(\\Psi,\\Theta^*)$'),
            ),
            'z_inv': xr.DataArray(
                z_inv,
                dims=('psi0', 'theta_star'),
                coords={'psi0': psi0, 'theta_star': theta_star},
                attrs=_attrs('z_inv', 'm', 'z = z(psi, theta*)', '$z(\\Psi,\\Theta^*)$'),
            ),
        },
        coords={
            'R': xr.DataArray(R, dims=('R',), attrs=_attrs('R', 'm', 'Major radius', 'R')),
            'z': xr.DataArray(z, dims=('z',), attrs=_attrs('z', 'm', 'Height', 'z')),
            'psi0': xr.DataArray(
                psi0,
                dims=('psi0',),
                attrs={
                    **_attrs('psi0', 'Wb', 'Reference flux', '$\\Psi_0$'),
                    'psi_axis': float(psi0[0]),
                    'psi_boundary': float(psi0[-1]),
                },
            ),
            'thetageom': xr.DataArray(
                thetageom,
                dims=('thetageom',),
                attrs=_attrs('thetageom', 'rad', 'Geometrical poloidal angle', '$\\Theta_{geom}$'),
            ),
            'theta_star': xr.DataArray(
                theta_star,
                dims=('theta_star',),
                attrs=_attrs('theta_star', 'rad', 'Magnetic poloidal angle', '$\\Theta^*$'),
            ),
        },
    )

    ones = np.ones_like(psi_2d)
    zeros = np.zeros_like(psi_2d)
    deriv = xr.Dataset(
        {
            'R': xr.DataArray(RR, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('R', 'm', 'Major radius', 'R')),
            'dR_dpsi': xr.DataArray(ones, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dR_dpsi', 'm/Wb', '', '')),
            'dz_dpsi': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dz_dpsi', 'm/Wb', '', '')),
            'dphi_dpsi': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dphi_dpsi', 'rad/Wb', '', '')),
            'dR_dtheta': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dR_dtheta', 'm/rad', '', '')),
            'dz_dtheta': xr.DataArray(ones, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dz_dtheta', 'm/rad', '', '')),
            'dphi_dtheta': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dphi_dtheta', 'rad/rad', '', '')),
            'dR_dzeta': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dR_dzeta', 'm/rad', '', '')),
            'dz_dzeta': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dz_dzeta', 'm/rad', '', '')),
            'dphi_dzeta': xr.DataArray(ones, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dphi_dzeta', 'rad/rad', '', '')),
            'dPsi_dr': xr.DataArray(ones, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dPsi_dr', 'Wb/m', '', '')),
            'dPsi_dphi': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dPsi_dphi', 'Wb/rad', '', '')),
            'dPsi_dz': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dPsi_dz', 'Wb/m', '', '')),
            'dTheta_dr': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dTheta_dr', 'rad/m', '', '')),
            'dTheta_dphi': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dTheta_dphi', 'rad/rad', '', '')),
            'dTheta_dz': xr.DataArray(ones, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dTheta_dz', 'rad/m', '', '')),
            'dzeta_dr': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dzeta_dr', 'rad/m', '', '')),
            'dzeta_dphi': xr.DataArray(ones, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dzeta_dphi', 'rad/rad', '', '')),
            'dzeta_dz': xr.DataArray(zeros, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dzeta_dz', 'rad/m', '', '')),
        }
    )

    return magnetic_coordinates(
        coords=coords, deriv=deriv, Raxis=0.0, zaxis=0.0, pad=0
    )


def test_nu_is_the_canonical_gauge_shift():
    mag = _build_synthetic_magnetic_coordinates()

    assert "nu_shift" not in mag.coords
    assert mag.coords["nu"].attrs["name"] == "nu"
    assert mag.coords["nu"].attrs["gauge_relation"] == "zeta = phi + nu"

    transformed = mag(R=np.array([1.5]), z=np.array([0.0]))
    assert "nu_shift" not in transformed
    assert transformed["nu"].attrs["name"] == "nu"


def test_magnetic_coordinates_requires_nu_gauge_table():
    complete = _build_synthetic_magnetic_coordinates()
    incomplete_coords = complete.coords.drop_vars("nu")

    with pytest.raises(ValueError, match="gauge shift 'nu'"):
        magnetic_coordinates(
            coords=incomplete_coords,
            deriv=complete.deriv,
            Raxis=complete.Raxis,
            zaxis=complete.zaxis,
            pad=complete.nthtpad,
        )


def test_scalar_transform_roundtrip_mag_cyl_mag():
    mag = _build_synthetic_magnetic_coordinates()

    psi = mag.coords.psi0.values
    theta = mag.coords.thetageom.values
    zeta = np.linspace(0.0, 2.0 * np.pi, 16)

    field_values = psi[:, None, None] + 0.25 * np.cos(zeta)[None, None, :]
    field_values = np.broadcast_to(
        field_values,
        (psi.size, theta.size, zeta.size),
    ).copy()

    field_mag = xr.DataArray(
        field_values,
        dims=('psi', 'theta', 'zeta'),
        coords={'psi': psi, 'theta': theta, 'zeta': zeta},
        attrs={'name': 'test_scalar'},
    )

    field_cyl = mag.mag2cyl_scalar(
        field=field_mag,
        R=psi,
        z=mag.coords.z.values,
        phi=zeta,
    )

    field_back = mag.cyl2mag_scalar(field_cyl)

    expected = (
        field_back.psi.values[:, None, None]
        + 0.25 * np.cos(field_back.zeta.values)[None, None, :]
    )
    expected = np.broadcast_to(expected, field_back.shape)

    assert field_back.dims == ('psi', 'theta', 'zeta')
    assert field_back.shape == (psi.size, theta.size, zeta.size)
    assert field_back.zeta.attrs['name'] == 'zeta'
    assert field_back.zeta.attrs['gauge_relation'] == 'zeta = phi + nu'
    assert np.allclose(field_back.values, expected, rtol=0.0, atol=1.0e-12)
    assert np.allclose(field_back.values, field_mag.values, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("origin", [0.0, -np.pi])
def test_scalar_transform_wraps_nonzero_nu_on_endpoint_excluded_grid(origin):
    nzeta = 16
    zeta = origin + np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    nu_value = 2.0 * (zeta[1] - zeta[0])
    mag = _build_synthetic_magnetic_coordinates(nu_value=nu_value)

    psi = mag.coords.psi0.values
    theta = mag.coords.thetageom.values
    field_values = psi[:, None, None] + np.cos(zeta)[None, None, :]
    field_values = np.broadcast_to(
        field_values,
        (psi.size, theta.size, zeta.size),
    ).copy()
    field_mag = xr.DataArray(
        field_values,
        dims=('psi', 'theta', 'zeta'),
        coords={'psi': psi, 'theta': theta, 'zeta': zeta},
    )

    field_cyl = mag.mag2cyl_scalar(
        field=field_mag,
        R=psi,
        z=np.array([0.0]),
    )
    np.testing.assert_array_equal(field_cyl.phi.values, zeta)
    expected_cyl = psi[None, :, None] + np.cos(zeta + nu_value)[:, None, None]
    np.testing.assert_allclose(field_cyl.values, expected_cyl, rtol=0.0, atol=1.0e-12)

    field_back = mag.cyl2mag_scalar(field_cyl)
    expected_back = (
        field_back.psi.values[:, None, None]
        + np.cos(field_back.zeta.values)[None, None, :]
    )
    expected_back = np.broadcast_to(expected_back, field_back.shape)
    np.testing.assert_allclose(field_back.values, expected_back, rtol=0.0, atol=1.0e-12)


def test_mag2cyl_wraps_endpoint_excluded_theta_grid():
    mag = _build_synthetic_magnetic_coordinates()
    psi = mag.coords.psi0.values
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    zeta = np.array([0.0])
    field_values = np.cos(theta)[None, :, None]
    field_values = np.broadcast_to(
        field_values,
        (psi.size, theta.size, zeta.size),
    ).copy()
    field_mag = xr.DataArray(
        field_values,
        dims=('psi', 'theta', 'zeta'),
        coords={'psi': psi, 'theta': theta, 'zeta': zeta},
    )

    dtheta = theta[1] - theta[0]
    R_eval = 1.5
    z_eval = R_eval * np.tan(-0.1 * dtheta)
    field_cyl = mag.mag2cyl_scalar(
        field=field_mag,
        R=np.array([R_eval]),
        z=np.array([z_eval]),
        phi=zeta,
    )

    np.testing.assert_allclose(field_cyl.values, 1.0, rtol=0.0, atol=1.0e-12)


def test_mag2cyl_discards_points_outside_psi_domain():
    mag = _build_synthetic_magnetic_coordinates()

    psi = mag.coords.psi0.values
    theta = mag.coords.thetageom.values
    zeta = np.linspace(0.0, 2.0 * np.pi, 8)

    field_values = psi[:, None, None] + 0.1 * np.sin(zeta)[None, None, :]
    field_values = np.broadcast_to(
        field_values,
        (psi.size, theta.size, zeta.size),
    ).copy()
    field_mag = xr.DataArray(
        field_values,
        dims=('psi', 'theta', 'zeta'),
        coords={'psi': psi, 'theta': theta, 'zeta': zeta},
    )

    # Internal valid psi range is [1.05, 1.95]; include outside points explicitly.
    R_eval = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
    z_eval = np.array([0.0])
    field_cyl = mag.mag2cyl_scalar(field_mag, R=R_eval, z=z_eval, phi=zeta)

    assert np.all(np.isnan(field_cyl.sel(R=1.0).values))
    assert np.all(np.isnan(field_cyl.sel(R=2.0).values))
    assert np.all(np.isfinite(field_cyl.sel(R=1.5).values))


def test_cyl2mag_scalar_returns_normalized_flux_coordinate():
    mag = _build_synthetic_magnetic_coordinates()

    R = mag.coords.R.values
    z = mag.coords.z.values
    phi = np.linspace(0.0, 2.0 * np.pi, 6)

    field_values = np.ones((R.size, z.size, phi.size))
    field_cyl = xr.DataArray(
        field_values,
        dims=('R', 'z', 'phi'),
        coords={'R': R, 'z': z, 'phi': phi},
    )

    out = mag.cyl2mag_scalar(field_cyl, return_psi_norm=True)

    assert np.isclose(out.psi.values.min(), 0.0)
    assert np.isclose(out.psi.values.max(), 1.0)
    assert out.psi.attrs['name'] == 'psi_norm'


def test_cyl2mag_scalar_returns_rhopol_coordinate():
    mag = _build_synthetic_magnetic_coordinates()

    R = mag.coords.R.values
    z = mag.coords.z.values
    phi = np.linspace(0.0, 2.0 * np.pi, 6)

    field_values = np.ones((R.size, z.size, phi.size))
    field_cyl = xr.DataArray(
        field_values,
        dims=('R', 'z', 'phi'),
        coords={'R': R, 'z': z, 'phi': phi},
    )

    out = mag.cyl2mag_scalar(field_cyl, return_rhopol=True)

    assert np.isclose(out.psi.values.min(), 0.0)
    assert np.isclose(out.psi.values.max(), 1.0)
    assert out.psi.attrs['name'] == 'rhopol'


def test_normalized_flux_mapping_preserves_descending_axis_to_boundary_direction():
    mag = _build_synthetic_magnetic_coordinates()
    mag.coords.psi0.attrs.update({
        'psi_axis': 1.95,
        'psi_boundary': 1.05,
        'normalization': (
            'psi_N = (psi - psi_axis) / (psi_boundary - psi_axis)'
        ),
    })

    flux_grid = mag._cyl2mag_build_flux_grid(  # noqa: SLF001 - mapping regression
        return_psi_norm=True,
        return_rhopol=False,
    )
    np.testing.assert_allclose(flux_grid['psi_eval'][[0, -1]], [1.95, 1.05])
    assert np.all(np.diff(flux_grid['psi_eval']) < 0.0)

    inverse = mag.transform_inverse(
        psi=np.array([0.0, 1.0]),
        thetamag=np.zeros(2),
        psi_is_norm=True,
    )
    # The synthetic inverse map is exactly R(psi, theta) = psi, so this also
    # verifies that normalized queries are converted to physical flux before
    # evaluating the strictly increasing spline coordinate.
    np.testing.assert_allclose(inverse['R_inv'].values, [1.95, 1.05])
    np.testing.assert_allclose(inverse['z_inv'].values, 0.0)

    rho_grid = mag._cyl2mag_build_flux_grid(  # noqa: SLF001 - mapping regression
        return_psi_norm=False,
        return_rhopol=True,
    )
    expected = 1.95 + rho_grid['Psi']**2 * (1.05 - 1.95)
    np.testing.assert_allclose(rho_grid['psi_eval'], expected)


def test_cyl2mag_scalar_rejects_conflicting_flux_coordinate_options():
    mag = _build_synthetic_magnetic_coordinates()

    R = mag.coords.R.values
    z = mag.coords.z.values
    phi = np.linspace(0.0, 2.0 * np.pi, 6)

    field_values = np.ones((R.size, z.size, phi.size))
    field_cyl = xr.DataArray(
        field_values,
        dims=('R', 'z', 'phi'),
        coords={'R': R, 'z': z, 'phi': phi},
    )

    with pytest.raises(ValueError):
        mag.cyl2mag_scalar(field_cyl, return_psi_norm=True, return_rhopol=True)


def test_cyl2mag_scalar_axis_is_theta_constant_for_rhopol_zero():
    mag = _build_synthetic_magnetic_coordinates()

    # Inject a small theta-dependent numerical perturbation on the axis row
    # of the inverse map to emulate interpolation noise near rhopol = 0.
    theta_star = mag.coords.theta_star.values
    mag.coords['R_inv'].values[0, :] = mag.coords.psi0.values[0] + 1.0e-3 * np.sin(theta_star)
    mag.coords['z_inv'].values[0, :] = 1.0e-3 * np.cos(theta_star)

    R = mag.coords.R.values
    z = mag.coords.z.values
    phi = np.linspace(0.0, 2.0 * np.pi, 8)
    RR, ZZ, PP = np.meshgrid(R, z, phi, indexing='ij')

    # Field with explicit R/z dependence so non-regular axis mapping would
    # create spurious theta variation.
    field_values = RR + 0.7 * ZZ + np.cos(PP)
    field_cyl = xr.DataArray(
        field_values,
        dims=('R', 'z', 'phi'),
        coords={'R': R, 'z': z, 'phi': phi},
    )

    out = mag.cyl2mag_scalar(field_cyl, return_rhopol=True)

    axis_slice = out.values[0, :, :]
    axis_reference = np.nanmean(axis_slice, axis=0, keepdims=True)
    assert np.allclose(axis_slice, axis_reference, rtol=0.0, atol=1.0e-12)


def test_cyl2mag_scalar_batched_ndarray_matches_scalar_calls():
    mag = _build_synthetic_magnetic_coordinates()

    R = mag.coords.R.values
    z = mag.coords.z.values
    phi = np.linspace(0.0, 2.0 * np.pi, 8)
    RR, ZZ, PP = np.meshgrid(R, z, phi, indexing='ij')

    f0 = RR + 0.3 * ZZ + np.cos(PP)
    f1 = 0.5 * RR - 0.2 * ZZ + np.sin(PP)
    field_batch = np.stack((f0, f1), axis=0)

    out_batch = mag.cyl2mag_scalar(field_batch, R=R, z=z, phi=phi)

    out0 = mag.cyl2mag_scalar(f0, R=R, z=z, phi=phi)
    out1 = mag.cyl2mag_scalar(f1, R=R, z=z, phi=phi)

    assert out_batch.dims == ('field', 'psi', 'theta', 'zeta')
    assert out_batch.shape[0] == 2
    assert np.allclose(out_batch.isel(field=0).values, out0.values, rtol=0.0, atol=1.0e-12)
    assert np.allclose(out_batch.isel(field=1).values, out1.values, rtol=0.0, atol=1.0e-12)


def test_cyl2mag_scalar_batched_dataarray_extra_dims():
    mag = _build_synthetic_magnetic_coordinates()

    R = mag.coords.R.values
    z = mag.coords.z.values
    phi = np.linspace(0.0, 2.0 * np.pi, 6)
    RR, ZZ, PP = np.meshgrid(R, z, phi, indexing='ij')

    base = RR + 0.1 * ZZ + np.cos(PP)
    vals = np.stack((base, 2.0 * base), axis=0)
    field_da = xr.DataArray(
        vals,
        dims=('channel', 'R', 'z', 'phi'),
        coords={'channel': ['A', 'B'], 'R': R, 'z': z, 'phi': phi},
        attrs={'name': 'batched_da'},
    ).transpose('R', 'channel', 'z', 'phi')

    out = mag.cyl2mag_scalar(field_da)
    out_a = mag.cyl2mag_scalar(field_da.sel(channel='A'))
    out_b = mag.cyl2mag_scalar(field_da.sel(channel='B'))

    assert out.dims == ('field', 'psi', 'theta', 'zeta')
    assert out.sizes['field'] == 2
    assert np.allclose(out.isel(field=0).values, out_a.values, rtol=0.0, atol=1.0e-12)
    assert np.allclose(out.isel(field=1).values, out_b.values, rtol=0.0, atol=1.0e-12)


def test_cyl2mag_scalar_dataset_multi_field_output():
    mag = _build_synthetic_magnetic_coordinates()

    R = mag.coords.R.values
    z = mag.coords.z.values
    phi = np.linspace(0.0, 2.0 * np.pi, 8)
    RR, ZZ, PP = np.meshgrid(R, z, phi, indexing='ij')

    a = xr.DataArray(RR + np.cos(PP), dims=('R', 'z', 'phi'), coords={'R': R, 'z': z, 'phi': phi})
    b = xr.DataArray(0.2 * ZZ + np.sin(PP), dims=('R', 'z', 'phi'), coords={'R': R, 'z': z, 'phi': phi})
    ds = xr.Dataset({'a': a, 'b': b})

    out = mag.cyl2mag_scalar(ds)

    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {'a', 'b'}
    assert out['a'].dims == ('psi', 'theta', 'zeta')
    assert out['b'].dims == ('psi', 'theta', 'zeta')
    assert np.allclose(out['a'].values, mag.cyl2mag_scalar(a).values, rtol=0.0, atol=1.0e-12)
    assert np.allclose(out['b'].values, mag.cyl2mag_scalar(b).values, rtol=0.0, atol=1.0e-12)


def test_cyl2mag_scalar_dict_multi_field_output():
    mag = _build_synthetic_magnetic_coordinates()

    R = mag.coords.R.values
    z = mag.coords.z.values
    phi = np.linspace(0.0, 2.0 * np.pi, 8)
    RR, ZZ, PP = np.meshgrid(R, z, phi, indexing='ij')

    a = RR + np.cos(PP)
    b = 0.2 * ZZ + np.sin(PP)

    out = mag.cyl2mag_scalar({'a': a, 'b': b}, R=R, z=z, phi=phi)

    assert isinstance(out, dict)
    assert set(out.keys()) == {'a', 'b'}
    assert out['a'].dims == ('psi', 'theta', 'zeta')
    assert out['b'].dims == ('psi', 'theta', 'zeta')
    assert np.allclose(out['a'].values, mag.cyl2mag_scalar(a, R=R, z=z, phi=phi).values, rtol=0.0, atol=1.0e-12)
    assert np.allclose(out['b'].values, mag.cyl2mag_scalar(b, R=R, z=z, phi=phi).values, rtol=0.0, atol=1.0e-12)


def test_cyl2mag_dataarray_with_field_dimension():
    """Batch dim named 'field' must not clash with internal stack dimension."""
    mag = _build_synthetic_magnetic_coordinates()
    R = mag.coords.R.values
    z = mag.coords.z.values
    da = xr.DataArray(
        np.arange(3 * len(R) * len(z), dtype=float).reshape(3, len(R), len(z)),
        dims=('field', 'R', 'z'),
        coords={'field': [0, 1, 2], 'R': R, 'z': z},
        attrs={'name': 'test'},
    )
    packed = mag._cyl2mag_pack_dataarray(da)
    assert packed['input_kind'] == 'batch_dataarray'
    assert packed['packed'].shape[0] == 3
    assert packed['specs'][0]['extra_dims'] == ['field']

    out = mag.cyl2mag_scalar(da)
    assert out.dims == ('field', 'psi', 'theta', 'zeta')
    assert out.sizes['field'] == 3
