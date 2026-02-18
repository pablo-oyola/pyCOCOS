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


def _build_synthetic_magnetic_coordinates() -> magnetic_coordinates:
    R = np.linspace(1.0, 2.0, 24)
    z = np.linspace(-0.6, 0.6, 25)
    psi0 = np.linspace(1.05, 1.95, 12)
    thetageom = np.linspace(0.0, 2.0 * np.pi, 64)
    theta_star = np.linspace(0.0, 2.0 * np.pi, 64)

    RR, ZZ = np.meshgrid(R, z, indexing='ij')

    psi_2d = RR.copy()
    theta_table = np.tile(thetageom, (psi0.size, 1))
    nu_table = np.zeros_like(theta_table)

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
                attrs=_attrs('nu', 'rad', 'Magnetic toroidal angle', '$\\nu$'),
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
            'psi0': xr.DataArray(psi0, dims=('psi0',), attrs=_attrs('psi0', 'Wb', 'Reference flux', '$\\Psi_0$')),
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
    invR = 1.0 / RR

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
            'dphi_dzeta': xr.DataArray(invR, dims=('R', 'z'), coords={'R': R, 'z': z}, attrs=_attrs('dphi_dzeta', 'rad/rad', '', '')),
        }
    )

    return magnetic_coordinates(coords=coords, deriv=deriv, Raxis=0.0, zaxis=0.0, pad=0)


def test_scalar_transform_roundtrip_mag_cyl_mag():
    mag = _build_synthetic_magnetic_coordinates()

    psi = mag.coords.psi0.values
    theta = mag.coords.thetageom.values
    nu = np.linspace(0.0, 2.0 * np.pi, 16)

    field_values = psi[:, None, None] + 0.25 * np.cos(nu)[None, None, :]
    field_values = np.broadcast_to(field_values, (psi.size, theta.size, nu.size)).copy()

    field_mag = xr.DataArray(
        field_values,
        dims=('psi', 'theta', 'nu'),
        coords={'psi': psi, 'theta': theta, 'nu': nu},
        attrs={'name': 'test_scalar'},
    )

    field_cyl = mag.mag2cyl_scalar(
        field=field_mag,
        R=psi,
        z=mag.coords.z.values,
        phi=nu,
    )

    field_back = mag.cyl2mag_scalar(field_cyl)

    expected = field_back.psi.values[:, None, None] + 0.25 * np.cos(field_back.nu.values)[None, None, :]
    expected = np.broadcast_to(expected, field_back.shape)

    assert field_back.dims == ('psi', 'theta', 'nu')
    assert field_back.shape == (psi.size, theta.size, nu.size)
    assert np.allclose(field_back.values, expected, rtol=0.0, atol=1.0e-12)
    assert np.allclose(field_back.values, field_mag.values, rtol=0.0, atol=1.0e-12)


def test_mag2cyl_discards_points_outside_psi_domain():
    mag = _build_synthetic_magnetic_coordinates()

    psi = mag.coords.psi0.values
    theta = mag.coords.thetageom.values
    nu = np.linspace(0.0, 2.0 * np.pi, 8)

    field_values = psi[:, None, None] + 0.1 * np.sin(nu)[None, None, :]
    field_values = np.broadcast_to(field_values, (psi.size, theta.size, nu.size)).copy()
    field_mag = xr.DataArray(
        field_values,
        dims=('psi', 'theta', 'nu'),
        coords={'psi': psi, 'theta': theta, 'nu': nu},
    )

    # Internal valid psi range is [1.05, 1.95]; include outside points explicitly.
    R_eval = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
    z_eval = np.array([0.0])
    field_cyl = mag.mag2cyl_scalar(field_mag, R=R_eval, z=z_eval, phi=nu)

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
