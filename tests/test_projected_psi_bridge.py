import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

from pycocos.core.equilibrium import (
    _fit_projected_coordinate_psi_bridge,
    _reflection_paired_coordinate_psi_grid,
)
from pycocos.core.magnetic_coordinates import magnetic_coordinates


def test_projected_psi_bridge_preserves_surface_flux_on_public_grid():
    Rgrid = np.linspace(1.0, 3.0, 41)
    zgrid = np.linspace(-1.0, 1.0, 41)
    RR, ZZ = np.meshgrid(Rgrid, zgrid, indexing="ij")
    x = RR - 2.0
    quartic_weight = 0.04
    psi_field = (
        x**2
        + ZZ**2
        + quartic_weight * (x**4 + ZZ**4)
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    angular_quartic = quartic_weight * (cosine**4 + sine**4)
    surface_psi = np.linspace(0.04, 0.64, 7)
    radius_squared = np.asarray(
        [
            (-1.0 + np.sqrt(1.0 + 4.0 * angular_quartic * target))
            / (2.0 * angular_quartic)
            for target in surface_psi
        ]
    )
    surface_R = 2.0 + np.sqrt(radius_squared) * cosine
    surface_z = np.sqrt(radius_squared) * sine

    corrected, audit = _fit_projected_coordinate_psi_bridge(
        Rgrid=Rgrid,
        zgrid=zgrid,
        psi_field=psi_field,
        surface_R=surface_R,
        surface_z=surface_z,
        surface_psi=surface_psi,
        flux_scale=1.0,
        reflection_z=0.0,
    )

    assert corrected.shape == psi_field.shape
    assert np.max(audit["initial_residual"]) > 1.0e-8
    assert np.max(audit["final_residual"]) <= 1.0e-8
    assert audit["final_symmetry_residual"] <= 1.0e-8
    assert audit["relative_grid_correction"] < 1.0e-4

    bridge = RectBivariateSpline(
        Rgrid,
        zgrid,
        corrected,
        kx=3,
        ky=3,
        s=0.0,
    )
    sampled = bridge.ev(
        surface_R.ravel(),
        surface_z.ravel(),
    ).reshape(surface_R.shape)
    np.testing.assert_allclose(
        sampled,
        np.broadcast_to(surface_psi[:, None], sampled.shape),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_reflection_knot_union_preserves_full_surface_family_without_fit():
    """Exercise the 264-by-256 projected-label shape used by DIII-D 202020."""
    tracing_R = np.linspace(1.0, 3.0, 81)
    tracing_z = np.linspace(-1.1, 1.3, 83)
    public_R = np.linspace(1.0, 3.0, 41)
    reflection_z = -0.037
    RR, ZZ = np.meshgrid(tracing_R, tracing_z, indexing="ij")
    radial_offset = RR - 2.0
    vertical_offset = ZZ - reflection_z
    vertical_weight = 1.3
    # The odd term represents source up-down asymmetry.  The paired evaluator
    # cancels it exactly, leaving analytic nested elliptical flux surfaces.
    tracing_psi = (
        radial_offset**2
        + vertical_weight * vertical_offset**2
        + 0.02 * vertical_offset**3
    )

    surface_radius = np.linspace(0.04, 0.72, 264)
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    surface_R = 2.0 + surface_radius[:, None] * np.cos(theta)[None, :]
    surface_z = (
        reflection_z
        + surface_radius[:, None]
        * np.sin(theta)[None, :]
        / np.sqrt(vertical_weight)
    )
    surface_psi = surface_radius**2

    union_R, union_z, union_psi = _reflection_paired_coordinate_psi_grid(
        tracing_R=tracing_R,
        tracing_z=tracing_z,
        tracing_psi=tracing_psi,
        public_R=public_R,
        reflection_z=reflection_z,
    )
    assert union_R.size == public_R.size
    assert union_z.size > tracing_z.size
    assert np.ptp(np.diff(union_z)) > 0.0
    np.testing.assert_allclose(
        union_z + union_z[::-1],
        2.0 * reflection_z,
        rtol=0.0,
        atol=2.0e-14,
    )

    bridge = RectBivariateSpline(union_R, union_z, union_psi, kx=3, ky=3)
    sampled = bridge.ev(surface_R.ravel(), surface_z.ravel()).reshape(
        surface_R.shape
    )
    reflected = bridge.ev(
        surface_R.ravel(),
        (2.0 * reflection_z - surface_z).ravel(),
    ).reshape(surface_R.shape)
    np.testing.assert_allclose(
        sampled,
        np.broadcast_to(surface_psi[:, None], sampled.shape),
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(sampled, reflected, rtol=0.0, atol=2.0e-12)

    corrected, audit = _fit_projected_coordinate_psi_bridge(
        Rgrid=union_R,
        zgrid=union_z,
        psi_field=union_psi,
        surface_R=surface_R,
        surface_z=surface_z,
        surface_psi=surface_psi,
        flux_scale=1.0,
        reflection_z=reflection_z,
    )
    np.testing.assert_array_equal(corrected, union_psi)
    assert np.max(audit["final_residual"]) <= 1.0e-8
    assert audit["final_symmetry_residual"] <= 1.0e-8
    assert audit["solver_iterations"] == 0


class _ZeroCoordinateMap:
    psi = np.asarray([-10.0, 10.0])

    @staticmethod
    def solve_theta(*, psi, **_kwargs):
        return np.zeros_like(psi)

    @staticmethod
    def evaluate(_name, psi, _theta):
        return np.zeros_like(psi)


def _transform_fixture(*, cubic_psi: bool) -> magnetic_coordinates:
    radial = np.linspace(1.0, 2.0, 7)
    vertical = np.asarray([-0.8, -0.43, -0.11, 0.08, 0.41, 0.9])
    RR, ZZ = np.meshgrid(radial, vertical, indexing="ij")
    psi = RR**3 + 0.4 * ZZ**3 + 0.2 * RR * ZZ
    psi_attrs = {
        "name": "psi",
        "units": "Wb/rad",
        "desc": "Poloidal flux",
        "short_name": "psi",
    }
    if cubic_psi:
        psi_attrs.update(
            {
                "interpolation_order_R": 3,
                "interpolation_order_z": 3,
            }
        )
    coords = xr.Dataset(
        {
            "psi": xr.DataArray(
                psi,
                dims=("R", "z"),
                coords={"R": radial, "z": vertical},
                attrs=psi_attrs,
            ),
            "theta": xr.DataArray(0.0, attrs=psi_attrs),
            "nu": xr.DataArray(0.0, attrs=psi_attrs),
        }
    )
    fixture = object.__new__(magnetic_coordinates)
    fixture.coords = coords
    fixture.Raxis = 1.5
    fixture.zaxis = 0.0
    fixture._coordinate_map = _ZeroCoordinateMap()
    return fixture


def test_public_transform_honors_projected_cubic_psi_metadata():
    sample_R = np.asarray([1.17, 1.46, 1.83])
    sample_z = np.asarray([-0.35, 0.17, 0.62])
    expected = sample_R**3 + 0.4 * sample_z**3 + 0.2 * sample_R * sample_z

    cubic = _transform_fixture(cubic_psi=True)._transform(
        sample_R,
        sample_z,
        fill_nan=False,
    )
    linear = _transform_fixture(cubic_psi=False)._transform(
        sample_R,
        sample_z,
        fill_nan=False,
    )

    np.testing.assert_allclose(cubic.psi, expected, rtol=0.0, atol=2.0e-13)
    assert np.max(np.abs(np.asarray(linear.psi) - expected)) > 1.0e-4
