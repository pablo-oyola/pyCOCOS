import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from pycocos.coordinates.accuracy import CoordinateAccuracy
from pycocos.coordinates.surfaces import (
    build_flux_constrained_surfaces,
    project_contours_to_flux,
)


_STANDARD_FLUX_TOLERANCE = (
    CoordinateAccuracy.standard().surface_flux_tolerance
)


def _asymmetric_flux_grid():
    R = np.linspace(1.0, 3.2, 241)
    z = np.linspace(-1.3, 1.5, 261)
    RR, ZZ = np.meshgrid(R, z, indexing="ij")
    x = RR - 2.0
    # Nested but not up-down symmetric about any horizontal midplane.
    psi = x**2 + ((ZZ - 0.18 * x**2) / 1.25) ** 2
    return R, z, psi


def _raw_asymmetric_surfaces(psigrid, ntheta=257):
    angle = np.linspace(0.0, 2.0 * np.pi, ntheta)
    rows_R = []
    rows_z = []
    for psi in psigrid:
        radius = np.sqrt(psi)
        R = 2.0 + radius * np.cos(angle)
        z = 0.18 * (R - 2.0) ** 2 + 1.25 * radius * np.sin(angle)
        # Add a small tracing/reconstruction error that projection must remove.
        R = R + 2.0e-4 * np.sin(5.0 * angle)
        z = z - 1.5e-4 * np.cos(4.0 * angle)
        rows_R.append(R)
        rows_z.append(z)
    return np.asarray(rows_R), np.asarray(rows_z)


def test_flux_surface_builder_preserves_up_down_asymmetry_and_exact_flux():
    Rgrid, zgrid, psi_field = _asymmetric_flux_grid()
    psigrid = np.linspace(0.04, 0.64, 9)
    R_raw, z_raw = _raw_asymmetric_surfaces(psigrid)

    surfaces = build_flux_constrained_surfaces(
        Rgrid=Rgrid,
        zgrid=zgrid,
        psi_field=psi_field,
        psigrid=psigrid,
        R_raw=R_raw,
        z_raw=z_raw,
        ntheta=256,
        spectral_max_mode=12,
        flux_scale=1.0,
    )

    assert surfaces.R.shape == (psigrid.size, 256)
    assert surfaces.z.shape == surfaces.R.shape
    assert np.all(
        surfaces.normalized_flux_residual <= _STANDARD_FLUX_TOLERANCE
    )
    assert np.all(np.sign(surfaces.signed_area) == np.sign(surfaces.signed_area[0]))
    assert np.all(
        surfaces.R[:, 0]
        >= np.max(surfaces.R, axis=1) - 2.0e-9
    )

    x = surfaces.R - 2.0
    reconstructed_flux = x**2 + (
        (surfaces.z - 0.18 * x**2) / 1.25
    ) ** 2
    np.testing.assert_allclose(
        reconstructed_flux,
        np.broadcast_to(psigrid[:, None], reconstructed_flux.shape),
        rtol=0.0,
        atol=1.1 * _STANDARD_FLUX_TOLERANCE,
    )

    # The vertical center depends on R, so reflecting z about a constant
    # midplane cannot reproduce the same surface.
    vertical_mean = np.mean(surfaces.z, axis=1)
    assert np.max(np.abs(vertical_mean)) > 1.0e-3
    assert np.ptp(vertical_mean) > 1.0e-3


def test_flux_surface_builder_handles_descending_physical_flux_labels():
    Rgrid, zgrid, psi_positive = _asymmetric_flux_grid()
    psigrid_positive = np.linspace(0.04, 0.49, 7)
    R_raw, z_raw = _raw_asymmetric_surfaces(psigrid_positive)
    psigrid = -psigrid_positive

    surfaces = build_flux_constrained_surfaces(
        Rgrid=Rgrid,
        zgrid=zgrid,
        psi_field=-psi_positive,
        psigrid=psigrid,
        R_raw=R_raw,
        z_raw=z_raw,
        ntheta=192,
        spectral_max_mode=10,
        flux_scale=1.0,
    )

    np.testing.assert_array_equal(surfaces.psi, psigrid)
    assert np.all(
        surfaces.normalized_flux_residual <= _STANDARD_FLUX_TOLERANCE
    )


def test_flux_surface_builder_uses_one_horizontal_gauge_for_asymmetric_surfaces():
    Rgrid, zgrid, psi_field = _asymmetric_flux_grid()
    psigrid = np.linspace(0.04, 0.49, 7)
    R_raw, z_raw = _raw_asymmetric_surfaces(psigrid)

    surfaces = build_flux_constrained_surfaces(
        Rgrid=Rgrid,
        zgrid=zgrid,
        psi_field=psi_field,
        psigrid=psigrid,
        R_raw=R_raw,
        z_raw=z_raw,
        ntheta=192,
        spectral_max_mode=10,
        flux_scale=1.0,
        gauge_z=0.0,
    )

    np.testing.assert_allclose(surfaces.z[:, 0], 0.0, atol=2.0e-10)
    assert np.all(surfaces.R[:, 0] > 2.0)
    assert np.all(
        surfaces.normalized_flux_residual <= _STANDARD_FLUX_TOLERANCE
    )


def test_reflection_paired_surfaces_preserve_parity_and_flux_labels():
    Rgrid, zgrid, psi_field = _asymmetric_flux_grid()
    psigrid = np.linspace(0.04, 0.49, 7)
    R_raw, z_raw = _raw_asymmetric_surfaces(psigrid)
    reflection_z = 0.037

    surfaces = build_flux_constrained_surfaces(
        Rgrid=Rgrid,
        zgrid=zgrid,
        psi_field=psi_field,
        psigrid=psigrid,
        R_raw=R_raw,
        z_raw=z_raw,
        ntheta=192,
        spectral_max_mode=10,
        flux_scale=1.0,
        gauge_z=reflection_z,
        reflection_z=reflection_z,
    )

    reflection = np.concatenate(
        ([0], np.arange(surfaces.theta.size - 1, 0, -1))
    )
    scale = max(float(np.ptp(surfaces.R)), float(np.ptp(surfaces.z)))
    np.testing.assert_allclose(
        surfaces.R,
        surfaces.R[:, reflection],
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps * scale,
    )
    np.testing.assert_allclose(
        surfaces.z - reflection_z,
        -(surfaces.z[:, reflection] - reflection_z),
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps * scale,
    )

    spline = RectBivariateSpline(Rgrid, zgrid, psi_field, s=0.0)
    reflected_z = 2.0 * reflection_z - surfaces.z
    paired_flux = 0.5 * (
        spline.ev(surfaces.R.ravel(), surfaces.z.ravel())
        + spline.ev(surfaces.R.ravel(), reflected_z.ravel())
    ).reshape(surfaces.R.shape)
    np.testing.assert_allclose(
        paired_flux,
        np.broadcast_to(psigrid[:, None], paired_flux.shape),
        rtol=0.0,
        atol=1.1 * _STANDARD_FLUX_TOLERANCE,
    )
    assert np.all(
        surfaces.normalized_flux_residual <= _STANDARD_FLUX_TOLERANCE
    )


def test_flux_surface_builder_retains_explicit_strict_accuracy_profile():
    Rgrid, zgrid, psi_field = _asymmetric_flux_grid()
    psigrid = np.linspace(0.04, 0.36, 4)
    R_raw, z_raw = _raw_asymmetric_surfaces(psigrid)
    strict_tolerance = CoordinateAccuracy.strict().surface_flux_tolerance

    surfaces = build_flux_constrained_surfaces(
        Rgrid=Rgrid,
        zgrid=zgrid,
        psi_field=psi_field,
        psigrid=psigrid,
        R_raw=R_raw,
        z_raw=z_raw,
        ntheta=128,
        spectral_max_mode=10,
        flux_scale=1.0,
        projection_tolerance=strict_tolerance,
    )

    assert np.all(surfaces.normalized_flux_residual <= strict_tolerance)


def test_batched_flux_projection_stops_updating_converged_vertices():
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    exact = np.stack((np.cos(theta), 2.0 * np.cos(theta)))
    vertical = np.stack((np.sin(theta), 2.0 * np.sin(theta)))
    radial = exact.copy()
    radial[1, ::2] += 2.0e-2
    evaluation_sizes = []

    def psi_value(R, z):
        evaluation_sizes.append(np.asarray(R).size)
        return np.asarray(R) ** 2 + np.asarray(z) ** 2

    def psi_gradient(R, z):
        return 2.0 * np.asarray(R), 2.0 * np.asarray(z)

    projected_R, projected_z, residual = project_contours_to_flux(
        R=radial,
        z=vertical,
        target_psi=np.array([1.0, 4.0]),
        psi_value=psi_value,
        psi_gradient=psi_gradient,
        flux_scale=4.0,
        tolerance=1.0e-10,
    )

    assert evaluation_sizes[0] == radial.size
    assert min(evaluation_sizes[1:]) < evaluation_sizes[0]
    assert np.all(residual <= 1.0e-10)
    np.testing.assert_allclose(
        projected_R[0],
        radial[0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        projected_z[0],
        vertical[0],
        rtol=0.0,
        atol=0.0,
    )


def test_batched_flux_projection_rejects_nonfinite_flux_values():
    with pytest.raises(ValueError, match="non-finite psi"):
        project_contours_to_flux(
            R=np.ones((2, 8)),
            z=np.zeros((2, 8)),
            target_psi=np.array([1.0, 2.0]),
            psi_value=lambda R, z: np.full(np.asarray(R).shape, np.nan),
            psi_gradient=lambda R, z: (
                np.ones(np.asarray(R).shape),
                np.zeros(np.asarray(z).shape),
            ),
            flux_scale=1.0,
            tolerance=1.0e-7,
        )
