import numpy as np
import pytest

from pycocos.coordinates.coordinate_map import SpectralCoordinateMap
from pycocos.coordinates.coordinate_map_numba import (
    axisymmetric_differential_kernel,
)


def _asymmetric_map():
    psi = np.linspace(0.04, 0.81, 13)
    theta = np.linspace(0.0, 2.0 * np.pi, 129)
    radial = np.sqrt(psi)[:, None]
    angle = theta[None, :]
    R = (
        2.1
        + radial * np.cos(angle)
        + 0.08 * radial**2 * np.sin(2.0 * angle)
    )
    z = (
        1.35 * radial * np.sin(angle)
        + 0.06 * radial**2 * np.cos(2.0 * angle)
        + 0.04 * radial**2
    )
    nu = 0.03 * radial * np.sin(angle) + 0.01 * psi[:, None] * np.cos(3.0 * angle)
    return psi, theta, R, z, nu


def test_spectral_map_preserves_asymmetric_values_and_derivatives():
    psi, theta, R, z, nu = _asymmetric_map()
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )

    # Evaluate on represented radial surfaces so this test isolates the
    # periodic Fourier representation from radial interpolation error.
    psi_eval = psi[[2, 5, 9]]
    theta_eval = np.array([0.37, 1.41, 5.22])
    radial = np.sqrt(psi_eval)
    expected_R = (
        2.1
        + radial * np.cos(theta_eval)
        + 0.08 * psi_eval * np.sin(2.0 * theta_eval)
    )
    expected_z = (
        1.35 * radial * np.sin(theta_eval)
        + 0.06 * psi_eval * np.cos(2.0 * theta_eval)
        + 0.04 * psi_eval
    )
    expected_R_theta = (
        -radial * np.sin(theta_eval)
        + 0.16 * psi_eval * np.cos(2.0 * theta_eval)
    )
    expected_z_theta = (
        1.35 * radial * np.cos(theta_eval)
        - 0.12 * psi_eval * np.sin(2.0 * theta_eval)
    )

    np.testing.assert_allclose(
        coordinate_map.evaluate("R", psi_eval, theta_eval),
        expected_R,
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        coordinate_map.evaluate("z", psi_eval, theta_eval),
        expected_z,
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        coordinate_map.evaluate("R", psi_eval, theta_eval, dtheta=1),
        expected_R_theta,
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        coordinate_map.evaluate("z", psi_eval, theta_eval, dtheta=1),
        expected_z_theta,
        rtol=1.0e-10,
        atol=1.0e-11,
    )


def test_explicit_up_down_projection_rebuilds_one_symmetric_map():
    psi = np.linspace(0.04, 0.81, 13)
    theta = np.linspace(0.0, 2.0 * np.pi, 129)
    rho = np.sqrt(psi)[:, None]
    angle = theta[None, :]
    R = (
        2.1
        + rho * np.cos(angle)
        + 2.0e-4 * rho * np.sin(2.0 * angle)
    )
    z = (
        1.3 * rho * np.sin(angle)
        + 3.0e-4 * rho * np.cos(2.0 * angle)
    )
    nu = (
        0.03 * rho * np.sin(angle)
        + 1.0e-4 * rho * np.cos(angle)
    )
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
        R_axis=2.1,
        z_axis=0.0,
        enforce_up_down_symmetry=True,
        symmetry_tolerance=1.0e-3,
    )

    sample_theta = theta[:-1]
    sample_psi, sample_angle = np.meshgrid(
        psi[[2, 6, 10]],
        sample_theta,
        indexing="ij",
    )
    reflection = np.concatenate(
        ([0], np.arange(sample_theta.size - 1, 0, -1))
    )
    for field, parity in (("R", 1.0), ("z", -1.0), ("nu", -1.0)):
        values = coordinate_map.evaluate(field, sample_psi, sample_angle)
        reflected = np.take(values, reflection, axis=1)
        offset = 2.1 if field == "R" else 0.0
        np.testing.assert_allclose(
            values - offset,
            parity * (reflected - offset),
            rtol=0.0,
            atol=2.0e-13,
        )
        radial_derivative = coordinate_map.evaluate(
            field,
            sample_psi,
            sample_angle,
            dpsi=1,
        )
        np.testing.assert_allclose(
            radial_derivative,
            parity * np.take(radial_derivative, reflection, axis=1),
            rtol=0.0,
            atol=2.0e-12,
        )
        angular_derivative = coordinate_map.evaluate(
            field,
            sample_psi,
            sample_angle,
            dtheta=1,
        )
        np.testing.assert_allclose(
            angular_derivative,
            -parity * np.take(angular_derivative, reflection, axis=1),
            rtol=0.0,
            atol=2.0e-12,
        )

    assert coordinate_map.up_down_symmetry_audit["applied"]
    assert (
        np.max(
            coordinate_map.up_down_symmetry_audit["geometry_residual"]
        )
        < 1.0e-3
    )


def test_explicit_up_down_projection_rejects_asymmetric_map():
    psi, theta, R, z, nu = _asymmetric_map()
    with pytest.raises(ValueError, match="not sufficiently up-down symmetric"):
        SpectralCoordinateMap(
            psi=psi,
            theta=theta,
            R=R,
            z=z,
            nu=nu,
            R_axis=2.1,
            z_axis=0.0,
            enforce_up_down_symmetry=True,
            symmetry_tolerance=1.0e-4,
        )


def test_explicit_up_down_projection_requires_finite_gate():
    psi, theta, R, z, nu = _asymmetric_map()
    with pytest.raises(ValueError, match="symmetry_tolerance is required"):
        SpectralCoordinateMap(
            psi=psi,
            theta=theta,
            R=R,
            z=z,
            nu=nu,
            R_axis=2.1,
            z_axis=0.0,
            enforce_up_down_symmetry=True,
        )


def test_spectral_map_metrics_are_exact_reciprocals():
    psi, theta, R, z, nu = _asymmetric_map()
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )
    radial, angle = np.meshgrid(
        psi[2:-2],
        theta[3:-4:13],
        indexing="ij",
    )
    differentials = coordinate_map.differentials(radial, angle)

    identity = np.einsum(
        "...ik,...kj->...ij",
        differentials.metric_covariant,
        differentials.metric_contravariant,
    )
    expected = np.broadcast_to(np.eye(3), identity.shape)
    np.testing.assert_allclose(identity, expected, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        np.linalg.det(differentials.direct),
        differentials.jacobian,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_spectral_map_recovers_theta_on_asymmetric_surfaces():
    psi, theta, R, z, nu = _asymmetric_map()
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )
    psi_eval = np.array([0.12, 0.29, 0.53, 0.74])
    theta_expected = np.array([0.15, 1.9, 3.7, 5.8])
    values = coordinate_map.values(psi_eval, theta_expected)
    theta_found = coordinate_map.solve_theta(
        psi=psi_eval,
        R=values["R"],
        z=values["z"],
        initial_theta=theta_expected + 0.08,
    )
    error = np.angle(np.exp(1j * (theta_found - theta_expected)))
    np.testing.assert_allclose(error, 0.0, atol=2.0e-11)


def test_spectral_map_recovers_theta_from_opposite_initial_branch():
    psi, theta, R, z, nu = _asymmetric_map()
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )
    psi_eval = np.array([0.12, 0.29, 0.53, 0.74])
    theta_expected = np.array([0.15, 1.9, 3.7, 5.8])
    values = coordinate_map.values(psi_eval, theta_expected)
    theta_found = coordinate_map.solve_theta(
        psi=psi_eval,
        R=values["R"],
        z=values["z"],
        initial_theta=theta_expected + np.pi,
    )
    error = np.angle(np.exp(1j * (theta_found - theta_expected)))
    np.testing.assert_allclose(error, 0.0, atol=2.0e-11)


def test_spectral_map_recovers_clockwise_magnetic_angle_from_geometric_seed():
    psi = np.linspace(0.04, 0.81, 13)
    theta = np.linspace(0.0, 2.0 * np.pi, 129)
    radius = np.sqrt(psi)[:, None]
    angle = theta[None, :]
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=2.1 + radius * np.cos(angle),
        z=-1.3 * radius * np.sin(angle),
    )

    psi_eval = np.array([0.12, 0.29, 0.53, 0.74])
    theta_expected = np.array([0.15, 1.9, 3.7, 5.8])
    values = coordinate_map.values(psi_eval, theta_expected)
    geometric_seed = np.mod(-theta_expected, 2.0 * np.pi)
    theta_found = coordinate_map.solve_theta(
        psi=psi_eval,
        R=values["R"],
        z=values["z"],
        initial_theta=geometric_seed,
    )
    error = np.angle(np.exp(1j * (theta_found - theta_expected)))
    np.testing.assert_allclose(error, 0.0, atol=2.0e-11)


def test_sqrt_flux_fit_preserves_near_axis_physical_psi_derivative():
    psi = np.linspace(0.02, 0.9, 19) ** 2
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    rho = np.sqrt(psi)[:, None]
    angle = theta[None, :]
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=2.0 + rho * np.cos(angle),
        z=1.3 * rho * np.sin(angle),
        psi_axis=0.0,
        psi_boundary=1.0,
        R_axis=2.0,
        z_axis=0.0,
    )

    psi_eval = np.array([0.025**2, 0.08**2, 0.37**2, 0.83**2])
    theta_eval = np.array([0.2, 1.3, 3.8, 5.4])
    expected = np.cos(theta_eval) / (2.0 * np.sqrt(psi_eval))
    np.testing.assert_allclose(
        coordinate_map.evaluate(
            "R",
            psi_eval,
            theta_eval,
            dpsi=1,
        ),
        expected,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def _flux_constrained_elliptical_map():
    psi = np.linspace(0.04, 0.81, 17)
    theta = np.linspace(0.0, 2.0 * np.pi, 129)
    angle = theta[None, :]
    distortion = 0.035
    angular_scale = 1.0 + distortion * np.cos(2.0 * angle)
    radius = np.sqrt(psi)[:, None] * angular_scale
    R_axis = 2.0
    z_axis = -0.07
    elongation = 1.4
    R = R_axis + radius * np.cos(angle)
    z = z_axis + elongation * radius * np.sin(angle)
    nu = psi[:, None] * np.sin(angle)

    constraint_R = np.linspace(0.9, 3.1, 91)
    constraint_z = np.linspace(-1.65, 1.51, 97)
    RR, ZZ = np.meshgrid(constraint_R, constraint_z, indexing="ij")
    constraint_psi = (
        (RR - R_axis) ** 2
        + ((ZZ - z_axis) / elongation) ** 2
    )
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
        max_mode=3,
        psi_axis=0.0,
        psi_boundary=1.0,
        R_axis=R_axis,
        z_axis=z_axis,
        flux_constraint_R=constraint_R,
        flux_constraint_z=constraint_z,
        flux_constraint_psi=constraint_psi,
        flux_constraint_tolerance=1.0e-12,
    )
    return coordinate_map, distortion, elongation, R_axis, z_axis


def test_flux_constraint_relabels_values_and_exact_first_derivatives():
    (
        coordinate_map,
        distortion,
        elongation,
        R_axis,
        z_axis,
    ) = _flux_constrained_elliptical_map()
    psi = np.asarray([0.07, 0.23, 0.51, 0.76])
    theta = np.asarray([0.24, 1.37, 3.82, 5.41])
    radius = np.sqrt(psi)
    expected = {
        "R": R_axis + radius * np.cos(theta),
        "z": z_axis + elongation * radius * np.sin(theta),
        "R_psi": np.cos(theta) / (2.0 * radius),
        "z_psi": elongation * np.sin(theta) / (2.0 * radius),
        "R_theta": -radius * np.sin(theta),
        "z_theta": elongation * radius * np.cos(theta),
    }
    angular_scale = 1.0 + distortion * np.cos(2.0 * theta)
    expected_nu = psi * np.sin(theta) / angular_scale**2
    expected_nu_psi = np.sin(theta) / angular_scale**2
    expected_nu_theta = psi * (
        np.cos(theta) / angular_scale**2
        + 4.0
        * distortion
        * np.sin(theta)
        * np.sin(2.0 * theta)
        / angular_scale**3
    )

    np.testing.assert_allclose(
        coordinate_map.evaluate("R", psi, theta),
        expected["R"],
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        coordinate_map.evaluate("z", psi, theta),
        expected["z"],
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    for field, derivative, expected_name in (
        ("R", {"dpsi": 1}, "R_psi"),
        ("z", {"dpsi": 1}, "z_psi"),
        ("R", {"dtheta": 1}, "R_theta"),
        ("z", {"dtheta": 1}, "z_theta"),
    ):
        np.testing.assert_allclose(
            coordinate_map.evaluate(field, psi, theta, **derivative),
            expected[expected_name],
            rtol=5.0e-10,
            atol=5.0e-11,
        )
    np.testing.assert_allclose(
        coordinate_map.evaluate("nu", psi, theta),
        expected_nu,
        rtol=5.0e-10,
        atol=5.0e-11,
    )
    np.testing.assert_allclose(
        coordinate_map.evaluate("nu", psi, theta, dpsi=1),
        expected_nu_psi,
        rtol=5.0e-10,
        atol=5.0e-11,
    )
    np.testing.assert_allclose(
        coordinate_map.evaluate("nu", psi, theta, dtheta=1),
        expected_nu_theta,
        rtol=5.0e-10,
        atol=5.0e-11,
    )

    differential = coordinate_map.differentials(psi, theta)
    identity = np.einsum(
        "...ik,...kj->...ij",
        differential.metric_covariant,
        differential.metric_contravariant,
    )
    np.testing.assert_allclose(
        identity,
        np.broadcast_to(np.eye(3), identity.shape),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert np.all(np.isfinite(differential.jacobian))
    assert np.all(np.abs(differential.jacobian) > 0.0)

    values = coordinate_map.values(psi, theta)
    recovered = coordinate_map.solve_theta(
        psi=psi,
        R=values["R"],
        z=values["z"],
        initial_theta=theta + 0.08,
    )
    error = np.angle(np.exp(1j * (recovered - theta)))
    np.testing.assert_allclose(error, 0.0, atol=2.0e-11)

    audit = coordinate_map.flux_constraint_audit
    assert audit["applied"]
    assert audit["validation_normalized_residual"] <= 1.0e-12
    assert audit["validation_iterations"] <= 4
    assert audit["validation_minimum_abs_F_sigma"] > 0.0
    assert coordinate_map.flux_constraint_R is not None
    assert coordinate_map.flux_constraint_z is not None
    assert coordinate_map.flux_constraint_psi is not None


def test_flux_constraint_uses_bounded_roots_when_vector_newton_is_limited():
    psi = np.linspace(0.04, 0.81, 17)
    theta = np.linspace(0.0, 2.0 * np.pi, 129)
    angle = theta[None, :]
    radius = np.sqrt(psi)[:, None] * (1.0 + 0.035 * np.cos(2.0 * angle))
    R_axis = 2.0
    z_axis = -0.07
    elongation = 1.4
    R = R_axis + radius * np.cos(angle)
    z = z_axis + elongation * radius * np.sin(angle)
    constraint_R = np.linspace(0.9, 3.1, 91)
    constraint_z = np.linspace(-1.65, 1.51, 97)
    RR, ZZ = np.meshgrid(constraint_R, constraint_z, indexing="ij")
    constraint_psi = (
        (RR - R_axis) ** 2 + ((ZZ - z_axis) / elongation) ** 2
    )

    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        max_mode=3,
        psi_axis=0.0,
        psi_boundary=1.0,
        R_axis=R_axis,
        z_axis=z_axis,
        flux_constraint_R=constraint_R,
        flux_constraint_z=constraint_z,
        flux_constraint_psi=constraint_psi,
        flux_constraint_tolerance=1.0e-12,
        flux_constraint_max_iterations=1,
    )

    audit = coordinate_map.flux_constraint_audit
    assert audit["validation_bounded_root_fallback_count"] > 0
    assert audit["validation_bounded_root_fallback_iterations"] > 0
    assert audit["validation_normalized_residual"] <= 1.0e-12


def test_flux_constraint_angle_reseed_uses_constrained_geometry(monkeypatch):
    coordinate_map, *_ = _flux_constrained_elliptical_map()
    psi = np.asarray([0.07, 0.23, 0.51, 0.76])
    theta = np.asarray([0.24, 1.37, 3.82, 5.41])
    values = coordinate_map.values(psi, theta)
    original = coordinate_map._constrained_field_bundle
    sampled_shapes = []
    forced_initial_failures = 0

    def recorded_bundle(radial, angle):
        nonlocal forced_initial_failures
        sampled_shapes.append(np.broadcast_shapes(np.shape(radial), np.shape(angle)))
        values_out, radial_out, theta_out = original(radial, angle)
        if forced_initial_failures < 10 and np.ndim(angle) == 1:
            forced_initial_failures += 1
            theta_out = dict(theta_out)
            theta_out["R"] = np.full_like(theta_out["R"], np.nan)
        return values_out, radial_out, theta_out

    monkeypatch.setattr(coordinate_map, "_constrained_field_bundle", recorded_bundle)
    recovered = coordinate_map.solve_theta(
        psi=psi,
        R=values["R"],
        z=values["z"],
        initial_theta=theta + np.pi,
        max_iterations=10,
    )

    error = np.angle(np.exp(1j * (recovered - theta)))
    np.testing.assert_allclose(error, 0.0, atol=2.0e-11)
    assert any(shape == (psi.size, 64) for shape in sampled_shapes)


def test_flux_constraint_angle_bounded_fallback_handles_one_newton_step():
    coordinate_map, *_ = _flux_constrained_elliptical_map()
    psi = np.asarray([0.07, 0.23, 0.51, 0.76])
    theta = np.asarray([0.24, 1.37, 3.82, 5.41])
    values = coordinate_map.values(psi, theta)

    recovered = coordinate_map.solve_theta(
        psi=psi,
        R=values["R"],
        z=values["z"],
        initial_theta=theta + np.pi,
        max_iterations=1,
    )

    error = np.angle(np.exp(1j * (recovered - theta)))
    np.testing.assert_allclose(error, 0.0, atol=2.0e-11)


def test_flux_constraint_rejects_singular_radial_relabelling():
    psi = np.linspace(0.1, 0.8, 7)
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    angle = theta[None, :]
    R = np.broadcast_to(2.0 + 0.5 * np.cos(angle), (psi.size, theta.size))
    z = np.broadcast_to(0.5 * np.sin(angle), (psi.size, theta.size))
    constraint_R = np.linspace(1.0, 3.0, 41)
    constraint_z = np.linspace(-1.0, 1.0, 41)
    RR, ZZ = np.meshgrid(constraint_R, constraint_z, indexing="ij")
    constraint_psi = (RR - 2.0) ** 2 + ZZ**2

    with pytest.raises(ValueError, match=r"min\|F_sigma\|"):
        SpectralCoordinateMap(
            psi=psi,
            theta=theta,
            R=R,
            z=z,
            flux_constraint_R=constraint_R,
            flux_constraint_z=constraint_z,
            flux_constraint_psi=constraint_psi,
        )


def test_flux_constraint_cache_reuses_and_invalidates_mutated_inputs(monkeypatch):
    (
        coordinate_map,
        _,
        elongation,
        R_axis,
        z_axis,
    ) = _flux_constrained_elliptical_map()
    psi = np.asarray([0.09, 0.28, 0.63])
    theta = np.asarray([0.31, 2.07, 5.36])

    geometry_calls = 0
    original_constraint_geometry = coordinate_map._constraint_geometry

    def tracked_constraint_geometry(sigma, angle):
        nonlocal geometry_calls
        geometry_calls += 1
        return original_constraint_geometry(sigma, angle)

    monkeypatch.setattr(
        coordinate_map,
        "_constraint_geometry",
        tracked_constraint_geometry,
    )

    first = coordinate_map.evaluate("R", psi, theta)
    first_solve_calls = geometry_calls
    assert first_solve_calls > 0
    np.testing.assert_array_equal(
        coordinate_map.evaluate("R", psi, theta),
        first,
    )
    assert geometry_calls == first_solve_calls

    psi[0] = 0.16
    theta[-1] += 0.19
    updated_R = coordinate_map.evaluate("R", psi, theta)
    assert geometry_calls > first_solve_calls
    updated_solve_calls = geometry_calls
    np.testing.assert_allclose(
        updated_R,
        R_axis + np.sqrt(psi) * np.cos(theta),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        coordinate_map.evaluate("z", psi, theta),
        z_axis + elongation * np.sqrt(psi) * np.sin(theta),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    assert geometry_calls == updated_solve_calls


def test_omitted_flux_constraint_preserves_default_map_bitwise():
    psi, theta, R, z, nu = _asymmetric_map()
    baseline = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )
    explicit_default = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
        flux_constraint_R=None,
        flux_constraint_z=None,
        flux_constraint_psi=None,
    )
    psi_eval = np.asarray([0.12, 0.31, 0.67])
    theta_eval = np.asarray([0.27, 2.14, 5.62])
    for field in baseline.field_names:
        for derivative in ({}, {"dpsi": 1}, {"dtheta": 1}, {"dtheta": 2}):
            np.testing.assert_array_equal(
                baseline.evaluate(field, psi_eval, theta_eval, **derivative),
                explicit_default.evaluate(
                    field,
                    psi_eval,
                    theta_eval,
                    **derivative,
                ),
            )
    assert baseline.flux_constraint_audit == {"applied": False}
    assert explicit_default.flux_constraint_audit == {"applied": False}


def test_fused_field_bundle_matches_individual_spectral_evaluations():
    psi, theta, R, z, nu = _asymmetric_map()
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )
    radial = np.linspace(0.07, 0.77, 31)
    angle = np.linspace(0.11, 6.01, 31)
    orders = ((0, 0), (1, 0), (0, 1), (0, 2))

    bundle = coordinate_map._base_field_bundle(
        radial,
        angle,
        derivative_orders=orders,
    )

    for dpsi, dtheta in orders:
        for field in coordinate_map.field_names:
            np.testing.assert_allclose(
                bundle[(dpsi, dtheta)][field],
                coordinate_map._base_evaluate(
                    field,
                    radial,
                    angle,
                    dpsi=dpsi,
                    dtheta=dtheta,
                ),
                rtol=2.0e-14,
                atol=2.0e-14,
            )


def test_axisymmetric_differential_algebra_matches_dense_reference():
    psi, theta, R, z, nu = _asymmetric_map()
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )
    radial, angle = np.meshgrid(
        psi[2:-2],
        theta[3:-4:11],
        indexing="ij",
    )
    differential = coordinate_map.differentials(radial, angle)

    dense_inverse = np.linalg.inv(differential.direct)
    dense_covariant = np.einsum(
        "...ai,...aj->...ij",
        differential.direct,
        differential.direct,
    )
    dense_contravariant = np.einsum(
        "...ia,...ja->...ij",
        dense_inverse,
        dense_inverse,
    )
    np.testing.assert_allclose(
        differential.jacobian,
        np.linalg.det(differential.direct),
        rtol=3.0e-15,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        differential.inverse,
        dense_inverse,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        differential.metric_covariant,
        dense_covariant,
        rtol=3.0e-15,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        differential.metric_contravariant,
        dense_contravariant,
        rtol=5.0e-14,
        atol=5.0e-14,
    )


@pytest.mark.parametrize("constrained", [False, True])
def test_direct_differentials_match_full_coordinate_tangents(constrained):
    if constrained:
        coordinate_map, *_ = _flux_constrained_elliptical_map()
    else:
        psi, theta, R, z, nu = _asymmetric_map()
        coordinate_map = SpectralCoordinateMap(
            psi=psi,
            theta=theta,
            R=R,
            z=z,
            nu=nu,
        )
    radial = np.linspace(0.07, 0.76, 47)
    angle = np.linspace(0.13, 6.03, 47)

    tangents = coordinate_map.direct_differentials(radial, angle)
    full = coordinate_map.differentials(radial, angle)

    assert not hasattr(tangents, "inverse")
    assert not hasattr(tangents, "metric_covariant")
    assert not hasattr(tangents, "jacobian")
    for field in coordinate_map.field_names:
        np.testing.assert_allclose(
            tangents.values[field],
            full.values[field],
            rtol=2.0e-14,
            atol=2.0e-14,
        )
    np.testing.assert_allclose(
        tangents.direct,
        full.direct,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_compiled_axisymmetric_kernel_matches_numpy_path():
    psi, theta, R, z, nu = _asymmetric_map()
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )
    radial, angle = np.meshgrid(
        psi[2:-2],
        theta[3:-4:11],
        indexing="ij",
    )
    reference = coordinate_map.differentials(radial, angle)
    radius = reference.values["R"].reshape(-1)
    direct = reference.direct.reshape((-1, 3, 3))

    compiled = axisymmetric_differential_kernel(
        radius,
        direct[:, 0, 0],
        direct[:, 0, 1],
        direct[:, 2, 0],
        direct[:, 2, 1],
        -direct[:, 1, 0] / radius,
        -direct[:, 1, 1] / radius,
    )
    expected = (
        reference.direct,
        reference.inverse,
        reference.metric_covariant,
        reference.metric_contravariant,
        reference.jacobian,
    )
    for actual, target in zip(compiled, expected):
        np.testing.assert_allclose(
            actual.reshape(target.shape),
            target,
            rtol=5.0e-14,
            atol=5.0e-14,
        )


def test_flux_constraint_newton_stops_evaluating_converged_points():
    coordinate_map, *_ = _flux_constrained_elliptical_map()
    audit = coordinate_map.flux_constraint_audit

    assert audit["validation_newton_evaluation_rounds"] >= 2
    assert (
        audit["validation_newton_active_evaluations"]
        < audit["validation_newton_full_batch_equivalent_evaluations"]
    )


def test_theta_newton_stops_evaluating_converged_points(monkeypatch):
    coordinate_map, *_ = _flux_constrained_elliptical_map()
    radial = np.linspace(0.07, 0.76, 40)
    expected_theta = np.linspace(0.17, 5.97, radial.size)
    values = coordinate_map.values(radial, expected_theta)
    initial_theta = expected_theta.copy()
    initial_theta[radial.size // 2 :] += 0.2
    evaluated_sizes = []
    original = coordinate_map._constrained_field_bundle

    def recorded_bundle(psi, theta):
        evaluated_sizes.append(
            int(np.prod(np.broadcast_shapes(np.shape(psi), np.shape(theta))))
        )
        return original(psi, theta)

    monkeypatch.setattr(
        coordinate_map,
        "_constrained_field_bundle",
        recorded_bundle,
    )
    recovered = coordinate_map.solve_theta(
        psi=radial,
        R=values["R"],
        z=values["z"],
        initial_theta=initial_theta,
    )

    error = np.angle(np.exp(1j * (recovered - expected_theta)))
    np.testing.assert_allclose(error, 0.0, atol=2.0e-11)
    assert evaluated_sizes[0] == radial.size
    assert any(0 < size < radial.size for size in evaluated_sizes[1:])
    audit = coordinate_map.theta_inversion_audit
    assert (
        audit["newton_active_evaluations"]
        < audit["newton_full_batch_equivalent_evaluations"]
    )
    assert audit["failed_count"] == 0


def test_coordinate_map_state_round_trip_uses_only_non_object_arrays(tmp_path):
    coordinate_map, *_ = _flux_constrained_elliptical_map()
    state = coordinate_map.to_state()
    assert all(np.asarray(value).dtype != object for value in state.values())

    checkpoint = tmp_path / "coordinate_map_state.npz"
    np.savez(checkpoint, **state)
    with np.load(checkpoint, allow_pickle=False) as loaded:
        restored = SpectralCoordinateMap.from_state(loaded)

    radial = np.linspace(0.07, 0.76, 23)
    angle = np.linspace(0.13, 6.03, 23)
    original = coordinate_map.differentials(radial, angle)
    rebuilt = restored.differentials(radial, angle)
    for field in coordinate_map.field_names:
        np.testing.assert_allclose(
            rebuilt.values[field],
            original.values[field],
            rtol=2.0e-14,
            atol=2.0e-14,
        )
    np.testing.assert_allclose(
        rebuilt.direct,
        original.direct,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        rebuilt.inverse,
        original.inverse,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert restored.max_mode == coordinate_map.max_mode
    assert (
        restored.flux_constraint_tolerance
        == coordinate_map.flux_constraint_tolerance
    )
