import numpy as np

from pycocos.coordinates.coordinate_map import SpectralCoordinateMap


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
