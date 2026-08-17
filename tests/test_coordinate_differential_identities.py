import numpy as np

from pycocos.coordinates.coordinate_map import SpectralCoordinateMap
from pycocos.core.equilibrium import equilibrium


_R_AXIS = 1.5
_Z_AXIS = 0.0
_SHAPE = np.array(
    [
        [1.0, 0.22],
        [0.28, 0.72],
    ],
    dtype=np.float64,
)
_SHAPE_INVERSE = np.linalg.inv(_SHAPE)


def _mapped_values(
    psi: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radial, angle = np.broadcast_arrays(
        np.asarray(psi, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
    )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    R = _R_AXIS + radial * (
        _SHAPE[0, 0] * cosine + _SHAPE[0, 1] * sine
    )
    z = _Z_AXIS + radial * (
        _SHAPE[1, 0] * cosine + _SHAPE[1, 1] * sine
    )
    nu = (
        0.06 * radial * np.sin(angle)
        + 0.025 * radial**2 * np.cos(2.0 * angle)
    )
    return R, z, nu


def _spectral_map() -> SpectralCoordinateMap:
    psi = np.linspace(0.05, 0.32, 17)
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    radial, angle = np.meshgrid(psi, theta, indexing="ij")
    R, z, nu = _mapped_values(radial, angle)
    return SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
    )


def _analytic_grad_psi(theta: np.ndarray) -> np.ndarray:
    angle = np.asarray(theta, dtype=np.float64)
    polar_gradient = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    return np.einsum("ij,...j->...i", _SHAPE_INVERSE.T, polar_gradient)


def _builder_result(core_indices=None):
    R_grid = np.linspace(1.0, 2.0, 61)
    z_grid = np.linspace(-0.5, 0.5, 61)
    RR, ZZ = np.meshgrid(R_grid, z_grid, indexing="ij")
    cylindrical = np.stack((RR - _R_AXIS, ZZ - _Z_AXIS), axis=-1)
    polar = np.einsum("ij,...j->...i", _SHAPE_INVERSE, cylindrical)
    psi_Rz = np.linalg.norm(polar, axis=-1)

    zeros = np.zeros_like(psi_Rz)
    eq = equilibrium(
        rgrid=R_grid,
        zgrid=z_grid,
        br=zeros,
        bz=zeros,
        bphi=np.full_like(psi_Rz, 2.0),
        psi=psi_Rz,
        Raxis=_R_AXIS,
        zaxis=_Z_AXIS,
        psi_edge=0.36,
        psi_ax=0.0,
    )

    psi = np.linspace(0.06, 0.32, 17)
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    radial, angle = np.meshgrid(psi, theta, indexing="ij")
    R, z, nu = _mapped_values(radial, angle)
    signed_jacobian = (
        -R * np.linalg.det(_SHAPE) * radial
    )

    return eq._build_magnetic_coordinates_dataset(
        psigrid=psi,
        thtable=np.broadcast_to(theta, radial.shape),
        nutable=nu,
        jac=signed_jacobian,
        Rtransform=R,
        ztransform=z,
        R_fine=R_grid,
        z_fine=z_grid,
        qprof=np.linspace(1.1, 1.7, psi.size),
        Fprof=np.linspace(2.0, 1.8, psi.size),
        Iprof=np.linspace(0.2, 0.3, psi.size),
        ntht_pad=3,
        coordinate_system="boozer",
        core_indices=core_indices,
        radial_support_metadata={
            "family": "analytic",
            "requested_guard_surfaces": 3,
            "inner_guard_surfaces": (
                0 if core_indices is None else int(core_indices[0])
            ),
            "outer_guard_surfaces": (
                0
                if core_indices is None
                else int(psi.size - 1 - core_indices[-1])
            ),
        },
    )


def test_builder_keeps_hidden_radial_support_private():
    core_indices = np.arange(3, 14, dtype=np.int64)
    magnetic = _builder_result(core_indices=core_indices)

    assert magnetic.coords.sizes["psi0"] == core_indices.size
    assert magnetic.deriv.sizes["psi0"] == core_indices.size
    assert magnetic._coordinate_map.psi.size == 17  # noqa: SLF001
    assert magnetic.coords.attrs["radial_core_nsurface"] == core_indices.size
    assert magnetic.coords.attrs["radial_support_nsurface"] == 17
    assert magnetic.coords.attrs["radial_inner_guard_surfaces"] == 3
    assert magnetic.coords.attrs["radial_outer_guard_surfaces"] == 3
    assert (
        magnetic._coordinate_diagnostics["radial_support"]["core_nsurface"]  # noqa: SLF001
        == core_indices.size
    )


def test_hidden_radial_support_extends_rz_differential_domain():
    core_indices = np.arange(3, 14, dtype=np.int64)
    magnetic = _builder_result(core_indices=core_indices)

    R_grid = np.asarray(magnetic.coords.R, dtype=np.float64)
    z_grid = np.asarray(magnetic.coords.z, dtype=np.float64)
    RR, ZZ = np.meshgrid(R_grid, z_grid, indexing="ij")
    cylindrical = np.stack((RR - _R_AXIS, ZZ - _Z_AXIS), axis=-1)
    polar = np.einsum("ij,...j->...i", _SHAPE_INVERSE, cylindrical)
    psi_Rz = np.linalg.norm(polar, axis=-1)

    support_min = magnetic.coords.attrs["radial_support_psi_min"]
    support_max = magnetic.coords.attrs["radial_support_psi_max"]
    core_min = magnetic.coords.attrs["radial_core_psi_min"]
    core_max = magnetic.coords.attrs["radial_core_psi_max"]
    expected_support = (
        np.isfinite(psi_Rz)
        & (psi_Rz >= support_min)
        & (psi_Rz <= support_max)
    )
    expected_core = (
        np.isfinite(psi_Rz)
        & (psi_Rz >= core_min)
        & (psi_Rz <= core_max)
    )
    actual = np.asarray(
        magnetic.coords["inside_coordinate_domain"],
        dtype=bool,
    )

    np.testing.assert_array_equal(actual, expected_support)
    assert np.any(actual & ~expected_core)
    assert np.all(
        np.isfinite(
            np.asarray(magnetic.deriv["dTheta_dr"])[actual]
        )
    )
    for first, second in (
        ("psi", "theta"),
        ("psi", "zeta"),
        ("theta", "theta"),
    ):
        projected = magnetic.metric(
            first,
            second,
            tensor="contravariant",
            return_in="magnetic_coordinates",
        )
        assert np.all(np.isfinite(np.asarray(projected)[[0, -1]]))
    projected_jacobian = magnetic.jacobian(
        return_in="magnetic_coordinates",
    )
    assert np.all(
        np.isfinite(np.asarray(projected_jacobian)[[0, -1]])
    )


def _public_basis_matrices(magnetic):
    radius = np.broadcast_to(
        magnetic.coords.R.values[:, None],
        magnetic.deriv["dR_dpsi"].shape,
    )
    tangent = np.stack(
        (
            np.stack(
                (
                    magnetic.deriv["dR_dpsi"].values,
                    radius * magnetic.deriv["dphi_dpsi"].values,
                    magnetic.deriv["dz_dpsi"].values,
                ),
                axis=-1,
            ),
            np.stack(
                (
                    magnetic.deriv["dR_dtheta"].values,
                    radius * magnetic.deriv["dphi_dtheta"].values,
                    magnetic.deriv["dz_dtheta"].values,
                ),
                axis=-1,
            ),
            np.stack(
                (
                    magnetic.deriv["dR_dzeta"].values,
                    radius * magnetic.deriv["dphi_dzeta"].values,
                    magnetic.deriv["dz_dzeta"].values,
                ),
                axis=-1,
            ),
        ),
        axis=-1,
    )
    gradient = np.stack(
        (
            np.stack(
                (
                    magnetic.deriv["dPsi_dr"].values,
                    magnetic.deriv["dPsi_dphi"].values / radius,
                    magnetic.deriv["dPsi_dz"].values,
                ),
                axis=-1,
            ),
            np.stack(
                (
                    magnetic.deriv["dTheta_dr"].values,
                    magnetic.deriv["dTheta_dphi"].values / radius,
                    magnetic.deriv["dTheta_dz"].values,
                ),
                axis=-1,
            ),
            np.stack(
                (
                    magnetic.deriv["dzeta_dr"].values,
                    magnetic.deriv["dzeta_dphi"].values / radius,
                    magnetic.deriv["dzeta_dz"].values,
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )
    return radius, tangent, gradient


def test_asymmetric_spectral_map_obeys_flux_differential_identities():
    coordinate_map = _spectral_map()
    psi = np.array([0.075, 0.14, 0.23, 0.30])
    theta = np.array([0.31, 1.27, 3.88, 5.47])
    differential = coordinate_map.differentials(psi, theta)

    grad_psi_Rz = _analytic_grad_psi(theta)
    grad_psi = np.stack(
        (
            grad_psi_Rz[:, 0],
            np.zeros(theta.size),
            grad_psi_Rz[:, 1],
        ),
        axis=-1,
    )
    x_psi = differential.direct[..., :, 0]
    x_theta = differential.direct[..., :, 1]

    np.testing.assert_allclose(
        np.einsum("...a,...a->...", grad_psi, x_psi),
        1.0,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.einsum("...a,...a->...", grad_psi, x_theta),
        0.0,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        differential.inverse[..., 0, :],
        grad_psi,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_asymmetric_spectral_map_direct_inverse_and_jacobian_are_reciprocal():
    coordinate_map = _spectral_map()
    psi, theta = np.meshgrid(
        np.array([0.08, 0.16, 0.24, 0.30]),
        np.array([0.23, 1.41, 3.19, 5.61]),
        indexing="ij",
    )
    differential = coordinate_map.differentials(psi, theta)
    identity = np.broadcast_to(np.eye(3), differential.direct.shape)

    np.testing.assert_allclose(
        differential.inverse @ differential.direct,
        identity,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        differential.direct @ differential.inverse,
        identity,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    direct_det_Rz = (
        differential.inverse[..., 0, 0]
        * differential.inverse[..., 1, 2]
        - differential.inverse[..., 0, 2]
        * differential.inverse[..., 1, 0]
    )
    np.testing.assert_allclose(
        differential.jacobian,
        -differential.values["R"] / direct_det_Rz,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_equilibrium_builder_exports_one_reciprocal_differential_system():
    magnetic = _builder_result()
    mask = magnetic.coords["inside_coordinate_domain"].values
    radius, tangent, gradient = _public_basis_matrices(magnetic)

    grad_psi = gradient[..., 0, :]
    np.testing.assert_allclose(
        np.einsum("...a,...a->...", grad_psi[mask], tangent[..., :, 0][mask]),
        1.0,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        np.einsum("...a,...a->...", grad_psi[mask], tangent[..., :, 1][mask]),
        0.0,
        rtol=3.0e-12,
        atol=3.0e-12,
    )

    identity = np.broadcast_to(np.eye(3), tangent[mask].shape)
    np.testing.assert_allclose(
        gradient[mask] @ tangent[mask],
        identity,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        tangent[mask] @ gradient[mask],
        identity,
        rtol=3.0e-12,
        atol=3.0e-12,
    )

    direct_det_Rz = magnetic.deriv["direct_det_Rz"].values
    np.testing.assert_allclose(
        magnetic.deriv["jacobian"].values[mask],
        -radius[mask] / direct_det_Rz[mask],
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        np.linalg.det(tangent[mask]),
        magnetic.deriv["jacobian"].values[mask],
        rtol=2.0e-8,
        atol=2.0e-10,
    )
