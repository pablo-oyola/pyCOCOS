import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from pycocos.coordinates.compute_coordinates import (
    _compute_surface_coordinate_row,
    _normalize_magnetic_angle_closure,
    _trace_flux_surfaces,
    compute_magnetic_coordinates,
)
from pycocos.coordinates.field_lines import get_field_line, integrate_pol_field_line


R_AXIS = 2.0
Z_AXIS = 0.0
F_TOROIDAL = -3.0


def _circular_axisymmetric_field():
    """Return a divergence-free field whose poloidal lines are circles."""
    R = np.linspace(1.25, 2.75, 151)
    z = np.linspace(-0.75, 0.75, 151)
    RR, ZZ = np.meshgrid(R, z, indexing="ij")
    Br = -ZZ / RR
    Bz = (RR - R_AXIS) / RR
    Bphi = F_TOROIDAL / RR
    return R, z, Br, Bz, Bphi


@pytest.mark.parametrize(
    "tracer",
    [get_field_line, integrate_pol_field_line],
    ids=["full-field", "poloidal"],
)
@pytest.mark.parametrize("integration_sign", [-1, 1])
def test_field_line_closes_after_full_turn_for_both_orientations(
    tracer,
    integration_sign,
):
    R, z, Br, Bz, Bphi = _circular_axisymmetric_field()
    radius = 0.5
    max_points = 10_000

    output = tracer(
        R,
        z,
        Br,
        Bz,
        Bphi,
        R_AXIS + radius,
        Z_AXIS,
        tol=1.0e-2,
        Nmax=max_points,
        integration_sign=integration_sign,
    )
    npoints = output[-1]
    Rline = output[0][:npoints]
    zline = output[1][:npoints]

    assert 0 < npoints < max_points
    assert np.sign(zline[0] - Z_AXIS) == integration_sign
    assert zline.min() < Z_AXIS - 0.95 * radius
    assert zline.max() > Z_AXIS + 0.95 * radius
    assert zline[-1] == pytest.approx(Z_AXIS, abs=1.0e-14)
    assert Rline[-1] == pytest.approx(R_AXIS + radius, abs=1.0e-4)

    # ``npoints`` is a count, not the final zero-based index: slicing through
    # it includes the interpolated closure point and excludes unused storage.
    assert output[0][npoints] == 0.0


@pytest.mark.parametrize("integration_sign", [-1, 1])
def test_surface_tracing_refines_underresolved_near_axis_contour(
    integration_sign,
):
    R, z, Br, Bz, Bphi = _circular_axisymmetric_field()
    radius = 1.0e-3

    traced = _trace_flux_surfaces(
        Rgrid=R,
        zgrid=z,
        br=Br,
        bz=Bz,
        bphi=Bphi,
        R_at_psi=np.array([R_AXIS + radius]),
        zaxis=Z_AXIS,
        ntheta=64,
        integration_sign=integration_sign,
        minimum_points=36,
    )

    R_surface, z_surface = traced[:2]
    assert R_surface.shape == (1, 64)
    assert z_surface.shape == (1, 64)
    assert np.all(np.isfinite(R_surface))
    assert np.all(np.isfinite(z_surface))
    reconstructed_radius = np.hypot(
        R_surface[0] - R_AXIS,
        z_surface[0] - Z_AXIS,
    )
    np.testing.assert_allclose(
        reconstructed_radius,
        radius,
        rtol=1.0e-2,
        atol=2.0e-6,
    )


def test_magnetic_angle_closure_normalizes_only_small_quadrature_drift():
    theta = np.linspace(0.0, 2.0 * np.pi * (1.0 + 5.0e-6), 65)
    normalized = _normalize_magnetic_angle_closure(theta)

    assert normalized[0] == 0.0
    assert normalized[-1] == pytest.approx(2.0 * np.pi)
    assert np.all(np.diff(normalized) > 0.0)

    invalid = np.linspace(0.0, 2.0 * np.pi * (1.0 + 5.0e-4), 65)
    with pytest.raises(ValueError, match="does not close"):
        _normalize_magnetic_angle_closure(invalid)


def _analytic_boozer_jacobian(context):
    B = np.asarray(context["B"], dtype=float)
    return (context["I"] + context["q"] * context["F"]) / B**2


def test_symmetric_surface_row_preserves_reflection_parity():
    ntheta_geom = 64
    theta_geom_full = np.linspace(0.0, 2.0 * np.pi, ntheta_geom + 1)
    theta_geom = theta_geom_full[:-1]
    theta_table = np.linspace(0.0, 2.0 * np.pi, 65)
    radius = 0.47
    R = R_AXIS + radius * np.cos(theta_geom)
    z = Z_AXIS + radius * np.sin(theta_geom)
    Br = -z / R
    Bz = (R - R_AXIS) / R
    Bphi = F_TOROIDAL / R

    _, _, _, theta, nu, jacobian, R_inverse, z_inverse = (
        _compute_surface_coordinate_row(
            R=R,
            z=z,
            br_surface=Br,
            bz_surface=Bz,
            bphi_surface=Bphi,
            thetageom=theta_geom_full,
            theta_eval=theta_geom,
            thgeogrid=theta_table,
            thmaggrid=theta_table,
            coordinate_system="boozer",
            jacobian_func=_analytic_boozer_jacobian,
            br_interp=None,
            bz_interp=None,
            bphi_interp=None,
        )
    )

    np.testing.assert_allclose(
        theta + theta[::-1],
        2.0 * np.pi,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(nu + nu[::-1], 0.0, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        jacobian - jacobian[::-1],
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        R_inverse - R_inverse[::-1],
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        z_inverse + z_inverse[::-1],
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_strong_registered_jacobian_keeps_interpolated_angle_monotonic():
    ntheta_geom = 64
    theta_geom_full = np.linspace(0.0, 2.0 * np.pi, ntheta_geom + 1)
    theta_geom = theta_geom_full[:-1]
    theta_table = np.linspace(0.0, 2.0 * np.pi, 513)
    radius = 0.47
    R = R_AXIS + radius * np.cos(theta_geom)
    z = Z_AXIS + radius * np.sin(theta_geom)
    Br = -z / R
    Bz = (R - R_AXIS) / R
    Bphi = F_TOROIDAL / R

    def strongly_varying_jacobian(context):
        angle = np.linspace(
            0.0,
            2.0 * np.pi,
            np.asarray(context["R"]).size,
            endpoint=False,
        )
        return np.exp(5.0 * np.cos(7.0 * angle))

    *_, theta, _, jacobian, _, _ = _compute_surface_coordinate_row(
        R=R,
        z=z,
        br_surface=Br,
        bz_surface=Bz,
        bphi_surface=Bphi,
        thetageom=theta_geom_full,
        theta_eval=theta_geom,
        thgeogrid=theta_table,
        thmaggrid=theta_table,
        coordinate_system="strong_custom",
        jacobian_func=strongly_varying_jacobian,
        br_interp=None,
        bz_interp=None,
        bphi_interp=None,
    )

    assert theta[0] == 0.0
    assert theta[-1] == pytest.approx(2.0 * np.pi)
    assert np.all(np.diff(theta) > 0.0)
    assert np.all(np.isfinite(jacobian))
    assert np.all(jacobian > 0.0)


def test_up_down_projection_precedes_coordinate_and_jacobian_construction():
    R, z, Br, Bz, Bphi = _circular_axisymmetric_field()
    RR, ZZ = np.meshgrid(R, z, indexing="ij")
    psi = 0.5 * ((RR - R_AXIS) ** 2 + ZZ**2) + 1.0e-3 * ZZ
    target_psi = np.asarray([0.045, 0.08, 0.125])
    diagnostics = {}

    rows = compute_magnetic_coordinates(
        Rgrid=R,
        zgrid=z,
        br=Br,
        bz=Bz,
        bphi=Bphi,
        raxis=R_AXIS,
        zaxis=Z_AXIS,
        psigrid=target_psi,
        ltheta=65,
        phiclockwise=True,
        jacobian_func=_analytic_boozer_jacobian,
        R_at_psi=R_AXIS + np.sqrt(2.0 * target_psi),
        coordinate_system="boozer",
        spectral_max_mode=12,
        n_theta_geom=512,
        psi_field=psi,
        flux_scale=float(np.ptp(psi)),
        enforce_up_down_symmetry=True,
        symmetry_tolerance=2.0e-2,
        diagnostics=diagnostics,
    )

    _, _, _, theta, nu, jacobian, R_inverse, z_inverse = rows
    for values, parity in (
        (R_inverse, 1.0),
        (z_inverse, -1.0),
        (nu, -1.0),
        (jacobian, 1.0),
    ):
        np.testing.assert_allclose(
            values,
            parity * values[:, ::-1],
            rtol=0.0,
            atol=2.0e-9,
        )
    np.testing.assert_allclose(
        theta + theta[:, ::-1],
        2.0 * np.pi,
        rtol=0.0,
        atol=2.0e-9,
    )
    audit = diagnostics["up_down_symmetry"]
    assert audit["applied"]
    assert 1.0e-5 < np.max(audit["geometry_residual"]) < 2.0e-2
    assert np.max(audit["projected_flux_residual"]) < 1.0e-8
    projected_psi = RectBivariateSpline(
        R,
        z,
        diagnostics["coordinate_psi_field"],
        kx=3,
        ky=3,
        s=0.0,
    )
    psi_on_inverse_map = projected_psi.ev(
        R_inverse.ravel(),
        z_inverse.ravel(),
    ).reshape(R_inverse.shape)
    np.testing.assert_allclose(
        psi_on_inverse_map,
        np.broadcast_to(target_psi[:, None], psi_on_inverse_map.shape),
        rtol=0.0,
        atol=1.0e-8 * float(np.ptp(psi)),
    )


def test_circular_surface_rows_satisfy_boozer_identities():
    """Exercise the five pointwise identities on a self-contained field."""
    ntheta_geom = 720
    ntheta_table = 121
    theta_geom_full = np.linspace(0.0, 2.0 * np.pi, ntheta_geom + 1)
    theta_geom = theta_geom_full[:-1]
    theta_table = np.linspace(0.0, 2.0 * np.pi, ntheta_table)

    # Descending radii make psi=-r^2/2 strictly increasing, as required by
    # the two-dimensional splines used below and by the production builder.
    radii = np.linspace(0.6, 0.25, 6)
    psi = -0.5 * radii**2
    rows = []
    for radius in radii:
        R = R_AXIS + radius * np.cos(theta_geom)
        z = Z_AXIS + radius * np.sin(theta_geom)
        Br = -z / R
        Bz = (R - R_AXIS) / R
        Bphi = F_TOROIDAL / R
        rows.append(
            _compute_surface_coordinate_row(
                R=R,
                z=z,
                br_surface=Br,
                bz_surface=Bz,
                bphi_surface=Bphi,
                thetageom=theta_geom_full,
                theta_eval=theta_geom,
                thgeogrid=theta_table,
                thmaggrid=theta_table,
                coordinate_system="boozer",
                jacobian_func=_analytic_boozer_jacobian,
                br_interp=None,
                bz_interp=None,
                bphi_interp=None,
            )
        )

    q, F, I, theta, nu, jacobian, _, z_inverse = [
        np.asarray(values) for values in zip(*rows)
    ]
    assert z_inverse.min() < Z_AXIS - 0.2
    assert z_inverse.max() > Z_AXIS + 0.2

    # Match the endpoint convention enforced by the production dataset
    # builder before differentiating the direct theta table.
    theta[:, 0] = 0.0
    theta[:, -1] = 2.0 * np.pi
    theta_spline = RectBivariateSpline(psi, theta_table, theta)
    nu_spline = RectBivariateSpline(psi, theta_table, nu)
    jacobian_spline = RectBivariateSpline(psi, theta_table, jacobian)

    psi_sample, alpha = np.meshgrid(
        psi[1:-1],
        theta_table[5:-5],
        indexing="ij",
    )
    radius = np.sqrt(-2.0 * psi_sample)
    R = R_AXIS + radius * np.cos(alpha)
    z = Z_AXIS + radius * np.sin(alpha)

    def evaluate(spline, dx=0, dy=0):
        return spline.ev(
            psi_sample.ravel(),
            alpha.ravel(),
            dx=dx,
            dy=dy,
        ).reshape(psi_sample.shape)

    theta_psi = evaluate(theta_spline, dx=1)
    theta_alpha = evaluate(theta_spline, dy=1)
    nu_psi = evaluate(nu_spline, dx=1)
    nu_alpha = evaluate(nu_spline, dy=1)
    J = evaluate(jacobian_spline)

    psi_R = -(R - R_AXIS)
    psi_z = -z
    alpha_R = -z / radius**2
    alpha_z = (R - R_AXIS) / radius**2
    grad_psi = np.stack((psi_R, np.zeros_like(R), psi_z), axis=-1)
    grad_theta = np.stack(
        (
            theta_psi * psi_R + theta_alpha * alpha_R,
            np.zeros_like(R),
            theta_psi * psi_z + theta_alpha * alpha_z,
        ),
        axis=-1,
    )
    grad_zeta = np.stack(
        (
            nu_psi * psi_R + nu_alpha * alpha_R,
            1.0 / R,
            nu_psi * psi_z + nu_alpha * alpha_z,
        ),
        axis=-1,
    )
    B = np.stack((-z / R, F_TOROIDAL / R, (R - R_AXIS) / R), axis=-1)
    B_cross_grad_psi = np.cross(B, grad_psi)

    q_sample = np.broadcast_to(q[1:-1, np.newaxis], psi_sample.shape)
    F_sample = np.broadcast_to(F[1:-1, np.newaxis], psi_sample.shape)
    I_sample = np.broadcast_to(I[1:-1, np.newaxis], psi_sample.shape)
    h_sample = I_sample + q_sample * F_sample
    lhs = (
        J * np.sum(B * grad_theta, axis=-1),
        J * np.sum(B * grad_zeta, axis=-1),
        J * np.sum(B * B, axis=-1),
        J * np.sum(B_cross_grad_psi * grad_theta, axis=-1),
        J * np.sum(B_cross_grad_psi * grad_zeta, axis=-1),
    )
    rhs = (
        np.ones_like(psi_sample),
        q_sample,
        h_sample,
        F_sample,
        -I_sample,
    )

    np.testing.assert_allclose(lhs[0], rhs[0], rtol=1.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(lhs[3], rhs[3], rtol=1.0e-5, atol=1.0e-7)
    for index in (1, 2, 4):
        np.testing.assert_allclose(lhs[index], rhs[index], rtol=1.0e-8, atol=1.0e-10)
