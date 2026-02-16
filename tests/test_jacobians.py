import numpy as np

from pycocos.coordinates.jacobian_builders import (
    boozer_consistency_residual,
    make_jacobian_context,
)
from pycocos.coordinates.jacobians import (
    compute_boozer_jacobian,
    compute_equal_arc_jacobian,
    compute_hamada_jacobian,
    compute_pest_jacobian,
)
from pycocos.coordinates.registry import (
    JACOBIAN_REGISTRY,
    get_jacobian_function,
    register_coordinate_system,
)


def _surface_context(coordinate_system: str, ntheta: int = 256):
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    R = 1.8 + 0.15 * np.cos(theta)
    B = 2.2 + 0.35 * np.sin(theta)
    Bpol = 0.6 + 0.08 * np.cos(2.0 * theta)
    dlp = np.full(ntheta, 2.0 * np.pi / ntheta)
    return make_jacobian_context(
        coordinate_system=coordinate_system,
        R=R,
        B=B,
        Bpol=Bpol,
        dlp=dlp,
        I=0.9,
        F=2.3,
        q=1.6,
    )


def test_boozer_uses_h_over_b2_context_api():
    ctx = _surface_context("boozer")
    jac = compute_boozer_jacobian(ctx)
    h = ctx["I"] + ctx["q"] * ctx["F"]
    assert np.all(np.isfinite(jac))
    assert np.allclose(jac * (ctx["B"] ** 2), h, rtol=1.0e-11, atol=1.0e-11)
    assert boozer_consistency_residual(ctx, jac) < 1.0e-12


def test_boozer_legacy_signature_kept():
    B = np.array([2.0, 2.5, 3.0], dtype=float)
    jac = compute_boozer_jacobian(np.array([1.0]), np.array([2.0]), np.array([0.5]), B)
    assert np.allclose(jac, (1.0 + 0.5 * 2.0) / (B**2))


def test_hamada_is_theta_independent():
    ctx = _surface_context("hamada")
    jac = compute_hamada_jacobian(ctx)
    assert np.all(np.isfinite(jac))
    assert np.max(np.abs(jac - jac.mean())) < 1.0e-12


def test_pest_has_r2_shape_up_to_surface_scale():
    ctx = _surface_context("pest")
    jac = compute_pest_jacobian(ctx)
    ratio = jac / (ctx["R"] ** 2)
    ratio = ratio[np.isfinite(ratio)]
    assert ratio.size > 0
    assert np.std(ratio) < 1.0e-10


def test_equal_arc_has_r_over_gradpsi_shape_up_to_surface_scale():
    ctx = _surface_context("equal_arc")
    jac = compute_equal_arc_jacobian(ctx)
    grad_psi = np.abs(ctx["R"] * ctx["Bpol"])
    target = ctx["R"] / grad_psi
    ratio = jac / target
    ratio = ratio[np.isfinite(ratio)]
    assert ratio.size > 0
    assert np.std(ratio) < 1.0e-10


def test_registry_wraps_legacy_custom_callable():
    name = "legacy_context_wrapper_test"

    def legacy_callable(I, F, q, B):
        return (I[0] + q[0] * F[0]) / (B**2)

    register_coordinate_system(name, legacy_callable)
    try:
        jacobian_func = get_jacobian_function(name)
        ctx = _surface_context(name)
        out = jacobian_func(ctx)
        expected = (ctx["I"] + ctx["q"] * ctx["F"]) / (ctx["B"] ** 2)
        assert out.shape == ctx["B"].shape
        assert np.allclose(out, expected)
    finally:
        JACOBIAN_REGISTRY.pop(name, None)

