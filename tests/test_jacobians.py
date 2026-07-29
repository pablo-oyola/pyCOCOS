import numpy as np
import pytest

from pycocos.coordinates.jacobian_builders import (
    boozer_consistency_residual,
    make_jacobian_context,
    normalize_jacobian_to_two_pi,
    validate_jacobian,
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


def test_boozer_rejects_removed_four_argument_signature():
    B = np.array([2.0, 2.5, 3.0], dtype=float)
    with pytest.raises(TypeError):
        compute_boozer_jacobian(
            np.array([1.0]),
            np.array([2.0]),
            np.array([0.5]),
            B,
        )


def test_hamada_is_theta_independent():
    ctx = _surface_context("hamada")
    jac = compute_hamada_jacobian(ctx)
    assert np.array_equal(jac, np.ones_like(jac))


def test_pest_returns_raw_r2_shape():
    ctx = _surface_context("pest")
    jac = compute_pest_jacobian(ctx)
    assert np.allclose(jac, ctx["R"] ** 2)


def test_equal_arc_returns_raw_r_over_gradpsi_shape():
    ctx = _surface_context("equal_arc")
    jac = compute_equal_arc_jacobian(ctx)
    grad_psi = np.abs(ctx["R"] * ctx["Bpol"])
    target = ctx["R"] / grad_psi
    assert np.allclose(jac, target)


@pytest.mark.parametrize(
    "bad_callable",
    [
        lambda I, F, q, B: (I + q * F) / (B**2),
        lambda *args: np.asarray(args),
        lambda context, **kwargs: context["B"],
    ],
)
def test_registry_rejects_non_context_only_callables(bad_callable):
    with pytest.raises(TypeError, match="exactly one positional parameter"):
        register_coordinate_system("invalid_signature_test", bad_callable)


@pytest.mark.parametrize(
    ("bad_jacobian", "message"),
    [
        (np.full(256, np.nan), "finite"),
        (np.zeros(256), "nonzero"),
        (np.r_[np.ones(128), -np.ones(128)], "one sign"),
        (np.ones((16, 16)), "1D array"),
        (np.ones(255), "matching"),
    ],
)
def test_validate_jacobian_rejects_invalid_surface_arrays(bad_jacobian, message):
    ctx = _surface_context("validation")
    with pytest.raises(ValueError, match=message):
        validate_jacobian(ctx, bad_jacobian)


def test_custom_shape_normalization_closes_at_two_pi_and_is_scale_invariant():
    name = "custom_shape_normalization_test"

    def custom_shape(context):
        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            context["B"].size,
            endpoint=False,
        )
        return 0.8 + 0.2 * np.cos(theta) - 0.05 * np.sin(2.0 * theta)

    register_coordinate_system(name, custom_shape)
    try:
        jacobian_func = get_jacobian_function(name)
        ctx = _surface_context(name)
        raw = jacobian_func(ctx)
        normalized = normalize_jacobian_to_two_pi(ctx, raw)
        scaled_normalized = normalize_jacobian_to_two_pi(ctx, 17.0 * raw)

        grad_psi = np.abs(ctx["R"] * ctx["Bpol"])
        span = np.sum(
            ctx["R"] / (np.abs(normalized) * grad_psi) * ctx["dlp"]
        )

        assert span == pytest.approx(2.0 * np.pi, rel=1.0e-13)
        assert np.allclose(normalized, scaled_normalized, rtol=1.0e-13)
        assert get_jacobian_function(name) is custom_shape
    finally:
        JACOBIAN_REGISTRY.pop(name, None)


def test_normalization_preserves_negative_orientation():
    ctx = _surface_context("negative_orientation")
    normalized = normalize_jacobian_to_two_pi(
        ctx,
        -np.ones_like(ctx["B"]),
    )
    assert np.all(normalized < 0.0)
