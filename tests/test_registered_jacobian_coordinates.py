import numpy as np
import pytest

from pycocos.coordinates import compute_coordinates as compute_coordinates_mod
from pycocos.coordinates.registry import (
    JACOBIAN_REGISTRY,
    get_jacobian_function,
    register_coordinate_system,
)


_TWO_PI = 2.0 * np.pi


def _fake_integrate_pol_field_line(
    grr,
    gzz,
    br,
    bz,
    bphi,
    R,
    zaxis,
    tol=1.0e-3,
    Nmax=100000,
    integration_sign=1,
):
    del grr, gzz, br, bz, bphi, tol, Nmax, integration_sign
    theta = np.linspace(0.0, _TWO_PI, 513)
    minor_radius = 0.11 + 0.015 * (R - 1.3)
    rline = R + minor_radius * np.cos(theta)
    zline = zaxis + minor_radius * np.sin(theta)

    # The poloidal field follows the positively oriented contour and F=R*Bphi
    # is a flux function. These identities make the Boozer closure check an
    # independent test of the same coordinate assembly used by custom shapes.
    brline = -np.sin(theta)
    bzline = np.cos(theta)
    bphiline = 2.0 / rline
    return rline, zline, brline, bzline, bphiline, theta.size


@pytest.fixture
def coordinate_inputs(monkeypatch):
    monkeypatch.setattr(
        compute_coordinates_mod,
        "integrate_pol_field_line",
        _fake_integrate_pol_field_line,
    )
    Rgrid = np.linspace(1.0, 2.2, 24)
    zgrid = np.linspace(-0.4, 0.4, 24)
    RR, _ = np.meshgrid(Rgrid, zgrid, indexing="ij")
    return {
        "Rgrid": Rgrid,
        "zgrid": zgrid,
        "br": np.zeros_like(RR),
        "bz": np.ones_like(RR),
        "bphi": 2.0 / RR,
        "raxis": 1.3,
        "zaxis": 0.0,
        "psigrid": np.array([0.2, 0.7]),
        "R_at_psi": np.array([1.35, 1.75]),
        "ltheta": 33,
        "n_theta_geom": 192,
        "spectral_max_mode": 16,
        "phiclockwise": True,
    }


def _run_registered(coordinate_inputs, coordinate_system):
    return compute_coordinates_mod.compute_magnetic_coordinates(
        **coordinate_inputs,
        coordinate_system=coordinate_system,
        jacobian_func=get_jacobian_function(coordinate_system),
    )


def _assert_closed_monotonic_coordinate(result):
    _, _, _, theta, nu, jacobian, R_inverse, z_inverse = result
    np.testing.assert_allclose(theta[:, 0], 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        theta[:, -1],
        _TWO_PI,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert np.all(np.diff(theta, axis=1) > 0.0)

    assert np.all(np.isfinite(jacobian))
    assert np.all(np.abs(jacobian) > 1.0e-14)
    assert np.all(np.sign(jacobian) == np.sign(jacobian[:, :1]))

    np.testing.assert_allclose(nu[:, -1], nu[:, 0], atol=1.0e-12)
    np.testing.assert_allclose(R_inverse[:, -1], R_inverse[:, 0], atol=1.0e-12)
    np.testing.assert_allclose(z_inverse[:, -1], z_inverse[:, 0], atol=1.0e-12)


@pytest.mark.parametrize(
    "coordinate_system",
    ["boozer", "pest", "equal_arc", "hamada"],
)
def test_registered_builtin_coordinates_close_and_are_monotonic(
    coordinate_inputs,
    coordinate_system,
):
    result = _run_registered(coordinate_inputs, coordinate_system)
    _assert_closed_monotonic_coordinate(result)


def test_registered_non_power_shape_is_scale_invariant_after_normalization(
    coordinate_inputs,
):
    base_name = "non_power_shape_e2e"
    scaled_name = "scaled_non_power_shape_e2e"

    def non_power_shape(context):
        normalized_R = context["R"] / np.mean(context["R"]) - 1.0
        normalized_B = context["B"] / np.mean(context["B"]) - 1.0
        return np.exp(0.4 * normalized_R) * (
            1.0 + 0.3 * normalized_B**2
        )

    def scaled_non_power_shape(context):
        return 37.0 * non_power_shape(context)

    register_coordinate_system(base_name, non_power_shape)
    register_coordinate_system(scaled_name, scaled_non_power_shape)
    try:
        base = _run_registered(coordinate_inputs, base_name)
        scaled = _run_registered(coordinate_inputs, scaled_name)

        _assert_closed_monotonic_coordinate(base)
        _assert_closed_monotonic_coordinate(scaled)
        assert np.ptp(base[5], axis=1).min() > 1.0e-5
        for base_values, scaled_values in zip(base, scaled):
            np.testing.assert_allclose(
                base_values,
                scaled_values,
                rtol=3.0e-13,
                atol=3.0e-13,
            )
    finally:
        JACOBIAN_REGISTRY.pop(base_name, None)
        JACOBIAN_REGISTRY.pop(scaled_name, None)


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("nonfinite", "finite"),
        ("zero", "nonzero"),
        ("mixed_sign", "one sign"),
        ("wrong_shape", "matching"),
    ],
)
def test_registered_invalid_jacobian_is_rejected_by_coordinate_assembly(
    coordinate_inputs,
    invalid_kind,
    message,
):
    name = f"invalid_{invalid_kind}_e2e"

    def invalid_jacobian(context):
        size = context["B"].size
        if invalid_kind == "nonfinite":
            return np.full(size, np.nan)
        if invalid_kind == "zero":
            return np.zeros(size)
        if invalid_kind == "mixed_sign":
            return np.where(np.arange(size) < size // 2, -1.0, 1.0)
        return np.ones(size - 1)

    register_coordinate_system(name, invalid_jacobian)
    try:
        with pytest.raises(ValueError, match=message):
            _run_registered(coordinate_inputs, name)
    finally:
        JACOBIAN_REGISTRY.pop(name, None)
