import inspect

import numpy as np
import pytest

from pycocos.coordinates import compute_coordinates as compute_coordinates_mod
from pycocos.coordinates.registry import get_jacobian_function


def test_surface_tolerance_preserves_historical_positional_parameter_order():
    parameter_names = list(
        inspect.signature(
            compute_coordinates_mod.compute_magnetic_coordinates
        ).parameters
    )
    assert parameter_names[-4:] == [
        "enforce_up_down_symmetry",
        "symmetry_tolerance",
        "diagnostics",
        "surface_projection_tolerance",
    ]


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
    theta = np.linspace(0.0, 2.0 * np.pi, 513)
    radius = 0.1
    rline = R + radius * np.cos(theta)
    zline = zaxis + radius * np.sin(theta)
    brline = -np.sin(theta)
    bzline = np.cos(theta)
    # Axisymmetric equilibrium requirement: F=R*Bphi is a flux function.
    bphiline = 2.0 / rline
    return rline, zline, brline, bzline, bphiline, len(theta)


@pytest.mark.parametrize("coordinate_system", ["boozer", "pest", "equal_arc", "hamada"])
def test_compute_magnetic_coordinates_coordinate_system_path(monkeypatch, coordinate_system):
    monkeypatch.setattr(
        compute_coordinates_mod,
        "integrate_pol_field_line",
        _fake_integrate_pol_field_line,
    )

    Rgrid = np.linspace(1.0, 2.0, 16)
    zgrid = np.linspace(-0.5, 0.5, 16)
    br = np.zeros((16, 16))
    bz = np.ones((16, 16))
    bphi = np.full((16, 16), 2.0)
    psigrid = np.linspace(0.1, 0.9, 5)

    out = compute_coordinates_mod.compute_magnetic_coordinates(
        Rgrid=Rgrid,
        zgrid=zgrid,
        br=br,
        bz=bz,
        bphi=bphi,
        raxis=1.4,
        zaxis=0.0,
        psigrid=psigrid,
        ltheta=64,
        phiclockwise=True,
        jacobian_func=get_jacobian_function(coordinate_system),
        coordinate_system=coordinate_system,
    )

    qprof, Fprof, Iprof, thtable, nutable, jacobian, Rtransform, ztransform = out
    assert qprof.shape == (5,)
    assert Fprof.shape == (5,)
    assert Iprof.shape == (5,)
    assert thtable.shape == (5, 64)
    assert nutable.shape == (5, 64)
    assert jacobian.shape == (5, 64)
    assert Rtransform.shape == (5, 64)
    assert ztransform.shape == (5, 64)
    assert np.isfinite(jacobian).all()


@pytest.mark.parametrize(
    ("requested", "expected", "automatic"),
    [
        (None, 512, True),
        (384, 384, False),
    ],
)
def test_surface_quadrature_is_workload_scaled_or_explicit(
    monkeypatch,
    requested,
    expected,
    automatic,
):
    monkeypatch.setattr(
        compute_coordinates_mod,
        "integrate_pol_field_line",
        _fake_integrate_pol_field_line,
    )
    Rgrid = np.linspace(1.0, 2.0, 16)
    zgrid = np.linspace(-0.5, 0.5, 16)
    br = np.zeros((16, 16))
    bz = np.ones((16, 16))
    bphi = np.full((16, 16), 2.0)
    diagnostics = {}

    compute_coordinates_mod.compute_magnetic_coordinates(
        Rgrid=Rgrid,
        zgrid=zgrid,
        br=br,
        bz=bz,
        bphi=bphi,
        raxis=1.4,
        zaxis=0.0,
        psigrid=np.linspace(0.1, 0.9, 3),
        ltheta=65,
        spectral_max_mode=12,
        n_theta_geom=requested,
        diagnostics=diagnostics,
    )

    audit = diagnostics["surface_construction"]
    assert audit["n_theta_geom"] == expected
    assert audit["minimum_n_theta_geom"] == 260
    assert audit["automatic_n_theta_geom"] is automatic
    assert audit["trace_passes"] == 1
    assert audit["flux_surface_build_passes"] == 0


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.inf, np.nan])
def test_surface_projection_tolerance_must_be_positive_and_finite(invalid):
    Rgrid = np.linspace(1.0, 2.0, 8)
    zgrid = np.linspace(-0.5, 0.5, 8)
    field = np.ones((8, 8))
    with pytest.raises(ValueError, match="surface_projection_tolerance"):
        compute_coordinates_mod.compute_magnetic_coordinates(
            Rgrid=Rgrid,
            zgrid=zgrid,
            br=field,
            bz=field,
            bphi=field,
            raxis=1.4,
            zaxis=0.0,
            psigrid=np.array([0.1, 0.2]),
            ltheta=16,
            surface_projection_tolerance=invalid,
        )
