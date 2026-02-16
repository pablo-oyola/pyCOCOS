import numpy as np

from pycocos.coordinates import compute_coordinates as compute_coordinates_mod


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
    bphiline = np.full_like(theta, 2.0)
    return rline, zline, brline, bzline, bphiline, len(theta)


def test_compute_magnetic_coordinates_boozer_path(monkeypatch):
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

