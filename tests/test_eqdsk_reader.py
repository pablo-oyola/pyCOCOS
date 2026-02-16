import numpy as np

from pycocos.io import eqdsk as eqdsk_mod


def test_read_eqdsk_smoke(monkeypatch, tmp_path):
    nx = 4
    ny = 3
    fake_geqdsk = {
        "comment": "fake equilibrium",
        "bcentr": 2.0,
        "cpasma": 1.0e6,
        "nx": nx,
        "ny": ny,
        "rbdry": np.array([1.2, 1.5, 1.8]),
        "zbdry": np.array([0.0, 0.2, 0.0]),
        "rmagx": 1.4,
        "zmagx": 0.0,
        "psi": np.linspace(0.0, 1.0, nx * ny).reshape(nx, ny),
        "simagx": 0.0,
        "sibdry": 1.0,
        "fpol": np.linspace(2.0, 1.6, nx),
        "pres": np.linspace(2e3, 0.0, nx),
        "ffprime": np.zeros(nx),
        "pprime": np.zeros(nx),
        "qpsi": np.linspace(1.0, 2.0, nx),
        "rleft": 1.0,
        "rdim": 1.0,
        "zmid": 0.0,
        "zdim": 1.0,
    }

    monkeypatch.setattr(eqdsk_mod.freeqdsk.geqdsk, "read", lambda _f: fake_geqdsk)
    monkeypatch.setattr(eqdsk_mod, "assign_cocos", lambda *args, **kwargs: 1)
    monkeypatch.setattr(eqdsk_mod, "fromCocosNtoCocosM", lambda data, _: data)

    file_path = tmp_path / "fake.geqdsk"
    file_path.write_text("fake file content\n", encoding="utf-8")

    out = eqdsk_mod.read_eqdsk(str(file_path), cocos=1)
    assert out["lr"] == nx
    assert out["lz"] == ny
    assert out["Rgrid"].shape == (nx,)
    assert out["zgrid"].shape == (ny,)
    assert out["fpolrz"].shape == (nx, ny)
    assert out["cocos_in"] == 1
    assert out["cocos_out"] == 1

