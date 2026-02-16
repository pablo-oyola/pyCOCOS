import numpy as np

from pycocos.io.cocos import assign, cocos, transform_cocos


def test_assign_returns_valid_cocos_id():
    cocos_id = assign(
        q=2.0,
        ip=1.0e6,
        b0=2.5,
        psiaxis=0.0,
        psibndr=1.0,
        phiclockwise=False,
    )
    assert 1 <= cocos_id <= 18


def test_transform_identity_is_unity():
    cc = cocos(1)
    factors = transform_cocos(cc, cc)
    assert np.isclose(factors["PSI"], 1.0)
    assert np.isclose(factors["Q"], 1.0)
    assert np.isclose(factors["B"], 1.0)

