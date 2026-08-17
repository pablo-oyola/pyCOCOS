import inspect

import numpy as np
import pytest

from pycocos.io.cocos import (
    COCOSResolution,
    cocos,
    fromCocosNtoCocosM,
    identify_cocos,
    transform_cocos,
)


VALID_COCOS = tuple(range(1, 9)) + tuple(range(11, 19))


def _synthetic_eqdsk_dictionary():
    profile = np.array([1.0, 1.5, 2.0])
    psi = np.arange(12, dtype=float).reshape(4, 3) + 0.25
    return {
        "nx": 4,
        "ny": 3,
        "rdim": 1.2,
        "zdim": 0.8,
        "rcentr": 1.7,
        "rleft": 1.0,
        "zmid": 0.0,
        "rmagx": 1.5,
        "zmagx": 0.0,
        "simagx": 0.25,
        "sibdry": 1.25,
        "bcentr": 2.4,
        "cpasma": 1.0e6,
        "fpol": profile.copy(),
        "pres": 2.0 * profile,
        "ffprime": -0.2 * profile,
        "pprime": -0.1 * profile,
        "psi": psi,
        "qpsi": profile.copy(),
        "rbdry": np.array([1.1, 1.5, 1.9]),
        "zbdry": np.array([0.0, 0.3, 0.0]),
        "rlim": np.array([0.9, 2.1]),
        "zlim": np.array([-0.5, 0.5]),
    }
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, (1, 2, 11, 12)),
        ({"phiclockwise": False}, (1, 11)),
        ({"flux_normalization": "Wb/rad"}, (1, 2)),
        (
            {"phiclockwise": True, "flux_normalization": "Wb"},
            (12,),
        ),
    ],
)
def test_identify_cocos_retains_unresolved_candidates(kwargs, expected):
    resolution = identify_cocos(2.0, 1.0e6, 2.5, 0.0, 1.0, **kwargs)
    assert resolution.candidates == expected
    assert resolution.is_unique == (len(expected) == 1)
    assert resolution.cocos == (expected[0] if len(expected) == 1 else None)


@pytest.mark.parametrize(
    ("q", "psi_boundary", "expected"),
    [
        (1.0, 1.0, (1, 2)),
        (-1.0, -1.0, (3, 4)),
        (-1.0, 1.0, (5, 6)),
        (1.0, -1.0, (7, 8)),
    ],
)
def test_identify_cocos_sign_families(q, psi_boundary, expected):
    resolution = identify_cocos(
        q,
        1.0,
        1.0,
        0.0,
        psi_boundary,
        flux_normalization="Wb/rad",
    )
    assert resolution.candidates == expected


@pytest.mark.parametrize("name", ["q", "ip", "b0", "psi_difference"])
@pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
def test_identify_cocos_rejects_nonfinite_or_zero_sign_inputs(name, value):
    values = {
        "q": 1.0,
        "ip": 1.0,
        "b0": 1.0,
        "psiaxis": 0.0,
        "psibndr": 1.0,
    }
    if name == "psi_difference":
        values["psibndr"] = value
    else:
        values[name] = value
    with pytest.raises(ValueError, match="finite and nonzero"):
        identify_cocos(**values)


def test_identify_cocos_accepts_numpy_scalars():
    resolution = identify_cocos(
        np.float64(1.0),
        np.float64(1.0),
        np.float64(1.0),
        np.float64(0.0),
        np.float64(1.0),
        phiclockwise=np.bool_(False),
        flux_normalization="Wb/rad",
    )
    assert resolution.candidates == (1,)


def test_identify_cocos_rejects_invalid_orientation():
    with pytest.raises(TypeError, match="phiclockwise"):
        identify_cocos(1.0, 1.0, 1.0, 0.0, 1.0, phiclockwise="ccw")


def test_identify_cocos_rejects_invalid_flux_normalization():
    with pytest.raises(ValueError, match="flux_normalization"):
        identify_cocos(
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            flux_normalization="normalized",
        )


def test_identify_cocos_rejects_nonscalar_sign_input():
    with pytest.raises(TypeError, match="q must be a scalar"):
        identify_cocos(np.array([1.0, 2.0]), 1.0, 1.0, 0.0, 1.0)


def test_resolution_requires_unique_candidate():
    resolution = COCOSResolution((1, 2))
    with pytest.raises(ValueError, match=r"\(1, 2\)"):
        resolution.require_unique()


def test_new_identification_api_is_exported_from_pycocos_io():
    from pycocos.io import COCOSResolution as ExportedResolution
    from pycocos.io import identify_cocos as exported_identify_cocos

    assert ExportedResolution is COCOSResolution
    assert exported_identify_cocos is identify_cocos


def test_cocos_api_has_no_legacy_detection_surface():
    import pycocos.io as io

    assert not hasattr(io, "assign")
    assert not hasattr(cocos(1), "weberperrad")
    identify_signature = inspect.signature(identify_cocos)
    conversion_signature = inspect.signature(fromCocosNtoCocosM)
    assert tuple(identify_signature.parameters) == (
        "q",
        "ip",
        "b0",
        "psiaxis",
        "psibndr",
        "phiclockwise",
        "flux_normalization",
    )
    assert tuple(conversion_signature.parameters) == (
        "eqd",
        "cocos_m",
        "cocos_n",
    )
    assert all(
        parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        for parameter in identify_signature.parameters.values()
    )
    assert all(
        parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        for parameter in conversion_signature.parameters.values()
    )


def test_cocos_dictionary_conversion_requires_source_convention():
    with pytest.raises(TypeError, match="cocos_n"):
        fromCocosNtoCocosM(_synthetic_eqdsk_dictionary(), 1)


def test_transform_identity_is_unity():
    cc = cocos(1)
    factors = transform_cocos(cc, cc)
    assert np.isclose(factors["PSI"], 1.0)
    assert np.isclose(factors["Q"], 1.0)
    assert np.isclose(factors["B"], 1.0)


def test_transform_psi_has_one_length_and_field_scale():
    factors = transform_cocos(
        cocos(1),
        cocos(1),
        ld=(2.0, 3.0),
        lB=(4.0, 5.0),
    )
    expected = (5.0 / 4.0) * (3.0 / 2.0) ** 2
    assert np.isclose(factors["PSI"], expected)


@pytest.mark.parametrize("cocos_id", VALID_COCOS)
def test_cocos_descriptor_encodes_orientation_and_flux_normalization(cocos_id):
    descriptor = cocos(cocos_id)
    assert descriptor.phiclockwise == (cocos_id % 2 == 0)
    assert descriptor.flux_normalization == (
        "Wb" if cocos_id >= 11 else "Wb/rad"
    )


@pytest.mark.parametrize("cocos_id", [True, 1.0, "1"])
def test_cocos_descriptor_rejects_noninteger_ids(cocos_id):
    with pytest.raises(TypeError, match="integer COCOS ID"):
        cocos(cocos_id)


@pytest.mark.parametrize("cocos_id", VALID_COCOS)
def test_round_trip_each_cocos_through_internal_cocos_1(cocos_id):
    original = _synthetic_eqdsk_dictionary()
    internal = fromCocosNtoCocosM(
        original,
        cocos_m=1,
        cocos_n=cocos_id,
    )
    restored = fromCocosNtoCocosM(
        internal,
        cocos_m=cocos_id,
        cocos_n=1,
    )

    for key, expected in original.items():
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(restored[key], expected, rtol=1e-13, atol=1e-13)
        else:
            assert restored[key] == pytest.approx(expected)


@pytest.mark.parametrize("cocos_id", VALID_COCOS)
def test_transform_from_cocos_1_has_expected_signs(cocos_id):
    target = cocos(cocos_id)
    factors = transform_cocos(cocos(1), target)
    expected_psi = (
        target.sigma_RpZ
        * target.sigma_Bp
        * (2.0 * np.pi) ** target.exp_Bp
    )
    assert factors["PSI"] == pytest.approx(expected_psi)
    assert np.sign(factors["B"]) == target.sigma_RpZ
    assert np.sign(factors["F"]) == target.sigma_RpZ
    assert np.sign(factors["Q"]) == target.sigma_rhotp


@pytest.mark.parametrize("base_cocos", range(1, 9))
def test_wb_and_wb_per_radian_pairs_have_expected_two_pi_scaling(base_cocos):
    factors = transform_cocos(cocos(base_cocos), cocos(base_cocos + 10))
    assert factors["PSI"] == pytest.approx(2.0 * np.pi)
    assert factors["PPRIME"] == pytest.approx(1.0 / (2.0 * np.pi))
    assert factors["FFPRIME"] == pytest.approx(1.0 / (2.0 * np.pi))
