import json

import numpy as np
import pytest

from pycocos.coordinates.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    CheckpointIntegrityError,
    CheckpointMismatchError,
    coordinate_checkpoint_key,
    extract_array_group,
    load_coordinate_checkpoint,
    namespace_array_group,
    write_coordinate_checkpoint,
)
from pycocos.coordinates.coordinate_map import SpectralCoordinateMap


def _source_arrays():
    return {
        "R": np.linspace(1.0, 3.0, 5),
        "z": np.linspace(-1.0, 1.0, 7),
        "psi": np.arange(35, dtype=np.float64).reshape(5, 7),
    }


def _state_arrays():
    return {
        **namespace_array_group(
            "map",
            {
                "radial_knots": np.linspace(0.0, 1.0, 6),
                "fourier_modes": np.arange(4, dtype=np.int32),
            },
        ),
        **namespace_array_group(
            "dataset",
            {
                "theta": np.linspace(0.0, 2.0 * np.pi, 9),
                "labels": np.array(["psi", "theta", "zeta"]),
                "valid": np.array([True, False, True]),
                "iteration_count": np.array(7, dtype=np.int64),
            },
        ),
    }


def _config():
    return {
        "coordinate_system": "boozer",
        "accuracy": {"flux": 1.0e-6, "theta": np.float64(1.0e-8)},
        "resolution": (256, 512),
    }


def test_key_is_deterministic_over_mapping_order_layout_and_endianness():
    source = _source_arrays()
    first = coordinate_checkpoint_key(
        source,
        _config(),
        algorithm_version="spectral-map-v2",
    )

    reordered_source = {
        "psi": np.asfortranarray(source["psi"].astype(">f8")),
        "z": source["z"].astype(">f8"),
        "R": source["R"].copy(),
    }
    reordered_config = {
        "resolution": [256, 512],
        "accuracy": {"theta": 1.0e-8, "flux": 1.0e-6},
        "coordinate_system": "boozer",
    }
    second = coordinate_checkpoint_key(
        reordered_source,
        reordered_config,
        algorithm_version="spectral-map-v2",
    )

    assert first == second
    assert len(first) == 64


def test_write_and_verified_load_round_trip(tmp_path):
    checkpoint = write_coordinate_checkpoint(
        tmp_path,
        source_arrays=_source_arrays(),
        build_config=_config(),
        state_arrays=_state_arrays(),
        algorithm_version="spectral-map-v2",
        metadata={
            "array_groups": {
                "map": ["radial_knots", "fourier_modes"],
                "dataset": ["theta", "labels", "valid", "iteration_count"],
            },
            "dataset_dims": {"theta": ["theta"]},
        },
    )

    assert checkpoint.name == coordinate_checkpoint_key(
        _source_arrays(),
        _config(),
        algorithm_version="spectral-map-v2",
    )
    assert {entry.name for entry in checkpoint.iterdir()} == {
        "arrays.npz",
        "manifest.json",
        "manifest.sha256",
    }

    loaded = load_coordinate_checkpoint(
        checkpoint,
        expected_source_arrays=_source_arrays(),
        expected_build_config=_config(),
        expected_algorithm_version="spectral-map-v2",
    )
    map_state = loaded.array_group("map")
    dataset_state = extract_array_group(loaded.arrays, "dataset")
    np.testing.assert_array_equal(
        map_state["radial_knots"],
        np.linspace(0.0, 1.0, 6),
    )
    np.testing.assert_array_equal(
        map_state["fourier_modes"],
        np.arange(4, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        dataset_state["labels"],
        np.array(["psi", "theta", "zeta"]),
    )
    np.testing.assert_array_equal(
        dataset_state["valid"],
        np.array([True, False, True]),
    )
    assert dataset_state["iteration_count"].shape == ()
    assert dataset_state["iteration_count"].item() == 7
    assert loaded.metadata["dataset_dims"] == {"theta": ["theta"]}

    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["format"] == CHECKPOINT_FORMAT
    assert manifest["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert manifest["algorithm_version"] == "spectral-map-v2"
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["source_digest"]
    assert manifest["payload"]["sha256"]
    assert all(
        descriptor["sha256"]
        for descriptor in manifest["payload"]["arrays"].values()
    )


def test_spectral_coordinate_map_state_uses_generic_array_group(tmp_path):
    psi = np.linspace(0.04, 0.81, 7)
    theta = np.linspace(0.0, 2.0 * np.pi, 33)
    rho = np.sqrt(psi)[:, None]
    angle = theta[None, :]
    R = 2.1 + rho * np.cos(angle)
    z = 1.3 * rho * np.sin(angle)
    nu = 0.02 * rho * np.sin(angle)
    coordinate_map = SpectralCoordinateMap(
        psi=psi,
        theta=theta,
        R=R,
        z=z,
        nu=nu,
        max_mode=8,
    )
    state = namespace_array_group("spectral_map", coordinate_map.to_state())

    checkpoint = write_coordinate_checkpoint(
        tmp_path,
        source_arrays={"R_grid": R, "z_grid": z, "psi_grid": psi},
        build_config={"coordinate_system": "boozer", "max_mode": 8},
        state_arrays=state,
        algorithm_version="spectral-map-v2",
    )
    loaded = load_coordinate_checkpoint(checkpoint)
    restored = SpectralCoordinateMap.from_state(
        loaded.array_group("spectral_map")
    )

    psi_eval = np.array([0.09, 0.25, 0.64])
    theta_eval = np.array([0.3, 2.0, 5.1])
    for field in ("R", "z", "nu"):
        np.testing.assert_allclose(
            restored.evaluate(field, psi_eval, theta_eval),
            coordinate_map.evaluate(field, psi_eval, theta_eval),
            rtol=0.0,
            atol=2.0e-13,
        )


def test_existing_checkpoint_is_not_overwritten(tmp_path):
    kwargs = {
        "source_arrays": _source_arrays(),
        "build_config": _config(),
        "state_arrays": _state_arrays(),
        "algorithm_version": "spectral-map-v2",
    }
    checkpoint = write_coordinate_checkpoint(tmp_path, **kwargs)
    original_manifest = (checkpoint / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        write_coordinate_checkpoint(tmp_path, **kwargs)
    assert (checkpoint / "manifest.json").read_bytes() == original_manifest

    reused = write_coordinate_checkpoint(
        tmp_path,
        **kwargs,
        reuse_existing=True,
    )
    assert reused == checkpoint


def test_reuse_rejects_different_payload_for_the_same_input_key(tmp_path):
    kwargs = {
        "source_arrays": _source_arrays(),
        "build_config": _config(),
        "state_arrays": _state_arrays(),
        "algorithm_version": "spectral-map-v2",
    }
    write_coordinate_checkpoint(tmp_path, **kwargs)
    changed_state = _state_arrays()
    changed_state["map::radial_knots"] = np.linspace(0.0, 2.0, 6)

    with pytest.raises(FileExistsError, match="different payload"):
        write_coordinate_checkpoint(
            tmp_path,
            **{**kwargs, "state_arrays": changed_state},
            reuse_existing=True,
        )


def test_manifest_corruption_is_detected_before_json_is_trusted(tmp_path):
    checkpoint = write_coordinate_checkpoint(
        tmp_path,
        source_arrays=_source_arrays(),
        build_config=_config(),
        state_arrays=_state_arrays(),
        algorithm_version="spectral-map-v2",
    )
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(CheckpointIntegrityError, match="manifest SHA-256"):
        load_coordinate_checkpoint(checkpoint)


def test_payload_corruption_is_detected_before_npz_is_loaded(tmp_path):
    checkpoint = write_coordinate_checkpoint(
        tmp_path,
        source_arrays=_source_arrays(),
        build_config=_config(),
        state_arrays=_state_arrays(),
        algorithm_version="spectral-map-v2",
    )
    payload = checkpoint / "arrays.npz"
    data = bytearray(payload.read_bytes())
    data[len(data) // 2] ^= 0x01
    payload.write_bytes(data)

    with pytest.raises(CheckpointIntegrityError, match="payload SHA-256"):
        load_coordinate_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"expected_algorithm_version": "spectral-map-v3"}, "algorithm"),
        (
            {"expected_build_config": {**_config(), "resolution": [128, 256]}},
            "configuration",
        ),
        (
            {
                "expected_source_arrays": {
                    **_source_arrays(),
                    "R": np.linspace(1.1, 3.1, 5),
                }
            },
            "source arrays",
        ),
    ],
)
def test_load_rejects_valid_checkpoint_for_different_request(
    tmp_path,
    expected,
    message,
):
    checkpoint = write_coordinate_checkpoint(
        tmp_path,
        source_arrays=_source_arrays(),
        build_config=_config(),
        state_arrays=_state_arrays(),
        algorithm_version="spectral-map-v2",
    )

    with pytest.raises(CheckpointMismatchError, match=message):
        load_coordinate_checkpoint(checkpoint, **expected)


@pytest.mark.parametrize(
    "bad_array",
    [
        np.array([object()], dtype=object),
        np.array([1.0 + 2.0j]),
        np.array(["x"], dtype=object),
    ],
)
def test_pickle_requiring_or_nonprimitive_arrays_are_rejected(
    tmp_path,
    bad_array,
):
    with pytest.raises(TypeError, match="unsupported dtype"):
        write_coordinate_checkpoint(
            tmp_path,
            source_arrays=_source_arrays(),
            build_config=_config(),
            state_arrays={"unsafe": bad_array},
            algorithm_version="spectral-map-v2",
        )
    assert not list(tmp_path.glob(".tmp-coordinate-checkpoint-*"))


def test_empty_source_identity_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one source array"):
        write_coordinate_checkpoint(
            tmp_path,
            source_arrays={},
            build_config=_config(),
            state_arrays={},
            algorithm_version="spectral-map-v2",
        )


def test_failed_write_cleans_atomic_staging_directory(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(np, "savez", fail_save)
    with pytest.raises(OSError, match="injected write failure"):
        write_coordinate_checkpoint(
            tmp_path,
            source_arrays=_source_arrays(),
            build_config=_config(),
            state_arrays=_state_arrays(),
            algorithm_version="spectral-map-v2",
        )

    assert not list(tmp_path.glob(".tmp-coordinate-checkpoint-*"))
    assert not [path for path in tmp_path.iterdir() if path.is_dir()]


def test_directory_name_is_part_of_integrity_contract(tmp_path):
    checkpoint = write_coordinate_checkpoint(
        tmp_path,
        source_arrays=_source_arrays(),
        build_config=_config(),
        state_arrays=_state_arrays(),
        algorithm_version="spectral-map-v2",
    )
    renamed = tmp_path / ("0" * 64)
    checkpoint.rename(renamed)

    with pytest.raises(CheckpointIntegrityError, match="directory name"):
        load_coordinate_checkpoint(renamed)
