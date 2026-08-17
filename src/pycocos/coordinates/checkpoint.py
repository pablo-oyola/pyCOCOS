"""Content-addressed checkpoints for expensive coordinate construction.

This module deliberately has no dependency on :mod:`xarray` or on a concrete
coordinate-map class.  Callers flatten the primitive arrays that represent
their state, and keep dimensions, attributes, and reconstruction information
in JSON-compatible metadata.  That makes the on-disk boundary small, explicit,
and usable by both ``SpectralCoordinateMap`` and xarray dataset serializers.

Checkpoints are directories named by a SHA-256 key::

    <root>/<key>/manifest.json
                 manifest.sha256
                 arrays.npz

The key identifies the source arrays, normalized build configuration, and
coordinate algorithm version.  Payload hashes are recorded separately so a
checkpoint can be rejected if either its manifest or stored arrays are damaged.
NPZ payloads are loaded with ``allow_pickle=False`` and object arrays are never
accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np


CHECKPOINT_FORMAT = "pycocos-coordinate-checkpoint"
CHECKPOINT_FORMAT_VERSION = 1
_ARRAYS_FILENAME = "arrays.npz"
_MANIFEST_FILENAME = "manifest.json"
_MANIFEST_DIGEST_FILENAME = "manifest.sha256"
_KEY_DOMAIN = b"pycocos-coordinate-checkpoint-key-v1\0"
_ARRAY_DOMAIN = b"pycocos-coordinate-array-v1\0"
_NAMESPACE_SEPARATOR = "::"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CoordinateCheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointIntegrityError(CoordinateCheckpointError):
    """Raised when stored content does not match its manifest."""


class CheckpointMismatchError(CoordinateCheckpointError):
    """Raised when a valid checkpoint does not match requested inputs."""


@dataclass(frozen=True)
class LoadedCoordinateCheckpoint:
    """A fully verified checkpoint loaded into memory."""

    key: str
    path: Path
    arrays: Mapping[str, np.ndarray]
    build_config: Mapping[str, Any]
    metadata: Mapping[str, Any]
    manifest: Mapping[str, Any]

    def array_group(self, namespace: str) -> dict[str, np.ndarray]:
        """Return payload arrays stored under ``namespace``.

        This is intended for callers that keep, for example, spectral-map and
        dataset arrays in the same checkpoint without coupling this module to
        those concrete object types.
        """

        return extract_array_group(self.arrays, namespace)


def _normalize_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _normalize_json_value(value: Any, *, location: str) -> Any:
    """Return a deterministic, strict JSON representation of ``value``."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{location} contains a non-finite float")
        # Do not let a semantically insignificant sign bit change a key.
        return 0.0 if number == 0.0 else number
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{location} keys must be strings")
            key = _normalize_string(raw_key)
            if key in normalized:
                raise ValueError(
                    f"{location} has duplicate keys after Unicode normalization: "
                    f"{key!r}"
                )
            normalized[key] = _normalize_json_value(
                item,
                location=f"{location}.{key}",
            )
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _normalize_json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} contains unsupported value type {type(value).__name__!r}"
    )


def normalize_build_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a build configuration for deterministic hashing.

    Mapping order and tuple/list distinctions do not affect the result.  NumPy
    scalar values are converted to their Python equivalents.  Non-finite
    floating-point values and unordered containers are rejected.
    """

    if not isinstance(config, Mapping):
        raise TypeError("build_config must be a mapping")
    normalized = _normalize_json_value(config, location="build_config")
    assert isinstance(normalized, dict)
    return normalized


def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    normalized = _normalize_json_value(metadata, location="metadata")
    assert isinstance(normalized, dict)
    return normalized


def _validate_array_name(name: str, *, location: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{location} array names must be strings")
    normalized = _normalize_string(name)
    if not normalized:
        raise ValueError(f"{location} array names must not be empty")
    if normalized in {".", ".."} or any(
        token in normalized for token in ("/", "\\", "\x00")
    ):
        raise ValueError(f"unsafe {location} array name {name!r}")
    return normalized


def _canonical_array(value: Any, *, name: str) -> np.ndarray:
    """Return a safe, C-contiguous, endian-stable primitive array."""

    array = np.asarray(value)
    kind = array.dtype.kind
    if kind not in "biufSU":
        raise TypeError(
            f"array {name!r} has unsupported dtype {array.dtype}; only bool, "
            "integer, floating-point, byte-string, and Unicode arrays are "
            "checkpoint-safe"
        )
    if kind in "iu" and array.dtype.itemsize not in (1, 2, 4, 8):
        raise TypeError(f"array {name!r} has unsupported integer dtype {array.dtype}")
    if kind == "f" and array.dtype.itemsize not in (2, 4, 8):
        raise TypeError(f"array {name!r} has unsupported float dtype {array.dtype}")

    dtype = array.dtype
    if kind in "iufU" and dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    # ``np.ascontiguousarray`` promotes a zero-dimensional array to shape (1,),
    # which would make scalar state fail an exact round trip.  ``np.array``
    # preserves dimensionality while still giving us canonical C-order bytes.
    canonical = np.array(array, dtype=dtype, order="C", copy=True)

    if kind == "f" and canonical.size:
        # Normalize representations that are numerically identical but can have
        # different bits on input.  This also avoids platform-specific NaN
        # payloads changing content keys.
        canonical = canonical.copy()
        canonical[canonical == 0.0] = 0.0
        canonical[np.isnan(canonical)] = np.nan
    return canonical


def _canonical_arrays(
    arrays: Mapping[str, Any],
    *,
    location: str,
    require_nonempty: bool,
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping):
        raise TypeError(f"{location} must be a mapping")
    if require_nonempty and not arrays:
        raise ValueError(f"{location} must contain at least one source array")
    result: dict[str, np.ndarray] = {}
    for raw_name, value in arrays.items():
        name = _validate_array_name(raw_name, location=location)
        if name in result:
            raise ValueError(
                f"{location} has duplicate names after Unicode normalization: "
                f"{name!r}"
            )
        result[name] = _canonical_array(value, name=name)
    return {name: result[name] for name in sorted(result)}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _array_descriptor(array: np.ndarray) -> dict[str, Any]:
    header = {
        "dtype": array.dtype.str,
        "shape": [int(length) for length in array.shape],
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(_ARRAY_DOMAIN)
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\0")
    # A zero-byte multidimensional array can have a zero stride that Python's
    # memoryview refuses to cast.  Its dtype and exact shape are already part
    # of the hashed header, and it has no content bytes to add.
    if array.nbytes:
        digest.update(memoryview(array).cast("B"))
    return {
        **header,
        "nbytes": int(array.nbytes),
        "sha256": digest.hexdigest(),
    }


def _array_descriptors(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: _array_descriptor(arrays[name]) for name in sorted(arrays)}


def _checkpoint_key_from_material(key_material: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_KEY_DOMAIN)
    digest.update(_canonical_json_bytes(key_material))
    return digest.hexdigest()


def coordinate_checkpoint_key(
    source_arrays: Mapping[str, Any],
    build_config: Mapping[str, Any],
    *,
    algorithm_version: str,
) -> str:
    """Return the deterministic key for source data and build configuration."""

    algorithm = _normalize_algorithm_version(algorithm_version)
    sources = _canonical_arrays(
        source_arrays,
        location="source_arrays",
        require_nonempty=True,
    )
    key_material = {
        "algorithm_version": algorithm,
        "build_config": normalize_build_config(build_config),
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "source_arrays": _array_descriptors(sources),
    }
    return _checkpoint_key_from_material(key_material)


def coordinate_checkpoint_path(
    root: str | os.PathLike[str],
    key: str,
) -> Path:
    """Return ``root/key`` after validating the SHA-256 checkpoint key."""

    if not isinstance(key, str) or _SHA256_RE.fullmatch(key) is None:
        raise ValueError("checkpoint key must be a lowercase SHA-256 hex digest")
    return Path(root) / key


def namespace_array_group(
    namespace: str,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Prefix primitive state arrays with a reversible namespace.

    For example, a serializer may combine ``namespace_array_group("map", ...)``
    and ``namespace_array_group("dataset", ...)`` in one payload while keeping
    object reconstruction outside this generic module.
    """

    namespace = _validate_array_name(namespace, location="namespace")
    if _NAMESPACE_SEPARATOR in namespace:
        raise ValueError(
            f"namespace must not contain {_NAMESPACE_SEPARATOR!r}"
        )
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping")
    grouped: dict[str, Any] = {}
    for raw_name, value in arrays.items():
        name = _validate_array_name(raw_name, location="array group")
        if _NAMESPACE_SEPARATOR in name:
            raise ValueError(
                f"array group names must not contain {_NAMESPACE_SEPARATOR!r}"
            )
        grouped[f"{namespace}{_NAMESPACE_SEPARATOR}{name}"] = value
    return grouped


def extract_array_group(
    arrays: Mapping[str, np.ndarray],
    namespace: str,
) -> dict[str, np.ndarray]:
    """Extract and un-prefix arrays previously grouped under ``namespace``."""

    namespace = _validate_array_name(namespace, location="namespace")
    prefix = f"{namespace}{_NAMESPACE_SEPARATOR}"
    return {
        name[len(prefix) :]: value
        for name, value in arrays.items()
        if name.startswith(prefix)
    }


def _normalize_algorithm_version(algorithm_version: str) -> str:
    if not isinstance(algorithm_version, str):
        raise TypeError("algorithm_version must be a string")
    normalized = _normalize_string(algorithm_version).strip()
    if not normalized:
        raise ValueError("algorithm_version must not be empty")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _json_without_duplicate_keys(data: bytes) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointIntegrityError(
                    f"manifest contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=object_pairs)
    except CheckpointIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError("manifest is not valid UTF-8 JSON") from exc


def _validate_manifest_structure(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise CheckpointIntegrityError("manifest root must be a JSON object")
    try:
        format_name = manifest["format"]
        format_version = manifest["format_version"]
        key = manifest["checkpoint_key"]
        key_material = manifest["key_material"]
        source_digest = manifest["source_digest"]
        payload = manifest["payload"]
        metadata = manifest["metadata"]
        algorithm_version = manifest["algorithm_version"]
        hash_algorithm = manifest["hash_algorithm"]
    except KeyError as exc:
        raise CheckpointIntegrityError(
            f"manifest is missing required field {exc.args[0]!r}"
        ) from exc

    if format_name != CHECKPOINT_FORMAT:
        raise CheckpointIntegrityError(
            f"unsupported checkpoint format {format_name!r}"
        )
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointIntegrityError(
            f"unsupported checkpoint format version {format_version!r}"
        )
    if not isinstance(key, str) or _SHA256_RE.fullmatch(key) is None:
        raise CheckpointIntegrityError("manifest checkpoint key is invalid")
    if not isinstance(key_material, dict):
        raise CheckpointIntegrityError("manifest key_material must be an object")
    if (
        not isinstance(source_digest, str)
        or _SHA256_RE.fullmatch(source_digest) is None
    ):
        raise CheckpointIntegrityError("manifest source_digest is invalid")
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError("manifest payload must be an object")
    if not isinstance(metadata, dict):
        raise CheckpointIntegrityError("manifest metadata must be an object")
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise CheckpointIntegrityError("manifest algorithm_version is invalid")
    if hash_algorithm != "sha256":
        raise CheckpointIntegrityError("manifest hash_algorithm must be 'sha256'")
    return manifest


def _verify_existing_payload(
    checkpoint: LoadedCoordinateCheckpoint,
    expected_arrays: Mapping[str, np.ndarray],
) -> None:
    expected = _array_descriptors(expected_arrays)
    actual = checkpoint.manifest["payload"].get("arrays")
    if actual != expected:
        raise FileExistsError(
            f"checkpoint {checkpoint.path} already exists with a different payload"
        )


def write_coordinate_checkpoint(
    root: str | os.PathLike[str],
    *,
    source_arrays: Mapping[str, Any],
    build_config: Mapping[str, Any],
    state_arrays: Mapping[str, Any],
    algorithm_version: str,
    metadata: Mapping[str, Any] | None = None,
    compressed: bool = False,
    reuse_existing: bool = False,
) -> Path:
    """Atomically write a content-addressed coordinate checkpoint.

    Existing checkpoints are never overwritten.  With ``reuse_existing=True``,
    a pre-existing checkpoint is fully verified and reused only when its input
    identity and payload array descriptors match the requested checkpoint.
    Otherwise :class:`FileExistsError` is raised.
    """

    algorithm = _normalize_algorithm_version(algorithm_version)
    sources = _canonical_arrays(
        source_arrays,
        location="source_arrays",
        require_nonempty=True,
    )
    states = _canonical_arrays(
        state_arrays,
        location="state_arrays",
        require_nonempty=False,
    )
    normalized_config = normalize_build_config(build_config)
    normalized_metadata = _normalize_metadata(metadata)
    source_descriptors = _array_descriptors(sources)
    key_material = {
        "algorithm_version": algorithm,
        "build_config": normalized_config,
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "source_arrays": source_descriptors,
    }
    key = _checkpoint_key_from_material(key_material)

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    target = coordinate_checkpoint_path(root_path, key)
    if target.exists():
        if not reuse_existing:
            raise FileExistsError(f"checkpoint already exists: {target}")
        existing = load_coordinate_checkpoint(
            target,
            expected_source_arrays=sources,
            expected_build_config=normalized_config,
            expected_algorithm_version=algorithm,
        )
        _verify_existing_payload(existing, states)
        return target

    temporary = Path(
        tempfile.mkdtemp(prefix=".tmp-coordinate-checkpoint-", dir=root_path)
    )
    try:
        arrays_path = temporary / _ARRAYS_FILENAME
        with arrays_path.open("xb") as handle:
            save = np.savez_compressed if compressed else np.savez
            save(handle, **states)
            handle.flush()
            os.fsync(handle.fileno())

        payload = {
            "arrays": _array_descriptors(states),
            "compressed": bool(compressed),
            "file": _ARRAYS_FILENAME,
            "sha256": _sha256_file(arrays_path),
        }
        source_digest = hashlib.sha256(
            _canonical_json_bytes(source_descriptors)
        ).hexdigest()
        manifest = {
            "algorithm_version": algorithm,
            "checkpoint_key": key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "hash_algorithm": "sha256",
            "key_material": key_material,
            "metadata": normalized_metadata,
            "payload": payload,
            "source_digest": source_digest,
        }
        manifest_bytes = (
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_bytes(temporary / _MANIFEST_FILENAME, manifest_bytes)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        _write_bytes(
            temporary / _MANIFEST_DIGEST_FILENAME,
            f"{manifest_digest}\n".encode("ascii"),
        )
        _fsync_directory(temporary)

        try:
            os.rename(temporary, target)
        except FileExistsError:
            if not reuse_existing:
                raise FileExistsError(f"checkpoint already exists: {target}")
            existing = load_coordinate_checkpoint(
                target,
                expected_source_arrays=sources,
                expected_build_config=normalized_config,
                expected_algorithm_version=algorithm,
            )
            _verify_existing_payload(existing, states)
            return target
        _fsync_directory(root_path)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate_array_descriptor_schema(
    name: str,
    descriptor: Any,
    *,
    location: str,
) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise CheckpointIntegrityError(
            f"invalid descriptor for {location} array {name!r}"
        )
    required = {"dtype", "shape", "order", "nbytes", "sha256"}
    if not required.issubset(descriptor):
        raise CheckpointIntegrityError(
            f"incomplete descriptor for {location} array {name!r}"
        )
    try:
        dtype = np.dtype(descriptor["dtype"])
    except (TypeError, ValueError) as exc:
        raise CheckpointIntegrityError(
            f"invalid dtype for {location} array {name!r}"
        ) from exc
    if dtype.kind not in "biufSU":
        raise CheckpointIntegrityError(
            f"unsafe dtype for {location} array {name!r}"
        )
    shape = descriptor["shape"]
    if not isinstance(shape, list) or any(
        isinstance(length, bool)
        or not isinstance(length, int)
        or length < 0
        for length in shape
    ):
        raise CheckpointIntegrityError(
            f"invalid shape for {location} array {name!r}"
        )
    if descriptor["order"] != "C":
        raise CheckpointIntegrityError(
            f"invalid order for {location} array {name!r}"
        )
    nbytes = descriptor["nbytes"]
    expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if (
        isinstance(nbytes, bool)
        or not isinstance(nbytes, int)
        or nbytes != expected_nbytes
    ):
        raise CheckpointIntegrityError(
            f"invalid byte count for {location} array {name!r}"
        )
    digest = descriptor["sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise CheckpointIntegrityError(
            f"invalid hash for {location} array {name!r}"
        )
    return descriptor


def _manifest_array_descriptors(
    value: Any,
    *,
    location: str = "payload",
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointIntegrityError("payload arrays must be an object")
    descriptors: dict[str, Any] = {}
    for raw_name, descriptor in value.items():
        try:
            name = _validate_array_name(raw_name, location="manifest payload")
        except (TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(str(exc)) from exc
        if name != raw_name:
            raise CheckpointIntegrityError(
                f"invalid descriptor name for {location} array {raw_name!r}"
            )
        descriptors[name] = _validate_array_descriptor_schema(
            name,
            descriptor,
            location=location,
        )
    return descriptors


def load_coordinate_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_source_arrays: Mapping[str, Any] | None = None,
    expected_build_config: Mapping[str, Any] | None = None,
    expected_algorithm_version: str | None = None,
) -> LoadedCoordinateCheckpoint:
    """Load a checkpoint only after verifying all manifests and array hashes."""

    checkpoint_path = Path(path)
    manifest_path = checkpoint_path / _MANIFEST_FILENAME
    digest_path = checkpoint_path / _MANIFEST_DIGEST_FILENAME
    arrays_path = checkpoint_path / _ARRAYS_FILENAME
    for required in (manifest_path, digest_path, arrays_path):
        if not required.is_file():
            raise CheckpointIntegrityError(
                f"checkpoint is missing required file {required.name!r}"
            )

    try:
        stored_manifest_digest = digest_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise CheckpointIntegrityError("cannot read manifest digest") from exc
    if _SHA256_RE.fullmatch(stored_manifest_digest) is None:
        raise CheckpointIntegrityError("manifest digest is invalid")
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_digest != stored_manifest_digest:
        raise CheckpointIntegrityError("manifest SHA-256 verification failed")

    manifest = _validate_manifest_structure(
        _json_without_duplicate_keys(manifest_bytes)
    )
    key = manifest["checkpoint_key"]
    if checkpoint_path.name != key:
        raise CheckpointIntegrityError(
            "checkpoint directory name does not match its content key"
        )
    key_material = manifest["key_material"]
    if _checkpoint_key_from_material(key_material) != key:
        raise CheckpointIntegrityError("checkpoint key verification failed")
    try:
        if _normalize_json_value(
            key_material,
            location="manifest.key_material",
        ) != key_material:
            raise CheckpointIntegrityError("key material is not normalized")
        if _normalize_json_value(
            manifest["metadata"],
            location="manifest.metadata",
        ) != manifest["metadata"]:
            raise CheckpointIntegrityError("metadata is not normalized")
    except (TypeError, ValueError) as exc:
        raise CheckpointIntegrityError(str(exc)) from exc

    try:
        manifest_algorithm = key_material["algorithm_version"]
        manifest_config = key_material["build_config"]
        source_descriptors = _manifest_array_descriptors(
            key_material["source_arrays"],
            location="source",
        )
    except KeyError as exc:
        raise CheckpointIntegrityError(
            f"key material is missing field {exc.args[0]!r}"
        ) from exc
    if key_material.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointIntegrityError("key material format is invalid")
    if key_material.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointIntegrityError("key material format version is invalid")
    if manifest["algorithm_version"] != manifest_algorithm:
        raise CheckpointIntegrityError(
            "manifest algorithm version disagrees with its key material"
        )
    source_digest = hashlib.sha256(
        _canonical_json_bytes(source_descriptors)
    ).hexdigest()
    if source_digest != manifest["source_digest"]:
        raise CheckpointIntegrityError("source descriptor digest verification failed")

    payload = manifest["payload"]
    if payload.get("file") != _ARRAYS_FILENAME:
        raise CheckpointIntegrityError("manifest references an invalid payload file")
    payload_digest = payload.get("sha256")
    if (
        not isinstance(payload_digest, str)
        or _SHA256_RE.fullmatch(payload_digest) is None
    ):
        raise CheckpointIntegrityError("payload digest is invalid")
    if _sha256_file(arrays_path) != payload_digest:
        raise CheckpointIntegrityError("payload SHA-256 verification failed")
    expected_descriptors = _manifest_array_descriptors(payload.get("arrays"))

    loaded_arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_descriptors):
                raise CheckpointIntegrityError(
                    "payload member names do not match the manifest"
                )
            for name in sorted(expected_descriptors):
                array = _canonical_array(archive[name], name=name)
                if _array_descriptor(array) != expected_descriptors[name]:
                    raise CheckpointIntegrityError(
                        f"payload array {name!r} failed descriptor verification"
                    )
                loaded_arrays[name] = array
    except CheckpointIntegrityError:
        raise
    except Exception as exc:
        raise CheckpointIntegrityError("cannot decode NPZ payload safely") from exc

    if expected_algorithm_version is not None:
        expected_algorithm = _normalize_algorithm_version(
            expected_algorithm_version
        )
        if manifest_algorithm != expected_algorithm:
            raise CheckpointMismatchError(
                "checkpoint algorithm version does not match the request"
            )
    if expected_build_config is not None:
        expected_config = normalize_build_config(expected_build_config)
        if manifest_config != expected_config:
            raise CheckpointMismatchError(
                "checkpoint build configuration does not match the request"
            )
    if expected_source_arrays is not None:
        expected_sources = _canonical_arrays(
            expected_source_arrays,
            location="expected_source_arrays",
            require_nonempty=True,
        )
        if _array_descriptors(expected_sources) != source_descriptors:
            raise CheckpointMismatchError(
                "checkpoint source arrays do not match the request"
            )

    return LoadedCoordinateCheckpoint(
        key=key,
        path=checkpoint_path,
        arrays=loaded_arrays,
        build_config=manifest_config,
        metadata=manifest["metadata"],
        manifest=manifest,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointIntegrityError",
    "CheckpointMismatchError",
    "CoordinateCheckpointError",
    "LoadedCoordinateCheckpoint",
    "coordinate_checkpoint_key",
    "coordinate_checkpoint_path",
    "extract_array_group",
    "load_coordinate_checkpoint",
    "namespace_array_group",
    "normalize_build_config",
    "write_coordinate_checkpoint",
]
