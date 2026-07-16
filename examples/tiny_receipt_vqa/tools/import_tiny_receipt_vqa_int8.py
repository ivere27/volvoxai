#!/usr/bin/env python3
"""Normalize the published TinyReceiptVQA INT8 SafeTensors release.

This is deliberately a development-only weight adapter.  The published release
contains a weight-only manifest: it does not include a Volvox graph, activation
quantization parameters, or output goldens.  Consequently this tool writes
only a standard SafeTensors weight file and a mergeable
``weights_quantization`` fragment.  It never tries to invent a graph or an
end-to-end W8A8 calibration.

``--conversion-contract`` is an optional *external* attestation.  When it is
supplied, this adapter validates a deliberately narrow, hash-bound contract
for the separately authored graph specification, activation calibration,
preprocessing, explicit adapter-family route, and executable golden cases.
It records that validation in the fragment, but still neither constructs nor
executes a graph.  Omitting the option retains the backwards-compatible
weight-only normalizer behaviour.

The source release stores state-dict tensors as ``quantized.<key>`` +
``scale.<key>`` or ``float32.<key>``.  Source offsets are intentionally not
read: named SafeTensors entries, their dtypes, and their shapes are the
authoritative payload.  Rank-4 INT8 tensors require an explicit ``--conv-key``
selection before being repacked OIHW -> OHWI.  Rank-2 INT8 tensors are copied
byte-for-byte in their existing order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from safetensors.numpy import load_file, save_file


SOURCE_FORMAT = "tiny_receipt_vqa_int8_safetensors_v1"
SOURCE_RUNTIME = "pytorch-reference-dequantized-safetensors-v1"
SOURCE_QUANTIZATION = "symmetric per-output-channel INT8"
SOURCE_LAYOUT = {
    "int8_per_out": "quantized.<state_dict_key>",
    "scale_fp32": "scale.<state_dict_key>",
    "float32": "float32.<state_dict_key>",
}

FRAGMENT_FORMAT = "volvoxai-weights-quantization-fragment-v1"
NORMALIZED_SAFETENSORS_FORMAT = "volvoxai-normalized-int8-weights-v1"
NORMALIZED_INT8_PREFIX = "tiny_receipt_vqa.int8."
NORMALIZED_FLOAT32_PREFIX = "tiny_receipt_vqa.float32."

CONVERSION_CONTRACT_FORMAT = "volvoxai-tiny-receipt-vqa-w8a8-conversion-contract-v1"
GRAPH_SPEC_FORMAT = "volvoxai-tiny-receipt-vqa-graph-spec-v1"
ACTIVATION_CALIBRATION_FORMAT = "volvoxai-tiny-receipt-vqa-activation-calibration-v1"
PREPROCESSING_FORMAT = "volvoxai-tiny-receipt-vqa-preprocessing-v1"
GOLDENS_FORMAT = "volvoxai-tiny-receipt-vqa-w8a8-goldens-v1"
EXPLICIT_FAMILY_ORDER = (
    "phone",
    "address",
    "store",
    "item_row",
    "item_math",
    "item_lookup",
    "math",
    "other",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """The release manifest or its named SafeTensors payload is unsupported."""


class _DuplicateJsonKeyError(ValueError):
    """JSON objects used as a fail-closed contract may not repeat a key."""


@dataclass(frozen=True)
class TensorSpec:
    key: str
    source_name: str
    normalized_name: str
    source_dtype: str
    shape: tuple[int, ...]
    transform: str


@dataclass(frozen=True)
class NormalizationResult:
    source_weights: Path
    normalized_weights: Path
    quantization_fragment: Path
    quantized_tensor_count: int
    float32_tensor_count: int
    conversion_contract_validated: bool


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ConversionContractValidation:
    path: Path
    contract_sha256: str
    source_manifest_sha256: str
    source_weights_sha256: str
    graph_id: str
    graph_sha256: str
    activation_edge_count: int
    calibration_sha256: str
    calibration_sample_count: int
    preprocessing_sha256: str
    routing: dict[str, Any]
    routing_sha256: str
    goldens_sha256: str
    golden_case_count: int


def _as_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{label} must be an object")
    return value


def _require_exact_keys(record: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(record)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    pieces: list[str] = []
    if missing:
        pieces.append(f"missing keys {missing!r}")
    if unexpected:
        pieces.append(f"unexpected keys {unexpected!r}")
    raise ManifestValidationError(f"{label} must contain exactly {sorted(expected)!r}; " + "; ".join(pieces))


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{label} must be a non-empty string")
    return value


def _require_integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ManifestValidationError(f"{label} must be at least {minimum}")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ManifestValidationError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ManifestValidationError(f"{label} must be a finite number")
    return number


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ManifestValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _json_object_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestValidationError(f"could not read {label} {path}: {error}") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_json_object_no_duplicate_keys,
            parse_constant=_reject_json_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ManifestValidationError(f"{label} {path} is not valid JSON: {error}") from error
    return _as_record(value, label)


def _sha256_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise ManifestValidationError(f"{label} is missing or is not a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ManifestValidationError(f"could not hash {label} {path}: {error}") from error
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManifestValidationError(f"could not canonicalize conversion contract value: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_string(value: Any, expected: str, label: str) -> None:
    if value != expected:
        raise ManifestValidationError(
            f"{label} must be {expected!r}, got {value!r}"
        )


def _release_weight_filename(manifest: dict[str, Any]) -> str:
    files = _as_record(manifest.get("files"), "manifest.files")
    filename = files.get("model_safetensors")
    if not isinstance(filename, str) or not filename:
        raise ManifestValidationError("manifest.files.model_safetensors must be a non-empty string")
    # The published release layout puts the file next to manifest.json.  Do not
    # silently follow a user-controlled relative path outside that package.
    if "/" in filename or "\\" in filename or Path(filename).suffix != ".safetensors":
        raise ManifestValidationError(
            "manifest.files.model_safetensors must be a local .safetensors filename"
        )
    return filename


def _validate_release_contract(manifest: Any) -> tuple[dict[str, Any], str]:
    root = _as_record(manifest, "manifest")
    _require_exact_string(root.get("format"), SOURCE_FORMAT, "manifest.format")
    _require_exact_string(root.get("runtime"), SOURCE_RUNTIME, "manifest.runtime")

    safetensors = _as_record(root.get("safetensors"), "manifest.safetensors")
    if safetensors.get("layout") != SOURCE_LAYOUT:
        raise ManifestValidationError(
            "manifest.safetensors.layout is not the published TinyReceiptVQA SafeTensors layout"
        )
    _require_exact_string(
        safetensors.get("quantization"),
        SOURCE_QUANTIZATION,
        "manifest.safetensors.quantization",
    )

    tensors = _as_record(root.get("tensors"), "manifest.tensors")
    if not tensors:
        raise ManifestValidationError("manifest.tensors must not be empty")
    return tensors, _release_weight_filename(root)


def _shape_from_manifest(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{label}.shape must be an array")
    shape: list[int] = []
    for index, dimension in enumerate(value):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ManifestValidationError(
                f"{label}.shape[{index}] must be a positive integer"
            )
        shape.append(dimension)
    return tuple(shape)


def _normalized_name(source_dtype: str, key: str) -> str:
    if source_dtype == "int8_per_out":
        return NORMALIZED_INT8_PREFIX + key
    if source_dtype == "float32":
        return NORMALIZED_FLOAT32_PREFIX + key
    raise AssertionError(f"unsupported internal source dtype: {source_dtype}")


def _build_specs(tensors: dict[str, Any], conv_keys: Iterable[str]) -> list[TensorSpec]:
    selected_conv_keys = list(conv_keys)
    if any(not isinstance(key, str) or not key for key in selected_conv_keys):
        raise ManifestValidationError("every --conv-key must be a non-empty state-dict key")
    if len(selected_conv_keys) != len(set(selected_conv_keys)):
        raise ManifestValidationError("--conv-key was specified more than once for the same key")
    selected_conv = set(selected_conv_keys)

    specs: list[TensorSpec] = []
    seen_normalized_names: set[str] = set()
    for key, raw_info in tensors.items():
        if not isinstance(key, str) or not key or "\x00" in key:
            raise ManifestValidationError("every manifest.tensors key must be a non-empty SafeTensors name")
        info = _as_record(raw_info, f"manifest.tensors[{key!r}]")
        source_dtype = info.get("dtype")
        if source_dtype not in ("int8_per_out", "float32"):
            raise ManifestValidationError(
                f"manifest.tensors[{key!r}].dtype must be 'int8_per_out' or 'float32'"
            )
        shape = _shape_from_manifest(info.get("shape"), f"manifest.tensors[{key!r}]")

        if source_dtype == "int8_per_out":
            if len(shape) == 4:
                if key not in selected_conv:
                    raise ManifestValidationError(
                        f"rank-4 INT8 tensor {key!r} requires an explicit --conv-key {key!r} "
                        "to permit OIHW -> OHWI conversion"
                    )
                transform = "OIHW_to_OHWI"
            elif len(shape) == 2:
                if key in selected_conv:
                    raise ManifestValidationError(
                        f"--conv-key {key!r} names a rank-2 tensor; only rank-4 OIHW tensors are Conv candidates"
                    )
                transform = "identity"
            else:
                raise ManifestValidationError(
                    f"INT8 tensor {key!r} has rank {len(shape)}; this bounded adapter supports only rank-2 "
                    "identity or explicitly selected rank-4 Conv weights"
                )
            source_name = f"quantized.{key}"
        else:
            if key in selected_conv:
                raise ManifestValidationError(
                    f"--conv-key {key!r} refers to a float32 tensor, not a rank-4 INT8 Conv weight"
                )
            transform = "identity"
            source_name = f"float32.{key}"

        normalized_name = _normalized_name(source_dtype, key)
        if normalized_name in seen_normalized_names:
            raise ManifestValidationError(
                f"normalization would create a duplicate tensor name {normalized_name!r}"
            )
        seen_normalized_names.add(normalized_name)
        specs.append(TensorSpec(
            key=key,
            source_name=source_name,
            normalized_name=normalized_name,
            source_dtype=source_dtype,
            shape=shape,
            transform=transform,
        ))

    unknown_conv = selected_conv - {spec.key for spec in specs}
    if unknown_conv:
        raise ManifestValidationError(
            f"--conv-key does not name a manifest tensor: {sorted(unknown_conv)!r}"
        )
    return specs


def _load_named_safetensors(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ManifestValidationError(f"published SafeTensors payload is missing: {path}")
    try:
        tensors = load_file(str(path))
    except Exception as error:  # safetensors exposes a Rust exception type.
        raise ManifestValidationError(f"could not read SafeTensors payload {path}: {error}") from error
    if not isinstance(tensors, dict):
        raise ManifestValidationError(f"SafeTensors payload {path} did not produce a tensor map")
    return tensors


def _require_array(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ManifestValidationError(f"{label} is not a NumPy tensor")
    return value


def _require_tensor(array: Any, *, name: str, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    tensor = _require_array(array, f"SafeTensors entry {name!r}")
    if tensor.dtype != dtype:
        raise ManifestValidationError(
            f"SafeTensors entry {name!r} has dtype {tensor.dtype}, expected {dtype}"
        )
    if tuple(tensor.shape) != shape:
        raise ManifestValidationError(
            f"SafeTensors entry {name!r} has shape {list(tensor.shape)}, expected {list(shape)}"
        )
    return tensor


def _require_scale(array: Any, *, name: str, output_channels: int) -> np.ndarray:
    scale = _require_tensor(
        array,
        name=name,
        dtype=np.dtype(np.float32),
        shape=(output_channels,),
    )
    if not np.all(np.isfinite(scale)) or not np.all(scale > 0.0):
        raise ManifestValidationError(
            f"SafeTensors scale {name!r} must contain only finite positive float32 values"
        )
    return scale


def _source_name_set(specs: Iterable[TensorSpec]) -> set[str]:
    names: set[str] = set()
    for spec in specs:
        names.add(spec.source_name)
        if spec.source_dtype == "int8_per_out":
            names.add(f"scale.{spec.key}")
    return names


def _validate_source_name_set(source: dict[str, np.ndarray], specs: Iterable[TensorSpec]) -> None:
    expected = _source_name_set(specs)
    actual = set(source)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        pieces: list[str] = []
        if missing:
            pieces.append(f"missing named entries: {missing!r}")
        if unexpected:
            pieces.append(f"unexpected named entries: {unexpected!r}")
        raise ManifestValidationError("SafeTensors entry set does not match manifest; " + "; ".join(pieces))


def _checked_destination(source: Path, normalized: Path, fragment: Path) -> None:
    source_resolved = source.resolve()
    normalized_resolved = normalized.resolve()
    fragment_resolved = fragment.resolve()
    if normalized_resolved == source_resolved:
        raise ManifestValidationError("--out must not overwrite the published source SafeTensors file")
    if fragment_resolved == source_resolved:
        raise ManifestValidationError("--quantization-out must not overwrite the published source SafeTensors file")
    if normalized_resolved == fragment_resolved:
        raise ManifestValidationError("--out and --quantization-out must be different files")


def _contract_local_file(contract_path: Path, value: Any, label: str) -> Path:
    filename = _require_nonempty_string(value, label)
    if "\x00" in filename or "\\" in filename:
        raise ManifestValidationError(f"{label} must use a local POSIX-style relative path")
    candidate = Path(filename)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ManifestValidationError(f"{label} must stay below the conversion-contract directory")
    root = contract_path.parent.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ManifestValidationError(
            f"{label} must stay below the conversion-contract directory"
        ) from error
    if resolved == contract_path.resolve():
        raise ManifestValidationError(f"{label} must not reference the conversion contract itself")
    return resolved


def _validated_artifact_reference(
    contract_path: Path,
    value: Any,
    label: str,
) -> ArtifactReference:
    record = _as_record(value, label)
    _require_exact_keys(record, {"file", "sha256"}, label)
    path = _contract_local_file(contract_path, record.get("file"), f"{label}.file")
    declared_sha256 = _require_sha256(record.get("sha256"), f"{label}.sha256")
    actual_sha256 = _sha256_file(path, label)
    if actual_sha256 != declared_sha256:
        raise ManifestValidationError(
            f"{label} SHA-256 mismatch: declared {declared_sha256}, actual {actual_sha256}"
        )
    return ArtifactReference(path=path, sha256=actual_sha256)


def _validate_graph_spec(
    graph: Any,
    *,
    source_manifest_sha256: str,
    source_weights_sha256: str,
) -> tuple[str, dict[str, tuple[str, tuple[int, ...]]]]:
    root = _as_record(graph, "graph specification")
    _require_exact_keys(
        root,
        {"format", "id", "source_manifest_sha256", "source_weights_sha256", "activation_edges"},
        "graph specification",
    )
    _require_exact_string(root.get("format"), GRAPH_SPEC_FORMAT, "graph specification.format")
    graph_id = _require_nonempty_string(root.get("id"), "graph specification.id")
    if "\x00" in graph_id:
        raise ManifestValidationError("graph specification.id must not contain NUL")
    manifest_hash = _require_sha256(
        root.get("source_manifest_sha256"),
        "graph specification.source_manifest_sha256",
    )
    if manifest_hash != source_manifest_sha256:
        raise ManifestValidationError(
            "graph specification.source_manifest_sha256 does not bind the supplied source manifest"
        )
    weights_hash = _require_sha256(
        root.get("source_weights_sha256"),
        "graph specification.source_weights_sha256",
    )
    if weights_hash != source_weights_sha256:
        raise ManifestValidationError(
            "graph specification.source_weights_sha256 does not bind the supplied SafeTensors payload"
        )

    raw_edges = root.get("activation_edges")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ManifestValidationError("graph specification.activation_edges must be a non-empty array")
    descriptors: dict[str, tuple[str, tuple[int, ...]]] = {}
    for index, raw_edge in enumerate(raw_edges):
        label = f"graph specification.activation_edges[{index}]"
        edge = _as_record(raw_edge, label)
        _require_exact_keys(edge, {"id", "dtype", "shape", "quantization"}, label)
        edge_id = _require_nonempty_string(edge.get("id"), f"{label}.id")
        if "\x00" in edge_id:
            raise ManifestValidationError(f"{label}.id must not contain NUL")
        if edge_id in descriptors:
            raise ManifestValidationError(f"graph specification.activation_edges repeats id {edge_id!r}")
        dtype = edge.get("dtype")
        if dtype not in ("int8", "uint8"):
            raise ManifestValidationError(f"{label}.dtype must be 'int8' or 'uint8'")
        _require_exact_string(edge.get("quantization"), "per_tensor", f"{label}.quantization")
        shape = _shape_from_manifest(edge.get("shape"), label)
        descriptors[edge_id] = (dtype, shape)
    return graph_id, descriptors


def _validate_activation_calibration(
    calibration: Any,
    *,
    contract_path: Path,
    graph_sha256: str,
    graph_descriptors: dict[str, tuple[str, tuple[int, ...]]],
) -> int:
    root = _as_record(calibration, "activation calibration")
    _require_exact_keys(
        root,
        {"format", "graph_sha256", "dataset_file", "dataset_sha256", "sample_count", "edges"},
        "activation calibration",
    )
    _require_exact_string(root.get("format"), ACTIVATION_CALIBRATION_FORMAT, "activation calibration.format")
    calibration_graph_sha256 = _require_sha256(
        root.get("graph_sha256"),
        "activation calibration.graph_sha256",
    )
    if calibration_graph_sha256 != graph_sha256:
        raise ManifestValidationError(
            "activation calibration.graph_sha256 does not bind the supplied graph specification"
        )
    dataset_path = _contract_local_file(
        contract_path,
        root.get("dataset_file"),
        "activation calibration.dataset_file",
    )
    dataset_sha256 = _require_sha256(root.get("dataset_sha256"), "activation calibration.dataset_sha256")
    actual_dataset_sha256 = _sha256_file(dataset_path, "activation calibration dataset")
    if actual_dataset_sha256 != dataset_sha256:
        raise ManifestValidationError(
            "activation calibration.dataset_sha256 does not match activation calibration.dataset_file"
        )
    sample_count = _require_integer(root.get("sample_count"), "activation calibration.sample_count", minimum=1)

    raw_edges = root.get("edges")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ManifestValidationError("activation calibration.edges must be a non-empty array")
    seen: set[str] = set()
    for index, raw_edge in enumerate(raw_edges):
        label = f"activation calibration.edges[{index}]"
        edge = _as_record(raw_edge, label)
        _require_exact_keys(
            edge,
            {"id", "dtype", "shape", "scale", "zero_point", "observed_min", "observed_max"},
            label,
        )
        edge_id = _require_nonempty_string(edge.get("id"), f"{label}.id")
        if edge_id in seen:
            raise ManifestValidationError(f"activation calibration.edges repeats id {edge_id!r}")
        seen.add(edge_id)
        expected = graph_descriptors.get(edge_id)
        if expected is None:
            raise ManifestValidationError(f"activation calibration names unknown graph activation edge {edge_id!r}")
        dtype, shape = expected
        if edge.get("dtype") != dtype:
            raise ManifestValidationError(f"{label}.dtype does not match graph activation edge {edge_id!r}")
        if _shape_from_manifest(edge.get("shape"), label) != shape:
            raise ManifestValidationError(f"{label}.shape does not match graph activation edge {edge_id!r}")
        scale = _require_finite_number(edge.get("scale"), f"{label}.scale")
        if scale <= 0.0:
            raise ManifestValidationError(f"{label}.scale must be positive")
        zero_point = _require_integer(edge.get("zero_point"), f"{label}.zero_point")
        minimum, maximum = (-128, 127) if dtype == "int8" else (0, 255)
        if zero_point < minimum or zero_point > maximum:
            raise ManifestValidationError(
                f"{label}.zero_point must be in [{minimum}, {maximum}] for {dtype}"
            )
        observed_min = _require_finite_number(edge.get("observed_min"), f"{label}.observed_min")
        observed_max = _require_finite_number(edge.get("observed_max"), f"{label}.observed_max")
        if observed_min > observed_max:
            raise ManifestValidationError(f"{label}.observed_min must not exceed observed_max")

    missing = sorted(set(graph_descriptors) - seen)
    if missing:
        raise ManifestValidationError(
            f"activation calibration is missing graph activation edge descriptors: {missing!r}"
        )
    return sample_count


def _validate_preprocessing(preprocessing: Any) -> None:
    root = _as_record(preprocessing, "preprocessing contract")
    _require_exact_keys(root, {"format", "input", "resize", "output"}, "preprocessing contract")
    _require_exact_string(root.get("format"), PREPROCESSING_FORMAT, "preprocessing contract.format")

    input_spec = _as_record(root.get("input"), "preprocessing contract.input")
    _require_exact_keys(input_spec, {"color_space", "value_min", "value_max"}, "preprocessing contract.input")
    _require_nonempty_string(input_spec.get("color_space"), "preprocessing contract.input.color_space")
    input_min = _require_finite_number(input_spec.get("value_min"), "preprocessing contract.input.value_min")
    input_max = _require_finite_number(input_spec.get("value_max"), "preprocessing contract.input.value_max")
    if input_min >= input_max:
        raise ManifestValidationError("preprocessing contract.input.value_min must be less than value_max")

    resize = _as_record(root.get("resize"), "preprocessing contract.resize")
    _require_exact_keys(resize, {"width", "height", "interpolation"}, "preprocessing contract.resize")
    _require_integer(resize.get("width"), "preprocessing contract.resize.width", minimum=1)
    _require_integer(resize.get("height"), "preprocessing contract.resize.height", minimum=1)
    _require_nonempty_string(resize.get("interpolation"), "preprocessing contract.resize.interpolation")

    output = _as_record(root.get("output"), "preprocessing contract.output")
    _require_exact_keys(output, {"dtype", "layout", "scale", "offset"}, "preprocessing contract.output")
    _require_exact_string(output.get("dtype"), "float32", "preprocessing contract.output.dtype")
    _require_nonempty_string(output.get("layout"), "preprocessing contract.output.layout")
    scale = _require_finite_number(output.get("scale"), "preprocessing contract.output.scale")
    if scale == 0.0:
        raise ManifestValidationError("preprocessing contract.output.scale must be non-zero")
    _require_finite_number(output.get("offset"), "preprocessing contract.output.offset")


def _validate_explicit_routing(value: Any) -> dict[str, Any]:
    routing = _as_record(value, "routing contract")
    _require_exact_keys(routing, {"mode", "family_order", "route_scope", "batch_policy"}, "routing contract")
    _require_exact_string(routing.get("mode"), "explicit_family_v1", "routing contract.mode")
    family_order = routing.get("family_order")
    if family_order != list(EXPLICIT_FAMILY_ORDER):
        raise ManifestValidationError(
            "routing contract.family_order must exactly match the TinyReceiptVQA explicit-family order "
            f"{list(EXPLICIT_FAMILY_ORDER)!r}"
        )
    _require_exact_string(routing.get("route_scope"), "whole_execution", "routing contract.route_scope")
    _require_exact_string(routing.get("batch_policy"), "homogeneous", "routing contract.batch_policy")
    return {
        "mode": "explicit_family_v1",
        "family_order": list(EXPLICIT_FAMILY_ORDER),
        "route_scope": "whole_execution",
        "batch_policy": "homogeneous",
    }


def _validate_goldens(
    goldens: Any,
    *,
    contract_path: Path,
    source_manifest_sha256: str,
    source_weights_sha256: str,
    graph_sha256: str,
    calibration_sha256: str,
    preprocessing_sha256: str,
    routing_sha256: str,
) -> int:
    root = _as_record(goldens, "goldens contract")
    _require_exact_keys(
        root,
        {
            "format",
            "source_manifest_sha256",
            "source_weights_sha256",
            "graph_sha256",
            "calibration_sha256",
            "preprocessing_sha256",
            "routing_sha256",
            "cases",
        },
        "goldens contract",
    )
    _require_exact_string(root.get("format"), GOLDENS_FORMAT, "goldens contract.format")
    expected_hashes = {
        "source_manifest_sha256": source_manifest_sha256,
        "source_weights_sha256": source_weights_sha256,
        "graph_sha256": graph_sha256,
        "calibration_sha256": calibration_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "routing_sha256": routing_sha256,
    }
    for key, expected in expected_hashes.items():
        actual = _require_sha256(root.get(key), f"goldens contract.{key}")
        if actual != expected:
            raise ManifestValidationError(
                f"goldens contract.{key} does not bind the validated {key.removesuffix('_sha256')}"
            )

    cases = root.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ManifestValidationError("goldens contract.cases must be a non-empty array")
    case_ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        label = f"goldens contract.cases[{index}]"
        case = _as_record(raw_case, label)
        _require_exact_keys(
            case,
            {"id", "input_file", "input_sha256", "family_id", "expected_token_ids"},
            label,
        )
        case_id = _require_nonempty_string(case.get("id"), f"{label}.id")
        if case_id in case_ids:
            raise ManifestValidationError(f"goldens contract.cases repeats id {case_id!r}")
        case_ids.add(case_id)
        input_path = _contract_local_file(contract_path, case.get("input_file"), f"{label}.input_file")
        expected_input_sha256 = _require_sha256(case.get("input_sha256"), f"{label}.input_sha256")
        if _sha256_file(input_path, f"golden input {case_id!r}") != expected_input_sha256:
            raise ManifestValidationError(f"{label}.input_sha256 does not match {label}.input_file")
        _require_integer(case.get("family_id"), f"{label}.family_id", minimum=0)
        if case["family_id"] >= len(EXPLICIT_FAMILY_ORDER):
            raise ManifestValidationError(f"{label}.family_id is outside the explicit routing-family range")
        token_ids = case.get("expected_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ManifestValidationError(f"{label}.expected_token_ids must be a non-empty array")
        for token_index, token_id in enumerate(token_ids):
            _require_integer(token_id, f"{label}.expected_token_ids[{token_index}]", minimum=0)
    return len(cases)


def _validate_conversion_contract(
    conversion_contract_path: Path,
    *,
    manifest_path: Path,
    source_weights_path: Path,
) -> ConversionContractValidation:
    contract_path = Path(conversion_contract_path)
    root = _load_json_object(contract_path, "conversion contract")
    _require_exact_keys(
        root,
        {"format", "source", "graph", "activation_calibration", "preprocessing", "routing", "goldens"},
        "conversion contract",
    )
    _require_exact_string(root.get("format"), CONVERSION_CONTRACT_FORMAT, "conversion contract.format")

    source = _as_record(root.get("source"), "conversion contract.source")
    _require_exact_keys(source, {"manifest_sha256", "weights_sha256"}, "conversion contract.source")
    source_manifest_sha256 = _require_sha256(
        source.get("manifest_sha256"),
        "conversion contract.source.manifest_sha256",
    )
    source_weights_sha256 = _require_sha256(
        source.get("weights_sha256"),
        "conversion contract.source.weights_sha256",
    )
    actual_manifest_sha256 = _sha256_file(manifest_path, "published source manifest")
    if actual_manifest_sha256 != source_manifest_sha256:
        raise ManifestValidationError("conversion contract.source.manifest_sha256 does not match --manifest")
    actual_weights_sha256 = _sha256_file(source_weights_path, "published source SafeTensors payload")
    if actual_weights_sha256 != source_weights_sha256:
        raise ManifestValidationError(
            "conversion contract.source.weights_sha256 does not match the published SafeTensors payload"
        )

    graph_ref = _validated_artifact_reference(contract_path, root.get("graph"), "conversion contract.graph")
    calibration_ref = _validated_artifact_reference(
        contract_path,
        root.get("activation_calibration"),
        "conversion contract.activation_calibration",
    )
    preprocessing_ref = _validated_artifact_reference(
        contract_path,
        root.get("preprocessing"),
        "conversion contract.preprocessing",
    )
    goldens_ref = _validated_artifact_reference(contract_path, root.get("goldens"), "conversion contract.goldens")
    artifact_paths = [graph_ref.path, calibration_ref.path, preprocessing_ref.path, goldens_ref.path]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ManifestValidationError(
            "conversion contract graph, calibration, preprocessing, and goldens must be distinct files"
        )

    graph_id, graph_descriptors = _validate_graph_spec(
        _load_json_object(graph_ref.path, "graph specification"),
        source_manifest_sha256=actual_manifest_sha256,
        source_weights_sha256=actual_weights_sha256,
    )
    calibration_sample_count = _validate_activation_calibration(
        _load_json_object(calibration_ref.path, "activation calibration"),
        contract_path=contract_path,
        graph_sha256=graph_ref.sha256,
        graph_descriptors=graph_descriptors,
    )
    _validate_preprocessing(_load_json_object(preprocessing_ref.path, "preprocessing contract"))
    routing = _validate_explicit_routing(root.get("routing"))
    routing_sha256 = _canonical_json_sha256(routing)
    golden_case_count = _validate_goldens(
        _load_json_object(goldens_ref.path, "goldens contract"),
        contract_path=contract_path,
        source_manifest_sha256=actual_manifest_sha256,
        source_weights_sha256=actual_weights_sha256,
        graph_sha256=graph_ref.sha256,
        calibration_sha256=calibration_ref.sha256,
        preprocessing_sha256=preprocessing_ref.sha256,
        routing_sha256=routing_sha256,
    )
    return ConversionContractValidation(
        path=contract_path,
        contract_sha256=_sha256_file(contract_path, "conversion contract"),
        source_manifest_sha256=actual_manifest_sha256,
        source_weights_sha256=actual_weights_sha256,
        graph_id=graph_id,
        graph_sha256=graph_ref.sha256,
        activation_edge_count=len(graph_descriptors),
        calibration_sha256=calibration_ref.sha256,
        calibration_sample_count=calibration_sample_count,
        preprocessing_sha256=preprocessing_ref.sha256,
        routing=routing,
        routing_sha256=routing_sha256,
        goldens_sha256=goldens_ref.sha256,
        golden_case_count=golden_case_count,
    )


def normalize_release(
    manifest_path: Path,
    normalized_weights_path: Path,
    quantization_fragment_path: Path,
    *,
    conv_keys: Iterable[str] = (),
    conversion_contract_path: Path | None = None,
) -> NormalizationResult:
    """Validate and normalize one published TinyReceiptVQA INT8 release.

    The caller must explicitly list every rank-4 INT8 Conv state-dict key in
    ``conv_keys``.  The resulting fragment's ``weights_quantization`` object
    is intended to be merged into a separately authored Volvox graph config.
    """

    manifest_path = Path(manifest_path)
    normalized_weights_path = Path(normalized_weights_path)
    quantization_fragment_path = Path(quantization_fragment_path)
    manifest = _load_json_object(manifest_path, "manifest")

    tensors, source_filename = _validate_release_contract(manifest)
    specs = _build_specs(tensors, conv_keys)
    source_weights_path = manifest_path.parent / source_filename
    _checked_destination(source_weights_path, normalized_weights_path, quantization_fragment_path)
    contract_validation = (
        _validate_conversion_contract(
            Path(conversion_contract_path),
            manifest_path=manifest_path,
            source_weights_path=source_weights_path,
        )
        if conversion_contract_path is not None
        else None
    )
    source = _load_named_safetensors(source_weights_path)
    _validate_source_name_set(source, specs)

    normalized: dict[str, np.ndarray] = {}
    weights_quantization: dict[str, dict[str, Any]] = {}
    normalized_tensors: dict[str, dict[str, Any]] = {}
    quantized_count = 0
    float32_count = 0

    for spec in specs:
        if spec.source_dtype == "int8_per_out":
            raw = _require_tensor(
                source[spec.source_name],
                name=spec.source_name,
                dtype=np.dtype(np.int8),
                shape=spec.shape,
            )
            scale_name = f"scale.{spec.key}"
            scales = _require_scale(
                source[scale_name],
                name=scale_name,
                output_channels=spec.shape[0],
            )
            # Do not dequantize or recalculate integer values.  The only byte
            # reordering is the explicit semantic OIHW -> OHWI Conv conversion.
            if spec.transform == "OIHW_to_OHWI":
                value = np.ascontiguousarray(np.transpose(raw, (0, 2, 3, 1)))
            else:
                value = np.array(raw, copy=True, order="C")
            normalized[spec.normalized_name] = value
            weights_quantization[spec.normalized_name] = {
                "scheme": "per_axis",
                "axis": 0,
                "scales": [float(item) for item in scales],
                "zero_points": [0] * spec.shape[0],
            }
            quantized_count += 1
        else:
            raw = _require_tensor(
                source[spec.source_name],
                name=spec.source_name,
                dtype=np.dtype(np.float32),
                shape=spec.shape,
            )
            normalized[spec.normalized_name] = np.array(raw, copy=True, order="C")
            float32_count += 1

        normalized_tensors[spec.key] = {
            "source_name": spec.source_name,
            "normalized_name": spec.normalized_name,
            "dtype": "int8" if spec.source_dtype == "int8_per_out" else "float32",
            "shape": list(normalized[spec.normalized_name].shape),
            "transform": spec.transform,
        }

    # Values are proven finite above; this additionally protects a future
    # change from writing a non-portable JSON NaN into the config fragment.
    fragment = {
        "format": FRAGMENT_FORMAT,
        "source_contract": {
            "format": SOURCE_FORMAT,
            "runtime": SOURCE_RUNTIME,
            "weights_file": source_filename,
            "tensor_layout": SOURCE_LAYOUT,
            "quantization": SOURCE_QUANTIZATION,
        },
        "normalization": {
            "normalized_safetensors_format": NORMALIZED_SAFETENSORS_FORMAT,
            "tensor_name_scheme": {
                "int8": NORMALIZED_INT8_PREFIX + "<state_dict_key>",
                "float32": NORMALIZED_FLOAT32_PREFIX + "<state_dict_key>",
            },
            "conv_transform": "selected rank-4 OIHW -> OHWI",
            "rank2_int8_transform": "identity",
        },
        "normalized_tensors": normalized_tensors,
        "weights_quantization": weights_quantization,
    }
    if contract_validation is not None:
        fragment["conversion_contract_validation"] = {
            "format": CONVERSION_CONTRACT_FORMAT,
            "attestation": (
                "validated external artifact identities and descriptors only; "
                "this normalizer did not construct or execute a graph"
            ),
            "contract_sha256": contract_validation.contract_sha256,
            "source": {
                "manifest_sha256": contract_validation.source_manifest_sha256,
                "weights_sha256": contract_validation.source_weights_sha256,
            },
            "graph": {
                "id": contract_validation.graph_id,
                "sha256": contract_validation.graph_sha256,
                "activation_edge_count": contract_validation.activation_edge_count,
            },
            "activation_calibration": {
                "sha256": contract_validation.calibration_sha256,
                "sample_count": contract_validation.calibration_sample_count,
            },
            "preprocessing": {
                "sha256": contract_validation.preprocessing_sha256,
            },
            "routing": contract_validation.routing,
            "routing_sha256": contract_validation.routing_sha256,
            "goldens": {
                "sha256": contract_validation.goldens_sha256,
                "case_count": contract_validation.golden_case_count,
            },
        }
    try:
        fragment_json = json.dumps(fragment, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ManifestValidationError(f"could not serialize quantization fragment: {error}") from error

    normalized_weights_path.parent.mkdir(parents=True, exist_ok=True)
    quantization_fragment_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_file(
            normalized,
            str(normalized_weights_path),
            metadata={
                "format": NORMALIZED_SAFETENSORS_FORMAT,
                "source_format": SOURCE_FORMAT,
            },
        )
        quantization_fragment_path.write_text(fragment_json, encoding="utf-8")
    except (OSError, ValueError) as error:
        raise ManifestValidationError(f"could not write normalized output: {error}") from error

    return NormalizationResult(
        source_weights=source_weights_path,
        normalized_weights=normalized_weights_path,
        quantization_fragment=quantization_fragment_path,
        quantized_tensor_count=quantized_count,
        float32_tensor_count=float32_count,
        conversion_contract_validated=contract_validation is not None,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only TinyReceiptVQA INT8 manifest normalizer. "
            "Writes weights plus a quantization fragment; it does not build a graph or calibration."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path,
                        help="published int8/manifest.json")
    parser.add_argument("--out", required=True, type=Path,
                        help="normalized Volvox SafeTensors output path")
    parser.add_argument("--quantization-out", required=True, type=Path,
                        help="JSON fragment containing weights_quantization")
    parser.add_argument(
        "--conversion-contract",
        type=Path,
        help=(
            "optional hash-bound external W8A8 conversion contract; validates graph-spec, "
            "calibration, preprocessing, explicit family routing, and goldens without constructing a graph"
        ),
    )
    parser.add_argument(
        "--conv-key",
        action="append",
        default=[],
        metavar="STATE_DICT_KEY",
        help=(
            "explicit rank-4 OIHW Conv state-dict key; repeat once for every rank-4 INT8 Conv "
            "tensor (rank-2 INT8 tensors stay byte-for-byte unchanged)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = normalize_release(
            args.manifest,
            args.out,
            args.quantization_out,
            conv_keys=args.conv_key,
            conversion_contract_path=args.conversion_contract,
        )
    except ManifestValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "source_weights": str(result.source_weights),
        "normalized_weights": str(result.normalized_weights),
        "quantization_fragment": str(result.quantization_fragment),
        "quantized_tensor_count": result.quantized_tensor_count,
        "float32_tensor_count": result.float32_tensor_count,
        "conversion_contract_validated": result.conversion_contract_validated,
        "graph_constructed": False,
        "activation_calibration_constructed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
