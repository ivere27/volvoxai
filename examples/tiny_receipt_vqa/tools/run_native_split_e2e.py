#!/usr/bin/env python3
"""Run the deterministic TinyReceipt split package through the public native CLI.

The orchestrator deliberately treats ``native/volvoxai run`` as the only
native execution interface.  It does not import native runtime internals.  One
encoder invocation produces ordinary raw tensors, then each autoregressive
decoder step is another ordinary fixed-length forward.

The saved result contains package-relative identities and content hashes, never
machine-local paths.  Floating-point acceptance is owned by the versioned ONNX
Runtime reference: every router value must satisfy

    abs(actual - expected) <= atol + rtol * abs(expected)

Token IDs, family selection, question encoding, and text hashes are exact.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

TOOLS_DIRECTORY = Path(__file__).resolve().parent
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIRECTORY))

from tiny_receipt_tokenizer import (  # noqa: E402
    TinyReceiptTokenizer,
    TokenizerContractError,
)


PACKAGE_FORMAT = "volvoxai-tiny-receipt-vqa-split-onnx-package-v1"
REFERENCE_SCHEMA = "volvoxai.tiny-receipt-split-e2e-reference/v1"
RESULT_SCHEMA = "volvoxai.tiny-receipt-split-native-e2e-result/v1"
RUNTIME_EVIDENCE_SCHEMA = "volvoxai.runtime-evidence"
WORKLOAD_ID = "synthetic-exact-f32-v1"
PROMPT = "phone number last one"
IMAGE_WIDTH = 672
IMAGE_HEIGHT = 320
QUESTION_LENGTH = 192
DECODER_LENGTH = 192
MAX_NEW_TOKENS = 4
MINIMUM_DECODER_STEPS = 2
IMAGE_SHA256 = "028acedd12b13cfcd706b8c364f41e82dd80fd34218e61612e31c0c9ad5474fa"
QUESTION_TOKEN_IDS_SHA256 = (
    "36a00a0f67e466c6386ffb4220616b1a2745f9ed1fd3a172f2b1fc4803c2ab36"
)
FAMILY_ORDER = (
    "phone",
    "address",
    "store",
    "item_row",
    "item_math",
    "item_lookup",
    "math",
    "other",
)
DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "split_int8_e2e_ort_cpu.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_PATTERNS = {
    "compilationId": re.compile(r"^native-compiled-[0-9]+$"),
    "definitionId": re.compile(r"^native-graph-[0-9]+$"),
    "executionId": re.compile(r"^native-execution-[0-9]+$"),
    "contextId": re.compile(r"^native-context-[0-9]+$"),
    "weightRevisionId": re.compile(r"^native-weight-[0-9]+$"),
}
_ADAPTER_REVISION_RE = re.compile(r"^native-adapter-[0-9]+:[0-9]+$")
_ASSET_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[\\/]|\\\\)")
_CUDA_DEVICE_BANNER_RE = re.compile(
    r"\[CUDA\] device ([0-9]+): (.+) \(compute ([0-9]+)\.([0-9]+)\)"
)
_SOFTWARE_DEVICE_RE = re.compile(
    r"\b(?:swift\s*shader|llvmpipe|lavapipe|softpipe|"
    r"software(?:\s+rasterizer)?|microsoft\s+basic\s+render|cpu)\b",
    re.IGNORECASE,
)


class E2EError(RuntimeError):
    """A package, CLI execution, evidence, or oracle contract failed closed."""


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True)
class Asset:
    path: Path
    relative_path: str
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class GraphDefinition:
    kind: str
    graph: Asset
    weights: Asset
    export_report: Asset
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]


@dataclass(frozen=True)
class Package:
    root: Path
    manifest_sha256: str
    variant: str
    vocabulary: tuple[str, ...]
    vocabulary_sha256: str
    token_ids: Mapping[str, int]
    tokenizer: TinyReceiptTokenizer
    decoder_output: str
    encoder: GraphDefinition
    decoder: GraphDefinition


@dataclass(frozen=True)
class Workload:
    image: array.array
    question_token_ids: tuple[int, ...]
    question_ids: array.array
    family_ids: array.array


@dataclass(frozen=True)
class CudaDeviceAttestation:
    index: int
    name: str
    compute_capability: str

    def public_identity(self) -> dict[str, str]:
        return {
            "name": self.name,
            "computeCapability": self.compute_capability,
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise E2EError(f"could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise E2EError(f"{label} must contain a JSON object")
    return value


def _exact_keys(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise E2EError(f"{label} must contain exactly {', '.join(keys)}")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise E2EError(f"{label} must be a non-empty string")
    return value


def _sha256_string(value: Any, label: str) -> str:
    result = _nonempty_string(value, label)
    if not _SHA256_RE.fullmatch(result):
        raise E2EError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise E2EError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise E2EError(f"{label} must be a finite number")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise E2EError(f"could not hash package asset {path.name!r}: {error}") from error
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _int32_bytes(values: Sequence[int] | array.array) -> bytes:
    result = array.array("i", values)
    if result.itemsize != 4:
        raise E2EError("native Python int32 storage is not four bytes")
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def _float32_bytes(values: Sequence[float] | array.array) -> bytes:
    result = array.array("f", values)
    if result.itemsize != 4:
        raise E2EError("native Python float32 storage is not four bytes")
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def _deterministic_image() -> array.array:
    image = array.array("f")
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            quantized = (17 * x + 29 * y + 7 * (x ^ y)) & 255
            image.append((quantized - 128) / 128)
    return image


def _write_raw(path: Path, values: array.array) -> None:
    if values.typecode == "f":
        data = _float32_bytes(values)
    elif values.typecode == "i":
        data = _int32_bytes(values)
    else:
        raise E2EError(f"unsupported raw array typecode {values.typecode!r}")
    try:
        path.write_bytes(data)
    except OSError as error:
        raise E2EError(f"could not write temporary tensor {path.name!r}: {error}") from error


def _read_raw(path: Path, typecode: str, count: int, label: str) -> array.array:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise E2EError(f"could not read temporary {label}: {error}") from error
    expected = count * 4
    if len(data) != expected:
        raise E2EError(f"{label} must contain {expected} bytes, got {len(data)}")
    result = array.array(typecode)
    result.frombytes(data)
    if result.itemsize != 4:
        raise E2EError(f"{label} host element size is not four bytes")
    if sys.byteorder != "little":
        result.byteswap()
    return result


def _numeric_summary(values: Sequence[int | float], label: str) -> dict[str, int | float]:
    if not values:
        raise E2EError(f"{label} must be non-empty")
    minimum = math.inf
    maximum = -math.inf
    total = 0.0
    absolute_sum = 0.0
    sum_squares = 0.0
    for index, raw in enumerate(values):
        value = _finite_number(raw, f"{label}[{index}]")
        minimum = min(minimum, value)
        maximum = max(maximum, value)
        total += value
        absolute_sum += abs(value)
        sum_squares += value * value
    return {
        "count": len(values),
        "min": minimum,
        "max": maximum,
        "sum": total,
        "absSum": absolute_sum,
        "sumSquares": sum_squares,
    }


def _relative_asset_path(value: Any, label: str) -> str:
    raw = _nonempty_string(value, label)
    try:
        raw.encode("ascii")
    except UnicodeEncodeError as error:
        raise E2EError(f"{label} must be an ASCII package-relative path") from error
    if "\\" in raw or "?" in raw or "#" in raw:
        raise E2EError(f"{label} must be a package-relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or str(path) != raw or any(
        part in ("", ".", "..") or not _ASSET_PART_RE.fullmatch(part)
        for part in path.parts
    ):
        raise E2EError(f"{label} must be a canonical package-relative path")
    return raw


def _asset(root: Path, value: Any, label: str) -> Asset:
    record = _exact_keys(value, ("path", "bytes", "sha256"), label)
    relative = _relative_asset_path(record["path"], f"{label}.path")
    byte_size = record["bytes"]
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
        raise E2EError(f"{label}.bytes must be a positive integer")
    digest = _sha256_string(record["sha256"], f"{label}.sha256")
    path = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise E2EError(f"{label} may not traverse a symbolic link")
    if not path.is_file():
        raise E2EError(f"{label} package asset is missing")
    try:
        actual_size = path.stat().st_size
    except OSError as error:
        raise E2EError(f"could not inspect {label}: {error}") from error
    if actual_size != byte_size or _sha256_file(path) != digest:
        raise E2EError(f"{label} size or SHA-256 does not match package_manifest.json")
    return Asset(path, relative, byte_size, digest)


def _semantic_names(
    value: Any,
    expected: Sequence[str],
    label: str,
) -> dict[str, str]:
    mapping = _exact_keys(value, expected, label)
    result = {
        semantic: _nonempty_string(mapping[semantic], f"{label}.{semantic}")
        for semantic in expected
    }
    if len(set(result.values())) != len(result):
        raise E2EError(f"{label} canonical tensor names must be unique")
    return result


def _graph_definition(
    root: Path,
    value: Any,
    kind: str,
    input_semantics: Sequence[str],
    output_semantics: Sequence[str],
) -> GraphDefinition:
    graph_record = _exact_keys(
        value,
        ("graph", "weights", "export_report", "inputs", "outputs"),
        f"graphs.{kind}",
    )
    graph = _asset(root, graph_record["graph"], f"graphs.{kind}.graph")
    weights = _asset(root, graph_record["weights"], f"graphs.{kind}.weights")
    export_report = _asset(
        root,
        graph_record["export_report"],
        f"graphs.{kind}.export_report",
    )
    inputs = _semantic_names(
        graph_record["inputs"], input_semantics, f"graphs.{kind}.inputs"
    )
    outputs = _semantic_names(
        graph_record["outputs"], output_semantics, f"graphs.{kind}.outputs"
    )
    document = _read_json(graph.path, f"{kind} graph")
    if document.get("format") != "volvox-graph/v1":
        raise E2EError(f"{kind} graph must use format 'volvox-graph/v1'")
    graph_inputs = document.get("inputs")
    graph_outputs = document.get("outputs")
    if not isinstance(graph_inputs, dict) or not set(inputs.values()).issubset(graph_inputs):
        raise E2EError(f"{kind} manifest input names are absent from graph.json")
    if not isinstance(graph_outputs, list) or graph_outputs != list(outputs.values()):
        raise E2EError(f"{kind} manifest output names do not match graph.json")
    return GraphDefinition(
        kind=kind,
        graph=graph,
        weights=weights,
        export_report=export_report,
        inputs=inputs,
        outputs=outputs,
    )


def load_package(package_directory: Path) -> Package:
    root = package_directory
    if root.is_symlink() or not root.is_dir():
        raise E2EError("--package must name a real package directory")
    manifest_path = root / "package_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise E2EError("package_manifest.json is missing or symbolic")
    manifest = _read_json(manifest_path, "package_manifest.json")
    if manifest.get("format") != PACKAGE_FORMAT:
        raise E2EError(f"package format must equal {PACKAGE_FORMAT!r}")

    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        raise E2EError("package assets must be an object")
    _asset(root, assets.get("config"), "assets.config")
    vocab_asset = _asset(root, assets.get("vocab"), "assets.vocab")
    vocab_document = _read_json(vocab_asset.path, "vocab.json")
    try:
        tokenizer = TinyReceiptTokenizer.from_documents(
            manifest.get("tokenizer"),
            vocab_document,
        )
    except TokenizerContractError as error:
        raise E2EError(f"invalid package tokenizer: {error}") from error
    vocabulary = tokenizer.vocabulary
    expected_ids = tokenizer.token_ids

    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict) or (
        preprocessing.get("layout"),
        preprocessing.get("shape"),
        preprocessing.get("color"),
    ) != ("NCHW", [1, 1, IMAGE_HEIGHT, IMAGE_WIDTH], "grayscale"):
        raise E2EError("package preprocessing must be grayscale F32 NCHW [1,1,320,672]")
    families = manifest.get("families")
    if (
        not isinstance(families, dict)
        or families.get("auto_id") != -1
        or families.get("ordered_names") != list(FAMILY_ORDER)
        or families.get("name_to_id")
        != {name: index for index, name in enumerate(FAMILY_ORDER)}
    ):
        raise E2EError("package family order or AUTO=-1 contract is invalid")
    generation = manifest.get("generation")
    common_generation = {
        "strategy": "greedy-autoregressive",
        "decoder_input_length": DECODER_LENGTH,
        "maximum_new_tokens": 191,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tie_policy": "first-index",
    }
    if not isinstance(generation, dict) or any(
        generation.get(key) != expected
        for key, expected in common_generation.items()
    ):
        raise E2EError("package generation contract is invalid")
    if (
        set(generation) == {*common_generation, "logits_row"}
        and generation.get("logits_row") == "prefix_length_minus_one"
    ):
        decoder_output = "logits"
    elif (
        set(generation)
        == {*common_generation, "decoder_output", "token_ids_row"}
        and generation.get("decoder_output") == "token_ids"
        and generation.get("token_ids_row") == "prefix_length_minus_one"
    ):
        decoder_output = "token_ids"
    else:
        raise E2EError("package generation contract is invalid")
    if manifest.get("mask_semantics") != {
        "memory_padding_mask": "nonzero_means_blocked"
    }:
        raise E2EError("memory_padding_mask must use nonzero_means_blocked")

    graphs = manifest.get("graphs")
    if not isinstance(graphs, dict) or set(graphs) != {"encoder", "decoder"}:
        raise E2EError("package must contain exactly encoder and decoder graphs")
    encoder = _graph_definition(
        root,
        graphs["encoder"],
        "encoder",
        ("image", "question_ids", "family_ids"),
        ("memory", "memory_padding_mask", "router_logits", "selected_family_ids"),
    )
    decoder_record = graphs["decoder"]
    decoder_manifest_inputs = (
        decoder_record.get("inputs")
        if isinstance(decoder_record, dict)
        else None
    )
    base_decoder_inputs = {
        "decoder_input_ids",
        "memory",
        "memory_padding_mask",
        "family_ids",
    }
    if not isinstance(decoder_manifest_inputs, dict):
        raise E2EError("graphs.decoder.inputs must be an object")
    if set(decoder_manifest_inputs) == base_decoder_inputs:
        decoder_input_semantics = (
            "decoder_input_ids",
            "memory",
            "memory_padding_mask",
            "family_ids",
        )
    elif set(decoder_manifest_inputs) == {*base_decoder_inputs, "v4_keep"}:
        decoder_input_semantics = (
            "decoder_input_ids",
            "memory",
            "memory_padding_mask",
            "family_ids",
            "v4_keep",
        )
    else:
        raise E2EError(
            "graphs.decoder.inputs must contain the split inputs and optional v4_keep"
        )
    decoder = _graph_definition(
        root,
        decoder_record,
        "decoder",
        decoder_input_semantics,
        (decoder_output,),
    )
    variant_record = manifest.get("variant")
    variant = (
        variant_record.get("requested")
        if isinstance(variant_record, dict)
        else None
    )
    if variant != "int8-w8a8":
        raise E2EError("the shared E2E reference requires the int8-w8a8 package variant")
    vocabulary_sha256 = tokenizer.package_vocabulary_sha256
    return Package(
        root=root,
        manifest_sha256=_sha256_file(manifest_path),
        variant=variant,
        vocabulary=tuple(vocabulary),
        vocabulary_sha256=vocabulary_sha256,
        token_ids=dict(expected_ids),
        tokenizer=tokenizer,
        decoder_output=decoder_output,
        encoder=encoder,
        decoder=decoder,
    )


def create_workload(package: Package) -> Workload:
    image = _deterministic_image()
    if _sha256_bytes(_float32_bytes(image)) != IMAGE_SHA256:
        raise E2EError("deterministic image bytes do not match workload v1")

    cleaned = re.sub(r"\s+", " ", PROMPT.replace("\n", " ")).strip()
    question = package.tokenizer.encode(
        cleaned,
        add_eos=True,
        max_len=QUESTION_LENGTH,
    )
    question_ids = array.array(
        "i",
        question
        + [package.token_ids["pad"]] * (QUESTION_LENGTH - len(question)),
    )
    return Workload(
        image=image,
        question_token_ids=tuple(question),
        question_ids=question_ids,
        family_ids=array.array("i", [-1]),
    )


def _validate_reference(
    reference: Mapping[str, Any],
    vocabulary_size: int | None = None,
) -> Mapping[str, Any]:
    _exact_keys(
        reference,
        ("schema", "provenance", "package", "workload", "expected"),
        "reference",
    )
    if reference["schema"] != REFERENCE_SCHEMA:
        raise E2EError(f"reference schema must equal {REFERENCE_SCHEMA!r}")
    provenance = _exact_keys(
        reference["provenance"],
        ("kind", "sourceFormat", "sourceVariant", "provider", "description"),
        "reference.provenance",
    )
    if (
        provenance["kind"] != "onnx-runtime-oracle"
        or provenance["sourceFormat"] != "tiny_receipt_vqa_split_onnx_v1"
        or provenance["sourceVariant"] != "int8-w8a8"
    ):
        raise E2EError("reference provenance is not the split INT8 ONNX Runtime oracle")
    _nonempty_string(provenance["provider"], "reference.provenance.provider")
    _nonempty_string(provenance["description"], "reference.provenance.description")

    package = _exact_keys(
        reference["package"],
        ("format", "assets", "vocabularySha256"),
        "reference.package",
    )
    if package["format"] != PACKAGE_FORMAT:
        raise E2EError("reference package format is invalid")
    package_assets = _exact_keys(
        package["assets"],
        ("encoderGraph", "encoderWeights", "decoderGraph", "decoderWeights"),
        "reference.package.assets",
    )
    for name, digest in package_assets.items():
        _sha256_string(digest, f"reference.package.assets.{name}")
    _sha256_string(
        package["vocabularySha256"], "reference.package.vocabularySha256"
    )

    workload = _exact_keys(
        reference["workload"],
        ("id", "prompt", "family", "maxNewTokens", "minimumDecoderSteps", "image"),
        "reference.workload",
    )
    image = _exact_keys(
        workload["image"],
        ("generator", "dtype", "shape", "byteOrder", "sha256"),
        "reference.workload.image",
    )
    computed_image_sha256 = _sha256_bytes(
        _float32_bytes(_deterministic_image())
    )
    if computed_image_sha256 != IMAGE_SHA256:
        raise E2EError("internal deterministic image contract is inconsistent")
    if (
        workload["id"] != WORKLOAD_ID
        or workload["prompt"] != PROMPT
        or workload["family"] != "auto"
        or workload["maxNewTokens"] != MAX_NEW_TOKENS
        or workload["minimumDecoderSteps"] != MINIMUM_DECODER_STEPS
        or image["generator"]
        != "q=(17*x+29*y+7*(x^y))&255; f32=(q-128)/128"
        or image["dtype"] != "float32"
        or image["shape"] != [1, 1, IMAGE_HEIGHT, IMAGE_WIDTH]
        or image["byteOrder"] != "little"
    ):
        raise E2EError("reference workload does not match the deterministic v1 contract")
    _sha256_string(image["sha256"], "reference.workload.image.sha256")
    if image["sha256"] != computed_image_sha256:
        raise E2EError(
            "reference.workload.image.sha256 fails its internal integrity check"
        )

    expected = _exact_keys(
        reference["expected"],
        (
            "family",
            "familyId",
            "requestedFamily",
            "requestedFamilyId",
            "questionTokenIds",
            "questionTokenIdsSha256",
            "tokenIds",
            "tokenIdsSha256",
            "textUtf8Sha256",
            "stoppedAtEos",
            "routerLogits",
        ),
        "reference.expected",
    )
    maximum_token_id = (
        vocabulary_size - 1 if vocabulary_size is not None else (1 << 31) - 1
    )
    if vocabulary_size is not None and vocabulary_size <= 0:
        raise E2EError("reference vocabulary size must be positive")
    for field, maximum in (
        ("questionTokenIds", QUESTION_LENGTH),
        ("tokenIds", MAX_NEW_TOKENS),
    ):
        values = expected[field]
        if (
            not isinstance(values, list)
            or len(values) > maximum
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > maximum_token_id
                for value in values
            )
        ):
            raise E2EError(f"reference.expected.{field} contains invalid token IDs")
    for field in (
        "questionTokenIdsSha256",
        "tokenIdsSha256",
        "textUtf8Sha256",
    ):
        _sha256_string(expected[field], f"reference.expected.{field}")
    recomputed_token_hashes = {
        "questionTokenIdsSha256": _sha256_bytes(
            _int32_bytes(expected["questionTokenIds"])
        ),
        "tokenIdsSha256": _sha256_bytes(_int32_bytes(expected["tokenIds"])),
    }
    for field, recomputed in recomputed_token_hashes.items():
        if expected[field] != recomputed:
            raise E2EError(
                f"reference.expected.{field} fails its internal integrity check"
            )
    router = _exact_keys(
        expected["routerLogits"],
        ("values", "sha256", "summary", "atol", "rtol"),
        "reference.expected.routerLogits",
    )
    if not isinstance(router["values"], list) or len(router["values"]) != 8:
        raise E2EError("reference router logits must contain eight values")
    for index, value in enumerate(router["values"]):
        _finite_number(value, f"reference router logits[{index}]")
    _sha256_string(router["sha256"], "reference router SHA-256")
    recomputed_router_sha256 = _sha256_bytes(_float32_bytes(router["values"]))
    if router["sha256"] != recomputed_router_sha256:
        raise E2EError("reference router SHA-256 fails its internal integrity check")
    summary = _exact_keys(
        router["summary"],
        ("count", "min", "max", "sum", "absSum", "sumSquares"),
        "reference router summary",
    )
    if summary["count"] != 8:
        raise E2EError("reference router summary count must equal eight")
    for field in ("min", "max", "sum", "absSum", "sumSquares"):
        _finite_number(summary[field], f"reference router summary {field}")
    recomputed_summary = _numeric_summary(
        router["values"], "reference router logits"
    )
    for field, recomputed in recomputed_summary.items():
        if summary[field] != recomputed:
            raise E2EError(
                f"reference router summary {field} fails its internal integrity check"
            )
    for field in ("atol", "rtol"):
        if _finite_number(router[field], f"reference router {field}") < 0:
            raise E2EError(f"reference router {field} must be non-negative")
    return reference


def _package_identity(package: Package) -> dict[str, Any]:
    return {
        "format": PACKAGE_FORMAT,
        "assets": {
            "encoderGraph": package.encoder.graph.sha256,
            "encoderWeights": package.encoder.weights.sha256,
            "decoderGraph": package.decoder.graph.sha256,
            "decoderWeights": package.decoder.weights.sha256,
        },
        "vocabularySha256": package.vocabulary_sha256,
    }


def _validate_route(value: Any, label: str) -> None:
    route = _exact_keys(value, ("tierFallback", "operator"), label)
    operator = _exact_keys(
        route["operator"],
        ("attestation", "used", "offendingNode"),
        f"{label}.operator",
    )
    if (
        route["tierFallback"] is not False
        or operator["attestation"] != "reported"
        or operator["used"] is not False
        or operator["offendingNode"] is not None
    ):
        raise E2EError(f"{label} does not attest an exact no-fallback route")


def _validate_revisions(value: Any, label: str, execution: bool) -> None:
    revisions = _exact_keys(
        value,
        (
            "topologyRevision",
            "weightRevision",
            "weightRevisionId",
            "adapterRevisionId",
            "adapterRevisionIds",
        ),
        label,
    )
    for field in ("topologyRevision", "weightRevision"):
        if not isinstance(revisions[field], str) or not re.fullmatch(
            r"(?:0|[1-9][0-9]*)", revisions[field]
        ):
            raise E2EError(f"{label}.{field} must be an unsigned decimal string")
    if not _IDENTITY_PATTERNS["weightRevisionId"].fullmatch(
        _nonempty_string(revisions["weightRevisionId"], f"{label}.weightRevisionId")
    ):
        raise E2EError(f"{label}.weightRevisionId is not a native identity")
    current = revisions["adapterRevisionId"]
    if current is not None and (
        not isinstance(current, str) or not _ADAPTER_REVISION_RE.fullmatch(current)
    ):
        raise E2EError(f"{label}.adapterRevisionId is invalid")
    all_revisions = revisions["adapterRevisionIds"]
    if not isinstance(all_revisions, list) or (execution and not all_revisions):
        raise E2EError(f"{label}.adapterRevisionIds is invalid")
    for revision in all_revisions:
        if revision is not None and (
            not isinstance(revision, str)
            or not _ADAPTER_REVISION_RE.fullmatch(revision)
        ):
            raise E2EError(f"{label}.adapterRevisionIds contains an invalid identity")
    if execution:
        expected_current = all_revisions[0] if len(all_revisions) == 1 else None
        if current != expected_current:
            raise E2EError(f"{label}.adapterRevisionId is inconsistent")
    elif current is not None and current not in all_revisions:
        raise E2EError(f"{label}.adapterRevisionId is not declared")


def _validate_device(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise E2EError(f"{label} must be a non-empty string map")
    for key, entry in value.items():
        _nonempty_string(key, f"{label} key")
        _nonempty_string(entry, f"{label}.{key}")


def _validate_runtime_evidence(
    evidence: Mapping[str, Any],
    backend: str,
    expected_outputs: Sequence[str],
) -> dict[str, Any]:
    _exact_keys(
        evidence,
        ("schema", "version", "compilation", "execution", "stableResult"),
        "runtime evidence",
    )
    if evidence["schema"] != RUNTIME_EVIDENCE_SCHEMA or evidence["version"] != 1:
        raise E2EError("unsupported runtime evidence schema or version")
    compilation = _exact_keys(
        evidence["compilation"],
        (
            "compilationId",
            "policy",
            "selectedBackend",
            "selectedDevice",
            "definitionId",
            "revisions",
            "route",
        ),
        "runtime evidence compilation",
    )
    policy = _exact_keys(
        compilation["policy"],
        ("mode", "backend", "operatorFallback"),
        "runtime evidence compilation.policy",
    )
    if policy != {
        "mode": "require",
        "backend": backend,
        "operatorFallback": "forbid",
    } or compilation["selectedBackend"] != backend:
        raise E2EError(f"runtime evidence does not prove strict {backend!r} compilation")
    for field in ("compilationId", "definitionId"):
        value = _nonempty_string(compilation[field], f"compilation.{field}")
        if not _IDENTITY_PATTERNS[field].fullmatch(value):
            raise E2EError(f"compilation.{field} is not a native identity")
    _validate_device(compilation["selectedDevice"], "compilation.selectedDevice")
    _validate_revisions(compilation["revisions"], "compilation.revisions", False)
    _validate_route(compilation["route"], "compilation.route")

    execution = _exact_keys(
        evidence["execution"],
        (
            "executionId",
            "contextId",
            "backend",
            "device",
            "outcome",
            "revisions",
            "route",
            "decodeState",
        ),
        "runtime evidence execution",
    )
    if (
        execution["backend"] != backend
        or execution["outcome"] != "success"
        or execution["device"] != compilation["selectedDevice"]
    ):
        raise E2EError("runtime execution does not match strict compilation")
    for field in ("executionId", "contextId"):
        value = _nonempty_string(execution[field], f"execution.{field}")
        if not _IDENTITY_PATTERNS[field].fullmatch(value):
            raise E2EError(f"execution.{field} is not a native identity")
    _validate_device(execution["device"], "execution.device")
    _validate_revisions(execution["revisions"], "execution.revisions", True)
    _validate_route(execution["route"], "execution.route")
    if any(
        execution["revisions"][field] != compilation["revisions"][field]
        for field in ("topologyRevision", "weightRevision", "weightRevisionId")
    ):
        raise E2EError("execution revisions do not match compilation")
    decode = _exact_keys(
        execution["decodeState"],
        ("operation", "mode", "cacheState", "cacheGeneration", "position"),
        "execution.decodeState",
    )
    if decode != {
        "operation": "execute",
        "mode": None,
        "cacheState": "not-applicable",
        "cacheGeneration": None,
        "position": None,
    }:
        raise E2EError("ordinary CLI run emitted unexpected decode state")

    stable = _exact_keys(
        evidence["stableResult"],
        (
            "outputs",
            "freshCallerOwnedReads",
            "readableAfterContextClose",
            "contextClosedBeforeResult",
            "resultClosedAfterVerification",
        ),
        "runtime evidence stableResult",
    )
    for field in (
        "freshCallerOwnedReads",
        "readableAfterContextClose",
        "contextClosedBeforeResult",
        "resultClosedAfterVerification",
    ):
        if stable[field] is not True:
            raise E2EError(f"runtime evidence stableResult.{field} must be true")
    outputs = stable["outputs"]
    if not isinstance(outputs, list) or not outputs:
        raise E2EError("runtime evidence stable outputs must be non-empty")
    names: list[str] = []
    bytes_per_element = {"float32": 4, "int32": 4, "int8": 1, "uint8": 1}
    for index, output in enumerate(outputs):
        descriptor = _exact_keys(
            output,
            ("name", "shape", "dtype", "location", "byteLength"),
            f"stable output {index}",
        )
        name = _nonempty_string(descriptor["name"], f"stable output {index}.name")
        if name in names:
            raise E2EError(f"runtime evidence contains duplicate output {name!r}")
        names.append(name)
        shape = descriptor["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise E2EError(f"stable output {name!r} has an invalid shape")
        dtype = descriptor["dtype"]
        if dtype not in bytes_per_element or descriptor["location"] not in (
            "host",
            "device",
        ):
            raise E2EError(f"stable output {name!r} has an invalid dtype or location")
        elements = math.prod(shape)
        if descriptor["byteLength"] != elements * bytes_per_element[dtype]:
            raise E2EError(f"stable output {name!r} has an invalid byteLength")
    if names != list(expected_outputs):
        raise E2EError("runtime evidence output names do not match package_manifest.json")
    return {
        "schema": RUNTIME_EVIDENCE_SCHEMA,
        "version": 1,
        "backend": backend,
        "device": _json_clone(compilation["selectedDevice"]),
        "strictNoFallback": True,
        "stableOutputs": names,
    }


def _parse_cuda_device_attestation(
    stdout: str,
    stderr: str,
) -> CudaDeviceAttestation:
    banner_lines = [
        line
        for value in (stdout, stderr)
        for line in value.splitlines()
        if line.startswith("[CUDA] device")
    ]
    if not banner_lines:
        raise E2EError(
            "required CUDA physical-device banner is missing from a native "
            "CLI invocation"
        )
    identities: list[CudaDeviceAttestation] = []
    for line in banner_lines:
        match = _CUDA_DEVICE_BANNER_RE.fullmatch(line)
        if match is None:
            raise E2EError("malformed CUDA physical-device banner")
        index = int(match.group(1))
        name = match.group(2)
        major = int(match.group(3))
        minor = int(match.group(4))
        if (
            name != name.strip()
            or len(name) > 256
            or "/" in name
            or "\\" in name
            or _ABSOLUTE_PATH_RE.match(name)
            or any(
                ord(character) < 0x20 or ord(character) > 0x7E
                for character in name
            )
        ):
            raise E2EError("CUDA device banner contains a non-path-safe device name")
        if _SOFTWARE_DEVICE_RE.search(name):
            raise E2EError("CUDA device banner identifies a software device")
        if major <= 0 or minor < 0 or minor > 99:
            raise E2EError(
                "CUDA device banner contains an invalid compute capability"
            )
        identities.append(
            CudaDeviceAttestation(
                index=index,
                name=name,
                compute_capability=f"{major}.{minor}",
            )
        )
    identity = identities[0]
    if any(candidate != identity for candidate in identities[1:]):
        raise E2EError(
            "CUDA physical-device banners disagree within one native CLI invocation"
        )
    return identity


def _safe_cli_text(
    value: str,
    package_root: Path,
    artifact_root: Path,
    binary: Path,
) -> str:
    sanitized = value
    replacements = (
        (str(package_root), "<package>"),
        (str(artifact_root), "<artifacts>"),
        (str(binary), "<native-binary>"),
    )
    for source, target in replacements:
        if source:
            sanitized = sanitized.replace(source, target)
    return sanitized.strip()[:4000]


def _run_cli(
    *,
    binary: Path,
    graph: GraphDefinition,
    backend: str,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    report_path: Path,
    artifact_root: Path,
    package_root: Path,
    timeout_seconds: float,
    require_cuda_device: bool,
) -> tuple[dict[str, Any], CudaDeviceAttestation | None]:
    command = [
        str(binary),
        "run",
        str(graph.graph.path),
        "--weights",
        str(graph.weights.path),
        f"--{backend}",
    ]
    for semantic, path in inputs.items():
        command.extend(["--input", f"{graph.inputs[semantic]}={path}"])
    for semantic, path in outputs.items():
        command.extend(["--output", f"{graph.outputs[semantic]}={path}"])
    command.extend(["--report-json", str(report_path)])
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    try:
        process = subprocess.run(
            command,
            cwd=artifact_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise E2EError(
            "native CLI could not complete: "
            + _safe_cli_text(str(error), package_root, artifact_root, binary)
        ) from error
    if process.returncode != 0:
        diagnostic = _safe_cli_text(
            process.stderr or process.stdout,
            package_root,
            artifact_root,
            binary,
        )
        raise E2EError(
            f"native CLI exited with status {process.returncode}"
            + (f": {diagnostic}" if diagnostic else "")
        )
    cuda_device = (
        _parse_cuda_device_attestation(process.stdout, process.stderr)
        if require_cuda_device
        else None
    )
    evidence = _read_json(report_path, "native runtime evidence")
    return (
        _validate_runtime_evidence(
            evidence,
            backend,
            list(graph.outputs.values()),
        ),
        cuda_device,
    )


def _argmax_first(values: Sequence[float]) -> int:
    if not values:
        raise E2EError("cannot select a token from empty logits")
    best_index = 0
    best_value = _finite_number(values[0], "logits[0]")
    for index in range(1, len(values)):
        value = _finite_number(values[index], f"logits[{index}]")
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


def _compare_reference(
    *,
    package_identity: Mapping[str, Any],
    output: Mapping[str, Any],
    reference: Mapping[str, Any],
    vocabulary_size: int,
) -> dict[str, Any]:
    _validate_reference(reference, vocabulary_size)
    if package_identity != reference["package"]:
        raise E2EError("runtime package identity does not match the oracle reference")
    expected = reference["expected"]
    exact_fields = (
        "family",
        "familyId",
        "requestedFamily",
        "requestedFamilyId",
        "questionTokenIdsSha256",
        "tokenIdsSha256",
        "textUtf8Sha256",
        "stoppedAtEos",
    )
    for field in exact_fields:
        if output[field] != expected[field]:
            raise E2EError(f"output {field} does not match the oracle")
    for field in ("questionTokenIds", "tokenIds"):
        if output[field] != expected[field]:
            raise E2EError(f"output {field} does not match the oracle")
    if len(output["tokenIds"]) < MINIMUM_DECODER_STEPS:
        raise E2EError("oracle workload did not execute at least two decoder forwards")
    expected_router = expected["routerLogits"]
    actual_router = output["routerLogits"]
    value_tolerances: list[float] = []
    for index, (actual, wanted) in enumerate(
        zip(actual_router["values"], expected_router["values"], strict=True)
    ):
        tolerance = expected_router["atol"] + expected_router["rtol"] * abs(wanted)
        value_tolerances.append(tolerance)
        if abs(actual - wanted) > tolerance:
            raise E2EError(
                f"router logit {index} exceeds the reference tolerance {tolerance}"
            )
    expected_summary = expected_router["summary"]
    actual_summary = actual_router["summary"]
    if actual_summary["count"] != expected_summary["count"]:
        raise E2EError("router summary count does not match the oracle")
    summary_bounds = {
        "min": max(value_tolerances),
        "max": max(value_tolerances),
        "sum": sum(value_tolerances),
        "absSum": sum(value_tolerances),
        "sumSquares": sum(
            2.0 * abs(wanted) * tolerance + tolerance * tolerance
            for wanted, tolerance in zip(
                expected_router["values"], value_tolerances, strict=True
            )
        ),
    }
    for field, bound in summary_bounds.items():
        if abs(actual_summary[field] - expected_summary[field]) > bound:
            raise E2EError(
                f"router summary {field} exceeds its derived tolerance {bound}"
            )
    return {
        "schema": reference["schema"],
        "provenanceKind": reference["provenance"]["kind"],
        "matched": True,
        "exactTokenIds": True,
        "routerWithinTolerance": True,
        "routerSummaryWithinTolerance": True,
        "routerTolerance": {
            "rule": "abs(actual-expected) <= atol + rtol*abs(expected)",
            "atol": expected_router["atol"],
            "rtol": expected_router["rtol"],
        },
    }


def _assert_path_safe_json(value: Any, label: str = "result") -> None:
    if isinstance(value, str):
        if _ABSOLUTE_PATH_RE.match(value):
            raise E2EError(f"{label} contains an absolute path")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_path_safe_json(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_path_safe_json(item, f"{label}.{key}")


@contextlib.contextmanager
def _artifact_workspace(requested: Path | None) -> Iterator[tuple[Path, bool]]:
    if requested is None:
        with tempfile.TemporaryDirectory(prefix="volvoxai-tiny-receipt-e2e-") as directory:
            yield Path(directory), False
        return
    if requested.is_symlink() or not requested.is_dir():
        raise E2EError("--artifact-dir must be an existing real directory")
    try:
        if any(requested.iterdir()):
            raise E2EError("--artifact-dir must be empty")
    except OSError as error:
        raise E2EError(f"could not inspect --artifact-dir: {error}") from error
    yield requested, True


def run_native_split_e2e(
    *,
    package_directory: Path,
    binary: Path,
    backend: str,
    reference_path: Path,
    artifact_directory: Path | None = None,
    timeout_seconds: float = 600.0,
    require_cuda_device: bool = False,
) -> dict[str, Any]:
    if backend not in ("cpu", "cuda"):
        raise E2EError("backend must be 'cpu' or 'cuda'")
    if require_cuda_device and backend != "cuda":
        raise E2EError("--require-cuda-device requires the CUDA backend")
    if package_directory.is_symlink():
        raise E2EError("--package may not be a symbolic link")
    if reference_path.is_symlink():
        raise E2EError("--reference may not be a symbolic link")
    if artifact_directory is not None and artifact_directory.is_symlink():
        raise E2EError("--artifact-dir may not be a symbolic link")
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise E2EError("--binary must name an executable regular file")
    try:
        package_directory = package_directory.resolve(strict=True)
        binary = binary.resolve(strict=True)
        reference_path = reference_path.resolve(strict=True)
        artifact_directory = (
            artifact_directory.resolve(strict=True)
            if artifact_directory is not None
            else None
        )
    except OSError as error:
        raise E2EError("could not resolve an E2E input") from error
    package = load_package(package_directory)
    reference = _validate_reference(
        _read_json(reference_path, "E2E reference"),
        len(package.vocabulary),
    )
    workload = create_workload(package)
    package_identity = _package_identity(package)

    with _artifact_workspace(artifact_directory) as (artifacts, retained):
        paths = {
            "image": artifacts / "image.f32",
            "question_ids": artifacts / "question_ids.i32",
            "family_ids": artifacts / "family_ids.i32",
            "memory": artifacts / "memory.f32",
            "memory_padding_mask": artifacts / "memory_padding_mask.i32",
            "router_logits": artifacts / "router_logits.f32",
            "selected_family_ids": artifacts / "selected_family_ids.i32",
            "decoder_input_ids": artifacts / "decoder_input_ids.i32",
            "decoder_keep": artifacts / "decoder_keep.i32",
            "decoder_family_ids": artifacts / "decoder_family_ids.i32",
            "logits": artifacts / "logits.f32",
            "token_ids": artifacts / "token_ids.i32",
        }
        _write_raw(paths["image"], workload.image)
        _write_raw(paths["question_ids"], workload.question_ids)
        _write_raw(paths["family_ids"], workload.family_ids)

        evidence: list[dict[str, Any]] = []
        encoder_evidence, cuda_device = _run_cli(
            binary=binary,
            graph=package.encoder,
            backend=backend,
            inputs={
                "image": paths["image"],
                "question_ids": paths["question_ids"],
                "family_ids": paths["family_ids"],
            },
            outputs={
                "memory": paths["memory"],
                "memory_padding_mask": paths["memory_padding_mask"],
                "router_logits": paths["router_logits"],
                "selected_family_ids": paths["selected_family_ids"],
            },
            report_path=artifacts / "encoder_evidence.json",
            artifact_root=artifacts,
            package_root=package.root,
            timeout_seconds=timeout_seconds,
            require_cuda_device=require_cuda_device,
        )
        evidence.append({"stage": "encoder", "step": None, **encoder_evidence})
        runtime_device = encoder_evidence["device"]
        router_logits = _read_raw(
            paths["router_logits"], "f", 8, "encoder router logits"
        )
        selected = _read_raw(
            paths["selected_family_ids"], "i", 1, "selected family ID"
        )
        selected_family_id = int(selected[0])
        if not 0 <= selected_family_id < len(FAMILY_ORDER):
            raise E2EError("encoder selected an invalid family ID")
        mask = _read_raw(
            paths["memory_padding_mask"],
            "i",
            402,
            "encoder memory padding mask",
        )
        if any(value not in (0, 1) for value in mask):
            raise E2EError("encoder memory_padding_mask must contain I32 0/1 flags")
        # Validate the complete memory byte count before feeding it back through
        # the public CLI. Keeping it on disk avoids a redundant Python copy.
        _read_raw(paths["memory"], "f", 402 * 320, "encoder memory")

        decoder_ids = array.array(
            "i",
            [package.token_ids["pad"]] * DECODER_LENGTH,
        )
        decoder_ids[0] = package.token_ids["bos"]
        decoder_keep = array.array("i", [0] * DECODER_LENGTH)
        decoder_keep[0] = 1
        decoder_family = array.array("i", [selected_family_id])
        _write_raw(paths["decoder_family_ids"], decoder_family)
        generated: list[int] = []
        stopped_at_eos = False
        prefix_length = 1
        for step in range(MAX_NEW_TOKENS):
            _write_raw(paths["decoder_input_ids"], decoder_ids)
            decoder_inputs = {
                "decoder_input_ids": paths["decoder_input_ids"],
                "memory": paths["memory"],
                "memory_padding_mask": paths["memory_padding_mask"],
                "family_ids": paths["decoder_family_ids"],
            }
            if "v4_keep" in package.decoder.inputs:
                _write_raw(paths["decoder_keep"], decoder_keep)
                decoder_inputs["v4_keep"] = paths["decoder_keep"]
            decoder_outputs = {
                package.decoder_output: paths[package.decoder_output]
            }
            decoder_evidence, decoder_cuda_device = _run_cli(
                binary=binary,
                graph=package.decoder,
                backend=backend,
                inputs=decoder_inputs,
                outputs=decoder_outputs,
                report_path=artifacts / f"decoder_{step:02d}_evidence.json",
                artifact_root=artifacts,
                package_root=package.root,
                timeout_seconds=timeout_seconds,
                require_cuda_device=require_cuda_device,
            )
            if decoder_evidence["device"] != runtime_device:
                raise E2EError(
                    "native runtime device evidence changed between CLI invocations"
                )
            if require_cuda_device and decoder_cuda_device != cuda_device:
                raise E2EError(
                    "CUDA physical-device banner changed between CLI invocations"
                )
            evidence.append({"stage": "decoder", "step": step, **decoder_evidence})
            row = prefix_length - 1
            if package.decoder_output == "token_ids":
                selected_tokens = _read_raw(
                    paths["token_ids"],
                    "i",
                    DECODER_LENGTH,
                    f"decoder token IDs step {step}",
                )
                next_token = int(selected_tokens[row])
                if not 0 <= next_token < len(package.vocabulary):
                    raise E2EError(
                        f"decoder emitted out-of-vocabulary token {next_token}"
                    )
            else:
                logits = _read_raw(
                    paths["logits"],
                    "f",
                    DECODER_LENGTH * len(package.vocabulary),
                    f"decoder logits step {step}",
                )
                row_offset = row * len(package.vocabulary)
                next_token = _argmax_first(
                    logits[row_offset : row_offset + len(package.vocabulary)]
                )
            generated.append(next_token)
            if next_token == package.token_ids["eos"]:
                stopped_at_eos = True
                break
            decoder_ids[prefix_length] = next_token
            decoder_keep[prefix_length] = 1
            prefix_length += 1
        if len(generated) < MINIMUM_DECODER_STEPS:
            raise E2EError(
                f"decoder executed {len(generated)} forward(s), fewer than required "
                f"{MINIMUM_DECODER_STEPS}"
            )

        text = package.tokenizer.decode(generated)
        output = {
            "family": FAMILY_ORDER[selected_family_id],
            "familyId": selected_family_id,
            "requestedFamily": "auto",
            "requestedFamilyId": -1,
            "questionTokenIds": list(workload.question_token_ids),
            "questionTokenIdsSha256": _sha256_bytes(
                _int32_bytes(workload.question_token_ids)
            ),
            "tokenIds": generated,
            "tokenIdsSha256": _sha256_bytes(_int32_bytes(generated)),
            "text": text,
            "textUtf8Sha256": _sha256_bytes(text.encode("utf-8")),
            "stoppedAtEos": stopped_at_eos,
            "routerLogits": {
                "values": list(router_logits),
                "sha256": _sha256_bytes(_float32_bytes(router_logits)),
                "summary": _numeric_summary(router_logits, "router logits"),
            },
        }
        devices = [_json_clone(runtime_device)]
        provider = {
            "strictNoFallback": True,
            "invocationCount": len(evidence),
            "compilationCount": len(evidence),
            "executionCount": len(evidence),
            "encoderExecutions": 1,
            "decoderExecutions": len(generated),
            "devices": devices,
            "evidence": evidence,
        }
        if require_cuda_device:
            if cuda_device is None:
                raise E2EError("required CUDA physical-device attestation is missing")
            provider["cudaDevice"] = cuda_device.public_identity()
        result = {
            "schema": RESULT_SCHEMA,
            "status": "pass",
            "backend": backend,
            "package": package_identity,
            "workload": {
                "id": WORKLOAD_ID,
                "prompt": PROMPT,
                "family": "auto",
                "maxNewTokens": MAX_NEW_TOKENS,
                "minimumDecoderSteps": MINIMUM_DECODER_STEPS,
                "image": {
                    "generator": "q=(17*x+29*y+7*(x^y))&255; f32=(q-128)/128",
                    "dtype": "float32",
                    "shape": [1, 1, IMAGE_HEIGHT, IMAGE_WIDTH],
                    "byteOrder": "little",
                    "sha256": IMAGE_SHA256,
                    "summary": _numeric_summary(workload.image, "workload image"),
                },
            },
            "output": output,
            "provider": provider,
            "lifecycle": {
                "cliProcessPerForward": True,
                "temporaryArtifacts": "retained" if retained else "released",
            },
            "reference": _compare_reference(
                package_identity=package_identity,
                output=output,
                reference=reference,
                vocabulary_size=len(package.vocabulary),
            ),
        }
        _assert_path_safe_json(result)
        return result


def _open_json_output_parent(path: Path) -> int:
    if not path.name or path.name in (".", ".."):
        raise E2EError("--json-out must name a file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        if path.is_absolute():
            current_fd = os.open(path.anchor, flags)
            components = path.parts[1:-1]
        else:
            current_fd = os.open(".", flags)
            components = path.parts[:-1]
    except OSError as error:
        raise E2EError(
            "--json-out parent directory does not exist or is inaccessible"
        ) from error
    try:
        for component in components:
            if component in ("", "."):
                continue
            try:
                descriptor = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise E2EError(
                    "--json-out parent directory does not exist or is inaccessible"
                ) from error
            if stat.S_ISLNK(descriptor.st_mode):
                raise E2EError(
                    "--json-out parent components may not be symbolic links"
                )
            if not stat.S_ISDIR(descriptor.st_mode):
                raise E2EError("--json-out parent component is not a directory")
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise E2EError(
                        "--json-out parent components may not be symbolic links"
                    ) from error
                raise E2EError(
                    "--json-out parent directory does not exist or is inaccessible"
                ) from error
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _atomic_write_new_json(path: Path, serialized: str) -> None:
    parent_fd = _open_json_output_parent(path)
    temporary_name = (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    temporary_fd = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            flags,
            0o666,
            dir_fd=parent_fd,
        )
        with os.fdopen(
            temporary_fd,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            temporary_fd = -1
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise E2EError(
                "--json-out refuses to overwrite an existing file"
            ) from error
        except OSError as error:
            raise E2EError(
                "could not atomically publish --json-out"
            ) from error
    except E2EError:
        raise
    except OSError as error:
        raise E2EError(f"could not write --json-out: {error}") from error
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
        os.close(parent_fd)


def _write_result(result: Mapping[str, Any], destination: str) -> None:
    serialized = json.dumps(
        result,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if destination == "-":
        sys.stdout.write(serialized)
        return
    path = Path(destination)
    _atomic_write_new_json(path, serialized)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the shared TinyReceipt split INT8 E2E workload through only "
            "the public native/volvoxai CLI."
        )
    )
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("native/volvoxai"),
        help="already-built inference CLI (default: native/volvoxai)",
    )
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--cpu", action="store_const", dest="backend", const="cpu")
    backend.add_argument("--cuda", action="store_const", dest="backend", const="cuda")
    parser.add_argument(
        "--require-cuda-device",
        action="store_true",
        help=(
            "require the native CUDA physical-device banner on every forward "
            "and reject software or changing identities"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="versioned ONNX Runtime oracle JSON",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help=(
            "existing empty temporary directory; retained for inspection. "
            "Without this option an automatically cleaned directory is used"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="timeout for each encoder/decoder CLI invocation",
    )
    parser.add_argument(
        "--json-out",
        default="-",
        help="new result file, or '-' for stdout (default)",
    )
    arguments = parser.parse_args(argv)
    if (
        not math.isfinite(arguments.timeout_seconds)
        or arguments.timeout_seconds <= 0
    ):
        parser.error("--timeout-seconds must be positive and finite")
    if arguments.require_cuda_device and arguments.backend != "cuda":
        parser.error("--require-cuda-device requires --cuda")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        result = run_native_split_e2e(
            package_directory=arguments.package,
            binary=arguments.binary,
            backend=arguments.backend,
            reference_path=arguments.reference,
            artifact_directory=arguments.artifact_dir,
            timeout_seconds=arguments.timeout_seconds,
            require_cuda_device=arguments.require_cuda_device,
        )
        _write_result(result, arguments.json_out)
    except E2EError as error:
        print(f"[TinyReceipt native split E2E] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
