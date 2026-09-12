#!/usr/bin/env python3
"""Import the typed TinyReceipt split-ONNX release into a VolvoxAI package.

The importer is deliberately local and model-specific. It accepts only the
explicit-KV producer contract, proves its positive-length cache sentinel
against ONNX Runtime, and then delegates both graphs to the repository's
generic ONNX exporter. It never loads a PyTorch checkpoint and never downloads
model data.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.exporter.capabilities import (  # noqa: E402
    classify_package,
    expand_targets,
    normalize_targets,
)
from tools.exporter.generated.kernel_registry import (  # noqa: E402
    TARGETS as GENERIC_EXPORT_TARGETS,
)

SOURCE_FORMAT = "tiny_receipt_vqa_split_kv_onnx_v2"
PACKAGE_FORMAT = "volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2"
BPE_VOCAB_SIZE = 1536
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
BPE_ATOMIC_TOKENS = (
    "<field>",
    "</field>",
    "<value>",
    "</value>",
    "<op>",
    "</op>",
    "<answer>",
    "</answer>",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)
BPE_BYTE_TOKENS = tuple(f"<0x{value:02X}>" for value in range(256))
BPE_VOCAB_KEYS = frozenset({
    "type",
    "version",
    "vocab_size",
    "itos",
    "merges",
    "normalization",
    "atomic_tokens",
    "byte_tokens",
    "unused_tokens",
    "special_tokens",
    "tokenizer_hash",
})
BPE_WHITESPACE_CODEPOINTS = frozenset(
    (
        0x0009,
        0x000A,
        0x000B,
        0x000C,
        0x000D,
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
    + tuple(range(0x2000, 0x200B))
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
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 672
IMAGE_CHANNELS = 1
IMAGE_TOKENS = 210
MAX_Q = 192
MAX_T = 192
SOURCE_FILES = {
    "manifest": "manifest.json",
    "encoder": "encoder_model.onnx",
    "decoder": "decoder_model.onnx",
    "config": "config.json",
    "vocab": "vocab.json",
}
INT8_W8A8_SOURCE_FILES = {
    "encoder_int8_w8a8": "encoder_model_int8.onnx",
    "decoder_int8_w8a8": "decoder_model_int8.onnx",
}
CLI_VARIANTS = ("fp32", "int8-w8a8")
MANIFEST_VARIANT_KEYS = {
    "fp32": "fp32",
    "int8-w8a8": "int8_w8a8",
}
SELECTED_MODEL_KEYS = {
    "fp32": {
        "encoder": "encoder",
        "decoder": "decoder",
    },
    "int8-w8a8": {
        "encoder": "encoder_int8_w8a8",
        "decoder": "decoder_int8_w8a8",
    },
}
BASE_SOURCE_FILE_KEYS = frozenset({
    "encoder",
    "decoder",
    "config",
    "vocab",
    "encoder_sha256",
    "encoder_bytes",
    "decoder_sha256",
    "decoder_bytes",
})
INT8_W8A8_SOURCE_FILE_KEYS = frozenset({
    "encoder_int8_w8a8",
    "decoder_int8_w8a8",
    "encoder_int8_w8a8_sha256",
    "encoder_int8_w8a8_bytes",
    "decoder_int8_w8a8_sha256",
    "decoder_int8_w8a8_bytes",
})
GENERIC_EXPORTER = REPOSITORY_ROOT / "tools" / "export_safetensors.py"
GRAPH_FORMAT = "volvox-graph/v1"
CANONICAL_DIMENSIONS = ("B", "Q", "T", "M")
KV_CANONICAL_DIMENSIONS = ("B", "Q", "M", "P", "R")
KV_LAYERS = 4
KV_HEADS = 8
KV_HEAD_WIDTH = 40
KV_CROSS_NAMES = tuple(
    name
    for layer in range(KV_LAYERS)
    for name in (f"cross_k_{layer}", f"cross_v_{layer}")
)
KV_PAST_NAMES = tuple(
    name
    for layer in range(KV_LAYERS)
    for name in (f"past_k_{layer}", f"past_v_{layer}")
)
KV_PRESENT_NAMES = tuple(
    name
    for layer in range(KV_LAYERS)
    for name in (f"present_k_{layer}", f"present_v_{layer}")
)
SHAPE_PROFILES = {
    "short": {"B": 1, "Q": 16, "T": 16, "M": 226},
    "representative": {"B": 1, "Q": 64, "T": 64, "M": 274},
    "maximum": {"B": 1, "Q": 192, "T": 192, "M": 402},
}
POSITION_INPUTS = {
    "encoder": ("question_position_ids", "Q"),
    "decoder": ("decoder_position_ids", "T"),
}
CAUSAL_MASK_INPUT = "decoder_causal_mask"


class ImportFailure(RuntimeError):
    """A fail-closed source or publication contract violation."""


class _DuplicateKey(ValueError):
    pass


class _NonFiniteJsonConstant(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise _NonFiniteJsonConstant(f"non-finite numeric constant {value!r}")


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateKey,
        _NonFiniteJsonConstant,
    ) as error:
        raise ImportFailure(f"{label} is not valid duplicate-free UTF-8 JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise ImportFailure(f"{label} root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _regular_file(root: Path, name: str) -> Path:
    path = root / name
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ImportFailure(f"required source file {name!r} is unavailable: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ImportFailure(f"required source file {name!r} must be a regular non-symlink file")
    return path


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ImportFailure(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ImportFailure(f"{label} must be a non-negative integer")
    return value


def _validated_max_batch_size(value: Any, producer_maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > producer_maximum
    ):
        raise ImportFailure(
            "max batch size must be an integer from 1 through "
            f"the producer maximum {producer_maximum}"
        )
    return value


def _tokenizer_fingerprint(vocab: Mapping[str, Any]) -> str:
    unhashed = dict(vocab)
    unhashed.pop("tokenizer_hash", None)
    encoded = json.dumps(
        unhashed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_bpe_vocabulary(
    vocab: Mapping[str, Any],
    manifest_tokenizer: Any,
    *,
    vocab_size: int,
) -> dict[str, Any]:
    label = "source byte_fallback_bpe vocabulary"
    _require_exact_keys(vocab, BPE_VOCAB_KEYS, label)
    if (
        vocab_size != BPE_VOCAB_SIZE
        or vocab.get("type") != "byte_fallback_bpe"
        or type(vocab.get("version")) is not int
        or vocab.get("version") != 1
        or type(vocab.get("vocab_size")) is not int
        or vocab.get("vocab_size") != vocab_size
        or vocab.get("normalization") != "NFC"
    ):
        raise ImportFailure(
            f"{label} must declare byte_fallback_bpe v1 with {BPE_VOCAB_SIZE} tokens"
        )

    itos = vocab.get("itos")
    if (
        not isinstance(itos, list)
        or len(itos) != vocab_size
        or any(not isinstance(item, str) or not item for item in itos)
        or len(set(itos)) != len(itos)
        or itos[: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS)
    ):
        raise ImportFailure(f"{label}.itos is not a unique {vocab_size}-token table")
    itos_set = frozenset(itos)

    atomic_tokens = vocab.get("atomic_tokens")
    byte_tokens = vocab.get("byte_tokens")
    unused_tokens = vocab.get("unused_tokens")
    special_tokens = vocab.get("special_tokens")
    if atomic_tokens != list(BPE_ATOMIC_TOKENS):
        raise ImportFailure(f"{label}.atomic_tokens is not the structured OCR contract")
    if byte_tokens != list(BPE_BYTE_TOKENS):
        raise ImportFailure(f"{label}.byte_tokens must contain <0x00> through <0xFF>")
    if special_tokens != {
        "pad": "<pad>",
        "bos": "<bos>",
        "eos": "<eos>",
        "unk": "<unk>",
    }:
        raise ImportFailure(f"{label}.special_tokens is invalid")
    if (
        not isinstance(unused_tokens, list)
        or any(not isinstance(item, str) or not item for item in unused_tokens)
        or len(set(unused_tokens)) != len(unused_tokens)
        or not set(unused_tokens).issubset(itos_set)
    ):
        raise ImportFailure(f"{label}.unused_tokens is invalid")
    required_tokens = frozenset((*SPECIAL_TOKENS, *BPE_ATOMIC_TOKENS, *BPE_BYTE_TOKENS))
    if not required_tokens.issubset(itos_set):
        raise ImportFailure(f"{label}.itos omits required special, atomic, or byte tokens")

    merges = vocab.get("merges")
    if not isinstance(merges, list):
        raise ImportFailure(f"{label}.merges must be an ordered array")
    merge_pairs: list[tuple[str, str]] = []
    for index, value in enumerate(merges):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ImportFailure(f"{label}.merges[{index}] must contain two token strings")
        left, right = value
        merged = left + right
        if left not in itos_set or right not in itos_set or merged not in itos_set:
            raise ImportFailure(f"{label}.merges[{index}] references an absent token")
        if left in required_tokens or right in required_tokens:
            raise ImportFailure(
                f"{label}.merges[{index}] crosses an atomic or byte-fallback boundary"
            )
        if any(character in "0123456789" for character in merged) or any(
            ord(character) in BPE_WHITESPACE_CODEPOINTS for character in merged
        ):
            raise ImportFailure(
                f"{label}.merges[{index}] crosses a digit or whitespace boundary"
            )
        merge_pairs.append((left, right))
    if len(set(merge_pairs)) != len(merge_pairs):
        raise ImportFailure(f"{label}.merges repeats a merge pair")

    tokenizer_hash = vocab.get("tokenizer_hash")
    if (
        not isinstance(tokenizer_hash, str)
        or len(tokenizer_hash) != 64
        or any(character not in "0123456789abcdef" for character in tokenizer_hash)
        or tokenizer_hash != _tokenizer_fingerprint(vocab)
    ):
        raise ImportFailure(f"{label}.tokenizer_hash does not match vocabulary contents")

    source_contract = {
        "type": "byte_fallback_bpe",
        "version": 1,
        "vocab_size": vocab_size,
        "normalization": "NFC",
        "tokenizer_hash": tokenizer_hash,
    }
    if not isinstance(manifest_tokenizer, Mapping):
        raise ImportFailure("source manifest tokenizer must be an object")
    _require_exact_keys(
        manifest_tokenizer,
        frozenset(source_contract),
        "source manifest tokenizer",
    )
    if (
        type(manifest_tokenizer.get("version")) is not int
        or type(manifest_tokenizer.get("vocab_size")) is not int
        or manifest_tokenizer != source_contract
    ):
        raise ImportFailure(
            "source manifest tokenizer does not match the byte_fallback_bpe vocabulary"
        )
    return {
        **source_contract,
        "itos_key": "itos",
        "merges_key": "merges",
        "token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
    }


def _validate_vocabulary(
    config: Mapping[str, Any],
    vocab: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    vocab_size = _positive_int(config.get("vocab_size"), "source config.vocab_size")
    return vocab_size, _validate_bpe_vocabulary(
        vocab,
        manifest.get("tokenizer"),
        vocab_size=vocab_size,
    )


def _tensor_signature(value: Any) -> tuple[int, list[int | str | None]]:
    tensor_type = value.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append(None)
    return int(tensor_type.elem_type), dimensions


def _nested_tensor_protos(message: Any, tensor_proto_type: type[Any]):
    """Yield every TensorProto reachable through a protobuf message tree."""

    if isinstance(message, tensor_proto_type):
        yield message
        return
    list_fields = getattr(message, "ListFields", None)
    if not callable(list_fields):
        return
    for field, value in list_fields():
        if field.cpp_type != field.CPPTYPE_MESSAGE:
            continue
        if field.is_repeated:
            for item in value:
                yield from _nested_tensor_protos(item, tensor_proto_type)
        else:
            yield from _nested_tensor_protos(value, tensor_proto_type)


def _nested_nodes(message: Any, node_proto_type: type[Any]):
    """Yield nodes from the main graph, subgraphs, and local functions."""

    if isinstance(message, node_proto_type):
        yield message
    list_fields = getattr(message, "ListFields", None)
    if not callable(list_fields):
        return
    for field, value in list_fields():
        if field.cpp_type != field.CPPTYPE_MESSAGE:
            continue
        if field.is_repeated:
            for item in value:
                yield from _nested_nodes(item, node_proto_type)
        else:
            yield from _nested_nodes(value, node_proto_type)


def _initializer_values(initializer: Any, numpy_helper: Any, label: str) -> list[Any]:
    try:
        return numpy_helper.to_array(initializer).reshape(-1).tolist()
    except Exception as error:
        raise ImportFailure(f"{label} initializer could not be decoded: {error}") from error


def _node_axis(node: Any) -> int | None:
    axes = [
        int(attribute.i)
        for attribute in node.attribute
        if attribute.name == "axis"
    ]
    if len(axes) > 1:
        raise ImportFailure(f"QDQ node {node.name or node.op_type!r} repeats axis")
    return axes[0] if axes else None


def _validate_u8s8_static_qdq(model: Any, *, label: str, onnx: Any) -> frozenset[str]:
    """Validate the producer's static per-tensor-U8/per-channel-S8 QDQ form."""

    initializers = {item.name: item for item in model.graph.initializer}
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)

    def initializer(name: str, purpose: str) -> Any:
        value = initializers.get(name)
        if value is None:
            raise ImportFailure(
                f"{label} {purpose} must be a static initializer"
            )
        return value

    def positive_f32_scale(name: str, purpose: str) -> Any:
        value = initializer(name, purpose)
        if int(value.data_type) != int(onnx.TensorProto.FLOAT):
            raise ImportFailure(f"{label} {purpose} must use FLOAT scale data")
        numbers = _initializer_values(value, onnx.numpy_helper, f"{label} {purpose}")
        if not numbers or any(
            not math.isfinite(float(number)) or float(number) <= 0.0
            for number in numbers
        ):
            raise ImportFailure(f"{label} {purpose} scales must be finite and positive")
        return value

    quantize_nodes = [
        node for node in model.graph.node if node.op_type == "QuantizeLinear"
    ]
    dequantize_nodes = [
        node for node in model.graph.node if node.op_type == "DequantizeLinear"
    ]
    if not quantize_nodes or not dequantize_nodes:
        raise ImportFailure(f"{label} is not a static U8S8 QDQ graph")

    for node in quantize_nodes:
        if len(node.input) != 3 or len(node.output) != 1 or node.attribute:
            raise ImportFailure(
                f"{label} QuantizeLinear nodes must use scalar scale/zero-point inputs"
            )
        scale = positive_f32_scale(node.input[1], "activation quantization")
        zero_point = initializer(node.input[2], "activation zero point")
        if tuple(scale.dims) or tuple(zero_point.dims):
            raise ImportFailure(
                f"{label} activation quantization must be per-tensor"
            )
        if int(zero_point.data_type) != int(onnx.TensorProto.UINT8):
            raise ImportFailure(f"{label} activation quantization must produce UINT8")
        output_consumers = consumers.get(node.output[0], [])
        if (
            len(output_consumers) != 1
            or output_consumers[0].op_type != "DequantizeLinear"
        ):
            raise ImportFailure(
                f"{label} QuantizeLinear output must feed exactly one DequantizeLinear"
            )

    signed_weight_count = 0
    for node in dequantize_nodes:
        if len(node.input) != 3 or len(node.output) != 1:
            raise ImportFailure(
                f"{label} DequantizeLinear nodes require explicit static scale/zero point"
            )
        source_name, scale_name, zero_name = node.input
        source_producer = producers.get(source_name)
        axis = _node_axis(node)
        if source_producer is not None:
            if source_producer.op_type != "QuantizeLinear":
                raise ImportFailure(
                    f"{label} activation DequantizeLinear must consume QuantizeLinear"
                )
            if tuple(source_producer.input[1:3]) != (scale_name, zero_name):
                raise ImportFailure(
                    f"{label} activation QDQ boundaries must share quantization parameters"
                )
            if axis is not None:
                raise ImportFailure(
                    f"{label} activation DequantizeLinear must remain per-tensor"
                )
            continue

        source = initializer(source_name, "dequantized integer tensor")
        scale = positive_f32_scale(scale_name, "integer dequantization")
        zero_point = initializer(zero_name, "integer zero point")
        source_dtype = int(source.data_type)
        if source_dtype not in {
            int(onnx.TensorProto.INT8),
            int(onnx.TensorProto.INT32),
        }:
            raise ImportFailure(
                f"{label} static dequantization must use signed INT8 weights or INT32 bias"
            )
        if int(zero_point.data_type) != source_dtype:
            raise ImportFailure(
                f"{label} dequantization zero point must match its integer tensor dtype"
            )
        if axis is None or axis < 0 or axis >= len(source.dims):
            raise ImportFailure(
                f"{label} signed weights and bias must declare a valid channel axis"
            )
        channel_count = int(source.dims[axis])
        if (
            tuple(scale.dims) != (channel_count,)
            or tuple(zero_point.dims) != (channel_count,)
        ):
            raise ImportFailure(
                f"{label} signed weights and bias must use per-output-channel parameters"
            )
        zero_values = _initializer_values(
            zero_point,
            onnx.numpy_helper,
            f"{label} integer zero point",
        )
        if any(int(value) != 0 for value in zero_values):
            raise ImportFailure(f"{label} signed QDQ tensors must use zero zero-points")
        if source_dtype == int(onnx.TensorProto.INT8):
            signed_weight_count += 1

    target_ops = frozenset(
        node.op_type
        for node in model.graph.node
        if node.op_type in {"Conv", "MatMul", "Gemm"}
    )
    if not target_ops or signed_weight_count == 0:
        raise ImportFailure(f"{label} has no signed W8 quantized compute")
    for node in model.graph.node:
        if node.op_type not in target_ops:
            continue
        if len(node.input) < 2:
            raise ImportFailure(f"{label} {node.op_type} has an incomplete QDQ input set")
        data_dq = producers.get(node.input[0])
        weight_dq = producers.get(node.input[1])
        if (
            data_dq is None
            or data_dq.op_type != "DequantizeLinear"
            or weight_dq is None
            or weight_dq.op_type != "DequantizeLinear"
        ):
            raise ImportFailure(
                f"{label} {node.op_type} must consume explicit QDQ operands"
            )
        data_quantize = producers.get(data_dq.input[0])
        if data_quantize is None or data_quantize.op_type != "QuantizeLinear":
            raise ImportFailure(
                f"{label} {node.op_type} activation must use UINT8 QDQ"
            )
        weight_source = initializers.get(weight_dq.input[0])
        weight_quantize = producers.get(weight_dq.input[0])
        if node.op_type in {"Conv", "Gemm"} and (
            weight_source is None
            or int(weight_source.data_type) != int(onnx.TensorProto.INT8)
        ):
            raise ImportFailure(
                f"{label} {node.op_type} weights must use signed INT8 QDQ"
            )
        if node.op_type == "MatMul" and not (
            (
                weight_source is not None
                and int(weight_source.data_type) == int(onnx.TensorProto.INT8)
            )
            or (
                weight_quantize is not None
                and weight_quantize.op_type == "QuantizeLinear"
            )
        ):
            raise ImportFailure(
                f"{label} MatMul second operand must use S8 weights or U8 activations"
            )
    return target_ops


def _validate_onnx(
    path: Path,
    *,
    label: str,
    expected_inputs: Mapping[str, tuple[int, list[int | str]]],
    expected_outputs: Mapping[str, tuple[int, list[int | str]]],
    expected_opset: int,
    require_u8s8_qdq: bool = False,
) -> frozenset[str]:
    try:
        import onnx
    except ImportError as error:  # pragma: no cover - production dependency.
        raise ImportFailure("the local ONNX package is required to validate this source") from error
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as error:
        raise ImportFailure(f"{label} ONNX loading failed: {error}") from error
    for tensor in _nested_tensor_protos(model, onnx.TensorProto):
        if (
            int(tensor.data_location) == int(onnx.TensorProto.EXTERNAL)
            or tensor.external_data
        ):
            raise ImportFailure(f"{label} contains external tensor data")
    custom_nodes = sorted({
        str(node.domain)
        for node in _nested_nodes(model, onnx.NodeProto)
        if node.domain not in {"", "ai.onnx"}
    })
    if custom_nodes:
        raise ImportFailure(
            f"{label} contains unsupported custom-domain nodes {custom_nodes}"
        )
    try:
        onnx.checker.check_model(model)
    except Exception as error:
        raise ImportFailure(f"{label} ONNX validation failed: {error}") from error
    standard_opsets = [
        int(item.version) for item in model.opset_import if item.domain in {"", "ai.onnx"}
    ]
    if standard_opsets != [expected_opset]:
        raise ImportFailure(
            f"{label} must declare exactly ai.onnx opset {expected_opset}, got {standard_opsets}"
        )
    inputs = {item.name: _tensor_signature(item) for item in model.graph.input}
    outputs = {item.name: _tensor_signature(item) for item in model.graph.output}
    if inputs != dict(expected_inputs):
        raise ImportFailure(f"{label} input signature differs from the typed producer contract")
    if outputs != dict(expected_outputs):
        raise ImportFailure(f"{label} output signature differs from the typed producer contract")
    if require_u8s8_qdq:
        return _validate_u8s8_static_qdq(model, label=label, onnx=onnx)
    return frozenset()


_MONOMIAL_ONE = (0, 0, 0, 0)


def _poly(items: Mapping[tuple[int, ...], Fraction | int]):
    return tuple(sorted(
        (powers, Fraction(coefficient))
        for powers, coefficient in items.items()
        if coefficient
    ))


def _poly_add(left, right, *, scale: int = 1):
    result = dict(left)
    for powers, coefficient in right:
        result[powers] = result.get(powers, Fraction(0)) + scale * coefficient
    return _poly(result)


def _poly_multiply(left, right):
    result: dict[tuple[int, ...], Fraction] = {}
    for left_powers, left_coefficient in left:
        for right_powers, right_coefficient in right:
            powers = tuple(
                left_powers[index] + right_powers[index]
                for index in range(len(_MONOMIAL_ONE))
            )
            result[powers] = (
                result.get(powers, Fraction(0))
                + left_coefficient * right_coefficient
            )
    return _poly(result)


@dataclass(frozen=True, eq=False)
class _SymbolicExtent:
    """Small exact rational-polynomial domain for ONNX shape expressions."""

    numerator: tuple[tuple[tuple[int, ...], Fraction], ...]
    denominator: tuple[tuple[tuple[int, ...], Fraction], ...]
    preferred_symbol: str | None = None

    @staticmethod
    def constant(value: int, preferred_symbol: str | None = None) -> "_SymbolicExtent":
        return _SymbolicExtent(
            _poly({_MONOMIAL_ONE: Fraction(value)}),
            _poly({_MONOMIAL_ONE: Fraction(1)}),
            preferred_symbol,
        )

    @staticmethod
    def symbol(
        index: int,
        preferred_symbol: str | None = None,
    ) -> "_SymbolicExtent":
        powers = [0] * len(_MONOMIAL_ONE)
        if index < 0 or index >= len(powers):
            raise ImportFailure(f"symbolic extent index {index} is outside the domain")
        powers[index] = 1
        return _SymbolicExtent(
            _poly({tuple(powers): Fraction(1)}),
            _poly({_MONOMIAL_ONE: Fraction(1)}),
            preferred_symbol,
        )

    def __add__(self, other: Any) -> "_SymbolicExtent":
        right = _as_symbolic_extent(other)
        return _SymbolicExtent(
            _poly_add(
                _poly_multiply(self.numerator, right.denominator),
                _poly_multiply(right.numerator, self.denominator),
            ),
            _poly_multiply(self.denominator, right.denominator),
        )

    def __radd__(self, other: Any) -> "_SymbolicExtent":
        return self + other

    def __sub__(self, other: Any) -> "_SymbolicExtent":
        right = _as_symbolic_extent(other)
        return _SymbolicExtent(
            _poly_add(
                _poly_multiply(self.numerator, right.denominator),
                _poly_multiply(right.numerator, self.denominator),
                scale=-1,
            ),
            _poly_multiply(self.denominator, right.denominator),
        )

    def __rsub__(self, other: Any) -> "_SymbolicExtent":
        return _as_symbolic_extent(other) - self

    def __mul__(self, other: Any) -> "_SymbolicExtent":
        right = _as_symbolic_extent(other)
        return _SymbolicExtent(
            _poly_multiply(self.numerator, right.numerator),
            _poly_multiply(self.denominator, right.denominator),
        )

    def __rmul__(self, other: Any) -> "_SymbolicExtent":
        return self * other

    def __truediv__(self, other: Any) -> "_SymbolicExtent":
        right = _as_symbolic_extent(other)
        if not right.numerator:
            raise ImportFailure("ONNX shape expression divides by zero")
        return _SymbolicExtent(
            _poly_multiply(self.numerator, right.denominator),
            _poly_multiply(self.denominator, right.numerator),
        )

    def __floordiv__(self, other: Any) -> "_SymbolicExtent":
        # Every producer shape division in this model is exact.  Retaining it
        # as a rational expression lets equality prove exactness below.
        return self / other

    def __pow__(self, exponent: int) -> "_SymbolicExtent":
        if not isinstance(exponent, int) or exponent < 0:
            raise ImportFailure("ONNX shape powers must be non-negative integers")
        result = _SymbolicExtent.constant(1)
        for _ in range(exponent):
            result *= self
        return result

    def equivalent(self, other: Any) -> bool:
        right = _as_symbolic_extent(other)
        return _poly_multiply(self.numerator, right.denominator) == _poly_multiply(
            right.numerator,
            self.denominator,
        )

    def integer(self) -> int | None:
        if not self.numerator:
            return 0
        numerator = dict(self.numerator)
        denominator = dict(self.denominator)
        if set(numerator) != set(denominator):
            return None
        ratio: Fraction | None = None
        for powers, denominator_coefficient in denominator.items():
            if denominator_coefficient == 0:
                return None
            current = numerator[powers] / denominator_coefficient
            if ratio is None:
                ratio = current
            elif current != ratio:
                return None
        if ratio is None or ratio.denominator != 1:
            return None
        return int(ratio.numerator)


def _as_symbolic_extent(value: Any) -> _SymbolicExtent:
    if isinstance(value, _SymbolicExtent):
        return value
    # NumPy object ufuncs preserve zero-rank arrays around Python objects.
    # Treat that representation as the scalar it contains so symbolic shape
    # arithmetic is independent of the producer/NumPy scalar convention.
    if getattr(value, "shape", None) == () and callable(getattr(value, "item", None)):
        return _as_symbolic_extent(value.item())
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImportFailure(f"unsupported ONNX symbolic extent value {value!r}")
    return _SymbolicExtent.constant(value)


def _parse_symbolic_extent(value: str, *, role: str) -> _SymbolicExtent | None:
    symbols = {
        "batch": _SymbolicExtent.symbol(3, "B"),
        "B": _SymbolicExtent.symbol(3, "B"),
        "question_length": _SymbolicExtent.symbol(0),
        "Q": _SymbolicExtent.symbol(0),
        "target_length": _SymbolicExtent.symbol(1),
        "T": _SymbolicExtent.symbol(1),
        "memory_length": (
            _SymbolicExtent.symbol(0) + 210
            if role == "encoder"
            else _SymbolicExtent.symbol(2)
        ),
        "M": (
            _SymbolicExtent.symbol(0) + 210
            if role == "encoder"
            else _SymbolicExtent.symbol(2)
        ),
    }

    def lower(node: ast.AST) -> _SymbolicExtent:
        if isinstance(node, ast.Expression):
            return lower(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return _SymbolicExtent.constant(node.value)
        if isinstance(node, ast.Name) and node.id in symbols:
            return symbols[node.id]
        if isinstance(node, ast.BinOp):
            left = lower(node.left)
            right = lower(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                return left / right
            if isinstance(node.op, ast.Pow):
                exponent = right.integer()
                if exponent is not None:
                    return left ** exponent
        raise ValueError("unsupported shape expression")

    try:
        return lower(ast.parse(value, mode="eval"))
    except (SyntaxError, ValueError, ImportFailure):
        return None


def _value_symbolic_shape(value: Any, *, role: str) -> list[_SymbolicExtent | None] | None:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    result: list[_SymbolicExtent | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(_SymbolicExtent.constant(int(dimension.dim_value)))
        elif dimension.HasField("dim_param"):
            result.append(_parse_symbolic_extent(str(dimension.dim_param), role=role))
        else:
            result.append(None)
    return result


def _onnx_int_attribute(node: Any, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _symbolic_integer(value: Any, label: str) -> int:
    extent = _as_symbolic_extent(value)
    result = extent.integer()
    if result is None:
        raise ImportFailure(f"{label} must be an immutable integer")
    return result


def _symbolic_array(value: Any, np: Any) -> Any:
    source = np.asarray(value)
    result = np.empty(source.shape, dtype=object)
    for index in np.ndindex(source.shape):
        item = source[index].item()
        if isinstance(item, bool) or not isinstance(item, int):
            raise ImportFailure("ONNX shape tensor must contain integers")
        result[index] = _SymbolicExtent.constant(item)
    if source.shape == ():
        return np.asarray(_SymbolicExtent.constant(int(source.item())), dtype=object)
    return result


def _symbolic_product(values: Sequence[_SymbolicExtent]) -> _SymbolicExtent:
    result = _SymbolicExtent.constant(1)
    for value in values:
        result *= value
    return result


def _symbolic_broadcast_shape(
    left: Sequence[_SymbolicExtent | None] | None,
    right: Sequence[_SymbolicExtent | None] | None,
    *,
    label: str,
) -> list[_SymbolicExtent] | None:
    if left is None or right is None or any(value is None for value in [*left, *right]):
        return None
    result: list[_SymbolicExtent] = []
    padded_left = [_SymbolicExtent.constant(1)] * (max(len(left), len(right)) - len(left)) + [
        value for value in left if value is not None
    ]
    padded_right = [_SymbolicExtent.constant(1)] * (max(len(left), len(right)) - len(right)) + [
        value for value in right if value is not None
    ]
    for left_extent, right_extent in zip(padded_left, padded_right):
        if left_extent.equivalent(1):
            result.append(right_extent)
        elif right_extent.equivalent(1) or left_extent.equivalent(right_extent):
            result.append(left_extent)
        else:
            raise ImportFailure(f"{label} has incompatible symbolic broadcast extents")
    return result


def _resolved_reshape_target(
    input_shape: Sequence[_SymbolicExtent | None] | None,
    raw_target: Sequence[_SymbolicExtent],
    *,
    allowzero: bool,
    label: str,
) -> list[_SymbolicExtent]:
    if input_shape is None or any(value is None for value in input_shape):
        raise ImportFailure(f"{label} input shape is not symbolically complete")
    concrete_input = [value for value in input_shape if value is not None]
    result: list[_SymbolicExtent | None] = []
    inferred_index: int | None = None
    for index, value in enumerate(raw_target):
        integer = value.integer()
        if integer == 0 and not allowzero:
            if index >= len(concrete_input):
                raise ImportFailure(f"{label} zero-copy axis is outside the input rank")
            result.append(concrete_input[index])
        elif integer == -1:
            if inferred_index is not None:
                raise ImportFailure(f"{label} contains multiple inferred dimensions")
            inferred_index = index
            result.append(None)
        elif integer is not None and integer <= 0:
            raise ImportFailure(f"{label} contains non-positive reshape extent {integer}")
        else:
            result.append(value)
    if inferred_index is not None:
        known = [value for value in result if value is not None]
        result[inferred_index] = _symbolic_product(concrete_input) / _symbolic_product(known)
    resolved = [value for value in result if value is not None]
    if len(resolved) != len(result) or not _symbolic_product(resolved).equivalent(
        _symbolic_product(concrete_input)
    ):
        raise ImportFailure(f"{label} does not preserve the symbolic element count")
    return resolved


def _canonical_reshape_initializer(
    input_shape: Sequence[_SymbolicExtent | None] | None,
    output_shape: Sequence[_SymbolicExtent],
    *,
    label: str,
) -> list[int]:
    if input_shape is None or any(value is None for value in input_shape):
        raise ImportFailure(f"{label} input shape is not symbolically complete")
    concrete_input = [value for value in input_shape if value is not None]
    if not _symbolic_product(concrete_input).equivalent(_symbolic_product(output_shape)):
        raise ImportFailure(f"{label} does not preserve the symbolic element count")
    result: list[int | None] = []
    inferred: list[int] = []
    for index, extent in enumerate(output_shape):
        if (
            extent.preferred_symbol is not None
            and index < len(concrete_input)
            and extent.equivalent(concrete_input[index])
        ):
            result.append(0)
            continue
        integer = extent.integer()
        if integer is not None:
            if integer <= 0:
                raise ImportFailure(f"{label} resolved to non-positive extent {integer}")
            result.append(integer)
        elif index < len(concrete_input) and extent.equivalent(concrete_input[index]):
            result.append(0)
        else:
            inferred.append(index)
            result.append(None)
    if len(inferred) > 1:
        raise ImportFailure(
            f"{label} needs {len(inferred)} independent inferred dimensions; "
            "ONNX Reshape permits at most one"
        )
    if inferred:
        result[inferred[0]] = -1
    return [int(value) for value in result if value is not None]


def _canonical_symbolic_dimension(
    value: _SymbolicExtent,
    *,
    role: str,
) -> int | str:
    if value.preferred_symbol == "B" or value.equivalent(
        _SymbolicExtent.symbol(3, "B")
    ):
        return "B"
    integer = value.integer()
    if integer is not None:
        if integer <= 0:
            raise ImportFailure(f"canonical symbolic dimension resolved to {integer}")
        return integer
    if role == "encoder":
        if value.equivalent(_SymbolicExtent.symbol(0)):
            return "Q"
        if value.equivalent(_SymbolicExtent.symbol(0) + IMAGE_TOKENS):
            return "M"
    else:
        if value.equivalent(_SymbolicExtent.symbol(1)):
            return "T"
        if value.equivalent(_SymbolicExtent.symbol(2)):
            return "M"
    raise ImportFailure("rewritten Reshape produced a non-canonical symbolic dimension")


def _set_existing_value_shape(
    model: Any,
    name: str,
    shape: Sequence[_SymbolicExtent],
    *,
    role: str,
) -> None:
    value = next(
        (
            item
            for item in [*model.graph.input, *model.graph.value_info, *model.graph.output]
            if item.name == name
        ),
        None,
    )
    if value is None:
        raise ImportFailure(f"{role} rewritten Reshape output {name!r} has no ValueInfo")
    tensor_shape = value.type.tensor_type.shape
    del tensor_shape.dim[:]
    for extent in shape:
        dimension = tensor_shape.dim.add()
        try:
            canonical = _canonical_symbolic_dimension(extent, role=role)
        except ImportFailure as error:
            raise ImportFailure(
                f"{role} rewritten Reshape output {name!r}: {error}"
            ) from error
        if isinstance(canonical, int):
            dimension.dim_value = canonical
        else:
            dimension.dim_param = canonical


def _symbolic_shapes_equivalent(
    actual: Sequence[_SymbolicExtent | None] | None,
    expected: Sequence[_SymbolicExtent | int],
) -> bool:
    return (
        actual is not None
        and len(actual) == len(expected)
        and all(
            left is not None and left.equivalent(right)
            for left, right in zip(actual, expected)
        )
    )


def _symbolic_shapes_compatible(
    actual: Sequence[_SymbolicExtent | None] | None,
    expected: Sequence[_SymbolicExtent | int],
) -> bool:
    """Accept inference-unknown axes, but reject every contradictory known axis."""

    return (
        actual is not None
        and len(actual) == len(expected)
        and all(
            left is None or left.equivalent(right)
            for left, right in zip(actual, expected)
        )
    )


def _set_ints_attribute(node: Any, name: str, values: Sequence[int], *, onnx: Any) -> None:
    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.extend([onnx.helper.make_attribute(name, list(values))])


def _set_int_attribute(node: Any, name: str, value: int, *, onnx: Any) -> None:
    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.extend([onnx.helper.make_attribute(name, int(value))])


def _set_reshape_target(
    model: Any,
    node: Any,
    values: Sequence[int],
    *,
    name: str,
    onnx: Any,
) -> None:
    import numpy as np

    existing = {
        value.name
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    } | {
        initializer.name for initializer in model.graph.initializer
    } | {
        output
        for graph_node in model.graph.node
        for output in graph_node.output
        if output
    }
    if name in existing:
        raise ImportFailure(f"encoder attention initializer collision {name!r}")
    model.graph.initializer.extend([
        onnx.numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)
    ])
    del node.input[1:]
    node.input.extend([name])
    _set_int_attribute(node, "allowzero", 0, onnx=onnx)


def _replace_with_identity(node: Any, source: str) -> None:
    node.op_type = "Identity"
    node.domain = ""
    del node.input[:]
    node.input.extend([source])
    del node.attribute[:]


def _derived_batch_dimensions(model: Any, *, role: str, onnx: Any) -> list[str]:
    result = []
    batch = _SymbolicExtent.symbol(3, "B")
    for value in _nested_value_infos(model, onnx.ValueInfoProto):
        for dimension in value.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_param"):
                continue
            spelling = str(dimension.dim_param)
            extent = _parse_symbolic_extent(spelling, role=role)
            if extent is None or extent.equivalent(batch):
                continue
            depends_on_batch = any(
                powers[3] != 0
                for polynomial in (extent.numerator, extent.denominator)
                for powers, _coefficient in polynomial
            )
            if depends_on_batch:
                result.append(f"{value.name}:{spelling}")
    return sorted(set(result))


def _rewrite_encoder_attention_batch_layout(model: Any, *, onnx: Any) -> dict[str, int]:
    """Keep producer self-attention batch/head axes independently typed.

    The producer temporarily flattens ``B*H`` around Q/K/V and the mask, and
    ``B*M`` around the output projection. Those products are valid ONNX shape
    programs but cannot be represented as independent bounded Volvox
    dimensions. The attention MatMuls already use rank-four operands, so this
    pass removes only the flatten/unflatten round trips and authors the output
    projection as a rank-three MatMul plus Add. No product extent is renamed
    to an unrelated atomic dimension.
    """

    import numpy as np

    if not _derived_batch_dimensions(model, role="encoder", onnx=onnx):
        return {
            "attention_qkv_layouts_rewritten": 0,
            "attention_mask_layouts_rewritten": 0,
            "attention_key_transposes_rewritten": 0,
            "attention_output_projections_rewritten": 0,
        }

    nodes = list(model.graph.node)
    producer = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    consumers = _node_consumers(nodes)
    tensor_names = {
        name
        for node in nodes
        for name in [*node.input, *node.output]
        if name
    } | {
        value.name
        for value in [
            *model.graph.input,
            *model.graph.value_info,
            *model.graph.output,
            *model.graph.initializer,
        ]
    }
    shapes = {
        value.name: _value_symbolic_shape(value, role="encoder")
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    shape_values: dict[str, Any] = {}
    for name, array in _small_constant_arrays(model, onnx=onnx).items():
        if np.asarray(array).dtype.kind in "iu":
            shape_values[name] = _symbolic_array(array, np)
    batch = _SymbolicExtent.symbol(3, "B")
    memory = _SymbolicExtent.symbol(0) + IMAGE_TOKENS
    heads = _SymbolicExtent.constant(KV_HEADS)
    width = _SymbolicExtent.constant(KV_HEAD_WIDTH)
    feature = _SymbolicExtent.constant(KV_HEADS * KV_HEAD_WIDTH)

    def shape(name: str) -> list[_SymbolicExtent | None] | None:
        return shapes.get(name)

    def set_shape(name: str, extents: Sequence[_SymbolicExtent | int]) -> None:
        symbolic = [
            extent if isinstance(extent, _SymbolicExtent) else _SymbolicExtent.constant(extent)
            for extent in extents
        ]
        _set_existing_value_shape(model, name, symbolic, role="encoder")
        shapes[name] = symbolic

    def shape_value(name: str) -> Any:
        cached = shape_values.get(name)
        if cached is not None:
            return cached
        node_index = producer.get(name)
        if node_index is None:
            raise ImportFailure(f"encoder attention shape tensor {name!r} has no producer")
        node = nodes[node_index]
        if node.op_type == "Shape":
            source_shape = shape(node.input[0])
            if source_shape is None or any(extent is None for extent in source_shape):
                raise ImportFailure("encoder attention Shape input is not symbolically complete")
            start = _onnx_int_attribute(node, "start", 0)
            end = _onnx_int_attribute(node, "end", len(source_shape))
            result = np.asarray(source_shape[start:end], dtype=object)
        elif node.op_type == "Slice":
            data = np.asarray(shape_value(node.input[0]), dtype=object)
            starts = [
                _symbolic_integer(value, "encoder attention Slice start")
                for value in np.asarray(shape_value(node.input[1]), dtype=object).reshape(-1)
            ]
            ends = [
                _symbolic_integer(value, "encoder attention Slice end")
                for value in np.asarray(shape_value(node.input[2]), dtype=object).reshape(-1)
            ]
            axes = (
                [
                    _symbolic_integer(value, "encoder attention Slice axis")
                    for value in np.asarray(
                        shape_value(node.input[3]), dtype=object
                    ).reshape(-1)
                ]
                if len(node.input) > 3 and node.input[3]
                else list(range(len(starts)))
            )
            steps = (
                [
                    _symbolic_integer(value, "encoder attention Slice step")
                    for value in np.asarray(
                        shape_value(node.input[4]), dtype=object
                    ).reshape(-1)
                ]
                if len(node.input) > 4 and node.input[4]
                else [1] * len(starts)
            )
            if not (
                len(starts) == len(ends) == len(axes) == len(steps)
            ):
                raise ImportFailure(
                    "encoder attention Slice parameter lengths differ"
                )
            normalized_axes: list[int] = []
            for axis, step in zip(axes, steps):
                if step == 0:
                    raise ImportFailure("encoder attention Slice step must be nonzero")
                normalized_axis = axis + data.ndim if axis < 0 else axis
                if normalized_axis < 0 or normalized_axis >= data.ndim:
                    raise ImportFailure("encoder attention Slice axis is out of range")
                if normalized_axis in normalized_axes:
                    raise ImportFailure("encoder attention Slice axes must be unique")
                normalized_axes.append(normalized_axis)
            slices = [slice(None)] * data.ndim
            for start, end, axis, step in zip(
                starts, ends, normalized_axes, steps
            ):
                slices[axis] = slice(start, end, step)
            result = data[tuple(slices)]
        elif node.op_type == "Concat":
            result = np.concatenate(
                [np.atleast_1d(shape_value(input_name)) for input_name in node.input],
                axis=_onnx_int_attribute(node, "axis", 0),
            )
        elif node.op_type in {"Squeeze", "Unsqueeze"}:
            source = np.asarray(shape_value(node.input[0]), dtype=object)
            axes = [
                _symbolic_integer(value, f"encoder attention {node.op_type} axis")
                for value in np.asarray(
                    shape_value(node.input[1]), dtype=object
                ).reshape(-1)
            ] if len(node.input) > 1 else []
            if node.op_type == "Squeeze":
                result = (
                    np.squeeze(source)
                    if not axes
                    else np.squeeze(source, axis=tuple(axes))
                )
            else:
                result = source
                for axis in sorted(axes):
                    result = np.expand_dims(result, axis=axis)
        elif node.op_type == "Gather":
            indices = np.asarray([
                _symbolic_integer(value, "encoder attention Gather index")
                for value in np.asarray(
                    shape_value(node.input[1]), dtype=object
                ).reshape(-1)
            ], dtype=np.int64).reshape(
                np.asarray(shape_value(node.input[1])).shape
            )
            result = np.take(
                shape_value(node.input[0]),
                indices,
                axis=_onnx_int_attribute(node, "axis", 0),
            )
        elif node.op_type in {"Add", "Sub", "Mul", "Div"}:
            left = shape_value(node.input[0])
            right = shape_value(node.input[1])
            result = {
                "Add": lambda: left + right,
                "Sub": lambda: left - right,
                "Mul": lambda: left * right,
                "Div": lambda: left / right,
            }[node.op_type]()
        elif node.op_type == "Cast":
            result = shape_value(node.input[0])
        elif node.op_type == "Reshape":
            target = [
                _symbolic_integer(value, "encoder attention shape-value Reshape")
                for value in np.asarray(
                    shape_value(node.input[1]), dtype=object
                ).reshape(-1)
            ]
            result = np.asarray(
                shape_value(node.input[0]), dtype=object
            ).reshape(tuple(target))
        else:
            raise ImportFailure(
                f"encoder attention shape tensor {name!r} uses unsupported "
                f"{node.op_type}"
            )
        shape_values[name] = result
        return result

    def resolved_reshape_shape(
        node: Any,
        input_shape: Sequence[_SymbolicExtent | None],
    ) -> list[_SymbolicExtent]:
        if len(node.input) < 2:
            raise ImportFailure("encoder attention Reshape has no target")
        target = [
            _as_symbolic_extent(value)
            for value in np.asarray(shape_value(node.input[1]), dtype=object).reshape(-1)
        ]
        return _resolved_reshape_target(
            input_shape,
            target,
            allowzero=bool(_onnx_int_attribute(node, "allowzero", 0)),
            label=f"encoder attention {node.name or 'Reshape'}",
        )

    def single_consumer(name: str, op_type: str) -> int:
        uses = consumers.get(name, [])
        if (
            len(uses) != 1
            or uses[0][1] != 0
            or nodes[uses[0][0]].op_type != op_type
        ):
            raise ImportFailure(
                f"encoder attention tensor {name!r} must feed input 0 of one "
                f"{op_type}"
            )
        return uses[0][0]

    qkv_count = 0
    for final_index, final in enumerate(nodes):
        if (
            final.op_type != "Reshape"
            or not final.input
            or not final.output
            or not _symbolic_shapes_equivalent(
                shape(final.output[0]), [batch, heads, memory, width]
            )
        ):
            continue
        transpose_index = producer.get(final.input[0])
        if transpose_index is None or nodes[transpose_index].op_type != "Transpose":
            continue
        transpose = nodes[transpose_index]
        first_index = producer.get(transpose.input[0])
        if first_index is None or nodes[first_index].op_type != "Reshape":
            continue
        first = nodes[first_index]
        if not (
            _symbolic_shapes_equivalent(shape(first.input[0]), [memory, batch, feature])
            and _symbolic_shapes_equivalent(
                shape(first.output[0]), [memory, batch * KV_HEADS, width]
            )
            and _symbolic_shapes_equivalent(
                shape(transpose.output[0]), [batch * KV_HEADS, memory, width]
            )
        ):
            continue
        if not _symbolic_shapes_equivalent(
            resolved_reshape_shape(first, [memory, batch, feature]),
            [memory, batch * KV_HEADS, width],
        ):
            raise ImportFailure("encoder attention Q/K/V flatten target changed")
        transpose_permutation = next(
            (
                list(attribute.ints)
                for attribute in transpose.attribute
                if attribute.name == "perm"
            ),
            None,
        )
        if transpose_permutation != [1, 0, 2]:
            raise ImportFailure("encoder attention Q/K/V transpose changed")
        if not _symbolic_shapes_equivalent(
            resolved_reshape_shape(
                final, [batch * KV_HEADS, memory, width]
            ),
            [batch, heads, memory, width],
        ):
            raise ImportFailure("encoder attention Q/K/V restore target changed")
        if consumers.get(first.output[0], []) != [(transpose_index, 0)]:
            raise ImportFailure("encoder attention Q/K/V head layout is shared")
        if consumers.get(transpose.output[0], []) != [(final_index, 0)]:
            raise ImportFailure("encoder attention Q/K/V batch layout is shared")
        _set_reshape_target(
            model,
            first,
            [0, 0, KV_HEADS, KV_HEAD_WIDTH],
            name=f"__volvox_encoder_attention_qkv_shape_{first_index}",
            onnx=onnx,
        )
        set_shape(first.output[0], [memory, batch, heads, width])
        _set_ints_attribute(transpose, "perm", [1, 2, 0, 3], onnx=onnx)
        set_shape(transpose.output[0], [batch, heads, memory, width])
        _replace_with_identity(final, transpose.output[0])
        set_shape(final.output[0], [batch, heads, memory, width])
        qkv_count += 1

    mask_count = 0
    for flattened in nodes:
        if (
            flattened.op_type != "Reshape"
            or not flattened.input
            or not flattened.output
            or not _symbolic_shapes_equivalent(
                shape(flattened.input[0]), [batch, heads, 1, memory]
            )
            or not _symbolic_shapes_equivalent(
                shape(flattened.output[0]), [batch * KV_HEADS, 1, memory]
            )
        ):
            continue
        restored_index = single_consumer(flattened.output[0], "Reshape")
        restored = nodes[restored_index]
        if not _symbolic_shapes_equivalent(
            resolved_reshape_shape(
                flattened, [batch, heads, 1, memory]
            ),
            [batch * KV_HEADS, 1, memory],
        ):
            raise ImportFailure("encoder attention mask flatten target changed")
        if not _symbolic_shapes_equivalent(
            shape(restored.output[0]), [batch, heads, 1, memory]
        ):
            raise ImportFailure("encoder attention mask restored the wrong batch layout")
        if not _symbolic_shapes_equivalent(
            resolved_reshape_shape(
                restored, [batch * KV_HEADS, 1, memory]
            ),
            [batch, heads, 1, memory],
        ):
            raise ImportFailure("encoder attention mask restore target changed")
        _replace_with_identity(flattened, flattened.input[0])
        set_shape(flattened.output[0], [batch, heads, 1, memory])
        _replace_with_identity(restored, flattened.output[0])
        set_shape(restored.output[0], [batch, heads, 1, memory])
        mask_count += 1

    key_count = 0
    for flattened in nodes:
        if (
            flattened.op_type != "Reshape"
            or not flattened.input
            or not flattened.output
            or not _symbolic_shapes_equivalent(
                shape(flattened.input[0]), [batch, heads, memory, width]
            )
        ):
            continue
        transpose_index = single_consumer(flattened.output[0], "Transpose")
        transpose = nodes[transpose_index]
        restored_index = single_consumer(transpose.output[0], "Reshape")
        restored = nodes[restored_index]
        flattened_shape = resolved_reshape_shape(
            flattened,
            [batch, heads, memory, width],
        )
        if not _symbolic_shapes_equivalent(
            flattened_shape, [batch * KV_HEADS, memory, width]
        ):
            raise ImportFailure("encoder attention key flatten target changed")
        if not _symbolic_shapes_compatible(
            shape(flattened.output[0]), [batch * KV_HEADS, memory, width]
        ):
            raise ImportFailure("encoder attention key flatten metadata changed")
        permutation = next(
            (list(attribute.ints) for attribute in transpose.attribute if attribute.name == "perm"),
            None,
        )
        if permutation != [0, 2, 1]:
            raise ImportFailure("encoder attention key transpose permutation changed")
        transposed_shape = [batch * KV_HEADS, width, memory]
        if not _symbolic_shapes_compatible(
            shape(transpose.output[0]), transposed_shape
        ):
            raise ImportFailure("encoder attention key transpose metadata changed")
        restored_shape = resolved_reshape_shape(restored, transposed_shape)
        if not _symbolic_shapes_equivalent(
            restored_shape, [batch, heads, width, memory]
        ):
            raise ImportFailure("encoder attention key restore target changed")
        if not _symbolic_shapes_compatible(
            shape(restored.output[0]), [batch, heads, width, memory]
        ):
            raise ImportFailure(
                "encoder attention key restore metadata changed for "
                f"{restored.output[0]!r}: {shape(restored.output[0])!r}"
            )
        _replace_with_identity(flattened, flattened.input[0])
        set_shape(flattened.output[0], [batch, heads, memory, width])
        _set_ints_attribute(transpose, "perm", [0, 1, 3, 2], onnx=onnx)
        set_shape(transpose.output[0], [batch, heads, width, memory])
        _replace_with_identity(restored, transpose.output[0])
        set_shape(restored.output[0], [batch, heads, width, memory])
        key_count += 1

    initializers = {value.name: value for value in model.graph.initializer}
    projection_replacements: dict[int, list[Any]] = {}
    projection_count = 0
    for index, gemm in enumerate(nodes):
        if (
            gemm.op_type != "Gemm"
            or len(gemm.input) != 3
            or ".self_attn.out_proj.weight" not in gemm.input[1]
            or not gemm.output
        ):
            continue
        activation_name = gemm.input[0]
        activation_path: list[Any] = []
        while activation_name in producer and nodes[producer[activation_name]].op_type in {
            "QuantizeLinear", "DequantizeLinear",
        }:
            passthrough = nodes[producer[activation_name]]
            activation_path.append(passthrough)
            activation_name = passthrough.input[0]
        reshape_index = producer.get(activation_name)
        if reshape_index is None or nodes[reshape_index].op_type != "Reshape":
            raise ImportFailure("encoder attention projection input is not a Reshape")
        reshape = nodes[reshape_index]
        transpose_index = producer.get(reshape.input[0])
        if transpose_index is None or nodes[transpose_index].op_type != "Transpose":
            raise ImportFailure("encoder attention projection input has no layout Transpose")
        transpose = nodes[transpose_index]
        if not _symbolic_shapes_equivalent(
            shape(transpose.input[0]), [batch, heads, memory, width]
        ) or not _symbolic_shapes_equivalent(
            shape(transpose.output[0]), [memory, batch, heads, width]
        ):
            raise ImportFailure("encoder attention projection has the wrong source layout")
        transpose_permutation = next(
            (
                list(attribute.ints)
                for attribute in transpose.attribute
                if attribute.name == "perm"
            ),
            None,
        )
        if transpose_permutation != [2, 0, 1, 3]:
            raise ImportFailure("encoder attention projection transpose changed")
        if consumers.get(transpose.output[0], []) != [(reshape_index, 0)]:
            raise ImportFailure("encoder attention projection source layout is shared")
        original_projection_shape = resolved_reshape_shape(
            reshape,
            [memory, batch, heads, width],
        )
        if not _symbolic_shapes_equivalent(
            original_projection_shape,
            [batch * memory, feature],
        ):
            raise ImportFailure("encoder attention projection flatten target changed")
        _set_ints_attribute(transpose, "perm", [0, 2, 1, 3], onnx=onnx)
        set_shape(transpose.output[0], [batch, memory, heads, width])
        _set_reshape_target(
            model,
            reshape,
            [0, 0, KV_HEADS * KV_HEAD_WIDTH],
            name=f"__volvox_encoder_attention_projection_shape_{index}",
            onnx=onnx,
        )
        set_shape(reshape.output[0], [batch, memory, feature])
        for passthrough in reversed(activation_path):
            set_shape(passthrough.output[0], [batch, memory, feature])
        current_activation = reshape.output[0]
        for passthrough in reversed(activation_path):
            passthrough_index = producer[passthrough.output[0]]
            if consumers.get(current_activation, []) != [(passthrough_index, 0)]:
                raise ImportFailure(
                    "encoder attention projection quantization layout is shared"
                )
            current_activation = passthrough.output[0]
        if (
            current_activation != gemm.input[0]
            or consumers.get(current_activation, []) != [(index, 0)]
        ):
            raise ImportFailure("encoder attention projection activation is shared")

        weight_name = gemm.input[1]
        attributes = {attribute.name: attribute for attribute in gemm.attribute}
        if set(attributes) != {"transA", "transB", "alpha", "beta"} or (
            attributes["transA"].i != 0
            or attributes["transB"].i != 1
            or attributes["alpha"].f != 1.0
            or attributes["beta"].f != 1.0
        ):
            raise ImportFailure("encoder attention projection Gemm attributes changed")
        weight_dq = (
            nodes[producer[weight_name]]
            if weight_name in producer
            and nodes[producer[weight_name]].op_type == "DequantizeLinear"
            else None
        )
        raw_weight_name = weight_dq.input[0] if weight_dq is not None else weight_name
        raw_weight = initializers.get(raw_weight_name)
        if raw_weight is None or tuple(raw_weight.dims) != (
            KV_HEADS * KV_HEAD_WIDTH,
            KV_HEADS * KV_HEAD_WIDTH,
        ):
            raise ImportFailure("encoder attention projection weight is not immutable 320x320")
        if consumers.get(weight_name, []) != [(index, 1)]:
            raise ImportFailure("encoder attention projection weight is unexpectedly shared")
        if weight_dq is not None:
            weight_dq_index = producer[weight_name]
            if (
                len(weight_dq.input) != 3
                or _onnx_int_attribute(weight_dq, "axis", 1) != 0
                or consumers.get(raw_weight_name, []) != [(weight_dq_index, 0)]
            ):
                raise ImportFailure(
                    "encoder attention projection weight DQ layout is not axis-0 private"
                )
            scale = initializers.get(weight_dq.input[1])
            zero = initializers.get(weight_dq.input[2])
            if scale is None or zero is None:
                raise ImportFailure("encoder attention projection weight DQ is not immutable")
            if (
                consumers.get(weight_dq.input[1], []) != [(weight_dq_index, 1)]
                or consumers.get(weight_dq.input[2], []) != [(weight_dq_index, 2)]
            ):
                raise ImportFailure(
                    "encoder attention projection weight DQ parameters are shared"
                )
            scale_array = onnx.numpy_helper.to_array(scale)
            zero_array = onnx.numpy_helper.to_array(zero)
            if (
                scale_array.dtype != np.dtype(np.float32)
                or scale_array.shape != (KV_HEADS * KV_HEAD_WIDTH,)
                or not np.all(np.isfinite(scale_array))
                or np.any(scale_array <= 0)
                or zero_array.dtype != np.dtype(np.int8)
                or zero_array.shape != (KV_HEADS * KV_HEAD_WIDTH,)
            ):
                raise ImportFailure("encoder attention projection weight DQ descriptor changed")
        transposed_weight = np.ascontiguousarray(
            onnx.numpy_helper.to_array(raw_weight).T
        )
        raw_weight.CopyFrom(
            onnx.numpy_helper.from_array(transposed_weight, raw_weight_name)
        )
        if weight_dq is not None:
            _set_int_attribute(weight_dq, "axis", 1, onnx=onnx)

        original_output = gemm.output[0]
        matrix_output = f"__volvox_encoder_attention_projection_{projection_count}"
        if matrix_output in tensor_names:
            raise ImportFailure(f"encoder attention tensor collision {matrix_output!r}")
        tensor_names.add(matrix_output)
        model.graph.value_info.extend([
            onnx.helper.make_tensor_value_info(
                matrix_output,
                onnx.TensorProto.FLOAT,
                ["B", "M", KV_HEADS * KV_HEAD_WIDTH],
            )
        ])
        shapes[matrix_output] = [batch, memory, feature]
        projection_replacements[index] = [
            onnx.helper.make_node(
                "MatMul",
                [gemm.input[0], weight_name],
                [matrix_output],
                name=f"{gemm.name or f'Gemm_{index}'}__rank3",
            ),
            onnx.helper.make_node(
                "Add",
                [matrix_output, gemm.input[2]],
                [original_output],
                name=gemm.name or f"volvox_encoder_attention_projection_{index}",
            ),
        ]
        set_shape(original_output, [batch, memory, feature])

        current = original_output
        while True:
            uses = consumers.get(current, [])
            if len(uses) != 1:
                raise ImportFailure(
                    f"encoder attention projection tensor {current!r} must have one consumer"
                )
            consumer = nodes[uses[0][0]]
            if consumer.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
                break
            set_shape(consumer.output[0], [batch, memory, feature])
            current = consumer.output[0]
        restored = consumer
        if restored.op_type != "Reshape" or not restored.output:
            raise ImportFailure("encoder attention projection has no restoring Reshape")
        original_restored_shape = resolved_reshape_shape(
            restored,
            [batch * memory, feature],
        )
        if not _symbolic_shapes_equivalent(
            original_restored_shape,
            [memory, batch, feature],
        ):
            raise ImportFailure("encoder attention projection restore target changed")
        _replace_with_identity(restored, current)
        set_shape(restored.output[0], [batch, memory, feature])
        final_index = single_consumer(restored.output[0], "Transpose")
        final = nodes[final_index]
        final_permutation = next(
            (list(attribute.ints) for attribute in final.attribute if attribute.name == "perm"),
            None,
        )
        if final_permutation != [1, 0, 2]:
            raise ImportFailure("encoder attention projection final transpose changed")
        _replace_with_identity(final, restored.output[0])
        set_shape(final.output[0], [batch, memory, feature])
        projection_count += 1

    if projection_replacements:
        rewritten_nodes = []
        for index, node in enumerate(nodes):
            rewritten_nodes.extend(projection_replacements.get(index, [node]))
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)

    expected = {
        "Q/K/V layouts": (qkv_count, 18),
        "mask layouts": (mask_count, 1),
        "key transposes": (key_count, 6),
        "output projections": (projection_count, 6),
    }
    mismatches = [
        f"{label}={actual}, expected {wanted}"
        for label, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ImportFailure(
            "encoder derived batch layout is not the qualified attention pattern: "
            + "; ".join(mismatches)
        )
    remaining = _derived_batch_dimensions(model, role="encoder", onnx=onnx)
    if remaining:
        raise ImportFailure(
            "encoder attention rewrite retained unsafe derived batch extents "
            f"{remaining}"
        )
    return {
        "attention_qkv_layouts_rewritten": qkv_count,
        "attention_mask_layouts_rewritten": mask_count,
        "attention_key_transposes_rewritten": key_count,
        "attention_output_projections_rewritten": projection_count,
    }


def _canonicalize_derived_dimension_expressions(
    model: Any,
    *,
    role: str,
    onnx: Any,
) -> int:
    rewritten = 0
    for value in _nested_value_infos(model, onnx.ValueInfoProto):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_param"):
                continue
            spelling = str(dimension.dim_param)
            if spelling in CANONICAL_DIMENSIONS:
                continue
            symbolic = _parse_symbolic_extent(spelling, role=role)
            if symbolic is None:
                continue
            canonical = _canonical_symbolic_dimension(symbolic, role=role)
            dimension.Clear()
            if isinstance(canonical, int):
                dimension.dim_value = canonical
            else:
                dimension.dim_param = canonical
            rewritten += 1
    return rewritten


def _refresh_inference_unknown_value_infos(model: Any, *, role: str, onnx: Any) -> int:
    """Re-infer only producer-generated ``unk__*`` intermediates after rewrites."""

    def has_unknown(value: Any) -> bool:
        tensor_type = value.type.tensor_type
        return tensor_type.HasField("shape") and any(
            dimension.HasField("dim_param")
            and str(dimension.dim_param).startswith("unk__")
            for dimension in tensor_type.shape.dim
        )

    public_unknown = [
        value.name
        for value in [*model.graph.input, *model.graph.output]
        if has_unknown(value)
    ]
    if public_unknown:
        raise ImportFailure(
            f"{role} public tensors retained inference-only symbols {public_unknown}"
        )
    unknown_names = {
        value.name for value in model.graph.value_info if has_unknown(value)
    }
    if not unknown_names:
        return 0
    retained = [
        value for value in model.graph.value_info if value.name not in unknown_names
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    inferred = onnx.shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )
    inferred_by_name = {
        value.name: value for value in inferred.graph.value_info
    }
    missing = sorted(unknown_names - set(inferred_by_name))
    if missing:
        raise ImportFailure(
            f"{role} could not re-infer rewritten intermediate shapes {missing}"
        )
    refreshed = [inferred_by_name[name] for name in sorted(unknown_names)]
    still_unknown = sorted(
        value.name for value in refreshed if has_unknown(value)
    )
    if still_unknown:
        raise ImportFailure(
            f"{role} retained unresolved rewritten shapes {still_unknown}"
        )
    model.graph.value_info.extend(refreshed)
    return len(refreshed)


def _rewrite_dynamic_reshape_targets(model: Any, *, role: str, onnx: Any) -> dict[str, Any]:
    """Replace runtime shape-value programs with exact ONNX Reshape constants."""

    import numpy as np

    nodes = list(model.graph.node)
    shapes: dict[str, list[_SymbolicExtent | None] | None] = {
        value.name: _value_symbolic_shape(value, role=role)
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    values: dict[str, Any] = {}
    dynamic_values: set[str] = set()
    shape_program_outputs: set[str] = set()
    initializer_names = {value.name for value in model.graph.initializer}
    for initializer in model.graph.initializer:
        array = onnx.numpy_helper.to_array(initializer)
        if array.dtype.kind in "iu" and array.size <= 64:
            values[initializer.name] = _symbolic_array(array, np)

    new_initializers = []
    rewritten = 0
    for index, node in enumerate(nodes):
        op = node.op_type
        produced_symbolic = False
        if op == "Constant" and node.output:
            attribute = next(
                (item for item in node.attribute if item.name == "value"),
                None,
            )
            if attribute is not None:
                array = onnx.numpy_helper.to_array(attribute.t)
                if array.dtype.kind in "iu" and array.size <= 64:
                    values[node.output[0]] = _symbolic_array(array, np)
        elif op == "Shape" and node.input and node.output:
            input_shape = shapes.get(node.input[0])
            if input_shape is None or any(value is None for value in input_shape):
                raise ImportFailure(
                    f"{role} {node.name or f'Shape[{index}]'} input shape is not symbolic"
                )
            start = _onnx_int_attribute(node, "start", 0)
            end = _onnx_int_attribute(node, "end", len(input_shape))
            values[node.output[0]] = np.asarray(input_shape[start:end], dtype=object)
            produced_symbolic = True
        elif op in {"Squeeze", "Unsqueeze"} and node.input[0] in values:
            array = values[node.input[0]]
            axes_value = values.get(node.input[1]) if len(node.input) > 1 else None
            axes = (
                [_symbolic_integer(value, f"{role} {op} axis") for value in np.asarray(
                    axes_value,
                    dtype=object,
                ).reshape(-1)]
                if axes_value is not None
                else []
            )
            if op == "Squeeze":
                values[node.output[0]] = (
                    np.squeeze(array)
                    if not axes
                    else np.squeeze(array, axis=tuple(axes))
                )
            else:
                result = array
                for axis in sorted(axes):
                    result = np.expand_dims(result, axis=axis)
                values[node.output[0]] = result
            produced_symbolic = node.input[0] in dynamic_values
        elif op == "Gather" and all(name in values for name in node.input[:2]):
            indices = np.asarray([
                _symbolic_integer(value, f"{role} Gather index")
                for value in np.asarray(values[node.input[1]], dtype=object).reshape(-1)
            ], dtype=np.int64).reshape(np.asarray(values[node.input[1]]).shape)
            values[node.output[0]] = np.take(
                values[node.input[0]],
                indices,
                axis=_onnx_int_attribute(node, "axis", 0),
            )
            produced_symbolic = any(name in dynamic_values for name in node.input[:2])
        elif op == "Slice" and all(name in values for name in node.input[1:3]) and node.input[0] in values:
            starts = [
                _symbolic_integer(value, f"{role} Slice start")
                for value in np.asarray(values[node.input[1]], dtype=object).reshape(-1)
            ]
            ends = [
                _symbolic_integer(value, f"{role} Slice end")
                for value in np.asarray(values[node.input[2]], dtype=object).reshape(-1)
            ]
            axes = (
                [
                    _symbolic_integer(value, f"{role} Slice axis")
                    for value in np.asarray(values[node.input[3]], dtype=object).reshape(-1)
                ]
                if len(node.input) > 3 and node.input[3] in values
                else list(range(len(starts)))
            )
            steps = (
                [
                    _symbolic_integer(value, f"{role} Slice step")
                    for value in np.asarray(values[node.input[4]], dtype=object).reshape(-1)
                ]
                if len(node.input) > 4 and node.input[4] in values
                else [1] * len(starts)
            )
            slices = [slice(None)] * np.asarray(values[node.input[0]]).ndim
            for start, end, axis, step in zip(starts, ends, axes, steps):
                slices[axis] = slice(start, end, step)
            values[node.output[0]] = values[node.input[0]][tuple(slices)]
            produced_symbolic = any(name in dynamic_values for name in node.input)
        elif op == "Concat" and all(name in values for name in node.input):
            values[node.output[0]] = np.concatenate(
                [np.atleast_1d(values[name]) for name in node.input],
                axis=_onnx_int_attribute(node, "axis", 0),
            )
            produced_symbolic = any(name in dynamic_values for name in node.input)
        elif op in {"Add", "Sub", "Mul", "Div"} and all(
            name in values for name in node.input[:2]
        ):
            left = values[node.input[0]]
            right = values[node.input[1]]
            values[node.output[0]] = {
                "Add": lambda: left + right,
                "Sub": lambda: left - right,
                "Mul": lambda: left * right,
                "Div": lambda: left / right,
            }[op]()
            produced_symbolic = any(name in dynamic_values for name in node.input[:2])
        elif op == "Cast" and node.input[0] in values:
            values[node.output[0]] = values[node.input[0]]
            produced_symbolic = node.input[0] in dynamic_values

        if op == "Reshape" and len(node.input) > 1 and node.input[1] in values:
            raw_target = [
                _as_symbolic_extent(value)
                for value in np.asarray(values[node.input[1]], dtype=object).reshape(-1)
            ]
            input_shape = shapes.get(node.input[0])
            output_shape = _resolved_reshape_target(
                input_shape,
                raw_target,
                allowzero=bool(_onnx_int_attribute(node, "allowzero", 0)),
                label=f"{role} {node.name or f'Reshape[{index}]'}",
            )
            shapes[node.output[0]] = output_shape
            _set_existing_value_shape(
                model,
                node.output[0],
                output_shape,
                role=role,
            )
            if node.input[1] in dynamic_values:
                replacement = _canonical_reshape_initializer(
                    input_shape,
                    output_shape,
                    label=f"{role} {node.name or f'Reshape[{index}]'}",
                )
                name = f"__volvox_{role}_reshape_shape_{index}"
                if name in initializer_names:
                    raise ImportFailure(f"{role} canonical initializer collision {name!r}")
                initializer_names.add(name)
                new_initializers.append(
                    onnx.numpy_helper.from_array(np.asarray(replacement, dtype=np.int64), name)
                )
                node.input[1] = name
                allowzero = next(
                    (item for item in node.attribute if item.name == "allowzero"),
                    None,
                )
                if allowzero is None:
                    node.attribute.extend([onnx.helper.make_attribute("allowzero", 0)])
                else:
                    allowzero.i = 0
                rewritten += 1
            if node.input[0] in values:
                target = [
                    _symbolic_integer(value, f"{role} shape-value Reshape target")
                    for value in raw_target
                ]
                values[node.output[0]] = np.asarray(
                    values[node.input[0]],
                    dtype=object,
                ).reshape(tuple(target))
                produced_symbolic = node.input[0] in dynamic_values
        elif op == "Transpose" and node.input and shapes.get(node.input[0]) is not None:
            input_shape = shapes[node.input[0]]
            permutation = [
                int(value)
                for value in next(
                    (
                        attribute.ints
                        for attribute in node.attribute
                        if attribute.name == "perm"
                    ),
                    reversed(range(len(input_shape))),
                )
            ]
            shapes[node.output[0]] = [input_shape[axis] for axis in permutation]
        elif op in {"Identity", "Cast"} and node.input:
            shapes[node.output[0]] = shapes.get(node.input[0])
        elif op == "Expand" and len(node.input) > 1 and node.input[1] in values:
            target_shape = [
                _as_symbolic_extent(value)
                for value in np.asarray(values[node.input[1]], dtype=object).reshape(-1)
            ]
            shapes[node.output[0]] = _symbolic_broadcast_shape(
                shapes.get(node.input[0]),
                target_shape,
                label=f"{role} {node.name or f'Expand[{index}]'}",
            )

        if produced_symbolic:
            dynamic_values.update(node.output)
            shape_program_outputs.update(node.output)

    if new_initializers:
        model.graph.initializer.extend(new_initializers)
    return {
        "reshape_targets_rewritten": rewritten,
        "values": values,
        "dynamic_values": dynamic_values,
        "shape_program_outputs": shape_program_outputs,
    }


def _small_constant_arrays(model: Any, *, onnx: Any) -> dict[str, Any]:
    arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
        if math.prod(initializer.dims) <= 512
    }
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        attribute = next(
            (item for item in node.attribute if item.name == "value"),
            None,
        )
        if attribute is None:
            continue
        array = onnx.numpy_helper.to_array(attribute.t)
        if array.size <= 512:
            arrays[node.output[0]] = array
    return arrays


def _add_semantic_input(
    model: Any,
    *,
    name: str,
    dtype: int,
    shape: Sequence[int | str],
    onnx: Any,
) -> None:
    tensor_names = {
        value.name
        for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]
    } | {initializer.name for initializer in model.graph.initializer}
    tensor_names.update(
        output
        for node in model.graph.node
        for output in node.output
        if output
    )
    if name in tensor_names:
        raise ImportFailure(f"canonical semantic input name {name!r} already exists")
    model.graph.input.extend([
        onnx.helper.make_tensor_value_info(name, dtype, list(shape))
    ])


def _constant_vector(
    values: Mapping[str, Any],
    name: str,
    *,
    label: str,
) -> list[int] | None:
    value = values.get(name)
    if value is None:
        return None
    import numpy as np

    return [
        _symbolic_integer(item, label)
        for item in np.asarray(value, dtype=object).reshape(-1)
    ]


def _rewrite_position_slice(
    model: Any,
    *,
    role: str,
    analysis: Mapping[str, Any],
    onnx: Any,
) -> tuple[dict[int, list[Any]], set[int], int]:
    import numpy as np

    position_input, dimension_name = POSITION_INPUTS[role]
    source_ids = "question_ids" if role == "encoder" else "decoder_input_ids"
    position_weight = "model.q_pos" if role == "encoder" else "model.y_pos"
    _add_semantic_input(
        model,
        name=position_input,
        dtype=onnx.TensorProto.INT64,
        shape=["B", dimension_name],
        onnx=onnx,
    )
    initializer = next(
        (value for value in model.graph.initializer if value.name == position_weight),
        None,
    )
    if initializer is None:
        return {}, set(), 0
    weight = onnx.numpy_helper.to_array(initializer)
    if list(weight.shape) != [1, MAX_Q if role == "encoder" else MAX_T, 320]:
        raise ImportFailure(
            f"{role} learned position table {position_weight!r} has the wrong shape"
        )
    values = analysis["values"]
    expected_extent = (
        _SymbolicExtent.symbol(0)
        if role == "encoder"
        else _SymbolicExtent.symbol(1)
    )
    candidates: list[int] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Slice" or not node.input or node.input[0] != position_weight:
            continue
        starts = _constant_vector(values, node.input[1], label=f"{role} position Slice starts")
        axes = (
            _constant_vector(values, node.input[3], label=f"{role} position Slice axes")
            if len(node.input) > 3 and node.input[3]
            else [0]
        )
        steps = (
            _constant_vector(values, node.input[4], label=f"{role} position Slice steps")
            if len(node.input) > 4 and node.input[4]
            else [1]
        )
        end_value = values.get(node.input[2]) if len(node.input) > 2 else None
        ends = (
            list(np.asarray(end_value, dtype=object).reshape(-1))
            if end_value is not None
            else []
        )
        if (
            starts == [0]
            and axes == [1]
            and steps == [1]
            and len(ends) == 1
            and _as_symbolic_extent(ends[0]).equivalent(expected_extent)
            and len(node.output) == 1
        ):
            candidates.append(index)
    if len(candidates) != 1:
        raise ImportFailure(
            f"{role} must contain exactly one learned-position Slice over {dimension_name}"
        )
    index = candidates[0]
    node = model.graph.node[index]
    _set_existing_value_shape(
        model,
        node.output[0],
        [
            _SymbolicExtent.symbol(3, preferred_symbol="B"),
            expected_extent,
            _SymbolicExtent.constant(320),
        ],
        role=role,
    )
    axes_name = f"__volvox_{role}_position_squeeze_axes"
    squeezed_name = f"__volvox_{role}_position_table"
    model.graph.initializer.extend([
        onnx.numpy_helper.from_array(np.asarray([0], dtype=np.int64), axes_name)
    ])
    replacements = {
        index: [
            onnx.helper.make_node(
                "Squeeze",
                [position_weight, axes_name],
                [squeezed_name],
                name=f"volvox_{role}_position_table",
            ),
            onnx.helper.make_node(
                "Gather",
                [squeezed_name, position_input],
                [node.output[0]],
                axis=0,
                name=f"volvox_{role}_active_positions",
            ),
        ]
    }
    return replacements, {index}, 1


def _node_consumers(nodes: Sequence[Any]) -> dict[str, list[tuple[int, int]]]:
    consumers: dict[str, list[tuple[int, int]]] = {}
    for node_index, node in enumerate(nodes):
        for input_index, name in enumerate(node.input):
            if name:
                consumers.setdefault(name, []).append((node_index, input_index))
    return consumers


def _rewire_tensor(nodes: Sequence[Any], source: str, replacement: str) -> None:
    for node in nodes:
        for index, name in enumerate(node.input):
            if name == source:
                node.input[index] = replacement


def _rewrite_mask_shape_programs(
    model: Any,
    *,
    role: str,
    analysis: Mapping[str, Any],
    onnx: Any,
) -> tuple[dict[int, list[Any]], set[int]]:
    import numpy as np

    nodes = list(model.graph.node)
    arrays = _small_constant_arrays(model, onnx=onnx)
    producer = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    consumers = _node_consumers(nodes)
    replacements: dict[int, list[Any]] = {}
    removed: set[int] = set()
    if role == "decoder":
        _add_semantic_input(
            model,
            name=CAUSAL_MASK_INPUT,
            dtype=onnx.TensorProto.FLOAT,
            shape=["T", "T"],
            onnx=onnx,
        )
        trilu_nodes = [
            index for index, node in enumerate(nodes) if node.op_type == "Trilu"
        ]
        if trilu_nodes:
            if len(trilu_nodes) != 1:
                raise ImportFailure("decoder must contain at most one causal Trilu")
            trilu_index = trilu_nodes[0]
            trilu = nodes[trilu_index]
            if (
                _onnx_int_attribute(trilu, "upper", 1) != 1
                or len(trilu.input) < 2
                or trilu.input[1] not in arrays
                or np.asarray(arrays[trilu.input[1]]).shape != ()
                or int(np.asarray(arrays[trilu.input[1]]).item()) != 1
            ):
                raise ImportFailure("decoder causal Trilu must be strict upper-triangular")
            ones_index = producer.get(trilu.input[0])
            if ones_index is None or nodes[ones_index].op_type != "Expand":
                raise ImportFailure("decoder causal Trilu must expand an immutable true scalar")
            ones = nodes[ones_index]
            if (
                ones.input[0] not in arrays
                or np.asarray(arrays[ones.input[0]]).shape != ()
                or bool(np.asarray(arrays[ones.input[0]]).item()) is not True
            ):
                raise ImportFailure("decoder causal Trilu source must be true")
            target = analysis["values"].get(ones.input[1])
            expected_t = _SymbolicExtent.symbol(1)
            if target is None:
                raise ImportFailure("decoder causal mask target is not symbolic")
            target_values = list(np.asarray(target, dtype=object).reshape(-1))
            if len(target_values) != 2 or any(
                not _as_symbolic_extent(value).equivalent(expected_t)
                for value in target_values
            ):
                raise ImportFailure("decoder causal mask target must be exactly [T,T]")
            where_uses = consumers.get(trilu.output[0], [])
            causal_where_uses = [
                (consumer, port)
                for consumer, port in where_uses
                if nodes[consumer].op_type == "Where" and port == 0
            ]
            if (
                len(causal_where_uses) != 1
                or any(nodes[consumer].op_type not in {"Where", "Shape"} for consumer, _ in where_uses)
            ):
                raise ImportFailure("decoder causal Trilu must feed one Where condition")
            where_index = causal_where_uses[0][0]
            where = nodes[where_index]
            if where.op_type != "Where" or len(where.input) != 3:
                raise ImportFailure("decoder causal Trilu consumer must be Where")
            true_value = arrays.get(where.input[1])
            zeros_index = producer.get(where.input[2])
            if (
                true_value is None
                or np.asarray(true_value).shape != ()
                or not np.isneginf(float(np.asarray(true_value).item()))
                or zeros_index is None
                or nodes[zeros_index].op_type != "Expand"
            ):
                raise ImportFailure(
                    "decoder causal Where must select -Infinity or expanded zero"
                )
            zero_source = arrays.get(nodes[zeros_index].input[0])
            if (
                zero_source is None
                or np.asarray(zero_source).shape != ()
                or float(np.asarray(zero_source).item()) != 0.0
            ):
                raise ImportFailure("decoder causal Where false branch must be zero")
            _rewire_tensor(nodes, where.output[0], CAUSAL_MASK_INPUT)
            removed.update({ones_index, trilu_index, zeros_index, where_index})

    consumers = _node_consumers(nodes)
    for index, node in enumerate(nodes):
        if index in removed or node.op_type != "Expand" or len(node.input) < 2:
            continue
        source = arrays.get(node.input[0])
        if source is None or np.asarray(source).shape != ():
            continue
        uses = consumers.get(node.output[0], [])
        if (
            float(np.asarray(source).item()) == 0.0
            and node.input[1] in analysis["dynamic_values"]
            and uses
            and all(nodes[consumer].op_type == "Where" and port == 2 for consumer, port in uses)
        ):
            _rewire_tensor(nodes, node.output[0], node.input[0])
            removed.add(index)

    if role == "encoder":
        for index, node in enumerate(nodes):
            if index in removed or node.op_type != "Expand" or len(node.input) < 2:
                continue
            source = arrays.get(node.input[0])
            target = analysis["values"].get(node.input[1])
            if (
                source is None
                or np.asarray(source).shape != ()
                or bool(np.asarray(source).item()) is not False
                or target is None
            ):
                continue
            target_values = list(np.asarray(target, dtype=object).reshape(-1))
            if (
                len(target_values) == 2
                and target_values[0].equivalent(
                    _SymbolicExtent.symbol(3, "B")
                )
                and target_values[1].equivalent(IMAGE_TOKENS)
            ):
                axes_name = "__volvox_encoder_image_padding_axes"
                ones_name = "__volvox_encoder_image_padding_ones"
                column_name = "__volvox_encoder_family_column"
                float_column_name = "__volvox_encoder_family_column_f32"
                zero_name = "__volvox_encoder_family_zero"
                int_zero_name = "__volvox_encoder_family_zero_i32"
                model.graph.initializer.extend([
                    onnx.numpy_helper.from_array(
                        np.asarray([1], dtype=np.int64), axes_name,
                    ),
                    onnx.numpy_helper.from_array(
                        np.ones((1, IMAGE_TOKENS), dtype=np.int32), ones_name,
                    ),
                ])
                replacements[index] = [
                    onnx.helper.make_node(
                        "Unsqueeze",
                        ["family_ids", axes_name],
                        [column_name],
                        name="volvox_encoder_family_column",
                    ),
                    onnx.helper.make_node(
                        "Cast",
                        [column_name],
                        [float_column_name],
                        to=onnx.TensorProto.FLOAT,
                        name="volvox_encoder_family_column_f32",
                    ),
                    onnx.helper.make_node(
                        "Sub",
                        [float_column_name, float_column_name],
                        [zero_name],
                        name="volvox_encoder_family_zero",
                    ),
                    onnx.helper.make_node(
                        "Cast",
                        [zero_name],
                        [int_zero_name],
                        to=onnx.TensorProto.INT32,
                        name="volvox_encoder_family_zero_i32",
                    ),
                    onnx.helper.make_node(
                        "Equal",
                        [int_zero_name, ones_name],
                        [node.output[0]],
                        name="volvox_encoder_image_padding_mask",
                    ),
                ]
    return replacements, removed


def _apply_node_rewrites(
    model: Any,
    *,
    replacements: Mapping[int, Sequence[Any]],
    removed: set[int],
    shape_program_outputs: set[str],
) -> None:
    source_nodes = list(model.graph.node)
    rewritten: list[Any] = []
    for index, node in enumerate(source_nodes):
        if index in replacements:
            rewritten.extend(replacements[index])
        elif index not in removed:
            rewritten.append(node)

    graph_outputs = {value.name for value in model.graph.output}
    changed = True
    while changed:
        changed = False
        consumed = {
            name
            for node in rewritten
            for name in node.input
            if name
        } | graph_outputs
        kept = []
        for node in rewritten:
            if (
                node.output
                and (
                    all(output in shape_program_outputs for output in node.output)
                    or node.op_type == "Constant"
                )
                and not any(output in consumed for output in node.output)
            ):
                changed = True
                continue
            kept.append(node)
        rewritten = kept
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    if any(node.op_type == "Shape" for node in model.graph.node):
        names = [node.name or "<unnamed>" for node in model.graph.node if node.op_type == "Shape"]
        raise ImportFailure(f"normalized ONNX retained runtime Shape nodes {names}")
    used_initializers = {
        name
        for node in model.graph.node
        for name in node.input
        if name
    } | {value.name for value in model.graph.output}
    retained_initializers = [
        value for value in model.graph.initializer if value.name in used_initializers
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)


def _rewrite_dynamic_authoring_graph(
    model: Any,
    *,
    role: str,
    onnx: Any,
) -> dict[str, int]:
    analysis = _rewrite_dynamic_reshape_targets(model, role=role, onnx=onnx)
    replacements, removed, position_count = _rewrite_position_slice(
        model,
        role=role,
        analysis=analysis,
        onnx=onnx,
    )
    mask_replacements, mask_removed = _rewrite_mask_shape_programs(
        model,
        role=role,
        analysis=analysis,
        onnx=onnx,
    )
    overlap = sorted(set(replacements) & set(mask_replacements))
    if overlap:
        raise ImportFailure(f"{role} dynamic rewrite collision at nodes {overlap}")
    replacements.update(mask_replacements)
    removed.update(mask_removed)
    _apply_node_rewrites(
        model,
        replacements=replacements,
        removed=removed,
        shape_program_outputs=analysis["shape_program_outputs"],
    )
    expression_count = _canonicalize_derived_dimension_expressions(
        model,
        role=role,
        onnx=onnx,
    )
    return {
        "reshape_targets_rewritten": int(analysis["reshape_targets_rewritten"]),
        "position_slice_rewritten": position_count,
        "causal_mask_hoisted": int(role == "decoder" and bool(removed)),
        "derived_dimension_expressions_rewritten": expression_count,
    }


def _normalized_onnx_symbols(
    source: Path,
    destination: Path,
    *,
    role: str,
) -> dict[str, int]:
    """Author a temporary canonical-symbol ONNX input for the generic exporter.

    The producer's encoder publishes the expression ``question_length + 210``.
    It is deliberately replaced with the plain output symbol ``M``
    before export.  That spelling is only an import-format cue: staged package
    qualification separately requires a canonical Concat proof for
    ``M = Q + 210``.
    """

    try:
        import onnx
    except ImportError as error:  # pragma: no cover - production dependency.
        raise ImportFailure(
            "the local ONNX package is required to author canonical shape symbols"
        ) from error
    if role == "encoder":
        replacements = {
            "batch": "B",
            "question_length": "Q",
            "question_length + 210": "M",
        }
        required = frozenset(replacements)
    elif role == "decoder":
        replacements = {
            "batch": "B",
            "target_length": "T",
            "memory_length": "M",
        }
        required = frozenset(replacements)
    else:  # Defensive internal API boundary.
        raise ImportFailure(f"unsupported normalized ONNX role {role!r}")
    try:
        model = onnx.load_model(str(source), load_external_data=False)
    except Exception as error:
        raise ImportFailure(f"{role} canonical ONNX loading failed: {error}") from error

    seen: set[str] = set()
    for value_info in _nested_value_infos(model, onnx.ValueInfoProto):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_param"):
                continue
            source_name = str(dimension.dim_param)
            replacement = replacements.get(source_name)
            if replacement is not None:
                seen.add(source_name)
                dimension.dim_param = replacement
    missing = sorted(required - seen)
    if missing:
        raise ImportFailure(
            f"{role} ONNX is missing canonicalization source symbols {missing}"
        )
    remaining = sorted({
        str(dimension.dim_param)
        for value_info in _nested_value_infos(model, onnx.ValueInfoProto)
        for dimension in value_info.type.tensor_type.shape.dim
        if dimension.HasField("dim_param")
        and str(dimension.dim_param) in replacements
    })
    if remaining:
        raise ImportFailure(
            f"{role} ONNX retained non-canonical shape symbols {remaining}"
        )
    try:
        inferred_source = onnx.shape_inference.infer_shapes(
            model,
            strict_mode=True,
            data_prop=True,
        )
        declared_names = {
            value.name
            for value in [
                *model.graph.input,
                *model.graph.value_info,
                *model.graph.output,
            ]
        }
        model.graph.value_info.extend([
            value
            for value in inferred_source.graph.value_info
            if value.name not in declared_names
        ])
        rewrite_report = _rewrite_dynamic_authoring_graph(
            model,
            role=role,
            onnx=onnx,
        )
        refreshed_unknowns = _refresh_inference_unknown_value_infos(
            model,
            role=role,
            onnx=onnx,
        )
        rewrite_report["inference_unknowns_refreshed"] = refreshed_unknowns
        rewrite_report["derived_dimension_expressions_rewritten"] += (
            _canonicalize_derived_dimension_expressions(
                model,
                role=role,
                onnx=onnx,
            )
        )
        onnx.checker.check_model(model)
        inferred_model = onnx.shape_inference.infer_shapes(
            model,
            strict_mode=True,
            data_prop=True,
        )
        onnx.checker.check_model(inferred_model)
        onnx.save_model(model, str(destination), save_as_external_data=False)
    except Exception as error:
        if isinstance(error, ImportFailure):
            raise
        raise ImportFailure(f"{role} canonical ONNX authoring failed: {error}") from error
    return rewrite_report


def _normalized_kv_onnx_symbols(
    source: Path,
    destination: Path,
    *,
    role: str,
) -> dict[str, int]:
    """Normalize the producer-specific v1 ValueInfo spelling.

    The one-token decoder already carries semantic position and mask inputs
    and therefore needs no graph rewrite.
    Only symbolic metadata is canonicalized; the executable node list and
    initializers remain byte-for-byte semantically equivalent.
    """

    try:
        import onnx
    except ImportError as error:  # pragma: no cover
        raise ImportFailure(
            "the local ONNX package is required to author KV shape symbols"
        ) from error
    if role == "encoder":
        replacements: dict[str, int | str] = {
            "batch": "B",
            "question_length": "Q",
            "question_length + 210": "M",
            "memory_length": "M",
        }
        required = {"batch", "question_length", "question_length + 210"}
    elif role == "decoder":
        replacements = {
            "batch": "B",
            "memory_length": "M",
            "past_length": "P",
            "present_length": "R",
        }
        required = set(replacements)
    else:
        raise ImportFailure(f"unsupported KV normalized ONNX role {role!r}")
    try:
        model = onnx.load_model(str(source), load_external_data=False)
        if role == "decoder":
            model = onnx.shape_inference.infer_shapes(
                model, strict_mode=True, data_prop=True
            )
    except Exception as error:
        raise ImportFailure(f"{role} KV canonical ONNX loading failed: {error}") from error

    seen: set[str] = set()
    rewritten = 0
    for value_info in _nested_value_infos(model, onnx.ValueInfoProto):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_param"):
                continue
            spelling = str(dimension.dim_param)
            replacement = replacements.get(spelling)
            if replacement is None:
                continue
            seen.add(spelling)
            dimension.Clear()
            if isinstance(replacement, int):
                dimension.dim_value = replacement
            else:
                dimension.dim_param = replacement
            rewritten += 1
    missing = sorted(required - seen)
    if missing:
        raise ImportFailure(f"{role} KV ONNX is missing source symbols {missing}")

    if role == "decoder":
        # Torch/ONNX shape inference leaves head width (and, after QDQ
        # insertion, some batch axes) as producer-local ``unk__*`` symbols.
        # The one-token public ABI fixes those extents unambiguously.
        for value_info in _nested_value_infos(model, onnx.ValueInfoProto):
            dimensions = value_info.type.tensor_type.shape.dim
            if len(dimensions) >= 2 and (
                dimensions[0].HasField("dim_param")
                and str(dimensions[0].dim_param).startswith("unk__")
            ):
                dimensions[0].Clear()
                dimensions[0].dim_param = "B"
                rewritten += 1
            if len(dimensions) == 4 and (
                dimensions[3].HasField("dim_param")
                and str(dimensions[3].dim_param).startswith("unk__")
            ):
                dimensions[3].Clear()
                dimensions[3].dim_value = KV_HEAD_WIDTH
                rewritten += 1
            if (
                len(dimensions) == 4
                and value_info.name.startswith("present_v_")
                and dimensions[2].HasField("dim_param")
                and str(dimensions[2].dim_param).startswith("unk__")
            ):
                dimensions[2].Clear()
                dimensions[2].dim_param = "R"
                rewritten += 1

    if role == "encoder":
        outputs = {value.name: value for value in model.graph.output}
        for name in KV_CROSS_NAMES:
            value = outputs.get(name)
            if value is None:
                raise ImportFailure(f"encoder KV ONNX is missing {name}")
            dimensions = value.type.tensor_type.shape.dim
            if len(dimensions) != 4:
                raise ImportFailure(f"encoder KV output {name} must have rank four")
            dimensions[0].Clear()
            dimensions[0].dim_param = "B"
            dimensions[1].Clear()
            dimensions[1].dim_value = KV_HEADS
            dimensions[2].Clear()
            dimensions[2].dim_param = "M"
            dimensions[3].Clear()
            dimensions[3].dim_value = KV_HEAD_WIDTH
            rewritten += 4

    semantic_rewrite: dict[str, int] = {}
    executable_nodes_rewritten = 0
    if role == "encoder":
        # The producer derives attention reshape targets and the learned
        # question-position slice from runtime Shape tensors.  Those programs
        # mix data and shape consumers, so make the position sequence semantic
        # and rewrite only the proven shape-only subgraphs.
        inferred_source = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        declared_names = {
            value.name
            for value in [
                *model.graph.input, *model.graph.value_info, *model.graph.output
            ]
        }
        model.graph.value_info.extend([
            value for value in inferred_source.graph.value_info
            if value.name not in declared_names
        ])
        nodes_before = len(model.graph.node)
        semantic_rewrite.update(
            _rewrite_encoder_attention_batch_layout(model, onnx=onnx)
        )
        semantic_rewrite = _rewrite_dynamic_authoring_graph(
            model, role=role, onnx=onnx
        ) | semantic_rewrite
        executable_nodes_rewritten = nodes_before - len(model.graph.node)
        semantic_rewrite["inference_unknowns_refreshed"] = (
            _refresh_inference_unknown_value_infos(model, role=role, onnx=onnx)
        )
        semantic_rewrite["derived_dimension_expressions_rewritten"] += (
            _canonicalize_derived_dimension_expressions(
                model, role=role, onnx=onnx
            )
        )
    public_expected: dict[str, list[int | str]]
    if role == "encoder":
        public_expected = {
            "image": ["B", 1, IMAGE_HEIGHT, IMAGE_WIDTH],
            "question_ids": ["B", "Q"],
            "family_ids": ["B"],
            "question_position_ids": ["B", "Q"],
            "memory": ["B", "M", 320],
            "memory_padding_mask": ["B", "M"],
            "router_logits": ["B", len(FAMILY_ORDER)],
            "selected_family_ids": ["B"],
            **{
                name: ["B", KV_HEADS, "M", KV_HEAD_WIDTH]
                for name in KV_CROSS_NAMES
            },
        }
    else:
        public_expected = {
            "decoder_input_ids": ["B", 1],
            "position_ids": ["B"],
            "family_ids": ["B"],
            "memory_padding_mask": ["B", "M"],
            "past_padding_mask": ["B", "P"],
            **{
                name: ["B", KV_HEADS, "M", KV_HEAD_WIDTH]
                for name in KV_CROSS_NAMES
            },
            **{
                name: ["B", KV_HEADS, "P", KV_HEAD_WIDTH]
                for name in KV_PAST_NAMES
            },
            "logits": ["B", 1, BPE_VOCAB_SIZE],
            "present_padding_mask": ["B", "R"],
            **{
                name: ["B", KV_HEADS, "R", KV_HEAD_WIDTH]
                for name in KV_PRESENT_NAMES
            },
        }
    public = {
        value.name: value for value in [*model.graph.input, *model.graph.output]
    }
    for name, expected_shape in public_expected.items():
        value = public.get(name)
        if value is None or _tensor_signature(value)[1] != expected_shape:
            raise ImportFailure(
                f"{role} KV canonical public tensor {name!r} has the wrong shape"
            )
    try:
        onnx.checker.check_model(model)
        onnx.save_model(model, str(destination), save_as_external_data=False)
    except Exception as error:
        raise ImportFailure(f"{role} KV canonical ONNX authoring failed: {error}") from error
    return {
        "value_info_dimensions_rewritten": rewritten,
        "executable_nodes_rewritten": executable_nodes_rewritten,
        **semantic_rewrite,
    }


def _verify_dynamic_authoring_parity(
    source: Path,
    normalized: Path,
    *,
    role: str,
    max_batch_size: int = 1,
) -> dict[str, Any]:
    """Compare the semantic-input rewrite with the producer at active extents."""

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - production dependency.
        raise ImportFailure(
            "NumPy and ONNX Runtime are required for dynamic authoring parity"
        ) from error

    absolute_tolerance = 4.0e-6
    relative_tolerance = 1.0e-5

    cases = [
        ("short", SHAPE_PROFILES["short"], 1),
        ("representative", SHAPE_PROFILES["representative"], 1),
        ("maximum", SHAPE_PROFILES["maximum"], 1),
        ("short_after_maximum", SHAPE_PROFILES["short"], 1),
        ("representative_after_short", SHAPE_PROFILES["representative"], 1),
    ]
    if max_batch_size > 1:
        cases.append((
            "batch_max_distinct_lanes",
            SHAPE_PROFILES["short"],
            max_batch_size,
        ))
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        original_session = ort.InferenceSession(
            str(source),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        normalized_session = ort.InferenceSession(
            str(normalized),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        raise ImportFailure(f"{role} dynamic authoring parity setup failed: {error}") from error

    maximum_difference = 0.0
    maximum_relative_difference = 0.0
    reports = []
    for case_name, profile, batch_size in cases:
        if role == "encoder":
            active = profile["Q"]
            question_ids = (
                np.arange(batch_size * active, dtype=np.int64).reshape(
                    batch_size, active
                ) % 64
            ) + 4
            images = np.zeros(
                (batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
                dtype=np.float32,
            )
            family_ids = np.asarray([-1], dtype=np.int64)
            if batch_size > 1:
                images += np.arange(batch_size, dtype=np.float32).reshape(
                    batch_size, 1, 1, 1
                ) * np.float32(0.03125)
                family_ids = np.arange(batch_size, dtype=np.int64) % len(FAMILY_ORDER)
            original_inputs = {
                "image": images,
                "question_ids": question_ids,
                "family_ids": family_ids,
            }
            normalized_inputs = {
                **original_inputs,
                POSITION_INPUTS[role][0]: np.broadcast_to(
                    np.arange(active, dtype=np.int64),
                    (batch_size, active),
                ).copy(),
            }
        else:
            active = profile["T"]
            memory_length = profile["M"]
            causal_mask = np.zeros((active, active), dtype=np.float32)
            causal_mask[np.triu_indices(active, 1)] = -np.inf
            original_inputs = {
                "decoder_input_ids": (
                    np.arange(
                        batch_size * active, dtype=np.int64
                    ).reshape(batch_size, active) % 64
                ) + 1,
                "memory": np.zeros(
                    (batch_size, memory_length, 320), dtype=np.float32
                ),
                "memory_padding_mask": np.zeros(
                    (batch_size, memory_length),
                    dtype=np.bool_,
                ),
                "family_ids": (
                    np.arange(batch_size, dtype=np.int64) % len(FAMILY_ORDER)
                ),
            }
            if batch_size > 1:
                original_inputs["memory"] += np.arange(
                    batch_size, dtype=np.float32
                ).reshape(batch_size, 1, 1) * np.float32(0.03125)
            normalized_inputs = {
                **original_inputs,
                POSITION_INPUTS[role][0]: np.broadcast_to(
                    np.arange(active, dtype=np.int64),
                    (batch_size, active),
                ).copy(),
                CAUSAL_MASK_INPUT: causal_mask,
            }
        try:
            original_outputs = original_session.run(None, original_inputs)
            normalized_outputs = normalized_session.run(None, normalized_inputs)
        except Exception as error:
            raise ImportFailure(
                f"{role} dynamic authoring parity case {case_name!r} failed: {error}"
            ) from error
        if len(original_outputs) != len(normalized_outputs):
            raise ImportFailure(f"{role} dynamic authoring parity output count changed")
        case_difference = 0.0
        case_relative_difference = 0.0
        for output_index, (original, authored) in enumerate(
            zip(original_outputs, normalized_outputs)
        ):
            if original.shape != authored.shape or original.dtype != authored.dtype:
                raise ImportFailure(
                    f"{role} dynamic authoring parity output {output_index} ABI changed"
                )
            if original.ndim == 0 or original.shape[0] != batch_size:
                raise ImportFailure(
                    f"{role} dynamic authoring parity output {output_index} "
                    "did not preserve the requested leading batch axis"
                )
            if original.dtype.kind == "f":
                if not (
                    np.all(np.isfinite(original))
                    and np.all(np.isfinite(authored))
                ):
                    raise ImportFailure(
                        f"{role} dynamic authoring parity case {case_name!r} "
                        f"produced non-finite output {output_index}"
                    )
                absolute = np.abs(original - authored)
                difference = float(np.max(absolute, initial=0.0))
                relative_difference = float(np.max(
                    absolute / np.maximum(
                        np.abs(original),
                        np.asarray(1.0e-6, dtype=original.dtype),
                    ),
                    initial=0.0,
                ))
                if not np.allclose(
                    original,
                    authored,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                ):
                    raise ImportFailure(
                        f"{role} dynamic authoring parity case {case_name!r} "
                        f"changed output {output_index} by abs={difference}, "
                        f"rel={relative_difference}"
                    )
                case_difference = max(case_difference, difference)
                case_relative_difference = max(
                    case_relative_difference, relative_difference
                )
            elif not np.array_equal(original, authored):
                raise ImportFailure(
                    f"{role} dynamic authoring parity case {case_name!r} "
                    f"changed output {output_index}"
                )
        if role == "decoder":
            original_greedy = np.argmax(original_outputs[0], axis=-1)
            normalized_greedy = np.argmax(normalized_outputs[0], axis=-1)
            if not np.array_equal(original_greedy, normalized_greedy):
                raise ImportFailure(
                    f"decoder dynamic authoring parity case {case_name!r} changed greedy IDs"
                )
        maximum_difference = max(maximum_difference, case_difference)
        maximum_relative_difference = max(
            maximum_relative_difference, case_relative_difference
        )
        reports.append({
            "case": case_name,
            "B": batch_size,
            "Q": profile["Q"],
            "T": profile["T"],
            "M": profile["M"],
            "max_abs_difference": case_difference,
            "max_relative_difference": case_relative_difference,
        })
    return {
        "status": "passed",
        "provider": "CPUExecutionProvider",
        "cases": reports,
        "maximum_absolute_difference": maximum_difference,
        "maximum_relative_difference": maximum_relative_difference,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "greedy_argmax": "matched" if role == "decoder" else "not_applicable",
    }


def _nested_value_infos(message: Any, value_info_proto_type: type[Any]):
    """Yield public/intermediate ValueInfoProto records, including subgraphs."""

    if isinstance(message, value_info_proto_type):
        yield message
        return
    list_fields = getattr(message, "ListFields", None)
    if not callable(list_fields):
        return
    for field, value in list_fields():
        if field.cpp_type != field.CPPTYPE_MESSAGE:
            continue
        if field.is_repeated:
            for item in value:
                yield from _nested_value_infos(item, value_info_proto_type)
        else:
            yield from _nested_value_infos(value, value_info_proto_type)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ImportFailure(
            f"{label} has the wrong key set; missing={missing}, unsupported={unknown}"
        )


def _validate_variant_file_reference(
    variant: Mapping[str, Any],
    files: Mapping[str, Any],
    *,
    role: str,
    file_key: str,
    label: str,
) -> None:
    if variant.get(role) != files.get(file_key):
        raise ImportFailure(f"{label}.{role} does not match files.{file_key}")
    for suffix in ("bytes", "sha256"):
        if variant.get(f"{role}_{suffix}") != files.get(f"{file_key}_{suffix}"):
            raise ImportFailure(
                f"{label}.{role}_{suffix} does not match files.{file_key}_{suffix}"
            )


def _validate_optimized_graph_summary(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ImportFailure(f"{label} must be an object")
    _require_exact_keys(value, frozenset({"nodes", "initializers"}), label)
    nodes = value["nodes"]
    initializers = value["initializers"]
    if not isinstance(nodes, Mapping) or not isinstance(initializers, Mapping):
        raise ImportFailure(f"{label} node/initializer summaries must be objects")
    for summary_label, summary in (("nodes", nodes), ("initializers", initializers)):
        if not summary or any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for name, count in summary.items()
        ):
            raise ImportFailure(f"{label}.{summary_label} must contain positive counts")
    if not {"QuantizeLinear", "DequantizeLinear"}.issubset(nodes):
        raise ImportFailure(f"{label} does not report preserved QDQ boundaries")
    if not {"FLOAT", "UINT8", "INT8", "INT32"}.issubset(initializers):
        raise ImportFailure(f"{label} does not report U8/S8/I32 initializer storage")


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImportFailure(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ImportFailure(f"{label} is outside its valid range")
    return result


def _validate_heldout_summary(value: Any, files: Mapping[str, Any]) -> None:
    label = "source manifest variants.int8_w8a8.validation.heldout"
    if value == {"status": "not_included"}:
        return
    if not isinstance(value, Mapping):
        raise ImportFailure(f"{label} must be an object")
    _require_exact_keys(
        value,
        frozenset({
            "status",
            "evaluation_mode",
            "dataset",
            "records_total",
            "batch_size",
            "routing",
            "runtime",
            "comparison",
            "artifacts",
            "summary",
            "timing",
            "timing_note",
        }),
        label,
    )
    if (
        value.get("status") != "passed"
        or value.get("evaluation_mode") != "final_split_onnx_files"
        or value.get("routing") != "learned_router"
        or not isinstance(value.get("dataset"), str)
        or not value["dataset"]
        or value.get("timing_note")
        != "Static W8A8 speed depends on QDQ fusion and the selected execution provider."
    ):
        raise ImportFailure(f"{label} provenance is invalid")
    records_total = _positive_int(value.get("records_total"), f"{label}.records_total")
    _positive_int(value.get("batch_size"), f"{label}.batch_size")

    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ImportFailure(f"{label}.runtime must be an object")
    _require_exact_keys(
        runtime,
        frozenset({"library", "provider", "version"}),
        f"{label}.runtime",
    )
    if (
        runtime.get("library") != "onnxruntime"
        or runtime.get("provider") not in {
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        }
        or not isinstance(runtime.get("version"), str)
        or not runtime["version"]
    ):
        raise ImportFailure(f"{label}.runtime is invalid")

    comparison = value.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ImportFailure(f"{label}.comparison must be an object")
    _require_exact_keys(
        comparison,
        frozenset({
            "n",
            "changed_predictions",
            "answer_agreement",
            "prediction_exact_agreement",
            "selected_family_agreement",
        }),
        f"{label}.comparison",
    )
    if comparison.get("n") != records_total:
        raise ImportFailure(f"{label}.comparison sample count is inconsistent")
    changed_predictions = _nonnegative_int(
        comparison.get("changed_predictions"),
        f"{label}.comparison.changed_predictions",
    )
    if changed_predictions > records_total:
        raise ImportFailure(f"{label}.comparison changed count is inconsistent")
    for key in (
        "answer_agreement",
        "prediction_exact_agreement",
        "selected_family_agreement",
    ):
        score = _finite_number(comparison.get(key), f"{label}.comparison.{key}")
        if score < 0.0 or score > 1.0:
            raise ImportFailure(f"{label}.comparison.{key} must be a probability")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ImportFailure(f"{label}.artifacts must be an object")
    _require_exact_keys(
        artifacts,
        frozenset({"fp32", "int8"}),
        f"{label}.artifacts",
    )
    artifact_keys = {
        "fp32": {"encoder": "encoder", "decoder": "decoder"},
        "int8": {
            "encoder": "encoder_int8_w8a8",
            "decoder": "decoder_int8_w8a8",
        },
    }
    for artifact_variant, roles in artifact_keys.items():
        variant_assets = artifacts.get(artifact_variant)
        if not isinstance(variant_assets, Mapping):
            raise ImportFailure(f"{label}.artifacts.{artifact_variant} must be an object")
        _require_exact_keys(
            variant_assets,
            frozenset({"encoder", "decoder"}),
            f"{label}.artifacts.{artifact_variant}",
        )
        for role, file_key in roles.items():
            record = variant_assets.get(role)
            if not isinstance(record, Mapping):
                raise ImportFailure(
                    f"{label}.artifacts.{artifact_variant}.{role} must be an object"
                )
            _require_exact_keys(
                record,
                frozenset({"filename", "bytes", "sha256"}),
                f"{label}.artifacts.{artifact_variant}.{role}",
            )
            if record != {
                "filename": files[file_key],
                "bytes": files[f"{file_key}_bytes"],
                "sha256": files[f"{file_key}_sha256"],
            }:
                raise ImportFailure(
                    f"{label}.artifacts.{artifact_variant}.{role} is inconsistent"
                )

    summary = value.get("summary")
    if not isinstance(summary, Mapping):
        raise ImportFailure(f"{label}.summary must be an object")
    _require_exact_keys(summary, frozenset({"fp32", "int8"}), f"{label}.summary")
    metric_keys = frozenset({
        "answer_exact",
        "n",
        "recomputed_answer_exact",
        "recomputed_n",
        "target_exact",
    })
    for summary_variant in ("fp32", "int8"):
        variant_summary = summary.get(summary_variant)
        if not isinstance(variant_summary, Mapping):
            raise ImportFailure(f"{label}.summary.{summary_variant} must be an object")
        _require_exact_keys(
            variant_summary,
            frozenset({"phone_number", "address", "overall"}),
            f"{label}.summary.{summary_variant}",
        )
        family_total = 0
        for scope in ("phone_number", "address", "overall"):
            metrics = variant_summary.get(scope)
            if not isinstance(metrics, Mapping):
                raise ImportFailure(
                    f"{label}.summary.{summary_variant}.{scope} must be an object"
                )
            _require_exact_keys(
                metrics,
                metric_keys,
                f"{label}.summary.{summary_variant}.{scope}",
            )
            count = _positive_int(
                metrics.get("n"),
                f"{label}.summary.{summary_variant}.{scope}.n",
            )
            if metrics.get("recomputed_n") != count:
                raise ImportFailure(
                    f"{label}.summary.{summary_variant}.{scope} recomputed count differs"
                )
            if scope == "overall" and count != records_total:
                raise ImportFailure(
                    f"{label}.summary.{summary_variant}.overall count is inconsistent"
                )
            if scope != "overall":
                family_total += count
            for metric in ("answer_exact", "recomputed_answer_exact", "target_exact"):
                score = _finite_number(
                    metrics.get(metric),
                    f"{label}.summary.{summary_variant}.{scope}.{metric}",
                )
                if score < 0.0 or score > 1.0:
                    raise ImportFailure(
                        f"{label}.summary.{summary_variant}.{scope}.{metric} "
                        "must be a probability"
                    )
        if family_total != records_total:
            raise ImportFailure(
                f"{label}.summary.{summary_variant} family counts are inconsistent"
            )

    timing = value.get("timing")
    if not isinstance(timing, Mapping):
        raise ImportFailure(f"{label}.timing must be an object")
    _require_exact_keys(timing, frozenset({"fp32", "int8"}), f"{label}.timing")
    timing_file_keys = {
        "fp32": ("encoder", "decoder"),
        "int8": ("encoder_int8_w8a8", "decoder_int8_w8a8"),
    }
    for timing_variant, (encoder_key, decoder_key) in timing_file_keys.items():
        entry = timing.get(timing_variant)
        if not isinstance(entry, Mapping):
            raise ImportFailure(f"{label}.timing.{timing_variant} must be an object")
        _require_exact_keys(
            entry,
            frozenset({
                "encoder",
                "decoder",
                "encoder_seconds",
                "decoder_seconds",
                "decoder_calls",
                "total_seconds",
                "providers",
            }),
            f"{label}.timing.{timing_variant}",
        )
        if (
            entry.get("encoder") != files[encoder_key]
            or entry.get("decoder") != files[decoder_key]
            or not isinstance(entry.get("providers"), list)
            or not entry["providers"]
            or any(
                provider not in {
                    "CPUExecutionProvider",
                    "CUDAExecutionProvider",
                }
                for provider in entry["providers"]
            )
        ):
            raise ImportFailure(f"{label}.timing.{timing_variant} is invalid")
        _positive_int(
            entry.get("decoder_calls"),
            f"{label}.timing.{timing_variant}.decoder_calls",
        )
        encoder_seconds = _finite_number(
            entry.get("encoder_seconds"),
            f"{label}.timing.{timing_variant}.encoder_seconds",
            minimum=0.0,
        )
        decoder_seconds = _finite_number(
            entry.get("decoder_seconds"),
            f"{label}.timing.{timing_variant}.decoder_seconds",
            minimum=0.0,
        )
        total_seconds = _finite_number(
            entry.get("total_seconds"),
            f"{label}.timing.{timing_variant}.total_seconds",
            minimum=0.0,
        )
        if not math.isclose(
            total_seconds,
            encoder_seconds + decoder_seconds,
            rel_tol=1.0e-9,
            abs_tol=1.0e-6,
        ):
            raise ImportFailure(
                f"{label}.timing.{timing_variant} total is inconsistent"
            )


def _validate_producer_variants(
    manifest: Mapping[str, Any],
    files: Mapping[str, Any],
    *,
    has_int8_w8a8: bool,
    maximum_prefix_length: int,
) -> None:
    variants = manifest.get("variants")
    if variants is None and not has_int8_w8a8:
        return
    if not isinstance(variants, Mapping):
        raise ImportFailure("source manifest variants must be an object")
    expected_variant_keys = {"fp32"}
    if has_int8_w8a8:
        expected_variant_keys.add("int8_w8a8")
    _require_exact_keys(
        variants,
        frozenset(expected_variant_keys),
        "source manifest variants",
    )

    fp32 = variants.get("fp32")
    if not isinstance(fp32, Mapping):
        raise ImportFailure("source manifest variants.fp32 must be an object")
    _require_exact_keys(
        fp32,
        frozenset({
            "encoder",
            "encoder_bytes",
            "encoder_sha256",
            "decoder",
            "decoder_bytes",
            "decoder_sha256",
            "activation_dtype",
            "compute_dtype",
            "weight_dtype",
        }),
        "source manifest variants.fp32",
    )
    if {
        "activation_dtype": fp32.get("activation_dtype"),
        "compute_dtype": fp32.get("compute_dtype"),
        "weight_dtype": fp32.get("weight_dtype"),
    } != {
        "activation_dtype": "float32",
        "compute_dtype": "float32",
        "weight_dtype": "float32",
    }:
        raise ImportFailure("source manifest variants.fp32 has the wrong dtype contract")
    for role in ("encoder", "decoder"):
        _validate_variant_file_reference(
            fp32,
            files,
            role=role,
            file_key=role,
            label="source manifest variants.fp32",
        )

    if not has_int8_w8a8:
        return
    int8_variant = variants.get("int8_w8a8")
    if not isinstance(int8_variant, Mapping):
        raise ImportFailure("source manifest variants.int8_w8a8 must be an object")
    _require_exact_keys(
        int8_variant,
        frozenset({
            "format",
            "mode",
            "quant_format",
            "scheme",
            "activation_dtype",
            "activation_granularity",
            "weight_dtype",
            "weight_granularity",
            "accumulation_dtype",
            "operators",
            "encoder",
            "encoder_bytes",
            "encoder_sha256",
            "decoder",
            "decoder_bytes",
            "decoder_sha256",
            "calibration",
            "validation",
        }),
        "source manifest variants.int8_w8a8",
    )
    contract = {
        "format": "tiny_receipt_vqa_onnx_w8a8_u8s8_qdq_v1",
        "mode": "static_w8a8_qdq",
        "quant_format": "QDQ",
        "scheme": "U8S8",
        "activation_dtype": "uint8",
        "activation_granularity": "per_tensor",
        "weight_dtype": "int8",
        "weight_granularity": "per_output_channel",
        "accumulation_dtype": "int32",
        "operators": ["Conv", "MatMul", "Gemm"],
    }
    if {key: int8_variant.get(key) for key in contract} != contract:
        raise ImportFailure(
            "source manifest variants.int8_w8a8 is not the exact static U8S8 QDQ contract"
        )
    for role in ("encoder", "decoder"):
        _validate_variant_file_reference(
            int8_variant,
            files,
            role=role,
            file_key=f"{role}_int8_w8a8",
            label="source manifest variants.int8_w8a8",
        )

    calibration = int8_variant.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ImportFailure(
            "source manifest variants.int8_w8a8.calibration must be an object"
        )
    _require_exact_keys(
        calibration,
        frozenset({
            "source",
            "seed",
            "records",
            "receipt_limit",
            "prefixes_per_record",
            "decoder_feeds",
            "heldout_used",
            "family_counts",
        }),
        "source manifest variants.int8_w8a8.calibration",
    )
    if (
        calibration.get("source") != "synthetic_training_split"
        or calibration.get("heldout_used") is not False
    ):
        raise ImportFailure(
            "source manifest variants.int8_w8a8 calibration provenance is invalid"
        )
    calibration_counts = {}
    for key in ("seed", "records", "receipt_limit", "prefixes_per_record", "decoder_feeds"):
        calibration_counts[key] = _positive_int(
            calibration.get(key),
            f"source manifest variants.int8_w8a8.calibration.{key}",
        )
    if calibration_counts["receipt_limit"] < calibration_counts["records"] or (
        calibration_counts["decoder_feeds"]
        != calibration_counts["records"] * calibration_counts["prefixes_per_record"]
    ):
        raise ImportFailure(
            "source manifest variants.int8_w8a8 calibration counts are inconsistent"
        )
    family_counts = calibration.get("family_counts")
    calibrated_families = frozenset(FAMILY_ORDER[:6])
    if (
        not isinstance(family_counts, Mapping)
        or set(family_counts) != calibrated_families
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for count in family_counts.values()
        )
        or sum(family_counts.values()) != calibration_counts["records"]
    ):
        raise ImportFailure(
            "source manifest variants.int8_w8a8 family calibration counts are invalid"
        )

    validation = int8_variant.get("validation")
    if not isinstance(validation, Mapping):
        raise ImportFailure(
            "source manifest variants.int8_w8a8.validation must be an object"
        )
    _require_exact_keys(
        validation,
        frozenset({"onnx_checker", "heldout", "optimized_graph", "runtime"}),
        "source manifest variants.int8_w8a8.validation",
    )
    if validation.get("onnx_checker") != "passed":
        raise ImportFailure(
            "source manifest variants.int8_w8a8 ONNX checker validation is invalid"
        )
    _validate_heldout_summary(validation.get("heldout"), files)
    optimized_graph = validation.get("optimized_graph")
    if not isinstance(optimized_graph, Mapping):
        raise ImportFailure(
            "source manifest variants.int8_w8a8 optimized_graph must be an object"
        )
    _require_exact_keys(
        optimized_graph,
        frozenset({"encoder", "decoder"}),
        "source manifest variants.int8_w8a8.validation.optimized_graph",
    )
    for role in ("encoder", "decoder"):
        _validate_optimized_graph_summary(
            optimized_graph[role],
            f"source manifest variants.int8_w8a8.validation.optimized_graph.{role}",
        )
    if not {"QLinearConv", "QLinearMatMul", "QGemm"}.issubset(
        optimized_graph["encoder"]["nodes"]
    ) or not {"QLinearMatMul", "QGemm"}.issubset(
        optimized_graph["decoder"]["nodes"]
    ):
        raise ImportFailure(
            "source manifest variants.int8_w8a8 optimized graph lacks U8S8 compute"
        )

    runtime = validation.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ImportFailure(
            "source manifest variants.int8_w8a8 runtime validation must be an object"
        )
    _require_exact_keys(
        runtime,
        frozenset({
            "dynamic_prefix_lengths",
            "family_ids",
            "onnxruntime",
            "onnxruntime_version",
            "provider",
        }),
        "source manifest variants.int8_w8a8.validation.runtime",
    )
    dynamic_prefix_lengths = runtime.get("dynamic_prefix_lengths")
    valid_dynamic_prefix_lengths = (
        isinstance(dynamic_prefix_lengths, list)
        and len(dynamic_prefix_lengths) == calibration_counts["prefixes_per_record"]
        and all(
            type(length) is int and 0 < length <= maximum_prefix_length
            for length in dynamic_prefix_lengths
        )
        and dynamic_prefix_lengths[0] == 1
        and dynamic_prefix_lengths == sorted(set(dynamic_prefix_lengths))
    )
    if (
        not valid_dynamic_prefix_lengths
        or runtime.get("family_ids") != [-1, *range(8)]
        or runtime.get("onnxruntime") != "passed"
        or runtime.get("provider") != "CPUExecutionProvider"
        or not isinstance(runtime.get("onnxruntime_version"), str)
        or not runtime["onnxruntime_version"]
    ):
        raise ImportFailure(
            "source manifest variants.int8_w8a8 runtime validation is invalid"
        )


def _validate_kv_variant_manifest(
    manifest: Mapping[str, Any],
    files: Mapping[str, Any],
    *,
    has_int8_w8a8: bool,
) -> None:
    """Validate the v1 variant records without importing producer eval files.

    The release carries useful benchmark and heldout provenance, but those
    referenced reports are intentionally outside the four-file import trust
    boundary.  The importer closes over the graph identity, public numeric
    ABI, and quantization contract that affect the resulting package.
    """

    variants = manifest.get("variants")
    if not isinstance(variants, Mapping):
        raise ImportFailure("KV source manifest variants must be an object")
    expected = {"fp32"}
    if has_int8_w8a8:
        expected.add("int8_w8a8")
    if set(variants) != expected:
        raise ImportFailure("KV source manifest variants do not match its model files")
    fp32 = variants.get("fp32")
    if not isinstance(fp32, Mapping) or {
        "activation_dtype": fp32.get("activation_dtype"),
        "compute_dtype": fp32.get("compute_dtype"),
        "weight_dtype": fp32.get("weight_dtype"),
    } != {
        "activation_dtype": "float32",
        "compute_dtype": "float32",
        "weight_dtype": "float32",
    }:
        raise ImportFailure("KV source variants.fp32 has the wrong dtype contract")
    for role in ("encoder", "decoder"):
        _validate_variant_file_reference(
            fp32, files, role=role, file_key=role,
            label="KV source manifest variants.fp32",
        )
    if not has_int8_w8a8:
        return
    int8_variant = variants.get("int8_w8a8")
    if not isinstance(int8_variant, Mapping):
        raise ImportFailure("KV source variants.int8_w8a8 must be an object")
    contract = {
        "format": "tiny_receipt_vqa_onnx_w8a8_u8s8_qdq_v1",
        "mode": "static_w8a8_qdq",
        "quant_format": "QDQ",
        "scheme": "U8S8",
        "activation_dtype": "uint8",
        "activation_granularity": "per_tensor",
        "weight_dtype": "int8",
        "weight_granularity": "per_output_channel",
        "accumulation_dtype": "int32",
        "operators": ["Conv", "MatMul", "Gemm"],
    }
    if {key: int8_variant.get(key) for key in contract} != contract:
        raise ImportFailure("KV source INT8 variant is not the static U8S8 QDQ contract")
    for role in ("encoder", "decoder"):
        _validate_variant_file_reference(
            int8_variant, files, role=role, file_key=f"{role}_int8_w8a8",
            label="KV source manifest variants.int8_w8a8",
        )


def _validate_kv_source(
    source: Path,
    paths: dict[str, Path],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    vocab: Mapping[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    if manifest.get("source_format") != "safetensors" or manifest.get("opset") != 18:
        raise ImportFailure("KV source must declare safetensors provenance and opset 18")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ImportFailure("KV source manifest requires a files object")
    actual_file_keys = set(files)
    missing_base_keys = BASE_SOURCE_FILE_KEYS - actual_file_keys
    unknown_file_keys = actual_file_keys - (
        BASE_SOURCE_FILE_KEYS | INT8_W8A8_SOURCE_FILE_KEYS
    )
    if missing_base_keys or unknown_file_keys:
        raise ImportFailure(
            "KV source manifest files has the wrong key set; "
            f"missing={sorted(missing_base_keys)}, unsupported={sorted(unknown_file_keys)}"
        )
    present_int8_keys = actual_file_keys & INT8_W8A8_SOURCE_FILE_KEYS
    if present_int8_keys and present_int8_keys != INT8_W8A8_SOURCE_FILE_KEYS:
        raise ImportFailure("KV source INT8 files must use the complete six-key set")
    for key in ("encoder", "decoder", "config", "vocab"):
        if files.get(key) != SOURCE_FILES[key]:
            raise ImportFailure(f"KV source files.{key} must be {SOURCE_FILES[key]!r}")
    validated_model_keys = ["encoder", "decoder"]
    for key, filename in INT8_W8A8_SOURCE_FILES.items():
        if key in present_int8_keys:
            if files.get(key) != filename:
                raise ImportFailure(f"KV source files.{key} must be {filename!r}")
            paths[key] = _regular_file(source, filename)
            validated_model_keys.append(key)
    if variant == "int8-w8a8" and not present_int8_keys:
        raise ImportFailure("requested int8-w8a8 variant is absent from the KV source")
    for key in validated_model_keys:
        if files.get(f"{key}_bytes") != paths[key].stat().st_size:
            raise ImportFailure(f"KV source {key} byte size does not match its manifest")
        if files.get(f"{key}_sha256") != _sha256(paths[key]):
            raise ImportFailure(f"KV source {key} SHA-256 does not match its manifest")
    _validate_kv_variant_manifest(
        manifest, files, has_int8_w8a8=bool(present_int8_keys)
    )
    exporter_contract = manifest.get("exporter")
    if not isinstance(exporter_contract, Mapping):
        raise ImportFailure("KV source manifest requires exporter batch provenance")
    producer_max_batch_size = _positive_int(
        exporter_contract.get("max_batch_size"),
        "KV source manifest exporter.max_batch_size",
    )

    required_config = {
        "vocab_size": BPE_VOCAB_SIZE,
        "d_model": 320,
        "heads": KV_HEADS,
        "enc_layers": 6,
        "dec_layers": KV_LAYERS,
        "max_q_len": MAX_Q,
        "max_out_len": MAX_T,
        "img_tokens": IMAGE_TOKENS,
        "adapter_families": len(FAMILY_ORDER),
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ImportFailure(f"KV source config {key!r} must equal {expected}")
    if config.get("use_adapters") is not True or config.get("use_router") is not True:
        raise ImportFailure("KV source must enable adapters and the learned router")
    vocab_size, tokenizer = _validate_vocabulary(config, vocab, manifest)
    for label, document in (("config", config), ("vocab", vocab)):
        if _contains_private_absolute_path(document):
            raise ImportFailure(f"KV source {label} contains a private absolute path")

    families = manifest.get("adapter_families")
    if not isinstance(families, Mapping) or families.get("ordered_names") != list(FAMILY_ORDER):
        raise ImportFailure("KV source adapter family order is not canonical")
    if families.get("name_to_id") != {
        name: index for index, name in enumerate(FAMILY_ORDER)
    }:
        raise ImportFailure("KV source adapter family IDs are not canonical")
    generation = manifest.get("generation")
    if not isinstance(generation, Mapping) or {
        "bos_token_id": generation.get("bos_token_id"),
        "eos_token_id": generation.get("eos_token_id"),
        "pad_token_id": generation.get("pad_token_id"),
        "max_length": generation.get("max_length"),
    } != {
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        "max_length": MAX_T,
    } or "KV cache" not in str(generation.get("strategy")):
        raise ImportFailure("KV source generation contract is invalid")
    cache = manifest.get("kv_cache")
    if not isinstance(cache, Mapping) or (
        cache.get("format") != "tiny_receipt_vqa_default_kv_cache_v2"
        or cache.get("default_for") != ["fp32", "int8_w8a8"]
    ):
        raise ImportFailure("KV source cache declaration is invalid")
    if manifest.get("outputs") != {
        "encoder": [
            "memory", "memory_padding_mask", "router_logits",
            "selected_family_ids", *KV_CROSS_NAMES,
        ],
        "decoder": ["logits", "present_padding_mask", *KV_PRESENT_NAMES],
    }:
        raise ImportFailure("KV source output names are not the explicit-cache contract")

    try:
        from onnx import TensorProto
    except ImportError as error:  # pragma: no cover
        raise ImportFailure("the local ONNX package is required to validate this source") from error
    selected_keys = SELECTED_MODEL_KEYS[variant]
    require_u8s8_qdq = variant == "int8-w8a8"
    cross_output_shapes = {
        name: [
            "batch",
            KV_HEADS if require_u8s8_qdq else f"Transpose{name}_dim_1",
            "memory_length",
            KV_HEAD_WIDTH if require_u8s8_qdq else f"Transpose{name}_dim_3",
        ]
        for name in KV_CROSS_NAMES
    }
    encoder_quantized_ops = _validate_onnx(
        paths[selected_keys["encoder"]],
        label=f"{variant} KV encoder",
        expected_opset=18,
        expected_inputs={
            "image": (TensorProto.FLOAT, ["batch", 1, IMAGE_HEIGHT, IMAGE_WIDTH]),
            "question_ids": (TensorProto.INT64, ["batch", "question_length"]),
            "family_ids": (TensorProto.INT64, ["batch"]),
        },
        expected_outputs={
            "memory": (TensorProto.FLOAT, ["batch", "question_length + 210", 320]),
            "memory_padding_mask": (
                TensorProto.BOOL, ["batch", "question_length + 210"]
            ),
            "router_logits": (TensorProto.FLOAT, ["batch", len(FAMILY_ORDER)]),
            "selected_family_ids": (TensorProto.INT64, ["batch"]),
            **{
                name: (TensorProto.FLOAT, shape)
                for name, shape in cross_output_shapes.items()
            },
        },
        require_u8s8_qdq=require_u8s8_qdq,
    )
    decoder_inputs: dict[str, tuple[int, list[int | str]]] = {
        "decoder_input_ids": (TensorProto.INT64, ["batch", 1]),
        "position_ids": (TensorProto.INT64, ["batch"]),
        "family_ids": (TensorProto.INT64, ["batch"]),
        "memory_padding_mask": (TensorProto.BOOL, ["batch", "memory_length"]),
        "past_padding_mask": (TensorProto.BOOL, ["batch", "past_length"]),
    }
    decoder_inputs.update({
        name: (TensorProto.FLOAT, ["batch", KV_HEADS, "memory_length", KV_HEAD_WIDTH])
        for name in KV_CROSS_NAMES
    })
    decoder_inputs.update({
        name: (TensorProto.FLOAT, ["batch", KV_HEADS, "past_length", KV_HEAD_WIDTH])
        for name in KV_PAST_NAMES
    })
    decoder_outputs: dict[str, tuple[int, list[int | str]]] = {
        "logits": (TensorProto.FLOAT, ["batch", 1, vocab_size]),
        "present_padding_mask": (TensorProto.BOOL, ["batch", "present_length"]),
    }
    decoder_outputs.update({
        name: (TensorProto.FLOAT, ["batch", KV_HEADS, "present_length", KV_HEAD_WIDTH])
        for name in KV_PRESENT_NAMES
    })
    decoder_quantized_ops = _validate_onnx(
        paths[selected_keys["decoder"]],
        label=f"{variant} KV decoder",
        expected_opset=18,
        expected_inputs=decoder_inputs,
        expected_outputs=decoder_outputs,
        require_u8s8_qdq=require_u8s8_qdq,
    )
    if require_u8s8_qdq and (
        encoder_quantized_ops | decoder_quantized_ops
    ) != frozenset({"Conv", "MatMul", "Gemm"}):
        raise ImportFailure("KV INT8 graphs do not implement the declared operator set")
    hashes = {key: _sha256(path) for key, path in paths.items()}
    return {
        "paths": paths,
        "selected_paths": {role: paths[key] for role, key in selected_keys.items()},
        "selected_file_keys": dict(selected_keys),
        "variant": variant,
        "source_format": SOURCE_FORMAT,
        "cache_mode": "explicit-kv",
        "manifest": manifest,
        "config": config,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "tokenizer": tokenizer,
        "hashes": hashes,
        "memory_length": IMAGE_TOKENS + MAX_Q,
        "producer_max_batch_size": producer_max_batch_size,
    }


def validate_source(source: Path, *, variant: str = "fp32") -> dict[str, Any]:
    """Validate the sole explicit-KV producer contract."""

    if variant not in CLI_VARIANTS:
        raise ImportFailure(f"unsupported TinyReceipt source variant {variant!r}")

    try:
        root_metadata = source.lstat()
    except OSError as error:
        raise ImportFailure(f"source directory is unavailable: {error}") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ImportFailure("source must be a real local directory, not a symlink")
    paths = {key: _regular_file(source, name) for key, name in SOURCE_FILES.items()}
    manifest = _read_json(paths["manifest"], "source manifest")
    config = _read_json(paths["config"], "source config")
    vocab = _read_json(paths["vocab"], "source vocabulary")
    if manifest.get("format") != SOURCE_FORMAT:
        raise ImportFailure(f"source manifest format must be {SOURCE_FORMAT!r}")
    return _validate_kv_source(
        source, paths, manifest, config, vocab, variant=variant
    )

def verify_explicit_kv_sentinel(source: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the positive P=1 prefill is equivalent to the producer's P=0 prefill.

    Volvox bounded dimensions intentionally reject zero extents.  The v2
    package therefore carries one all-zero self-cache slot whose padding-mask
    bit is true.  This qualification executes every adapter family and proves
    that the visible token and newly appended cache are unchanged.
    """

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - production dependency.
        raise ImportFailure(
            "NumPy and ONNX Runtime are required for KV sentinel parity"
        ) from error

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(source["selected_paths"]["decoder"]),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(955740)
    memory_length = IMAGE_TOKENS + 1
    common: dict[str, Any] = {
        "decoder_input_ids": np.asarray([[1]], dtype=np.int64),
        "position_ids": np.asarray([0], dtype=np.int64),
        "memory_padding_mask": np.zeros((1, memory_length), dtype=np.bool_),
    }
    for name in KV_CROSS_NAMES:
        common[name] = rng.standard_normal(
            (1, KV_HEADS, memory_length, KV_HEAD_WIDTH), dtype=np.float32
        ) * np.float32(0.01)
    output_names = ["logits", "present_padding_mask", *KV_PRESENT_NAMES]
    maximum_logits_difference = 0.0
    maximum_cache_difference = 0.0
    for family_id in range(len(FAMILY_ORDER)):
        empty = dict(common)
        empty["family_ids"] = np.asarray([family_id], dtype=np.int64)
        empty["past_padding_mask"] = np.zeros((1, 0), dtype=np.bool_)
        sentinel = dict(common)
        sentinel["family_ids"] = np.asarray([family_id], dtype=np.int64)
        sentinel["past_padding_mask"] = np.ones((1, 1), dtype=np.bool_)
        for name in KV_PAST_NAMES:
            empty[name] = np.zeros(
                (1, KV_HEADS, 0, KV_HEAD_WIDTH), dtype=np.float32
            )
            sentinel[name] = np.zeros(
                (1, KV_HEADS, 1, KV_HEAD_WIDTH), dtype=np.float32
            )
        empty_outputs = session.run(output_names, empty)
        sentinel_outputs = session.run(output_names, sentinel)
        empty_logits = empty_outputs[0]
        sentinel_logits = sentinel_outputs[0]
        if (
            empty_logits.shape != (1, 1, BPE_VOCAB_SIZE)
            or sentinel_logits.shape != (1, 1, BPE_VOCAB_SIZE)
            or not np.all(np.isfinite(empty_logits))
            or not np.all(np.isfinite(sentinel_logits))
        ):
            raise ImportFailure(
                f"KV sentinel produced invalid logits for family {family_id}"
            )
        logits_difference = float(np.max(np.abs(
            empty_logits.astype(np.float64)
            - sentinel_logits.astype(np.float64)
        )))
        maximum_logits_difference = max(maximum_logits_difference, logits_difference)
        if not math.isfinite(logits_difference) or logits_difference > 1.0e-6 or int(
            np.argmax(empty_logits)
        ) != int(
            np.argmax(sentinel_logits)
        ):
            raise ImportFailure(
                f"KV sentinel changed family {family_id} decoder logits"
            )
        empty_mask = empty_outputs[1]
        sentinel_mask = sentinel_outputs[1]
        if (
            empty_mask.shape != (1, 1)
            or sentinel_mask.shape != (1, 2)
            or bool(sentinel_mask[0, 0]) is not True
            or not np.array_equal(empty_mask, sentinel_mask[:, 1:])
        ):
            raise ImportFailure("KV sentinel did not remain blocked in the present mask")
        for index, name in enumerate(KV_PRESENT_NAMES, start=2):
            empty_cache = empty_outputs[index]
            sentinel_cache = sentinel_outputs[index]
            if empty_cache.shape != (1, KV_HEADS, 1, KV_HEAD_WIDTH) or (
                sentinel_cache.shape != (1, KV_HEADS, 2, KV_HEAD_WIDTH)
            ) or not np.all(np.isfinite(empty_cache)) or not np.all(
                np.isfinite(sentinel_cache)
            ) or np.any(sentinel_cache[:, :, 0, :] != 0.0):
                raise ImportFailure(f"KV sentinel layout is invalid for {name}")
            difference = float(np.max(np.abs(
                empty_cache.astype(np.float64)
                - sentinel_cache[:, :, 1:, :].astype(np.float64)
            )))
            maximum_cache_difference = max(maximum_cache_difference, difference)
            if not math.isfinite(difference) or difference > 1.0e-6:
                raise ImportFailure(f"KV sentinel changed newly appended {name}")
    return {
        "provider": "CPUExecutionProvider",
        "families_verified": list(FAMILY_ORDER),
        "producer_initial_past_length": 0,
        "package_initial_past_length": 1,
        "sentinel_mask_value": 1,
        "logits_max_abs_difference": maximum_logits_difference,
        "present_cache_suffix_max_abs_difference": maximum_cache_difference,
        "greedy_argmax": "matched",
        "tolerance": 1.0e-6,
    }


def _run_exporter(command: Sequence[str]) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise ImportFailure(f"generic ONNX exporter failed: {detail}")


def _sanitize_export_report(
    path: Path,
    source_name: str,
    targets: Sequence[str],
) -> None:
    report = dict(_read_json(path, "staged exporter report"))
    source = report.get("source")
    if (
        report.get("format") != "volvox-export-report/v1"
        or not isinstance(source, Mapping)
    ):
        raise ImportFailure("staged exporter report has the wrong format")
    if (
        report.get("supported") is not True
        or report.get("published") is not True
    ):
        raise ImportFailure("staged exporter report does not attest publication")
    target_evidence = report.get("targets")
    if (
        not isinstance(target_evidence, Mapping)
        or target_evidence.get("requested") != list(targets)
        or target_evidence.get("resolved") != list(expand_targets(targets))
    ):
        raise ImportFailure(
            "staged exporter report does not attest the requested targets"
        )
    sanitized_source = dict(source)
    sanitized_source["path"] = source_name
    report["source"] = sanitized_source
    path.write_text(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _contains_private_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_private_absolute_path(key)
            or _contains_private_absolute_path(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_private_absolute_path(item) for item in value)
    if not isinstance(value, str):
        return False
    return (
        (len(value) > 1 and value.startswith(("/", "\\")))
        or value.lower().startswith("file:")
        or (
            len(value) >= 3
            and value[0].isalpha()
            and value[1] == ":"
            and value[2] in {"/", "\\"}
        )
    )


def _export_command(
    *,
    model: Path,
    output: Path,
    report: Path,
    targets: Sequence[str],
    weight_dtype: str,
    quant_mode: str,
    input_shapes: Mapping[str, str] | None,
    dimension_bounds: Mapping[str, Mapping[str, int]] | None,
    input_dtypes: Mapping[str, str],
    output_dtypes: Mapping[str, str],
    output_names: Sequence[str],
    allow_silu_numerical_migration: bool = False,
    allow_quantized_bias_folding_numerical_migration: bool = False,
    allow_static_qdq_qbatch_matmul_numerical_migration: bool = False,
    allow_static_qdq_groupnorm_silu_numerical_migration: bool = False,
    enable_exact_common_subexpression_elimination: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(GENERIC_EXPORTER),
        "--model",
        str(model),
        "--out",
        str(output),
        "--weight-dtype",
        weight_dtype,
        "--quant-mode",
        quant_mode,
        "--report",
        str(report),
        "--report-format",
        "json",
    ]
    for target in targets:
        command.extend(["--target", target])
    if allow_silu_numerical_migration:
        command.append("--allow-silu-numerical-migration")
    if allow_quantized_bias_folding_numerical_migration:
        command.append("--allow-quantized-bias-folding-migration")
    if allow_static_qdq_qbatch_matmul_numerical_migration:
        command.append("--allow-static-qdq-qbatch-matmul-migration")
    if allow_static_qdq_groupnorm_silu_numerical_migration:
        command.append("--allow-static-qdq-groupnorm-silu-migration")
    if enable_exact_common_subexpression_elimination:
        command.append("--enable-exact-common-subexpression-elimination")
    for name, shape in (input_shapes or {}).items():
        command.extend(["--input-shape", f"{name}={shape}"])
    for name, descriptor in (dimension_bounds or {}).items():
        minimum = descriptor.get("min")
        maximum = descriptor.get("max")
        multiple = descriptor.get("multiple_of")
        bound = f"{minimum}:{maximum}"
        if multiple is not None:
            bound += f":{multiple}"
        command.extend(["--dimension-bound", f"{name}={bound}"])
    for name, dtype in input_dtypes.items():
        command.extend(["--input-dtype", f"{name}={dtype}"])
    for name, dtype in output_dtypes.items():
        command.extend(["--output-dtype", f"{name}={dtype}"])
    for name in output_names:
        command.extend(["--output-name", name])
    return command


def _semantic_input_map(
    graph: Mapping[str, Any],
    expected: Mapping[str, tuple[list[int | str], str]],
    label: str,
) -> dict[str, str]:
    raw_inputs = graph.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ImportFailure(f"staged {label} graph has no inputs object")
    if len(raw_inputs) != len(expected):
        raise ImportFailure(f"staged {label} graph input set is incomplete")
    result: dict[str, str] = {}
    for index, ((canonical, descriptor), (source_name, expected_descriptor)) in enumerate(
        zip(raw_inputs.items(), expected.items())
    ):
        if not isinstance(descriptor, Mapping):
            raise ImportFailure(f"staged {label} input {canonical!r} is malformed")
        if canonical != f"input{index}":
            raise ImportFailure(
                f"staged {label} graph input order is not the canonical input0..N ABI"
            )
        result[source_name] = str(canonical)
        if (
            descriptor.get("shape"),
            descriptor.get("dtype"),
        ) != expected_descriptor:
            raise ImportFailure(f"staged {label} input {source_name!r} has the wrong ABI")
    return result


def _graph_tensor_descriptors(
    graph: Mapping[str, Any],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    descriptors: dict[str, Mapping[str, Any]] = {}
    raw_inputs = graph.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ImportFailure(f"staged {label} graph has no inputs object")
    for name, descriptor in raw_inputs.items():
        if isinstance(name, str) and isinstance(descriptor, Mapping):
            descriptors[name] = descriptor
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ImportFailure(f"staged {label} graph has no node list")
    for node in nodes:
        outputs = node.get("outputs") if isinstance(node, Mapping) else None
        if not isinstance(outputs, Mapping):
            continue
        for descriptor in outputs.values():
            if isinstance(descriptor, Mapping) and isinstance(
                descriptor.get("tensor"), str
            ):
                descriptors[descriptor["tensor"]] = descriptor
    return descriptors


def _tensor_reaches_output(
    graph: Mapping[str, Any],
    source: str,
    target: str,
) -> bool:
    if source == target:
        return True
    edges: dict[str, set[str]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        outputs = node.get("outputs")
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            continue
        output_names = {
            descriptor.get("tensor")
            for descriptor in outputs.values()
            if isinstance(descriptor, Mapping)
            and isinstance(descriptor.get("tensor"), str)
        }
        for input_name in inputs.values():
            if isinstance(input_name, str):
                edges.setdefault(input_name, set()).update(output_names)
    pending = [source]
    visited = {source}
    while pending:
        current = pending.pop()
        for output in edges.get(current, ()):
            if output == target:
                return True
            if output not in visited:
                visited.add(output)
                pending.append(output)
    return False


def _qualify_encoder_memory_relation(
    graph: Mapping[str, Any],
    *,
    memory_output: str,
) -> dict[str, Any]:
    """Require a live canonical Concat witness and verify M for all profiles."""

    try:
        from tools.exporter.operator_shape_contracts import (
            infer_concrete_operator_shapes,
        )
    except ImportError as error:  # pragma: no cover - repository invariant.
        raise ImportFailure(
            "canonical operator shape inference is unavailable"
        ) from error
    descriptors = _graph_tensor_descriptors(graph, "encoder")
    candidates: list[tuple[Mapping[str, Any], str]] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, Mapping) or node.get("opType") != "Concat":
            continue
        params = node.get("params")
        inputs = node.get("inputs")
        outputs = node.get("outputs")
        if (
            not isinstance(params, Mapping)
            or params.get("axis") != 1
            or not isinstance(inputs, Mapping)
            or len(inputs) != 2
            or not isinstance(outputs, Mapping)
            or len(outputs) != 1
        ):
            continue
        input_descriptors = [descriptors.get(name) for name in inputs.values()]
        output_descriptor = next(iter(outputs.values()))
        if (
            any(not isinstance(item, Mapping) for item in input_descriptors)
            or not isinstance(output_descriptor, Mapping)
            or output_descriptor.get("shape") != ["B", "M", 320]
            or output_descriptor.get("dtype") != "float32"
        ):
            continue
        input_shapes = [item.get("shape") for item in input_descriptors]
        if input_shapes != [["B", 210, 320], ["B", "Q", 320]]:
            continue
        output_name = output_descriptor.get("tensor")
        if not isinstance(output_name, str) or not _tensor_reaches_output(
            graph, output_name, memory_output
        ):
            continue
        candidates.append((node, output_name))
    if len(candidates) != 1:
        raise ImportFailure(
            "staged encoder must contain exactly one canonical [B,210,320] + "
            "[B,Q,320] Concat witness that reaches memory [B,M,320]"
        )

    for profile_name, profile in SHAPE_PROFILES.items():
        inferred = infer_concrete_operator_shapes(
            "Concat",
            {
                "inputs": {
                    "input0": {
                        "shape": [profile["B"], 210, 320],
                        "dtype": "float32",
                    },
                    "input1": {
                        "shape": [profile["B"], profile["Q"], 320],
                        "dtype": "float32",
                    },
                },
                "params": {"axis": 1},
            },
        )
        expected = (profile["B"], profile["M"], 320)
        if tuple(inferred["out"].shape) != expected:
            raise ImportFailure(
                "canonical Concat inference did not prove M=Q+210 for "
                f"profile {profile_name!r}"
            )
    witness, output_name = candidates[0]
    return {
        "operator": "Concat",
        "node_id": witness.get("id"),
        "concat_axis": 1,
        "fixed_image_tokens": 210,
        "dynamic_question_dimension": "Q",
        "derived_memory_dimension": "M",
        "witness_tensor": output_name,
        "profiles_verified": list(SHAPE_PROFILES),
    }


def _qualify_present_cache_relation(
    graph: Mapping[str, Any],
    *,
    present_outputs: Sequence[str],
) -> dict[str, Any]:
    """Require every exported self-cache to witness R=P+1 via Concat."""

    try:
        from tools.exporter.operator_shape_contracts import (
            infer_concrete_operator_shapes,
        )
    except ImportError as error:  # pragma: no cover
        raise ImportFailure("canonical operator shape inference is unavailable") from error
    descriptors = _graph_tensor_descriptors(graph, "decoder")
    producers = {
        descriptor.get("tensor"): node
        for node in graph.get("nodes", [])
        if isinstance(node, Mapping)
        for descriptor in (
            node.get("outputs", {}).values()
            if isinstance(node.get("outputs"), Mapping) else ()
        )
        if isinstance(descriptor, Mapping)
        and isinstance(descriptor.get("tensor"), str)
    }
    witnesses: list[str] = []
    for target in present_outputs:
        source = target
        node = producers.get(source)
        while isinstance(node, Mapping) and node.get("opType") in {
            "Identity", "QuantizeLinear", "DequantizeLinear", "Cast",
        }:
            inputs = node.get("inputs")
            data_input = inputs.get("input") if isinstance(inputs, Mapping) else None
            if not isinstance(data_input, str):
                break
            source = data_input
            node = producers.get(source)
        params = node.get("params") if isinstance(node, Mapping) else None
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        outputs = node.get("outputs") if isinstance(node, Mapping) else None
        input_descriptors = (
            [descriptors.get(name) for name in inputs.values()]
            if isinstance(inputs, Mapping) else []
        )
        output_descriptor = (
            next(iter(outputs.values()))
            if isinstance(outputs, Mapping) and len(outputs) == 1 else None
        )
        input_shapes = [
            item.get("shape") if isinstance(item, Mapping) else None
            for item in input_descriptors
        ]
        if (
            not isinstance(node, Mapping)
            or node.get("opType") != "Concat"
            or not isinstance(params, Mapping)
            or params.get("axis") != 2
            or input_shapes
            != [
                ["B", KV_HEADS, "P", KV_HEAD_WIDTH],
                ["B", KV_HEADS, 1, KV_HEAD_WIDTH],
            ]
            or not isinstance(output_descriptor, Mapping)
            or output_descriptor.get("shape")
            != ["B", KV_HEADS, "R", KV_HEAD_WIDTH]
            or output_descriptor.get("dtype") != "float32"
        ):
            raise ImportFailure(
                f"staged decoder output {target!r} must have one P+1 Concat witness"
            )
        witnesses.append(str(node.get("id")))
    for past_length in (1, 63, MAX_T - 1):
        inferred = infer_concrete_operator_shapes(
            "Concat",
            {
                "inputs": {
                    "input0": {
                        "shape": [1, KV_HEADS, past_length, KV_HEAD_WIDTH],
                        "dtype": "float32",
                    },
                    "input1": {
                        "shape": [1, KV_HEADS, 1, KV_HEAD_WIDTH],
                        "dtype": "float32",
                    },
                },
                "params": {"axis": 2},
            },
        )
        if tuple(inferred["out"].shape) != (
            1, KV_HEADS, past_length + 1, KV_HEAD_WIDTH
        ):
            raise ImportFailure("canonical Concat inference did not prove R=P+1")
    return {
        "operator": "Concat",
        "axis": 2,
        "past_dimension": "P",
        "fixed_current_tokens": 1,
        "derived_present_dimension": "R",
        "witness_count": len(witnesses),
        "witness_nodes": witnesses,
    }


def _validate_staged_graph(
    directory: Path,
    *,
    label: str,
    expected_dimensions: Mapping[str, Mapping[str, int]],
    expected_inputs: Mapping[str, tuple[list[int | str], str]],
    expected_outputs: Mapping[str, tuple[list[int | str], str]],
) -> tuple[Mapping[str, Any], dict[str, str]]:
    graph_path = directory / "graph.json"
    weights_path = directory / "model.safetensors"
    report_path = directory / "export_report.json"
    for path in (graph_path, weights_path, report_path):
        if not path.is_file() or path.is_symlink():
            raise ImportFailure(f"staged {label} asset {path.name!r} is missing")
    graph = _read_json(graph_path, f"staged {label} graph")
    allowed_root = {
        "format", "dimensions", "inputs", "nodes", "outputs",
        "banks", "quantization",
    }
    if set(graph) - allowed_root or not {
        "format", "dimensions", "inputs", "nodes", "outputs",
    }.issubset(graph):
        raise ImportFailure(f"staged {label} graph does not use the closed v1 root schema")
    if graph.get("format") != GRAPH_FORMAT:
        raise ImportFailure(f"staged {label} graph has the wrong format")
    # Bank slot-count dimensions are synthesized per weight bank by the
    # exporter; they are additive and never part of this model's input contract.
    staged_dimensions = {
        name: value for name, value in (graph.get("dimensions") or {}).items()
        if not name.startswith("bank_")
    }
    if staged_dimensions != dict(expected_dimensions):
        raise ImportFailure(
            f"staged {label} graph has the wrong canonical dimension contract"
        )
    semantic_inputs = _semantic_input_map(graph, expected_inputs, label)
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ImportFailure(f"staged {label} graph has no node list")
    tensor_descriptors: dict[str, tuple[Any, Any]] = {
        name: (descriptor.get("shape"), descriptor.get("dtype"))
        for name, descriptor in graph["inputs"].items()
        if isinstance(descriptor, Mapping)
    }
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ImportFailure(f"staged {label} graph contains a malformed node")
        outputs = node.get("outputs")
        if not isinstance(outputs, Mapping) or not outputs:
            raise ImportFailure(f"staged {label} graph contains an untyped node")
        for port, output_descriptor in outputs.items():
            if (
                not isinstance(port, str)
                or not isinstance(output_descriptor, Mapping)
                or set(output_descriptor) != {"tensor", "shape", "dtype"}
                or not isinstance(output_descriptor.get("tensor"), str)
            ):
                raise ImportFailure(
                    f"staged {label} graph contains an invalid output descriptor"
                )
            tensor_descriptors[output_descriptor["tensor"]] = (
                output_descriptor.get("shape"), output_descriptor.get("dtype")
            )
    if graph.get("outputs") != list(expected_outputs):
        raise ImportFailure(f"staged {label} graph outputs are not semantic/stable")
    for name, descriptor in expected_outputs.items():
        if tensor_descriptors.get(name) != descriptor:
            raise ImportFailure(f"staged {label} output {name!r} has the wrong ABI")
    if weights_path.stat().st_size <= 0:
        raise ImportFailure(f"staged {label} weights are empty")
    return graph, semantic_inputs


def _normalize_distribution_modes(root: Path) -> None:
    try:
        descendants = sorted(root.rglob("*"), key=lambda path: path.as_posix())
        for path in descendants:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise ImportFailure(
                    "staged package contains an unsupported filesystem entry"
                )
            path.chmod(0o755 if stat.S_ISDIR(metadata.st_mode) else 0o644)
        root.chmod(0o755)
    except OSError as error:
        raise ImportFailure("could not prepare distributable package permissions") from error


def _linux_rename_no_replace(source: Path, destination: Path) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ImportFailure("output directory already exists; refusing to overwrite it")
    if error_number in {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }:
        return False
    raise ImportFailure("atomic output publication failed") from OSError(
        error_number,
        os.strerror(error_number),
    )


def _publish_with_exclusive_reservation(stage: Path, destination: Path) -> None:
    """Portable no-overwrite fallback; the manifest remains the commit marker."""

    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ImportFailure(
            "output directory already exists; refusing to overwrite it"
        ) from error
    owns_destination = True
    try:
        entries = {path.name: path for path in stage.iterdir()}
        manifest = entries.pop("package_manifest.json")
        for name in sorted(entries):
            os.rename(entries[name], destination / name)
        os.rename(manifest, destination / manifest.name)
        destination.chmod(0o755)
        stage.rmdir()
        owns_destination = False
    except Exception:
        if owns_destination:
            shutil.rmtree(destination)
        raise


def _publish_directory_no_replace(stage: Path, destination: Path) -> None:
    if _linux_rename_no_replace(stage, destination):
        return
    if os.name == "nt":
        try:
            os.rename(stage, destination)
        except FileExistsError as error:
            raise ImportFailure(
                "output directory already exists; refusing to overwrite it"
            ) from error
        return
    _publish_with_exclusive_reservation(stage, destination)


def _import_kv_package(
    source: Mapping[str, Any],
    output_directory: Path,
    *,
    variant: str,
    targets: tuple[str, ...],
    weight_dtype: str,
    exporter: Callable[[Sequence[str]], None],
    parity: Mapping[str, Any],
    max_batch_size: int,
) -> Path:
    """Compile and publish the v1 one-token explicit-cache package."""

    output_parent = output_directory.parent
    source_config = source["config"]
    d_model = _positive_int(source_config["d_model"], "source config.d_model")
    source_hashes_before = dict(source["hashes"])
    original_selected_paths = source["selected_paths"]
    encoder_dimensions = {
        "B": {"min": 1, "max": max_batch_size},
        "Q": {"min": 1, "max": MAX_Q},
        "M": {"min": IMAGE_TOKENS + 1, "max": IMAGE_TOKENS + MAX_Q},
    }
    decoder_dimensions = {
        "B": {"min": 1, "max": max_batch_size},
        "M": {"min": IMAGE_TOKENS + 1, "max": IMAGE_TOKENS + MAX_Q},
        "P": {"min": 1, "max": MAX_T - 1},
        "R": {"min": 2, "max": MAX_T},
    }
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output_directory.name}.stage-", dir=output_parent
    ))
    try:
        encoder_directory = stage / "encoder"
        decoder_directory = stage / "decoder"
        authoring_directory = stage / ".authoring"
        encoder_directory.mkdir()
        decoder_directory.mkdir()
        authoring_directory.mkdir()
        shutil.copyfile(source["paths"]["config"], stage / "config.json")
        shutil.copyfile(source["paths"]["vocab"], stage / "vocab.json")
        selected_paths = {
            role: authoring_directory / original_selected_paths[role].name
            for role in ("encoder", "decoder")
        }
        authoring_reports = {
            role: _normalized_kv_onnx_symbols(
                original_selected_paths[role], selected_paths[role], role=role
            )
            for role in ("encoder", "decoder")
        }
        authoring_parity = {
            "encoder": _verify_dynamic_authoring_parity(
                original_selected_paths["encoder"],
                selected_paths["encoder"],
                role="encoder",
                max_batch_size=max_batch_size,
            ),
            "decoder": {
                "status": "not_applicable",
                "reason": "decoder executable nodes were not rewritten",
            },
        }
        encoder_outputs = (
            "memory", "memory_padding_mask", "router_logits",
            "selected_family_ids", *KV_CROSS_NAMES,
        )
        decoder_outputs = (
            "logits", "present_padding_mask", *KV_PRESENT_NAMES,
        )
        exporter(_export_command(
            model=selected_paths["encoder"],
            output=encoder_directory / "model.safetensors",
            report=encoder_directory / "export_report.json",
            targets=targets,
            weight_dtype=weight_dtype,
            quant_mode="preserve",
            input_shapes=None,
            dimension_bounds=encoder_dimensions,
            input_dtypes={
                "question_ids": "int32",
                "family_ids": "int32",
                "question_position_ids": "int32",
            },
            output_dtypes={
                "memory_padding_mask": "int32",
                "selected_family_ids": "int32",
            },
            output_names=encoder_outputs,
            allow_silu_numerical_migration=True,
            allow_quantized_bias_folding_numerical_migration=(
                variant == "int8-w8a8"
            ),
            allow_static_qdq_qbatch_matmul_numerical_migration=(
                variant == "int8-w8a8"
            ),
            enable_exact_common_subexpression_elimination=True,
        ))
        exporter(_export_command(
            model=selected_paths["decoder"],
            output=decoder_directory / "model.safetensors",
            report=decoder_directory / "export_report.json",
            targets=targets,
            weight_dtype=weight_dtype,
            quant_mode="preserve",
            input_shapes=None,
            dimension_bounds=decoder_dimensions,
            input_dtypes={
                "decoder_input_ids": "int32",
                "position_ids": "int32",
                "family_ids": "int32",
                "memory_padding_mask": "int32",
                "past_padding_mask": "int32",
            },
            output_dtypes={"present_padding_mask": "int32"},
            output_names=decoder_outputs,
            allow_silu_numerical_migration=True,
            allow_quantized_bias_folding_numerical_migration=(
                variant == "int8-w8a8"
            ),
            allow_static_qdq_qbatch_matmul_numerical_migration=(
                variant == "int8-w8a8"
            ),
            enable_exact_common_subexpression_elimination=True,
        ))
        for role in ("encoder", "decoder"):
            _sanitize_export_report(
                stage / role / "export_report.json",
                original_selected_paths[role].name,
                targets,
            )
        shutil.rmtree(authoring_directory)

        encoder_expected_inputs = {
            "image": (["B", IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH], "float32"),
            "question_ids": (["B", "Q"], "int32"),
            "family_ids": (["B"], "int32"),
            "question_position_ids": (["B", "Q"], "int32"),
        }
        encoder_expected_outputs = {
            "memory": (["B", "M", d_model], "float32"),
            "memory_padding_mask": (["B", "M"], "int32"),
            "router_logits": (["B", len(FAMILY_ORDER)], "float32"),
            "selected_family_ids": (["B"], "int32"),
            **{
                name: (["B", KV_HEADS, "M", KV_HEAD_WIDTH], "float32")
                for name in KV_CROSS_NAMES
            },
        }
        decoder_expected_inputs = {
            "decoder_input_ids": (["B", 1], "int32"),
            "position_ids": (["B"], "int32"),
            "family_ids": (["B"], "int32"),
            "memory_padding_mask": (["B", "M"], "int32"),
            "past_padding_mask": (["B", "P"], "int32"),
            **{
                name: (["B", KV_HEADS, "M", KV_HEAD_WIDTH], "float32")
                for name in KV_CROSS_NAMES
            },
            **{
                name: (["B", KV_HEADS, "P", KV_HEAD_WIDTH], "float32")
                for name in KV_PAST_NAMES
            },
        }
        decoder_expected_outputs = {
            "logits": (["B", 1, source["vocab_size"]], "float32"),
            "present_padding_mask": (["B", "R"], "int32"),
            **{
                name: (["B", KV_HEADS, "R", KV_HEAD_WIDTH], "float32")
                for name in KV_PRESENT_NAMES
            },
        }
        encoder_graph, encoder_inputs = _validate_staged_graph(
            encoder_directory,
            label="encoder",
            expected_dimensions=encoder_dimensions,
            expected_inputs=encoder_expected_inputs,
            expected_outputs=encoder_expected_outputs,
        )
        decoder_graph, decoder_inputs = _validate_staged_graph(
            decoder_directory,
            label="decoder",
            expected_dimensions=decoder_dimensions,
            expected_inputs=decoder_expected_inputs,
            expected_outputs=decoder_expected_outputs,
        )
        memory_qualification = _qualify_encoder_memory_relation(
            encoder_graph, memory_output="memory"
        )
        present_qualification = _qualify_present_cache_relation(
            decoder_graph, present_outputs=KV_PRESENT_NAMES
        )
        staged_package_classes = {
            "encoder": classify_package(encoder_graph),
            "decoder": classify_package(decoder_graph),
        }
        if variant == "int8-w8a8" and any(
            package_class != "hybrid"
            for package_class in staged_package_classes.values()
        ):
            raise ImportFailure(
                "generic exporter must classify both KV INT8 graphs as hybrid"
            )
        for label, document in (
            ("encoder graph", encoder_graph),
            ("decoder graph", decoder_graph),
            ("encoder export report", _read_json(
                encoder_directory / "export_report.json", "encoder export report"
            )),
            ("decoder export report", _read_json(
                decoder_directory / "export_report.json", "decoder export report"
            )),
        ):
            if _contains_private_absolute_path(document):
                raise ImportFailure(f"staged {label} contains a private absolute path")
        if {
            key: _sha256(path) for key, path in source["paths"].items()
        } != source_hashes_before:
            raise ImportFailure("source files changed while the KV package was imported")

        graph_assets: dict[str, Any] = {}
        for role, directory in (
            ("encoder", encoder_directory), ("decoder", decoder_directory)
        ):
            graph_assets[role] = {
                "graph": _asset_record(directory / "graph.json", f"{role}/graph.json"),
                "weights": _asset_record(
                    directory / "model.safetensors", f"{role}/model.safetensors"
                ),
                "export_report": _asset_record(
                    directory / "export_report.json", f"{role}/export_report.json"
                ),
            }
        offline_target_evidence = (
            {"offline_target": targets[0]}
            if len(targets) == 1
            else {
                "offline_target_attestation": {
                    "requested": list(targets),
                    "resolved": list(expand_targets(targets)),
                }
            }
        )
        manifest = {
            "format": PACKAGE_FORMAT,
            "source": {
                "format": SOURCE_FORMAT,
                "variant": variant,
                "manifest": {
                    "path": SOURCE_FILES["manifest"],
                    "bytes": source["paths"]["manifest"].stat().st_size,
                    "sha256": source_hashes_before["manifest"],
                },
                "encoder_onnx": {
                    "path": original_selected_paths["encoder"].name,
                    "bytes": original_selected_paths["encoder"].stat().st_size,
                    "sha256": source_hashes_before[
                        source["selected_file_keys"]["encoder"]
                    ],
                },
                "decoder_onnx": {
                    "path": original_selected_paths["decoder"].name,
                    "bytes": original_selected_paths["decoder"].stat().st_size,
                    "sha256": source_hashes_before[
                        source["selected_file_keys"]["decoder"]
                    ],
                },
            },
            "variant": {
                "requested": variant,
                "producer_key": MANIFEST_VARIANT_KEYS[variant],
                "export_quant_mode": "preserve",
                "compiled_graph_package_class": staged_package_classes,
                "complete_w8a8_fusion": False,
            },
            "assets": {
                "config": _asset_record(stage / "config.json", "config.json"),
                "vocab": _asset_record(stage / "vocab.json", "vocab.json"),
            },
            "tokenizer": dict(source["tokenizer"]),
            "preprocessing": {
                "layout": "NCHW",
                "shape": [1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
                "color": "grayscale",
                "resize": {
                    "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT,
                    "method": "bilinear",
                },
                "normalization": "(x / 255 - 0.5) / 0.5",
            },
            "families": {
                "auto_id": -1,
                "ordered_names": list(FAMILY_ORDER),
                "name_to_id": {
                    name: index for index, name in enumerate(FAMILY_ORDER)
                },
            },
            "routing": {
                "mode": "runtime",
                "family_inputs": {
                    "encoder": encoder_inputs["family_ids"],
                    "decoder": decoder_inputs["family_ids"],
                },
            },
            "generation": {
                "strategy": "greedy-autoregressive-explicit-kv",
                "maximum_target_length": MAX_T,
                "maximum_new_tokens": MAX_T - 1,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "logits_row": "current_token",
                "tie_policy": "first-index",
            },
            "shape_contract": {
                "graph_shape_mode": "bounded-explicit-kv-v2",
                "dimensions": {
                    "B": {"min": 1, "max": max_batch_size},
                    "Q": {"min": 1, "max": MAX_Q},
                    "M": {"min": IMAGE_TOKENS + 1, "max": IMAGE_TOKENS + MAX_Q},
                    "P": {"min": 1, "max": MAX_T - 1},
                    "R": {"min": 2, "max": MAX_T},
                },
                "fixed_geometry": {
                    "image": [1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
                    "image_tokens": IMAGE_TOKENS,
                    "feature_width": d_model,
                    "attention_heads": KV_HEADS,
                    "attention_head_width": KV_HEAD_WIDTH,
                    "decoder_layers": KV_LAYERS,
                    "adapter_families": len(FAMILY_ORDER),
                },
                "relations": {
                    "encoder_memory": {
                        "operator": "Concat",
                        "axis": 1,
                        "fixed_image_tokens": IMAGE_TOKENS,
                        "dynamic_question_dimension": "Q",
                        "derived_memory_dimension": "M",
                    },
                    "present_cache": {
                        "operator": "Concat",
                        "axis": 2,
                        "past_dimension": "P",
                        "fixed_current_tokens": 1,
                        "derived_present_dimension": "R",
                    },
                },
                "semantic_inputs": {
                    "question_position_ids": {
                        "shape": ["B", "Q"],
                        "values": "zero_based_contiguous",
                    },
                },
            },
            "cache_contract": {
                "format": "masked-zero-sentinel-v1",
                "layers": KV_LAYERS,
                "heads": KV_HEADS,
                "head_width": KV_HEAD_WIDTH,
                "past_dimension": "P",
                "present_dimension": "R",
                "initial_past_length": 1,
                "sentinel_mask_value": 1,
                "cache_dtype": "float32",
            },
            "graphs": {
                "encoder": {
                    **graph_assets["encoder"],
                    "inputs": encoder_inputs,
                    "outputs": {name: name for name in encoder_outputs},
                },
                "decoder": {
                    **graph_assets["decoder"],
                    "inputs": decoder_inputs,
                    "outputs": {name: name for name in decoder_outputs},
                },
            },
            "mask_semantics": {
                "memory_padding_mask": "nonzero_means_blocked",
                "past_padding_mask": "nonzero_means_blocked",
            },
            "validation": {
                "onnx_checker": "passed",
                "kv_authoring_normalization": authoring_reports,
                "kv_authoring_parity": authoring_parity,
                "empty_vs_masked_sentinel_parity": dict(parity),
                "canonical_memory_relation": memory_qualification,
                "canonical_present_relation": present_qualification,
                **offline_target_evidence,
                "source_variant": variant,
                "strict_runtime_execution": "not-run",
            },
        }
        if _contains_private_absolute_path(manifest):
            raise ImportFailure("KV package manifest contains a private absolute path")
        (stage / "package_manifest.json").write_text(
            json.dumps(
                manifest, allow_nan=False, ensure_ascii=False,
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        _normalize_distribution_modes(stage)
        _publish_directory_no_replace(stage, output_directory)
    except Exception:
        if stage.exists() and stage.parent == output_parent and stage.name.startswith(
            f".{output_directory.name}.stage-"
        ):
            shutil.rmtree(stage)
        raise
    return output_directory


def import_package(
    source_directory: Path,
    output_directory: Path,
    *,
    variant: str = "fp32",
    targets: Sequence[str] | None = None,
    weight_dtype: str = "auto",
    exporter: Callable[[Sequence[str]], None] = _run_exporter,
    parity_verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    max_batch_size: int = 1,
) -> Path:
    requested_targets = normalize_targets(targets)
    source_directory = source_directory.resolve(strict=True)
    output_parent = output_directory.parent.resolve(strict=True)
    output_directory = output_parent / output_directory.name
    if os.path.lexists(output_directory):
        raise ImportFailure("output directory already exists; refusing to overwrite it")
    if source_directory == output_directory or source_directory in output_directory.parents:
        raise ImportFailure("output directory must not be inside the immutable source")

    source = validate_source(source_directory, variant=variant)
    max_batch_size = _validated_max_batch_size(
        max_batch_size,
        source["producer_max_batch_size"],
    )
    verifier = parity_verifier or verify_explicit_kv_sentinel
    parity = dict(verifier(source))
    return _import_kv_package(
        source,
        output_directory,
        variant=variant,
        targets=requested_targets,
        weight_dtype=weight_dtype,
        exporter=exporter,
        parity=parity,
        max_batch_size=max_batch_size,
    )

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a local typed TinyReceipt split-ONNX release."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--variant",
        default="fp32",
        choices=CLI_VARIANTS,
        help="Select the producer FP32 or static U8S8 W8A8 QDQ graphs.",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=GENERIC_EXPORT_TARGETS,
        metavar="TARGET",
        help=(
            "Required generic-exporter target; repeat to require their "
            "intersection (default: portable)."
        ),
    )
    parser.add_argument(
        "--weight-dtype",
        default="auto",
        choices=("auto", "float32", "float16"),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help=(
            "retain the producer's leading batch axis with bounded domain "
            "1..N (bounded by the source manifest; default: 1)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        imported = import_package(
            args.source,
            args.out_dir,
            variant=args.variant,
            targets=args.targets,
            weight_dtype=args.weight_dtype,
            max_batch_size=args.max_batch_size,
        )
    except (ImportFailure, OSError, ValueError) as error:
        print(f"[TinyReceipt split ONNX import] {error}", file=sys.stderr)
        return 1
    print(imported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
