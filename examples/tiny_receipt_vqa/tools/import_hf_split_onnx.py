#!/usr/bin/env python3
"""Import the typed TinyReceipt split-ONNX release into a VolvoxAI package.

The importer is deliberately local and model-specific.  It verifies the
producer contract, proves the fixed-padding export against ONNX Runtime, and
then delegates both graphs to the repository's generic ONNX exporter.  It
never loads a PyTorch checkpoint and never downloads model data.
"""

from __future__ import annotations

import argparse
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
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SOURCE_FORMAT = "tiny_receipt_vqa_split_onnx_v1"
PACKAGE_FORMAT = "volvoxai-tiny-receipt-vqa-split-onnx-package-v1"
CHAR_VOCAB_SIZE = 760
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


def _validate_char_vocabulary(
    vocab: Mapping[str, Any],
    *,
    vocab_size: int,
) -> dict[str, Any]:
    itos = vocab.get("itos")
    if (
        vocab_size != CHAR_VOCAB_SIZE
        or set(vocab) != {"itos"}
        or not isinstance(itos, list)
        or len(itos) != vocab_size
        or any(not isinstance(item, str) for item in itos)
        or itos[: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS)
    ):
        raise ImportFailure(
            "source vocab must be the exact 760-entry CharVocab contract"
        )
    return {
        "type": "char-vocab",
        "version": 1,
        "itos_key": "itos",
        "token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
    }


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
    if set(vocab) == {"itos"}:
        return vocab_size, _validate_char_vocabulary(vocab, vocab_size=vocab_size)
    if vocab.get("type") == "byte_fallback_bpe":
        return vocab_size, _validate_bpe_vocabulary(
            vocab,
            manifest.get("tokenizer"),
            vocab_size=vocab_size,
        )
    raise ImportFailure(
        "source vocab must be the legacy CharVocab or byte_fallback_bpe v1 contract"
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


def validate_source(source: Path, *, variant: str = "fp32") -> dict[str, Any]:
    """Validate a producer directory and return immutable import metadata."""

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
    if manifest.get("source_format") != "safetensors" or manifest.get("opset") != 18:
        raise ImportFailure("source manifest must declare safetensors provenance and opset 18")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ImportFailure("source manifest requires a files object")
    actual_file_keys = set(files)
    missing_base_keys = BASE_SOURCE_FILE_KEYS - actual_file_keys
    if missing_base_keys:
        raise ImportFailure(
            "source manifest files is missing required base keys "
            f"{sorted(missing_base_keys)}"
        )
    unknown_file_keys = actual_file_keys - (
        BASE_SOURCE_FILE_KEYS | INT8_W8A8_SOURCE_FILE_KEYS
    )
    if unknown_file_keys:
        raise ImportFailure(
            "source manifest files contains unsupported keys "
            f"{sorted(unknown_file_keys)}"
        )
    present_int8_keys = actual_file_keys & INT8_W8A8_SOURCE_FILE_KEYS
    if present_int8_keys and present_int8_keys != INT8_W8A8_SOURCE_FILE_KEYS:
        missing_int8_keys = INT8_W8A8_SOURCE_FILE_KEYS - present_int8_keys
        raise ImportFailure(
            "source manifest INT8 W8A8 files must use the complete six-key set; "
            f"missing {sorted(missing_int8_keys)}"
        )
    for key in ("encoder", "decoder", "config", "vocab"):
        if files.get(key) != SOURCE_FILES[key]:
            raise ImportFailure(f"source manifest files.{key} must be {SOURCE_FILES[key]!r}")
    validated_model_keys = ["encoder", "decoder"]
    if present_int8_keys:
        for key, filename in INT8_W8A8_SOURCE_FILES.items():
            if files.get(key) != filename:
                raise ImportFailure(
                    f"source manifest files.{key} must be {filename!r}"
                )
            paths[key] = _regular_file(source, filename)
            validated_model_keys.append(key)
    if variant == "int8-w8a8" and not present_int8_keys:
        raise ImportFailure(
            "requested int8-w8a8 variant is absent from the source manifest"
        )
    for key in validated_model_keys:
        actual_bytes = paths[key].stat().st_size
        actual_hash = _sha256(paths[key])
        if files.get(f"{key}_bytes") != actual_bytes:
            raise ImportFailure(f"source {key} byte size does not match its manifest")
        if files.get(f"{key}_sha256") != actual_hash:
            raise ImportFailure(f"source {key} SHA-256 does not match its manifest")
    _validate_producer_variants(
        manifest,
        files,
        has_int8_w8a8=bool(present_int8_keys),
        maximum_prefix_length=_positive_int(
            config.get("max_out_len"),
            "source config.max_out_len",
        ),
    )

    required_config = {
        "d_model": 320,
        "heads": 8,
        "enc_layers": 6,
        "dec_layers": 4,
        "max_q_len": 192,
        "max_out_len": 192,
        "img_tokens": 210,
        "adapter_families": 8,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ImportFailure(f"source config {key!r} must equal {expected}")
    if config.get("use_adapters") is not True or config.get("use_router") is not True:
        raise ImportFailure("source config must enable adapters and the learned router")

    vocab_size, tokenizer = _validate_vocabulary(config, vocab, manifest)
    for label, document in (("config", config), ("vocab", vocab)):
        if _contains_private_absolute_path(document):
            raise ImportFailure(
                f"source {label} contains a private absolute path"
            )

    families = manifest.get("adapter_families")
    if not isinstance(families, Mapping):
        raise ImportFailure("source manifest requires adapter_families")
    if families.get("ordered_names") != list(FAMILY_ORDER):
        raise ImportFailure("source adapter family order is not canonical")
    if families.get("name_to_id") != {
        name: index for index, name in enumerate(FAMILY_ORDER)
    }:
        raise ImportFailure("source adapter family IDs do not match the declared order")

    generation = manifest.get("generation")
    if not isinstance(generation, Mapping) or {
        "bos_token_id": generation.get("bos_token_id"),
        "eos_token_id": generation.get("eos_token_id"),
        "pad_token_id": generation.get("pad_token_id"),
        "max_length": generation.get("max_length"),
    } != {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_length": 192,
    }:
        raise ImportFailure("source generation IDs/length do not match config and vocabulary")
    if manifest.get("outputs") != {
        "encoder": [
            "memory",
            "memory_padding_mask",
            "router_logits",
            "selected_family_ids",
        ],
        "decoder": ["logits"],
    }:
        raise ImportFailure("source output names are not the split-ONNX contract")

    try:
        from onnx import TensorProto
    except ImportError as error:  # pragma: no cover - handled in _validate_onnx.
        raise ImportFailure("the local ONNX package is required to validate this source") from error
    selected_keys = SELECTED_MODEL_KEYS[variant]
    require_u8s8_qdq = variant == "int8-w8a8"
    encoder_quantized_ops = _validate_onnx(
        paths[selected_keys["encoder"]],
        label=f"{variant} encoder",
        expected_opset=18,
        expected_inputs={
            "image": (TensorProto.FLOAT, ["batch", 1, 320, 672]),
            "question_ids": (TensorProto.INT64, ["batch", "question_length"]),
            "family_ids": (TensorProto.INT64, ["batch"]),
        },
        expected_outputs={
            "memory": (TensorProto.FLOAT, ["batch", "question_length + 210", 320]),
            "memory_padding_mask": (
                TensorProto.BOOL,
                ["batch", "question_length + 210"],
            ),
            "router_logits": (TensorProto.FLOAT, ["batch", 8]),
            "selected_family_ids": (TensorProto.INT64, ["batch"]),
        },
        require_u8s8_qdq=require_u8s8_qdq,
    )
    decoder_quantized_ops = _validate_onnx(
        paths[selected_keys["decoder"]],
        label=f"{variant} decoder",
        expected_opset=18,
        expected_inputs={
            "decoder_input_ids": (
                TensorProto.INT64,
                ["batch", "target_length"],
            ),
            "memory": (TensorProto.FLOAT, ["batch", "memory_length", 320]),
            "memory_padding_mask": (
                TensorProto.BOOL,
                ["batch", "memory_length"],
            ),
            "family_ids": (TensorProto.INT64, ["batch"]),
        },
        expected_outputs={
            "logits": (
                TensorProto.FLOAT,
                ["batch", "target_length", vocab_size],
            ),
        },
        require_u8s8_qdq=require_u8s8_qdq,
    )
    if require_u8s8_qdq and (
        encoder_quantized_ops | decoder_quantized_ops
    ) != frozenset({"Conv", "MatMul", "Gemm"}):
        raise ImportFailure(
            "int8-w8a8 encoder/decoder do not implement the declared quantized operator set"
        )
    hashes = {key: _sha256(path) for key, path in paths.items()}
    return {
        "paths": paths,
        "selected_paths": {
            role: paths[key] for role, key in selected_keys.items()
        },
        "selected_file_keys": dict(selected_keys),
        "variant": variant,
        "manifest": manifest,
        "config": config,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "tokenizer": tokenizer,
        "hashes": hashes,
        "memory_length": _positive_int(config["img_tokens"], "config.img_tokens")
        + _positive_int(config["max_q_len"], "config.max_q_len"),
    }


def verify_static_padding(source: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the fixed Q/T=192 lowering against the dynamic source graphs."""

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - production dependency.
        raise ImportFailure(
            "NumPy and ONNX Runtime are required for fixed-padding parity"
        ) from error

    paths = source.get("selected_paths", source["paths"])
    config = source["config"]
    q_length = int(config["max_q_len"])
    target_length = int(config["max_out_len"])
    image = np.zeros((1, 1, 320, 672), dtype=np.float32)
    question = np.asarray([[4, 5, 2]], dtype=np.int64)
    padded_question = np.zeros((1, q_length), dtype=np.int64)
    padded_question[:, : question.shape[1]] = question
    automatic_family = np.asarray([-1], dtype=np.int64)

    try:
        encoder = ort.InferenceSession(
            str(paths["encoder"]),
            providers=["CPUExecutionProvider"],
        )
        dynamic_encoder = encoder.run(
            None,
            {
                "image": image,
                "question_ids": question,
                "family_ids": automatic_family,
            },
        )
        fixed_encoder = encoder.run(
            None,
            {
                "image": image,
                "question_ids": padded_question,
                "family_ids": automatic_family,
            },
        )
    except Exception as error:
        raise ImportFailure(f"encoder fixed-padding parity execution failed: {error}") from error

    dynamic_memory, dynamic_mask, dynamic_router, dynamic_selected = dynamic_encoder
    fixed_memory, fixed_mask, fixed_router, fixed_selected = fixed_encoder
    if dynamic_mask.dtype != np.bool_ or fixed_mask.dtype != np.bool_:
        raise ImportFailure("source memory_padding_mask must be BOOL")
    if np.any(dynamic_mask) or int(np.count_nonzero(~fixed_mask)) != dynamic_memory.shape[1]:
        raise ImportFailure("source memory_padding_mask does not mark padded positions as true")
    fixed_valid_memory = fixed_memory[~fixed_mask].reshape(dynamic_memory.shape)
    memory_difference = float(np.max(np.abs(dynamic_memory - fixed_valid_memory)))
    router_difference = float(np.max(np.abs(dynamic_router - fixed_router)))
    if not np.array_equal(dynamic_selected, fixed_selected):
        raise ImportFailure("fixed question padding changes the selected family")

    decoder_ids = np.asarray([[1, 4, 5]], dtype=np.int64)
    padded_decoder_ids = np.zeros((1, target_length), dtype=np.int64)
    padded_decoder_ids[:, : decoder_ids.shape[1]] = decoder_ids
    try:
        del encoder
        decoder = ort.InferenceSession(
            str(paths["decoder"]),
            providers=["CPUExecutionProvider"],
        )
        dynamic_logits = decoder.run(
            None,
            {
                "decoder_input_ids": decoder_ids,
                "memory": dynamic_memory,
                "memory_padding_mask": dynamic_mask,
                "family_ids": dynamic_selected.astype(np.int64, copy=False),
            },
        )[0]
        fixed_logits = decoder.run(
            None,
            {
                "decoder_input_ids": padded_decoder_ids,
                "memory": fixed_memory,
                "memory_padding_mask": fixed_mask,
                "family_ids": fixed_selected.astype(np.int64, copy=False),
            },
        )[0]
    except Exception as error:
        raise ImportFailure(f"decoder fixed-padding parity execution failed: {error}") from error
    logits_difference = float(
        np.max(np.abs(dynamic_logits - fixed_logits[:, : decoder_ids.shape[1], :]))
    )
    float_outputs = (
        dynamic_memory,
        fixed_memory,
        dynamic_router,
        fixed_router,
        dynamic_logits,
        fixed_logits,
    )
    if any(not bool(np.all(np.isfinite(value))) for value in float_outputs) or any(
        not bool(np.isfinite(value))
        for value in (memory_difference, router_difference, logits_difference)
    ):
        raise ImportFailure("fixed-padding parity produced non-finite outputs or differences")
    tolerance = 1.0e-4
    if max(memory_difference, router_difference, logits_difference) > tolerance:
        raise ImportFailure(
            "fixed-padding parity exceeded tolerance "
            f"{tolerance}: memory={memory_difference}, router={router_difference}, "
            f"logits={logits_difference}"
        )
    if int(np.argmax(dynamic_logits[0, -1])) != int(
        np.argmax(fixed_logits[0, decoder_ids.shape[1] - 1])
    ):
        raise ImportFailure("fixed-padding parity changes greedy first-index ArgMax")
    return {
        "provider": "CPUExecutionProvider",
        "question_length": int(question.shape[1]),
        "target_length": int(decoder_ids.shape[1]),
        "memory_max_abs_difference": memory_difference,
        "router_max_abs_difference": router_difference,
        "decoder_valid_logits_max_abs_difference": logits_difference,
        "greedy_argmax": "matched",
        "tolerance": tolerance,
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


def _sanitize_export_report(path: Path, source_name: str) -> None:
    report = dict(_read_json(path, "staged exporter report"))
    source = report.get("source")
    if report.get("format") != "volvox-export-report/v1" or not isinstance(source, Mapping):
        raise ImportFailure("staged exporter report has the wrong format")
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
    target: str,
    weight_dtype: str,
    quant_mode: str,
    input_shapes: Mapping[str, str],
    input_dtypes: Mapping[str, str],
    output_dtypes: Mapping[str, str],
    output_names: Sequence[str],
    image_input: str | None = None,
    defer_static_qdq_layout_optimization: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(GENERIC_EXPORTER),
        "--model",
        str(model),
        "--out",
        str(output),
        "--target",
        target,
        "--weight-dtype",
        weight_dtype,
        "--quant-mode",
        quant_mode,
        "--report",
        str(report),
        "--report-format",
        "json",
    ]
    if defer_static_qdq_layout_optimization:
        command.append("--defer-static-qdq-layout-optimization")
    for name, shape in input_shapes.items():
        command.extend(["--input-shape", f"{name}={shape}"])
    for name, dtype in input_dtypes.items():
        command.extend(["--input-dtype", f"{name}={dtype}"])
    for name, dtype in output_dtypes.items():
        command.extend(["--output-dtype", f"{name}={dtype}"])
    for name in output_names:
        command.extend(["--output-name", name])
    if image_input is not None:
        command.extend(["--image-normalization", f"{image_input}=minus-one-one"])
    return command


def _semantic_input_map(
    graph: Mapping[str, Any],
    expected: Mapping[str, tuple[list[int], str]],
    label: str,
) -> dict[str, str]:
    raw_inputs = graph.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ImportFailure(f"staged {label} graph has no inputs object")
    result: dict[str, str] = {}
    for canonical, descriptor in raw_inputs.items():
        if not isinstance(descriptor, Mapping):
            raise ImportFailure(f"staged {label} input {canonical!r} is malformed")
        source_name = descriptor.get("source_name")
        if not isinstance(source_name, str) or source_name in result:
            raise ImportFailure(f"staged {label} graph has invalid source input names")
        result[source_name] = str(canonical)
        expected_descriptor = expected.get(source_name)
        if expected_descriptor is None or (
            descriptor.get("shape"),
            descriptor.get("dtype"),
        ) != expected_descriptor:
            raise ImportFailure(f"staged {label} input {source_name!r} has the wrong ABI")
    if set(result) != set(expected):
        raise ImportFailure(f"staged {label} graph input set is incomplete")
    return result


def _validate_staged_graph(
    directory: Path,
    *,
    label: str,
    expected_inputs: Mapping[str, tuple[list[int], str]],
    expected_outputs: Mapping[str, tuple[list[int], str]],
) -> tuple[Mapping[str, Any], dict[str, str]]:
    graph_path = directory / "graph.json"
    weights_path = directory / "model.safetensors"
    report_path = directory / "export_report.json"
    for path in (graph_path, weights_path, report_path):
        if not path.is_file() or path.is_symlink():
            raise ImportFailure(f"staged {label} asset {path.name!r} is missing")
    graph = _read_json(graph_path, f"staged {label} graph")
    if graph.get("format") != "volvox-graph/v1":
        raise ImportFailure(f"staged {label} graph has the wrong format")
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
        shapes = node.get("outputs_shape")
        dtypes = node.get("outputs_dtype")
        if not all(isinstance(item, Mapping) for item in (outputs, shapes, dtypes)):
            raise ImportFailure(f"staged {label} graph contains an untyped node")
        for port, tensor_name in outputs.items():
            tensor_descriptors[str(tensor_name)] = (shapes.get(port), dtypes.get(port))
    if graph.get("outputs") != list(expected_outputs):
        raise ImportFailure(f"staged {label} graph outputs are not semantic/stable")
    for name, descriptor in expected_outputs.items():
        if tensor_descriptors.get(name) != descriptor:
            raise ImportFailure(f"staged {label} output {name!r} has the wrong ABI")
    if weights_path.stat().st_size <= 0:
        raise ImportFailure(f"staged {label} weights are empty")
    return graph, semantic_inputs


_CANONICAL_BYTE_OPS = frozenset({
    "QConv2D",
    "QAdd",
    "QLinear",
    "QMatMul",
    "QGemm",
    "QBatchMatMul",
    "QEmbedding",
    "QLayerNorm",
    "QGroupNorm",
    "QMaskedMean",
    "QSDPA",
    "QGELU",
    "QSiLU",
    "QArgMax",
})
_BYTE_STRUCTURAL_OPS = frozenset({
    "Identity",
    "Reshape",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Concat",
    "MaxPool2D",
    "ResizeNearest2D",
    "RequantizeLinear",
})
_PACKAGE_CLASSES = frozenset({"fp32", "w8a32", "w8a8-v1", "hybrid"})


def _descriptor_package_class(graph: Mapping[str, Any]) -> str:
    """Mirror the generic exporter's descriptor-derived package classifier."""

    raw_nodes = graph.get("nodes")
    nodes = raw_nodes if isinstance(raw_nodes, list) else []
    node_ops = {
        str(node.get("opType"))
        for node in nodes
        if isinstance(node, Mapping) and isinstance(node.get("opType"), str)
    }
    execution_dtypes = {
        descriptor.get("dtype")
        for descriptor in (
            graph.get("inputs").values()
            if isinstance(graph.get("inputs"), Mapping)
            else ()
        )
        if isinstance(descriptor, Mapping)
    }
    for node in nodes:
        if isinstance(node, Mapping) and isinstance(
            node.get("outputs_dtype"), Mapping
        ):
            execution_dtypes.update(node["outputs_dtype"].values())

    byte_execution = bool(execution_dtypes & {"int8", "uint8"})
    float_execution = "float32" in execution_dtypes
    all_byte_graph_ops = bool(node_ops & _CANONICAL_BYTE_OPS) and node_ops <= (
        _CANONICAL_BYTE_OPS | _BYTE_STRUCTURAL_OPS
    )
    if byte_execution and all_byte_graph_ops and not float_execution:
        return "w8a8-v1"
    if byte_execution or bool(node_ops & _CANONICAL_BYTE_OPS):
        return "hybrid"
    return "fp32"


def _staged_package_class(graph: Mapping[str, Any], label: str) -> str:
    actual = _descriptor_package_class(graph)
    source = graph.get("source")
    declared = source.get("package_class") if isinstance(source, Mapping) else None
    if declared is not None and declared not in _PACKAGE_CLASSES:
        raise ImportFailure(
            f"staged {label} graph declares an unknown package class"
        )
    if declared is not None and declared != actual:
        raise ImportFailure(
            f"staged {label} graph package class does not match its live descriptors"
        )
    return actual


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


def import_package(
    source_directory: Path,
    output_directory: Path,
    *,
    variant: str = "fp32",
    target: str = "portable",
    weight_dtype: str = "auto",
    exporter: Callable[[Sequence[str]], None] = _run_exporter,
    parity_verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]] = verify_static_padding,
) -> Path:
    source_directory = source_directory.resolve(strict=True)
    output_parent = output_directory.parent.resolve(strict=True)
    output_directory = output_parent / output_directory.name
    if os.path.lexists(output_directory):
        raise ImportFailure("output directory already exists; refusing to overwrite it")
    if source_directory == output_directory or source_directory in output_directory.parents:
        raise ImportFailure("output directory must not be inside the immutable source")
    source = validate_source(source_directory, variant=variant)
    source_hashes_before = dict(source["hashes"])
    parity = dict(parity_verifier(source))
    original_selected_paths = source["selected_paths"]
    selected_paths = dict(original_selected_paths)
    quant_mode = "preserve"

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.stage-",
            dir=output_parent,
        )
    )
    try:
        encoder_directory = stage / "encoder"
        decoder_directory = stage / "decoder"
        encoder_directory.mkdir()
        decoder_directory.mkdir()
        shutil.copyfile(source["paths"]["config"], stage / "config.json")
        shutil.copyfile(source["paths"]["vocab"], stage / "vocab.json")

        exporter(_export_command(
            model=selected_paths["encoder"],
            output=encoder_directory / "model.safetensors",
            report=encoder_directory / "export_report.json",
            target=target,
            weight_dtype=weight_dtype,
            quant_mode=quant_mode,
            input_shapes={
                "image": "1x1x320x672",
                "question_ids": "1x192",
                "family_ids": "1",
            },
            input_dtypes={
                "question_ids": "int32",
                "family_ids": "int32",
            },
            output_dtypes={
                "memory_padding_mask": "int32",
                "selected_family_ids": "int32",
            },
            output_names=(
                "memory",
                "memory_padding_mask",
                "router_logits",
                "selected_family_ids",
            ),
            image_input="input0",
            defer_static_qdq_layout_optimization=(variant == "int8-w8a8"),
        ))
        exporter(_export_command(
            model=selected_paths["decoder"],
            output=decoder_directory / "model.safetensors",
            report=decoder_directory / "export_report.json",
            target=target,
            weight_dtype=weight_dtype,
            quant_mode=quant_mode,
            input_shapes={
                "decoder_input_ids": "1x192",
                "memory": "1x402x320",
                "memory_padding_mask": "1x402",
                "family_ids": "1",
            },
            input_dtypes={
                "decoder_input_ids": "int32",
                "memory_padding_mask": "int32",
                "family_ids": "int32",
            },
            output_dtypes={},
            output_names=("logits",),
            defer_static_qdq_layout_optimization=(variant == "int8-w8a8"),
        ))
        _sanitize_export_report(
            encoder_directory / "export_report.json",
            original_selected_paths["encoder"].name,
        )
        _sanitize_export_report(
            decoder_directory / "export_report.json",
            original_selected_paths["decoder"].name,
        )

        encoder_graph, encoder_inputs = _validate_staged_graph(
            encoder_directory,
            label="encoder",
            expected_inputs={
                "image": ([1, 1, 320, 672], "float32"),
                "question_ids": ([1, 192], "int32"),
                "family_ids": ([1], "int32"),
            },
            expected_outputs={
                "memory": ([1, 402, 320], "float32"),
                "memory_padding_mask": ([1, 402], "int32"),
                "router_logits": ([1, 8], "float32"),
                "selected_family_ids": ([1], "int32"),
            },
        )
        decoder_graph, decoder_inputs = _validate_staged_graph(
            decoder_directory,
            label="decoder",
            expected_inputs={
                "decoder_input_ids": ([1, 192], "int32"),
                "memory": ([1, 402, 320], "float32"),
                "memory_padding_mask": ([1, 402], "int32"),
                "family_ids": ([1], "int32"),
            },
            expected_outputs={
                "logits": ([1, 192, source["vocab_size"]], "float32")
            },
        )
        staged_package_classes = {
            "encoder": _staged_package_class(encoder_graph, "encoder"),
            "decoder": _staged_package_class(decoder_graph, "decoder"),
        }
        if variant == "int8-w8a8" and any(
            package_class not in {"hybrid", "w8a8-v1"}
            for package_class in staged_package_classes.values()
        ):
            raise ImportFailure(
                "generic exporter did not preserve the selected INT8 QDQ graph"
            )
        complete_w8a8_fusion = all(
            package_class == "w8a8-v1"
            for package_class in staged_package_classes.values()
        )
        for label, document in (
            ("encoder graph", encoder_graph),
            ("decoder graph", decoder_graph),
            (
                "encoder export report",
                _read_json(
                    encoder_directory / "export_report.json",
                    "encoder export report",
                ),
            ),
            (
                "decoder export report",
                _read_json(
                    decoder_directory / "export_report.json",
                    "decoder export report",
                ),
            ),
        ):
            if _contains_private_absolute_path(document):
                raise ImportFailure(f"staged {label} contains a private absolute path")
        if {
            key: _sha256(path) for key, path in source["paths"].items()
        } != source_hashes_before:
            raise ImportFailure("source files changed while the package was being imported")

        variant_manifest: dict[str, Any] = {
            "requested": variant,
            "producer_key": MANIFEST_VARIANT_KEYS[variant],
            "export_quant_mode": quant_mode,
            "compiled_graph_package_class": staged_package_classes,
            "complete_w8a8_fusion": complete_w8a8_fusion,
        }
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
            "variant": variant_manifest,
            "assets": {
                "config": _asset_record(stage / "config.json", "config.json"),
                "vocab": _asset_record(stage / "vocab.json", "vocab.json"),
            },
            "tokenizer": dict(source["tokenizer"]),
            "preprocessing": {
                "layout": "NCHW",
                "shape": [1, 1, 320, 672],
                "color": "grayscale",
                "resize": {"width": 672, "height": 320, "method": "bilinear"},
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
                "strategy": "greedy-autoregressive",
                "decoder_input_length": 192,
                "maximum_new_tokens": 191,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "logits_row": "prefix_length_minus_one",
                "tie_policy": "first-index",
            },
            "graphs": {
                "encoder": {
                    "graph": _asset_record(
                        encoder_directory / "graph.json",
                        "encoder/graph.json",
                    ),
                    "weights": _asset_record(
                        encoder_directory / "model.safetensors",
                        "encoder/model.safetensors",
                    ),
                    "export_report": _asset_record(
                        encoder_directory / "export_report.json",
                        "encoder/export_report.json",
                    ),
                    "inputs": encoder_inputs,
                    "outputs": {
                        "memory": "memory",
                        "memory_padding_mask": "memory_padding_mask",
                        "router_logits": "router_logits",
                        "selected_family_ids": "selected_family_ids",
                    },
                },
                "decoder": {
                    "graph": _asset_record(
                        decoder_directory / "graph.json",
                        "decoder/graph.json",
                    ),
                    "weights": _asset_record(
                        decoder_directory / "model.safetensors",
                        "decoder/model.safetensors",
                    ),
                    "export_report": _asset_record(
                        decoder_directory / "export_report.json",
                        "decoder/export_report.json",
                    ),
                    "inputs": decoder_inputs,
                    "outputs": {"logits": "logits"},
                },
            },
            "mask_semantics": {
                "memory_padding_mask": "nonzero_means_blocked",
            },
            "validation": {
                "onnx_checker": "passed",
                "fixed_padding_parity": parity,
                "offline_target": target,
                "source_variant": variant,
                "strict_runtime_execution": "not-run",
            },
        }
        if _contains_private_absolute_path(manifest):
            raise ImportFailure("package manifest contains a private absolute path")
        (stage / "package_manifest.json").write_text(
            json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
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
        default="portable",
        choices=("portable", "browser", "cpu-js", "wasm", "webgpu", "native-cpu"),
    )
    parser.add_argument(
        "--weight-dtype",
        default="auto",
        choices=("auto", "float32", "float16"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        imported = import_package(
            args.source,
            args.out_dir,
            variant=args.variant,
            target=args.target,
            weight_dtype=args.weight_dtype,
        )
    except (ImportFailure, OSError, ValueError) as error:
        print(f"[TinyReceipt split ONNX import] {error}", file=sys.stderr)
        return 1
    print(imported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
