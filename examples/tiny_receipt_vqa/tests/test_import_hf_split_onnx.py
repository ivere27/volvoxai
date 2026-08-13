from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = EXAMPLE_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as error:  # pragma: no cover - environment dependency.
    raise unittest.SkipTest(f"TinyReceipt split importer dependencies unavailable: {error}")


importer = importlib.import_module("import_hf_split_onnx")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_model_record(source: Path, key: str) -> None:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = source / manifest["files"][key]
    manifest["files"][f"{key}_bytes"] = path.stat().st_size
    manifest["files"][f"{key}_sha256"] = _sha256(path)
    variant_key = "int8_w8a8" if key.endswith("_int8_w8a8") else "fp32"
    role = key.split("_", 1)[0]
    variant = manifest["variants"].get(variant_key)
    if variant is not None:
        variant[f"{role}_bytes"] = path.stat().st_size
        variant[f"{role}_sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _value(name: str, dtype: int, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def _constant(name: str, output: str, value: np.ndarray):
    return helper.make_node(
        "Constant",
        [],
        [output],
        name=name,
        value=numpy_helper.from_array(value),
    )


def _write_source(root: Path) -> Path:
    root.mkdir()
    encoder = helper.make_model(
        helper.make_graph(
            [
                _constant("memory_value", "memory", np.zeros((1, 1, 320), np.float32)),
                _constant("mask_value", "memory_padding_mask", np.zeros((1, 1), np.bool_)),
                _constant("router_value", "router_logits", np.zeros((1, 8), np.float32)),
                _constant("family_value", "selected_family_ids", np.zeros((1,), np.int64)),
            ],
            "encoder",
            [
                _value("image", TensorProto.FLOAT, ["batch", 1, 320, 672]),
                _value("question_ids", TensorProto.INT64, ["batch", "question_length"]),
                _value("family_ids", TensorProto.INT64, ["batch"]),
            ],
            [
                _value(
                    "memory",
                    TensorProto.FLOAT,
                    ["batch", "question_length + 210", 320],
                ),
                _value(
                    "memory_padding_mask",
                    TensorProto.BOOL,
                    ["batch", "question_length + 210"],
                ),
                _value("router_logits", TensorProto.FLOAT, ["batch", 8]),
                _value("selected_family_ids", TensorProto.INT64, ["batch"]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    decoder = helper.make_model(
        helper.make_graph(
            [
                _constant("logits_value", "logits", np.zeros((1, 1, 760), np.float32)),
            ],
            "decoder",
            [
                _value(
                    "decoder_input_ids",
                    TensorProto.INT64,
                    ["batch", "target_length"],
                ),
                _value("memory", TensorProto.FLOAT, ["batch", "memory_length", 320]),
                _value(
                    "memory_padding_mask",
                    TensorProto.BOOL,
                    ["batch", "memory_length"],
                ),
                _value("family_ids", TensorProto.INT64, ["batch"]),
            ],
            [
                _value(
                    "logits",
                    TensorProto.FLOAT,
                    ["batch", "target_length", 760],
                ),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    encoder_path = root / "encoder_model.onnx"
    decoder_path = root / "decoder_model.onnx"
    onnx.save(encoder, encoder_path)
    onnx.save(decoder, decoder_path)
    config = {
        "vocab_size": 760,
        "d_model": 320,
        "heads": 8,
        "enc_layers": 6,
        "dec_layers": 4,
        "ff_mult": 4,
        "dropout": 0.1,
        "max_q_len": 192,
        "max_out_len": 192,
        "img_tokens": 210,
        "use_adapters": True,
        "use_router": True,
        "adapter_bottleneck": 64,
        "adapter_families": 8,
    }
    vocab = {
        "itos": ["<pad>", "<bos>", "<eos>", "<unk>"]
        + [f"token-{index}" for index in range(4, 760)]
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    manifest = {
        "format": importer.SOURCE_FORMAT,
        "source_format": "safetensors",
        "opset": 18,
        "files": {
            "encoder": "encoder_model.onnx",
            "decoder": "decoder_model.onnx",
            "config": "config.json",
            "vocab": "vocab.json",
            "encoder_sha256": _sha256(encoder_path),
            "encoder_bytes": encoder_path.stat().st_size,
            "decoder_sha256": _sha256(decoder_path),
            "decoder_bytes": decoder_path.stat().st_size,
        },
        "outputs": {
            "encoder": [
                "memory",
                "memory_padding_mask",
                "router_logits",
                "selected_family_ids",
            ],
            "decoder": ["logits"],
        },
        "adapter_families": {
            "ordered_names": list(importer.FAMILY_ORDER),
            "name_to_id": {
                name: index for index, name in enumerate(importer.FAMILY_ORDER)
            },
        },
        "generation": {
            "strategy": "greedy autoregressive",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "max_length": 192,
        },
        "exporter": {"max_batch_size": 8},
        "variants": {
            "fp32": {
                "activation_dtype": "float32",
                "compute_dtype": "float32",
                "weight_dtype": "float32",
                "encoder": "encoder_model.onnx",
                "encoder_bytes": encoder_path.stat().st_size,
                "encoder_sha256": _sha256(encoder_path),
                "decoder": "decoder_model.onnx",
                "decoder_bytes": decoder_path.stat().st_size,
                "decoder_sha256": _sha256(decoder_path),
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _upgrade_source_to_bpe(root)
    return root


def _upgrade_source_to_kv(source: Path) -> Path:
    encoder_path = source / "encoder_model.onnx"
    encoder = onnx.load(encoder_path)
    for name in importer.KV_CROSS_NAMES:
        encoder.graph.node.append(_constant(
            f"{name}_value",
            name,
            np.zeros((1, importer.KV_HEADS, 1, importer.KV_HEAD_WIDTH), np.float32),
        ))
        encoder.graph.output.append(_value(
            name,
            TensorProto.FLOAT,
            [
                "batch",
                f"Transpose{name}_dim_1",
                "memory_length",
                f"Transpose{name}_dim_3",
            ],
        ))
    encoder.ir_version = 11
    onnx.save(encoder, encoder_path)

    decoder_inputs = [
        _value("decoder_input_ids", TensorProto.INT64, ["batch", 1]),
        _value("position_ids", TensorProto.INT64, ["batch"]),
        _value("family_ids", TensorProto.INT64, ["batch"]),
        _value("memory_padding_mask", TensorProto.BOOL, ["batch", "memory_length"]),
        _value("past_padding_mask", TensorProto.BOOL, ["batch", "past_length"]),
    ]
    decoder_inputs.extend(
        _value(name, TensorProto.FLOAT,
               ["batch", importer.KV_HEADS, "memory_length", importer.KV_HEAD_WIDTH])
        for name in importer.KV_CROSS_NAMES
    )
    decoder_inputs.extend(
        _value(name, TensorProto.FLOAT,
               ["batch", importer.KV_HEADS, "past_length", importer.KV_HEAD_WIDTH])
        for name in importer.KV_PAST_NAMES
    )
    nodes = [
        helper.make_node(
            "Cast", ["decoder_input_ids"], ["decoder_token_f32"],
            name="decoder_token_f32_value", to=TensorProto.FLOAT,
        ),
        _constant("logits_axis_value", "logits_axis", np.asarray([2], np.int64)),
        helper.make_node(
            "Unsqueeze", ["decoder_token_f32", "logits_axis"], ["decoder_token_row"],
            name="decoder_token_row_value",
        ),
        _constant(
            "logits_repeats_value", "logits_repeats",
            np.asarray([1, 1, importer.BPE_VOCAB_SIZE], np.int64),
        ),
        helper.make_node(
            "Tile", ["decoder_token_row", "logits_repeats"], ["logits"],
            name="logits_value",
        ),
        _constant(
            "cache_axis_value", "cache_axis", np.asarray([3], np.int64),
        ),
        helper.make_node(
            "Unsqueeze", ["decoder_token_row", "cache_axis"], ["current_cache_scalar"],
            name="current_cache_scalar_value",
        ),
        _constant(
            "cache_repeats_value", "cache_repeats",
            np.asarray([1, importer.KV_HEADS, 1, importer.KV_HEAD_WIDTH], np.int64),
        ),
        helper.make_node(
            "Tile", ["current_cache_scalar", "cache_repeats"], ["current_cache_base"],
            name="current_cache_base_value",
        ),
        _constant("pad_value", "pad_value", np.asarray(0, np.int64)),
        helper.make_node(
            "Equal", ["decoder_input_ids", "pad_value"], ["current_padding_mask"],
            name="current_padding_mask_value",
        ),
        helper.make_node(
            "Concat", ["past_padding_mask", "current_padding_mask"],
            ["present_padding_mask"], name="present_padding_mask_concat", axis=1,
        ),
    ]
    for index, (past_name, present_name) in enumerate(zip(
        importer.KV_PAST_NAMES, importer.KV_PRESENT_NAMES
    )):
        current = f"current_cache_{index}"
        nodes.extend([
            helper.make_node(
                "Identity", ["current_cache_base"], [current],
                name=f"{current}_value",
            ),
            helper.make_node(
                "Concat", [past_name, current], [present_name],
                name=f"{present_name}_concat", axis=2,
            ),
        ])
    decoder_outputs = [
        _value("logits", TensorProto.FLOAT, ["batch", 1, importer.BPE_VOCAB_SIZE]),
        _value("present_padding_mask", TensorProto.BOOL, ["batch", "present_length"]),
    ]
    decoder_outputs.extend(
        _value(name, TensorProto.FLOAT,
               ["batch", importer.KV_HEADS, "present_length", importer.KV_HEAD_WIDTH])
        for name in importer.KV_PRESENT_NAMES
    )
    decoder = helper.make_model(
        helper.make_graph(nodes, "kv-decoder", decoder_inputs, decoder_outputs),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    # The test environment's ONNX Runtime supports IR versions through 11.
    decoder.ir_version = 11
    decoder_path = source / "decoder_model.onnx"
    onnx.save(decoder, decoder_path)

    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = importer.SOURCE_FORMAT
    manifest["kv_cache"] = {
        "format": "tiny_receipt_vqa_default_kv_cache_v2",
        "default_for": ["fp32", "int8_w8a8"],
    }
    manifest["generation"]["strategy"] = (
        "greedy autoregressive one-token decoding with per-layer KV cache"
    )
    manifest["outputs"] = {
        "encoder": [
            "memory", "memory_padding_mask", "router_logits",
            "selected_family_ids", *importer.KV_CROSS_NAMES,
        ],
        "decoder": ["logits", "present_padding_mask", *importer.KV_PRESENT_NAMES],
    }
    for key, path in (("encoder", encoder_path), ("decoder", decoder_path)):
        manifest["files"][f"{key}_bytes"] = path.stat().st_size
        manifest["files"][f"{key}_sha256"] = _sha256(path)
        manifest["variants"]["fp32"][f"{key}_bytes"] = path.stat().st_size
        manifest["variants"]["fp32"][f"{key}_sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source


def _set_decoder_vocab_size(source: Path, vocab_size: int) -> None:
    decoder_path = source / "decoder_model.onnx"
    decoder = onnx.load(decoder_path)
    decoder.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = vocab_size
    constant = next(
        node for node in decoder.graph.node if node.name == "logits_value"
    )
    value = next(
        attribute for attribute in constant.attribute if attribute.name == "value"
    )
    value.t.CopyFrom(
        numpy_helper.from_array(np.zeros((1, 1, vocab_size), np.float32))
    )
    onnx.save(decoder, decoder_path)
    _refresh_model_record(source, "decoder")


def _upgrade_source_to_bpe(source: Path) -> dict:
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["vocab_size"] = importer.BPE_VOCAB_SIZE
    config_path.write_text(json.dumps(config), encoding="utf-8")

    itos = [
        *importer.SPECIAL_TOKENS,
        *importer.BPE_ATOMIC_TOKENS,
        *importer.BPE_BYTE_TOKENS,
        "a",
        "b",
        "ab",
    ]
    itos.extend(
        f"piece-{index}"
        for index in range(importer.BPE_VOCAB_SIZE - len(itos))
    )
    vocab = {
        "type": "byte_fallback_bpe",
        "version": 1,
        "vocab_size": importer.BPE_VOCAB_SIZE,
        "itos": itos,
        "merges": [["a", "b"]],
        "normalization": "NFC",
        "atomic_tokens": list(importer.BPE_ATOMIC_TOKENS),
        "byte_tokens": list(importer.BPE_BYTE_TOKENS),
        "unused_tokens": [],
        "special_tokens": {
            "pad": "<pad>",
            "bos": "<bos>",
            "eos": "<eos>",
            "unk": "<unk>",
        },
    }
    vocab["tokenizer_hash"] = hashlib.sha256(
        json.dumps(
            vocab,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (source / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

    _set_decoder_vocab_size(source, importer.BPE_VOCAB_SIZE)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"] = {
        "type": "byte_fallback_bpe",
        "version": 1,
        "vocab_size": importer.BPE_VOCAB_SIZE,
        "normalization": "NFC",
        "tokenizer_hash": vocab["tokenizer_hash"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return vocab


def _append_static_u8s8_qdq(model, prefix: str) -> None:
    def add_compute(
        op_type: str,
        activation: np.ndarray,
        weight: np.ndarray,
        *,
        weight_axis: int,
        **attributes,
    ) -> None:
        stem = f"{prefix}_{op_type.lower()}"
        activation_name = f"{stem}_activation"
        activation_scale = f"{stem}_activation_scale"
        activation_zero = f"{stem}_activation_zero"
        quantized_activation = f"{stem}_activation_u8"
        dequantized_activation = f"{stem}_activation_f32"
        weight_name = f"{stem}_weight_s8"
        weight_scale = f"{stem}_weight_scale"
        weight_zero = f"{stem}_weight_zero"
        dequantized_weight = f"{stem}_weight_f32"
        output = f"{stem}_output"
        channel_count = int(weight.shape[weight_axis])
        model.graph.initializer.extend([
            numpy_helper.from_array(activation, activation_name),
            numpy_helper.from_array(np.asarray(0.125, np.float32), activation_scale),
            numpy_helper.from_array(np.asarray(127, np.uint8), activation_zero),
            numpy_helper.from_array(weight, weight_name),
            numpy_helper.from_array(
                np.full((channel_count,), 0.25, np.float32),
                weight_scale,
            ),
            numpy_helper.from_array(
                np.zeros((channel_count,), np.int8),
                weight_zero,
            ),
        ])
        model.graph.node.extend([
            helper.make_node(
                "QuantizeLinear",
                [activation_name, activation_scale, activation_zero],
                [quantized_activation],
                name=f"{stem}_quantize",
            ),
            helper.make_node(
                "DequantizeLinear",
                [quantized_activation, activation_scale, activation_zero],
                [dequantized_activation],
                name=f"{stem}_activation_dequantize",
            ),
            helper.make_node(
                "DequantizeLinear",
                [weight_name, weight_scale, weight_zero],
                [dequantized_weight],
                name=f"{stem}_weight_dequantize",
                axis=weight_axis,
            ),
            helper.make_node(
                op_type,
                [dequantized_activation, dequantized_weight],
                [output],
                name=f"{stem}_compute",
                **attributes,
            ),
        ])

    add_compute(
        "Conv",
        np.ones((1, 1, 2, 2), np.float32),
        np.ones((1, 1, 1, 1), np.int8),
        weight_axis=0,
    )
    add_compute(
        "MatMul",
        np.ones((1, 2), np.float32),
        np.ones((2, 3), np.int8),
        weight_axis=1,
    )
    add_compute(
        "Gemm",
        np.ones((1, 2), np.float32),
        np.ones((3, 2), np.int8),
        weight_axis=0,
        transB=1,
    )


def _add_int8_w8a8_variant(source: Path) -> dict[str, Path]:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    for key, filename in importer.INT8_W8A8_SOURCE_FILES.items():
        base_key = key.split("_", 1)[0]
        path = source / filename
        model = onnx.load(source / importer.SOURCE_FILES[base_key])
        if base_key == "encoder":
            for output in model.graph.output:
                if output.name not in importer.KV_CROSS_NAMES:
                    continue
                dims = output.type.tensor_type.shape.dim
                dims[1].dim_param = ""
                dims[1].dim_value = importer.KV_HEADS
                dims[3].dim_param = ""
                dims[3].dim_value = importer.KV_HEAD_WIDTH
        _append_static_u8s8_qdq(model, base_key)
        onnx.save(model, path)
        manifest["files"][key] = filename
        manifest["files"][f"{key}_bytes"] = path.stat().st_size
        manifest["files"][f"{key}_sha256"] = _sha256(path)
        paths[key] = path
    optimized_graph = {
        "encoder": {
            "nodes": {
                "QuantizeLinear": 3,
                "DequantizeLinear": 6,
                "QLinearConv": 1,
                "QLinearMatMul": 1,
                "QGemm": 1,
            },
            "initializers": {"FLOAT": 1, "UINT8": 1, "INT8": 1, "INT32": 1},
        },
        "decoder": {
            "nodes": {
                "QuantizeLinear": 3,
                "DequantizeLinear": 6,
                "QLinearMatMul": 1,
                "QGemm": 1,
            },
            "initializers": {"FLOAT": 1, "UINT8": 1, "INT8": 1, "INT32": 1},
        },
    }
    manifest["variants"]["int8_w8a8"] = {
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
        "encoder": paths["encoder_int8_w8a8"].name,
        "encoder_bytes": paths["encoder_int8_w8a8"].stat().st_size,
        "encoder_sha256": _sha256(paths["encoder_int8_w8a8"]),
        "decoder": paths["decoder_int8_w8a8"].name,
        "decoder_bytes": paths["decoder_int8_w8a8"].stat().st_size,
        "decoder_sha256": _sha256(paths["decoder_int8_w8a8"]),
        "calibration": {
            "source": "synthetic_training_split",
            "seed": 71,
            "records": 6,
            "receipt_limit": 6,
            "prefixes_per_record": 3,
            "decoder_feeds": 18,
            "heldout_used": False,
            "family_counts": {
                name: 1 for name in importer.FAMILY_ORDER[:6]
            },
        },
        "validation": {
            "onnx_checker": "passed",
            "heldout": {"status": "not_included"},
            "optimized_graph": optimized_graph,
            "runtime": {
                "dynamic_prefix_lengths": [1, 39, 77],
                "family_ids": [-1, *range(8)],
                "onnxruntime": "passed",
                "onnxruntime_version": "fixture",
                "provider": "CPUExecutionProvider",
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return paths


class FakeExporter:
    def __init__(
        self,
        fail_on_call: int | None = None,
        *,
        vocab_size: int = importer.BPE_VOCAB_SIZE,
    ):
        self.commands: list[list[str]] = []
        self.model_signatures: list[dict[str, dict[str, tuple[int, list]]]] = []
        self.fail_on_call = fail_on_call
        self.vocab_size = vocab_size

    @staticmethod
    def _option(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    @staticmethod
    def _options(command: list[str], name: str) -> list[str]:
        return [
            command[index + 1]
            for index, value in enumerate(command)
            if value == name
        ]

    def __call__(self, raw_command) -> None:
        command = list(raw_command)
        self.commands.append(command)
        if self.fail_on_call == len(self.commands):
            raise importer.ImportFailure("injected exporter failure")
        output = Path(self._option(command, "--out"))
        report = Path(self._option(command, "--report"))
        authored_model = onnx.load(self._option(command, "--model"), load_external_data=False)
        self.model_signatures.append({
            "inputs": {
                value.name: importer._tensor_signature(value)
                for value in authored_model.graph.input
            },
            "outputs": {
                value.name: importer._tensor_signature(value)
                for value in authored_model.graph.output
            },
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake-safetensors")
        requested_targets = self._options(command, "--target")
        report.write_text(json.dumps({
            "format": "volvox-export-report/v1",
            "supported": True,
            "published": True,
            "targets": {
                "requested": requested_targets,
                "resolved": list(importer.expand_targets(requested_targets)),
            },
            "source": {
                "path": self._option(command, "--model"),
                "format": "onnx",
            },
        }), encoding="utf-8")
        if output.parent.name == "encoder":
            inputs = {
                "input0": {
                    "shape": ["B", 1, 320, 672],
                    "dtype": "float32",
                },
                "input1": {
                    "shape": ["B", "Q"],
                    "dtype": "int32",
                },
                "input2": {
                    "shape": ["B"],
                    "dtype": "int32",
                },
                "input3": {
                    "shape": ["B", "Q"],
                    "dtype": "int32",
                },
            }
            outputs = {
                "memory": (["B", "M", 320], "float32"),
                "memory_padding_mask": (["B", "M"], "int32"),
                "router_logits": (["B", 8], "float32"),
                "selected_family_ids": (["B"], "int32"),
            }
            dimensions = {
                "B": {"min": 1, "max": 1},
                "Q": {"min": 1, "max": 192},
                "M": {"min": 211, "max": 402},
            }
        else:
            inputs = {
                "input0": {
                    "shape": ["B", "T"],
                    "dtype": "int32",
                },
                "input1": {
                    "shape": ["B", "M", 320],
                    "dtype": "float32",
                },
                "input2": {
                    "shape": ["B", "M"],
                    "dtype": "int32",
                },
                "input3": {
                    "shape": ["B"],
                    "dtype": "int32",
                },
                "input4": {
                    "shape": ["B", "T"],
                    "dtype": "int32",
                },
                "input5": {
                    "shape": ["T", "T"],
                    "dtype": "float32",
                },
            }
            outputs = {"logits": (["B", "T", self.vocab_size], "float32")}
            dimensions = {
                "B": {"min": 1, "max": 1},
                "T": {"min": 1, "max": 192},
                "M": {"min": 211, "max": 402},
            }
        nodes = []
        if "_int8" in Path(self._option(command, "--model")).stem:
            first_input = next(iter(inputs))
            nodes.append({
                "id": "preserved-quantize",
                "opType": "QuantizeLinear",
                "inputs": {
                    "input": first_input,
                    "scale": "fixture_scale",
                    "zero_point": "fixture_zero_point",
                },
                "outputs": {"out": {
                    "tensor": "fixture_u8",
                    "shape": inputs[first_input]["shape"],
                    "dtype": "uint8",
                }},
                "params": {},
            })
        if output.parent.name == "encoder":
            nodes.extend([
                {
                    "id": "fixture-image-tokens",
                    "opType": "Identity",
                    "inputs": {"input": "input0"},
                    "outputs": {"out": {
                        "tensor": "fixture_image_tokens",
                        "shape": ["B", 210, 320],
                        "dtype": "float32",
                    }},
                    "params": {},
                },
                {
                    "id": "fixture-question-tokens",
                    "opType": "Identity",
                    "inputs": {"input": "input1"},
                    "outputs": {"out": {
                        "tensor": "fixture_question_tokens",
                        "shape": ["B", "Q", 320],
                        "dtype": "float32",
                    }},
                    "params": {},
                },
                {
                    "id": "fixture-memory-concat",
                    "opType": "Concat",
                    "inputs": {
                        "input0": "fixture_image_tokens",
                        "input1": "fixture_question_tokens",
                    },
                    "outputs": {"out": {
                        "tensor": "fixture_encoded_sequence",
                        "shape": ["B", "M", 320],
                        "dtype": "float32",
                    }},
                    "params": {"axis": 1},
                },
            ])
        for index, (name, (shape, dtype)) in enumerate(outputs.items()):
            source = (
                "fixture_encoded_sequence"
                if output.parent.name == "encoder" and name == "memory"
                else next(iter(inputs))
            )
            nodes.append({
                "id": f"output-{index}",
                "opType": "Identity",
                "inputs": {"input": source},
                "outputs": {"out": {
                    "tensor": name,
                    "shape": shape,
                    "dtype": dtype,
                }},
                "params": {},
            })
        graph = {
            "format": "volvox-graph/v1",
            "dimensions": dimensions,
            "inputs": inputs,
            "nodes": nodes,
            "outputs": list(outputs),
        }
        (output.parent / "graph.json").write_text(
            json.dumps(graph),
            encoding="utf-8",
        )


class FakeKVExporter(FakeExporter):
    def __call__(self, raw_command) -> None:
        command = list(raw_command)
        self.commands.append(command)
        if self.fail_on_call == len(self.commands):
            raise importer.ImportFailure("injected exporter failure")
        output_path = Path(self._option(command, "--out"))
        report_path = Path(self._option(command, "--report"))
        authored_model = onnx.load(
            self._option(command, "--model"), load_external_data=False
        )
        self.model_signatures.append({
            "inputs": {
                value.name: importer._tensor_signature(value)
                for value in authored_model.graph.input
            },
            "outputs": {
                value.name: importer._tensor_signature(value)
                for value in authored_model.graph.output
            },
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-kv-safetensors")
        requested_targets = self._options(command, "--target")
        report_path.write_text(json.dumps({
            "format": "volvox-export-report/v1",
            "supported": True,
            "published": True,
            "targets": {
                "requested": requested_targets,
                "resolved": list(importer.expand_targets(requested_targets)),
            },
            "source": {"path": self._option(command, "--model"), "format": "onnx"},
        }), encoding="utf-8")
        batch_bound = next(
            value for value in self._options(command, "--dimension-bound")
            if value.startswith("B=")
        )
        batch_maximum = int(batch_bound.split("=", 1)[1].split(":")[1])

        nodes = []
        if output_path.parent.name == "encoder":
            inputs = {
                "input0": {"shape": ["B", 1, 320, 672], "dtype": "float32"},
                "input1": {"shape": ["B", "Q"], "dtype": "int32"},
                "input2": {"shape": ["B"], "dtype": "int32"},
                "input3": {"shape": ["B", "Q"], "dtype": "int32"},
            }
            dimensions = {
                "B": {"min": 1, "max": batch_maximum},
                "Q": {"min": 1, "max": 192},
                "M": {"min": 211, "max": 402},
            }
            outputs = {
                "memory": (["B", "M", 320], "float32"),
                "memory_padding_mask": (["B", "M"], "int32"),
                "router_logits": (["B", 8], "float32"),
                "selected_family_ids": (["B"], "int32"),
                **{
                    name: (["B", 8, "M", 40], "float32")
                    for name in importer.KV_CROSS_NAMES
                },
            }
            nodes.extend([
                {
                    "id": "image-tokens", "opType": "Identity",
                    "inputs": {"input": "input0"},
                    "outputs": {"out": {"tensor": "image_tokens", "shape": ["B", 210, 320], "dtype": "float32"}},
                    "params": {},
                },
                {
                    "id": "question-tokens", "opType": "Identity",
                    "inputs": {"input": "input1"},
                    "outputs": {"out": {"tensor": "question_tokens", "shape": ["B", "Q", 320], "dtype": "float32"}},
                    "params": {},
                },
                {
                    "id": "memory-concat", "opType": "Concat",
                    "inputs": {"input0": "image_tokens", "input1": "question_tokens"},
                    "outputs": {"out": {"tensor": "memory", "shape": ["B", "M", 320], "dtype": "float32"}},
                    "params": {"axis": 1},
                },
            ])
            for index, (name, (shape, dtype)) in enumerate(outputs.items()):
                if name == "memory":
                    continue
                nodes.append({
                    "id": f"encoder-output-{index}", "opType": "Identity",
                    "inputs": {"input": "input0"},
                    "outputs": {"out": {"tensor": name, "shape": shape, "dtype": dtype}},
                    "params": {},
                })
        else:
            inputs = {
                "input0": {"shape": ["B", 1], "dtype": "int32"},
                "input1": {"shape": ["B"], "dtype": "int32"},
                "input2": {"shape": ["B"], "dtype": "int32"},
                "input3": {"shape": ["B", "M"], "dtype": "int32"},
                "input4": {"shape": ["B", "P"], "dtype": "int32"},
            }
            for index, _name in enumerate(importer.KV_CROSS_NAMES):
                inputs[f"input{index + 5}"] = {
                    "shape": ["B", 8, "M", 40], "dtype": "float32",
                }
            for index, _name in enumerate(importer.KV_PAST_NAMES):
                inputs[f"input{index + 13}"] = {
                    "shape": ["B", 8, "P", 40], "dtype": "float32",
                }
            dimensions = {
                "B": {"min": 1, "max": batch_maximum},
                "M": {"min": 211, "max": 402},
                "P": {"min": 1, "max": 191},
                "R": {"min": 2, "max": 192},
            }
            outputs = {
                "logits": (["B", 1, importer.BPE_VOCAB_SIZE], "float32"),
                "present_padding_mask": (["B", "R"], "int32"),
                **{
                    name: (["B", 8, "R", 40], "float32")
                    for name in importer.KV_PRESENT_NAMES
                },
            }
            for index, name in enumerate(importer.KV_PRESENT_NAMES):
                current = f"current_cache_{index}"
                nodes.extend([
                    {
                        "id": f"current-{index}", "opType": "Identity",
                        "inputs": {"input": "input0"},
                        "outputs": {"out": {"tensor": current, "shape": ["B", 8, 1, 40], "dtype": "float32"}},
                        "params": {},
                    },
                    {
                        "id": f"present-{index}", "opType": "Concat",
                        "inputs": {"input0": f"input{index + 13}", "input1": current},
                        "outputs": {"out": {"tensor": name, "shape": ["B", 8, "R", 40], "dtype": "float32"}},
                        "params": {"axis": 2},
                    },
                ])
            for index, name in enumerate(("logits", "present_padding_mask")):
                shape, dtype = outputs[name]
                nodes.append({
                    "id": f"decoder-output-{index}", "opType": "Identity",
                    "inputs": {"input": "input0"},
                    "outputs": {"out": {"tensor": name, "shape": shape, "dtype": dtype}},
                    "params": {},
                })
        if "_int8" in Path(self._option(command, "--model")).stem:
            first_input = next(iter(inputs))
            nodes.append({
                "id": "preserved-quantize",
                "opType": "QuantizeLinear",
                "inputs": {
                    "input": first_input,
                    "scale": "fixture_scale",
                    "zero_point": "fixture_zero_point",
                },
                "outputs": {"out": {
                    "tensor": "fixture_u8",
                    "shape": inputs[first_input]["shape"],
                    "dtype": "uint8",
                }},
                "params": {},
            })
        graph = {
            "format": "volvox-graph/v1",
            "dimensions": dimensions,
            "inputs": inputs,
            "nodes": nodes,
            "outputs": list(outputs),
        }
        (output_path.parent / "graph.json").write_text(
            json.dumps(graph), encoding="utf-8"
        )


def _parity(_source):
    return {
        "provider": "fixture",
        "memory_max_abs_difference": 0.0,
        "router_max_abs_difference": 0.0,
        "decoder_valid_logits_max_abs_difference": 0.0,
        "greedy_argmax": "matched",
        "tolerance": 1.0e-4,
    }


def _heldout_summary(files, *, changed_predictions: int):
    def artifact(file_key: str):
        return {
            "filename": files[file_key],
            "bytes": files[f"{file_key}_bytes"],
            "sha256": files[f"{file_key}_sha256"],
        }

    def metrics(n: int):
        return {
            "answer_exact": 1.0,
            "n": n,
            "recomputed_answer_exact": 1.0,
            "recomputed_n": n,
            "target_exact": 1.0,
        }

    def timing(encoder_key: str, decoder_key: str):
        return {
            "encoder": files[encoder_key],
            "decoder": files[decoder_key],
            "encoder_seconds": 0.25,
            "decoder_seconds": 0.75,
            "decoder_calls": 2,
            "total_seconds": 1.0,
            "providers": ["CPUExecutionProvider"],
        }

    variant_summary = {
        "phone_number": metrics(1),
        "address": metrics(1),
        "overall": metrics(2),
    }
    return {
        "status": "passed",
        "evaluation_mode": "final_split_onnx_files",
        "dataset": "heldout-fixture",
        "records_total": 2,
        "batch_size": 1,
        "routing": "learned_router",
        "runtime": {
            "library": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "version": "fixture",
        },
        "comparison": {
            "n": 2,
            "changed_predictions": changed_predictions,
            "answer_agreement": 1.0,
            "prediction_exact_agreement": 1.0,
            "selected_family_agreement": 1.0,
        },
        "artifacts": {
            "fp32": {
                "encoder": artifact("encoder"),
                "decoder": artifact("decoder"),
            },
            "int8": {
                "encoder": artifact("encoder_int8_w8a8"),
                "decoder": artifact("decoder_int8_w8a8"),
            },
        },
        "summary": {
            "fp32": variant_summary,
            "int8": variant_summary,
        },
        "timing": {
            "fp32": timing("encoder", "decoder"),
            "int8": timing("encoder_int8_w8a8", "decoder_int8_w8a8"),
        },
        "timing_note": (
            "Static W8A8 speed depends on QDQ fusion and the selected "
            "execution provider."
        ),
    }


class TinyReceiptSplitImporterTests(unittest.TestCase):
    def test_staged_graph_rejects_retired_shape_system_field(self):
        with tempfile.TemporaryDirectory(
            prefix="volvoxai-split-retired-shape-system-"
        ) as temporary:
            stage = Path(temporary)
            (stage / "graph.json").write_text(json.dumps({
                "format": importer.GRAPH_FORMAT,
                "shape_system": "volvox-bounded-shape/v1",
                "dimensions": {},
                "inputs": {},
                "nodes": [],
                "outputs": [],
            }), encoding="utf-8")
            (stage / "model.safetensors").write_bytes(b"fixture")
            (stage / "export_report.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                importer.ImportFailure,
                "does not use the closed v1 root schema",
            ):
                importer._validate_staged_graph(
                    stage,
                    label="fixture",
                    expected_dimensions={},
                    expected_inputs={},
                    expected_outputs={},
                )

    def test_explicit_kv_sentinel_rejects_nonfinite_logits(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-kv-nan-") as temporary:
            source = _upgrade_source_to_kv(_write_source(Path(temporary) / "source"))
            validated = importer.validate_source(source)
            decoder_path = Path(validated["selected_paths"]["decoder"])
            decoder = onnx.load(decoder_path)
            logits_node = next(
                node for node in decoder.graph.node if node.name == "logits_value"
            )
            logits_node.output[0] = "unused_finite_logits"
            decoder.graph.node.append(_constant(
                "nonfinite_logits",
                "logits",
                np.full((1, 1, importer.BPE_VOCAB_SIZE), np.nan, np.float32),
            ))
            onnx.save(decoder, decoder_path)
            with self.assertRaisesRegex(importer.ImportFailure, "invalid logits"):
                importer.verify_explicit_kv_sentinel(validated)

    def test_explicit_kv_source_proves_sentinel_and_publishes_typed_v2_package(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-kv-import-") as temporary:
            root = Path(temporary)
            source = _upgrade_source_to_kv(_write_source(root / "source"))
            validated = importer.validate_source(source)
            source_hashes = {
                key: _sha256(path) for key, path in validated["paths"].items()
            }
            parity = importer.verify_explicit_kv_sentinel(validated)
            self.assertEqual(parity["families_verified"], list(importer.FAMILY_ORDER))
            self.assertEqual(parity["producer_initial_past_length"], 0)
            self.assertEqual(parity["package_initial_past_length"], 1)
            self.assertEqual(parity["sentinel_mask_value"], 1)
            self.assertEqual(parity["logits_max_abs_difference"], 0.0)
            self.assertEqual(parity["present_cache_suffix_max_abs_difference"], 0.0)

            fake = FakeKVExporter()

            def verified_parity(import_source):
                self.assertEqual(import_source["cache_mode"], "explicit-kv")
                return parity

            output = root / "package"
            importer.import_package(
                source,
                output,
                exporter=fake,
                parity_verifier=verified_parity,
            )
            manifest = json.loads(
                (output / "package_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["format"], importer.PACKAGE_FORMAT)
            self.assertEqual(
                manifest["variant"]["compiled_graph_package_class"],
                {"encoder": "fp32", "decoder": "fp32"},
            )
            self.assertIs(
                manifest["variant"]["complete_w8a8_fusion"],
                False,
            )
            self.assertEqual(
                set(manifest["shape_contract"]),
                {
                    "graph_shape_mode", "dimensions", "fixed_geometry",
                    "relations", "semantic_inputs",
                },
            )
            self.assertEqual(
                manifest["shape_contract"]["dimensions"],
                {
                    "B": {"min": 1, "max": 1},
                    "Q": {"min": 1, "max": 192},
                    "M": {"min": 211, "max": 402},
                    "P": {"min": 1, "max": 191},
                    "R": {"min": 2, "max": 192},
                },
            )
            self.assertEqual(
                manifest["shape_contract"]["semantic_inputs"],
                {
                    "question_position_ids": {
                        "shape": ["B", "Q"],
                        "values": "zero_based_contiguous",
                    }
                },
            )
            self.assertEqual(
                manifest["cache_contract"],
                {
                    "format": "masked-zero-sentinel-v1",
                    "layers": 4,
                    "heads": 8,
                    "head_width": 40,
                    "past_dimension": "P",
                    "present_dimension": "R",
                    "initial_past_length": 1,
                    "sentinel_mask_value": 1,
                    "cache_dtype": "float32",
                },
            )
            self.assertEqual(
                manifest["generation"]["strategy"],
                "greedy-autoregressive-explicit-kv",
            )
            self.assertEqual(len(manifest["graphs"]["encoder"]["inputs"]), 4)
            self.assertEqual(len(manifest["graphs"]["encoder"]["outputs"]), 12)
            self.assertEqual(len(manifest["graphs"]["decoder"]["inputs"]), 21)
            self.assertEqual(len(manifest["graphs"]["decoder"]["outputs"]), 10)
            self.assertEqual(
                manifest["validation"]["canonical_present_relation"]["witness_count"],
                8,
            )
            self.assertEqual(
                manifest["validation"]["kv_authoring_normalization"]["decoder"]
                ["executable_nodes_rewritten"],
                0,
            )
            self.assertEqual(
                manifest["validation"]["kv_authoring_parity"]["decoder"],
                {
                    "status": "not_applicable",
                    "reason": "decoder executable nodes were not rewritten",
                },
            )
            self.assertEqual(
                manifest["validation"]["offline_target"],
                "portable",
            )
            self.assertNotIn(
                "offline_target_attestation", manifest["validation"]
            )
            self.assertEqual(len(fake.commands), 2)
            self.assertTrue(all(
                [
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "--target"
                ] == ["portable"]
                for command in fake.commands
            ))
            self.assertTrue(all(
                command.count("--allow-silu-numerical-migration") == 1
                for command in fake.commands
            ))
            self.assertTrue(all(
                "--allow-static-qdq-qbatch-matmul-migration" not in command
                for command in fake.commands
            ))
            self.assertTrue(all(
                "--allow-static-qdq-groupnorm-silu-migration" not in command
                for command in fake.commands
            ))
            self.assertTrue(all(
                "--allow-quantized-bias-folding-migration" not in command
                for command in fake.commands
            ))
            self.assertTrue(all(
                "--defer-static-qdq-layout-optimization" not in command
                for command in fake.commands
            ))
            self.assertTrue(all(
                command.count(
                    "--enable-exact-common-subexpression-elimination"
                ) == 1
                for command in fake.commands
            ))
            decoder_bounds = [
                fake.commands[1][index + 1]
                for index, value in enumerate(fake.commands[1])
                if value == "--dimension-bound"
            ]
            self.assertEqual(
                decoder_bounds,
                ["B=1:1", "M=211:402", "P=1:191", "R=2:192"],
            )
            self.assertEqual(len(fake.model_signatures[0]["outputs"]), 12)
            self.assertEqual(len(fake.model_signatures[1]["inputs"]), 21)
            self.assertEqual(len(fake.model_signatures[1]["outputs"]), 10)
            self.assertEqual(
                {_key: _sha256(path) for _key, path in validated["paths"].items()},
                source_hashes,
            )

    def test_repeated_export_targets_are_intersected_and_recorded(self):
        with tempfile.TemporaryDirectory(
            prefix="volvoxai-split-kv-targets-"
        ) as temporary:
            root = Path(temporary)
            source = _upgrade_source_to_kv(_write_source(root / "source"))
            fake = FakeKVExporter()

            output = root / "package"
            importer.import_package(
                source,
                output,
                targets=(
                    "portable",
                    "backend:vulkan",
                    "backend:opengl",
                    "backend:cuda",
                    "backend:vulkan",
                ),
                exporter=fake,
                parity_verifier=lambda _source: {},
            )

            expected_requested = [
                "portable",
                "backend:vulkan",
                "backend:opengl",
                "backend:cuda",
            ]
            expected_resolved = [
                "cpu-js",
                "wasm",
                "webgpu",
                "native-cpu",
                "backend:vulkan",
                "backend:opengl",
                "backend:cuda",
            ]
            self.assertEqual(len(fake.commands), 2)
            for command in fake.commands:
                self.assertEqual(
                    [
                        command[index + 1]
                        for index, value in enumerate(command)
                        if value == "--target"
                    ],
                    expected_requested,
                )
            manifest = json.loads(
                (output / "package_manifest.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("offline_target", manifest["validation"])
            self.assertEqual(
                manifest["validation"]["offline_target_attestation"],
                {
                    "requested": expected_requested,
                    "resolved": expected_resolved,
                },
            )

    def test_mismatched_export_target_report_fails_before_publication(self):
        with tempfile.TemporaryDirectory(
            prefix="volvoxai-split-kv-target-report-"
        ) as temporary:
            root = Path(temporary)
            source = _upgrade_source_to_kv(_write_source(root / "source"))
            fake = FakeKVExporter()

            def mismatched_exporter(command):
                fake(command)
                report_path = Path(FakeKVExporter._option(command, "--report"))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["targets"] = {
                    "requested": ["portable"],
                    "resolved": ["cpu-js", "wasm", "webgpu", "native-cpu"],
                }
                report_path.write_text(json.dumps(report), encoding="utf-8")

            output = root / "package"
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "does not attest the requested targets",
            ):
                importer.import_package(
                    source,
                    output,
                    targets=("portable", "backend:vulkan"),
                    exporter=mismatched_exporter,
                    parity_verifier=lambda _source: {},
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".package.stage-*")), [])

    def test_target_cli_repeats_generic_exporter_targets(self):
        arguments = importer.parse_args(
            [
                "--source", "source",
                "--out-dir", "package",
                "--target", "portable",
                "--target", "backend:vulkan",
                "--target", "backend:opengl",
                "--target", "backend:cuda",
            ]
        )
        self.assertEqual(
            arguments.targets,
            [
                "portable",
                "backend:vulkan",
                "backend:opengl",
                "backend:cuda",
            ],
        )
        self.assertIsNone(
            importer.parse_args([
                "--source", "source",
                "--out-dir", "package",
            ]).targets
        )
        self.assertEqual(
            importer.parse_args([
                "--source", "source",
                "--out-dir", "package",
            ]).max_batch_size,
            1,
        )
        self.assertEqual(
            importer.parse_args([
                "--source", "source",
                "--out-dir", "package",
                "--max-batch-size", "2",
            ]).max_batch_size,
            2,
        )

    def test_opt_in_batch_bound_is_one_package_transaction(self):
        with tempfile.TemporaryDirectory(
            prefix="volvoxai-split-kv-batch-two-"
        ) as temporary:
            root = Path(temporary)
            source = _upgrade_source_to_kv(_write_source(root / "source"))
            fake = FakeKVExporter()
            parity = {
                "status": "passed",
                "provider": "fixture",
                "cases": [{"case": "batch_max_distinct_lanes", "B": 2}],
                "maximum_absolute_difference": 0.0,
                "greedy_argmax": "not_applicable",
            }
            output = root / "package"
            with patch.object(
                importer,
                "_verify_dynamic_authoring_parity",
                return_value=parity,
            ):
                importer.import_package(
                    source,
                    output,
                    max_batch_size=2,
                    exporter=fake,
                    parity_verifier=lambda _source: {},
                )

            manifest = json.loads(
                (output / "package_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["shape_contract"]["dimensions"]["B"],
                {"min": 1, "max": 2},
            )
            self.assertEqual(
                manifest["validation"]["kv_authoring_parity"]["encoder"],
                parity,
            )
            for command in fake.commands:
                self.assertIn("B=1:2", [
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "--dimension-bound"
                ])
            for role in ("encoder", "decoder"):
                graph = json.loads(
                    (output / role / "graph.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    graph["dimensions"]["B"], {"min": 1, "max": 2}
                )

    def test_explicit_kv_int8_enables_narrow_quantized_migrations_for_both_graphs(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-kv-int8-") as temporary:
            root = Path(temporary)
            source = _upgrade_source_to_kv(_write_source(root / "source"))
            _add_int8_w8a8_variant(source)
            fake = FakeKVExporter()

            output = root / "package"
            importer.import_package(
                source,
                output,
                variant="int8-w8a8",
                exporter=fake,
                parity_verifier=lambda _source: {},
            )

            manifest = json.loads(
                (output / "package_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["variant"]["compiled_graph_package_class"],
                {"encoder": "hybrid", "decoder": "hybrid"},
            )
            self.assertIs(
                manifest["variant"]["complete_w8a8_fusion"],
                False,
            )
            self.assertEqual(len(fake.commands), 2)
            self.assertTrue(all(
                command.count(
                    "--allow-static-qdq-qbatch-matmul-migration"
                ) == 1
                for command in fake.commands
            ))
            self.assertTrue(all(
                command.count(
                    "--allow-quantized-bias-folding-migration"
                ) == 1
                for command in fake.commands
            ))
            self.assertTrue(all(
                "--allow-static-qdq-groupnorm-silu-migration" not in command
                for command in fake.commands
            ))
            self.assertTrue(all(
                "_int8" in Path(FakeKVExporter._option(command, "--model")).stem
                for command in fake.commands
            ))

    def test_explicit_kv_int8_rejects_every_nonhybrid_compiled_class(self):
        with tempfile.TemporaryDirectory(
            prefix="volvoxai-split-kv-int8-class-"
        ) as temporary:
            root = Path(temporary)
            source = _upgrade_source_to_kv(_write_source(root / "source"))
            _add_int8_w8a8_variant(source)

            for package_class in ("fp32", "w8a8-v1"):
                with self.subTest(package_class=package_class):
                    output = root / f"package-{package_class}"
                    with patch.object(
                        importer,
                        "classify_package",
                        return_value=package_class,
                    ):
                        with self.assertRaisesRegex(
                            importer.ImportFailure,
                            "must classify both KV INT8 graphs as hybrid",
                        ):
                            importer.import_package(
                                source,
                                output,
                                variant="int8-w8a8",
                                exporter=FakeKVExporter(),
                                parity_verifier=lambda _source: {},
                            )
                    self.assertFalse(output.exists())

    def test_explicit_kv_source_rejects_weakened_cache_and_present_shapes(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-kv-cache-") as temporary:
            source = _upgrade_source_to_kv(_write_source(Path(temporary) / "source"))
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["kv_cache"]["format"] = "unsupported-cache-contract"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure, "KV source cache declaration is invalid"
            ):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-kv-shape-") as temporary:
            source = _upgrade_source_to_kv(_write_source(Path(temporary) / "source"))
            decoder_path = source / "decoder_model.onnx"
            decoder = onnx.load(decoder_path)
            decoder.graph.output[2].type.tensor_type.shape.dim[2].dim_param = "past_length"
            onnx.save(decoder, decoder_path)
            _refresh_model_record(source, "decoder")
            with self.assertRaisesRegex(
                importer.ImportFailure, "KV decoder output signature differs"
            ):
                importer.validate_source(source)

    def test_wrong_source_format_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-wrong-format-") as temporary:
            source = _write_source(Path(temporary) / "source")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["format"] = "unsupported-source-format"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "source manifest format must be",
            ):
                importer.validate_source(source)

    def test_dynamic_reshape_normalizer_rejects_two_independent_inferred_extents(self):
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Shape",
                        ["data"],
                        ["source_shape"],
                        name="dynamic_shape",
                    ),
                    helper.make_node(
                        "Gather",
                        ["source_shape", "reorder"],
                        ["target_shape"],
                        name="swap_dynamic_extents",
                        axis=0,
                    ),
                    helper.make_node(
                        "Reshape",
                        ["data", "target_shape"],
                        ["output"],
                        name="unrepresentable_reshape",
                    ),
                ],
                "two-independent-extents",
                [_value("data", TensorProto.FLOAT, ["B", "T", "M"])],
                [_value("output", TensorProto.FLOAT, ["B", "M", "T"])],
                initializer=[
                    numpy_helper.from_array(
                        np.asarray([0, 2, 1], dtype=np.int64),
                        "reorder",
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )

        with self.assertRaisesRegex(
            importer.ImportFailure,
            "needs 2 independent inferred dimensions",
        ):
            importer._rewrite_dynamic_reshape_targets(
                model,
                role="decoder",
                onnx=onnx,
            )

    def test_zero_changed_predictions_is_a_valid_nonnegative_count(self):
        files = {}
        for index, key in enumerate(
            ("encoder", "decoder", "encoder_int8_w8a8", "decoder_int8_w8a8")
        ):
            files[key] = f"{key}.onnx"
            files[f"{key}_bytes"] = index + 1
            files[f"{key}_sha256"] = f"{index:064x}"
        importer._validate_heldout_summary(
            _heldout_summary(files, changed_predictions=0),
            files,
        )
        for invalid in (-1, True, 0.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    importer.ImportFailure,
                    "non-negative integer",
                ):
                    importer._validate_heldout_summary(
                        _heldout_summary(files, changed_predictions=invalid),
                        files,
                    )

    def test_explicit_kv_source_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-kv-hash-") as temporary:
            source = _upgrade_source_to_kv(_write_source(Path(temporary) / "source"))
            with (source / "encoder_model.onnx").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "byte size does not match",
            ):
                importer.validate_source(source)

    def test_explicit_kv_publication_is_atomic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-kv-atomic-") as temporary:
            root = Path(temporary)
            source = _upgrade_source_to_kv(_write_source(root / "source"))
            output = root / "package"
            with self.assertRaisesRegex(importer.ImportFailure, "injected exporter"):
                importer.import_package(
                    source,
                    output,
                    exporter=FakeKVExporter(fail_on_call=2),
                    parity_verifier=lambda _source: {},
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".package.stage-*")), [])

            output.mkdir()
            marker = output / "owner.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(importer.ImportFailure, "already exists"):
                importer.import_package(
                    source,
                    output,
                    exporter=FakeKVExporter(),
                    parity_verifier=lambda _source: {},
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
