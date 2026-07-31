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
    return root


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
        vocab_size: int = importer.CHAR_VOCAB_SIZE,
    ):
        self.commands: list[list[str]] = []
        self.fail_on_call = fail_on_call
        self.vocab_size = vocab_size

    @staticmethod
    def _option(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def __call__(self, raw_command) -> None:
        command = list(raw_command)
        self.commands.append(command)
        if self.fail_on_call == len(self.commands):
            raise importer.ImportFailure("injected exporter failure")
        output = Path(self._option(command, "--out"))
        report = Path(self._option(command, "--report"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake-safetensors")
        report.write_text(json.dumps({
            "format": "volvox-export-report/v1",
            "source": {
                "path": self._option(command, "--model"),
                "format": "onnx",
            },
        }), encoding="utf-8")
        if output.parent.name == "encoder":
            inputs = {
                "input0": {
                    "source_name": "image",
                    "shape": [1, 1, 320, 672],
                    "dtype": "float32",
                },
                "input1": {
                    "source_name": "question_ids",
                    "shape": [1, 192],
                    "dtype": "int32",
                },
                "input2": {
                    "source_name": "family_ids",
                    "shape": [1],
                    "dtype": "int32",
                },
            }
            outputs = {
                "memory": ([1, 402, 320], "float32"),
                "memory_padding_mask": ([1, 402], "int32"),
                "router_logits": ([1, 8], "float32"),
                "selected_family_ids": ([1], "int32"),
            }
        else:
            inputs = {
                "input0": {
                    "source_name": "decoder_input_ids",
                    "shape": [1, 192],
                    "dtype": "int32",
                },
                "input1": {
                    "source_name": "memory",
                    "shape": [1, 402, 320],
                    "dtype": "float32",
                },
                "input2": {
                    "source_name": "memory_padding_mask",
                    "shape": [1, 402],
                    "dtype": "int32",
                },
                "input3": {
                    "source_name": "family_ids",
                    "shape": [1],
                    "dtype": "int32",
                },
            }
            outputs = {"logits": ([1, 192, self.vocab_size], "float32")}
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
                "outputs": {"out": "fixture_u8"},
                "outputs_shape": {"out": inputs[first_input]["shape"]},
                "outputs_dtype": {"out": "uint8"},
                "params": {},
            })
        for index, (name, (shape, dtype)) in enumerate(outputs.items()):
            nodes.append({
                "id": f"output-{index}",
                "opType": "Identity",
                "inputs": {"input": next(iter(inputs))},
                "outputs": {"out": name},
                "outputs_shape": {"out": shape},
                "outputs_dtype": {"out": dtype},
                "params": {},
            })
        graph = {
            "format": "volvox-graph/v1",
            "source": {
                "onnx": Path(self._option(command, "--model")).name,
                "package_class": (
                    "hybrid"
                    if "_int8" in Path(self._option(command, "--model")).stem
                    else "fp32"
                ),
            },
            "inputs": inputs,
            "nodes": nodes,
            "outputs": list(outputs),
        }
        (output.parent / "graph.json").write_text(
            json.dumps(graph),
            encoding="utf-8",
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

    def test_import_delegates_both_graphs_and_publishes_one_typed_manifest(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-import-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            output = root / "package"
            fake = FakeExporter()

            importer.import_package(
                source,
                output,
                exporter=fake,
                parity_verifier=_parity,
            )
            manifest = json.loads(
                (output / "package_manifest.json").read_text(encoding="utf-8")
            )
            encoder_report = json.loads(
                (output / "encoder" / "export_report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["format"], importer.PACKAGE_FORMAT)
        self.assertEqual(
            manifest["tokenizer"],
            {
                "type": "char-vocab",
                "version": 1,
                "itos_key": "itos",
                "token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
            },
        )
        self.assertEqual(len(fake.commands), 2)
        self.assertIn("question_ids=int32", fake.commands[0])
        self.assertIn("memory_padding_mask=int32", fake.commands[0])
        self.assertIn("decoder_input_ids=int32", fake.commands[1])
        self.assertNotIn(
            "--defer-static-qdq-layout-optimization", fake.commands[0],
        )
        self.assertNotIn(
            "--defer-static-qdq-layout-optimization", fake.commands[1],
        )
        self.assertEqual(
            manifest["graphs"]["encoder"]["inputs"],
            {"image": "input0", "question_ids": "input1", "family_ids": "input2"},
        )
        self.assertEqual(
            manifest["routing"],
            {
                "mode": "runtime",
                "family_inputs": {
                    "encoder": "input2",
                    "decoder": "input3",
                },
            },
        )
        self.assertEqual(
            manifest["mask_semantics"]["memory_padding_mask"],
            "nonzero_means_blocked",
        )
        self.assertEqual(manifest["generation"]["logits_row"], "prefix_length_minus_one")
        self.assertEqual(encoder_report["source"]["path"], "encoder_model.onnx")
        self.assertNotIn(str(source), json.dumps(manifest))
        self.assertNotIn(str(source), json.dumps(encoder_report))

    def test_bpe1536_source_is_validated_and_packaged_with_its_tokenizer_contract(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-bpe-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            source_vocab = _upgrade_source_to_bpe(source)
            validated = importer.validate_source(source)
            output = root / "package"
            fake = FakeExporter(vocab_size=importer.BPE_VOCAB_SIZE)

            importer.import_package(
                source,
                output,
                exporter=fake,
                parity_verifier=_parity,
            )
            manifest = json.loads(
                (output / "package_manifest.json").read_text(encoding="utf-8")
            )
            packaged_vocab = json.loads(
                (output / "vocab.json").read_text(encoding="utf-8")
            )
            decoder_graph = json.loads(
                (output / "decoder" / "graph.json").read_text(encoding="utf-8")
            )

        expected_tokenizer = {
            "type": "byte_fallback_bpe",
            "version": 1,
            "vocab_size": importer.BPE_VOCAB_SIZE,
            "normalization": "NFC",
            "tokenizer_hash": source_vocab["tokenizer_hash"],
            "itos_key": "itos",
            "merges_key": "merges",
            "token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
        }
        self.assertEqual(validated["vocab_size"], importer.BPE_VOCAB_SIZE)
        self.assertEqual(validated["tokenizer"], expected_tokenizer)
        self.assertEqual(manifest["tokenizer"], expected_tokenizer)
        self.assertEqual(packaged_vocab, source_vocab)
        self.assertEqual(
            decoder_graph["nodes"][-1]["outputs_shape"]["out"],
            [1, 192, importer.BPE_VOCAB_SIZE],
        )

    def test_bpe_tokenizer_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-bpe-hash-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            _upgrade_source_to_bpe(source)
            vocab_path = source / "vocab.json"
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab["tokenizer_hash"] = "0" * 64
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

            with self.assertRaisesRegex(importer.ImportFailure, "tokenizer_hash"):
                importer.validate_source(source)

    def test_publication_is_deterministic_and_distribution_readable(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-deterministic-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            outputs = (root / "package-a", root / "package-b")
            snapshots = []
            for output in outputs:
                importer.import_package(
                    source,
                    output,
                    exporter=FakeExporter(),
                    parity_verifier=_parity,
                )
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o755)
                snapshot = {}
                for path in sorted(output.rglob("*"), key=lambda item: item.as_posix()):
                    relative = path.relative_to(output).as_posix()
                    if path.is_dir():
                        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)
                    else:
                        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
                        snapshot[relative] = path.read_bytes()
                snapshots.append(snapshot)
            self.assertEqual(snapshots[0], snapshots[1])

    def test_second_export_failure_leaves_no_package_or_stage(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-atomic-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            output = root / "package"
            with self.assertRaisesRegex(importer.ImportFailure, "injected"):
                importer.import_package(
                    source,
                    output,
                    exporter=FakeExporter(fail_on_call=2),
                    parity_verifier=_parity,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".package.stage-*")), [])

    def test_concurrent_empty_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-publish-race-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            output = root / "package"
            fake = FakeExporter()

            def create_destination_after_export(command):
                fake(command)
                if len(fake.commands) == 2:
                    output.mkdir()

            with self.assertRaisesRegex(
                importer.ImportFailure,
                "already exists",
            ):
                importer.import_package(
                    source,
                    output,
                    exporter=create_destination_after_export,
                    parity_verifier=_parity,
                )
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(root.glob(".package.stage-*")), [])

    def test_hash_and_signature_changes_fail_before_export(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-source-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            (source / "encoder_model.onnx").write_bytes(
                (source / "encoder_model.onnx").read_bytes() + b"x"
            )
            with self.assertRaisesRegex(importer.ImportFailure, "byte size"):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-signature-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            model = onnx.load(source / "decoder_model.onnx")
            model.graph.output[0].name = "wrong_logits"
            model.graph.node[0].output[0] = "wrong_logits"
            onnx.save(model, source / "decoder_model.onnx")
            _refresh_model_record(source, "decoder")
            with self.assertRaisesRegex(importer.ImportFailure, "output signature"):
                importer.validate_source(source)

    def test_int8_w8a8_variant_is_selected_hashed_and_described_honestly(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-complete-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            int8_paths = _add_int8_w8a8_variant(source)
            validated = importer.validate_source(source, variant="int8-w8a8")
            self.assertEqual(
                {key: validated["paths"][key] for key in int8_paths},
                int8_paths,
            )
            self.assertEqual(
                validated["selected_paths"],
                {
                    "encoder": int8_paths["encoder_int8_w8a8"],
                    "decoder": int8_paths["decoder_int8_w8a8"],
                },
            )
            for key, path in int8_paths.items():
                self.assertEqual(validated["hashes"][key], _sha256(path))

            output = root / "package"
            fake = FakeExporter()
            importer.import_package(
                source,
                output,
                variant="int8-w8a8",
                exporter=fake,
                parity_verifier=_parity,
            )
            manifest = json.loads(
                (output / "package_manifest.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (output / "encoder" / "export_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                FakeExporter._option(fake.commands[0], "--model").endswith(
                    "encoder_model_int8.onnx"
                )
            )
            self.assertEqual(
                FakeExporter._option(fake.commands[0], "--quant-mode"),
                "preserve",
            )
            self.assertIn(
                "--defer-static-qdq-layout-optimization", fake.commands[0],
            )
            self.assertIn(
                "--defer-static-qdq-layout-optimization", fake.commands[1],
            )
            self.assertEqual(manifest["source"]["variant"], "int8-w8a8")
            self.assertEqual(
                manifest["source"]["encoder_onnx"]["path"],
                "encoder_model_int8.onnx",
            )
            self.assertEqual(manifest["variant"]["requested"], "int8-w8a8")
            self.assertEqual(
                Path(FakeExporter._option(fake.commands[0], "--model")),
                int8_paths["encoder_int8_w8a8"],
            )
            self.assertEqual(
                Path(FakeExporter._option(fake.commands[1], "--model")),
                int8_paths["decoder_int8_w8a8"],
            )
            self.assertEqual(
                manifest["variant"]["compiled_graph_package_class"],
                {"encoder": "hybrid", "decoder": "hybrid"},
            )
            self.assertFalse(manifest["variant"]["complete_w8a8_fusion"])
            self.assertEqual(report["source"]["path"], "encoder_model_int8.onnx")
            self.assertNotIn(str(source), json.dumps(manifest))
            self.assertNotIn(str(source), json.dumps(report))
            self.assertFalse((output / ".static-qdq-source").exists())

    def test_int8_runtime_accepts_declared_dynamic_prefix_lengths_within_bounds(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-prefixes-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            _upgrade_source_to_bpe(source)
            _add_int8_w8a8_variant(source)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runtime = manifest["variants"]["int8_w8a8"]["validation"]["runtime"]
            runtime["dynamic_prefix_lengths"] = [1, 10, 20]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            validated = importer.validate_source(source, variant="int8-w8a8")
            self.assertEqual(validated["vocab_size"], importer.BPE_VOCAB_SIZE)

            for invalid in ([2, 10, 20], [1, 20, 20], [1, 20, 193], [1, True, 20]):
                with self.subTest(invalid=invalid):
                    runtime["dynamic_prefix_lengths"] = invalid
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(
                        importer.ImportFailure,
                        "runtime validation",
                    ):
                        importer.validate_source(source, variant="int8-w8a8")

    def test_int8_import_rejects_a_claimed_hybrid_identity_graph(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-class-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            _add_int8_w8a8_variant(source)
            output = root / "package"
            fake = FakeExporter()

            def claim_hybrid_without_live_descriptors(command):
                fake(command)
                graph_path = Path(FakeExporter._option(list(command), "--out")).parent / "graph.json"
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                graph["nodes"] = [
                    node for node in graph["nodes"]
                    if node["opType"] == "Identity"
                ]
                graph["source"]["package_class"] = "hybrid"
                graph_path.write_text(json.dumps(graph), encoding="utf-8")

            with self.assertRaisesRegex(
                importer.ImportFailure,
                "does not match its live descriptors",
            ):
                importer.import_package(
                    source,
                    output,
                    variant="int8-w8a8",
                    exporter=claim_hybrid_without_live_descriptors,
                    parity_verifier=_parity,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".package.stage-*")), [])

    def test_int8_source_mutation_prevents_atomic_publication(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-mutate-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            int8_paths = _add_int8_w8a8_variant(source)
            output = root / "package"
            fake = FakeExporter()

            def mutate_int8_after_validation(command):
                fake(command)
                if len(fake.commands) == 1:
                    path = int8_paths["encoder_int8_w8a8"]
                    path.write_bytes(path.read_bytes() + b"x")

            with self.assertRaisesRegex(
                importer.ImportFailure,
                "source files changed",
            ):
                importer.import_package(
                    source,
                    output,
                    variant="int8-w8a8",
                    exporter=mutate_int8_after_validation,
                    parity_verifier=_parity,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".package.stage-*")), [])

    def test_incomplete_unknown_legacy_or_invalid_int8_files_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-incomplete-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["encoder_int8_w8a8"] = "encoder_model_int8.onnx"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "complete six-key set",
            ):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-legacy-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["encoder_int8_weight_only"] = "encoder_model_int8.onnx"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(importer.ImportFailure, "unsupported keys"):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-unknown-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["encoder_dynamic"] = "encoder_dynamic.onnx"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(importer.ImportFailure, "unsupported keys"):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-name-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            _add_int8_w8a8_variant(source)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["encoder_int8_w8a8"] = "wrong.onnx"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "encoder_model_int8.onnx",
            ):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-symlink-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            int8_paths = _add_int8_w8a8_variant(source)
            encoder_int8 = int8_paths["encoder_int8_w8a8"]
            encoder_int8.unlink()
            encoder_int8.symlink_to(int8_paths["decoder_int8_w8a8"].name)
            with self.assertRaisesRegex(importer.ImportFailure, "non-symlink"):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-size-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            int8_paths = _add_int8_w8a8_variant(source)
            path = int8_paths["decoder_int8_w8a8"]
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(importer.ImportFailure, "byte size"):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-hash-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            _add_int8_w8a8_variant(source)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["decoder_int8_w8a8_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(importer.ImportFailure, "SHA-256"):
                importer.validate_source(source)

    def test_int8_variant_contract_and_qdq_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-contract-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            _add_int8_w8a8_variant(source)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["variants"]["int8_w8a8"]["scheme"] = "S8S8"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(importer.ImportFailure, "exact static U8S8"):
                importer.validate_source(source, variant="int8-w8a8")

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-qdq-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            _add_int8_w8a8_variant(source)
            encoder_path = source / "encoder_model_int8.onnx"
            model = onnx.load(encoder_path)
            quantize = next(
                node for node in model.graph.node if node.op_type == "QuantizeLinear"
            )
            initializers = {item.name: item for item in model.graph.initializer}
            initializers[quantize.input[2]].data_type = TensorProto.INT8
            onnx.save(model, encoder_path)
            _refresh_model_record(source, "encoder_int8_w8a8")
            with self.assertRaisesRegex(importer.ImportFailure, "produce UINT8"):
                importer.validate_source(source, variant="int8-w8a8")

    def test_missing_int8_variant_and_cli_selection_are_explicit(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-int8-absent-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            with self.assertRaisesRegex(importer.ImportFailure, "variant is absent"):
                importer.validate_source(source, variant="int8-w8a8")

        defaults = importer.parse_args(["--source", "source", "--out-dir", "package"])
        selected = importer.parse_args([
            "--source",
            "source",
            "--out-dir",
            "package",
            "--variant",
            "int8-w8a8",
        ])
        self.assertEqual(defaults.variant, "fp32")
        self.assertEqual(defaults.weight_dtype, "auto")
        self.assertEqual(selected.variant, "int8-w8a8")

        invalid_cli_requests = (
            [
                "--source", "source", "--out-dir", "package",
                "--variant", "unknown",
            ],
        )
        for arguments in invalid_cli_requests:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        importer.parse_args(arguments)

    def test_unused_custom_import_is_allowed_but_custom_nodes_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-unused-domain-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            encoder_path = source / "encoder_model.onnx"
            model = onnx.load(encoder_path)
            model.opset_import.append(helper.make_opsetid("private.unused", 1))
            onnx.save(model, encoder_path)
            _refresh_model_record(source, "encoder")
            importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-custom-node-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            encoder_path = source / "encoder_model.onnx"
            model = onnx.load(encoder_path)
            model.opset_import.append(helper.make_opsetid("private.used", 1))
            model.graph.node.append(
                helper.make_node(
                    "PrivateIdentity",
                    ["image"],
                    ["private_result"],
                    domain="private.used",
                )
            )
            onnx.save(model, encoder_path)
            _refresh_model_record(source, "encoder")
            with self.assertRaisesRegex(importer.ImportFailure, "custom-domain nodes"):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-local-function-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            encoder_path = source / "encoder_model.onnx"
            model = onnx.load(encoder_path)
            model.functions.append(
                helper.make_function(
                    "local.wrapper",
                    "Wrapper",
                    ["x"],
                    ["y"],
                    [
                        helper.make_node(
                            "PrivateIdentity",
                            ["x"],
                            ["y"],
                            domain="private.function",
                        )
                    ],
                    [helper.make_opsetid("private.function", 1)],
                )
            )
            onnx.save(model, encoder_path)
            _refresh_model_record(source, "encoder")
            with self.assertRaisesRegex(importer.ImportFailure, "custom-domain nodes"):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-custom-subgraph-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            encoder_path = source / "encoder_model.onnx"
            model = onnx.load(encoder_path)
            condition_name = "private_condition"
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(True, np.bool_), condition_name)
            )
            branch_output = _value("branch_output", TensorProto.FLOAT, [1])
            private_branch = helper.make_graph(
                [
                    helper.make_node(
                        "PrivateConstant",
                        [],
                        ["branch_output"],
                        domain="private.subgraph",
                    )
                ],
                "private_branch",
                [],
                [branch_output],
            )
            standard_branch = helper.make_graph(
                [
                    _constant(
                        "standard_constant",
                        "branch_output",
                        np.zeros((1,), np.float32),
                    )
                ],
                "standard_branch",
                [],
                [branch_output],
            )
            model.graph.node.append(
                helper.make_node(
                    "If",
                    [condition_name],
                    ["unused_branch_result"],
                    then_branch=private_branch,
                    else_branch=standard_branch,
                )
            )
            onnx.save(model, encoder_path)
            _refresh_model_record(source, "encoder")
            with self.assertRaisesRegex(importer.ImportFailure, "custom-domain nodes"):
                importer.validate_source(source)

    def test_required_asset_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-symlink-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            config = source / "config.json"
            real_config = source / "config.real.json"
            config.rename(real_config)
            config.symlink_to(real_config.name)
            with self.assertRaisesRegex(importer.ImportFailure, "non-symlink"):
                importer.validate_source(source)

    def test_constant_attribute_external_tensor_data_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-external-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            encoder_path = source / "encoder_model.onnx"
            model = onnx.load(encoder_path)
            tensor = model.graph.node[0].attribute[0].t
            tensor.ClearField("raw_data")
            tensor.data_location = TensorProto.EXTERNAL
            location = tensor.external_data.add()
            location.key = "location"
            location.value = "constant.bin"
            onnx.save_model(model, encoder_path, save_as_external_data=False)
            _refresh_model_record(source, "encoder")

            with self.assertRaisesRegex(
                importer.ImportFailure,
                "encoder contains external tensor data",
            ):
                importer.validate_source(source)

    def test_publishable_json_assets_reject_private_absolute_paths(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-config-path-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            config_path = source / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["producer_cache"] = str((root / "sensitive-config").resolve())
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "source config contains a private absolute path",
            ):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-vocab-path-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            vocab_path = source / "vocab.json"
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab["itos"][4] = str((root / "sensitive-vocab").resolve())
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "source vocab contains a private absolute path",
            ):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-config-key-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            config_path = source / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["producer"] = {
                str((root / "sensitive-config-key").resolve()): "private"
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "source config contains a private absolute path",
            ):
                importer.validate_source(source)

        with tempfile.TemporaryDirectory(prefix="volvoxai-split-file-uri-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            config_path = source / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["producer_cache"] = "FILE:" + "/" + "private/cache"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "source config contains a private absolute path",
            ):
                importer.validate_source(source)

    def test_non_finite_json_and_parity_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-split-json-number-") as temporary:
            root = Path(temporary)
            source = _write_source(root / "source")
            config_path = source / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["producer_score"] = float("nan")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "non-finite numeric constant",
            ):
                importer.validate_source(source)

        class FakeSession:
            def __init__(self, path, providers):
                self.kind = Path(path).name
                self.calls = 0
                self.providers = providers

            def run(self, _outputs, inputs):
                self.calls += 1
                if self.kind == "encoder":
                    length = int(inputs["question_ids"].shape[1]) + 210
                    router = np.zeros((1, 8), dtype=np.float32)
                    router[0, 0] = np.nan
                    mask = np.zeros((1, length), dtype=np.bool_)
                    if int(inputs["question_ids"].shape[1]) == 192:
                        mask[:, 213:] = True
                    return [
                        np.zeros((1, length, 320), dtype=np.float32),
                        mask,
                        router,
                        np.zeros((1,), dtype=np.int64),
                    ]
                length = int(inputs["decoder_input_ids"].shape[1])
                return [np.zeros((1, length, 760), dtype=np.float32)]

        source = {
            "paths": {
                "encoder": Path("encoder"),
                "decoder": Path("decoder"),
            },
            "config": {"max_q_len": 192, "max_out_len": 192},
        }
        fake_runtime = types.SimpleNamespace(InferenceSession=FakeSession)
        with patch.dict(sys.modules, {"onnxruntime": fake_runtime}):
            with self.assertRaisesRegex(
                importer.ImportFailure,
                "non-finite outputs or differences",
            ):
                importer.verify_static_padding(source)


if __name__ == "__main__":
    unittest.main()
