"""Hermetic contract tests for the receipt digit reader importer.

No model data, no network, no ONNX Runtime. These cover the checks that decide
whether a package is publishable at all: release integrity, the producer
format/opset contract, and the derivation of the per-request ABI and optional
physical-batch domain from the producer manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.receipt_digit_reader.tools import import_hf_onnx as importer  # noqa: E402

CONFIG = {
    "model_type": "receipt-digit-reader",
    "format": importer.SOURCE_FORMAT,
    "slots": 16,
    "phone_slots": 12,
    "street_slots": 4,
    "num_classes": 11,
    "blank_class": 10,
    "input": {"channels": 1, "height": 320, "width": 672},
}
MANIFEST = {
    "format": importer.SOURCE_FORMAT,
    "opset": 18,
    "inputs": [{"name": "image", "shape": ["batch", 1, 320, 672], "dtype": "float32"}],
    "outputs": [{"name": "slot_logits", "shape": ["batch", 16, 11], "dtype": "float32"}],
}


def _published_graph(
    batch_symbol: str | None = None,
    max_batch_size: int = 1,
) -> dict:
    leading: int | str = 1 if batch_symbol is None else batch_symbol
    dimensions = {} if batch_symbol is None else {
        batch_symbol: {"min": 1, "max": max_batch_size, "multiple_of": 1},
    }
    return {
        "format": "volvox-graph/v1",
        "dimensions": dimensions,
        "inputs": {
            importer.INPUT_TENSOR: {
                "shape": [leading, 1, 320, 672], "dtype": "float32",
            },
        },
        "outputs": [importer.OUTPUT_TENSOR],
        "nodes": [{
            "id": "node_0",
            "opType": "Identity",
            "inputs": {"input": importer.INPUT_TENSOR},
            "outputs": {"out": {
                "tensor": importer.OUTPUT_TENSOR,
                "dtype": "float32",
                "shape": [leading, 16, 11],
            }},
            "params": {},
        }],
    }


def _publish_manifest(
    out_dir: Path,
    *,
    batch_symbol: str | None,
    max_batch_size: int,
) -> dict:
    return importer.write_manifest(
        out_dir,
        variant="fp32", source=out_dir,
        checksums={"model.onnx": "0" * 64}, config=CONFIG,
        input_shape=(1, 1, 320, 672), dimension_bounds={"slot": 16},
        batch_symbol=batch_symbol, max_batch_size=max_batch_size,
        targets=("portable",), resolved_targets=("cpu-js",), ptq=None,
    )


def _write_release(root: Path, *, config=None, manifest=None, corrupt=False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "config.json": json.dumps(config if config is not None else CONFIG),
        "manifest.json": json.dumps(manifest if manifest is not None else MANIFEST),
        "model.onnx": "not a real graph",
    }
    lines = []
    for name, text in payload.items():
        (root / name).write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if corrupt and name == "model.onnx":
            digest = "0" * 64
        lines.append(f"{digest}  {name}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


class ReleaseVerificationTests(unittest.TestCase):
    def test_verifies_every_consumed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _write_release(Path(directory) / "release")
            verified = importer.verify_release(
                source, checked=("config.json", "manifest.json", "model.onnx"),
            )
            self.assertEqual(len(verified), 3)
            self.assertTrue(all(len(value) == 64 for value in verified.values()))

    def test_rejects_a_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _write_release(Path(directory) / "release", corrupt=True)
            with self.assertRaisesRegex(importer.ImportFailure, "SHA-256"):
                importer.verify_release(source, checked=("model.onnx",))

    def test_rejects_an_uncovered_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _write_release(Path(directory) / "release")
            with self.assertRaisesRegex(importer.ImportFailure, "does not cover"):
                importer.verify_release(source, checked=("model_int8.onnx",))

    def test_requires_a_checksum_list(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "release"
            source.mkdir()
            with self.assertRaisesRegex(importer.ImportFailure, "checksum list"):
                importer.verify_release(source, checked=("model.onnx",))


class StaticAbiTests(unittest.TestCase):
    def test_input_shape_is_batch_one_from_config(self):
        self.assertEqual(importer._static_input_shape(CONFIG), [1, 1, 320, 672])

    def test_rejects_a_non_positive_extent(self):
        broken = {**CONFIG, "input": {**CONFIG["input"], "height": 0}}
        with self.assertRaisesRegex(importer.ImportFailure, "positive"):
            importer._static_input_shape(broken)

    def test_rejects_a_missing_input_specification(self):
        with self.assertRaisesRegex(importer.ImportFailure, "input specification"):
            importer._static_input_shape({"format": importer.SOURCE_FORMAT})


class DynamicBatchContractTests(unittest.TestCase):
    @staticmethod
    def _write_symbolic_model(
        path: Path, *, input_symbol: str = "batch", output_symbol: str = "batch",
    ) -> None:
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Identity", ["image"], ["slot_logits"])],
            "symbolic_batch_fixture",
            [helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [input_symbol, 1, 320, 672],
            )],
            [helper.make_tensor_value_info(
                "slot_logits", TensorProto.FLOAT, [output_symbol, 1, 320, 672],
            )],
        )
        onnx.save(helper.make_model(graph), path)

    def test_max_batch_size_requires_a_positive_integer(self):
        self.assertEqual(importer._validated_max_batch_size(1), 1)
        self.assertEqual(importer._validated_max_batch_size(256), 256)
        self.assertEqual(importer._validated_max_batch_size(1 << 40), 1 << 40)
        for invalid in (0, -1, True, 1.5, "4"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(importer.ImportFailure, "max batch size"):
                    importer._validated_max_batch_size(invalid)

    def test_producer_input_and_output_must_share_one_named_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            self._write_symbolic_model(model)
            self.assertEqual(importer._producer_batch_symbol(model, MANIFEST), "batch")

            self._write_symbolic_model(model, output_symbol="other_batch")
            with self.assertRaisesRegex(importer.ImportFailure, "output batch symbols"):
                importer._producer_batch_symbol(model, MANIFEST)

    def test_dynamic_export_omits_static_shape_and_passes_exact_domain(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(importer, "_run") as run:
            root = Path(directory)
            importer.export_source_graph(
                root / "model.onnx", root / "out",
                input_shape=None,
                dimension_bounds={
                    "batch": {"min": 1, "max": 4, "multiple_of": 1},
                    "slot": {"min": 16, "max": 16},
                },
                targets=("portable",), report_path=None,
            )
            command = run.call_args.args[0]
            self.assertNotIn("--input-shape", command)
            bounds = [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--dimension-bound"
            ]
            self.assertEqual(bounds, ["batch=1:4:1", "slot=16:16"])

    def test_static_export_keeps_the_legacy_shape_concretization(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(importer, "_run") as run:
            root = Path(directory)
            importer.export_source_graph(
                root / "model.onnx", root / "out",
                input_shape=(1, 1, 320, 672),
                dimension_bounds={"slot": {"min": 16, "max": 16}},
                targets=(), report_path=None,
            )
            command = run.call_args.args[0]
            shape_index = command.index("--input-shape")
            self.assertEqual(command[shape_index + 1], "image=1x1x320x672")

    def test_manifest_separates_per_request_shape_from_graph_batch_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "graph.json").write_text(
                json.dumps(_published_graph("batch", 4)), encoding="utf-8",
            )
            manifest = _publish_manifest(
                out_dir, batch_symbol="batch", max_batch_size=4,
            )
            self.assertEqual(manifest["abi"]["input"]["shape"], [1, 1, 320, 672])
            self.assertEqual(manifest["abi"]["batch"], {
                "per_request": 1,
                "symbol": "batch",
                "min": 1,
                "max": 4,
                "multiple_of": 1,
            })

    def test_manifest_always_records_the_static_batch_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "graph.json").write_text(
                json.dumps(_published_graph()), encoding="utf-8",
            )
            manifest = _publish_manifest(
                out_dir, batch_symbol=None, max_batch_size=1,
            )
            self.assertEqual(manifest["abi"]["batch"], {
                "per_request": 1,
                "symbol": None,
                "min": 1,
                "max": 1,
                "multiple_of": 1,
            })

    def _assert_publication_rejected(
        self,
        graph: dict,
        pattern: str,
        *,
        batch_symbol: str | None = "batch",
        max_batch_size: int = 4,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "graph.json").write_text(
                json.dumps(graph), encoding="utf-8",
            )
            with self.assertRaisesRegex(importer.ImportFailure, pattern):
                _publish_manifest(
                    out_dir,
                    batch_symbol=batch_symbol,
                    max_batch_size=max_batch_size,
                )
            self.assertFalse((out_dir / "manifest.json").exists())
            self.assertEqual(list(out_dir.glob(".manifest.json.tmp-*")), [])

    def test_manifest_publication_rejects_an_exporter_input_tail_change(self):
        graph = _published_graph("batch", 4)
        graph["inputs"][importer.INPUT_TENSOR]["shape"] = ["batch", 1, 321, 672]
        self._assert_publication_rejected(graph, "input tail")

    def test_manifest_publication_rejects_an_exporter_output_batch_change(self):
        graph = _published_graph("batch", 4)
        graph["nodes"][0]["outputs"]["out"]["shape"][0] = "other_batch"
        self._assert_publication_rejected(graph, "public leading axes")

    def test_manifest_publication_rejects_missing_or_wrong_batch_domains(self):
        malformed = []
        missing = _published_graph("batch", 4)
        missing["dimensions"] = {}
        malformed.append(missing)
        wrong_maximum = _published_graph("batch", 4)
        wrong_maximum["dimensions"]["batch"]["max"] = 8
        malformed.append(wrong_maximum)
        wrong_multiple = _published_graph("batch", 4)
        wrong_multiple["dimensions"]["batch"]["multiple_of"] = 2
        malformed.append(wrong_multiple)
        wrong_spelling = _published_graph("batch", 4)
        wrong_spelling["dimensions"]["batch"].pop("multiple_of")
        wrong_spelling["dimensions"]["batch"]["multipleOf"] = 1
        malformed.append(wrong_spelling)

        for graph in malformed:
            with self.subTest(domain=graph["dimensions"]):
                self._assert_publication_rejected(graph, "dimension 'batch'")

    def test_manifest_publication_rejects_public_name_changes(self):
        extra_input = _published_graph("batch", 4)
        extra_input["inputs"]["other"] = {
            "shape": ["batch", 1], "dtype": "float32",
        }
        self._assert_publication_rejected(extra_input, "inputs must be exactly")

        wrong_output = _published_graph("batch", 4)
        wrong_output["outputs"] = ["digit_logits"]
        self._assert_publication_rejected(wrong_output, "outputs must be exactly")

    def test_manifest_publication_rejects_a_symbolic_static_batch(self):
        graph = _published_graph("batch", 1)
        self._assert_publication_rejected(
            graph,
            "literal leading batch extent 1",
            batch_symbol=None,
            max_batch_size=1,
        )


class SourceContractTests(unittest.TestCase):
    def test_rejects_a_foreign_source_format(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _write_release(
                Path(directory) / "release",
                config={**CONFIG, "format": "something_else"},
            )
            with self.assertRaisesRegex(importer.ImportFailure, "is not"):
                importer.import_release(
                    source, Path(directory) / "out", variant="fp32", targets=(),
                    calibration_path=None, activation_dtype="int8",
                    activation_scheme="asymmetric", float_ops=(),
                )

    def test_rejects_a_foreign_opset(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _write_release(
                Path(directory) / "release", manifest={**MANIFEST, "opset": 17},
            )
            with self.assertRaisesRegex(importer.ImportFailure, "opset"):
                importer.import_release(
                    source, Path(directory) / "out", variant="fp32", targets=(),
                    calibration_path=None, activation_dtype="int8",
                    activation_scheme="asymmetric", float_ops=(),
                )

    def test_rejects_an_unknown_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _write_release(Path(directory) / "release")
            with self.assertRaisesRegex(importer.ImportFailure, "unknown variant"):
                importer.import_release(
                    source, Path(directory) / "out", variant="fp16", targets=(),
                    calibration_path=None, activation_dtype="int8",
                    activation_scheme="asymmetric", float_ops=(),
                )

    def test_ptq_requires_a_calibration_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _write_release(Path(directory) / "release")
            with self.assertRaisesRegex(importer.ImportFailure, "--calibration"):
                importer.import_release(
                    source, Path(directory) / "out", variant="ptq", targets=(),
                    calibration_path=None, activation_dtype="int8",
                    activation_scheme="asymmetric", float_ops=(),
                )


class DefaultPolicyTests(unittest.TestCase):
    def test_attention_scores_are_retained_in_float_by_default(self):
        # The readout score BatchMatMul reaches four digits of dynamic range,
        # so an 8-bit step destroys the phone head. Keeping it float is the
        # default the example documents and measures.
        self.assertIn("BatchMatMul", importer.DEFAULT_PTQ_FLOAT_OPS)

    def test_default_coverage_is_convolution_only(self):
        # Matching the producer's own INT8 coverage holds target_exact at
        # 0.9685 against 0.9690 for FP32 on the held-out split; widening it
        # costs real accuracy, so the default must not drift silently.
        for retained in ("Linear", "GroupNorm", "SiLU", "LayerNorm", "Add"):
            self.assertIn(retained, importer.DEFAULT_PTQ_FLOAT_OPS)
        self.assertNotIn("Conv2D", importer.DEFAULT_PTQ_FLOAT_OPS)


if __name__ == "__main__":
    unittest.main()
