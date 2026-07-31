from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.exporter import publication
from tools.exporter.errors import ExporterError
from tools.exporter.generated.optimizer_registry import (
    KERNEL_REGISTRY_SHA256,
    PIPELINE_RECIPES,
    REGISTRY_SHA256,
    SCHEMA_VERSION,
)
from tools.exporter.ir import OpNode, TensorDataRef, TensorValue, ValuePort
from tools.exporter.runtime_ir import import_runtime_package
from examples.tiny_receipt_vqa.tools import optimize_direct_int8
from examples.tiny_receipt_vqa.tools import optimize_split_fp32
from examples.tiny_receipt_vqa.tools import quantize_split_package
from tools.exporter.optimizer.safetensors_io import (
    read_safetensors,
    write_safetensors,
)
from examples.tiny_receipt_vqa.tests.test_split_package_manifest import (
    _add_export_reports,
    _calibration_document,
    _write_package,
)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _specialize_decoder_output(document, weights, **_):
    specialized = json.loads(json.dumps(document))
    specialized["nodes"].append({
        "opType": "ArgMax",
        "inputs": {"input": "logits"},
        "outputs": {"out": "token_ids"},
        "outputs_shape": {"out": [1]},
        "outputs_dtype": {"out": "int32"},
        "params": {"axis": -1, "keepdims": False},
    })
    specialized["outputs"] = ["token_ids"]
    return specialized, weights, SimpleNamespace(total_changes=1)


class SplitPackagePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="volvox-split-optimizer-publication-"
        )
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "output"

    def tearDown(self):
        self.temporary.cleanup()

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".output.publish-*")), [])

    def write_calibration(self) -> Path:
        calibration = self.root / "calibration.json"
        calibration.write_text(
            json.dumps(_calibration_document(self.source)),
            encoding="utf-8",
        )
        return calibration

    def test_apply_validation_failure_leaves_no_visible_destination(self):
        _write_package(self.source, specialized_graphs=True)
        source_before = _snapshot(self.source)
        with mock.patch.object(
            optimize_direct_int8,
            "validate_split_package",
            side_effect=ValueError("injected staged validation failure"),
        ):
            with self.assertRaisesRegex(ValueError, "injected staged validation"):
                optimize_direct_int8.optimize_package(self.source, self.output)

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_publication_is_transactional_for_runtime_contract_failures(self):
        cases = {
            "reserved-port": lambda document: document["nodes"][0].update(
                inputs={"constructor": "input0"}
            ),
            "unsafe-output": lambda document: document["nodes"][0][
                "outputs_shape"
            ].update({"out": [1 << 27, 1 << 27]}),
        }
        for case, mutate in cases.items():
            with self.subTest(case=case):
                source = self.root / f"source-{case}"
                output = self.root / f"output-{case}"
                _write_package(source, specialized_graphs=True)
                graph_path = source / "encoder" / "graph.json"
                document = json.loads(graph_path.read_text(encoding="utf-8"))
                mutate(document)
                graph_path.write_text(json.dumps(document), encoding="utf-8")
                source_before = _snapshot(source)

                with self.assertRaises(ExporterError):
                    optimize_direct_int8.optimize_package(source, output)

                self.assertFalse(output.exists())
                self.assertEqual(_snapshot(source), source_before)
                self.assertEqual(
                    list(self.root.glob(f".{output.name}.publish-*")), []
                )

    def test_requested_quantized_lift_failure_is_private_and_atomic(self):
        _write_package(self.source, specialized_graphs=True)
        source_before = _snapshot(self.source)
        with mock.patch.object(
            optimize_direct_int8,
            "_quantized_attention_inventory",
            side_effect=ValueError("injected quantized lift refusal"),
        ):
            with self.assertRaisesRegex(ValueError, "lift refusal"):
                optimize_direct_int8.optimize_package(
                    self.source, self.output, fuse_attention=True,
                )

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_apply_publishes_typed_graph_and_tensor_rewrites(self):
        _write_package(self.source, specialized_graphs=True)

        def typed_rewrite(graph, tensors, **_):
            output = graph.outputs[0]
            producer = next(
                node for node in graph.nodes
                if output in node.output_map().values()
            )
            port = next(
                item.name for item in producer.outputs
                for value in (item.value,)
                if value == output
            )
            intermediate = f"{output}_typed"
            output_tensor = graph.tensors[output]
            graph.add_tensor(TensorValue(
                name=intermediate,
                shape=output_tensor.shape,
                dtype=output_tensor.dtype,
                source_dtype=output_tensor.source_dtype,
            ))
            producer.outputs = tuple(
                ValuePort(
                    item.name,
                    intermediate if item.name == port else item.value,
                    item.position,
                )
                for item in producer.outputs
            )
            zero = f"{output}_typed_zero"
            graph.add_tensor(TensorValue(
                name=zero,
                shape=(1,),
                dtype="float32",
                source_dtype="float32",
                initializer=True,
                data=TensorDataRef(zero),
            ))
            tensors[zero] = np.zeros((1,), dtype=np.float32)
            graph.add_node(OpNode.from_maps(
                name=f"{output}.typed-add",
                op_type="Add",
                inputs={"a": intermediate, "b": zero},
                outputs={"out": output},
            ))
            graph.invalidate_analyses()
            graph.verify()
            return SimpleNamespace(
                total_changes=1,
                runs=(),
                metadata=None,
            )

        with mock.patch.object(
            optimize_direct_int8,
            "optimize_runtime_graph",
            side_effect=typed_rewrite,
        ) as optimize:
            summary = optimize_direct_int8.optimize_package(
                self.source, self.output,
            )

        self.assertEqual(optimize.call_count, 2)
        self.assertEqual(summary["encoder"]["verified_rewrites"], 1)
        self.assertEqual(summary["decoder"]["verified_rewrites"], 1)
        for kind, output in (("encoder", "memory"), ("decoder", "logits")):
            graph = json.loads(
                (self.output / kind / "graph.json").read_text(encoding="utf-8")
            )
            self.assertEqual(graph["nodes"][-1]["opType"], "Add")
            self.assertIn(
                f"{output}_typed_zero",
                read_safetensors(self.output / kind / "model.safetensors"),
            )

    def test_direct_optimizer_refreshes_rewritten_export_report_identity(self):
        _write_package(self.source, specialized_graphs=True)
        _add_export_reports(self.source)

        optimize_direct_int8.optimize_package(self.source, self.output)

        manifest = json.loads(
            (self.output / "package_manifest.json").read_text(encoding="utf-8")
        )
        for kind in ("encoder", "decoder"):
            entry = manifest["graphs"][kind]["export_report"]
            payload = (self.output / entry["path"]).read_bytes()
            self.assertEqual(entry["bytes"], len(payload))
            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())

    def test_fp32_optimizer_publishes_active_registry_receipt(self):
        _write_package(self.source, specialized_graphs=False)
        _add_export_reports(self.source)

        summary = optimize_split_fp32.optimize_package(
            self.source,
            self.output,
            attention_fusion=False,
        )

        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(
            REGISTRY_SHA256,
            "e8f5571be9ec778433d873d075e737af7e339ea1e259328386d2bf7e13712d28",
        )
        recipe = PIPELINE_RECIPES["runtime-fp32-pre-ptq"]
        self.assertEqual(recipe["version"], 1)
        manifest = json.loads(
            (self.output / "package_manifest.json").read_text(encoding="utf-8")
        )
        for kind in ("encoder", "decoder"):
            entry = manifest["graphs"][kind]["export_report"]
            payload = (self.output / entry["path"]).read_bytes()
            report = json.loads(payload)
            receipt = report["optimization"]
            self.assertEqual(receipt["format"], "volvox-optimizer-pipeline-report/v1")
            self.assertEqual(receipt["contract"], "fp32-pre-ptq")
            self.assertEqual(receipt["source_nodes"], summary[kind]["before"])
            self.assertEqual(receipt["final_nodes"], summary[kind]["after"])
            self.assertEqual(len(receipt["phases"]), 1)
            phase = receipt["phases"][0]
            self.assertEqual(phase["name"], "fp32-pre-ptq")
            self.assertEqual(phase["total_changes"], summary[kind]["verified_rewrites"])
            pipeline = phase["pipeline"]
            self.assertEqual(pipeline["registry"], {
                "schema_version": SCHEMA_VERSION,
                "sha256": REGISTRY_SHA256,
                "kernel_registry_sha256": KERNEL_REGISTRY_SHA256,
            })
            self.assertEqual(
                pipeline["recipe"]["id"], "runtime-fp32-pre-ptq"
            )
            self.assertEqual(
                pipeline["recipe"]["version"], recipe["version"]
            )
            self.assertEqual(entry["bytes"], len(payload))
            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())

    def test_apply_never_publishes_undefined_or_forward_inputs(self):
        for case in ("undefined", "forward"):
            with self.subTest(case=case):
                source = self.root / f"source-{case}"
                output = self.root / f"output-{case}"
                _write_package(source, specialized_graphs=True)
                graph_path = source / "encoder" / "graph.json"
                document = json.loads(graph_path.read_text(encoding="utf-8"))
                if case == "undefined":
                    document["nodes"][0]["inputs"]["input"] = "missing"
                else:
                    document["nodes"] = [
                        {
                            "opType": "Identity",
                            "inputs": {"input": "future"},
                            "outputs": {"out": "memory"},
                            "outputs_shape": {"out": [1, 1]},
                            "outputs_dtype": {"out": "float32"},
                        },
                        {
                            "opType": "Identity",
                            "inputs": {"input": "input0"},
                            "outputs": {"out": "future"},
                            "outputs_shape": {"out": [1, 1]},
                            "outputs_dtype": {"out": "float32"},
                        },
                    ]
                graph_path.write_text(json.dumps(document), encoding="utf-8")
                source_before = _snapshot(source)

                with self.assertRaisesRegex(
                    ExporterError, "references unavailable tensor"
                ):
                    optimize_direct_int8.optimize_package(source, output)

                self.assertFalse(output.exists())
                self.assertEqual(_snapshot(source), source_before)
                self.assertEqual(
                    list(self.root.glob(f".{output.name}.publish-*")), []
                )

    def test_apply_never_publishes_nonfinite_json_or_unsafe_shapes(self):
        for case in ("nan-param", "unsafe-dimension"):
            with self.subTest(case=case):
                source = self.root / f"source-{case}"
                output = self.root / f"output-{case}"
                _write_package(source, specialized_graphs=True)
                graph_path = source / "encoder" / "graph.json"
                document = json.loads(graph_path.read_text(encoding="utf-8"))
                if case == "nan-param":
                    document["nodes"][0]["params"] = {"alpha": float("nan")}
                else:
                    document["nodes"][0]["outputs_shape"]["out"] = [1 << 53]
                graph_path.write_text(
                    json.dumps(document, allow_nan=True), encoding="utf-8"
                )
                source_before = _snapshot(source)

                expected = (
                    "non-finite JSON number"
                    if case == "nan-param"
                    else "JSON's safe range"
                )
                with self.assertRaisesRegex(ExporterError, expected):
                    optimize_direct_int8.optimize_package(source, output)

                self.assertFalse(output.exists())
                self.assertEqual(_snapshot(source), source_before)
                self.assertEqual(
                    list(self.root.glob(f".{output.name}.publish-*")), []
                )

    def test_apply_never_publishes_invalid_runtime_operator_contract(self):
        _write_package(self.source, specialized_graphs=True)
        graph_path = self.source / "encoder" / "graph.json"
        document = json.loads(graph_path.read_text(encoding="utf-8"))
        document["nodes"][0]["opType"] = "QLinear"
        graph_path.write_text(json.dumps(document), encoding="utf-8")
        source_before = _snapshot(self.source)

        with self.assertRaisesRegex(
            ExporterError, "requires exact input/weight/bias operands"
        ):
            optimize_direct_int8.optimize_package(self.source, self.output)

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_apply_never_publishes_invalid_float_operator_contract(self):
        _write_package(self.source, specialized_graphs=True)
        graph_path = self.source / "encoder" / "graph.json"
        document = json.loads(graph_path.read_text(encoding="utf-8"))
        document["nodes"][0]["opType"] = "Conv2D"
        graph_path.write_text(json.dumps(document), encoding="utf-8")
        source_before = _snapshot(self.source)

        with self.assertRaisesRegex(ExporterError, "weight"):
            optimize_direct_int8.optimize_package(self.source, self.output)

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_fp32_never_publishes_producer_collisions(self):
        _write_package(self.source, specialized_graphs=False)
        graph_path = self.source / "encoder" / "graph.json"
        document = json.loads(graph_path.read_text(encoding="utf-8"))
        document["nodes"].append({
            "opType": "Identity",
            "inputs": {"input": "input0"},
            "outputs": {"out": "memory"},
            "outputs_shape": {"out": [1, 1]},
            "outputs_dtype": {"out": "float32"},
        })
        graph_path.write_text(json.dumps(document), encoding="utf-8")
        source_before = _snapshot(self.source)

        with self.assertRaisesRegex(ExporterError, "multiply defined"):
            optimize_split_fp32.optimize_package(
                self.source,
                self.output,
                attention_fusion=False,
            )

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_fp32_never_publishes_unknown_operator_spelling(self):
        _write_package(self.source, specialized_graphs=False)
        graph_path = self.source / "encoder" / "graph.json"
        document = json.loads(graph_path.read_text(encoding="utf-8"))
        document["nodes"][0]["opType"] = "DefinitelyNotAnOperator"
        graph_path.write_text(json.dumps(document), encoding="utf-8")
        source_before = _snapshot(self.source)

        with self.assertRaisesRegex(
            ExporterError, "no exact emitted-descriptor validator"
        ):
            optimize_split_fp32.optimize_package(
                self.source,
                self.output,
                attention_fusion=False,
            )

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_fp32_never_publishes_malformed_known_operator_contracts(self):
        for case in ("conv-without-weight", "noncanonical-softmax-axis"):
            with self.subTest(case=case):
                source = self.root / f"source-{case}"
                output = self.root / f"output-{case}"
                _write_package(source, specialized_graphs=False)
                graph_path = source / "encoder" / "graph.json"
                document = json.loads(graph_path.read_text(encoding="utf-8"))
                if case == "conv-without-weight":
                    document["nodes"][0]["opType"] = "Conv2D"
                    expected = "weight"
                else:
                    document["nodes"][0]["opType"] = "Softmax"
                    document["nodes"][0]["params"] = {"axis": 0}
                    expected = "canonical last axis"
                graph_path.write_text(json.dumps(document), encoding="utf-8")
                source_before = _snapshot(source)

                with self.assertRaisesRegex(ExporterError, expected):
                    optimize_split_fp32.optimize_package(
                        source,
                        output,
                        attention_fusion=False,
                    )

                self.assertFalse(output.exists())
                self.assertEqual(_snapshot(source), source_before)
                self.assertEqual(
                    list(self.root.glob(f".{output.name}.publish-*")), []
                )

    def test_fp32_partial_weight_write_failure_is_private_and_cleaned(self):
        _write_package(self.source, specialized_graphs=False)
        source_before = _snapshot(self.source)
        writes = 0

        def fail_second_write(path, tensors, **kwargs):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("injected decoder weight write failure")
            return write_safetensors(path, tensors, **kwargs)

        with mock.patch.object(
            optimize_split_fp32,
            "write_safetensors",
            side_effect=fail_second_write,
        ):
            with self.assertRaisesRegex(OSError, "decoder weight write failure"):
                optimize_split_fp32.optimize_package(
                    self.source,
                    self.output,
                    attention_fusion=False,
                )

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_ptq_validation_failure_leaves_no_visible_destination(self):
        _write_package(self.source, specialized_graphs=True)
        calibration = self.write_calibration()
        source_before = _snapshot(self.source)
        with (
            mock.patch.object(
                quantize_split_package, "quantize_graph", return_value={}
            ),
            mock.patch.object(
                quantize_split_package,
                "optimize_runtime_package",
                side_effect=_specialize_decoder_output,
            ),
            mock.patch.object(
                quantize_split_package,
                "_refresh_package_class",
                return_value="hybrid",
            ),
            mock.patch.object(
                quantize_split_package,
                "validate_split_package",
                side_effect=ValueError("injected staged PTQ validation failure"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "staged PTQ validation"):
                quantize_split_package.quantize_package(
                    self.source, self.output, calibration
                )

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_ptq_publisher_refuses_public_input_or_routing_specialization(self):
        _write_package(self.source, specialized_graphs=False)
        calibration = self.write_calibration()
        source_before = _snapshot(self.source)

        def remove_family_input(graph, *_args, **_kwargs):
            family_input = "input2" if "input2" in graph.inputs else "input3"
            graph.inputs.remove(family_input)
            graph.tensors[family_input].public_input = False
            return {}

        with (
            mock.patch.object(
                quantize_split_package,
                "quantize_graph",
                side_effect=remove_family_input,
            ),
            mock.patch.object(
                quantize_split_package,
                "_refresh_package_class",
                return_value="hybrid",
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "changed public input descriptors",
            ):
                quantize_split_package.quantize_package(
                    self.source, self.output, calibration,
                )

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_split_quantizer_uses_typed_profile_and_central_runtime_ir(self):
        document = {
            "format": "volvox-graph/v1",
            "inputs": {"x": {"shape": [1, 3], "dtype": "float32"}},
            "outputs": ["y"],
            "nodes": [{
                "id": "dense",
                "opType": "Linear",
                "inputs": {"input": "x", "weight": "weight", "bias": "bias"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [1, 2]},
                "outputs_dtype": {"out": "float32"},
                "params": {"weight_layout": "IN_OUT"},
            }],
        }
        weights = {
            "weight": np.asarray(
                [[0.5, -0.25], [1.0, 0.75], [-0.5, 0.25]],
                dtype=np.float32,
            ),
            "bias": np.asarray([0.125, -0.25], dtype=np.float32),
        }
        graph = import_runtime_package(document, weights)
        summary = quantize_split_package.quantize_graph(
            graph,
            weights,
            {
                "x": {"min": -2.0, "max": 1.0},
                "y": {"min": -1.5, "max": 1.5},
            },
            sample_count=2,
            sample_digest="b" * 64,
        )
        self.assertEqual(summary["qlinear"], 1)
        self.assertEqual(summary["quantize_nodes"], 1)
        self.assertEqual(summary["dequantize_nodes"], 1)
        self.assertEqual(summary["retained_f32"], [])
        self.assertEqual(
            summary["optimizer"]["pipeline"]["recipe"]["id"],
            "runtime-ptq-authoring",
        )
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QLinear", "DequantizeLinear"],
        )
        qlinear = graph.nodes[1]
        qweight = graph.tensors[qlinear.input_map()["weight"]]
        self.assertEqual(qweight.shape, (2, 3))
        self.assertEqual(qweight.quantization.axis, 0)
        self.assertNotIn("weight", weights)
        self.assertNotIn("bias", weights)

    def test_ptq_never_publishes_an_unresolved_public_output(self):
        _write_package(self.source, specialized_graphs=True)
        graph_path = self.source / "encoder" / "graph.json"
        document = json.loads(graph_path.read_text(encoding="utf-8"))
        document["outputs"] = ["missing"]
        graph_path.write_text(json.dumps(document), encoding="utf-8")
        manifest_path = self.source / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["graphs"]["encoder"]["outputs"] = {"missing": "missing"}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        calibration = self.write_calibration()
        source_before = _snapshot(self.source)

        with (
            mock.patch.object(
                quantize_split_package, "quantize_graph", return_value={}
            ),
            mock.patch.object(
                quantize_split_package,
                "optimize_runtime_package",
                side_effect=_specialize_decoder_output,
            ),
            mock.patch.object(
                quantize_split_package,
                "_refresh_package_class",
                return_value="hybrid",
            ),
        ):
            with self.assertRaises(ExporterError) as caught:
                quantize_split_package.quantize_package(
                    self.source, self.output, calibration
                )
            self.assertEqual(caught.exception.diagnostic.code, "VXRTIR016")

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_ptq_partial_weight_write_failure_is_private_and_cleaned(self):
        _write_package(self.source, specialized_graphs=True)
        calibration = self.write_calibration()
        source_before = _snapshot(self.source)
        writes = 0

        def fail_second_write(path, tensors, **kwargs):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("injected decoder PTQ weight write failure")
            return write_safetensors(path, tensors, **kwargs)

        with (
            mock.patch.object(
                quantize_split_package, "quantize_graph", return_value={}
            ),
            mock.patch.object(
                quantize_split_package,
                "optimize_runtime_package",
                side_effect=_specialize_decoder_output,
            ),
            mock.patch.object(
                quantize_split_package,
                "_refresh_package_class",
                return_value="hybrid",
            ),
            mock.patch.object(
                quantize_split_package,
                "write_safetensors",
                side_effect=fail_second_write,
            ),
        ):
            with self.assertRaisesRegex(OSError, "decoder PTQ weight write failure"):
                quantize_split_package.quantize_package(
                    self.source, self.output, calibration
                )

        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()

    def test_final_directory_rename_failure_exposes_no_partial_package(self):
        _write_package(self.source, specialized_graphs=True)
        source_before = _snapshot(self.source)
        real_replace = os.replace

        def fail_final_replace(source, destination):
            if Path(destination) == self.output:
                raise OSError("injected final directory rename failure")
            return real_replace(source, destination)

        with mock.patch.object(
            publication.os, "replace", side_effect=fail_final_replace
        ):
            with self.assertRaises(ExporterError) as caught:
                optimize_direct_int8.optimize_package(self.source, self.output)
        self.assertEqual(caught.exception.diagnostic.code, "VXPUB007")
        self.assertFalse(self.output.exists())
        self.assertEqual(_snapshot(self.source), source_before)
        self.assert_no_staging()


if __name__ == "__main__":
    unittest.main()
