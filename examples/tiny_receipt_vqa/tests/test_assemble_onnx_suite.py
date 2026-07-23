import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = EXAMPLE_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

assembler = importlib.import_module("assemble_onnx_suite")


def graph_document(
    inputs,
    output_name,
    output_shape,
    *,
    output_dtype="int32",
    package_class="fp32",
    nodes=None,
    outputs=None,
):
    return {
        "format": assembler.GRAPH_FORMAT,
        "source": {
            "onnx": "fixture.onnx",
            "frontend": assembler.ONNX_FRONTEND,
            "package_class": package_class,
        },
        "inputs": inputs,
        "nodes": nodes if nodes is not None else [
            {
                "id": "terminal",
                "opType": "ArgMax",
                "inputs": {"input": next(iter(inputs))},
                "outputs": {"out": output_name},
                "outputs_shape": {"out": output_shape},
                "outputs_dtype": {"out": output_dtype},
                "params": {"axis": -1, "keepdims": 0},
            }
        ],
        "outputs": outputs if outputs is not None else [output_name],
    }


class OnnxSuiteFixture:
    def __init__(
        self,
        root: Path,
        *,
        mask_mode="keep-mask-i32",
        router_precision="w8a8-v1",
        family_precision="w8a32",
    ):
        self.root = root
        self.router = root / "exports" / "router"
        self.families = {
            family: root / "exports" / "families" / family
            for family in assembler.FAMILY_ORDER
        }
        self.mask_mode = mask_mode
        self.router_precision = router_precision
        self.family_precision = family_precision
        self.mask_name = (
            "router_keep" if mask_mode == "keep-mask-i32" else "router_weights"
        )
        self.write_router()
        for family in assembler.FAMILY_ORDER:
            self.write_family(family)

    @staticmethod
    def write_export(
        directory: Path,
        graph: dict,
        tensors: dict[str, np.ndarray],
        *,
        metadata: dict[str, str] | None = None,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / assembler.GRAPH_FILENAME).write_text(
            json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        save_file(
            tensors,
            str(directory / assembler.WEIGHTS_FILENAME),
            metadata=metadata,
        )

    def router_graph(self):
        mask_dtype = "int32" if self.mask_mode == "keep-mask-i32" else "float32"
        inputs = {
            "q_ids": {"shape": [1, 8], "dtype": "int32"},
            self.mask_name: {"shape": [1, 8], "dtype": mask_dtype},
        }
        if self.router_precision == "w8a8-v1":
            descriptor = {"scheme": "per_tensor", "scale": 0.125, "zero_point": 0}
            graph = graph_document(
                inputs, "family_id", [1], package_class=self.router_precision,
                nodes=[
                    {
                        "id": "embed", "opType": "QEmbedding",
                        "inputs": {"input": "q_ids", "weight": "router_embedding"},
                        "outputs": {"out": "question_tokens"},
                        "outputs_shape": {"out": [1, 8, 4]},
                        "outputs_dtype": {"out": "int8"},
                        "params": {},
                    },
                    {
                        "id": "masked_mean", "opType": "QMaskedMean",
                        "inputs": {"input": "question_tokens", "mask": self.mask_name},
                        "outputs": {"out": "pooled"},
                        "outputs_shape": {"out": [1, 4]},
                        "outputs_dtype": {"out": "int8"},
                        "params": {},
                    },
                    {
                        "id": "router_logits", "opType": "QLinear",
                        "inputs": {"input": "pooled", "weight": "router_weight", "bias": "router_bias"},
                        "outputs": {"out": "router_logits"},
                        "outputs_shape": {"out": [1, 8]},
                        "outputs_dtype": {"out": "int8"},
                        "params": {},
                    },
                    {
                        "id": "terminal", "opType": "QArgMax",
                        "inputs": {"input": "router_logits"},
                        "outputs": {"out": "family_id"},
                        "outputs_shape": {"out": [1]},
                        "outputs_dtype": {"out": "int32"},
                        "params": {"axis": -1},
                    },
                ],
            )
            graph["quantization"] = {
                "format": "volvox-affine-safetensors/v1",
                "tensors": {
                "router_embedding": {
                    "scheme": "per_axis", "axis": 0,
                    "scale_tensor": "router_embedding_scale",
                    "zero_point_tensor": "router_embedding_zero_point",
                },
                "router_weight": {
                    "scheme": "per_axis", "axis": 0,
                    "scale_tensor": "router_weight_scale",
                    "zero_point_tensor": "router_weight_zero_point",
                },
                **{
                    name: {
                        "scheme": descriptor["scheme"],
                        "scale_tensor": "router_activation_scale",
                        "zero_point_tensor": "router_activation_zero_point",
                    }
                    for name in ("question_tokens", "pooled", "router_logits")
                },
            }}
            all_masked = "zero-vector-via-qmaskedmean"
        else:
            graph = graph_document(
                inputs, "family_id", [1], package_class=self.router_precision,
                nodes=[
                    {
                        "id": "embed", "opType": "Embedding",
                        "inputs": {"input": "q_ids", "weight": "router_embedding"},
                        "outputs": {"out": "question_tokens"},
                        "outputs_shape": {"out": [1, 8, 4]},
                        "outputs_dtype": {"out": "float32"}, "params": {},
                    },
                    {
                        "id": "expand_mask", "opType": "Unsqueeze",
                        "inputs": {"input": self.mask_name},
                        "outputs": {"out": "mask_3d"},
                        "outputs_shape": {"out": [1, 8, 1]},
                        "outputs_dtype": {"out": "float32"}, "params": {"axes": [2]},
                    },
                    {
                        "id": "mask_tokens", "opType": "Mul",
                        "inputs": {"a": "question_tokens", "b": "mask_3d"},
                        "outputs": {"out": "masked_tokens"},
                        "outputs_shape": {"out": [1, 8, 4]},
                        "outputs_dtype": {"out": "float32"}, "params": {},
                    },
                    {
                        "id": "sequence_last", "opType": "Transpose",
                        "inputs": {"input": "masked_tokens"},
                        "outputs": {"out": "masked_tokens_d_s"},
                        "outputs_shape": {"out": [1, 4, 8]},
                        "outputs_dtype": {"out": "float32"}, "params": {"perm": [0, 2, 1]},
                    },
                    {
                        "id": "weighted_sum", "opType": "ReduceSum",
                        "inputs": {"input": "masked_tokens_d_s"},
                        "outputs": {"out": "pooled"},
                        "outputs_shape": {"out": [1, 4]},
                        "outputs_dtype": {"out": "float32"},
                        "params": {"axis": -1, "keepdims": False},
                    },
                    {
                        "id": "router_logits", "opType": "Linear",
                        "inputs": {"input": "pooled", "weight": "router_weight", "bias": "router_bias"},
                        "outputs": {"out": "router_logits"},
                        "outputs_shape": {"out": [1, 8]},
                        "outputs_dtype": {"out": "float32"}, "params": {"weight_layout": "IN_OUT"},
                    },
                    {
                        "id": "terminal", "opType": "ArgMax",
                        "inputs": {"input": "router_logits"},
                        "outputs": {"out": "family_id"},
                        "outputs_shape": {"out": [1]},
                        "outputs_dtype": {"out": "int32"},
                        "params": {"axis": -1, "keepdims": False, "select_last_index": 0},
                    },
                ],
            )
            all_masked = "zero-vector-via-normalized-zero-weights"
        graph["source"]["features"] = {
            "router": [{
                "source_node": "terminal", "output": "family_id",
                "tie_policy": "first-index", "semantic_mask_inputs": [self.mask_name],
                "all_masked": all_masked,
            }],
        }
        return graph

    def family_graph(self):
        inputs = {
            "image": {"shape": [1, 2, 2, 1], "dtype": "float32", "image_normalization": "minus-one-one"},
            "q_ids": {"shape": [1, 4], "dtype": "int32"},
            "router_keep": {"shape": [1, 4], "dtype": "int32"},
            "memory_keep": {"shape": [1, 7], "dtype": "int32"},
            "y_ids": {"shape": [1, 3], "dtype": "int32"},
            "y_keep": {"shape": [1, 3], "dtype": "int32"},
        }
        graph = graph_document(
            inputs, "token_ids", [1, 3], output_dtype="float32",
            package_class=self.family_precision,
            nodes=[
                {
                    "id": "flatten_image", "opType": "Reshape",
                    "inputs": {"input": "image"}, "outputs": {"out": "image_features"},
                    "outputs_shape": {"out": [1, 4]}, "outputs_dtype": {"out": "float32"}, "params": {},
                },
                {
                    "id": "decoder_logits", "opType": "Linear",
                    "inputs": {
                        "input": "image_features", "weight": "family_weight", "bias": "family_bias",
                        **({"weight_scale": "family_weight_scale"} if self.family_precision == "w8a32" else {}),
                    },
                    "outputs": {"out": "token_ids"},
                    "outputs_shape": {"out": [1, 3]}, "outputs_dtype": {"out": "float32"},
                    "params": {"weight_layout": "OUT_IN" if self.family_precision == "w8a32" else "IN_OUT"},
                },
            ],
        )
        if self.family_precision == "w8a32":
            graph["quantization"] = {
                "format": "volvox-affine-safetensors/v1",
                "tensors": {
                    "family_weight": {
                        "scheme": "per_axis",
                        "axis": 0,
                        "scale_tensor": "family_weight_scale",
                        "zero_point_tensor": "family_weight_zero_point",
                    }
                },
            }
        return graph

    def router_tensors(self):
        if self.router_precision == "w8a8-v1":
            return {
                "router_embedding": np.zeros((16, 4), dtype=np.int8),
                "router_embedding_scale": np.full(16, 0.25, dtype=np.float32),
                "router_embedding_zero_point": np.zeros(16, dtype=np.int8),
                "router_weight": np.zeros((8, 4), dtype=np.int8),
                "router_weight_scale": np.full(8, 0.25, dtype=np.float32),
                "router_weight_zero_point": np.zeros(8, dtype=np.int8),
                "router_activation_scale": np.asarray([0.125], dtype=np.float32),
                "router_activation_zero_point": np.zeros(1, dtype=np.int8),
                "router_bias": np.zeros(8, dtype=np.int32),
            }
        return {
            "router_embedding": np.zeros((16, 4), dtype=np.float32),
            "router_weight": np.zeros((4, 8), dtype=np.float32),
            "router_bias": np.zeros(8, dtype=np.float32),
        }

    def family_tensors(self):
        if self.family_precision == "w8a32":
            return {
                "family_weight": np.zeros((3, 4), dtype=np.int8),
                "family_weight_scale": np.ones(3, dtype=np.float32),
                "family_weight_zero_point": np.zeros(3, dtype=np.int8),
                "family_bias": np.zeros(3, dtype=np.float32),
            }
        return {
            "family_weight": np.zeros((4, 3), dtype=np.float32),
            "family_bias": np.zeros(3, dtype=np.float32),
        }

    def write_router(self, graph=None):
        self.write_export(
            self.router, self.router_graph() if graph is None else graph,
            self.router_tensors(),
        )

    def write_family(self, family, graph=None):
        self.write_export(
            self.families[family], self.family_graph() if graph is None else graph,
            self.family_tensors(),
        )

    def assemble(self, destination: Path, **overrides):
        options = {
            "router_mask_mode": self.mask_mode,
            "router_mask_input": self.mask_name,
            "router_precision": self.router_precision,
            "family_precision": self.family_precision,
            "router_profiles": ["portable", "native-cpu", "portable"],
            "family_profiles": ["native-cpu", "portable"],
            "producer_hashes": {"suite_onnx": "a" * 64},
        }
        options.update(overrides)
        return assembler.assemble_onnx_suite(
            self.router, self.families, destination, **options
        )


class TinyReceiptOnnxSuiteAssemblerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="volvoxai-tiny-receipt-onnx-suite-"
        )
        self.root = Path(self.temporary.name)
        self.fixture = OnnxSuiteFixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_assembles_exact_suite_without_transforming_graphs_or_weights(self):
        destination = self.root / "assembled"
        source_router_graph = (self.fixture.router / assembler.GRAPH_FILENAME).read_bytes()
        source_router_weights = (self.fixture.router / assembler.WEIGHTS_FILENAME).read_bytes()

        result = self.fixture.assemble(destination)

        self.assertEqual(result.output_dir, destination)
        self.assertEqual(result.manifest_path, destination / "package_manifest.json")
        self.assertEqual(result.router_graph_path.read_bytes(), source_router_graph)
        self.assertEqual(
            (destination / "router" / assembler.WEIGHTS_FILENAME).read_bytes(),
            source_router_weights,
        )
        self.assertEqual(len(result.family_graph_paths), len(assembler.FAMILY_ORDER))

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["format"], assembler.PACKAGE_FORMAT)
        self.assertEqual(manifest["family_order"], list(assembler.FAMILY_ORDER))
        self.assertNotIn("suite_profiles", manifest)
        self.assertEqual(
            manifest["target_qualification"],
            {
                "offline_common_targets": [
                    "cpu-js", "wasm", "webgpu", "native-cpu"
                ],
                "qualified_suite_profiles": [],
                "strict_no_fallback_execution": "not-run",
                "reason": (
                    "the assembler performs descriptor validation but receives no "
                    "strict runtime execution attestation"
                ),
            },
        )
        self.assertEqual(manifest["router"]["precision"], "w8a8-v1")
        self.assertEqual(
            manifest["router"]["requested_profiles"],
            ["portable", "native-cpu"],
        )
        self.assertEqual(
            manifest["router"]["semantic_mask"],
            {
                "input": "router_keep",
                "mode": "keep-mask-i32",
                "semantics": "I32 nonzero exactly for kept question tokens",
            },
        )
        self.assertEqual(
            manifest["explicit_family_contract"]["precision"], "w8a32"
        )
        self.assertEqual(
            manifest["routing"]["id_to_family"],
            [
                {"id": index, "family": family}
                for index, family in enumerate(assembler.FAMILY_ORDER)
            ],
        )
        self.assertEqual(set(manifest["explicit_families"]), set(assembler.FAMILY_ORDER))
        self.assertEqual(manifest["producer_hashes"], {"suite_onnx": "a" * 64})

        router_record = manifest["router"]
        self.assertEqual(
            router_record["graph"]["sha256"],
            self.sha256(destination / router_record["graph"]["file"]),
        )
        self.assertEqual(
            router_record["weights"]["sha256"],
            self.sha256(destination / router_record["weights"]["file"]),
        )
        for family in assembler.FAMILY_ORDER:
            source = self.fixture.families[family]
            copied = destination / "families" / family
            self.assertEqual(
                (copied / assembler.GRAPH_FILENAME).read_bytes(),
                (source / assembler.GRAPH_FILENAME).read_bytes(),
            )
            self.assertEqual(
                (copied / assembler.WEIGHTS_FILENAME).read_bytes(),
                (source / assembler.WEIGHTS_FILENAME).read_bytes(),
            )

    def test_accepts_explicit_normalized_fp32_router_weights_mode(self):
        fixture = OnnxSuiteFixture(
            self.root / "normalized-router",
            mask_mode="normalized-weights-f32",
            router_precision="fp32",
            family_precision="fp32",
        )
        result = fixture.assemble(
            self.root / "normalized-assembled",
            router_precision="fp32",
            family_precision="fp32",
            router_profiles=["browser"],
            family_profiles=["browser"],
        )
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["router"]["semantic_mask"],
            {
                "input": "router_weights",
                "mode": "normalized-weights-f32",
                "semantics": "F32 normalized token weights with padded tokens equal to zero",
            },
        )

    def test_rejects_missing_extra_or_duplicate_family_mappings(self):
        missing = dict(self.fixture.families)
        missing.pop("phone")
        with self.assertRaisesRegex(assembler.AssemblyError, "missing.*phone"):
            assembler.assemble_onnx_suite(
                self.fixture.router,
                missing,
                self.root / "missing-output",
                router_mask_mode="keep-mask-i32",
                router_mask_input="router_keep",
                router_precision="w8a8-v1",
                family_precision="w8a8-v1",
                router_profiles=["portable"],
                family_profiles=["portable"],
            )
        self.assertFalse((self.root / "missing-output").exists())

        extra = dict(self.fixture.families)
        extra["invented"] = self.fixture.families["phone"]
        with self.assertRaisesRegex(assembler.AssemblyError, "unexpected.*invented"):
            assembler.assemble_onnx_suite(
                self.fixture.router,
                extra,
                self.root / "extra-output",
                router_mask_mode="keep-mask-i32",
                router_mask_input="router_keep",
                router_precision="w8a8-v1",
                family_precision="w8a8-v1",
                router_profiles=["portable"],
                family_profiles=["portable"],
            )
        with self.assertRaisesRegex(assembler.AssemblyError, "duplicate family mapping"):
            assembler._family_assignments(
                ["phone=/first", "phone=/second"]
            )

    def test_rejects_missing_canonical_artifacts_before_publication(self):
        (self.fixture.families["store"] / assembler.WEIGHTS_FILENAME).unlink()
        destination = self.root / "missing-artifact-output"
        with self.assertRaisesRegex(
            assembler.AssemblyError, "family store export is missing model.safetensors"
        ):
            self.fixture.assemble(destination)
        self.assertFalse(destination.exists())

    def test_rejects_any_family_input_or_output_schema_drift(self):
        input_drift = self.fixture.family_graph()
        input_drift["inputs"]["q_ids"]["shape"] = [1, 5]
        self.fixture.write_family("math", input_drift)
        destination = self.root / "input-drift-output"
        with self.assertRaisesRegex(assembler.AssemblyError, "family math.*differs"):
            self.fixture.assemble(destination)
        self.assertFalse(destination.exists())

        self.fixture.write_family("math")
        output_drift = self.fixture.family_graph()
        next(node for node in output_drift["nodes"] if node["id"] == "decoder_logits")[
            "outputs_shape"
        ]["out"] = [1, 2]
        self.fixture.write_family("other", output_drift)
        with self.assertRaisesRegex(assembler.AssemblyError, "family other.*differs"):
            self.fixture.assemble(self.root / "output-drift-output")

    def test_rejects_implicit_node_output_dtype(self):
        graph = self.fixture.family_graph()
        graph["nodes"][0].pop("outputs_dtype")
        self.fixture.write_family("phone", graph)
        with self.assertRaisesRegex(
            assembler.AssemblyError,
            r"nodes\[0\]\.outputs_dtype must be an object",
        ):
            self.fixture.assemble(self.root / "implicit-dtype-output")

    def test_rejects_router_semantic_contract_mismatch(self):
        destination = self.root / "bad-router-output"
        with self.assertRaisesRegex(
            assembler.AssemblyError, "q_ids and semantic mask inputs must be distinct"
        ):
            self.fixture.assemble(
                destination,
                router_mask_input="q_ids",
            )
        self.assertFalse(destination.exists())

        bad_route = self.fixture.router_graph()
        bad_route["nodes"][-1]["outputs_dtype"]["out"] = "float32"
        self.fixture.write_router(bad_route)
        with self.assertRaisesRegex(
            assembler.AssemblyError, "family ID output must be unquantized int32"
        ):
            self.fixture.assemble(self.root / "bad-route-output")

        unattested = self.fixture.router_graph()
        unattested["source"].pop("features")
        self.fixture.write_router(unattested)
        with self.assertRaisesRegex(
            assembler.AssemblyError, "lacks an exact masked-mean.*feature attestation"
        ):
            self.fixture.assemble(self.root / "unattested-router-output")

    def test_rejects_nonterminal_or_wrong_width_router_argmax(self):
        wrong_width = self.fixture.router_graph()
        next(node for node in wrong_width["nodes"] if node["id"] == "router_logits")[
            "outputs_shape"
        ]["out"] = [1, 7]
        self.fixture.write_router(wrong_width)
        with self.assertRaisesRegex(
            assembler.AssemblyError, r"\[1,8\] logits.*final family axis"
        ):
            self.fixture.assemble(self.root / "wrong-width")

        nonterminal = self.fixture.router_graph()
        nonterminal["nodes"].append(
            {
                "id": "dispatch_inside_graph",
                "opType": "Identity",
                "inputs": {"input": "family_id"},
                "outputs": {"out": "dispatched"},
                "outputs_shape": {"out": [1]},
                "outputs_dtype": {"out": "int32"},
                "params": {},
            }
        )
        self.fixture.write_router(nonterminal)
        with self.assertRaisesRegex(assembler.AssemblyError, "terminate the graph"):
            self.fixture.assemble(self.root / "nonterminal")

    def test_precision_and_profiles_are_validated_from_exported_graphs(self):
        mismatched = self.fixture.router_graph()
        mismatched["source"]["package_class"] = "fp32"
        self.fixture.write_router(mismatched)
        with self.assertRaisesRegex(assembler.AssemblyError, "does not match exported"):
            self.fixture.assemble(self.root / "precision-mismatch")

        self.fixture.write_router()
        with self.assertRaisesRegex(assembler.AssemblyError, "must be one of"):
            self.fixture.assemble(
                self.root / "unknown-profile", router_profiles=["invented"]
            )

        result = self.fixture.assemble(
            self.root / "expanded-intersection",
            router_profiles=["portable"],
            family_profiles=["browser"],
        )
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["target_qualification"]["offline_common_targets"],
            ["cpu-js", "wasm", "webgpu"],
        )
        self.assertEqual(
            manifest["target_qualification"]["strict_no_fallback_execution"],
            "not-run",
        )

    def test_rejects_unqualified_or_malformed_metadata(self):
        with self.assertRaisesRegex(assembler.AssemblyError, "nonempty target intersection"):
            self.fixture.assemble(
                self.root / "disjoint",
                router_profiles=["browser"],
                family_profiles=["native-cpu"],
            )
        with self.assertRaisesRegex(assembler.AssemblyError, "router precision must match"):
            self.fixture.assemble(
                self.root / "bad-precision", router_precision="FP32"
            )
        with self.assertRaisesRegex(assembler.AssemblyError, "lowercase SHA-256"):
            self.fixture.assemble(
                self.root / "bad-hash",
                producer_hashes={"router_onnx": "A" * 64},
            )

    def test_rejects_legacy_safetensors_quantization_metadata(self):
        self.fixture.write_export(
            self.fixture.router,
            self.fixture.router_graph(),
            self.fixture.router_tensors(),
            metadata={"weights_quantization": "{}"},
        )

        with self.assertRaisesRegex(
            assembler.AssemblyError,
            "legacy safetensors quantization metadata",
        ):
            self.fixture.assemble(self.root / "legacy-metadata")

    def test_manifest_is_deterministic_for_identical_inputs(self):
        first = self.fixture.assemble(self.root / "first")
        second = self.fixture.assemble(self.root / "second")
        self.assertEqual(
            first.manifest_path.read_bytes(), second.manifest_path.read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
