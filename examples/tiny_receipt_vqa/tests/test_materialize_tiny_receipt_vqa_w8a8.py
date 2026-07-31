import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.exporter.quantization_storage import (
    QUANTIZATION_FORMAT,
    validate_external_quantization,
)


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = EXAMPLE_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
except ImportError as error:  # pragma: no cover - environment dependency.
    raise unittest.SkipTest(f"TinyReceipt materializer dependencies are unavailable: {error}")


materializer = importlib.import_module("materialize_tiny_receipt_vqa_w8a8")


class TinyReleaseFixture:
    """Tiny but structurally complete named SafeTensors source release."""

    def __init__(self, root: Path):
        self.root = root
        self.release = root / "int8"
        self.release.mkdir()
        self.manifest_path = self.release / "manifest.json"
        self.weights_path = self.release / "model_int8.safetensors"
        self.config = {
            "vocab_size": 7,
            "d_model": 4,
            "heads": 1,
            "enc_layers": 1,
            "dec_layers": 1,
            "ff_mult": 1,
            "dropout": 0.0,
            "max_q_len": 2,
            "max_out_len": 2,
            "img_tokens": 210,
            "use_adapters": True,
            "use_router": True,
            "adapter_bottleneck": 2,
            "adapter_families": 8,
            "lora_r": 1,
            "lora_alpha": 2.0,
            "lora_dropout": 0.0,
            "lora_targets": "transformer",
        }
        self.dimensions = materializer._validate_dimensions(self.config)
        self.expected = materializer._expected_source_shapes(self.dimensions)
        self.named = {}

    def write(self):
        manifest_tensors = {}
        for offset, (key, (dtype, shape)) in enumerate(self.expected.items()):
            if dtype == np.dtype(np.int8):
                values = ((np.arange(np.prod(shape), dtype=np.int32) + offset) % 15 - 7).astype(np.int8).reshape(shape)
                scales = np.full((shape[0],), np.float32(0.25), dtype=np.float32)
                self.named[f"quantized.{key}"] = values
                self.named[f"scale.{key}"] = scales
                manifest_tensors[key] = {"dtype": "int8_per_out", "shape": list(shape)}
            else:
                values = np.full(shape, np.float32(0.0625), dtype=np.float32)
                self.named[f"float32.{key}"] = values
                manifest_tensors[key] = {"dtype": "float32", "shape": list(shape)}

        # Keep the fixture's source tying exact while making the head-bias
        # conversion observable at a half integer accumulator boundary.
        self.named["quantized.head.weight"] = self.named["quantized.tok.weight"].copy()
        self.named["scale.head.weight"] = self.named["scale.tok.weight"].copy()
        self.named["float32.out_bias"] = np.full((7,), np.float32(0.046875), dtype=np.float32)
        self.named["float32.img_pos"] = np.full((1, 210, 4), np.float32(0.02), dtype=np.float32)
        self.named["float32.q_pos"] = np.full((1, 2, 4), np.float32(0.015), dtype=np.float32)
        self.named["float32.y_pos"] = np.full((1, 2, 4), np.float32(0.01), dtype=np.float32)
        self.named["float32.type_img"] = np.full((1, 1, 4), np.float32(0.01), dtype=np.float32)
        self.named["float32.type_q"] = np.full((1, 1, 4), np.float32(0.01), dtype=np.float32)

        manifest = {
            "format": materializer.SOURCE_FORMAT,
            "runtime": materializer.SOURCE_RUNTIME,
            "files": {"model_safetensors": self.weights_path.name},
            "safetensors": {
                "layout": materializer.SOURCE_LAYOUT,
                "quantization": materializer.SOURCE_QUANTIZATION,
            },
            "config": self.config,
            "vocab": {"itos": ["<pad>", "<bos>", "<eos>", "<unk>", "a", "b", "c"]},
            "tensors": manifest_tensors,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        save_file(self.named, str(self.weights_path))
        return self


def assert_no_implicit_inputs(test: unittest.TestCase, graph: dict, weight_names: set[str]):
    available = set(graph["inputs"]) | set(weight_names)
    for node in graph["nodes"]:
        for input_name in node["inputs"].values():
            test.assertIn(input_name, available, f"{node['id']} references implicit input {input_name}")
        available.update(node["outputs"].values())


def referenced_i8_weights(graph: dict, weights: dict[str, np.ndarray]) -> set[str]:
    table = graph.get("quantization", {}).get("tensors", {})
    parameters = {
        descriptor[key]
        for descriptor in table.values()
        for key in ("scale_tensor", "zero_point_tensor")
    }
    return {
        name
        for node in graph["nodes"]
        for name in node["inputs"].values()
        if (
            name not in parameters
            and name in weights
            and weights[name].dtype == np.dtype(np.int8)
        )
    }


def load_scoped_weights(root: Path, record: dict) -> dict[str, np.ndarray]:
    paths = record.get("weight_files")
    if not isinstance(paths, list) or not paths:
        raise AssertionError("graph record requires non-empty weight_files")
    merged: dict[str, np.ndarray] = {}
    for relative in paths:
        if not isinstance(relative, str) or not relative:
            raise AssertionError("weight_files entries must be non-empty strings")
        shard = load_file(str(root / relative))
        duplicates = set(merged).intersection(shard)
        if duplicates:
            raise AssertionError(f"scoped weight files overlap: {sorted(duplicates)!r}")
        merged.update(shard)
    return merged


class TinyReceiptMaterializerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-tiny-receipt-materializer-")
        self.root = Path(self.temporary.name)
        self.fixture = TinyReleaseFixture(self.root).write()

    def tearDown(self):
        self.temporary.cleanup()

    def test_pure_transforms_preserve_layout_split_and_nearest_even(self):
        conv = np.arange(2 * 3 * 2 * 2, dtype=np.int8).reshape(2, 3, 2, 2)
        transformed = materializer.transpose_conv_oihw_to_ohwi(conv)
        self.assertEqual(transformed.shape, (2, 2, 2, 3))
        self.assertTrue(np.array_equal(transformed, np.transpose(conv, (0, 2, 3, 1))))

        packed = np.arange(3 * 4 * 4, dtype=np.int8).reshape(12, 4)
        bias = np.arange(12, dtype=np.float32)
        scales = np.arange(1, 13, dtype=np.float32) / 10
        (q_weight, q_bias, q_scale), (k_weight, k_bias, k_scale), (v_weight, v_bias, v_scale) = \
            materializer.split_mha_in_proj(packed, bias, scales)
        self.assertTrue(np.array_equal(q_weight, packed[:4]))
        self.assertTrue(np.array_equal(k_weight, packed[4:8]))
        self.assertTrue(np.array_equal(v_weight, packed[8:]))
        self.assertTrue(np.array_equal(q_bias, bias[:4]))
        self.assertTrue(np.array_equal(k_scale, scales[4:8]))
        self.assertTrue(np.array_equal(v_scale, scales[8:]))

        rounded = materializer.round_ties_to_even_i32(
            np.asarray([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=np.float32), "fixture"
        )
        self.assertEqual(rounded.tolist(), [0, 2, 2, 0, -2, -2])

    def test_accepts_distinct_volvox_trained_source_contract_and_dynamic_smoke_stem(self):
        alternate_root = self.root / "volvox-trained"
        alternate_root.mkdir()
        alternate = TinyReleaseFixture(alternate_root)
        alternate.config["stem_channels"] = [2, 3, 4, 4, 4]
        alternate.dimensions = materializer._validate_dimensions(alternate.config)
        alternate.expected = materializer._expected_source_shapes(alternate.dimensions)
        alternate.write()
        manifest = json.loads(alternate.manifest_path.read_text(encoding="utf-8"))
        manifest["format"] = materializer.VOLVOX_TRAINED_SOURCE_FORMAT
        manifest["runtime"] = materializer.VOLVOX_TRAINED_SOURCE_RUNTIME
        manifest["safetensors"]["layout"] = materializer.VOLVOX_TRAINED_SOURCE_LAYOUT
        for key, (dtype, shape) in alternate.expected.items():
            if dtype == np.dtype(np.int8):
                alternate.named[f"quantized.{key}.zero_point"] = np.zeros(
                    (shape[0],), dtype=np.int8
                )
        alternate.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        save_file(alternate.named, str(alternate.weights_path))

        release = materializer.load_release(alternate.manifest_path)
        self.assertEqual(release.dimensions.stem_channels, (2, 3, 4, 4, 4))
        result = materializer.materialize_release(
            alternate.manifest_path, alternate_root / "package", activation_scale=0.125
        )
        package = json.loads(result.package_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(package["source"]["format"], materializer.VOLVOX_TRAINED_SOURCE_FORMAT)
        self.assertEqual(package["source"]["runtime"], materializer.VOLVOX_TRAINED_SOURCE_RUNTIME)
        weights = load_file(str(result.weights_path))
        self.assertEqual(weights["w.stem.0.net.0.weight"].shape, (2, 3, 3, 1))

    def test_rejects_quantization_parameter_name_collisions(self):
        release = materializer.load_release(self.fixture.manifest_path)
        occupied = materializer.WeightBuilder(release)
        occupied._insert("collision_scale", np.ones((2,), dtype=np.float32))
        occupied._insert("collision", np.ones((2, 2), dtype=np.int8))
        with self.assertRaisesRegex(materializer.MaterializationError, "quantization parameter.*collides"):
            occupied._register_per_axis_quantization(
                "collision", np.asarray([0.25, 0.5], dtype=np.float32)
            )

        reserved = materializer.WeightBuilder(release)
        reserved._insert("collision", np.ones((2, 2), dtype=np.int8))
        reserved._register_per_axis_quantization(
            "collision", np.asarray([0.25, 0.5], dtype=np.float32)
        )
        with self.assertRaisesRegex(materializer.MaterializationError, "collides with a quantization parameter"):
            reserved._insert("collision_scale", np.ones((2,), dtype=np.float32))

    def test_materializes_complete_typed_router_and_eight_family_graphs(self):
        out_dir = self.root / "package"
        result = materializer.materialize_release(
            self.fixture.manifest_path, out_dir, activation_scale=0.125
        )
        self.assertEqual(result.constant_saturation_count, 0)
        package = json.loads(result.package_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(package["format"], materializer.PACKAGE_FORMAT)
        self.assertEqual(package["weights"]["file"], "model.safetensors")
        self.assertEqual(
            package["router"]["weight_files"],
            [f"{materializer.SCOPED_WEIGHTS_DIR}/router.safetensors"],
        )
        self.assertEqual(package["router"]["inputs"], {"q_ids": "q_ids", "router_keep": "router_keep"})
        self.assertEqual(package["vocab"]["token_ids"], {"pad": 0, "bos": 1, "eos": 2, "unk": 3})
        self.assertEqual(package["family_order"], list(materializer.FAMILY_ORDER))
        self.assertEqual(set(package["explicit_families"]), set(materializer.FAMILY_ORDER))
        for family in materializer.FAMILY_ORDER:
            self.assertEqual(
                package["explicit_families"][family]["weight_files"],
                [
                    f"{materializer.SCOPED_WEIGHTS_DIR}/common.safetensors",
                    f"{materializer.SCOPED_WEIGHTS_DIR}/explicit_family_{family}.safetensors",
                ],
            )
        self.assertEqual(
            package["explicit_families"]["phone"]["interface"]["inputs"],
            {"image": "image", "q_ids": "q_ids", "router_keep": "router_keep", "memory_keep": "memory_keep", "y_ids": "y_ids", "y_keep": "y_keep"},
        )
        self.assertEqual(package["preprocessing"]["model_input_dtype"], "float32")

        weights = load_file(str(result.weights_path))
        with safe_open(str(result.weights_path), framework="numpy") as handle:
            weights_metadata = handle.metadata() or {}
        self.assertEqual(weights_metadata["format"], materializer.WEIGHTS_FORMAT)
        self.assertNotIn("weights_quantization_storage", weights_metadata)
        self.assertNotIn("weights_quantization", weights_metadata)
        scoped_metadata = {
            f"{materializer.SCOPED_WEIGHTS_DIR}/router.safetensors": "router",
            f"{materializer.SCOPED_WEIGHTS_DIR}/common.safetensors":
                "explicit-family-common",
            **{
                f"{materializer.SCOPED_WEIGHTS_DIR}/explicit_family_{family}.safetensors":
                    f"explicit-family:{family}"
                for family in materializer.FAMILY_ORDER
            },
        }
        for relative_path, scope in scoped_metadata.items():
            with safe_open(str(out_dir / relative_path), framework="numpy") as handle:
                self.assertEqual(
                    handle.metadata() or {},
                    {**weights_metadata, "scope": scope},
                )
        common_names = set(load_file(
            str(out_dir / materializer.SCOPED_WEIGHTS_DIR / "common.safetensors")
        ))
        for family in materializer.FAMILY_ORDER:
            family_only_names = set(load_file(str(
                out_dir
                / materializer.SCOPED_WEIGHTS_DIR
                / f"explicit_family_{family}.safetensors"
            )))
            self.assertTrue(common_names.isdisjoint(family_only_names), family)
        source_conv = self.fixture.named["quantized.stem.0.net.0.weight"]
        self.assertTrue(np.array_equal(weights["w.stem.0.net.0.weight"], np.transpose(source_conv, (0, 2, 3, 1))))
        source_mha = self.fixture.named["quantized.encoder.layers.0.self_attn.in_proj_weight"]
        self.assertTrue(np.array_equal(weights["w.encoder.layers.0.self_attn.q.weight"], source_mha[:4]))
        self.assertTrue(np.array_equal(weights["w.encoder.layers.0.self_attn.k.weight"], source_mha[4:8]))
        self.assertTrue(np.array_equal(weights["w.encoder.layers.0.self_attn.v.weight"], source_mha[8:]))
        self.assertIn("w.tok.weight", weights)
        self.assertNotIn("w.head.weight", weights)
        # 0.046875 / (0.125 * 0.25) = 1.5, so the generated accumulator
        # bias must use ties-to-even rather than half-away-from-zero.
        self.assertEqual(weights["b.family_phone.decoder.head.i32"].tolist(), [2] * 7)
        self.assertEqual(weights["c.img_pos"].dtype, np.dtype(np.int8))

        router = json.loads(result.router_graph_path.read_text(encoding="utf-8"))
        phone_path = out_dir / package["explicit_families"]["phone"]["graph"]
        phone = json.loads(phone_path.read_text(encoding="utf-8"))
        router_scoped = load_scoped_weights(out_dir, package["router"])
        phone_scoped = load_scoped_weights(
            out_dir, package["explicit_families"]["phone"]
        )
        self.assertEqual(
            set(router_scoped), materializer._graph_weight_names(router, weights)
        )
        self.assertEqual(
            set(phone_scoped), materializer._graph_weight_names(phone, weights)
        )
        validate_external_quantization(router, router_scoped)
        validate_external_quantization(phone, phone_scoped)
        assert_no_implicit_inputs(self, router, set(router_scoped))
        assert_no_implicit_inputs(self, phone, set(phone_scoped))
        for graph, terminal in ((router, "router_family"), (phone, "token_ids")):
            self.assertEqual(graph["format"], materializer.GRAPH_FORMAT)
            self.assertEqual(graph["graph_profile"], materializer.W8A8_GRAPH_PROFILE)
            self.assertEqual(graph["outputs"], [terminal])
            self.assertEqual(graph["output_contract"]["name"], terminal)
            self.assertTrue(graph["activation_edge_bindings"])
            self.assertEqual(graph["quantization"]["format"], QUANTIZATION_FORMAT)
            self.assertNotIn("weights_quantization", graph)
            self.assertNotIn("weights_quantization_storage", graph)
            resolved = validate_external_quantization(graph, weights)
            referenced_weights = referenced_i8_weights(graph, weights)
            self.assertLessEqual(referenced_weights, set(resolved))
            for node in graph["nodes"]:
                if node["opType"] == "QArgMax":
                    self.assertEqual(node["outputs_dtype"]["out"], "int32")
                else:
                    self.assertEqual(node["outputs_dtype"]["out"], "int8")
                    descriptor = resolved[node["outputs"]["out"]]
                    self.assertEqual(descriptor.scheme, "per_tensor")
                    self.assertEqual(descriptor.zero_points.tolist(), [0])
                    self.assertNotIn("outputs_quantization", node)
            assert_no_implicit_inputs(self, graph, set(weights))
        self.assertEqual(phone["inputs"]["image"]["dtype"], "float32")
        self.assertEqual(phone["nodes"][0]["opType"], "QuantizeLinear")
        self.assertEqual(phone["nodes"][-1]["opType"], "QArgMax")
        self.assertEqual(phone["nodes"][-1]["outputs"]["out"], "token_ids")
        self.assertEqual(phone["nodes"][-2]["inputs"]["weight"], "w.tok.weight")

        router_i8_weights = referenced_i8_weights(router, weights)
        phone_i8_weights = referenced_i8_weights(phone, weights)
        self.assertEqual(router_i8_weights, {
            "c.q_pos",
            "c.q_type",
            "w.router.net.0.weight",
            "w.router.net.2.weight",
            "w.tok.weight",
        })
        self.assertIn("w.memory_adapters.0.down.weight", phone_i8_weights)
        self.assertNotIn("w.memory_adapters.1.down.weight", phone_i8_weights)

        all_graphs = [router] + [
            json.loads(path.read_text(encoding="utf-8"))
            for path in result.explicit_graph_paths
        ]
        all_scoped_names = set(router_scoped)
        for family, graph in zip(materializer.FAMILY_ORDER, all_graphs[1:]):
            scoped = load_scoped_weights(
                out_dir, package["explicit_families"][family]
            )
            self.assertEqual(
                set(scoped), materializer._graph_weight_names(graph, weights)
            )
            validate_external_quantization(graph, scoped)
            assert_no_implicit_inputs(self, graph, set(scoped))
            all_scoped_names.update(scoped)
        self.assertEqual(all_scoped_names, set(weights))
        all_referenced_i8_weights = set().union(
            *(referenced_i8_weights(graph, weights) for graph in all_graphs)
        )
        graph_tables = [graph["quantization"]["tensors"] for graph in all_graphs]
        declared_i8_weights = {
            name
            for table in graph_tables
            for name in table
            if name in weights and weights[name].dtype == np.dtype(np.int8)
        }
        self.assertEqual(declared_i8_weights, all_referenced_i8_weights)
        for graph in all_graphs:
            table = graph["quantization"]["tensors"]
            resolved = validate_external_quantization(graph, weights)
            for name, descriptor in table.items():
                allowed = {"scheme", "scale_tensor", "zero_point_tensor"}
                if descriptor["scheme"] == "per_axis":
                    allowed.add("axis")
                self.assertEqual(set(descriptor), allowed)
                self.assertIn(name, resolved)
                self.assertEqual(weights[descriptor["scale_tensor"]].dtype, np.dtype(np.float32))
                target_dtype = weights[name].dtype if name in weights else np.dtype(np.int8)
                self.assertEqual(weights[descriptor["zero_point_tensor"]].dtype, target_dtype)

        expected_scale_tensors = {
            "w.tok.weight": self.fixture.named["scale.tok.weight"],
            "w.stem.0.net.0.weight": self.fixture.named["scale.stem.0.net.0.weight"],
            "w.encoder.layers.0.self_attn.q.weight":
                self.fixture.named["scale.encoder.layers.0.self_attn.in_proj_weight"][:4],
            "w.encoder.layers.0.linear1.lora_b.weight": np.asarray(
                self.fixture.named["scale.encoder.layers.0.linear1.lora_b.weight"]
                * self.fixture.dimensions.lora_scale,
                dtype=np.float32,
            ),
        }
        for name, expected_scales in expected_scale_tensors.items():
            self.assertIn(name, declared_i8_weights)
            self.assertTrue(np.array_equal(weights[f"{name}_scale"], expected_scales), name)
        q_pos_scale, _ = materializer.derive_static_constant_i8_scale(
            self.fixture.named["float32.q_pos"]
        )
        q_pos_descriptor = router["quantization"]["tensors"]["c.q_pos"]
        self.assertEqual(q_pos_descriptor["scheme"], "per_tensor")
        self.assertTrue(np.array_equal(
            weights["c.q_pos_scale"], np.asarray([q_pos_scale], dtype=np.float32)
        ))

    def test_materializes_hybrid_w8a8_router_and_w8a32_family_graphs(self):
        out_dir = self.root / "w8a32-package"
        result = materializer.materialize_release(
            self.fixture.manifest_path,
            out_dir,
            activation_scale=0.125,
            family_execution="w8a32",
        )
        self.assertEqual(result.family_execution, "w8a32")
        package = json.loads(result.package_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(package["format"], materializer.PACKAGE_FORMAT)
        self.assertEqual(package["execution_variant"], {
            "router": "w8a8",
            "explicit_families": "w8a32",
            "hybrid": True,
            "selection_time": "package_materialization",
        })
        self.assertEqual(package["activation_scale_profile"]["scope"], "router_only")
        self.assertEqual(
            package["router"]["weight_files"],
            [f"{materializer.SCOPED_WEIGHTS_DIR}/router.safetensors"],
        )
        self.assertEqual(
            package["explicit_families"]["phone"]["weight_files"],
            [
                f"{materializer.SCOPED_WEIGHTS_DIR}/common.safetensors",
                f"{materializer.SCOPED_WEIGHTS_DIR}/explicit_family_phone.safetensors",
            ],
        )

        weights = load_file(str(result.weights_path))
        with safe_open(str(result.weights_path), framework="numpy") as handle:
            weights_metadata = handle.metadata() or {}
        self.assertEqual(weights_metadata["format"], materializer.W8A32_WEIGHTS_FORMAT)
        self.assertNotIn("weights_quantization_storage", weights_metadata)
        self.assertNotIn("weights_quantization", weights_metadata)
        self.assertEqual(weights["f.tok.embedding"].dtype, np.dtype(np.float32))
        expected_embedding = (
            self.fixture.named["quantized.tok.weight"].astype(np.float32)
            * self.fixture.named["scale.tok.weight"][:, np.newaxis]
        )
        self.assertTrue(np.array_equal(weights["f.tok.embedding"], expected_embedding))
        self.assertTrue(np.array_equal(weights["c.img_pos.f32"], self.fixture.named["float32.img_pos"]))
        self.assertTrue(np.array_equal(
            weights["w.tok.weight_scale"], self.fixture.named["scale.tok.weight"]
        ))
        self.assertNotIn("s.w.tok.weight", weights)
        # Unlike W8A8 accumulator bias conversion, W8A32 preserves the exact
        # source value at the deliberately chosen 1.5-accumulator boundary.
        self.assertEqual(weights["b.f32.tok.weight"].tolist(), [0.046875] * 7)

        router = json.loads(result.router_graph_path.read_text(encoding="utf-8"))
        phone = json.loads((out_dir / "explicit_family_phone.graph.json").read_text(encoding="utf-8"))
        self.assertEqual(router["format"], materializer.GRAPH_FORMAT)
        self.assertEqual(router["graph_profile"], materializer.W8A8_GRAPH_PROFILE)
        self.assertEqual(router["quantization"]["format"], QUANTIZATION_FORMAT)
        self.assertNotIn("weights_quantization", router)
        self.assertNotIn("weights_quantization_storage", router)
        validate_external_quantization(router, weights)
        validate_external_quantization(
            router, load_scoped_weights(out_dir, package["router"])
        )
        self.assertTrue(any(node["opType"] == "QLinear" for node in router["nodes"]))
        self.assertEqual(phone["format"], materializer.GRAPH_FORMAT)
        self.assertEqual(phone["graph_profile"], materializer.W8A32_GRAPH_PROFILE)
        self.assertNotIn("weights_quantization_storage", phone)
        self.assertNotIn("weights_quantization", phone)
        self.assertEqual(phone["quantization"]["format"], QUANTIZATION_FORMAT)
        validate_external_quantization(phone, weights)
        validate_external_quantization(
            phone,
            load_scoped_weights(
                out_dir, package["explicit_families"]["phone"]
            ),
        )
        self.assertEqual(phone["activation_edge_bindings"], [])
        self.assertFalse(any(node["opType"] == "QuantizeLinear" for node in phone["nodes"]))
        self.assertFalse(any(node["opType"] in {
            "QAdd", "QEmbedding", "QGELU", "QGroupNorm", "QLayerNorm", "QSiLU", "QSDPA",
        } for node in phone["nodes"]))
        self.assertTrue(any(node["opType"] == "QConv2D" for node in phone["nodes"]))
        self.assertTrue(any(node["opType"] == "CrossSDPA" for node in phone["nodes"]))
        for node in phone["nodes"]:
            if node["opType"] == "Linear":
                self.assertIn("weight_scale", node["inputs"])
                self.assertEqual(node["inputs"]["weight_scale"], f"{node['inputs']['weight']}_scale")
                self.assertEqual(node["params"]["weight_layout"], "OUT_IN")
                self.assertEqual(weights[node["inputs"]["bias"]].dtype, np.dtype(np.float32))
            if node["opType"] == "QConv2D":
                self.assertTrue(node["params"]["weight_only"])
            if node["opType"] == "Embedding":
                self.assertEqual(node["inputs"]["weight"], "f.tok.embedding")
            if node["opType"] == "ArgMax":
                self.assertEqual(node["outputs_dtype"]["out"], "int32")
                self.assertEqual(node["params"], {"axis": -1, "keepdims": 0})
            else:
                self.assertEqual(node["outputs_dtype"]["out"], "float32")
                self.assertNotIn("outputs_quantization", node)
        self.assertEqual(phone["nodes"][-1]["opType"], "ArgMax")
        self.assertEqual(phone["nodes"][-2]["inputs"]["weight"], "w.tok.weight")
        assert_no_implicit_inputs(self, phone, set(weights))

    def test_rejects_broken_tied_head_before_writing_package(self):
        self.fixture.named["quantized.head.weight"] = self.fixture.named["quantized.head.weight"].copy()
        self.fixture.named["quantized.head.weight"][0, 0] ^= np.int8(1)
        save_file(self.fixture.named, str(self.fixture.weights_path))
        with self.assertRaisesRegex(materializer.MaterializationError, "tied tok.weight/head.weight"):
            materializer.materialize_release(
                self.fixture.manifest_path, self.root / "bad-package", activation_scale=0.125
            )
        self.assertFalse((self.root / "bad-package").exists())

    def test_development_profile_rejects_missing_bound_probe_and_marks_zero_only_reuse(self):
        calibration = self.root / "calibration.json"
        calibration.write_text(json.dumps({
            "format": "volvoxai-tiny-receipt-vqa-w8a8-development-calibration-v1",
            "profile": {
                "id": "fixture-profile",
                "activation_quantization": {
                    "global_fallback": {"scale": 0.25},
                    "probe_boundaries": {
                        "known": {
                            "dtype": "int8", "symmetric": True, "zero_point": 0, "scale": 0.125,
                            "observed_zero_only": True,
                            "scale_source": "global_fallback_for_zero_only_probe",
                        },
                    },
                },
            },
        }), encoding="utf-8")
        profile = materializer.ActivationProfile.from_development_calibration(calibration)
        self.assertEqual(float(profile.scale_for("known-edge", source_probe="known")), 0.125)
        with self.assertRaisesRegex(materializer.MaterializationError, "missing required source probe"):
            profile.scale_for("missing-edge", source_probe="missing")
        record = profile.manifest_record()
        self.assertEqual(record["zero_only_probe_global_fallback_logical_edges"], ["known-edge"])
        self.assertFalse(record["qualified_per_edge_calibration"])


if __name__ == "__main__":
    unittest.main()
