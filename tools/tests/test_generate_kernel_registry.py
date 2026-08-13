from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.generate_kernel_registry import (
    ROOT,
    Variant,
    load_registry,
    render_c,
    render_c_full,
    render_python,
    render_python_full,
    render_ts,
    render_ts_full,
)


REGISTRY_PROTO = ROOT / "proto/kernel_registry.proto"
PUBLIC_PROTO = ROOT / "proto/volvoxai.proto"


class KernelRegistryGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(REGISTRY_PROTO, PUBLIC_PROTO)

    def test_canonical_inventory_separates_runtime_and_exporter_support(self):
        by_id = {backend.runtime_id: backend for backend in self.registry.backends}
        self.assertEqual(len(by_id["cpu-js"].runtime_operators), 92)
        self.assertEqual(len(by_id["wasm"].runtime_operators), 92)
        self.assertEqual(len(by_id["webgpu"].runtime_operators), 87)
        self.assertEqual(len(by_id["webnn"].runtime_operators), 16)
        self.assertEqual(len(by_id["native-cpu"].runtime_operators), 77)
        self.assertEqual(len(by_id["nnapi"].runtime_operators), 3)
        self.assertEqual(len(by_id["cpu-js"].qualified_operators), 68)
        self.assertEqual(len(by_id["native-cpu"].qualified_operators), 66)
        self.assertEqual(len(by_id["webnn"].qualified_operators), 8)
        cuda_qualified = {
            "OPERATOR_KIND_ADD",
            "OPERATOR_KIND_ARG_MAX",
            "OPERATOR_KIND_BATCH_MATMUL",
            "OPERATOR_KIND_CAST",
            "OPERATOR_KIND_CLIP",
            "OPERATOR_KIND_CONCAT",
            "OPERATOR_KIND_CONV_2D",
            "OPERATOR_KIND_DEQUANTIZE_LINEAR",
            "OPERATOR_KIND_DIV",
            "OPERATOR_KIND_EMBEDDING",
            "OPERATOR_KIND_EQUAL",
            "OPERATOR_KIND_EXPAND",
            "OPERATOR_KIND_GELU",
            "OPERATOR_KIND_GATHER",
            "OPERATOR_KIND_GREATER_OR_EQUAL",
            "OPERATOR_KIND_GROUP_NORM",
            "OPERATOR_KIND_LAYER_NORM",
            "OPERATOR_KIND_LINEAR",
            "OPERATOR_KIND_MUL",
            "OPERATOR_KIND_NOT",
            "OPERATOR_KIND_Q_ADD",
            "OPERATOR_KIND_Q_ARG_MAX",
            "OPERATOR_KIND_Q_BATCH_MATMUL",
            "OPERATOR_KIND_Q_CONV_2D",
            "OPERATOR_KIND_Q_EMBEDDING",
            "OPERATOR_KIND_Q_GELU",
            "OPERATOR_KIND_Q_GEMM",
            "OPERATOR_KIND_Q_GROUP_NORM",
            "OPERATOR_KIND_Q_LAYER_NORM",
            "OPERATOR_KIND_Q_LINEAR",
            "OPERATOR_KIND_Q_MASKED_MEAN",
            "OPERATOR_KIND_Q_MATMUL",
            "OPERATOR_KIND_Q_SDPA",
            "OPERATOR_KIND_Q_SILU",
            "OPERATOR_KIND_QUANTIZE_LINEAR",
            "OPERATOR_KIND_REDUCE_SUM",
            "OPERATOR_KIND_REQUANTIZE_LINEAR",
            "OPERATOR_KIND_RESHAPE",
            "OPERATOR_KIND_SILU",
            "OPERATOR_KIND_SLICE",
            "OPERATOR_KIND_SOFTMAX",
            "OPERATOR_KIND_SQUEEZE",
            "OPERATOR_KIND_SUB",
            "OPERATOR_KIND_TRANSPOSE",
            "OPERATOR_KIND_UNSQUEEZE",
            "OPERATOR_KIND_WHERE",
        }
        self.assertEqual(set(by_id["cuda"].qualified_operators), cuda_qualified)
        self.assertEqual(by_id["native-cpu"].support_mode, "RUNTIME_SUPPORT_MODE_DYNAMIC")
        self.assertEqual(by_id["cuda"].support_mode, "RUNTIME_SUPPORT_MODE_DYNAMIC")
        newly_qualified = {
            "OPERATOR_KIND_RMS_NORM",
            "OPERATOR_KIND_BATCH_NORM_2D",
            "OPERATOR_KIND_UPSAMPLE_NEAREST_2D",
            "OPERATOR_KIND_PRELU",
        }
        for backend_id in ("cpu-js", "wasm", "webgpu", "native-cpu"):
            self.assertTrue(
                newly_qualified.issubset(by_id[backend_id].qualified_operators),
                backend_id,
            )
        unsupported_native = {
            "OPERATOR_KIND_CROSS_ATTENTION",
            "OPERATOR_KIND_CONV_1D",
            "OPERATOR_KIND_CONV_TRANSPOSE_2D",
            "OPERATOR_KIND_AVERAGE_POOL_2D",
            "OPERATOR_KIND_INTERPOLATE_1D",
            "OPERATOR_KIND_MASK",
            "OPERATOR_KIND_PAD",
            "OPERATOR_KIND_GATHER_ELEMENTS",
            "OPERATOR_KIND_BROADCAST",
            "OPERATOR_KIND_CONCAT2",
            "OPERATOR_KIND_NON_MAX_SUPPRESSION",
            "OPERATOR_KIND_SPATIAL_SOFTARGMAX_Y",
            "OPERATOR_KIND_MEAN_HEIGHT",
            "OPERATOR_KIND_PROFILE_X",
            "OPERATOR_KIND_PROFILE_Y",
        }
        self.assertEqual(
            set(by_id["cpu-js"].runtime_operators) - set(by_id["native-cpu"].runtime_operators),
            unsupported_native,
        )
        for backend_id in ("vulkan", "opengl", "metal"):
            self.assertEqual(
                set(by_id[backend_id].qualified_operators), cuda_qualified
            )
            self.assertEqual(by_id[backend_id].support_mode, "RUNTIME_SUPPORT_MODE_DYNAMIC")

    def test_wasm_has_an_explicit_route_for_every_runtime_operator(self):
        wasm = next(backend for backend in self.registry.backends if backend.runtime_id == "wasm")
        self.assertFalse(wasm.default_route)
        self.assertEqual(
            {route.operator for route in wasm.routes},
            set(wasm.runtime_operators),
        )
        routes = {
            self.registry.graph_names[route.operator]: route.route
            for route in wasm.routes
        }
        self.assertEqual(routes["QLinear"], "qlinear")
        self.assertEqual(routes["LogSoftmax"], "softmax")

    def test_cuda_qbatch_dp4a_variant_is_explicit(self):
        variant = next(
            item for item in self.registry.variants
            if item.id == "cuda.qbatch-matmul.dp4a"
        )
        self.assertEqual(variant.backend, "BACKEND_KIND_CUDA")
        self.assertEqual(
            variant.operators, ("OPERATOR_KIND_Q_BATCH_MATMUL",)
        )
        self.assertEqual(
            variant.entrypoint_id, "vx_cuda_qbatch_matmul_i8u8"
        )
        self.assertEqual(
            variant.required_features,
            ("cuda.device-residency", "cuda.sm61-dp4a"),
        )
        self.assertIn("arbitrary-k-tail", variant.predicate_id)

    def test_qbatch_packed_dot_variants_are_feature_gated(self):
        for backend, feature, entrypoint in (
            (
                "BACKEND_KIND_WEBGPU",
                "webgpu.packed-4x8-integer-dot-product",
                "qBatchMatMulDot.wgsl",
            ),
            (
                "BACKEND_KIND_VULKAN",
                "vulkan.packed-4x8-integer-dot-product",
                "qBatchMatMulDot",
            ),
        ):
            variant = next(
                item
                for item in self.registry.variants
                if item.backend == backend
                and item.id.endswith(".qbatch-matmul.dot")
            )
            self.assertEqual(
                variant.operators, ("OPERATOR_KIND_Q_BATCH_MATMUL",)
            )
            self.assertEqual(variant.entrypoint_id, entrypoint)
            self.assertEqual(variant.required_features, (feature,))
            self.assertIn("i32-safe", variant.predicate_id)

    def test_cuda_dense_and_conv_dp4a_tiers_match_physical_dispatch(self):
        by_id = {item.id: item for item in self.registry.variants}
        dense_operators = (
            "OPERATOR_KIND_Q_LINEAR",
            "OPERATOR_KIND_Q_MATMUL",
            "OPERATOR_KIND_Q_GEMM",
        )
        for variant_id, entrypoint, predicate_fragment, priority in (
            ("cuda.qlinear.warp-dp4a", "vx_cuda_qlinear_warp_dp4a_i8u8",
             "k-ge32-warp-launch-capacity", 300),
            ("cuda.qlinear.thread-dp4a", "vx_cuda_qlinear_i8u8",
             "small-k-or-warp-launch-capacity", 200),
        ):
            variant = by_id[variant_id]
            self.assertEqual(variant.operators, dense_operators)
            self.assertEqual(variant.entrypoint_id, entrypoint)
            self.assertEqual(
                variant.required_features,
                ("cuda.device-residency", "cuda.sm61-dp4a"),
            )
            self.assertIn(predicate_fragment, variant.predicate_id)
            self.assertIn("arbitrary-k-tail", variant.predicate_id)
            self.assertEqual(variant.priority, priority)
        for variant_id, entrypoint, predicate_fragment, priority in (
            ("cuda.qconv2d.warp-dp4a",
             "vx_cuda_qconv2d_warp_dp4a_i8u8",
             "input-per-group-ge4-terms-ge64-warp-launch-capacity", 300),
            ("cuda.qconv2d.thread-dp4a", "vx_cuda_qconv2d_i8u8",
             "small-reduction-or-warp-launch-capacity", 200),
        ):
            variant = by_id[variant_id]
            self.assertEqual(variant.operators, ("OPERATOR_KIND_Q_CONV_2D",))
            self.assertEqual(variant.entrypoint_id, entrypoint)
            self.assertEqual(
                variant.required_features,
                ("cuda.device-residency", "cuda.sm61-dp4a"),
            )
            self.assertIn(predicate_fragment, variant.predicate_id)
            self.assertIn("per-spatial-tail", variant.predicate_id)
            self.assertEqual(variant.priority, priority)

    def test_native_gpu_qconv_variants_match_physical_dispatches(self):
        by_id = {item.id: item for item in self.registry.variants}
        vulkan_conv = by_id["vulkan.conv2d.regular-out16"]
        self.assertEqual(
            vulkan_conv.operators, ("OPERATOR_KIND_CONV_2D",)
        )
        self.assertEqual(vulkan_conv.entrypoint_id, "conv2DRegularOut16")
        self.assertEqual(vulkan_conv.required_features, ())
        self.assertIn("groups1", vulkan_conv.predicate_id)
        self.assertIn("output-channels16", vulkan_conv.predicate_id)
        self.assertEqual(vulkan_conv.priority, 250)
        vulkan_conv_c3 = by_id["vulkan.conv2d.c3-out16"]
        self.assertEqual(
            vulkan_conv_c3.entrypoint_id, "conv2DRegularC3Out16"
        )
        self.assertIn("input-channels3", vulkan_conv_c3.predicate_id)
        self.assertEqual(vulkan_conv_c3.priority, 260)
        self.assertEqual(
            by_id["vulkan.conv2d.scalar"].entrypoint_id, "conv2D"
        )
        self.assertEqual(by_id["vulkan.conv2d.scalar"].priority, 100)
        webgpu_conv = by_id["webgpu.conv2d.regular-out16"]
        self.assertEqual(
            webgpu_conv.operators, ("OPERATOR_KIND_CONV_2D",)
        )
        self.assertEqual(webgpu_conv.entrypoint_id, "conv2DRegularOut16.wgsl")
        self.assertEqual(webgpu_conv.required_features, ())
        self.assertIn("output-channels16", webgpu_conv.predicate_id)
        self.assertEqual(webgpu_conv.priority, 250)
        self.assertEqual(
            by_id["webgpu.conv2d.scalar"].entrypoint_id, "conv2D.wgsl"
        )
        vulkan_dot = by_id["vulkan.qconv2d.dot-tiled"]
        self.assertEqual(
            vulkan_dot.operators, ("OPERATOR_KIND_Q_CONV_2D",)
        )
        self.assertEqual(vulkan_dot.entrypoint_id, "qConv2DInt8DotTiled")
        self.assertEqual(
            vulkan_dot.required_features,
            ("vulkan.packed-4x8-integer-dot-product",),
        )
        self.assertIn("groups1", vulkan_dot.predicate_id)
        self.assertIn("output-channels4", vulkan_dot.predicate_id)
        self.assertIn("i32-safe", vulkan_dot.predicate_id)

        for variant_id, entrypoint in (
            ("vulkan.qconv2d.tiled", "qConv2DInt8Tiled"),
            ("vulkan.qconv2d.scalar", "qConv2DInt8"),
            ("opengl.qconv2d.tiled", "qConv2DInt8Tiled"),
            ("opengl.qconv2d.scalar", "qConv2DInt8"),
        ):
            variant = by_id[variant_id]
            self.assertEqual(
                variant.operators, ("OPERATOR_KIND_Q_CONV_2D",)
            )
            self.assertEqual(variant.entrypoint_id, entrypoint)
            self.assertEqual(variant.required_features, ())

        self.assertIn(
            "workgroup-8x4",
            by_id["vulkan.qconv2d.tiled"].predicate_id,
        )
        self.assertIn(
            "workgroup-8x4",
            by_id["opengl.qconv2d.tiled"].predicate_id,
        )
        self.assertEqual(
            by_id["vulkan.qconv2d.dot-tiled"].priority,
            300,
        )
        self.assertEqual(by_id["vulkan.qconv2d.tiled"].priority, 200)
        self.assertEqual(by_id["opengl.qconv2d.tiled"].priority, 200)

        opengl_qbatch = by_id["opengl.qbatch-matmul.scalar"]
        self.assertEqual(
            opengl_qbatch.operators,
            ("OPERATOR_KIND_Q_BATCH_MATMUL",),
        )
        self.assertEqual(opengl_qbatch.entrypoint_id, "qBatchMatMul")
        self.assertEqual(opengl_qbatch.required_features, ())
        self.assertIn("i32-safe", opengl_qbatch.predicate_id)

    def test_shape_contracts_exactly_cover_runtime_vocabulary_and_canonical_ids(self):
        by_operator = {
            self.registry.graph_names[contract.operator]: contract
            for contract in self.registry.shape_contracts
        }
        self.assertEqual(len(by_operator), 92)
        self.assertEqual(set(by_operator), set(self.registry.graph_names.values()))
        self.assertEqual(
            {
                contract.classification
                for contract in self.registry.shape_contracts
            },
            {
                "SHAPE_CONTRACT_CLASSIFICATION_CANONICAL",
                "SHAPE_CONTRACT_CLASSIFICATION_BOUNDED_VALUE_DEPENDENT",
            },
        )
        expected_wave_a = {
            "Identity": "volvox.shape.identity.v1",
            "ReLU": "volvox.shape.activation-preserve.v1",
            "LeakyReLU": "volvox.shape.activation-preserve.v1",
            "PReLU": "volvox.shape.activation-preserve.v1",
            "GELU": "volvox.shape.activation-preserve.v1",
            "SiLU": "volvox.shape.activation-preserve.v1",
            "Sigmoid": "volvox.shape.activation-preserve.v1",
            "HardSwish": "volvox.shape.activation-preserve.v1",
            "HardSigmoid": "volvox.shape.activation-preserve.v1",
            "Tanh": "volvox.shape.activation-preserve.v1",
            "Sin": "volvox.shape.activation-preserve.v1",
            "Cos": "volvox.shape.activation-preserve.v1",
            "Clip": "volvox.shape.activation-preserve.v1",
            "Softmax": "volvox.shape.activation-preserve.v1",
            "LogSoftmax": "volvox.shape.activation-preserve.v1",
            "Cast": "volvox.shape.cast-preserve.v1",
            "QuantizeLinear": "volvox.shape.quantize-linear.v1",
            "DequantizeLinear": "volvox.shape.dequantize-linear.v1",
            "LayerNorm": "volvox.shape.feature-norm.v1",
            "RMSNorm": "volvox.shape.feature-norm.v1",
            "GroupNorm": "volvox.shape.group-norm-nhwc.v1",
            "Linear": "volvox.shape.dense-last-axis.v1",
            "Gemm": "volvox.shape.dense-last-axis.v1",
            "MatMul": "volvox.shape.dense-last-axis.v1",
            "Embedding": "volvox.shape.embedding-prefix.v1",
            "Add": "volvox.shape.exact-binary.v1",
            "Mul": "volvox.shape.exact-binary.v1",
        }
        expected_wave_b = {
            "Sub": "volvox.shape.broadcast-arithmetic.v1",
            "Div": "volvox.shape.broadcast-arithmetic.v1",
            "ReduceSum": "volvox.shape.reduction.v1",
            "ReduceMean": "volvox.shape.reduction.v1",
            "ArgMax": "volvox.shape.arg-max.v1",
            "Equal": "volvox.shape.broadcast-comparison.v1",
            "GreaterOrEqual": "volvox.shape.broadcast-comparison.v1",
            "Where": "volvox.shape.where-broadcast.v1",
            "Reshape": "volvox.shape.reshape-static-target.v1",
            "Flatten": "volvox.shape.flatten.v1",
            "Squeeze": "volvox.shape.squeeze.v1",
            "Unsqueeze": "volvox.shape.unsqueeze.v1",
            "Transpose": "volvox.shape.transpose.v1",
            "Concat": "volvox.shape.concat.v1",
            "Split": "volvox.shape.split.v1",
            "Slice": "volvox.shape.slice-static.v1",
            "Pad": "volvox.shape.pad-static.v1",
            "Expand": "volvox.shape.expand-static-target.v1",
            "Gather": "volvox.shape.gather.v1",
            "GatherElements": "volvox.shape.gather-elements.v1",
        }
        expected_spatial = {
            "BatchMatMul": "volvox.shape.batch-matmul.v1",
            "Conv1D": "volvox.shape.conv-1d.v1",
            "Conv2D": "volvox.shape.conv-2d.v1",
            "ConvTranspose2D": "volvox.shape.conv-transpose-2d.v1",
            "MaxPool2D": "volvox.shape.max-pool-2d.v1",
            "AveragePool2D": "volvox.shape.average-pool-2d.v1",
            "GlobalAveragePool": "volvox.shape.global-average-pool.v1",
            "Resize": "volvox.shape.resize.v1",
            "ResizeNearest2D": "volvox.shape.resize-nearest-2d.v1",
            "UpsampleNearest2D": "volvox.shape.upsample-nearest-2d.v1",
        }
        expected_attention = {
            "SDPA": "volvox.shape.sdpa.v1",
            "CrossSDPA": "volvox.shape.cross-sdpa.v1",
            "RoPE": "volvox.shape.rope-preserve.v1",
        }
        expected_quantized = {
            "QLinear": "volvox.shape.q-dense-last-axis.v1",
            "QMatMul": "volvox.shape.q-dense-last-axis.v1",
            "QGemm": "volvox.shape.q-dense-last-axis.v1",
            "QBatchMatMul": "volvox.shape.q-batch-matmul.v1",
            "QConv2D": "volvox.shape.q-conv-2d.v1",
            "QAdd": "volvox.shape.q-exact-binary.v1",
            "QEmbedding": "volvox.shape.q-embedding-prefix.v1",
            "QGELU": "volvox.shape.q-activation-preserve.v1",
            "QSiLU": "volvox.shape.q-activation-preserve.v1",
            "QLayerNorm": "volvox.shape.q-feature-norm.v1",
            "QGroupNorm": "volvox.shape.q-group-norm-nhwc.v1",
            "QMaskedMean": "volvox.shape.q-masked-mean.v1",
            "QSDPA": "volvox.shape.q-sdpa.v1",
            "QArgMax": "volvox.shape.q-arg-max.v1",
        }
        expected_final = {
            "MoERouter": "volvox.shape.moe-router.v1",
            "MoELinear": "volvox.shape.moe-linear.v1",
            "CrossAttention": "volvox.shape.cross-attention.v1",
            "BatchNorm2D": "volvox.shape.batch-norm-2d.v1",
            "Interpolate1D": "volvox.shape.interpolate-1d.v1",
            "Not": "volvox.shape.logical-not.v1",
            "Mask": "volvox.shape.mask.v1",
            "Broadcast": "volvox.shape.broadcast-static-target.v1",
            "Concat2": "volvox.shape.concat-2.v1",
            "RequantizeLinear": "volvox.shape.requantize-linear.v1",
            "SSMScan": "volvox.shape.ssm-scan.v1",
            "SelectiveScan": "volvox.shape.selective-scan.v1",
            "SpatialSoftargmaxY": "volvox.shape.spatial-softargmax-y.v1",
            "MeanHeight": "volvox.shape.mean-height.v1",
            "ProfileX": "volvox.shape.profile-x.v1",
            "ProfileY": "volvox.shape.profile-y.v1",
            "Dropout": "volvox.shape.dropout-preserve.v1",
        }
        self.assertEqual(
            {
                operator: contract.shape_function_id
                for operator, contract in by_operator.items()
                if contract.classification == "SHAPE_CONTRACT_CLASSIFICATION_CANONICAL"
            },
            expected_wave_a | expected_wave_b | expected_spatial |
            expected_attention | expected_quantized | expected_final,
        )
        self.assertEqual(
            by_operator["NonMaxSuppression"].classification,
            "SHAPE_CONTRACT_CLASSIFICATION_BOUNDED_VALUE_DEPENDENT",
        )

    def test_shape_contract_projections_include_immutable_lookups(self):
        python = render_python(self.registry)
        typescript = render_ts(self.registry)
        native = render_c(self.registry)
        self.assertIn(b"OPERATOR_SHAPE_CONTRACTS = MappingProxyType", python)
        self.assertIn(b"operatorShapeContracts = Object.freeze", typescript)
        self.assertIn(b"operatorShapeContract(operator: string)", typescript)
        self.assertIn(b"vx_kernel_shape_contract_find", native)

    def test_full_variants_do_not_leak_into_inference_outputs(self):
        full = Variant(
            "native-cpu.test.backward",
            "BACKEND_KIND_NATIVE_CPU",
            ("OPERATOR_KIND_LINEAR",),
            "KERNEL_PROFILE_FULL",
            "KERNEL_PHASE_BACKWARD",
            "vx_test_training_backward",
            "test.training-predicate",
            (),
            1,
        )
        registry = dataclasses.replace(
            self.registry, variants=self.registry.variants + (full,)
        )
        inference = render_python(registry) + render_ts(registry) + render_c(registry)
        full_outputs = (
            render_python_full(registry) + render_ts_full(registry) + render_c_full(registry)
        )
        self.assertNotIn(b"vx_test_training_backward", inference)
        self.assertIn(b"vx_test_training_backward", full_outputs)

    def test_rejects_duplicate_operator_inside_a_named_set(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        needle = (
            'name: "runtime-linear"\n'
            "    operator: OPERATOR_KIND_MATMUL\n"
        )
        mutated = source.replace(
            needle,
            needle + "    operator: OPERATOR_KIND_MATMUL\n",
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate operator"):
            self._load_mutated(mutated)

    def test_rejects_unknown_operator_runtime_registration(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated = source.replace(
            'name: "runtime-linear"\n    operator: OPERATOR_KIND_MATMUL',
            'name: "runtime-linear"\n    operator: OPERATOR_KIND_NOT_DECLARED',
            1,
        )
        with self.assertRaisesRegex(ValueError, "unknown operator"):
            self._load_mutated(mutated)

    def test_native_gpu_qualification_is_shared(self):
        by_target = {
            backend.exporter_target: backend
            for backend in self.registry.backends
            if backend.exporter_target
        }
        expected = by_target["backend:cuda"].qualified_operators
        self.assertTrue(expected)
        for target in ("backend:vulkan", "backend:opengl", "backend:metal"):
            self.assertEqual(by_target[target].qualified_operators, expected)

    def test_rejects_missing_shape_contract_route(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated, count = re.subn(
            r"^  shape_contract: \{ operator: OPERATOR_KIND_MATMUL .*\n",
            "",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        self.assertEqual(count, 1)
        with self.assertRaisesRegex(ValueError, "exactly cover OperatorKind"):
            self._load_mutated(mutated)

    def test_rejects_duplicate_shape_contract_route(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        match = re.search(
            r"^  shape_contract: \{ operator: OPERATOR_KIND_MATMUL .*\n",
            source,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(match)
        assert match is not None
        mutated = source.replace(match.group(0), match.group(0) * 2, 1)
        with self.assertRaisesRegex(ValueError, "duplicate shape-contract operator"):
            self._load_mutated(mutated)

    def test_rejects_invalid_shape_function_id(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated = source.replace(
            'shape_function_id: "volvox.shape.dense-last-axis.v1"',
            'shape_function_id: "Dynamic Shape Function"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "invalid stable ID"):
            self._load_mutated(mutated)

    def test_rejects_invalid_shape_contract_classification(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated = source.replace(
            "classification: SHAPE_CONTRACT_CLASSIFICATION_CANONICAL",
            "classification: SHAPE_CONTRACT_CLASSIFICATION_UNSPECIFIED",
            1,
        )
        with self.assertRaisesRegex(ValueError, "invalid shape-contract classification"):
            self._load_mutated(mutated)

    def test_rejects_inference_training_phase(self):
        source = REGISTRY_PROTO.read_text(encoding="utf-8")
        mutated = source.replace(
            "phase: KERNEL_PHASE_FORWARD",
            "phase: KERNEL_PHASE_BACKWARD",
            1,
        )
        with self.assertRaisesRegex(ValueError, "non-forward phase"):
            self._load_mutated(mutated)

    def test_generated_files_are_current_and_protoc_accepts_schema(self):
        command = [sys.executable, str(ROOT / "tools/generate_kernel_registry.py"), "--check"]
        if shutil.which("protoc"):
            command.append("--protoc-check")
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def _load_mutated(self, source: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel_registry.proto"
            path.write_text(source, encoding="utf-8")
            return load_registry(path, PUBLIC_PROTO)


if __name__ == "__main__":
    unittest.main()
