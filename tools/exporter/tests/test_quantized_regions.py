from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from tools.exporter.optimizer.analysis import AnalysisManager, GraphAnalysis
from tools.exporter.optimizer.candidate import RewritePolicy, RewriteSemantics
from tools.exporter.optimizer.quantized_regions import (
    QuantizedRegionAnalysis,
    QuantizedRegionCandidateAnalysis,
    RegionCandidatePlacement,
)
from tools.exporter.runtime_ir import import_runtime_package


def _node(identifier, op_type, inputs, output, shape, dtype, params=None):
    node = {
        "id": identifier,
        "opType": op_type,
        "inputs": dict(inputs),
        "outputs": {"out": {
            "tensor": output, "shape": list(shape), "dtype": dtype,
        }},
        "params": {},
    }
    if params is not None:
        node["params"] = dict(params)
    return node


def _affine(scale, zero):
    return {
        "scheme": "per_tensor",
        "scale_tensor": scale,
        "zero_point_tensor": zero,
    }


def _single_input_package(compute_nodes, *, shape=(2, 4), outputs=None, extra=None):
    tensors = {
        "in_scale": np.asarray([0.125], dtype=np.float32),
        "in_zero": np.asarray([-3], dtype=np.int8),
        "out_scale": np.asarray([0.0625], dtype=np.float32),
        "out_zero": np.asarray([7], dtype=np.uint8),
        **(extra or {}),
    }
    nodes = [
        _node(
            "dq", "DequantizeLinear",
            {"input": "input_byte", "scale": "in_scale", "zero_point": "in_zero"},
            "decoded", shape, "float32",
        ),
        *compute_nodes,
        _node(
            "q", "QuantizeLinear",
            {"input": compute_nodes[-1]["outputs"]["out"]["tensor"],
             "scale": "out_scale", "zero_point": "out_zero"},
            "output_byte", shape, "uint8",
        ),
    ]
    document = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"input_byte": {"shape": list(shape), "dtype": "int8"}},
        "outputs": list(outputs or ("output_byte",)),
        "nodes": nodes,
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "input_byte": _affine("in_scale", "in_zero"),
                "output_byte": _affine("out_scale", "out_zero"),
            },
        },
    }
    return document, tensors


def _closed_broadcast_graph():
    shape = (2, 4)
    document, tensors = _single_input_package(
        [
            _node(
                "bias", "Add", {"a": "decoded", "b": "bias_value"},
                "biased", shape, "float32",
            ),
            _node(
                "gelu", "GELU", {"input": "biased"},
                "activated", shape, "float32", {"approximate": "none"},
            ),
        ],
        extra={
            "bias_value": np.tile(
                np.linspace(-0.2, 0.2, 4, dtype=np.float32), (2, 1),
            ),
        },
    )
    return import_runtime_package(document, tensors)


class QuantizedRegionAnalysisTests(unittest.TestCase):
    def test_discovers_closed_multi_op_region(self):
        shape = (2, 4)
        compute = [
            _node(
                "bias", "Add", {"a": "decoded", "b": "bias_value"},
                "biased", shape, "float32",
            ),
            _node(
                "gelu", "GELU", {"input": "biased"},
                "activated", shape, "float32", {"approximate": "none"},
            ),
        ]
        document, tensors = _single_input_package(
            compute,
            extra={
                "bias_value": np.tile(
                    np.linspace(-0.2, 0.2, 4, dtype=np.float32), (2, 1),
                ),
            },
        )

        report = QuantizedRegionAnalysis().run(
            import_runtime_package(document, tensors),
        )

        self.assertEqual(report.analysis_id, "quantized-regions/v1")
        self.assertEqual(len(report.regions), 1)
        region = report.regions[0]
        self.assertTrue(region.closed)
        self.assertTrue(region.multi_op)
        self.assertEqual(region.operators, ("Add", "GELU"))
        self.assertEqual(region.static_parameters, ("bias_value",))
        self.assertEqual(region.open_float_inputs, ())
        self.assertEqual(region.escaping_tensors, ())
        self.assertEqual(region.unquantized_intermediates, ("biased",))
        self.assertEqual(region.broadcasts, ())
        self.assertEqual(region.input_boundaries[0].byte_tensor, "input_byte")
        self.assertEqual(region.output_boundaries[0].byte_tensor, "output_byte")
        self.assertEqual(report.by_node["bias"], region.id)
        self.assertEqual(report.by_node["gelu"], region.id)

    def test_shared_public_float_escape_keeps_region_open(self):
        shape = (2, 4)
        compute = [
            _node(
                "bias", "Add", {"a": "decoded", "b": "bias_value"},
                "biased", shape, "float32",
            ),
            _node(
                "gelu", "GELU", {"input": "biased"},
                "activated", shape, "float32", {"approximate": "none"},
            ),
        ]
        document, tensors = _single_input_package(
            compute,
            outputs=("output_byte", "biased"),
            extra={"bias_value": np.zeros(shape, dtype=np.float32)},
        )

        region = QuantizedRegionAnalysis().run(
            import_runtime_package(document, tensors),
        ).regions[0]

        self.assertFalse(region.closed)
        self.assertEqual(region.escaping_tensors, ("biased",))
        self.assertEqual(len(region.output_boundaries), 1)

    def test_two_input_residual_norm_is_one_closed_region(self):
        shape = (2, 4)
        tensors = {
            "a_scale": np.asarray([0.125], dtype=np.float32),
            "a_zero": np.asarray([-3], dtype=np.int8),
            "b_scale": np.asarray([0.25], dtype=np.float32),
            "b_zero": np.asarray([129], dtype=np.uint8),
            "out_scale": np.asarray([0.0625], dtype=np.float32),
            "out_zero": np.asarray([7], dtype=np.uint8),
            "norm_weight": np.ones(4, dtype=np.float32),
            "norm_bias": np.zeros(4, dtype=np.float32),
        }
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {
                "a_byte": {"shape": list(shape), "dtype": "int8"},
                "b_byte": {"shape": list(shape), "dtype": "uint8"},
            },
            "outputs": ["out_byte"],
            "nodes": [
                _node(
                    "a-dq", "DequantizeLinear",
                    {"input": "a_byte", "scale": "a_scale", "zero_point": "a_zero"},
                    "a", shape, "float32",
                ),
                _node(
                    "b-dq", "DequantizeLinear",
                    {"input": "b_byte", "scale": "b_scale", "zero_point": "b_zero"},
                    "b", shape, "float32",
                ),
                _node("residual", "Add", {"a": "a", "b": "b"},
                      "sum", shape, "float32"),
                _node(
                    "norm", "LayerNorm",
                    {"input": "sum", "weight": "norm_weight", "bias": "norm_bias"},
                    "normalized", shape, "float32", {"eps": 1e-5, "d_model": 4},
                ),
                _node(
                    "q", "QuantizeLinear",
                    {"input": "normalized", "scale": "out_scale",
                     "zero_point": "out_zero"},
                    "out_byte", shape, "uint8",
                ),
            ],
            "quantization": {
                "format": "volvox-affine-safetensors/v1",
                "tensors": {
                    "a_byte": _affine("a_scale", "a_zero"),
                    "b_byte": _affine("b_scale", "b_zero"),
                    "out_byte": _affine("out_scale", "out_zero"),
                },
            },
        }

        region = QuantizedRegionAnalysis().run(
            import_runtime_package(document, tensors),
        ).regions[0]

        self.assertTrue(region.closed)
        self.assertEqual(region.operators, ("Add", "LayerNorm"))
        self.assertEqual(len(region.input_boundaries), 2)
        self.assertEqual(region.unquantized_intermediates, ("sum",))
        self.assertEqual(
            region.static_parameters, ("norm_bias", "norm_weight"),
        )

    def test_decomposed_silu_branch_is_one_region(self):
        shape = (1, 2, 3, 4)
        compute = [
            _node(
                "norm", "GroupNorm",
                {"input": "decoded", "weight": "norm_weight", "bias": "norm_bias"},
                "normalized", shape, "float32",
                {"num_groups": 2, "eps": 1e-5},
            ),
            _node(
                "sigmoid", "Sigmoid", {"input": "normalized"},
                "gate", shape, "float32",
            ),
            _node(
                "multiply", "Mul", {"a": "normalized", "b": "gate"},
                "activated", shape, "float32",
            ),
        ]
        document, tensors = _single_input_package(
            compute,
            shape=shape,
            extra={
                "norm_weight": np.ones(4, dtype=np.float32),
                "norm_bias": np.zeros(4, dtype=np.float32),
            },
        )

        region = QuantizedRegionAnalysis().run(
            import_runtime_package(document, tensors),
        ).regions[0]

        self.assertTrue(region.closed)
        self.assertEqual(region.operators, ("GroupNorm", "Sigmoid", "Mul"))
        self.assertEqual(
            region.unquantized_intermediates, ("gate", "normalized"),
        )

    def test_analysis_manager_registration_preserves_direct_api(self):
        graph = _closed_broadcast_graph()
        analysis = QuantizedRegionAnalysis()
        self.assertIsInstance(analysis, GraphAnalysis)
        manager = AnalysisManager(graph, (analysis,))

        managed = manager.get(QuantizedRegionAnalysis.name)
        self.assertIs(managed, manager.get(QuantizedRegionAnalysis.name))
        direct = QuantizedRegionAnalysis().run(graph)

        self.assertEqual(
            tuple(region.id for region in managed.regions),
            tuple(region.id for region in direct.regions),
        )
        self.assertEqual(managed.graph_fingerprint, graph.fingerprint())
        self.assertEqual(
            manager.cached_analyses,
            ("quantized-regions", "use-def"),
        )

    def test_closed_region_plans_portable_and_compiled_model_candidates(self):
        graph = _closed_broadcast_graph()
        regions = QuantizedRegionAnalysis()
        candidates = QuantizedRegionCandidateAnalysis()
        manager = AnalysisManager(graph, (regions, candidates))

        plan = manager.get(QuantizedRegionCandidateAnalysis.name)
        self.assertEqual(plan.analysis_id, "quantized-region-candidates/v1")
        self.assertEqual(len(plan.candidates), 2)
        portable = plan.portable_rewrites[0]
        compiled = plan.compiled_model_fusions[0]

        self.assertEqual(
            portable.placement,
            RegionCandidatePlacement.PORTABLE_LOGICAL_REWRITE,
        )
        self.assertTrue(portable.changes_portable_graph)
        self.assertFalse(portable.compiled_model_only)
        self.assertEqual(
            portable.semantics,
            frozenset({RewriteSemantics.NUMERICAL_MIGRATION}),
        )
        self.assertIn(
            "bind-registered-portable-logical-rewrite",
            portable.requirements,
        )
        self.assertEqual(portable.broadcast_edges, ())
        self.assertEqual(portable.unquantized_intermediates, ("biased",))

        self.assertEqual(
            compiled.placement,
            RegionCandidatePlacement.COMPILED_MODEL_FUSION,
        )
        self.assertTrue(compiled.compiled_model_only)
        self.assertFalse(compiled.changes_portable_graph)
        self.assertEqual(
            compiled.semantics,
            frozenset({RewriteSemantics.EXACT}),
        )
        self.assertIn("preserve-qdq-rounding-order", compiled.requirements)
        self.assertNotEqual(portable.candidate_id, compiled.candidate_id)
        self.assertEqual(
            plan.by_region[portable.region_id],
            (portable.candidate_id, compiled.candidate_id),
        )
        self.assertEqual(plan.permitted_by(RewritePolicy.exact()), (compiled,))
        self.assertEqual(
            plan.permitted_by(RewritePolicy.qualified()),
            (portable, compiled),
        )
        self.assertEqual(
            portable.input_domains[0].to_dict(),
            {
                "dtype": "int8",
                "scheme": "per_tensor",
                "scale": "in_scale",
                "zero_point": "in_zero",
                "axis": None,
            },
        )
        payload = portable.to_dict()
        self.assertEqual(payload["candidate_id"], portable.candidate_id)
        self.assertEqual(payload["placement"], "portable-logical-rewrite")
        with self.assertRaises(TypeError):
            plan.by_region["new"] = ()
        with self.assertRaises(FrozenInstanceError):
            portable.region_id = "changed"

        direct = QuantizedRegionCandidateAnalysis().run(graph)
        self.assertEqual(
            tuple(item.candidate_id for item in plan.candidates),
            tuple(item.candidate_id for item in direct.candidates),
        )

    def test_open_regions_do_not_produce_rewrite_or_backend_candidates(self):
        shape = (2, 4)
        compute = [
            _node(
                "bias", "Add", {"a": "decoded", "b": "bias_value"},
                "biased", shape, "float32",
            ),
            _node(
                "gelu", "GELU", {"input": "biased"},
                "activated", shape, "float32", {"approximate": "none"},
            ),
        ]
        document, tensors = _single_input_package(
            compute,
            outputs=("output_byte", "biased"),
            extra={"bias_value": np.zeros(shape, dtype=np.float32)},
        )
        graph = import_runtime_package(document, tensors)

        plan = QuantizedRegionCandidateAnalysis().run(graph)

        self.assertEqual(plan.candidates, ())
        self.assertEqual(dict(plan.by_region), {})

    def test_candidate_analysis_depends_on_region_analysis(self):
        graph = _closed_broadcast_graph()
        manager = AnalysisManager(graph, (
            QuantizedRegionAnalysis(),
            QuantizedRegionCandidateAnalysis(),
        ))
        manager.get(QuantizedRegionCandidateAnalysis.name)
        self.assertEqual(
            manager.cached_analyses,
            ("quantized-region-candidates", "quantized-regions", "use-def"),
        )

        manager.invalidate(QuantizedRegionAnalysis.name)

        self.assertEqual(manager.cached_analyses, ("use-def",))


if __name__ == "__main__":
    unittest.main()
