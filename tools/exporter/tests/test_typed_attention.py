from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.exporter.capabilities import validate_graph
from tools.exporter.ir import (
    AffineQuantization,
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    Provenance,
    TensorDataRef,
    TensorValue,
)
from tools.exporter.optimizer.typed_attention import (
    RuntimeAttentionLayoutPass,
    RuntimeFloatAttentionFusionPass,
    RuntimeKeepMaskPass,
)
from tools.exporter.optimizer.typed_attention_common import (
    ATTENTION_KEEP_MASK_SEMANTIC_ID,
)
from tools.exporter.optimizer.typed_pipeline import (
    default_runtime_pipeline,
    serialize_pipeline_report,
)
from tools.exporter.optimizer.typed_quantized_attention import (
    RuntimeQuantizedAttentionFusionPass,
    RuntimeQuantizedAttentionLayoutPass,
    quantized_attention_candidate_count,
)
from tools.exporter.optimizer.typed_specialization import (
    DerivedValueSelector,
    InputHoistingSpec,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.runtime_ir import export_runtime_package


def _provenance(name: str, op: str) -> tuple[Provenance, ...]:
    return (Provenance(
        source_format="onnx",
        source_name=name,
        source_op=op,
        location=f"attention.onnx:{name}",
    ),)


def _tensor(
    graph: GraphIR,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: str = "float32",
    quantization: AffineQuantization | None = None,
    initializer: bool = False,
    public_input: bool = False,
    public_output: bool = False,
) -> None:
    graph.add_tensor(TensorValue(
        name=name,
        shape=shape,
        dtype=dtype,
        source_dtype=dtype,
        quantization=quantization,
        initializer=initializer,
        public_input=public_input,
        public_output=public_output,
        data=TensorDataRef(name) if initializer else None,
    ))
    if public_input:
        graph.inputs.append(name)


def _node(
    graph: GraphIR,
    name: str,
    op: str,
    inputs: dict[str, str],
    output: str,
    shape: tuple[int, ...],
    *,
    dtype: str = "float32",
    quantization: AffineQuantization | None = None,
    params: dict | None = None,
) -> None:
    if params is None and op in {"Reshape", "Expand"}:
        params = {"shape": list(shape)}
    _tensor(graph, output, shape, dtype=dtype, quantization=quantization)
    graph.add_node(OpNode.from_maps(
        name=name,
        op_type=op,
        inputs=inputs,
        outputs={"out": output},
        attributes=(() if params is None else (
            OpAttribute("params", "volvox.params", params),
        )),
        provenance=_provenance(name, op),
    ))


def _output(graph: GraphIR, name: str) -> None:
    graph.tensors[name].public_output = True
    graph.outputs.append(name)


def _split(
    graph: GraphIR,
    prefix: str,
    root: str,
    *,
    sequence: int,
    heads: int = 2,
    head_dim: int = 4,
    key: bool = False,
    bad_permutation: bool = False,
) -> str:
    width = heads * head_dim
    _node(
        graph,
        f"{prefix}.reshape",
        "Reshape",
        {"input": root},
        f"{prefix}.split",
        (sequence, heads, head_dim),
    )
    if key:
        permutation = [1, 2, 0]
        transposed = (heads, head_dim, sequence)
    else:
        permutation = [1, 0, 2]
        transposed = (heads, sequence, head_dim)
    if bad_permutation:
        permutation = [0, 1, 2]
        transposed = (sequence, heads, head_dim)
    _node(
        graph,
        f"{prefix}.transpose",
        "Transpose",
        {"input": f"{prefix}.split"},
        f"{prefix}.transposed",
        transposed,
        params={"perm": permutation},
    )
    target = (
        (1, heads, head_dim, sequence)
        if key and not bad_permutation
        else (1, *transposed)
    )
    _node(
        graph,
        f"{prefix}.heads",
        "Reshape",
        {"input": f"{prefix}.transposed"},
        f"{prefix}.heads.out",
        target,
    )
    return f"{prefix}.heads.out"


def _merge(
    graph: GraphIR,
    source: str,
    *,
    queries: int,
    width: int = 8,
) -> str:
    _node(
        graph,
        "context.transpose",
        "Transpose",
        {"input": source},
        "context.transposed",
        (1, queries, 2, 4),
        params={"perm": [0, 2, 1, 3]},
    )
    _node(
        graph,
        "context.merge",
        "Reshape",
        {"input": "context.transposed"},
        "context",
        (queries, width),
    )
    return "context"


def _float_attention_graph(
    *,
    masked: bool = False,
    bad_query_permutation: bool = False,
) -> tuple[GraphIR, dict[str, np.ndarray]]:
    queries, keys, width = 3, 5, 8
    graph = GraphIR("volvoxai", "float-attention.json", dialect=IRDialect.RUNTIME)
    for name, shape in (
        ("q", (queries, width)),
        ("k", (keys, width)),
        ("v", (keys, width)),
    ):
        _tensor(graph, name, shape, public_input=True)
    q = _split(
        graph,
        "q",
        "q",
        sequence=queries,
        bad_permutation=bad_query_permutation,
    )
    k = _split(graph, "k", "k", sequence=keys, key=True)
    v = _split(graph, "v", "v", sequence=keys)
    _node(
        graph,
        "score.matmul",
        "BatchMatMul",
        {"a": q, "b": k},
        "score",
        (1, 2, queries, keys),
    )
    tensors: dict[str, np.ndarray] = {}
    score_input = "score"
    if masked:
        _tensor(graph, "padding", (1, keys), dtype="int32", public_input=True)
        for name, value in (
            ("mask.neg_inf", np.full((1, keys), -np.inf, dtype=np.float32)),
            ("mask.zero", np.zeros((1, keys), dtype=np.float32)),
        ):
            _tensor(graph, name, value.shape, initializer=True)
            tensors[name] = value
        _node(
            graph,
            "mask.where",
            "Where",
            {"condition": "padding", "a": "mask.neg_inf", "b": "mask.zero"},
            "mask.additive",
            (1, keys),
        )
        _node(
            graph,
            "mask.reshape",
            "Reshape",
            {"input": "mask.additive"},
            "mask.broadcastable",
            (1, 1, 1, keys),
        )
        _node(
            graph,
            "mask.expand",
            "Expand",
            {"input": "mask.broadcastable"},
            "mask.expanded",
            (1, 2, queries, keys),
        )
        _node(
            graph,
            "score.mask",
            "Add",
            {"a": "score", "b": "mask.expanded"},
            "score.masked",
            (1, 2, queries, keys),
        )
        score_input = "score.masked"
    _node(
        graph,
        "score.softmax",
        "Softmax",
        {"input": score_input},
        "probability",
        (1, 2, queries, keys),
        params={"axis": -1},
    )
    _node(
        graph,
        "context.matmul",
        "BatchMatMul",
        {"a": "probability", "b": v},
        "context.heads",
        (1, 2, queries, 4),
    )
    final = _merge(graph, "context.heads", queries=queries)
    _output(graph, final)
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def _shared_mask_attention_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    """Two independently fusible attention regions sharing one additive mask."""

    queries, keys, width = 3, 5, 8
    graph = GraphIR("volvoxai", "shared-mask.json", dialect=IRDialect.RUNTIME)
    tensors: dict[str, np.ndarray] = {}
    for block in ("first", "second"):
        for role, shape in (
            ("q", (queries, width)),
            ("k", (keys, width)),
            ("v", (keys, width)),
        ):
            _tensor(graph, f"{block}.{role}", shape, public_input=True)
    _tensor(graph, "padding", (1, keys), dtype="int32", public_input=True)
    for name, value in (
        ("mask.neg_inf", np.full((1, keys), -np.inf, dtype=np.float32)),
        ("mask.zero", np.zeros((1, keys), dtype=np.float32)),
    ):
        _tensor(graph, name, value.shape, initializer=True)
        tensors[name] = value
    _node(
        graph,
        "mask.where",
        "Where",
        {"condition": "padding", "a": "mask.neg_inf", "b": "mask.zero"},
        "mask.additive",
        (1, keys),
    )
    _node(
        graph, "mask.reshape", "Reshape", {"input": "mask.additive"},
        "mask.broadcastable", (1, 1, 1, keys),
    )
    _node(
        graph, "mask.expand", "Expand", {"input": "mask.broadcastable"},
        "mask.expanded", (1, 2, queries, keys),
    )
    for block in ("first", "second"):
        q = _split(graph, f"{block}.q", f"{block}.q", sequence=queries)
        k = _split(
            graph, f"{block}.k", f"{block}.k", sequence=keys, key=True,
        )
        v = _split(graph, f"{block}.v", f"{block}.v", sequence=keys)
        _node(
            graph, f"{block}.score.matmul", "BatchMatMul", {"a": q, "b": k},
            f"{block}.score", (1, 2, queries, keys),
        )
        _node(
            graph, f"{block}.score.mask", "Add",
            {"a": f"{block}.score", "b": "mask.expanded"},
            f"{block}.score.masked", (1, 2, queries, keys),
        )
        _node(
            graph, f"{block}.score.softmax", "Softmax",
            {"input": f"{block}.score.masked"}, f"{block}.probability",
            (1, 2, queries, keys), params={"axis": -1},
        )
        _node(
            graph, f"{block}.context.matmul", "BatchMatMul",
            {"a": f"{block}.probability", "b": v},
            f"{block}.context.heads", (1, 2, queries, 4),
        )
        _node(
            graph, f"{block}.context.transpose", "Transpose",
            {"input": f"{block}.context.heads"},
            f"{block}.context.transposed", (1, queries, 2, 4),
            params={"perm": [0, 2, 1, 3]},
        )
        _node(
            graph, f"{block}.context.merge", "Reshape",
            {"input": f"{block}.context.transposed"},
            f"{block}.context", (queries, width),
        )
        _output(graph, f"{block}.context")
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def _affine(
    graph: GraphIR,
    tensors: dict[str, np.ndarray],
    stem: str,
    *,
    dtype: str = "int8",
    scale: float = 0.125,
    zero: int = 0,
) -> AffineQuantization:
    scale_name = f"{stem}.scale"
    zero_name = f"{stem}.zero"
    _tensor(graph, scale_name, (1,), initializer=True)
    _tensor(graph, zero_name, (1,), dtype=dtype, initializer=True)
    tensors[scale_name] = np.asarray([scale], dtype=np.float32)
    tensors[zero_name] = np.asarray([zero], dtype=np.dtype(dtype))
    return AffineQuantization(
        "per_tensor", scale=scale_name, zero_point=zero_name,
    )


def _quantize_node(
    graph: GraphIR,
    name: str,
    source: str,
    output: str,
    shape: tuple[int, ...],
    affine: AffineQuantization,
) -> None:
    _node(
        graph,
        name,
        "QuantizeLinear",
        {
            "input": source,
            "scale": affine.scale,
            "zero_point": affine.zero_point,
        },
        output,
        shape,
        dtype=graph.tensors[affine.zero_point].dtype,
        quantization=affine,
    )


def _dequantize_node(
    graph: GraphIR,
    name: str,
    source: str,
    output: str,
    shape: tuple[int, ...],
) -> None:
    affine = graph.tensors[source].quantization
    assert affine is not None
    _node(
        graph,
        name,
        "DequantizeLinear",
        {
            "input": source,
            "scale": affine.scale,
            "zero_point": affine.zero_point,
        },
        output,
        shape,
    )


def _quantized_fusion_graph(
    *,
    bad_merge: bool = False,
) -> tuple[GraphIR, dict[str, np.ndarray]]:
    queries, keys, width = 3, 5, 8
    graph = GraphIR("volvoxai", "qdq-attention.json", dialect=IRDialect.RUNTIME)
    tensors: dict[str, np.ndarray] = {}
    for name, shape in (
        ("q", (queries, width)),
        ("k", (keys, width)),
        ("v", (keys, width)),
    ):
        _tensor(graph, name, shape, public_input=True)
    q_heads = _split(graph, "q", "q", sequence=queries)
    k_heads = _split(graph, "k", "k", sequence=keys, key=True)
    v_heads = _split(graph, "v", "v", sequence=keys)
    q_affine = _affine(graph, tensors, "q.domain", scale=0.125, zero=-3)
    k_affine = _affine(graph, tensors, "k.domain", dtype="uint8", scale=0.25, zero=127)
    v_affine = _affine(graph, tensors, "v.domain", scale=0.2, zero=2)
    score_affine = _affine(graph, tensors, "score.domain", scale=0.1, zero=0)
    probability_affine = _affine(
        graph, tensors, "probability.domain", dtype="uint8", scale=1 / 255, zero=0,
    )
    output_affine = _affine(graph, tensors, "output.domain", scale=0.15, zero=-1)
    _quantize_node(graph, "q.quantize", q_heads, "q.byte", (1, 2, queries, 4), q_affine)
    _quantize_node(graph, "k.quantize", k_heads, "k.byte", (1, 2, 4, keys), k_affine)
    _quantize_node(graph, "v.quantize", v_heads, "v.byte", (1, 2, keys, 4), v_affine)
    _node(
        graph,
        "score.qmatmul",
        "QBatchMatMul",
        {"a": "q.byte", "b": "k.byte"},
        "score.byte",
        (1, 2, queries, keys),
        dtype="int8",
        quantization=score_affine,
    )
    _dequantize_node(
        graph, "score.dequantize", "score.byte", "score.float", (1, 2, queries, keys),
    )
    _node(
        graph,
        "score.softmax",
        "Softmax",
        {"input": "score.float"},
        "probability.float",
        (1, 2, queries, keys),
        params={"axis": -1},
    )
    _quantize_node(
        graph,
        "probability.quantize",
        "probability.float",
        "probability.byte",
        (1, 2, queries, keys),
        probability_affine,
    )
    _node(
        graph,
        "context.qmatmul",
        "QBatchMatMul",
        {"a": "probability.byte", "b": "v.byte"},
        "context.byte",
        (1, 2, queries, 4),
        dtype="int8",
        quantization=output_affine,
    )
    _dequantize_node(
        graph,
        "context.dequantize",
        "context.byte",
        "context.heads",
        (1, 2, queries, 4),
    )
    _node(
        graph,
        "context.transpose",
        "Transpose",
        {"input": "context.heads"},
        "context.transposed",
        ((1, 2, queries, 4) if bad_merge else (1, queries, 2, 4)),
        params={"perm": ([0, 1, 2, 3] if bad_merge else [0, 2, 1, 3])},
    )
    _node(
        graph,
        "context.merge",
        "Reshape",
        {"input": "context.transposed"},
        "context",
        (queries, width),
    )
    _output(graph, "context")
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def _quantized_layout_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", "qsdpa-layout.json", dialect=IRDialect.RUNTIME)
    tensors: dict[str, np.ndarray] = {}
    for name, shape in (("q", (3, 8)), ("k", (5, 8)), ("v", (5, 8))):
        _tensor(graph, name, shape, public_input=True)
        _node(
            graph,
            f"{name}.batch",
            "Reshape",
            {"input": name},
            f"{name}.batch.out",
            (1, *shape),
        )
    affines = {
        name: _affine(graph, tensors, f"{name}.domain", scale=0.1 + index * 0.05)
        for index, name in enumerate(("q", "k", "v", "out"))
    }
    for name in ("q", "k", "v"):
        _quantize_node(
            graph,
            f"{name}.quantize",
            f"{name}.batch.out",
            f"{name}.byte",
            graph.tensors[f"{name}.batch.out"].shape,
            affines[name],
        )
    _node(
        graph,
        "attention",
        "QSDPA",
        {"q": "q.byte", "k": "k.byte", "v": "v.byte"},
        "out.byte",
        (1, 3, 8),
        dtype="int8",
        quantization=affines["out"],
        params={"heads": 2, "causal": False, "scale": 1.0},
    )
    _dequantize_node(
        graph, "out.dequantize", "out.byte", "out.batch", (1, 3, 8),
    )
    _node(
        graph,
        "out.unbatch",
        "Reshape",
        {"input": "out.batch"},
        "out",
        (3, 8),
    )
    _output(graph, "out")
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def _assert_portable(
    case: unittest.TestCase,
    graph: GraphIR,
    tensors: dict[str, np.ndarray],
) -> None:
    document, packaged = export_runtime_package(graph, tensors)
    result = validate_graph(document, ["portable"], weights=packaged)
    case.assertTrue(result.supported, result.diagnostics)


class RuntimeFloatAttentionFusionTests(unittest.TestCase):
    def test_fuses_complete_layout_atomically_and_matches_independent_numpy(self):
        graph, tensors = _float_attention_graph()
        q = np.arange(24, dtype=np.float32).reshape(3, 8) / np.float32(17.0)
        k = np.arange(40, dtype=np.float32).reshape(5, 8) / np.float32(23.0)
        v = np.flip(k, axis=0).copy()
        qh = q.reshape(3, 2, 4).transpose(1, 0, 2)[None]
        kh = k.reshape(5, 2, 4).transpose(1, 2, 0)[None]
        vh = v.reshape(5, 2, 4).transpose(1, 0, 2)[None]
        scores = np.matmul(qh, kh, dtype=np.float32)
        scores *= np.float32(0.5)
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores, dtype=np.float32)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True, dtype=np.float32)
        decomposed = np.matmul(probabilities, vh, dtype=np.float32)
        decomposed = decomposed.transpose(0, 2, 1, 3).reshape(3, 8)

        report = VerifiedPipeline((RuntimeFloatAttentionFusionPass(
            tensors, allow_numerical_migration=True,
        ),), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual([node.op_type for node in graph.nodes], ["CrossSDPA"])
        attention = graph.nodes[0]
        self.assertEqual(attention.input_map(), {"q": "q", "k": "k", "v": "v"})
        self.assertEqual(attention.output_map(), {"out": "context"})
        self.assertEqual(attention.attributes[0].value, {
            "heads": 2, "causal": False, "scale": 0.5,
        })
        # Independent fused spelling of the same mathematical contract.
        fused = np.matmul(
            _softmax(np.matmul(qh, kh, dtype=np.float32) * np.float32(0.5)),
            vh,
            dtype=np.float32,
        ).transpose(0, 2, 1, 3).reshape(3, 8)
        np.testing.assert_allclose(fused, decomposed, rtol=2e-6, atol=1e-6)
        _assert_portable(self, graph, tensors)

    def test_additive_padding_mask_becomes_i32_keep_mask_without_touching_sources(self):
        graph, tensors = _float_attention_graph(masked=True)
        original = {name: value.copy() for name, value in tensors.items()}

        VerifiedPipeline((RuntimeFloatAttentionFusionPass(
            tensors, allow_numerical_migration=True,
        ),), shape_profile={}).run(graph)

        attention = next(node for node in graph.nodes if node.op_type == "CrossSDPA")
        keep_name = attention.input_map()["mask"]
        keep = next(node for node in graph.nodes if node.output_map().get("out") == keep_name)
        self.assertEqual(keep.op_type, "Where")
        self.assertEqual(graph.tensors[keep_name].dtype, "int32")
        self.assertEqual(graph.tensors[keep_name].shape, (1, 5))
        self.assertFalse(attention.attributes[0].value["causal"])
        for name, value in original.items():
            np.testing.assert_array_equal(tensors[name], value)
        # The runtime `Where` needs exact shapes rather than broadcasting, and a
        # bounded-dynamic keep mask has no concrete extent to allocate, so each
        # branch is a scalar initializer widened by one `Expand`.
        zero = self._constant_branch(graph, tensors, keep.input_map()["a"])
        one = self._constant_branch(graph, tensors, keep.input_map()["b"])
        self.assertEqual(zero.dtype, np.int32)
        self.assertTrue(np.all(zero == 0))
        self.assertTrue(np.all(one == 1))
        _assert_portable(self, graph, tensors)

    def _constant_branch(self, graph, tensors, name):
        """Resolve a Where branch to its constant fill through one Expand."""

        if name in tensors:
            return tensors[name]
        producer = next(
            node for node in graph.nodes if node.output_map().get("out") == name
        )
        self.assertEqual(producer.op_type, "Expand")
        return tensors[producer.input_map()["input"]]

    def test_identical_derived_keep_mask_is_materialized_once_and_reused(self):
        graph, tensors = _shared_mask_attention_graph()

        report = VerifiedPipeline((RuntimeFloatAttentionFusionPass(
            tensors, allow_numerical_migration=True,
        ),), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 2)
        attentions = [node for node in graph.nodes if node.op_type == "CrossSDPA"]
        self.assertEqual(len(attentions), 2)
        masks = {node.input_map()["mask"] for node in attentions}
        self.assertEqual(len(masks), 1)
        keep_name = next(iter(masks))
        self.assertEqual(
            graph.tensors[keep_name].metadata["optimizer_derived_value"],
            {
                "semantic_id": ATTENTION_KEEP_MASK_SEMANTIC_ID,
                "source_value": "mask.additive",
                "condition": "padding",
                "suppress_when_true": True,
                "shape": [1, 5],
            },
        )
        keep_producers = [
            node for node in graph.nodes if node.output_map().get("out") == keep_name
        ]
        self.assertEqual(len(keep_producers), 1)
        _assert_portable(self, graph, tensors)

    def test_registry_orders_derived_value_hoisting_after_attention_rewrite(self):
        graph, tensors = _float_attention_graph(masked=True)
        specification = InputHoistingSpec(
            public_name="padding_keep",
            dtype="int32",
            derived_value=DerivedValueSelector(
                ATTENTION_KEEP_MASK_SEMANTIC_ID,
                "mask.additive",
            ),
        )
        pipeline = default_runtime_pipeline(
            tensor_data=tensors,
            input_hoistings=(specification,),
            allow_float_attention_numerical_migration=True,
            enable_fp32_pre_ptq_optimization=True,
            shape_profile={},
        )

        report = pipeline.run(graph)

        selected = report.metadata.pass_ids
        self.assertLess(
            selected.index("runtime-float-attention-fusion"),
            selected.index("runtime-input-hoisting"),
        )
        attention = next(node for node in graph.nodes if node.op_type == "CrossSDPA")
        keep_name = attention.input_map()["mask"]
        self.assertIn(keep_name, graph.inputs)
        self.assertTrue(graph.tensors[keep_name].public_input)
        self.assertEqual(
            graph.tensors[keep_name].metadata["runtime_input_fields"]["source_name"],
            "padding_keep",
        )
        self.assertFalse(any(
            node.output_map().get("out") == keep_name for node in graph.nodes
        ))
        document, packaged = export_runtime_package(graph, tensors)
        self.assertEqual(
            document["inputs"][keep_name],
            {"shape": [1, 5], "dtype": "int32"},
        )
        _assert_portable(self, graph, packaged)

    def test_refuses_ambiguous_head_permutation_without_partial_mutation(self):
        graph, tensors = _float_attention_graph(bad_query_permutation=True)
        before = graph.fingerprint()
        data_before = dict(tensors)
        pass_ = RuntimeFloatAttentionFusionPass(
            tensors, allow_numerical_migration=True,
        )

        report = VerifiedPipeline((pass_,), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), before)
        self.assertEqual(tensors, data_before)
        self.assertGreaterEqual(pass_.refused, 1)

    def test_requires_explicit_numerical_migration_policy(self):
        with self.assertRaisesRegex(ValueError, "numerical-migration"):
            RuntimeFloatAttentionFusionPass({}, allow_numerical_migration=False)


class RuntimeAttentionExactPassTests(unittest.TestCase):
    def test_layout_removes_only_proved_singleton_batch_wrappers(self):
        graph = GraphIR("volvoxai", "cross-layout.json", dialect=IRDialect.RUNTIME)
        for name, shape in (("q", (3, 8)), ("k", (5, 8)), ("v", (5, 8))):
            _tensor(graph, name, shape, public_input=True)
            _node(graph, f"{name}.batch", "Reshape", {"input": name},
                  f"{name}.batched", (1, *shape))
        _node(
            graph,
            "attention",
            "CrossSDPA",
            {"q": "q.batched", "k": "k.batched", "v": "v.batched"},
            "out.batched",
            (1, 3, 8),
            params={"heads": 2, "causal": False, "scale": 0.5},
        )
        _node(graph, "out.unbatch", "Reshape", {"input": "out.batched"},
              "out", (3, 8))
        _output(graph, "out")
        graph.verify(IRDialect.RUNTIME)

        report = VerifiedPipeline(
            (RuntimeAttentionLayoutPass(),), shape_profile={},
        ).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual([node.op_type for node in graph.nodes], ["CrossSDPA"])
        self.assertEqual(graph.nodes[0].input_map(), {"q": "q", "k": "k", "v": "v"})
        self.assertEqual(graph.nodes[0].output_map(), {"out": "out"})
        _assert_portable(self, graph, {})

    def test_exact_causal_keep_initializer_becomes_flag(self):
        graph = GraphIR("volvoxai", "keep.json", dialect=IRDialect.RUNTIME)
        tensors: dict[str, np.ndarray] = {}
        for name in ("q", "k", "v"):
            _tensor(graph, name, (3, 8), public_input=True)
        causal = np.tril(np.ones((3, 3), dtype=np.int32))
        _tensor(graph, "causal.keep", causal.shape, dtype="int32", initializer=True)
        tensors["causal.keep"] = causal
        _node(
            graph,
            "attention",
            "CrossSDPA",
            {"q": "q", "k": "k", "v": "v", "mask": "causal.keep"},
            "out",
            (3, 8),
            params={"heads": 2, "causal": False, "scale": 0.5},
        )
        _output(graph, "out")
        graph.verify(IRDialect.RUNTIME)

        report = VerifiedPipeline(
            (RuntimeKeepMaskPass(tensors),), shape_profile={},
        ).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertNotIn("mask", graph.nodes[0].input_map())
        self.assertTrue(graph.nodes[0].attributes[0].value["causal"])
        np.testing.assert_array_equal(tensors["causal.keep"], causal)
        _assert_portable(self, graph, tensors)


class RuntimeQuantizedAttentionTests(unittest.TestCase):
    def test_exact_layout_movement_reuses_every_affine_and_payload(self):
        graph, tensors = _quantized_layout_graph()
        affines = {
            name: graph.tensors[name].quantization
            for name in ("q.byte", "k.byte", "v.byte", "out.byte")
        }
        payloads = {name: value.tobytes() for name, value in tensors.items()}

        report = VerifiedPipeline(
            (RuntimeQuantizedAttentionLayoutPass(tensors),), shape_profile={},
        ).run(graph)

        self.assertEqual(report.total_changes, 1)
        attention = next(node for node in graph.nodes if node.op_type == "QSDPA")
        for port, shape in (("q", (3, 8)), ("k", (5, 8)), ("v", (5, 8))):
            name = attention.input_map()[port]
            self.assertEqual(graph.tensors[name].shape, shape)
            self.assertEqual(graph.tensors[name].quantization, affines[f"{port}.byte"])
        self.assertEqual(graph.tensors[attention.output_map()["out"]].shape, (3, 8))
        self.assertEqual(
            graph.tensors[attention.output_map()["out"]].quantization,
            affines["out.byte"],
        )
        output_dq = next(node for node in graph.nodes if node.op_type == "DequantizeLinear")
        self.assertEqual(output_dq.output_map(), {"out": "out"})
        self.assertEqual(
            {name: value.tobytes() for name, value in tensors.items()}, payloads,
        )
        _assert_portable(self, graph, tensors)

    def test_static_qdq_fusion_is_atomic_and_preserves_boundary_domains(self):
        graph, tensors = _quantized_fusion_graph()
        self.assertEqual(quantized_attention_candidate_count(graph), 1)
        affines = {
            name: graph.tensors[name].quantization
            for name in ("q.byte", "k.byte", "v.byte", "context.byte")
        }
        payloads = {name: value.tobytes() for name, value in tensors.items()}

        report = VerifiedPipeline((RuntimeQuantizedAttentionFusionPass(
            tensors, allow_numerical_migration=True,
        ),), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual(dict(report.runs[0].metrics), {
            "attention_candidates_fused": 1,
            "attention_candidates_refused": 0,
        })
        self.assertEqual(
            serialize_pipeline_report(report)["runs"][0]["metrics"],
            dict(report.runs[0].metrics),
        )
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QuantizeLinear", "QuantizeLinear", "QSDPA",
             "DequantizeLinear"],
        )
        attention = graph.nodes[-2]
        self.assertEqual(attention.input_map(), {
            "q": "q.byte", "k": "k.byte", "v": "v.byte",
        })
        self.assertEqual(attention.attributes[0].value, {
            "heads": 2, "causal": False, "scale": 1.0,
        })
        for name, affine in affines.items():
            self.assertEqual(graph.tensors[name].quantization, affine)
        self.assertEqual(graph.tensors["q.byte"].shape, (3, 8))
        self.assertEqual(graph.tensors["k.byte"].shape, (5, 8))
        self.assertEqual(graph.tensors["v.byte"].shape, (5, 8))
        self.assertEqual(graph.tensors["context.byte"].shape, (3, 8))
        self.assertEqual(graph.nodes[-1].output_map(), {"out": "context"})
        self.assertEqual(
            {name: value.tobytes() for name, value in tensors.items()}, payloads,
        )
        _assert_portable(self, graph, tensors)

    def test_fusion_refuses_bad_merge_and_leaves_graph_and_payloads_unchanged(self):
        graph, tensors = _quantized_fusion_graph(bad_merge=True)
        before = graph.fingerprint()
        payloads = {name: value.tobytes() for name, value in tensors.items()}
        pass_ = RuntimeQuantizedAttentionFusionPass(
            tensors, allow_numerical_migration=True,
        )

        report = VerifiedPipeline((pass_,), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), before)
        self.assertEqual(
            {name: value.tobytes() for name, value in tensors.items()}, payloads,
        )
        self.assertGreaterEqual(pass_.refused, 1)
        self.assertEqual(
            dict(report.runs[0].metrics)["attention_candidates_refused"],
            pass_.refused,
        )

    def test_requires_explicit_quantized_numerical_migration_policy(self):
        with self.assertRaisesRegex(ValueError, "numerical-migration"):
            RuntimeQuantizedAttentionFusionPass(
                {}, allow_numerical_migration=False,
            )


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponential = np.exp(shifted, dtype=np.float32)
    return exponential / np.sum(
        exponential, axis=-1, keepdims=True, dtype=np.float32,
    )


if __name__ == "__main__":
    unittest.main()
