#!/usr/bin/env python3
"""Export ONNX F32/I32 comparison oracles for required-backend execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper
from tools.generate_tts_validation import encoded


def fixtures():
    for dtype, values in ((np.float32, [-np.inf, -3.5, -0., 0., 2.25, np.inf, np.nan]),
                          (np.int32, [-2**31, -7, 0, 4, 2**31 - 1])):
        values = np.array(values, dtype=dtype)
        for operator in ("Equal", "GreaterOrEqual"):
            for layout, a, b in (("broadcast", values[:, None], values[None, :]),
                                 ("scalar", np.array(0, dtype=dtype), np.array(0, dtype=dtype)),
                                 ("scalar_tensor", np.array(0, dtype=dtype), values),
                                 ("tensor_scalar", values, np.array(0, dtype=dtype))):
                kind = TensorProto.FLOAT if dtype == np.float32 else TensorProto.INT32
                output_shape = np.broadcast_shapes(a.shape, b.shape)
                model = helper.make_model(helper.make_graph(
                    [helper.make_node(operator, ["a", "b"], ["y"])], "comparison",
                    [helper.make_tensor_value_info("a", kind, a.shape),
                     helper.make_tensor_value_info("b", kind, b.shape)],
                    [helper.make_tensor_value_info("y", TensorProto.BOOL, output_shape)]),
                    opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
                yield f"{operator}_{np.dtype(dtype).name}_{layout}", model, a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/comparison-validation/cases.json"))
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    cases = []
    for name, model, a, b in fixtures():
        expected = ort.InferenceSession(model.SerializeToString(), options,
            providers=["CPUExecutionProvider"]).run(None, {"a": a, "b": b})[0].astype(np.int32)
        # Author the small canonical IR directly so rank-zero runtime coverage
        # does not depend on the source importer's public scalar-shape policy.
        graph = {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {f"input{i}": {"shape": list(value.shape), "dtype": value.dtype.name}
                       for i, value in enumerate((a, b))},
            "outputs": ["output0"], "nodes": [{"id": "compare", "opType": model.graph.node[0].op_type,
                "inputs": {"a": "input0", "b": "input1"}, "params": {},
                "outputs": {"out": {"tensor": "output0", "shape": list(expected.shape), "dtype": "int32"}}}],
            "execution": {"entrypoints": [{"name": "evaluate", "effect": 1, "allowed_ranges": [1],
                "inputs": ["input0", "input1"], "outputs": ["output0"]}], "stateless": {}}}
        def tensor(name, value):
            result = encoded(name, value)
            result["shape"] = list(value.shape)
            return result
        cases.append({"name": name, "graph": graph, "weight_shards": [],
            "inputs": [tensor("input0", a), tensor("input1", b)],
            "expected": [tensor("output0", expected)]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}) + "\n")
    print(f"Wrote {len(cases)} ONNX comparison references to {args.out}")


if __name__ == "__main__":
    main()
