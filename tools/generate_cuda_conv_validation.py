#!/usr/bin/env python3
"""Independent ONNX Conv cases around the CUDA four-spatial-output route.

Each retained context receives two real inputs. Replay with
tools/verify_lstm_native.py --operator conv, optionally checking exact bits
against a separately captured legacy CUDA output file.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from tools.export_safetensors import export_onnx_model
from tools.generate_tts_validation import encoded


def fixtures():
    # N, I, O, H, W, KW, groups, dilation, stride, padding, bias, Relu.
    shapes = {
        "threshold_below": (1, 3, 64, 1, 2047, 3, 1, 1, 1, 1, True, False),
        "threshold_exact": (1, 3, 64, 1, 2048, 3, 1, 1, 1, 1, True, False),
        "width_channel_tail": (1, 8, 65, 1, 2051, 3, 1, 1, 1, 1, True, False),
        "dilated": (1, 5, 129, 1, 1027, 5, 1, 3, 1, 6, True, False),
        "stride_two": (1, 7, 64, 1, 4099, 3, 1, 1, 2, 1, True, False),
        "batch_tail": (2, 5, 65, 1, 1025, 3, 1, 1, 1, 1, True, False),
        "no_bias_relu": (1, 9, 64, 1, 2049, 7, 1, 5, 1, 15, False, True),
        "grouped_boundary": (1, 8, 64, 1, 2049, 3, 2, 1, 1, 1, True, False),
        "depthwise_boundary": (1, 64, 64, 1, 2049, 3, 64, 1, 1, 1, True, False),
        "channel_boundary": (1, 3, 31, 1, 4229, 3, 1, 1, 1, 1, True, False),
        "pointwise_boundary": (1, 3, 64, 1, 2049, 1, 1, 1, 1, 0, True, False),
        "height_boundary": (1, 3, 64, 2, 1025, 3, 1, 1, 1, 1, True, False),
    }
    rng = np.random.default_rng(39151)
    for name, shape in shapes.items():
        n, ic, oc, height, width, kw, groups, dilation, stride, padding, bias, relu = shape
        weight = rng.normal(0, .1, (oc, ic // groups, 1, kw)).astype(np.float32)
        constants = [numpy_helper.from_array(weight, "weight")]
        inputs = ["x", "weight"]
        if bias:
            constants.append(numpy_helper.from_array(rng.normal(0, .05, oc).astype(np.float32), "bias"))
            inputs.append("bias")
        output_width = (width + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
        nodes = [helper.make_node("Conv", inputs, ["conv" if relu else "y"],
            group=groups, dilations=[1, dilation], strides=[1, stride],
            pads=[0, padding, 0, padding])]
        if relu:
            nodes.append(helper.make_node("Relu", ["conv"], ["y"]))
        graph = helper.make_graph(nodes, name,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, ic, height, width])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, oc, height, output_width])], constants)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        requests = [rng.normal(0, .2, (n, ic, height, width)).astype(np.float32) for _ in range(2)]
        yield name, shape, model, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    cases = []
    with tempfile.TemporaryDirectory(prefix="volvox-cuda-conv-validation-") as directory:
        root = Path(directory)
        for name, _, model, inputs in fixtures():
            session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            requests = [{"inputs": [encoded("input0", x)],
                         "expected": [encoded("output0", session.run(None, {"x": x})[0])]}
                        for x in inputs]
            source, weight = root / f"{name}.onnx", root / name / "model.safetensors"
            weight.parent.mkdir()
            onnx.save(model, source)
            graph = export_onnx_model(str(source), str(weight))
            cases.append({"name": name, "graph": graph,
                "weight_shards": [base64.b64encode(weight.read_bytes()).decode()],
                "inputs": requests[0]["inputs"], "expected": requests[0]["expected"],
                "requests": requests})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}) + "\n")
    print(f"Wrote {len(cases)} independent ONNX Conv cases to {args.out}")


if __name__ == "__main__":
    main()
