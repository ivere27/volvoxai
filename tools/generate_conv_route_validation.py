#!/usr/bin/env python3
"""Independent Conv oracles for the vector route and its scalar boundaries.

Generate with ``python -m tools.generate_conv_route_validation --out FILE``;
replay FILE with tools/verify_tts_web.mjs or tools/verify_tts_native.py. Each
geometry uses two actual inputs, and the public runners execute every case
twice. No expected activation participates in inference.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.export_safetensors import export_onnx_model
from tools.generate_tts_validation import encoded


def fixtures():
    # n, ic, oc, h, w, kh, kw, groups, dy, dx, sy, sx, py, px, bias, relu
    shapes = {
        "tts_regular": (1, 8, 32, 1, 19, 1, 3, 1, 1, 1, 1, 1, 0, 1, True, False),
        "tts_dilated": (1, 16, 16, 1, 21, 1, 3, 1, 1, 3, 1, 1, 0, 3, True, False),
        "pointwise": (1, 17, 32, 1, 13, 1, 1, 1, 1, 1, 1, 1, 0, 0, True, False),
        "batch_spatial": (2, 3, 32, 9, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, False),
        "stride_dilation": (2, 5, 16, 11, 17, 3, 3, 1, 2, 2, 2, 2, 2, 2, True, False),
        "no_bias": (1, 5, 16, 1, 23, 1, 5, 1, 1, 1, 1, 1, 0, 2, False, False),
        "relu": (1, 8, 16, 1, 15, 1, 3, 1, 1, 1, 1, 1, 0, 1, True, True),
        "grouped_boundary": (1, 8, 32, 1, 17, 1, 3, 2, 1, 1, 1, 1, 0, 1, True, False),
        "depthwise_boundary": (1, 8, 16, 1, 17, 1, 3, 8, 1, 1, 1, 1, 0, 1, True, False),
        "single_channel_boundary": (1, 1, 16, 1, 17, 1, 3, 1, 1, 1, 1, 1, 0, 1, True, False),
        "partial_vector_boundary": (1, 8, 17, 1, 17, 1, 3, 1, 1, 1, 1, 1, 0, 1, True, False),
    }
    rng = np.random.default_rng(2041)
    for name, shape in shapes.items():
        n, ic, oc, h, w, kh, kw, groups, dy, dx, sy, sx, py, px, bias, relu = shape
        weight = rng.normal(0, .1, (oc, ic // groups, kh, kw)).astype(np.float32)
        b = rng.normal(0, .05, oc).astype(np.float32)
        oh = (h + 2 * py - dy * (kh - 1) - 1) // sy + 1
        ow = (w + 2 * px - dx * (kw - 1) - 1) // sx + 1
        for request in range(2):
            x = rng.normal(0, .2, (n, ic, h, w)).astype(np.float32)
            # Independent NCHW scalar reference with F32 product and sum.
            y = np.zeros((n, oc, oh, ow), np.float32)
            for ni in range(n):
                for out in range(oc):
                    group = out // (oc // groups)
                    for oy in range(oh):
                        for ox in range(ow):
                            total = np.float32(0)
                            for inp in range(ic // groups):
                                for ky in range(kh):
                                    iy = oy * sy + ky * dy - py
                                    if not 0 <= iy < h:
                                        continue
                                    for kx in range(kw):
                                        ix = ox * sx + kx * dx - px
                                        if 0 <= ix < w:
                                            total += x[ni, group * (ic // groups) + inp, iy, ix] * weight[out, inp, ky, kx]
                            y[ni, out, oy, ox] = total + (b[out] if bias else np.float32(0))
            if relu:
                y = np.maximum(y, np.float32(0))
            constants = [numpy_helper.from_array(weight, "weight")]
            inputs = ["x", "weight"]
            if bias:
                constants.append(numpy_helper.from_array(b, "bias"))
                inputs.append("bias")
            nodes = [helper.make_node("Conv", inputs, ["conv" if relu else "y"],
                group=groups, dilations=[dy, dx], strides=[sy, sx], pads=[py, px, py, px])]
            if relu:
                nodes.append(helper.make_node("Relu", ["conv"], ["y"]))
            graph = helper.make_graph(nodes, name,
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, y.shape)], constants)
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            yield f"{name}_input{request}", model, x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    with tempfile.TemporaryDirectory(prefix="volvox-conv-route-") as directory:
        root = Path(directory)
        for name, model, x, y in fixtures():
            source = root / f"{name}.onnx"
            onnx.save(model, source)
            package = root / name
            package.mkdir()
            weight = package / "model.safetensors"
            graph = export_onnx_model(str(source), str(weight))
            cases.append(dict(name=name, graph=graph,
                weight_shards=[base64.b64encode(weight.read_bytes()).decode()],
                inputs=[encoded("input0", x)], expected=[encoded("output0", y)]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"Wrote {len(cases)} independent Conv oracles")


if __name__ == "__main__":
    main()
