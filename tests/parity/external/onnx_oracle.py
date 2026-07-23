#!/usr/bin/env python3
"""External correctness oracle: run the *source* ONNX graph in ONNX Runtime and
emit signatures in the exact format tests/parity/lib/extract.mjs produces, so the
JS comparator can check VolvoxAI's tiers against an independent, industry-standard
runtime — not just against VolvoxAI's own pure-JS reference.

Reads the same seeded input fixtures the JS/native tiers use (out/fixtures/), so
every runtime sees byte-identical input. Writes out/<model>.onnx.json.

Usage:  python3 tests/parity/external/onnx_oracle.py [model_name ...]
"""
import json
import os
import sys
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PARITY))
OUT = os.path.join(PARITY, "out")
sys.path.insert(0, HERE)
from sigutil import signature, write_json_atomic  # noqa: E402

DT = {"f32": np.float32, "i32": np.int32, "u8": np.uint8, "i8": np.int8}


def read_fixture(model_name, inp):
    p = os.path.join(OUT, "fixtures", model_name, f"{inp['name']}.{ 'f32' if inp['dtype']=='f32' else inp['dtype'] }")
    arr = np.fromfile(p, dtype=DT[inp["dtype"]])
    return arr.reshape(inp["shape"])


def run_model(model_name, model):
    dst = os.path.join(OUT, f"{model_name}.onnx.json")
    try:
        os.unlink(dst)
    except FileNotFoundError:
        pass
    ext = model["external"]
    onnx_path = os.path.join(ROOT, ext["onnx"])
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    feed = {}
    for inp in model["inputs"]:
        onnx_name = ext["inputs"][inp["name"]]
        feed[onnx_name] = read_fixture(model_name, inp)

    out_names = list(ext["outputs"].keys())
    results = sess.run(out_names, feed)
    onnx_to_volvox = ext["outputs"]

    out_spec = {s["name"]: s for s in model["outputs"]}
    sigs = {}
    for onnx_name, val in zip(out_names, results):
        vname = onnx_to_volvox[onnx_name]
        spec = out_spec[vname]
        sigs[vname] = signature(
            val,
            topk=spec.get("topk", 10),
            shape=list(val.shape),
            sample_axes=spec.get("sampleAxes", []),
        )

    write_json_atomic(
        dst,
        {"model": model_name, "backend": "onnx", "ms": None, "sigs": sigs},
        indent=1,
    )
    print(f"onnx-oracle {model_name}: ok -> {os.path.relpath(dst, ROOT)}")


def main():
    policy = json.load(open(os.path.join(PARITY, "policy.json")))
    only = set(sys.argv[1:])
    ran = 0
    for name, model in policy["models"].items():
        if "onnx" not in model.get("external", {}):
            continue
        if only and name not in only:
            continue
        run_model(name, model)
        ran += 1
    if ran == 0:
        print("no external-oracle models to run (need policy.external + fixtures)")


if __name__ == "__main__":
    main()
