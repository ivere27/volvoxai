#!/usr/bin/env python3
"""Export ONNX LSTM reference cases for required-backend native/WebGPU checks."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from tools.export_safetensors import export_onnx_model
from tools.generate_tts_validation import encoded
from tools.lstm_fixtures import lstm_model, lstm_tail_fixture


def fixtures():
    rng = np.random.default_rng(9903)
    for direction in (0, 1, 2):
        for time, batch, width, hidden, optional in (
                (1, 1, 3, 1, False), (7, 2, 5, 9, True), (3, 1, 7, 257, True),
                (4, 1, 33, 65, False), (5, 2, 129, 257, True)):
            directions = 2 if direction == 2 else 1
            def random(shape): return rng.normal(0, .15, shape).astype(np.float32)
            x = random((time, batch, width))
            w, r = random((directions, 4 * hidden, width)), random((directions, 4 * hidden, hidden))
            bias = random((directions, 8 * hidden)) if optional else None
            h, c = (random((directions, batch, hidden)) for _ in range(2)) if optional else (None, None)
            yield f"lstm_{direction}_{time}_{batch}_{hidden}", lstm_model(x, w, r, bias, h, c, direction), x, {}
    # Nonzero input-projection compensation must survive the boundary into R.
    x = np.zeros((3, 2, 33), np.float32)
    x[..., 0], x[..., 1] = 1, 2**-24
    w = np.zeros((2, 4*65, 33), np.float32)
    w[:, 3*65:, :2] = 1
    r = np.zeros((2, 4*65, 65), np.float32)
    r[:, 3*65:, 0] = 2**-24
    h = np.ones((2, 2, 65), np.float32)
    c = np.zeros_like(h)
    yield "lstm_projection_correction", lstm_model(x, w, r, None, h, c, 2), x, {}
    tail = lstm_tail_fixture()
    yield "lstm_finite_tanh_tails", lstm_model(*tail), tail[0], {}
    w = rng.normal(0, .15, (2, 16, 3)).astype(np.float32)
    r = rng.normal(0, .15, (2, 16, 4)).astype(np.float32)
    for time in (1, 5, 8):
        x = rng.normal(0, .15, (time, 2, 3)).astype(np.float32)
        model = lstm_model(x, w, r, direction=2)
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "T"
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "T"
        yield f"lstm_bounded_time_{time}", model, x, {"dimension_bounds": {"T": {"min": 1, "max": 8}}}
    source = Path("build/tts-examples/assets/kokoro/onnx/model.onnx")
    if source.exists():
        model = onnx.load(source)
        node = next(node for node in model.graph.node if node.op_type == "LSTM")
        names = set(node.input[1:4])
        weights = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer if value.name in names}
        w, r, bias = (weights[name] for name in node.input[1:4])
        x = rng.normal(0, .5, (11, 1, w.shape[-1])).astype(np.float32)
        yield "kokoro_learned_lstm", lstm_model(x, w, r, bias, direction=2), x, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/lstm-validation/cases.json"))
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    cases = []
    with tempfile.TemporaryDirectory(prefix="volvox-lstm-validation-") as directory:
        root = Path(directory)
        for name, model, x, export_options in fixtures():
            session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            expected = session.run(None, {"x": x})[0]
            changed = np.ascontiguousarray(x * np.float32(1.125) + np.float32(.03125))
            changed_expected = session.run(None, {"x": changed})[0]
            source, weights = root / f"{name}.onnx", root / name / "model.safetensors"
            weights.parent.mkdir()
            onnx.save(model, source)
            graph = export_onnx_model(str(source), str(weights), **export_options)
            cases.append({"name": name, "graph": graph,
                "weight_shards": [base64.b64encode(weights.read_bytes()).decode()],
                "inputs": [encoded("input0", x)], "expected": [encoded("output0", expected)],
                "requests": [
                    {"inputs": [encoded("input0", x)], "expected": [encoded("output0", expected)]},
                    {"inputs": [encoded("input0", changed)], "expected": [encoded("output0", changed_expected)]}]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}) + "\n")
    print(f"Wrote {len(cases)} ONNX LSTM references to {args.out}")


if __name__ == "__main__":
    main()
