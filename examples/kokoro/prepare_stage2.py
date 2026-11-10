#!/usr/bin/env python3
"""Prepare the real final Kokoro generator stage with explicit norm lowering."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.kokoro.prepare import (HEAD_INPUT, OVERLAP_OUTPUT, SAMPLE_RATE, export_status,
    extract, relative, session, specialized_head, write_wave)
from examples.tts_common.artifacts import sha256, write_case

STAGE_INPUT = "/decoder/decoder/generator/Add_5_output_0"


def normalize_patterns(model):
    """Replace only centered spatial variance with epsilon inside its sqrt.

    ONNX InstanceNormalization specifies this same formula; implementations
    may accumulate differently. Original source ONNX remains the comparator.
    """
    model = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    values = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    shapes = {value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in list(model.graph.input) + list(model.graph.value_info)}
    def axes(node):
        attrs = {value.name: helper.get_attribute_value(value) for value in node.attribute}
        if attrs.get("keepdims", 1) != 1:
            return None
        result = values.get(node.input[1]) if len(node.input) > 1 else attrs.get("axes")
        return [int(value) for value in np.asarray(result).reshape(-1)] if result is not None else None
    def node(name, kind):
        result = producers.get(name)
        return result if result is not None and result.op_type == kind else None
    replaced = []
    for index, current in enumerate(model.graph.node):
        if current.op_type == "Pow":
            exponent = values.get(current.input[1])
            if exponent is not None and exponent.size == 1 and exponent.item() == 2:
                current.CopyFrom(helper.make_node("Mul", [current.input[0], current.input[0]],
                    list(current.output), name=current.name + "__square"))
                replaced.append({"kind": "square", "node": current.name})
            continue
        if current.op_type != "Div":
            continue
        sub, sqrt = node(current.input[0], "Sub"), node(current.input[1], "Sqrt")
        if sub is None or sqrt is None:
            continue
        add = node(sqrt.input[0], "Add")
        mean = node(sub.input[1], "ReduceMean")
        if add is None or mean is None or mean.input[0] != sub.input[0] or axes(mean) not in ([2], [-1]):
            continue
        variance, epsilon = None, None
        for var_name, eps_name in (add.input, reversed(add.input)):
            reduction, constant = node(var_name, "ReduceMean"), values.get(eps_name)
            if reduction is not None and constant is not None and constant.size == 1:
                variance, epsilon = reduction, float(constant.item())
        if variance is None or axes(variance) != axes(mean) or not np.isfinite(epsilon) or epsilon < 0:
            continue
        square = node(variance.input[0], "Mul")
        if square is None or list(square.input) != [sub.output[0], sub.output[0]]:
            continue
        shape = shapes.get(sub.input[0], [])
        if len(shape) != 3 or shape[1] <= 0:
            continue
        scale_name, bias_name = f"stage_norm_{index}_ones", f"stage_norm_{index}_zeros"
        model.graph.initializer.extend([
            numpy_helper.from_array(np.ones(shape[1], dtype=np.float32), scale_name),
            numpy_helper.from_array(np.zeros(shape[1], dtype=np.float32), bias_name),
        ])
        name = current.name
        current.CopyFrom(helper.make_node("InstanceNormalization", [sub.input[0], scale_name, bias_name],
            list(current.output), name=name + "__instance_norm", epsilon=epsilon))
        replaced.append({"kind": "instance_normalization", "node": name, "epsilon": epsilon,
            "source_axes": axes(mean), "channels": shape[1]})
    return model, replaced


def prepare(args):
    full_path = args.manifest.resolve()
    full = json.loads(full_path.read_text())
    root = full_path.parent / "generator-stage2"
    if root.exists():
        raise RuntimeError(f"Choose a fresh prepared whole-model directory; component already exists: {root}")
    model = onnx.load(full_path.parent / full["onnx"]["path"])
    inputs = dict(np.load(full_path.parent / full["cases"][0]["inputs"], allow_pickle=False))
    observed = copy.deepcopy(model)
    observed.graph.output.extend([helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        for name in (STAGE_INPUT, HEAD_INPUT, OVERLAP_OUTPUT)])
    audio, features, final_features, overlap = session(observed.SerializeToString(), args.threads).run(None, inputs)
    boundaries = {STAGE_INPUT: features, "style": inputs["style"]}
    source = extract(model, boundaries, {"waveform": audio})
    lowered = specialized_head(model, final_features, overlap, audio, boundaries=boundaries)
    lowered, rewrites = normalize_patterns(lowered)
    # Remove the now-unused decomposed variance and square subgraphs.
    lowered = extract(lowered, boundaries, {"waveform": audio})
    root.mkdir()
    onnx.save(source, root / "source.onnx")
    onnx.save(lowered, root / "specialized.onnx")
    reference = session(str(root / "source.onnx"), args.threads).run(None, boundaries)[0]
    candidate = session(str(root / "specialized.onnx"), args.threads).run(None, boundaries)[0]
    np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(reference, audio, rtol=1e-4, atol=1e-6)
    status = export_status(root / "specialized.onnx", root / "package", {name: list(value.shape) for name, value in boundaries.items()})
    if status["status"] == "ready":
        status["graph"] = relative(Path(status["graph"]), root)
        status["weights"] = [relative(Path(path), root) for path in status["weights"]]
        status["inputs"] = {"features": "input0", "style": "input1"}
    case = write_case(root, "hello", {"features": features, "style": inputs["style"]}, {"audio": reference}, sample_rate=SAMPLE_RATE)
    case["validation"] = {"deterministic": True, "reference_engine": "onnxruntime",
        "reason": "Captured real features and style; deterministic pretrained stage."}
    write_wave(root / "reference.wav", reference)
    manifest = {"format": "volvoxai-tts-example/v1", "model": {**full["model"], "name": "Kokoro pretrained generator stage 2"},
        "scope": "hybrid: ORT supplies generator stage-2 features; Volvox runs final residual/snake blocks, learned spectral head and waveform synthesis",
        "sample_rate": SAMPLE_RATE, "onnx": {"path": "source.onnx", "inputs": {"features": STAGE_INPUT, "style": "style"}, "outputs": {"audio": "waveform"}},
        "volvox": status, "cases": [case], "reference_audio": "reference.wav",
        "provenance": {"whole_model_manifest": "../manifest.json", "source_sha256": sha256(root / "source.onnx"),
            "specialized_sha256": sha256(root / "specialized.onnx"), "fixed_feature_shape": list(features.shape),
            "rewrites": rewrites, "normalization_semantics": "centered spatial variance; epsilon inside sqrt; unchanged style affine",
            "bit_identical_specialization": bool(np.array_equal(candidate, reference)),
            "specialization_max_absolute_error": float(np.max(np.abs(candidate-reference))),
            "whole_model_max_absolute_error": float(np.max(np.abs(reference-audio))),
            "source_node_count": len(source.graph.node), "specialized_node_count": len(lowered.graph.node)}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(root / "manifest.json"), "status": status,
        "rewrites": len(rewrites), "max_absolute_error": manifest["provenance"]["specialization_max_absolute_error"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("build/tts-examples/kokoro/manifest.json"))
    parser.add_argument("--threads", type=int, default=4)
    prepare(parser.parse_args())
