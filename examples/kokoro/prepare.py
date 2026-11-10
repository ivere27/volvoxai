#!/usr/bin/env python3
"""Kokoro source helpers and the complete VolvoxAI preparation entrypoint."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import sys
import time
import wave

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.kokoro.fetch import FILES, REPOSITORY, REVISION, fetch
from examples.tts_common.artifacts import sha256, write_case

SAMPLE_RATE = 24_000
HEAD_INPUT = "/decoder/decoder/generator/Div_4_output_0"
OVERLAP_OUTPUT = "/decoder/decoder/generator/istft/stft/ConvTranspose_output_0"
PREFIX = "/decoder/decoder/generator/istft/stft/"


def relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def session(model, threads: int):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])


def write_wave(path: Path, audio: np.ndarray) -> None:
    pcm = np.rint(np.clip(audio.reshape(-1), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def extract(model, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]):
    """Cut actual source tensors; retain the original operators and weights."""
    producers = {name: index for index, node in enumerate(model.graph.node) for name in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    needed_nodes, needed_weights = set(), set()

    def visit(name):
        if not name or name in inputs:
            return
        if name in initializers:
            needed_weights.add(name)
            return
        if name not in producers:
            raise RuntimeError(f"Missing explicit component boundary: {name}")
        index = producers[name]
        if index in needed_nodes:
            return
        needed_nodes.add(index)
        for value in model.graph.node[index].input:
            visit(value)

    for name in outputs:
        visit(name)
    nodes = [copy.deepcopy(node) for index, node in enumerate(model.graph.node) if index in needed_nodes]
    def value_info(name, array):
        if isinstance(array, onnx.ValueInfoProto):
            if array.name != name:
                raise ValueError("Explicit output metadata must use its source tensor name")
            return copy.deepcopy(array)
        return helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(array.dtype), list(array.shape))
    graph = helper.make_graph(nodes, "kokoro-pretrained-component",
        [value_info(name, array) for name, array in inputs.items()],
        [value_info(name, array) for name, array in outputs.items()],
        [copy.deepcopy(value) for name, value in initializers.items() if name in needed_weights])
    result = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    onnx.checker.check_model(result)
    return result


def specialized_head(model, activation: np.ndarray, overlap: np.ndarray, audio: np.ndarray, *, boundaries=None):
    """Specialize only the source iSTFT window-index program to a fixed length.

    Source ScatterND divides precisely the samples whose constant window sum
    exceeds FLT_MIN. Dividing the other samples by one preserves them. Keep
    division, scaling, and the two crops as separate operations, in source order.
    No model activations or learned parameters are replaced by constants.
    """
    head = extract(model, boundaries or {HEAD_INPUT: activation}, {OVERLAP_OUTPUT: overlap})
    values = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    window = values["decoder.decoder.generator.stft.istft.stft.window_sum"].reshape(-1)
    length = overlap.shape[-1]
    threshold = values[PREFIX + "Constant_9_output_0"].item()
    # Slice clamps to the table length. Source ScatterND leaves the remainder
    # untouched when the synthesized waveform is longer than this table.
    divisor = np.ones(length, dtype=np.float32)
    selected = window[:length]
    divisor[:selected.size] = np.where(selected > threshold, selected, np.float32(1))
    divisor = divisor.reshape(1, 1, -1)
    constants = {
        "head_window_divisor": divisor.astype(np.float32),
        "head_window_scale": values[PREFIX + "Constant_19_output_0"],
        "head_crop_start": values[PREFIX + "Constant_21_output_0"],
        "head_crop_end": values[PREFIX + "Constant_26_output_0"],
        "head_crop_axis": np.array([2], dtype=np.int64),
        "head_audio_shape": np.array(audio.shape, dtype=np.int64),
    }
    head.graph.initializer.extend(numpy_helper.from_array(value, name) for name, value in constants.items())
    head.graph.node.extend([
        helper.make_node("Div", [OVERLAP_OUTPUT, "head_window_divisor"], ["head_normalized"]),
        helper.make_node("Mul", ["head_normalized", "head_window_scale"], ["head_scaled"]),
        helper.make_node("Slice", ["head_scaled", "head_crop_start", "head_crop_end", "head_crop_axis"], ["head_cropped"]),
        helper.make_node("Reshape", ["head_cropped", "head_audio_shape"], ["waveform"]),
    ])
    del head.graph.output[:]
    head.graph.output.extend([helper.make_tensor_value_info("waveform", onnx.TensorProto.FLOAT, list(audio.shape))])
    onnx.checker.check_model(head)
    return head


def export_status(model: Path, package: Path, shapes: dict, input_dtypes=None):
    from tools.export_safetensors import export_onnx_model
    from tools.exporter.errors import ExporterError
    try:
        graph = export_onnx_model(str(model), str(package / "model.safetensors"), weight_dtype="float32",
            input_shapes=shapes, input_dtypes=input_dtypes)
    except ExporterError as error:
        return {"status": "unsupported", "reason": str(error), "diagnostic": error.diagnostic.to_dict()}
    return {"status": "ready", "graph": str(package / "graph.json"),
        "weights": [str(path) for path in sorted(package.glob("*.safetensors"))],
        "inputs": {name: f"input{index}" for index, name in enumerate(shapes)},
        "outputs": {"audio": "output0"}, "node_count": len(graph["nodes"])}


def compatibility(model) -> list[dict]:
    """Source evidence beyond the first failing node, not a support whitelist."""
    categories = [
        ("recurrent text and duration encoders", {"LSTM"},
         "Six bidirectional recurrent operators need an exact recurrence lowering."),
        ("duration-dependent output shape", {"Round", "Range"},
         "The first Range limit comes from predicted, rounded durations. Fixing token count does not fix waveform length."),
        ("source excitation and spectral analysis", {"STFT", "Atan", "Floor", "Resize"},
         "The source needs a 20-point STFT with hop 5, phase reconstruction, and both nearest and half-pixel linear resize."),
        ("inverse spectral window indexing", {"NonZero", "ScatterND"},
         "The supplied spectral-head component specializes this constant window-index program for its fixed feature shape."),
    ]
    return [{"category": title, "requirement": requirement,
        "nodes": [{"name": node.name, "op": node.op_type} for node in model.graph.node if node.op_type in ops]}
        for title, ops, requirement in categories]


def phoneme_inputs(assets: Path, phonemes: str, speed: float) -> dict[str, np.ndarray]:
    vocab = json.loads((assets / "tokenizer.json").read_text())["model"]["vocab"]
    missing = sorted(set(phonemes) - vocab.keys())
    if missing:
        raise ValueError(f"Phonemes absent from the pinned vocabulary: {missing}")
    ids = [vocab[character] for character in phonemes]
    if not 1 <= len(ids) <= 510 or not np.isfinite(speed) or speed <= 0:
        raise ValueError("Use 1..510 phoneme characters and finite positive speed")
    with np.errstate(over="ignore", under="ignore"):
        speed_f32 = np.float32(speed)
    if not np.isfinite(speed_f32) or speed_f32 <= 0:
        raise ValueError("Speed must remain finite and positive in F32")
    voices = np.fromfile(assets / "voices/af_heart.bin", dtype=np.float32).reshape(-1, 1, 256)
    return {"input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
        "style": voices[len(ids)].copy(), "speed": np.array([speed_f32], dtype=np.float32)}


def prepare(args):
    import onnxruntime as ort

    assets, root = args.assets.resolve(), args.output.resolve()
    if (root / "manifest.json").exists() or (root / "cases").exists():
        raise RuntimeError(f"Output already contains a prepared example; choose another --output: {root}")
    fetch(assets)
    root.mkdir(parents=True, exist_ok=True)
    inputs = phoneme_inputs(assets, args.phonemes, args.speed)
    path = assets / "onnx/model.onnx"
    model = onnx.load(path)
    random_nodes = [{"name": node.name, "op": node.op_type} for node in model.graph.node
        if node.op_type.startswith("Random") or node.op_type in {"Multinomial", "Bernoulli"}]
    # Observe genuine intermediate activations without writing another full model.
    observed = copy.deepcopy(model)
    observed.graph.output.extend([helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        for name in (HEAD_INPUT, OVERLAP_OUTPUT)])
    runtime = session(observed.SerializeToString(), args.threads)
    started = time.perf_counter()
    audio, activation, overlap = runtime.run(None, inputs)
    elapsed = time.perf_counter() - started
    repeated = runtime.run(["waveform"], inputs)[0]
    deterministic = not random_nodes and np.array_equal(audio, repeated)
    if not deterministic:
        raise RuntimeError("Pinned Kokoro graph was not repeatable; refusing deterministic comparison")
    del runtime, observed
    model_info = {"name": "Kokoro-82M-v1.0 FP32", "family": "kokoro",
        "source_url": f"https://huggingface.co/{REPOSITORY}/tree/{REVISION}",
        "revision": REVISION, "sha256": FILES["onnx/model.onnx"],
        "license": "Apache-2.0", "voice": "af_heart", "precision": "float32"}
    case = write_case(root, "hello", inputs, {"audio": audio}, sample_rate=SAMPLE_RATE)
    case.update(text=args.text, phonemes=args.phonemes, validation={"deterministic": True, "reference_engine": "onnxruntime",
        "reason": "No stochastic nodes; two real-model ORT executions were bit-identical."})
    write_wave(root / "reference.wav", audio)
    full_status = export_status(path, root / "package", {name: list(value.shape) for name, value in inputs.items()}, {"input_ids": "int32"})
    if full_status["status"] == "ready":
        full_status["graph"] = relative(Path(full_status["graph"]), root)
        full_status["weights"] = [relative(Path(value), root) for value in full_status["weights"]]
    manifest = {"format": "volvoxai-tts-example/v1", "model": model_info,
        "scope": "complete pretrained phonemes-to-waveform model", "sample_rate": SAMPLE_RATE,
        "onnx": {"path": relative(path, root), "inputs": {name: name for name in inputs}, "outputs": {"audio": "waveform"}},
        "volvox": full_status, "cases": [case], "reference_audio": "reference.wav",
        "provenance": {"phonemizer": "explicit caller-supplied phonemes", "random_nodes": random_nodes,
            "ort_version": ort.__version__, "baseline_seconds": elapsed,
            "operator_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
            "compatibility": compatibility(model)}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    component = root / "spectral-head"
    component.mkdir(exist_ok=True)
    original = extract(model, {HEAD_INPUT: activation}, {"waveform": audio})
    original_path = component / "source.onnx"
    onnx.save(original, original_path)
    lowered = specialized_head(model, activation, overlap, audio)
    lowered_path = component / "specialized.onnx"
    onnx.save(lowered, lowered_path)
    reference = session(str(original_path), args.threads).run(None, {HEAD_INPUT: activation})[0]
    candidate = session(str(lowered_path), args.threads).run(None, {HEAD_INPUT: activation})[0]
    if not np.array_equal(reference, candidate):
        raise RuntimeError(f"Spectral normalization specialization changed ORT output: {np.max(np.abs(reference-candidate))}")
    if not np.allclose(reference, audio, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Extracted source head does not reproduce the whole model's waveform")
    status = export_status(lowered_path, component / "package", {HEAD_INPUT: list(activation.shape)})
    if status["status"] == "ready":
        status["graph"] = relative(Path(status["graph"]), component)
        status["weights"] = [relative(Path(value), component) for value in status["weights"]]
        status["inputs"] = {"features": "input0"}
    head_case = write_case(component, "hello", {"features": activation}, {"audio": reference}, sample_rate=SAMPLE_RATE)
    head_case["validation"] = {"deterministic": True, "reference_engine": "onnxruntime",
        "reason": "Real captured features; deterministic pretrained head."}
    write_wave(component / "reference.wav", reference)
    head_manifest = {"format": "volvoxai-tts-example/v1", "model": {**model_info, "name": "Kokoro pretrained spectral synthesis head"},
        "scope": "hybrid: ORT supplies final generator features; Volvox executes learned conv_post and complete spectral waveform synthesis",
        "sample_rate": SAMPLE_RATE, "onnx": {"path": "source.onnx", "inputs": {"features": HEAD_INPUT}, "outputs": {"audio": "waveform"}},
        "volvox": status, "cases": [head_case], "reference_audio": "reference.wav",
        "provenance": {"whole_model_manifest": "../manifest.json", "source_sha256": sha256(original_path),
            "specialized_sha256": sha256(lowered_path), "fixed_feature_shape": list(activation.shape),
            "learned_boundary": HEAD_INPUT, "normalization_specialization": "fixed-length window index/scatter becomes per-sample divisor; ORT bit-identical",
            "whole_model_max_absolute_error": float(np.max(np.abs(reference-audio))),
            "source_node_count": len(original.graph.node), "specialized_node_count": len(lowered.graph.node)}}
    (component / "manifest.json").write_text(json.dumps(head_manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(root / "manifest.json"), "whole_model": full_status["status"],
        "component_manifest": str(component / "manifest.json"), "component": status["status"],
        "samples": audio.size, "sample_rate": SAMPLE_RATE}, indent=2))


if __name__ == "__main__":
    from examples.kokoro.complete import main
    main()
