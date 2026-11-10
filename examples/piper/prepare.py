#!/usr/bin/env python3
"""Prepare a complete Piper pipeline executed entirely by VolvoxAI."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tts_common.artifacts import sha256, write_case
from examples.piper.validation import compare_waveform, WAVEFORM_CRITERIA
from tools.export_safetensors import export_onnx_model
from examples.piper.lowering import (DURATION_NOISE, ACOUSTIC_NOISE, DURATION_OUTPUTS,
    duration_model, acoustic_model, externalize_noise)

REVISION = "375a0fe641dea077c2a47b4e9a056d6da521eed3"
VOICE = "en_US-lessac-low"
BASE_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/{REVISION}/en/en_US/lessac/low"
FILES = {
    f"{VOICE}.onnx": "f7d01dde371555732c4c314111ac79672b1a5ce2fc19266ab42178fd8df7f375",
    f"{VOICE}.onnx.json": "45754dfdebb3b8661c3fc564713772deec6e064feeb5b4e9594857dc7305193a",
    "MODEL_CARD": "f4f5b5ab354ab71dd2c71b571bcd0879b259ddfabaa5169b968a60427d0b73f3",
}
SOURCE_INPUTS = {"tokens": "input", "lengths": "input_lengths", "scales": "scales"}


def ensure_assets(directory: Path) -> Path:
    """Reuse verified files; publish downloads only after checksum validation."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = directory / name
        if destination.exists():
            if sha256(destination) != expected:
                raise ValueError(f"checksum mismatch: {destination}")
            continue
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                stream.flush()
                if sha256(temporary) != expected:
                    raise ValueError(f"download checksum mismatch for {name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return directory / f"{VOICE}.onnx"


def prepare_inputs(text: str, config: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    from piper_phonemize import phonemize_espeak

    sentences = phonemize_espeak(text, config["espeak"]["voice"])
    if len(sentences) != 1:
        raise ValueError("prepare one sentence at a time; each sentence is a separate Piper request")
    phonemes = sentences[0]
    mapping = config["phoneme_id_map"]
    ids = [*mapping["^"], *mapping["_"]]
    for phoneme in phonemes:
        if phoneme not in mapping:
            raise ValueError(f"voice vocabulary has no phoneme {phoneme!r}")
        ids.extend(mapping[phoneme])
        ids.extend(mapping["_"])
    ids.extend(mapping["$"])
    return {
        "tokens": np.asarray([ids], dtype=np.int64),
        "lengths": np.asarray([len(ids)], dtype=np.int64),
        "scales": np.asarray([config["inference"]["noise_scale"],
                              config["inference"]["length_scale"],
                              config["inference"]["noise_w"]], dtype=np.float32),
    }, phonemes


def session(model: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_stage(root, model, inputs, outputs):
    root.mkdir(parents=True, exist_ok=True)
    onnx.save(model, root / "model.onnx")
    report = {}
    graph = export_onnx_model(str(root / "model.onnx"), str(root / "package/model.safetensors"),
        input_dtypes={value.name: "int32" for value in model.graph.input
                      if value.type.tensor_type.elem_type == onnx.TensorProto.INT64},
        report_callback=report.update)
    write_json(root / "export-report.json", report)
    mapping = dict(zip((value.name for value in model.graph.input), graph["inputs"]))
    return {"status": "ready", "graph": "package/graph.json", "weights": ["package/model.safetensors"],
        "inputs": {semantic: mapping[name] for semantic, name in inputs.items()},
        "outputs": dict(zip(outputs, graph["outputs"]))}


def execute_stage(root, contract, inputs, args):
    from examples.tts_common.benchmark_native import Native
    engine = Native(root, contract, "cpu", args.threads, args.library, 120)
    try:
        engine.prepare(inputs)
        result, timing = engine.run()
        return result, {"route": engine.route, "timing": timing, "setup_ms": engine.setup_ms,
                        "metadata": engine.metadata}
    finally:
        engine.close()


def prepare(args) -> Path:
    if args.python_deps.is_dir():
        sys.path.insert(0, str(args.python_deps.resolve()))
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output directory is not empty; choose a new --out: {args.out}")
    source = ensure_assets(args.assets)
    config = json.loads((args.assets / f"{VOICE}.onnx.json").read_text())
    inputs, phonemes = prepare_inputs(args.text, config)
    tokens = inputs["tokens"].shape[1]
    if not 1 <= tokens <= args.max_tokens:
        raise ValueError(f"{tokens} phoneme IDs exceed admitted token limit {args.max_tokens}")
    rng = np.random.default_rng(args.seed)
    inputs["duration_noise"] = rng.standard_normal((1, 2, tokens)).astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    rate = int(config["audio"]["sample_rate"])
    duration_inputs = {**SOURCE_INPUTS, "duration_noise": DURATION_NOISE}
    duration_root = args.out / "duration"
    duration = export_stage(duration_root, duration_model(source, tokens), duration_inputs, DURATION_OUTPUTS)
    duration_outputs, duration_evidence = execute_stage(duration_root, duration, inputs, args)
    frames = int(duration_outputs["frames"].item())
    if not 1 <= frames <= args.max_frames:
        raise ValueError(f"Volvox duration stage returned {frames} frames; admitted range is 1..{args.max_frames}")
    inputs["acoustic_noise"] = rng.standard_normal((1, 192, frames)).astype(np.float32)
    acoustic_feed = {**duration_outputs, "scales": inputs["scales"], "acoustic_noise": inputs["acoustic_noise"]}
    acoustic_inputs = {**DURATION_OUTPUTS, "scales": "scales", "acoustic_noise": ACOUSTIC_NOISE}
    acoustic_root = args.out / "acoustic"
    acoustic = export_stage(acoustic_root, acoustic_model(source, tokens, frames), acoustic_inputs, {"audio": "output"})
    result, acoustic_evidence = execute_stage(acoustic_root, acoustic, acoustic_feed, args)

    # ONNX Runtime is used only after both numerical stages have run in Volvox.
    # The reference differs from the checkpoint solely by making its two IID
    # Gaussian draws explicit, so the engines compare the same request noise.
    reference = externalize_noise(onnx.load(source), tokens, frames)
    reference_path = args.out / "reference.onnx"
    onnx.save(reference, reference_path)
    full_inputs = {**SOURCE_INPUTS, "duration_noise": DURATION_NOISE, "acoustic_noise": ACOUSTIC_NOISE}
    expected = session(reference_path, args.threads).run(["output"],
        {full_inputs[name]: value for name, value in inputs.items()})[0]
    waveform_comparison = compare_waveform(result["audio"], expected)
    stage_reference_values = session(duration_root / "model.onnx", args.threads).run(
        list(DURATION_OUTPUTS.values()), {duration_inputs[name]: value for name, value in inputs.items()
                                         if name in duration_inputs})
    stage_reference = dict(zip(DURATION_OUTPUTS, stage_reference_values))
    for name, value in duration_outputs.items():
        if name in {"durations", "frames", "mask"}:
            np.testing.assert_array_equal(value, stage_reference[name])
        else:
            np.testing.assert_allclose(value, stage_reference[name], rtol=1e-4, atol=1e-5)
    acoustic_reference = session(acoustic_root / "model.onnx", args.threads).run(["output"],
        {acoustic_inputs[name]: value for name, value in acoustic_feed.items()})[0]
    np.testing.assert_allclose(result["audio"], acoustic_reference, rtol=1e-4, atol=1e-5)
    info = {"name": VOICE, "family": "piper", "revision": REVISION,
        "source_url": f"{BASE_URL}/{VOICE}.onnx", "sha256": FILES[f"{VOICE}.onnx"],
        "model_card_url": f"{BASE_URL}/MODEL_CARD",
        "dataset_license_url": "https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html"}
    provenance = {"numpy": np.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "onnxsim": importlib.metadata.version("onnxsim"),
        "piper_phonemize": importlib.metadata.version("piper-phonemize"),
        "phonemization_voice": config["espeak"]["voice"], "scales": inputs["scales"].tolist(),
        "noise_seed": args.seed, "threads": args.threads, "shape_source": "VolvoxAI duration stage",
        "source_operator_counts": dict(Counter(node.op_type for node in onnx.load(source).graph.node)),
        "lowering": "dense independent rational-quadratic spline lanes; constant shape/index folding; explicit Gaussian inputs"}
    validation = {"deterministic": True, "reference_engine": "onnxruntime",
        "reason": "both Gaussian draws are explicit request inputs with default voice scales"}
    case = write_case(args.out, "speech", inputs, {"audio": expected}, sample_rate=rate)
    case.update(text=args.text, phonemes=phonemes, validation={**validation, "waveform": WAVEFORM_CRITERIA})
    manifest = {"format": "volvoxai-tts-example/v1", "scope": "full-model", "model": info,
        "sample_rate": rate, "onnx": {"path": "reference.onnx", "inputs": full_inputs, "outputs": {"audio": "output"}},
        "source_onnx": relative(source, args.out), "cases": [case], "provenance": provenance,
        "volvox": {"status": "ready", "pipeline": [
            {"name": "duration", "manifest": "duration/manifest.json"},
            {"name": "acoustic", "manifest": "acoustic/manifest.json"}],
            "tokens": tokens, "frames": frames,
            "shape_feedback": {"stage": "duration", "output": "frames", "index": 0, "expected": frames},
            "execution": "all learned and numerical inference runs in VolvoxAI; host reads frame count to select shape"}}
    for name, root, contract, mapping, stage_feed, stage_outputs in (
            ("duration", duration_root, duration, duration_inputs,
             {key: value for key, value in inputs.items() if key != "acoustic_noise"}, stage_reference),
            ("acoustic", acoustic_root, acoustic, acoustic_inputs, acoustic_feed, {"audio": acoustic_reference})):
        output_mapping = DURATION_OUTPUTS if name == "duration" else {"audio": "output"}
        stage_case = write_case(root, "speech", stage_feed, stage_outputs,
                               audio_output="frames" if name == "duration" else "audio", sample_rate=rate)
        stage_case.update(text=args.text, validation=validation)
        stage = {"format": "volvoxai-tts-example/v1", "scope": f"full-model-{name}-stage", "model": info,
            "sample_rate": rate, "onnx": {"path": "model.onnx", "inputs": mapping, "outputs": output_mapping},
            "volvox": contract, "cases": [stage_case], "provenance": provenance}
        write_json(root / "manifest.json", stage)
    write_json(args.out / "preparation-report.json", {"duration": duration_evidence, "acoustic": acoustic_evidence,
        "tokens": tokens, "frames": frames, "comparison": waveform_comparison,
        "stage_validation": {"rtol": 1e-4, "atol": 1e-5, "duration_pass": True, "acoustic_pass": True}})
    write_json(args.out / "manifest.json", manifest)
    return args.out / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Hello world.")
    parser.add_argument("--assets", type=Path, default=ROOT / "build/tts-examples/assets/piper-lessac-low")
    parser.add_argument("--out", type=Path, default=ROOT / "build/tts-examples/piper-full")
    parser.add_argument("--python-deps", type=Path, default=ROOT / "build/tts-examples/python")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    if args.threads <= 0 or args.max_tokens <= 0 or args.max_frames <= 0:
        parser.error("threads and shape limits must be positive")
    print(prepare(args))


if __name__ == "__main__":
    main()
