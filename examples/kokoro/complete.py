#!/usr/bin/env python3
"""Execute the complete pretrained Kokoro model through VolvoxAI stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.kokoro.fetch import FILES, REPOSITORY, REVISION, fetch
from examples.kokoro.full_model import DEFAULT_MAX_FRAMES, TEXT, PREDICTOR, DURATIONS, encoder, fold_static_programs, lower_decoder
from examples.kokoro.prepare import SAMPLE_RATE, phoneme_inputs, session, write_wave
from examples.tts_common.artifacts import sha256, write_case
from examples.tts_common.benchmark_native import Native


def export_stage(model, root, inputs, outputs, *, input_dtypes=None, output_dtypes=None):
    from tools.export_safetensors import export_onnx_model
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source.onnx"
    onnx.save(model, path)
    graph = export_onnx_model(str(path), str(root / "package/model.safetensors"), weight_dtype="float32",
        output_names=list(outputs), input_dtypes=input_dtypes, output_dtypes=output_dtypes)
    contract = {"status": "ready", "graph": "package/graph.json", "weights": ["package/model.safetensors"],
        "inputs": {semantic: f"input{index}" for index, semantic in enumerate(inputs)},
        "outputs": {semantic: semantic for semantic in outputs}}
    return contract, len(graph["nodes"])


def run_stage(root, contract, inputs, args):
    runner = Native(root, contract, args.backend, args.threads, args.library, args.timeout)
    try:
        runner.prepare(inputs)
        outputs, timings = runner.run()
        return outputs, {"runtime": runner.metadata, "setup_ms": runner.setup_ms,
            "execution": timings, "route": runner.route}
    finally:
        runner.close()


def run(args):
    assets, root = args.assets.resolve(), args.output.resolve()
    if (root / "manifest.json").exists() or (root / "cases").exists():
        raise RuntimeError(f"Prepared request already exists: {root}; use a fresh --output")
    fetch(assets)
    root.mkdir(parents=True, exist_ok=True)
    inputs = phoneme_inputs(assets, args.phonemes, args.speed)
    tokens = inputs["input_ids"].shape[1]
    model_path = assets / "onnx/model.onnx"
    model = onnx.load(model_path)
    source_a = fold_static_programs(encoder(model, tokens, args.max_frames))
    encoder_names = {"text_features": TEXT, "predictor_features": PREDICTOR, "durations": DURATIONS}
    contract_a, nodes_a = export_stage(source_a, root / "encoder", inputs, encoder_names,
        input_dtypes={"input_ids": "int32"}, output_dtypes={"durations": "int32"})
    # This is the first computation of features and predicted durations.
    # ONNX Runtime is not invoked to choose a shape or supply an activation.
    values, evidence_a = run_stage(root / "encoder", contract_a, inputs, args)
    durations = np.asarray(values["durations"])
    if durations.dtype != np.int32 or durations.shape != (tokens,) or np.any(np.diff(durations.astype(np.int64), prepend=0) <= 0):
        raise RuntimeError("Volvox encoder returned invalid cumulative durations")
    frames = int(durations[-1])
    if frames > args.max_frames:
        raise RuntimeError(f"Predicted duration {frames} exceeds --max-frames {args.max_frames}")
    source_b, lowering = lower_decoder(model, tokens, frames)
    decoder_names = {"text_features": TEXT, "predictor_features": PREDICTOR, "durations": DURATIONS, "style": "style"}
    contract_b, nodes_b = export_stage(source_b, root / "decoder", decoder_names, {"audio": "waveform"},
        input_dtypes={DURATIONS: "int32"})
    decoder_inputs = {**values, "style": inputs["style"]}
    outputs, evidence_b = run_stage(root / "decoder", contract_b, decoder_inputs, args)
    audio = outputs["audio"]
    if audio.dtype != np.float32 or audio.shape != (1, frames * 600) or not np.isfinite(audio).all():
        raise RuntimeError("Volvox decoder returned an invalid waveform")
    write_wave(root / "volvox.wav", audio)
    np.save(root / "volvox.npy", audio, allow_pickle=False)
    validation = {"oracle": "not_run", "numerical_parity": "not_claimed"}
    reference = audio
    if args.validate:
        # The original, unmodified source graph is solely an independent
        # oracle. Its outputs do not participate in either Volvox stage.
        reference = session(str(model_path), args.threads).run(None, inputs)[0]
        validation = {"oracle": "onnxruntime", "source_model_sha256": FILES["onnx/model.onnx"],
            "same_shape": audio.shape == reference.shape, "rtol": args.rtol, "atol": args.atol}
        if audio.shape == reference.shape:
            difference = audio.astype(np.float64) - reference
            validation.update(max_absolute_error=float(np.max(np.abs(difference))),
                rmse=float(np.sqrt(np.mean(difference * difference))),
                snr_db=float(10 * np.log10(np.sum(reference.astype(np.float64) ** 2) / np.sum(difference * difference))) if np.any(difference) else None,
                strict_mismatch_count=int(np.count_nonzero(np.abs(difference) > args.atol + args.rtol * np.abs(reference))),
                numerical_parity=bool(np.allclose(audio, reference, rtol=args.rtol, atol=args.atol)))
        else:
            validation["numerical_parity"] = False
        write_wave(root / "reference.wav", reference)
    model_info = {"name": "Kokoro-82M-v1.0 FP32", "family": "kokoro", "revision": REVISION,
        "source_url": f"https://huggingface.co/{REPOSITORY}/tree/{REVISION}",
        "sha256": FILES["onnx/model.onnx"], "license": "Apache-2.0", "voice": "af_heart", "precision": "float32"}
    case = write_case(root, "request", inputs, {"audio": reference}, sample_rate=SAMPLE_RATE)
    case.update(text=args.text, phonemes=args.phonemes,
        validation={"deterministic": True, "reason": "The pinned source graph has no stochastic nodes.",
            "reference_engine": "onnxruntime" if args.validate else "volvoxai"})
    for name, contract, source_inputs, source_outputs, nodes, execution in [
        ("encoder", contract_a, {key: key for key in inputs}, encoder_names, nodes_a, evidence_a),
        ("decoder", contract_b, decoder_names, {"audio": "waveform"}, nodes_b, evidence_b),
    ]:
        stage = {"format": "volvoxai-tts-example/v1", "model": model_info,
            "scope": f"complete-model {name} stage; all numerical operations execute on VolvoxAI",
            "sample_rate": SAMPLE_RATE, "onnx": {"path": "source.onnx", "inputs": source_inputs, "outputs": source_outputs},
            "volvox": contract, "cases": [{"id": "request"}],
            "provenance": {"nodes": nodes, "execution": execution, "source_sha256": sha256(root / name / "source.onnx")}}
        (root / name / "manifest.json").write_text(json.dumps(stage, indent=2) + "\n")
    manifest = {"format": "volvoxai-tts-example/v1", "model": model_info,
        "scope": "complete pretrained phonemes-to-waveform model; Volvox encoder and decoder; host reads duration extent only",
        "sample_rate": SAMPLE_RATE,
        "onnx": {"path": str(Path(os.path.relpath(model_path, root))),
            "inputs": {key: key for key in inputs}, "outputs": {"audio": "waveform"}},
        "volvox": {"status": "ready", "pipeline": [
            {"name": "encoder", "manifest": "encoder/manifest.json"},
            {"name": "decoder", "manifest": "decoder/manifest.json"}], "tokens": tokens, "frames": frames, "max_frames": args.max_frames,
            "shape_feedback": {"stage": "encoder", "output": "durations", "index": -1, "expected": frames}},
        "cases": [case], "volvox_audio": "volvox.wav", "volvox_samples": "volvox.npy",
        "reference_audio": "reference.wav" if args.validate else None,
        "provenance": {"lowering": lowering, "validation": validation,
            "execution": {"encoder": evidence_a, "decoder": evidence_b},
            "captured_activations": False, "oracle_in_execution_path": False}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(root / "manifest.json"), "samples": audio.size,
        "frames": frames, "validation": validation}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("build/tts-examples/assets/kokoro"))
    parser.add_argument("--output", type=Path, default=Path("build/tts-examples/kokoro-complete"))
    parser.add_argument("--text", default="Hello world!")
    parser.add_argument("--phonemes", default="həlˈoʊ wˈɜːld!")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="Reject longer duration predictions; bounded I32 legalization preserves admitted requests")
    parser.add_argument("--backend", choices=["cpu", "cuda", "vulkan", "opengl", "metal"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--validate", action="store_true", help="Run the original ONNX model only as a post-execution oracle")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    manifest = run(args)
    if args.validate and not manifest["provenance"]["validation"]["numerical_parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
