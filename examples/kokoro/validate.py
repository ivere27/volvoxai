#!/usr/bin/env python3
"""Measure complete Kokoro output and the original oracle's F32 sensitivity.

This diagnostic never supplies a model activation to Volvox inference. It runs
the original ONNX checkpoint solely as an independent comparison oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.kokoro.full_model import TEXT, PREDICTOR, DURATIONS, encoder, decoder


def spectrum(audio):
    """Phase-insensitive diagnostic, 1024-sample Hann frames and 256 hop."""
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    if samples.size < 1024:
        samples = np.pad(samples, (0, 1024 - samples.size))
    positions = np.arange(0, samples.size - 1024 + 1, 256)[:, None] + np.arange(1024)[None, :]
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(1024) / 1024)
    return np.abs(np.fft.rfft(samples[positions] * window, axis=-1))


def metrics(actual, reference):
    if actual.shape != reference.shape:
        return {"same_shape": False, "actual_shape": list(actual.shape), "reference_shape": list(reference.shape)}
    difference = actual.astype(np.float64) - reference
    error_power = float(np.sum(difference * difference))
    reference_power = float(np.sum(reference.astype(np.float64) ** 2))
    reference_spectrum = spectrum(reference)
    return {"same_shape": True, "max_absolute_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "snr_db": float(10 * np.log10(reference_power / error_power)) if error_power and reference_power else None,
        "spectral_magnitude_relative_l2": float(np.linalg.norm(spectrum(actual) - reference_spectrum) / np.linalg.norm(reference_spectrum)),
        "strict_rtol": 1e-4, "strict_atol": 1e-5,
        "strict_mismatch_count": int(np.count_nonzero(np.abs(difference) > 1e-5 + 1e-4 * np.abs(reference))),
        "strict_parity": bool(np.allclose(actual, reference, rtol=1e-4, atol=1e-5)),
        "bit_identical": bool(np.array_equal(actual, reference))}


def runtime(model, threads=4, optimized=True):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])


def validate(path, *, phase_sensitivity=False):
    root = path.parent
    manifest = json.loads(path.read_text())
    source = root / manifest["onnx"]["path"]
    case = manifest["cases"][0]
    with np.load(root / case["inputs"], allow_pickle=False) as archive:
        inputs = dict(archive)
    reference = runtime(str(source)).run(None, inputs)[0]
    actual = np.load(root / manifest["volvox_samples"], allow_pickle=False)
    result = {"format": "volvoxai-kokoro-oracle-stability/v1", "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider", "source_model_sha256": manifest["model"]["sha256"],
        "spectral_metric": {"frame": 1024, "hop": 256, "window": "periodic Hann", "zero_pad_short_audio": True},
        "volvox_vs_original": metrics(actual, reference),
        "oracle_disable_vs_enable_optimizations": metrics(runtime(str(source), optimized=False).run(None, inputs)[0], reference),
        "oracle_threads_one_vs_four": metrics(runtime(str(source), threads=1).run(None, inputs)[0], reference)}
    if phase_sensitivity:
        model = onnx.load(source)
        token_count = inputs["input_ids"].shape[1]
        stage_a = encoder(model, token_count, manifest["volvox"]["max_frames"])
        features = runtime(stage_a.SerializeToString()).run(None, inputs)
        frames = int(features[-1][-1])
        original_inputs = {TEXT: features[0], PREDICTOR: features[1], DURATIONS: features[2].astype(np.int64), "style": inputs["style"]}
        stage_b = decoder(model, token_count, frames)
        stage_b.graph.output.extend([
            helper.make_tensor_value_info("/decoder/decoder/generator/STFT_output_0", onnx.TensorProto.FLOAT, [1, frames * 120 + 1, 11, 2]),
            helper.make_tensor_value_info("/decoder/decoder/generator/Where_1_output_0", onnx.TensorProto.FLOAT, [1, 11, frames * 120 + 1]),
        ])
        decoder_oracle = runtime(stage_b.SerializeToString())
        baseline = decoder_oracle.run(None, original_inputs)
        perturbed_inputs = {name: np.nextafter(value, np.float32(np.inf)) if name in (TEXT, PREDICTOR) else value
            for name, value in original_inputs.items()}
        perturbed = decoder_oracle.run(None, perturbed_inputs)
        phase_difference = perturbed[2] - baseline[2]
        magnitude = np.linalg.norm(baseline[1], axis=-1)
        result["one_ulp_encoder_feature_perturbation"] = {"waveform": metrics(perturbed[0], baseline[0]),
            "perturbation": "each text and predictor F32 feature moves one representable value toward positive infinity; durations and style unchanged",
            "stft_magnitude_relative_l2": float(np.linalg.norm(np.linalg.norm(perturbed[1], axis=-1) - magnitude) / np.linalg.norm(magnitude)),
            "phase_branch_jumps_over_six_radians": int(np.count_nonzero(np.abs(phase_difference) > 6)),
            "dc_nyquist_branch_jumps_over_six_radians": int(np.count_nonzero(np.abs(phase_difference[:, [0, 10], :]) > 6)),
            "dc_nyquist_max_imaginary": float(np.max(np.abs(baseline[1][:, :, [0, 10], 1])))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("build/tts-examples/kokoro-complete/manifest.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phase-sensitivity", action="store_true")
    args = parser.parse_args()
    report = validate(args.manifest.resolve(), phase_sensitivity=args.phase_sensitivity)
    output = args.out or args.manifest.parent / "oracle-stability.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
