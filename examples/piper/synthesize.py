#!/usr/bin/env python3
"""Synthesize Piper speech with both numerical stages on VolvoxAI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tts_common.benchmark_native import Native
from examples.piper.validation import compare_waveform


class Pipeline:
    """Retain both compiled stages and pass every activation as a runtime input."""
    def __init__(self, manifest_path, backend="cpu", threads=1, library=None):
        self.root = Path(manifest_path).resolve().parent
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.stages = []
        try:
            for entry in self.manifest["volvox"]["pipeline"]:
                path = self.root / entry["manifest"]
                stage = json.loads(path.read_text())
                engine = Native(path.parent, stage["volvox"], backend, threads, library, 120)
                self.stages.append((entry["name"], stage["volvox"], engine))
        except BaseException:
            self.close()
            raise

    def run(self, inputs):
        values = dict(inputs)
        evidence = {}
        started = time.perf_counter_ns()
        for name, contract, engine in self.stages:
            binding = time.perf_counter_ns()
            engine.prepare({key: values[key] for key in contract["inputs"]})
            binding_ms = (time.perf_counter_ns() - binding) / 1e6
            outputs, timing = engine.run()
            values.update(outputs)
            evidence[name] = {"input_binding_ms": binding_ms, **timing, "route": engine.route}
            if name == "duration":
                frames = int(outputs["frames"].item())
                if frames != self.manifest["volvox"]["frames"]:
                    raise ValueError(f"request produces {frames} frames; package admits "
                        f"{self.manifest['volvox']['frames']}. Prepare a new package for this text/noise seed.")
        return {"audio": values["audio"]}, {"request_ms": (time.perf_counter_ns() - started) / 1e6,
                                             "stages": evidence}

    def close(self):
        for _, _, engine in reversed(self.stages):
            engine.close()
        self.stages.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "build/tts-examples/piper-full/manifest.json")
    parser.add_argument("--backend", choices=["cpu", "cuda", "vulkan", "opengl", "metal"], default="cpu")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "build/tts-examples/piper-full/speech.wav")
    parser.add_argument("--verify", action="store_true", help="compare final audio with whole-source ONNX reference")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    with np.load(root / manifest["cases"][0]["inputs"], allow_pickle=False) as data:
        inputs = {name: data[name].copy() for name in data.files}
    pipeline = Pipeline(args.manifest, args.backend, args.threads, args.library)
    try:
        outputs, timing = pipeline.run(inputs)
        audio = outputs["audio"]
        if any(extent != 1 for extent in audio.shape[:-1]) or not np.isfinite(audio).all():
            raise ValueError("expected finite mono PCM output")
        report = {"scope": "full-model", "execution": manifest["volvox"]["execution"],
            "text": manifest["cases"][0]["text"], "sample_rate": manifest["sample_rate"],
            "samples": audio.size, "model_sha256": manifest["model"]["sha256"], **timing,
            "timing_scope": "both Volvox numerical stages including host copy/binding and shape check; excludes setup and verification"}
        if args.verify:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = args.threads
            options.inter_op_num_threads = 1
            reference = ort.InferenceSession(str(root / manifest["onnx"]["path"]), options,
                                            providers=["CPUExecutionProvider"])
            expected = reference.run([manifest["onnx"]["outputs"]["audio"]],
                {manifest["onnx"]["inputs"][name]: value for name, value in inputs.items()})[0]
            report["comparison"] = compare_waveform(audio, expected,
                manifest["cases"][0]["validation"]["waveform"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.rint(np.clip(audio.reshape(-1), -1, 1) * 32767).astype("<i2")
        with wave.open(str(args.out), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(manifest["sample_rate"])
            output.writeframes(pcm.tobytes())
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(args.out)
        print(json.dumps(report, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
