#!/usr/bin/env python3
"""Observe one raw-input-to-first-ID workflow without loading an HF/ORT oracle.

This is a single wall/RSS observation, not an idle or optimized latency claim.
Numerical qualification must already exist in the explicitly supplied receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from tools.llm_baseline.verify_multimodal_native import (Session, sha, tensors, attest, pinned_audio,
    prepare_whisper_features, prepare_receipt_image, stateless_execution, tensor_feedback_execution, pb)
from tools.exporter.hf_decoder import read_hf_checkpoint
from tools.exporter.hf_whisper import WhisperConfig, build_whisper_encoder_graph, build_whisper_decoder_graph


def memory():
    observation = {"process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
        (1 if sys.platform == "darwin" else 1024)}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split(); observation[key.rstrip(":") + "_bytes"] = int(value) * 1024
    return observation


def audio(args, manifest, correctness, directory):
    pin = pinned_audio(args.whisper_snapshot, manifest)
    expected = correctness["audio"]["expected_ids"][0]
    started = time.perf_counter_ns()
    clock = np.arange(24000, dtype=np.float32) / np.float32(16000)
    pcm = (.2 * np.sin(2 * np.pi * 220 * clock) + .05 * np.sin(2 * np.pi * 730 * clock)).astype(np.float32)
    features = prepare_whisper_features(pcm, manifest["audio"]["processor_config"])
    caller_prepared = time.perf_counter_ns()
    source, weights, _ = read_hf_checkpoint(args.whisper_snapshot)
    config = WhisperConfig.from_hf(source, max_context=16)
    encoder, ew = build_whisper_encoder_graph(config, weights)
    suppress = sorted(set(source["suppress_tokens"]) | set(range(50364, config.vocabulary)))
    decoder, dw = build_whisper_decoder_graph(config, weights, generate=True, suppressed_ids=suppress)
    del weights
    models_prepared = time.perf_counter_ns()
    session = Session(args.profile, directory, library_dir=args.library_dir, backend=args.backend)
    try:
        packages = {"encoder": session.model("encoder", encoder, ew), "decoder": session.model("decoder", decoder, dw)}
        ec = session.context("encoder")
        gc = session.context("decoder", generate=True, steps=3, eos_ids=[source["eos_token_id"]])
        compiled = time.perf_counter_ns()
        encoder_result = session.execute(ec, {"input_features": features}, encoder=True, owned=True)
        encoded = time.perf_counter_ns()
        prompt = correctness["audio"]["generation_policy"]["prompt_ids"]
        inputs = tensors({"token_ids": np.array([prompt], np.int32), "position_ids": np.arange(len(prompt), dtype=np.int32),
            "attention_mask": np.ones((1, len(prompt)), np.int32)})
        inputs.append(encoder_result.outputs[0])
        result = session.client.generate(pb.GenerateRequest(context_id=gc,
            initial_inputs=pb.GenerationInputs(values=inputs), budget=pb.GenerationBudget(max_steps=1)))
        observed = time.perf_counter_ns()
        session.routes.append(attest(result.report, args.backend))
        actual = session.read(result.outputs[0]).reshape(-1).tolist()
        assert actual == [expected]
        report = {"pin": pin, "first_ids": actual, "encoder_execute_calls": session.encoder_calls,
            "platform": session.platform, "packages": packages, "routes": session.routes, "owned_outputs": session.owned_inventory,
            "wall_ns": {"raw_input_to_first_result": observed - started,
                "caller_pcm_and_features": caller_prepared - started,
                "local_checkpoint_and_graph_preparation": models_prepared - caller_prepared,
                "runtime_load_compile_contexts": compiled - models_prepared,
                "first_encoder": encoded - compiled, "first_generate_call": observed - encoded},
            "memory_before_owner_release": memory()}
    finally: session.close()
    report["memory_after_owner_release"] = memory()
    return report


def image(args, manifest, correctness, directory):
    from examples.tiny_receipt_vqa.tools import verify_proto_backends as vqa
    reference = next(case for case in correctness["image"]["cases"] if case["jpeg"] == "receipt_en.jpg")
    source = ROOT / "examples/tiny_receipt_vqa/benchmarks/work/source"
    path = source / "examples/receipt_en.jpg"
    assert sha(path) == reference["jpeg_sha256"]
    package = args.vqa_package
    assert sha(package / "package_manifest.json") == correctness["image"]["package_manifest_sha256"]
    started = time.perf_counter_ns()
    pixels = prepare_receipt_image(path)
    ids = vqa.tokenizer(package).encode(vqa.clean(reference["question"]), add_eos=True, max_len=192)
    assert ids == reference["question_ids"]
    caller_prepared = time.perf_counter_ns()
    assets = json.loads((package / "package_manifest.json").read_text())["graphs"]
    graphs = {role: json.loads((package / assets[role]["graph"]["path"]).read_text()) for role in ("encoder", "decoder")}
    graphs["encoder"]["execution"] = stateless_execution(graphs["encoder"]["inputs"], graphs["encoder"]["outputs"])
    mapping = assets["decoder"]["inputs"]
    invariant = [mapping[name] for name in mapping if name.startswith("cross_") or name in ("family_ids", "memory_padding_mask")]
    feedback = {mapping[f"past_{kind}_{index}"]: assets["decoder"]["outputs"][f"present_{kind}_{index}"]
        for index in range(4) for kind in ("k", "v")}
    feedback[mapping["past_padding_mask"]] = assets["decoder"]["outputs"]["present_padding_mask"]
    graphs["decoder"]["execution"] = tensor_feedback_execution(graphs["decoder"]["inputs"],
        graphs["decoder"]["outputs"], feedback=feedback, position_input=mapping["position_ids"], invariant_inputs=invariant)
    models_prepared = time.perf_counter_ns()
    session = Session(args.profile, directory, library_dir=args.library_dir, backend=args.backend)
    try:
        packages = {}
        for role in ("encoder", "decoder"):
            graph_path = Path(directory) / f"{role}.graph.json"; graph_path.write_text(json.dumps(graphs[role], allow_nan=False))
            packages[role] = session.load(role, graph_path, package / assets[role]["weights"]["path"])
        ec, dc = session.context("encoder"), session.context("decoder")
        compiled = time.perf_counter_ns()
        encoder_inputs = {assets["encoder"]["inputs"][name]: value for name, value in vqa.encoder_feed(pixels, ids).items()}
        result = session.execute(ec, encoder_inputs, encoder=True, owned=True)
        encoded = time.perf_counter_ns()
        outputs = {value.name: value for value in result.outputs}
        conditioning = {name: session.read(outputs[value]) for name, value in assets["encoder"]["outputs"].items()}
        inputs = {mapping[name]: value for name, value in vqa.decoder_feed(conditioning).items()}
        result = session.execute(dc, inputs, entrypoint="initialize")
        observed = time.perf_counter_ns()
        logits = session.read(next(value for value in result.outputs if value.name == "logits"))
        actual = int(logits.argmax(-1)[0, 0]); assert actual == reference["expected_ids"][0]
        report = {"jpeg_sha256": reference["jpeg_sha256"], "first_ids": [actual], "encoder_execute_calls": session.encoder_calls,
            "platform": session.platform, "packages": packages, "routes": session.routes, "owned_outputs": session.owned_inventory,
            "wall_ns": {"raw_input_to_first_result": observed - started, "caller_jpeg_and_question": caller_prepared - started,
                "graph_metadata_preparation": models_prepared - caller_prepared, "runtime_load_compile_contexts": compiled - models_prepared,
                "first_encoder": encoded - compiled, "first_decoder_call_including_conditioning_readback": observed - encoded},
            "memory_before_owner_release": memory()}
    finally: session.close()
    report["memory_after_owner_release"] = memory()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=("audio", "image"), required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("inference", "full"), default="inference")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl", "metal"), default="cpu")
    parser.add_argument("--library-dir", type=Path, default=ROOT / "native")
    parser.add_argument("--whisper-snapshot", type=Path, default=Path("/tmp/volvoxai-h-stage-whisper-169d4a"))
    parser.add_argument("--vqa-package", type=Path, default=ROOT / "build/receipt-vqa-ptq-head-29ffb7e/fp32")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    if args.backend != "cpu" and args.profile != "full": parser.error("native GPU requires --profile full")
    correctness = json.loads(args.correctness.read_text()); assert correctness["status"] == "PASS"
    manifest = json.loads((ROOT / "tools/llm_baseline/multimodal_sensor.json").read_text())
    receipt = {"format": "volvox-multimodal-first-result/v1", "status": "RUNNING", "workflow": args.workflow,
        "profile": args.profile, "backend": args.backend, "library_sha256": sha(args.library_dir /
            ("libvolvoxai-lite.so" if args.profile == "inference" else "libvolvoxai.so")),
        "harness_sha256": sha(__file__), "correctness_receipt_sha256": sha(args.correctness),
        "scope": "one caller raw-input-to-first-ID wall observation, no HF/ORT capture, no idle qualification",
        "memory_scope": "whole Python process lifetime VmHWM/RSS plus explicit retained OWNED output inventory",
        "memory_at_workflow_entry": memory()}
    try:
        with tempfile.TemporaryDirectory(prefix="volvox-h-observe-", dir=args.scratch) as directory:
            receipt["observation"] = (audio if args.workflow == "audio" else image)(args, manifest, correctness, directory)
        assert "torch" not in sys.modules and "onnxruntime" not in sys.modules
        receipt["reference_capture_disabled"] = True
        receipt["status"] = "PASS"
    except BaseException as error:
        receipt["status"] = "FAIL"; receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__": main()
