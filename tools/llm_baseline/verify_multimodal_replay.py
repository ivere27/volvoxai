#!/usr/bin/env python3
"""Replay prepared independent H references through a required native backend.

The candidate process requires NumPy; raw JPEG additionally requires Pillow.
It imports neither HF nor ORT and never changes a stored numerical budget.
"""
from __future__ import annotations

from google.protobuf.json_format import MessageToDict

import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from tools.llm_baseline.verify_multimodal_native import (Session, sha, array_hash, compare, tensors, attest,
    prepare_receipt_image, prepare_whisper_features, pb)


def audio(session, bundle, corpus, arrays, manifest):
    policy = corpus["audio"]
    features = prepare_whisper_features(arrays["audio_pcm"].copy(), manifest["audio"]["processor_config"])
    assert array_hash(arrays["audio_pcm"]) == policy["pcm_sha256"]
    checks = {"features": compare(features, arrays["audio_features"], **policy["tolerances"]["features"])}
    ec = session.context("audio_encoder")
    encoded = session.execute(ec, {"input_features": features}, encoder=True, owned=True)
    memory_tensor = encoded.outputs[0]
    memory = session.read(memory_tensor); snapshot = array_hash(memory)
    checks["encoder"] = compare(memory, arrays["audio_memory"], **policy["tolerances"]["encoder"])
    dc = session.context("audio_decoder")
    prefix = list(policy["prompt_ids"]); checks["logits"] = []
    for index, token in enumerate(policy["expected_ids"]):
        packet = session.execute(dc, {"token_ids": np.array([prefix], np.int32),
            "position_ids": np.arange(len(prefix), dtype=np.int32), "attention_mask": np.ones((1, len(prefix)), np.int32),
            "encoder_memory": memory})
        logits = session.read(next(value for value in packet.outputs if value.name == "logits"))
        checks["logits"].append(compare(logits, arrays[f"audio_logits_{index}"], **policy["tolerances"]["logits"]))
        prefix.append(token)
    gc = session.context("audio_decoder", generate=True, steps=3, eos_ids=[policy["eos_id"]])
    initial = tensors({"token_ids": np.array([policy["prompt_ids"]], np.int32),
        "position_ids": np.arange(len(policy["prompt_ids"]), dtype=np.int32),
        "attention_mask": np.ones((1, len(policy["prompt_ids"])), np.int32)})
    # FullPrefixGenerationContract requires host-readable initial bindings.
    # read() above explicitly stages the device OWNED snapshot via CopyTensors;
    # Generate retains its own HOST snapshot, and model kernels still run on
    # the required GPU backend. This is not a device-resident invariant lease.
    if session.backend == "cpu":
        initial.append(memory_tensor)
    else:
        initial.extend(tensors({"encoder_memory": memory}, borrowed=True))
    ids, retained = [], []
    for index, budget in enumerate((1, 2)):
        packet = session.client.generate(pb.GenerateRequest(context_id=gc,
            initial_inputs=pb.GenerationInputs(values=initial) if not index else None,
            budget=pb.GenerationBudget(max_steps=budget)))
        session.routes.append(attest(packet.report, session.backend)); retained.extend(packet.outputs)
        ids.extend(session.read(packet.outputs[0]).reshape(-1).tolist()); memory[:] = 0
        if not packet.progress.can_continue: break
    assert ids == policy["expected_ids"]
    assert array_hash(session.read(memory_tensor)) == snapshot and session.encoder_calls == 1
    return {"kind": "raw-PCM-whisper", "checks": checks, "generated_ids": ids, "encoder_execute_calls": 1,
        "owned_conditioning_snapshot": True,
        "conditioning_handoff": "HOST OWNED binding" if session.backend == "cpu" else
            "public CopyTensors device OWNED to caller HOST; borrowed HOST snapshot accepted by Generate; upload to required GPU",
        "conditioning_device_to_host_bytes": 0 if session.backend == "cpu" else int(arrays["audio_memory"].nbytes),
        "borrowed_host_conditioning_mutation_after_acceptance": session.backend != "cpu"}, retained


def image(session, bundle, corpus, arrays):
    from examples.tiny_receipt_vqa.tools import verify_proto_backends as vqa
    contracts = corpus["image"]["contracts"]
    mapping = contracts["decoder"]["inputs"]
    invariant = [name for name in mapping if name.startswith("cross_") or name in ("family_ids", "memory_padding_mask")]
    results, retained = [], []
    for case in corpus["image"]["cases"]:
        assert sha(bundle / case["jpeg"]) == case["jpeg_sha256"]
        pixels = prepare_receipt_image(bundle / case["jpeg"])
        assert array_hash(pixels) == case["pixels_sha256"]
        ec, dc = session.context("image_encoder"), session.context("image_decoder")
        before = session.encoder_calls
        packet = session.execute(ec, {contracts["encoder"]["inputs"][name]: value
            for name, value in vqa.encoder_feed(pixels, case["question_ids"]).items()}, encoder=True, owned=True)
        outputs = {value.name: value for value in packet.outputs}
        encoded = {name: session.read(outputs[tensor]) for name, tensor in contracts["encoder"]["outputs"].items()}
        snapshots = {name: array_hash(value) for name, value in encoded.items()}
        checks = {}
        for name, value in encoded.items():
            oracle = arrays[f"{case['key']}_encoder_{name}"]
            if value.dtype.kind == "f": checks[name] = compare(value, oracle, **corpus["image"]["tolerances"])
            else: np.testing.assert_array_equal(value, oracle)
        pixels[:] = 0; feed = vqa.decoder_feed(encoded, 1); ids, logits_checks = [], []
        for index, expected in enumerate(case["expected_ids"]):
            packet = session.execute(dc, {mapping[name]: value for name, value in feed.items() if not index or name not in invariant},
                entrypoint="transition" if index else "initialize", reuse=[mapping[name] for name in invariant] if index else [])
            retained.extend(packet.outputs)
            observed = {value.name: session.read(value) for value in packet.outputs}
            actual = {name: observed[tensor] for name, tensor in contracts["decoder"]["outputs"].items()}
            logits_checks.append(compare(actual["logits"], arrays[f"{case['key']}_logits_{index}"], **corpus["image"]["tolerances"]))
            token = int(actual["logits"].argmax(-1)[0, 0]); assert token == expected; ids.append(token)
            vqa.advance(feed, actual, token, index)
            if not index:
                for name in invariant: feed[name][:] = 0
        assert session.encoder_calls - before == 1
        for name, tensor in contracts["encoder"]["outputs"].items(): assert array_hash(session.read(outputs[tensor])) == snapshots[name]
        results.append({"kind": "raw-JPEG-vqa", "jpeg": case["jpeg"], "encoder": checks, "logits": logits_checks,
            "generated_ids": ids, "encoder_execute_calls": 1, "conditioning_snapshot_and_reuse": True})
    return results, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, default=ROOT / "native")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl", "metal"), default="cpu")
    parser.add_argument("--profiles", nargs="+", choices=("inference", "full"), default=("inference", "full"))
    parser.add_argument("--workflows", nargs="+", choices=("audio", "image"), default=("audio", "image"))
    args = parser.parse_args()
    if args.backend != "cpu" and args.profiles != ["full"]: parser.error("native GPU requires --profiles full")
    corpus = json.loads((args.bundle / "corpus.json").read_text())
    assert corpus["status"] == "REFERENCE_PREPARED" and sha(args.bundle / "references.npz") == corpus["references_sha256"]
    arrays = np.load(args.bundle / "references.npz", allow_pickle=False)
    manifest = json.loads((ROOT / "tools/llm_baseline/multimodal_sensor.json").read_text())
    assert sha(ROOT / "tools/llm_baseline/multimodal_sensor.json") == corpus["manifest_sha256"]
    receipt = {"format": "volvox-multimodal-native-replay/v1", "status": "RUNNING", "backend": args.backend,
        "harness_sha256": sha(__file__), "corpus_sha256": sha(args.bundle / "corpus.json"), "results": [],
        "scope": "actual selected raw PCM/JPEG; independently prepared HF/ORT oracles; unchanged budgets; no candidate oracle engines"}
    try:
        for profile in args.profiles:
            with tempfile.TemporaryDirectory(prefix="volvox-h-replay-") as directory:
                session = Session(profile, directory, library_dir=args.library_dir, backend=args.backend)
                retained = []
                try:
                    for role, pin in corpus["packages"].items():
                        if role.split("_", 1)[0] not in args.workflows: continue
                        package = args.bundle / role
                        assert sha(package / "graph.json") == pin["graph_sha256"]
                        assert sha(package / "model.safetensors") == pin["weights_sha256"]
                        session.load(role, package / "graph.json", package / "model.safetensors")
                    results = []
                    if "audio" in args.workflows:
                        result, values = audio(session, args.bundle, corpus, arrays, manifest); results.append(result); retained.extend(values)
                    if "image" in args.workflows:
                        result, values = image(session, args.bundle, corpus, arrays); results.extend(result); retained.extend(values)
                    snapshots = [value.inline for value in retained]
                    record = {"profile": profile, "platform": session.platform, "library_sha256": sha(args.library_dir /
                        ("libvolvoxai-lite.so" if profile == "inference" else "libvolvoxai.so")),
                        "results": results, "routes": session.routes, "owned_outputs": session.owned_inventory}
                finally: session.close()
                assert snapshots == [value.inline for value in retained]
                record["inline_survives_all_owner_retirement"] = True; receipt["results"].append(record)
        assert "torch" not in sys.modules and "onnxruntime" not in sys.modules
        receipt["candidate_oracle_engines_absent"] = True; receipt["status"] = "PASS"
    except BaseException as error:
        receipt["status"] = "FAIL"; receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        if getattr(error, "report", None) is not None: receipt["error"]["report"] = MessageToDict(error.report, preserving_proto_field_name=True)
        raise
    finally:
        arrays.close()
        receipt["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__": main()
