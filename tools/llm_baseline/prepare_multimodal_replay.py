#!/usr/bin/env python3
"""Prepare pinned independent media/model oracles before a remote candidate run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from tools.llm_baseline.verify_multimodal_native import (sha, array_hash, pinned_audio, pinned_image,
    prepare_receipt_image, stateless_execution, tensor_feedback_execution)
from tools.exporter.hf_decoder import read_hf_checkpoint
from tools.exporter.hf_whisper import WhisperConfig, build_whisper_encoder_graph, build_whisper_decoder_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--whisper-snapshot", type=Path, default=Path("/tmp/volvoxai-h-stage-whisper-169d4a"))
    parser.add_argument("--vqa-package", type=Path, default=ROOT / "build/receipt-vqa-ptq-head-29ffb7e/fp32")
    args = parser.parse_args()
    if args.out.exists(): parser.error("output directory must be fresh")
    args.out.mkdir(parents=True)
    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
    from transformers.generation.logits_process import SuppressTokensLogitsProcessor
    from examples.tiny_receipt_vqa.tools import verify_proto_backends as vqa
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "tools/llm_baseline/multimodal_sensor.json").read_text())
    audio_pin = pinned_audio(args.whisper_snapshot, manifest)
    image_source, upstream, upstream_tokenizer, package_manifest, image_pin = pinned_image(args.vqa_package, manifest)
    source, weights, _ = read_hf_checkpoint(args.whisper_snapshot)
    config = WhisperConfig.from_hf(source, max_context=16)
    suppressed = sorted(set(source["suppress_tokens"]) | set(range(50364, config.vocabulary)))
    encoder, ew = build_whisper_encoder_graph(config, weights)
    decoder, dw = build_whisper_decoder_graph(config, weights, generate=True, suppressed_ids=suppressed)
    del weights
    packages = {}
    for role, graph, bank in (("audio_encoder", encoder, ew), ("audio_decoder", decoder, dw)):
        directory = args.out / role; directory.mkdir()
        (directory / "graph.json").write_text(json.dumps(graph, allow_nan=False))
        save_file(bank, str(directory / "model.safetensors"))
        packages[role] = {"graph_sha256": sha(directory / "graph.json"), "weights_sha256": sha(directory / "model.safetensors"),
            "weights_bytes": (directory / "model.safetensors").stat().st_size}
    hf = WhisperForConditionalGeneration.from_pretrained(args.whisper_snapshot, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").eval()
    tokenizer = WhisperTokenizer.from_pretrained(args.whisper_snapshot, local_files_only=True,
        language="english", task="transcribe", predict_timestamps=False, clean_up_tokenization_spaces=False)
    clock = np.arange(24000, dtype=np.float32) / np.float32(16000)
    pcm = (.2 * np.sin(2 * np.pi * 220 * clock) + .05 * np.sin(2 * np.pi * 730 * clock)).astype(np.float32)
    padded = np.zeros((1, 480000), np.float32); padded[0, :len(pcm)] = pcm
    features = WhisperFeatureExtractor.from_pretrained(args.whisper_snapshot, local_files_only=True)._np_extract_fbank_features(padded, "cpu")
    arrays = {"audio_pcm": pcm, "audio_features": features}
    prefix, ids, policy = list(tokenizer.prefix_tokens), [], SuppressTokensLogitsProcessor(suppressed)
    with torch.inference_mode():
        memory = hf.model.encoder(torch.from_numpy(features)).last_hidden_state
        arrays["audio_memory"] = memory.numpy()
        for index in range(3):
            tokens = torch.tensor([prefix]); logits = hf(encoder_outputs=(memory,), decoder_input_ids=tokens, use_cache=False).logits
            arrays[f"audio_logits_{index}"] = logits.numpy()
            token = int(policy(tokens, logits[:, -1].clone()).argmax(-1)[0]); ids.append(token); prefix.append(token)
            if token == source["eos_token_id"]: break
    corpus = {"format": "volvox-multimodal-oracles/v1", "status": "REFERENCE_PREPARED",
        "producer_sha256": sha(__file__), "manifest_sha256": sha(ROOT / "tools/llm_baseline/multimodal_sensor.json"),
        "oracle": {"transformers": transformers.__version__, "torch": torch.__version__,
            "hf_model_source_sha256": sha(sys.modules[hf.__class__.__module__].__file__), "onnx": "pinned original TinyReceiptVQA FP32 split graphs"},
        "packages": packages, "audio": {"pin": audio_pin, "prompt_ids": tokenizer.prefix_tokens, "expected_ids": ids,
            "suppress_ids": suppressed, "eos_id": source["eos_token_id"], "pcm_sha256": array_hash(pcm),
            "policy": "fixed English/transcribe/no-timestamps; HF forward+official fixed suppression; no begin-only suppression",
            "tolerances": {"features": {"atol": 1e-5, "rtol": 1e-5}, "encoder": {"atol": 2e-4, "rtol": 2e-4, "rms": 2e-5},
                "logits": {"atol": 2e-3, "rtol": 1e-3, "rms": 5e-4}}}, "image": {"cases": []}}
    package = args.vqa_package
    assets = package_manifest["graphs"]; mapping = assets["decoder"]["inputs"]
    for role in ("encoder", "decoder"):
        directory = args.out / f"image_{role}"; directory.mkdir()
        graph = json.loads((package / assets[role]["graph"]["path"]).read_text())
        if role == "encoder": graph["execution"] = stateless_execution(graph["inputs"], graph["outputs"])
        else:
            invariant = [mapping[name] for name in mapping if name.startswith("cross_") or name in ("family_ids", "memory_padding_mask")]
            feedback = {mapping[f"past_{kind}_{index}"]: assets[role]["outputs"][f"present_{kind}_{index}"]
                for index in range(4) for kind in ("k", "v")}
            feedback[mapping["past_padding_mask"]] = assets[role]["outputs"]["present_padding_mask"]
            graph["execution"] = tensor_feedback_execution(graph["inputs"], graph["outputs"], feedback=feedback,
                position_input=mapping["position_ids"], invariant_inputs=invariant)
        (directory / "graph.json").write_text(json.dumps(graph, allow_nan=False))
        shutil.copyfile(package / assets[role]["weights"]["path"], directory / "model.safetensors")
        corpus["packages"][f"image_{role}"] = {"graph_sha256": sha(directory / "graph.json"),
            "weights_sha256": sha(directory / "model.safetensors"), "weights_bytes": (directory / "model.safetensors").stat().st_size}
    for filename, question in (("receipt_en.jpg", "What is the store name?"), ("receipt_ko.jpg", "가게 이름은 무엇인가요?")):
        path = image_source / "examples" / filename
        shutil.copyfile(path, args.out / filename)
        pixels = prepare_receipt_image(path); question_ids = vqa.tokenizer(package).encode(vqa.clean(question), add_eos=True, max_len=192)
        np.testing.assert_array_equal(pixels, upstream.preprocess_image(path))
        np.testing.assert_array_equal(question_ids, upstream_tokenizer.encode(upstream.clean_text(question), 192).reshape(-1))
        reference = vqa.OrtRunner(image_source, "fp32")
        key = filename.removesuffix(".jpg")
        try:
            encoded = reference.run("encoder", vqa.encoder_feed(pixels, question_ids))
            for name, value in encoded.items(): arrays[f"{key}_encoder_{name}"] = value
            feed = vqa.decoder_feed(encoded, 0); tokens = []
            for index in range(191):
                output = reference.run("decoder", feed); arrays[f"{key}_logits_{index}"] = output["logits"]
                token = int(output["logits"].argmax(-1)[0, 0]); tokens.append(token); vqa.advance(feed, output, token, index)
                if token == 2: break
        finally: reference.close()
        assert tokens[-1] == 2
        corpus["image"]["cases"].append({"key": key, "jpeg": filename, "jpeg_sha256": sha(path), "question": question,
            "question_ids": question_ids, "pixels_sha256": array_hash(pixels), "expected_ids": tokens,
            "pinned_upstream_image_processor_exact": True, "pinned_upstream_question_tokenizer_exact": True})
    corpus["image"]["contracts"] = {role: {"inputs": assets[role]["inputs"], "outputs": assets[role]["outputs"]} for role in ("encoder", "decoder")}
    corpus["image"]["pin"] = image_pin
    corpus["image"]["tolerances"] = {"atol": 1e-4, "rtol": 1e-4}
    np.savez_compressed(args.out / "references.npz", **arrays)
    corpus["references_sha256"] = sha(args.out / "references.npz")
    (args.out / "corpus.json").write_text(json.dumps(corpus, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "REFERENCE_PREPARED", "audio_ids": ids, "image_cases": len(corpus["image"]["cases"]),
        "weight_bytes": sum(value["weights_bytes"] for value in corpus["packages"].values()), "references_bytes": (args.out / "references.npz").stat().st_size}))


if __name__ == "__main__": main()
