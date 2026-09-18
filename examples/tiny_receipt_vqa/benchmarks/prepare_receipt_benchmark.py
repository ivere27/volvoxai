#!/usr/bin/env python3
"""Freeze two benchmark inputs and references from a completed 2000-case audit.

This prepares data only; it performs no inference. Model payloads are reused,
and original ONNX files are copied beside the private benchmark specification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["validation", "digit-work", "vqa-work", "digit-source", "vqa-source", "digit-reference", "out"]:
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads((args.validation / "verification-summary.json").read_text())
    plan = json.loads((args.validation / "plan.json").read_text())
    assert validation["matrix_complete"] and validation["complete_cells"] == 88 and not validation["invalid"]
    args.out.mkdir(parents=True, exist_ok=True)
    spec = {"schema": "volvoxai.receipt-proto-benchmark-inputs/v1", "base_head": validation["base_head"],
            "case_selection": "first two held-out records, indices 0 and 1, fixed before timing",
            "validation_sha256": checksum(args.validation / "verification-summary.json"),
            "artifacts": plan["artifacts"], "libraries": plan["libraries"],
            "file_hashes": {}, "cases": [], "models": {}, "expected": {}}
    corpus = json.loads((args.vqa_work / "corpus/corpus.json").read_text())
    assert corpus["images_sha256"] == json.loads((args.digit_work / "corpus/corpus.json").read_text())["images_sha256"]
    for index in [0, 1]:
        raw = []
        for work in [args.vqa_work, args.digit_work]:
            with (work / "corpus/images.u8").open("rb") as stream:
                stream.seek(index * 320 * 672)
                raw.append(stream.read(320 * 672))
        assert raw[0] == raw[1] and len(raw[0]) == 320 * 672
        pixels = ((np.frombuffer(raw[0], dtype=np.uint8).astype(np.float32) / 255 - .5) / .5)
        output = args.out / f"case-{index}.f32"
        output.write_bytes(pixels.tobytes())
        spec["cases"].append({"index": index, "pixels": relative(output), "sha256": checksum(output),
                              "question_ids": corpus["evaluation"][index]["question_ids"]})
    for model, work, source in [("digit", args.digit_work, args.digit_source), ("vqa", args.vqa_work, args.vqa_source)]:
        spec["models"][model] = {"variants": {}, "onnx": {}}
        for variant in ["fp32", "int8", "ptq", "ptq-wasm"]:
            if model == "digit":
                folder = work / "corpus" / variant
                roles = {"digit": {"graph": relative(folder / "graph.json"), "weights": relative(folder / "model.safetensors"),
                         "inputs": {"image": "input0"}, "outputs": {"slot_logits": "slot_logits"}}}
            else:
                folder = work / variant
                manifest = json.loads((folder / "package_manifest.json").read_text())
                roles = {role: {"graph": relative(folder / value["graph"]["path"]),
                         "weights": relative(folder / value["weights"]["path"]),
                         "inputs": value["inputs"], "outputs": value["outputs"]}
                         for role, value in manifest["graphs"].items()}
            spec["models"][model]["variants"][variant] = roles
            for contract in roles.values():
                for kind in ["graph", "weights"]:
                    spec["file_hashes"][contract[kind]] = checksum(ROOT / contract[kind])
        destination = args.out / "onnx" / model
        destination.mkdir(parents=True, exist_ok=True)
        for variant in ["fp32", "int8"]:
            suffix = "_int8" if variant == "int8" else ""
            paths = {}
            for role in (["digit"] if model == "digit" else ["encoder", "decoder"]):
                filename = f"model{suffix}.onnx" if role == "digit" else f"{role}_model{suffix}.onnx"
                target = destination / filename
                shutil.copy2(source / filename, target)
                paths[role] = relative(target)
                spec["file_hashes"][relative(target)] = checksum(target)
            spec["models"][model]["onnx"][variant] = paths
    cached = {}
    for row in validation["matrix"]:
        key = "/".join(row[k] for k in ["model", "backend", "profile", "variant"])
        if row["model"] == "vqa":
            report = cached.setdefault(row["report"], json.loads((args.validation / row["report"]).read_text()))
            predictions = report["variants"][row["variant"]]["predictions"]
            values = {str(i): [predictions[i]["tokens"], predictions[i]["family"]] for i in [0, 1]}
        else:
            report = args.validation / row["report"]
            path = report.with_name(report.stem + "-" + row["variant"] + "-logits.f32")
            logits = np.fromfile(path, dtype=np.float32, count=2 * 16 * 11).reshape(2, 16, 11)
            values = {str(i): logits[i].argmax(-1).tolist() for i in [0, 1]}
        spec["expected"][key] = values
    ort = json.loads((args.vqa_work / "ort-2000.json").read_text())
    assert ort["samples"] == 2000
    for variant in ["fp32", "int8"]:
        predictions = ort["variants"][variant]["predictions"]
        spec["expected"][f"vqa/ort/inference/{variant}"] = {str(i): [predictions[i]["tokens"], predictions[i]["family"]] for i in [0, 1]}
    with np.load(args.digit_reference) as data:
        for variant in ["fp32", "int8"]:
            logits = data[f"ort_{variant}"]
            assert logits.shape == (2000, 16, 11)
            spec["expected"][f"digit/ort/inference/{variant}"] = {str(i): logits[i].argmax(-1).tolist() for i in [0, 1]}
    spec["reference_sources"] = {relative(args.digit_reference): checksum(args.digit_reference),
                                  relative(args.vqa_work / "ort-2000.json"): checksum(args.vqa_work / "ort-2000.json")}
    (args.out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"cases": [0, 1], "reference_routes": len(spec["expected"]), "spec_sha256": checksum(args.out / "spec.json")}))


if __name__ == "__main__":
    main()
