#!/usr/bin/env python3
"""Record the current host's original-ONNX VQA INT8 outputs independently.

The frozen 2000-case reference is retained. This bounded, two-input check uses
the existing accuracy driver and records any disagreement with that reference.
Run on the benchmark host, outside other benchmark measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.tiny_receipt_vqa.tools import verify_proto_backends as audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    files = spec["models"]["vqa"]["onnx"]["int8"]
    file_hashes = {path: sha(ROOT / path) for path in files.values()}
    assert all(spec["file_hashes"][p] == v for p, v in file_hashes.items())
    package = (ROOT / spec["models"]["vqa"]["variants"]["fp32"]["encoder"]["graph"]).parents[1]
    tok = audit.tokenizer(package)
    def answer(tokens):
        match = re.search(r"<answer>(.*?)</answer>", tok.decode(tokens))
        assert match is not None
        return audit.clean(match.group(1))
    runner = audit.OrtRunner((ROOT / files["encoder"]).parent, "int8")
    expected, comparisons = {}, {}
    try:
        for case in spec["cases"]:
            path = ROOT / case["pixels"]
            assert sha(path) == case["sha256"]
            pixels = np.fromfile(path, dtype=np.float32).reshape(1, 1, 320, 672)
            actual = []
            for _ in range(2):
                value = audit.generate(runner, pixels, case["question_ids"])
                assert value["eos"]
                actual.append([value["tokens"], value["family"]])
            assert actual[0] == actual[1], "independent ORT reference changed between repetitions"
            key = str(case["index"])
            expected[key] = actual[0]
            original = spec["expected"]["vqa/ort/inference/int8"][key]
            comparisons[key] = {"whole_tokens_equal": actual[0][0] == original[0],
                                "answer_equal": answer(actual[0][0]) == answer(original[0]),
                                "router_equal": actual[0][1] == original[1],
                                "original_decoder_steps": len(original[0]) - 1,
                                "current_decoder_steps": len(actual[0][0]) - 1}
    finally:
        runner.close()
    report = {"schema": "volvoxai.receipt-ort-benchmark-reference/v1", "model": "vqa", "variant": "int8",
              "spec_sha256": sha(args.spec), "audit_harness_sha256": sha(Path(audit.__file__)),
              "preparation_harness_sha256": sha(Path(__file__)), "onnxruntime": ort.__version__,
              "numpy": np.__version__, "python": platform.python_version(),
              "case_indices": [c["index"] for c in spec["cases"]], "file_hashes": file_hashes,
              "expected": expected, "repeated_outputs_identical": True,
              "original_2000_reference_comparison": comparisons}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reference_sha256": sha(args.out), "comparison": comparisons}))


if __name__ == "__main__":
    main()
