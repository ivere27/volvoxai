#!/usr/bin/env python3
"""Score receipt-digit-reader packages on a held-out split, next to ONNX Runtime.

The held-out split is a measurement set: it must not have entered training,
checkpoint selection, or PTQ calibration. This tool does not enforce that --
only the caller knows the provenance -- but it does report the exact record
counts so a disagreement is attributable rather than aggregate.

`--onnx` adds an ONNX Runtime column computed on the identical preprocessed
batch, so a VolvoxAI package is compared against the producer on the same
inputs rather than against a published number from a different pipeline.

    python3 -m examples.receipt_digit_reader.tools.eval_heldout \\
        --package build/receipt-digit-reader-fp32 \\
        --images heldout.f32 --labels heldout.json \\
        --onnx /path/to/model.onnx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decode_slots(argmax: Sequence[int], decode: Mapping[str, int]) -> tuple[str, str]:
    """Read each block until its first blank, matching ReceiptDigitSession."""

    blank = int(decode["blank_class"])
    phone_slots = int(decode["phone_slots"])
    slots = int(decode["slots"])

    def read(start: int, stop: int) -> str:
        digits = []
        for index in range(start, stop):
            if int(argmax[index]) == blank:
                break
            digits.append(str(int(argmax[index])))
        return "".join(digits)

    return read(0, phone_slots), read(phone_slots, slots)


def score(
    records: Sequence[tuple[str, str]],
    labels: Sequence[Mapping[str, str]],
    router=None,
) -> dict:
    """Reproduce the producer's held-out metric set exactly.

    The per-field rates use non-empty denominators: a receipt that carries no
    street number is not evidence that the street head is right. `target_exact`
    is the strict record rate over every receipt -- every phone digit and every
    street digit correct on the same image, whether or not the question asked
    for them -- so it is the number to quote when one wrong digit anywhere
    should count as a failure.
    """

    phone_hits = phone_total = street_hits = street_total = target_hits = 0
    families: dict[str, dict[str, int]] = {}
    answer_hits = answer_total = 0
    for (phone, street), label in zip(records, labels):
        if (not label["phone"] or phone == label["phone"]) and \
                (not label["street"] or street == label["street"]):
            target_hits += 1
        if label["phone"]:
            phone_total += 1
            phone_hits += int(phone == label["phone"])
        if label["street"]:
            street_total += 1
            street_hits += int(street == label["street"])
        if router is not None and label.get("question") is not None:
            family = router.route_family_from_question(label["question"])
            bucket = families.setdefault(family, {"n": 0, "ok": 0})
            bucket["n"] += 1
            correct = int(
                router.answer_from_record(label["question"], phone, street)
                == label.get("answer", "")
            )
            bucket["ok"] += correct
            answer_total += 1
            answer_hits += correct
    report = {
        "n": len(labels),
        "phone_digit_exact": phone_hits / max(1, phone_total),
        "street_exact": street_hits / max(1, street_total),
        "target_exact": target_hits / max(1, len(labels)),
    }
    if answer_total:
        report["answer_exact"] = answer_hits / answer_total
        report["by_family"] = {
            name: {"n": bucket["n"], "answer_exact": bucket["ok"] / max(1, bucket["n"])}
            for name, bucket in sorted(families.items())
        }
    return report


def load_router(source: Path | None):
    """Load the producer's shipped regex router, if one was supplied.

    The router ships with the model because the graph alone cannot answer a
    question. Scoring `answer_exact` against a different router would measure
    the router, not the reader, so the producer's own file is the reference.
    """

    if source is None:
        return None
    module_path = source / "question_router.py"
    if not module_path.is_file():
        raise SystemExit(f"question router is missing at {module_path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "receipt_digit_reader_question_router", module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_package(
    package: Path, batch: np.ndarray, decode: Mapping[str, int], input_name: str,
    backend: str = "cpu",
) -> tuple[list[tuple[str, str]], np.ndarray, float]:
    """Execute every sample through the native inference CLI.

    `backend` selects a runtime identity, so the same split can be scored on a
    device route. Each sample is one process, which pays device initialization
    every image -- fine for scoring a record, useless as a latency figure.
    """

    binary = REPOSITORY_ROOT / "native" / "volvoxai"
    if not binary.is_file():
        raise SystemExit(f"native runtime is not built at {binary}; run `make build_native`")
    records: list[tuple[str, str]] = []
    logits: list[np.ndarray] = []
    elapsed = 0.0
    classes = int(decode["num_classes"])
    slots = int(decode["slots"])
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        source = workspace / "input.f32"
        sink = workspace / "output.f32"
        for sample in batch:
            source.write_bytes(np.ascontiguousarray(sample, dtype=np.float32).tobytes())
            started = time.perf_counter()
            subprocess.run(
                [str(binary), "run", str(package),
                 "--input", f"{input_name}={source}",
                 "--output", f"slot_logits={sink}",
                 f"--{backend}"],
                check=True, capture_output=True, cwd=REPOSITORY_ROOT,
            )
            elapsed += time.perf_counter() - started
            values = np.fromfile(sink, dtype=np.float32).reshape(slots, classes)
            logits.append(values)
            records.append(decode_slots(values.argmax(-1), decode))
    return records, np.stack(logits), elapsed


def run_onnx(
    model: Path, batch: np.ndarray, decode: Mapping[str, int],
) -> tuple[list[tuple[str, str]], np.ndarray]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"],
    )
    name = session.get_inputs()[0].name
    records: list[tuple[str, str]] = []
    logits: list[np.ndarray] = []
    for sample in batch:
        values = session.run(None, {name: sample[None, ...]})[0].reshape(
            int(decode["slots"]), int(decode["num_classes"]),
        )
        logits.append(values)
        records.append(decode_slots(values.argmax(-1), decode))
    return records, np.stack(logits)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path,
                        help="raw F32 batch shaped [N, C, H, W]")
    parser.add_argument("--labels", required=True, type=Path,
                        help="JSON list of {phone, street} in batch order")
    parser.add_argument("--onnx", type=Path, help="producer ONNX for a side-by-side column")
    parser.add_argument("--router-source", type=Path,
                        help="release directory holding question_router.py; adds "
                             "answer_exact and the per-family breakdown")
    parser.add_argument("--backend", default="cpu",
                        help="runtime identity for the VolvoxAI column "
                             "(cpu, vulkan, opengl, cuda); default cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write the report as JSON")
    args = parser.parse_args(argv)

    manifest = _read_json(args.package / "manifest.json")
    decode = manifest["decode"]
    shape = manifest["abi"]["input"]["shape"]
    input_name = manifest["abi"]["input"]["name"]
    labels = _read_json(args.labels)
    batch = np.fromfile(args.images, dtype=np.float32).reshape(-1, *shape[1:])
    if len(batch) != len(labels):
        raise SystemExit(
            f"{len(batch)} images do not match {len(labels)} labels"
        )
    if args.limit:
        batch, labels = batch[: args.limit], labels[: args.limit]

    router = load_router(args.router_source)
    records, package_logits, elapsed = run_package(
        args.package, batch, decode, input_name, args.backend,
    )
    report: dict[str, Any] = {
        "package": str(args.package),
        "variant": manifest.get("variant"),
        "package_class": manifest.get("package_class"),
        "backend": args.backend,
        "volvoxai": {
            **score(records, labels, router),
            "ms_per_image_including_process_spawn": 1000 * elapsed / len(labels),
        },
    }
    if args.onnx is not None:
        onnx_records, onnx_logits = run_onnx(args.onnx, batch, decode)
        report["onnxruntime"] = score(onnx_records, labels, router)
        report["agreement"] = {
            "record_exact": sum(
                1 for a, b in zip(records, onnx_records) if a == b
            ) / len(labels),
            "slot_argmax": float(
                (package_logits.argmax(-1) == onnx_logits.argmax(-1)).mean()
            ),
            "max_abs_logit_difference": float(
                np.abs(package_logits - onnx_logits).max()
            ),
        }
    text = json.dumps(report, indent=1)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
