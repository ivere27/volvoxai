#!/usr/bin/env python3
"""Verify ONNX imports and C PTQ through the generated Python clients.

Supply the producer release, a calibration image directory, and a disjoint
evaluation image/annotation pair. The output directory retains fresh imports,
the calibrated package, logits, input identities and a numerical report.
No exporter-side quantization is used: author/calibrate/inspect/write are RPCs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx  # noqa: E402

from .eval_heldout import decode_slots, load_router, score  # noqa: E402
from .import_hf_onnx import (  # noqa: E402
    DEFAULT_PTQ_FLOAT_OPS, import_release, optimize_package,
)

pb = vx.pb

# Python installs only the full library, so name both release builds directly.
# Running the same PTQ export through each one keeps the profile-parity check.
LIBRARIES = {
    "inference": ROOT / "native" / "libvolvoxai-lite.so",
    "full": ROOT / "native" / "libvolvoxai.so",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def preprocess(path: Path, config: dict) -> np.ndarray:
    spec = config["input"]
    with Image.open(path) as image:
        image = image.convert("L").resize(
            (spec["width"], spec["height"]), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(((values - 0.5) / 0.5)[None, None])


def tensor(values: np.ndarray) -> object:
    return pb.Tensor(name="input0", shape=list(values.shape),
                     dtype=pb.DataType.DATA_TYPE_F32, inline=values.tobytes())


def load_model(client, runtime, package: Path):
    return client.load_model(pb.LoadModelRequest(
        runtime_id=runtime.runtime_id, graph_path=str(package / "graph.json"),
        weight_paths=[str(package / "model.safetensors")]))


class Runner:
    """One retained proto execution context; results are released per sample."""

    def __init__(self, stack: ExitStack, package: Path, library: Path, threads: int):
        host = vx.open_library(library)
        stack.callback(host.close)
        self.client = vx.VxInferenceServiceClient(host)
        runtime = self.client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=threads))
        model = load_model(self.client, runtime, package)
        compiled = self.client.compile_model(pb.CompileModelRequest(
            model_id=model.model_id,
            policy=pb.BackendPolicy(
                mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=["cpu"],
                operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        self.context = self.client.create_execution_context(pb.CreateExecutionContextRequest(
                                   compiled_model_id=compiled.compiled_model_id))

    def run(self, values: np.ndarray) -> np.ndarray:
        result = self.client.execute(pb.ExecuteRequest(
            context_id=self.context.context_id, inputs=[tensor(values)]))
        try:
            self.client.get_result(pb.ResultRef(result_id=result.result_id))
            output = self.client.read_output(pb.ReadOutputRequest(
                result_id=result.result_id, name="slot_logits")).tensor
            if output.dtype != pb.DataType.DATA_TYPE_F32:
                raise AssertionError(f"unexpected output dtype: {output.dtype}")
            produced = np.frombuffer(output.inline, dtype=np.float32).copy()
            return produced.reshape(tuple(output.shape))
        finally:
            self.client.release_result(pb.ResultRef(result_id=result.result_id))


def calibrate(package: Path, output: Path, paths: list[Path], config: dict,
              threads: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    host = vx.open_library("full")
    try:
        inference = vx.VxInferenceServiceClient(host)
        quantization = vx.VxQuantizationServiceClient(host)
        runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=threads))
        model = load_model(inference, runtime, package)
        authored = quantization.author_ptq_template(pb.AuthorPtqTemplateRequest(
            source_graph=(package / "graph.json").read_bytes(),
            weight_shards=[(package / "model.safetensors").read_bytes()],
            config=pb.PtqAuthoringConfig(
                activation_dtype=pb.DataType.DATA_TYPE_I8,
                activation_scheme=pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC,
                weight_dtype=pb.DataType.DATA_TYPE_I8,
                float_operators=list(DEFAULT_PTQ_FLOAT_OPS))))
        (output / "template.graph.json").write_bytes(authored.template_graph)
        # Keep the exact generated request for a second native/WASM host.
        request = pb.CreatePtqPlanRequest(
            model_id=model.model_id, template_graph=authored.template_graph,
            observers=list(authored.observers), layers=list(authored.layers),
            profile_names=["default"])
        plan = quantization.create_ptq_plan(request)
        request.model_id = 0
        (output / "plan.pb").write_bytes(request.to_bytes())
        started = time.perf_counter()
        with (output.parent / "calibration.f32").open("wb") as raw:
            for index, path in enumerate(paths):
                values = preprocess(path, config)
                raw.write(values.tobytes())
                calibrated = quantization.calibrate_ptq_plan(pb.CalibratePtqPlanRequest(
                    ptq_plan_id=plan.ptq_plan_id, profile_name="default",
                    sample_name=f"calibration-{index}", sample_count=1,
                    inputs=[tensor(values)]))
                if (index + 1) % 32 == 0 or index + 1 == len(paths):
                    print(f"calibration {index + 1}/{len(paths)}", flush=True)
        inspected = quantization.inspect_ptq_plan(pb.PtqPlanRef(ptq_plan_id=plan.ptq_plan_id))
        if not inspected.coverage.complete or inspected.calibration_samples != len(paths):
            raise AssertionError("incomplete calibration coverage")
        packed = quantization.write_ptq_package(pb.WritePtqPackageRequest(ptq_plan_id=plan.ptq_plan_id))
        (output / "graph.json").write_bytes(packed.graph)
        (output / "model.safetensors").write_bytes(packed.weights)
        # Exercise native path export as well as the portable byte result.
        with tempfile.TemporaryDirectory(prefix="path-export-", dir=output) as directory:
            path_output = Path(directory)
            quantization.write_ptq_package(pb.WritePtqPackageRequest(
                ptq_plan_id=plan.ptq_plan_id,
                output_graph_path=str(path_output / "graph.json"),
                output_weights_path=str(path_output / "model.safetensors")))
            if (path_output / "model.safetensors").read_bytes() != packed.weights:
                raise AssertionError("path and byte exports have different weights")
            if json.loads((path_output / "graph.json").read_bytes()) != json.loads(packed.graph):
                raise AssertionError("path and byte exports have different graphs")
        return {
            "quantized_nodes": authored.quantized_nodes,
            "retained_float_nodes": authored.retained_float_nodes,
            "layers": len(authored.layers), "observers": len(authored.observers),
            "activation_dtype": "int8", "activation_scheme": "asymmetric",
            "float_operators": list(DEFAULT_PTQ_FLOAT_OPS),
            "cpu_threads": threads,
            "plan": inspected.to_dict(), "path_byte_export_equal": True,
            "seconds": time.perf_counter() - started,
        }
    finally:
        host.close()


def compare(actual: np.ndarray, expected: np.ndarray, config: dict) -> dict:
    actual_ids, expected_ids = actual.argmax(-1), expected.argmax(-1)
    records = lambda ids: [decode_slots(row, config) for row in ids]
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "max_abs_logit_diff": float(difference.max()),
        "mean_abs_logit_diff": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "slot_argmax_agreement": float(np.mean(actual_ids == expected_ids)),
        "record_agreement": sum(a == b for a, b in zip(
            records(actual_ids), records(expected_ids))) / len(actual),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--calibration-images", required=True, type=Path)
    parser.add_argument("--eval-images", required=True, type=Path)
    parser.add_argument("--eval-annotations", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--calibration-count", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every annotation")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--threads", type=int, default=1)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-imports", action="store_true",
                       help="calibrate and evaluate existing imported/prepared packages")
    reuse.add_argument("--reuse-packages", action="store_true",
                       help="evaluate existing imports and calibrated package")
    args = parser.parse_args()
    if args.calibration_count < 1 or args.limit < 0 or args.threads < 1:
        parser.error("counts and threads must be positive (limit may be zero)")
    source, out = args.source.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    router = load_router(source)
    labels, evaluation = [], []
    for annotation in sorted(args.eval_annotations.glob("*.json")):
        image = args.eval_images / f"{annotation.stem}.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        document = json.loads(annotation.read_text())
        store = (document.get("receipt") or {}).get("store") or {}
        labels.append({
            "phone": router.digits_only(store.get("phone", "")),
            "street": router.street_no_from_address(store.get("address", "")),
            "question": str(document.get("question", "")),
            "answer": str(document.get("answer", "")),
        })
        evaluation.append(image)
    if not evaluation:
        raise ValueError("evaluation set is empty")
    # Compare normalized content as well as files: alternate encodings or
    # resolutions of the same network input must not enter both sets.
    evaluation_hashes = {sha256(preprocess(path, config).tobytes()) for path in evaluation}
    if args.limit:
        labels, evaluation = labels[:args.limit], evaluation[:args.limit]
    candidates = sorted(p for p in args.calibration_images.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(args.seed).shuffle(candidates)
    calibration, calibration_hashes = [], set()
    for path in candidates:
        digest = sha256(preprocess(path, config).tobytes())
        if digest in evaluation_hashes or digest in calibration_hashes:
            continue
        calibration.append(path)
        calibration_hashes.add(digest)
        if len(calibration) == args.calibration_count:
            break
    if len(calibration) != args.calibration_count:
        raise ValueError("insufficient distinct calibration images outside evaluation")
    report = {
        "schema": "volvoxai.receipt-proto-ptq-verification/v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {"onnxruntime": ort.__version__, "numpy": np.__version__},
        "runtime_sha256": {name: sha256(path.read_bytes())
                           for name, path in LIBRARIES.items()},
        "seed": args.seed, "cpu_threads": args.threads,
        "evaluation_samples": len(evaluation),
        "calibration_samples": len(calibration),
        "normalized_calibration_eval_overlap": len(calibration_hashes & evaluation_hashes),
        "source_sha256": {name: sha256((source / name).read_bytes())
                          for name in ("model.onnx", "model_int8.onnx")},
    }
    identities = {
        "calibration": sorted(calibration_hashes),
        "evaluation": sorted(evaluation_hashes),
    }
    identity_path = out / "input-identities.json"
    if args.reuse_packages and json.loads(identity_path.read_text()) != identities:
        raise ValueError("existing package input identities differ; rerun calibration")
    write_json(identity_path, identities)
    if args.reuse_imports or args.reuse_packages:
        for variant in ("fp32", "int8"):
            manifest = json.loads((out / variant / "manifest.json").read_text())
            for name, digest in manifest["source"]["sha256"].items():
                if sha256((source / name).read_bytes()) != digest:
                    raise ValueError(f"existing {variant} package comes from a different {name}")
    if not args.reuse_packages:
        if not args.reuse_imports:
            for variant in ("fp32", "int8"):
                import_release(source, out / variant, variant=variant, targets=[],
                               calibration_path=None, activation_dtype="int8",
                               activation_scheme="asymmetric",
                               float_ops=DEFAULT_PTQ_FLOAT_OPS)
                print(f"imported {variant}", flush=True)
            optimize_package(out / "fp32", out / "prepared", prepare_for_ptq=True)
        report["ptq"] = calibrate(out / "prepared", out / "ptq",
                                  calibration, config, args.threads)
        write_json(out / "ptq-report.json", report["ptq"])
    else:
        report["ptq"] = json.loads((out / "ptq-report.json").read_text())
    report["artifacts"] = {}
    for variant in ("fp32", "int8", "prepared", "ptq"):
        package = out / variant
        graph = json.loads((package / "graph.json").read_text())
        report["artifacts"][variant] = {
            "operators": dict(Counter(node["opType"] for node in graph["nodes"])),
            "files": {name: {"bytes": (package / name).stat().st_size,
                             "sha256": sha256((package / name).read_bytes())}
                      for name in ("graph.json", "model.safetensors")},
        }
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    routes = {}
    with ExitStack() as stack:
        for variant in ("fp32", "int8", "ptq"):
            routes[f"volvoxai_{variant}"] = Runner(stack, out / variant, LIBRARIES["inference"], args.threads).run
        # Loading the PTQ export into the full library independently also
        # checks that both generated service closures execute the same graph.
        routes["volvoxai_ptq_full"] = Runner(stack, out / "ptq", LIBRARIES["full"], args.threads).run
        for variant, filename in (("fp32", "model.onnx"), ("int8", "model_int8.onnx")):
            session = ort.InferenceSession(str(source / filename), options,
                                           providers=["CPUExecutionProvider"])
            routes[f"ort_{variant}"] = lambda values, s=session: s.run(None, {"image": values})[0]
        outputs = {name: [] for name in routes}
        started = time.perf_counter()
        with (out / "eval-smoke.f32").open("wb") as smoke:
            for index, path in enumerate(evaluation):
                values = preprocess(path, config)
                if index < 8:
                    smoke.write(values.tobytes())
                for name, run in routes.items():
                    logits = run(values)
                    if logits.shape != (1, config["slots"], config["num_classes"]) or not np.isfinite(logits).all():
                        raise AssertionError(f"{name}: invalid output for sample {index}")
                    outputs[name].append(logits[0])
                if (index + 1) % 50 == 0 or index + 1 == len(evaluation):
                    print(f"evaluation {index + 1}/{len(evaluation)} ({time.perf_counter() - started:.1f}s)", flush=True)
        outputs = {name: np.stack(values) for name, values in outputs.items()}
        np.savez(out / "logits.npz", **outputs)
        for name, logits in outputs.items():
            (out / f"{name}-smoke-logits.f32").write_bytes(logits[:8].tobytes())
    report["scores"] = {
        name: score([decode_slots(row, config) for row in logits.argmax(-1)], labels, router)
        for name, logits in outputs.items()
    }
    report["comparisons"] = {
        name: compare(outputs[actual], outputs[expected], config)
        for name, actual, expected in (
            ("fp32_import_vs_ort", "volvoxai_fp32", "ort_fp32"),
            ("int8_import_vs_ort", "volvoxai_int8", "ort_int8"),
            ("ptq_vs_fp32", "volvoxai_ptq", "volvoxai_fp32"),
            ("ptq_profiles", "volvoxai_ptq_full", "volvoxai_ptq"),
        )
    }
    report["checks"] = {
        "fp32_logit_parity": bool(np.allclose(outputs["volvoxai_fp32"], outputs["ort_fp32"], atol=1e-4, rtol=1e-4)),
        "ptq_profiles_equal": bool(np.array_equal(outputs["volvoxai_ptq"], outputs["volvoxai_ptq_full"])),
        "ptq_target_drop_at_most_one_point": report["scores"]["volvoxai_fp32"]["target_exact"] - report["scores"]["volvoxai_ptq"]["target_exact"] <= 0.01,
        "ptq_all_convolutions_quantized": report["artifacts"]["ptq"]["operators"].get("QConv2D", 0) == 11,
    }
    report["passed"] = all(report["checks"].values())
    write_json(out / "report.json", report)
    print(json.dumps({key: report[key] for key in ("scores", "comparisons", "checks", "passed")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
