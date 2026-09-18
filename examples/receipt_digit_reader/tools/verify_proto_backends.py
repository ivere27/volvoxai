#!/usr/bin/env python3
"""Prepare identical receipt inputs and verify native routes through proto.

Reports separate successful execution from numerical agreement. A requested
GPU must attest every node, with tier and operator fallback both forbidden.
Run each physical GPU route sequentially on an idle device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx  # noqa: E402

pb = vx.pb
HEIGHT, WIDTH = 320, 672
VARIANTS = ("fp32", "int8", "ptq", "ptq-wasm")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def normalize(gray):
    return np.ascontiguousarray((gray.astype(np.float32) / 255 - .5) / .5)


def records(logits):
    result = []
    for row in logits.argmax(-1):
        fields = []
        for block in (row[:12], row[12:]):
            digits = []
            for digit in block:
                if digit == 10:
                    break
                digits.append(str(digit))
            fields.append("".join(digits))
        result.append(tuple(fields))
    return result


def compare(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs_logit_diff": float(np.abs(delta).max()),
        "mean_abs_logit_diff": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "logits_close_atol_rtol_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "slot_argmax_agreement": float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
        "record_agreement": sum(a == b for a, b in zip(records(actual), records(expected))) / len(actual),
    }


def prepare(args):
    from PIL import Image

    out, verified = args.out.resolve(), args.verification.resolve()
    out.mkdir(parents=True, exist_ok=True)
    annotations = sorted(args.annotations.glob("*.json"))
    assert annotations
    identities = set()
    with (out / "images.u8").open("wb") as stream:
        for annotation in annotations:
            with Image.open(args.images / f"{annotation.stem}.jpg") as image:
                gray = np.asarray(image.convert("L").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR))
            stream.write(gray.tobytes())
            identities.add(hashlib.sha256(normalize(gray).tobytes()).hexdigest())
    expected = json.loads((verified / "input-identities.json").read_text())
    assert identities == set(expected["evaluation"]), "evaluation inputs changed"
    assert not identities.intersection(expected["calibration"]), "calibration/evaluation overlap"
    images = np.memmap(out / "images.u8", dtype=np.uint8, mode="r", shape=(len(annotations), HEIGHT, WIDTH))
    assert normalize(images[:8]).tobytes() == (verified / "eval-smoke.f32").read_bytes()
    manifest = {"schema": "volvoxai.receipt-backend-corpus/v1", "samples": len(annotations),
                "input_shape": [1, 1, HEIGHT, WIDTH], "output_shape": [1, 16, 11],
                "images_sha256": digest(out / "images.u8"), "normalized_inputs_match_native_evaluation": True,
                "calibration_eval_overlap": 0, "packages": {}}
    with np.load(verified / "logits.npz") as reference:
        for variant in VARIANTS:
            target = out / variant
            target.mkdir(exist_ok=True)
            for name in ("graph.json", "model.safetensors"):
                shutil.copyfile(verified / variant / name, target / name)
            logits = np.load(verified / "wasm-package-native-logits.npy") if variant == "ptq-wasm" else reference[f"volvoxai_{variant}"]
            assert logits.shape == (len(annotations), 16, 11)
            (target / "reference.f32").write_bytes(logits.astype(np.float32).tobytes())
            manifest["packages"][variant] = {name: digest(target / name)
                for name in ("graph.json", "model.safetensors", "reference.f32")}
    save_json(out / "corpus.json", manifest)
    print(json.dumps(manifest, indent=2))


def attest(report, backend):
    assert report.backend == backend, report.to_dict()
    assert report.route is not None and report.route.attested and report.route.provider == backend, report.to_dict()
    assert report.route.active_nodes == report.route.selected_nodes, report.to_dict()
    assert report.route.missing_nodes == report.route.fallback_nodes == 0, report.to_dict()
    assert not (report.fallback and report.fallback.operator_fallback_used), report.to_dict()
    assert not (report.compilation and report.compilation.tier_fallback_used), report.to_dict()


def run(args):
    corpus = args.corpus.resolve()
    manifest = json.loads((corpus / "corpus.json").read_text())
    count = min(args.limit or manifest["samples"], manifest["samples"])
    images = np.memmap(corpus / "images.u8", dtype=np.uint8, mode="r", shape=(manifest["samples"], HEIGHT, WIDTH))
    result = {"schema": "volvoxai.receipt-native-backend-verification/v1", "backend": args.backend,
              "profile": args.profile, "samples": count, "cpu_threads": 1,
              "library_sha256": digest(vx.find_library(args.profile)),
              "corpus_sha256": digest(corpus / "corpus.json"), "variants": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        host, entry = None, {"status": "failed", "stage": "load"}
        result["variants"][variant] = entry
        try:
            package = corpus / variant
            for name, checksum in manifest["packages"][variant].items():
                assert digest(package / name) == checksum, name
            host = vx.open_library(args.profile)
            result["platform"] = vx.VxPlatformServiceClient(host).get_platform_info(pb.Empty()).to_dict()
            client = vx.VxInferenceServiceClient(host)
            runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=pb.ModelPackage(graph_document=(package / "graph.json").read_bytes(),
                                        weight_shards=[(package / "model.safetensors").read_bytes()])))
            entry["stage"] = "compile"
            compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=[args.backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            attest(compiled.report, args.backend)
            entry["compilation"] = compiled.report.to_dict()
            context = client.create_execution_context(pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id))
            entry["stage"] = "execute"
            outputs = []
            started = time.perf_counter()
            for index in range(count):
                values = normalize(images[index])
                execution = client.execute(pb.ExecuteRequest(context_id=context.context_id,
                    inputs=[pb.Tensor(name="input0", shape=[1, 1, HEIGHT, WIDTH],
                                      dtype=pb.DataType.DATA_TYPE_F32, inline=values.tobytes())]))
                try:
                    attest(execution.report, args.backend)
                    if index == 0:
                        entry["execution"] = execution.report.to_dict()
                    deadline = time.monotonic() + 60
                    while True:
                        info = client.get_result(pb.ResultRef(result_id=execution.result_id))
                        if info.state != pb.ResultState.RESULT_STATE_PENDING:
                            assert info.state == pb.ResultState.RESULT_STATE_READY, info.to_dict()
                            break
                        assert time.monotonic() < deadline, "result timed out"
                        time.sleep(.001)
                    output = client.read_output(pb.ReadOutputRequest(
                        result_id=execution.result_id, name="slot_logits")).tensor
                    assert output.dtype == pb.DataType.DATA_TYPE_F32 and list(output.shape) == [1, 16, 11]
                    logits = np.frombuffer(output.inline, dtype=np.float32).copy().reshape(16, 11)
                    assert np.isfinite(logits).all()
                    outputs.append(logits)
                finally:
                    client.release_result(pb.ResultRef(result_id=execution.result_id))
                if (index + 1) % 250 == 0:
                    entry["completed_samples"] = index + 1
                    entry["seconds"] = time.perf_counter() - started
                    save_json(args.out, result)
                    print(f"{args.backend}/{args.profile}/{variant}: {index + 1}/{count}", flush=True)
            produced = np.stack(outputs)
            reference = np.fromfile(package / "reference.f32", dtype=np.float32).reshape(-1, 16, 11)[:count]
            entry.update(status="executed", stage="complete", seconds=time.perf_counter() - started,
                         comparison=compare(produced, reference))
            args.out.with_name(f"{args.out.stem}-{variant}-logits.f32").write_bytes(produced.tobytes())
            print(variant, json.dumps(entry["comparison"]), flush=True)
        except Exception as error:
            entry["error"] = str(error)
            print(variant, type(error).__name__, str(error), flush=True)
        finally:
            if host is not None:
                host.close()
            save_json(args.out, result)
    return 0 if all(item["status"] == "executed" for item in result["variants"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for name in ("verification", "images", "annotations", "out"):
        preparation.add_argument(f"--{name}", type=Path, required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--corpus", type=Path, required=True)
    execution.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl"), required=True)
    execution.add_argument("--profile", choices=("inference", "full"), required=True)
    execution.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    execution.add_argument("--limit", type=int, default=0)
    execution.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return 0
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
