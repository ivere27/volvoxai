#!/usr/bin/env python3
"""One-step public native Trainer smoke with an independent closed-form oracle."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("cpu", "vulkan", "opengl", "metal", "cuda")
INITIAL_WEIGHT = np.asarray([[0.2, -0.4], [0.1, 0.3]], dtype=np.float32)
INPUT = np.asarray([[1.0, 0.0]], dtype=np.float32)
LEARNING_RATE = 0.1
SOFTWARE_ADAPTER = re.compile(
    r"\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software rasterizer|"
    r"microsoft basic render|cpu)\b",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    raise RuntimeError(f"native GPU training smoke: {message}")


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def graph_document() -> dict[str, Any]:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"dtype": "float32", "shape": [1, 2]}},
        "nodes": [
            {
                "id": "projection",
                "opType": "Linear",
                "inputs": {"input": "x", "weight": "parameter"},
                "outputs": {
                    "out": {
                        "tensor": "logits",
                        "dtype": "float32",
                        "shape": [1, 2],
                    }
                },
                "params": {"weight_layout": "din_dout"},
            }
        ],
        "outputs": ["logits"],
    }


def expected_update() -> tuple[np.ndarray, np.ndarray, float]:
    logits = INPUT @ INITIAL_WEIGHT
    shifted = logits[0] - np.max(logits[0])
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
    loss = -math.log(float(probabilities[0]))
    gradient = probabilities.copy()
    gradient[0] -= 1.0
    updated = INITIAL_WEIGHT - LEARNING_RATE * np.outer(INPUT[0], gradient)
    return updated.astype(np.float32), (INPUT @ updated).astype(np.float32), loss


def physical_identity(backend: str, stdout: str, stderr: str) -> dict[str, str]:
    if backend == "vulkan":
        matches = re.findall(
            r"^\[VolvoxAI GPU\] Vulkan Compute initialized successfully! "
            r"Device: ([^;\r\n]+); packed INT8 dot: (enabled|unavailable)$",
            stdout,
            re.MULTILINE,
        )
        if len(matches) != 1 or matches[0][0].strip().lower() in {"unknown", "gpu"}:
            fail(f"Vulkan published {len(matches)} valid physical device identities")
        return {
            "backend": backend,
            "device": matches[0][0].strip(),
            "packedInt8Dot": matches[0][1],
        }
    if backend == "opengl":
        matches = re.findall(
            r"^\[VolvoxAI GPU\] OpenGL Compute initialized: "
            r"([^\r\n]+?) / ([^\r\n]+?) / ([^\r\n]+)$",
            stdout,
            re.MULTILINE,
        )
        if len(matches) != 1 or any(
            value.strip().lower() == "unknown" for value in matches[0][:2]
        ):
            fail(f"OpenGL published {len(matches)} valid physical device identities")
        return {
            "backend": backend,
            "vendor": matches[0][0].strip(),
            "device": matches[0][1].strip(),
            "version": matches[0][2].strip(),
        }
    if backend == "metal":
        matches = re.findall(
            r"^\[VolvoxAI GPU\] Metal initialized successfully on: ([^\r\n]+)$",
            stdout,
            re.MULTILINE,
        )
        if len(matches) != 1 or matches[0].strip().lower() in {"unknown", "gpu"}:
            fail(f"Metal published {len(matches)} valid physical device identities")
        return {"backend": backend, "device": matches[0].strip()}
    if backend == "cuda":
        matches = re.findall(
            r"^\[CUDA\] device ([0-9]+): ([^\r\n]+?) "
            r"\(compute ([0-9]+\.[0-9]+)\)$",
            stderr,
            re.MULTILINE,
        )
        if not matches or len(set(matches)) != 1:
            fail("CUDA did not publish one consistent physical device identity")
        index, device, compute = matches[0]
        executable = shutil.which("nvidia-smi")
        if executable is None:
            fail("physical CUDA evidence requires nvidia-smi")
        completed = run(
            [
                executable,
                f"--id={index}",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            cwd=ROOT,
        )
        devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if len(devices) != 1 or "," not in devices[0]:
            fail("nvidia-smi did not publish exactly the selected CUDA adapter")
        smi_name, driver = (part.strip() for part in devices[0].rsplit(",", 1))
        if smi_name.lower() != device.strip().lower():
            fail(f"CUDA selected {device!r}, but nvidia-smi index {index} reports {smi_name!r}")
        return {
            "backend": backend,
            "deviceIndex": index,
            "device": device.strip(),
            "compute": compute,
            "driver": driver,
            "initializations": str(len(matches)),
        }
    if backend == "cpu":
        return {"backend": backend, "device": "host-cpu"}
    fail(f"unsupported backend {backend!r}")


def write_fixture(directory: Path) -> dict[str, Path]:
    package = directory / "package"
    package.mkdir(parents=True)
    graph = package / "graph.json"
    weights = package / "model.safetensors"
    input_path = directory / "x.f32"
    targets = directory / "targets.i32"
    graph.write_text(f"{json.dumps(graph_document(), indent=2)}\n", encoding="utf-8")
    save_file({"parameter": INITIAL_WEIGHT}, weights)
    INPUT.tofile(input_path)
    np.asarray([0], dtype=np.int32).tofile(targets)
    return {
        "package": package,
        "graph": graph,
        "weights": weights,
        "input": input_path,
        "targets": targets,
    }


def backend_flag(backend: str) -> str:
    return f"--{backend}"


def validate_strict_runtime_report(report: dict[str, Any], backend: str) -> None:
    if (
        report.get("schema") != "volvoxai.native-cli.run-report"
        or report.get("version") != 1
    ):
        fail(f"{backend} reload omitted the native CLI run report")
    request = report.get("request", {})
    if request.get("backend") != backend:
        fail(f"{backend} reload report does not match the requested backend")

    expected_device = "host" if backend == "cpu" else f"builtin:{backend}"
    expected_stages = {
        "compile": "OPERATION_STAGE_COMPILE",
        "execute": "OPERATION_STAGE_EXECUTE",
    }
    for operation, stage in expected_stages.items():
        evidence = report.get(operation, {})
        route = evidence.get("route", {})
        active_nodes = route.get("activeNodes")
        if (
            evidence.get("status") != "NATIVE_STATUS_OK"
            or evidence.get("stage") != stage
            or evidence.get("backend") != backend
            or evidence.get("device") != expected_device
            or route.get("provider") != backend
            or route.get("builtin") is not True
            or route.get("attested") is not True
            or not isinstance(active_nodes, int)
            or isinstance(active_nodes, bool)
            or active_nodes <= 0
            or route.get("selectedNodes") != active_nodes
            or route.get("fallbackNodes") != 0
            or route.get("missingNodes") != 0
        ):
            fail(f"{backend} reload did not attest a strict {operation} route")

    if report.get("outputs") != [
        {
            "name": "logits",
            "dtype": "DATA_TYPE_F32",
            "byteLength": 8,
            "shape": [1, 2],
        }
    ]:
        fail(f"{backend} reload report does not describe the expected logits")


def train_backend(
    backend: str,
    *,
    native_full: Path,
    native_inference: Path,
    fixture: dict[str, Path],
    directory: Path,
    require_physical: bool,
) -> dict[str, Any]:
    output_weights = directory / f"trained-{backend}.safetensors"
    completed = run(
        [
            str(native_full),
            "train",
            str(fixture["package"]),
            "--targets",
            str(fixture["targets"]),
            "--logits",
            "logits",
            "--trainable",
            "parameter",
            "--output-weights",
            str(output_weights),
            "--input",
            f"x={fixture['input']}",
            "--optimizer",
            "sgd",
            "--learning-rate",
            str(LEARNING_RATE),
            "--microbatches",
            "1",
            backend_flag(backend),
        ],
        cwd=ROOT,
    )
    required_lines = (
        f"Training backend={backend} optimizer=sgd",
        f"update=yes backend={backend}",
        "Published weight revision 2 and exported 1 shard(s).",
    )
    for line in required_lines:
        if line not in completed.stdout:
            fail(f"{backend} training output is missing {line!r}")
    step_matches = re.findall(
        r"^microbatch=([0-9]+) optimizer_step=([0-9]+) "
        r"loss=([^\s]+) accumulated=([0-9]+) update=(yes|no) "
        r"backend=([a-z]+)$",
        completed.stdout,
        re.MULTILINE,
    )
    if len(step_matches) != 1:
        fail(f"{backend} published {len(step_matches)} parseable training steps")
    microbatch, optimizer_step, loss_text, accumulated, update, step_backend = (
        step_matches[0]
    )
    try:
        actual_loss = float(loss_text)
    except ValueError as error:
        fail(f"{backend} published an invalid loss {loss_text!r}: {error}")
    expected_weight, expected_logits, expected_loss = expected_update()
    if (
        microbatch != "1"
        or optimizer_step != "1"
        or accumulated != "1"
        or update != "yes"
        or step_backend != backend
        or not math.isfinite(actual_loss)
        or abs(actual_loss - expected_loss) > 2e-5
    ):
        fail(f"{backend} training step metadata or loss differs from the oracle")
    identity = physical_identity(backend, completed.stdout, completed.stderr)
    if require_physical and backend == "cpu":
        fail("--require-physical cannot be used with the CPU backend")
    if require_physical:
        description = " ".join(identity.values())
        if SOFTWARE_ADAPTER.search(description):
            fail(f"{backend} selected a software adapter ({description})")
        required = os.environ.get("VOLVOXAI_PARITY_GPU_ADAPTER", "").strip()
        if required and required.lower() not in description.lower():
            fail(f"{backend} adapter '{description}' does not match '{required}'")

    tensors = load_file(output_weights)
    if set(tensors) != {"parameter"}:
        fail(f"{backend} exported tensors {sorted(tensors)} instead of ['parameter']")
    np.testing.assert_allclose(tensors["parameter"], expected_weight, rtol=2e-5, atol=2e-6)

    logits_path = directory / f"logits-{backend}.f32"
    report_path = directory / f"inference-{backend}.json"
    inference_completed = run(
        [
            str(native_inference),
            "run",
            str(fixture["package"]),
            "--weights",
            str(output_weights),
            "--input",
            f"x={fixture['input']}",
            "--output",
            f"logits={logits_path}",
            "--report-json",
            str(report_path),
            backend_flag(backend),
        ],
        cwd=ROOT,
    )
    logits = np.fromfile(logits_path, dtype=np.float32).reshape(1, 2)
    np.testing.assert_allclose(logits, expected_logits, rtol=2e-5, atol=2e-6)
    inference_report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_strict_runtime_report(inference_report, backend)
    reload_identity = physical_identity(
        backend, inference_completed.stdout, inference_completed.stderr
    )
    if backend != "cpu" and reload_identity.get("device") != identity.get("device"):
        fail(f"{backend} training and reload selected different devices")
    return {
        "backend": backend,
        "status": "pass",
        "optimizer": "sgd",
        "optimizerSteps": 1,
        "loss": expected_loss,
        "lossAbsError": abs(actual_loss - expected_loss),
        "weightMaxAbsError": float(np.max(np.abs(tensors["parameter"] - expected_weight))),
        "reloadMaxAbsError": float(np.max(np.abs(logits - expected_logits))),
        "physicalDevice": identity,
        "reloadPhysicalDevice": reload_identity,
        "artifacts": {
            "weights": output_weights.name,
            "inferenceReport": report_path.name,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-full", type=Path, default=ROOT / "native" / "volvoxai")
    parser.add_argument("--native-inference", type=Path, default=ROOT / "native" / "volvoxai-lite")
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--require-physical", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = args.backend or ["cpu"]
    native_full = args.native_full.resolve()
    native_inference = args.native_inference.resolve()
    for executable in (native_full, native_inference):
        if not executable.is_file():
            fail(f"missing native executable {executable}")

    if args.work_dir:
        directory = args.work_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        scope = nullcontext(directory)
    else:
        scope = tempfile.TemporaryDirectory(prefix="volvoxai-native-training-")

    with scope as selected:
        directory = Path(selected)
        fixture = write_fixture(directory)
        cases = [
            train_backend(
                backend,
                native_full=native_full,
                native_inference=native_inference,
                fixture=fixture,
                directory=directory,
                require_physical=args.require_physical,
            )
            for backend in backends
        ]
        report = {
            "schema": "volvoxai.native-training-smoke",
            "version": 1,
            "status": "pass",
            "oracle": "closed-form-linear-cross-entropy-sgd",
            "cases": cases,
        }
        rendered = f"{json.dumps(report, indent=2)}\n"
        if args.report:
            args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
            args.report.resolve().write_text(rendered, encoding="utf-8")
        print(rendered, end="")


if __name__ == "__main__":
    main()
