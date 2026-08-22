#!/usr/bin/env python3
"""Audit TinyReceipt VQA component batching against independent ORT B=1 lanes.

The companion MJS worker proves Runtime dispatch and writes every output as a
raw typed array. This orchestrator authors distinct encoder/decoder lanes,
runs the original producer ONNX separately at B=1 for each lane, and compares
all outputs lane by lane. It does not exercise or make a batching claim about
TinyReceiptSplitSession, whose autoregressive API remains serialized B=1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort


SCHEMA = "volvoxai.tiny-receipt-vqa-runtime-batch-audit/v2"
INPUT_SCHEMA = "volvoxai.tiny-receipt-vqa-runtime-batch-inputs/v1"
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 672
HEADS = 8
HEAD_WIDTH = 40
MEMORY_LENGTH = 226
QUESTION_LENGTH = 16
PAST_LENGTH = 1


class AuditFailure(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AuditFailure(f"{path} must contain a JSON object")
    return value


def _variant_onnx(manifest: Mapping[str, Any], source: Path, role: str) -> Path:
    producer_key = manifest.get("variant", {}).get("producer_key")
    if producer_key == "fp32":
        filename = f"{role}_model.onnx"
    elif producer_key == "int8_w8a8":
        filename = f"{role}_model_int8.onnx"
    else:
        raise AuditFailure(f"unsupported package producer key {producer_key!r}")
    path = source / filename
    if not path.is_file():
        raise AuditFailure(f"source ONNX is missing: {path}")
    return path


def _encoder_lanes(count: int) -> list[dict[str, np.ndarray]]:
    lanes = []
    for lane in range(count):
        image = np.zeros((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        image += np.float32(lane * 0.03125)
        question_ids = (
            np.arange(QUESTION_LENGTH, dtype=np.int64) + 4 + lane * 17
        ) % 256
        lanes.append({
            "image": image,
            "question_ids": question_ids.reshape(1, QUESTION_LENGTH),
            "family_ids": np.asarray([lane % 8], dtype=np.int64),
            "question_position_ids": np.arange(
                QUESTION_LENGTH, dtype=np.int64
            ).reshape(1, QUESTION_LENGTH),
        })
    return lanes


def _deterministic_cache(
    lane: int,
    cache_index: int,
    length: int,
) -> np.ndarray:
    elements = HEADS * length * HEAD_WIDTH
    values = (
        ((np.arange(elements, dtype=np.int64) * (cache_index + 3)) % 97) - 48
    ).astype(np.float32)
    values *= np.float32(0.002)
    values += np.float32(lane * 0.03125 + cache_index * 0.0078125)
    return values.reshape(1, HEADS, length, HEAD_WIDTH)


def _decoder_lanes(count: int) -> list[dict[str, np.ndarray]]:
    lanes = []
    for lane in range(count):
        values: dict[str, np.ndarray] = {
            "decoder_input_ids": np.asarray([[1 + lane]], dtype=np.int64),
            "position_ids": np.asarray([0], dtype=np.int64),
            "family_ids": np.asarray([lane % 8], dtype=np.int64),
            "memory_padding_mask": np.zeros(
                (1, MEMORY_LENGTH), dtype=np.bool_
            ),
            "past_padding_mask": np.ones((1, PAST_LENGTH), dtype=np.bool_),
        }
        for index in range(4):
            values[f"cross_k_{index}"] = _deterministic_cache(
                lane, index * 2, MEMORY_LENGTH
            )
            values[f"cross_v_{index}"] = _deterministic_cache(
                lane, index * 2 + 1, MEMORY_LENGTH
            )
            values[f"past_k_{index}"] = _deterministic_cache(
                lane, index * 2 + 8, PAST_LENGTH
            )
            values[f"past_v_{index}"] = _deterministic_cache(
                lane, index * 2 + 9, PAST_LENGTH
            )
        lanes.append(values)
    return lanes


def _runtime_array(value: np.ndarray) -> np.ndarray:
    if value.dtype == np.dtype(np.float32):
        return np.ascontiguousarray(value)
    if value.dtype == np.dtype(np.bool_):
        return np.ascontiguousarray(value.astype(np.int32))
    if value.dtype == np.dtype(np.int64):
        return np.ascontiguousarray(value.astype(np.int32))
    raise AuditFailure(f"unsupported runtime fixture dtype {value.dtype}")


def _runtime_dtype(value: np.ndarray) -> str:
    if value.dtype == np.dtype(np.float32):
        return "float32"
    if value.dtype == np.dtype(np.int32):
        return "int32"
    raise AuditFailure(f"unsupported runtime dtype {value.dtype}")


def _write_fixture(
    directory: Path,
    lanes: Sequence[Mapping[str, np.ndarray]],
    graph_inputs: Mapping[str, str],
    graph_outputs: Mapping[str, str],
) -> tuple[Path, list[dict[str, np.ndarray]]]:
    authored_lanes = []
    runtime_lanes = []
    for lane_index, semantic in enumerate(lanes):
        authored: dict[str, Any] = {}
        runtime_semantic: dict[str, np.ndarray] = {}
        for semantic_name, graph_name in graph_inputs.items():
            if semantic_name not in semantic:
                raise AuditFailure(f"fixture is missing semantic input {semantic_name}")
            value = _runtime_array(semantic[semantic_name])
            runtime_semantic[semantic_name] = value
            path = directory / f"input-lane{lane_index}-{graph_name}.raw"
            value.tofile(path)
            authored[graph_name] = {
                "path": str(path),
                "dtype": _runtime_dtype(value),
                "shape": list(value.shape),
            }
        authored_lanes.append(authored)
        runtime_lanes.append(runtime_semantic)
    document = {
        "schema": INPUT_SCHEMA,
        "outputs": list(graph_outputs.values()),
        "lanes": authored_lanes,
    }
    path = directory / "inputs.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path, runtime_lanes


def _ort_inputs(role: str, lane: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if role == "encoder":
        names = ("image", "question_ids", "family_ids")
    else:
        names = (
            "decoder_input_ids", "position_ids", "family_ids",
            "memory_padding_mask", "past_padding_mask",
            *(f"cross_{kind}_{layer}" for layer in range(4) for kind in ("k", "v")),
            *(f"past_{kind}_{layer}" for layer in range(4) for kind in ("k", "v")),
        )
    return {name: lane[name] for name in names}


def _run_ort_lanes(
    model_path: Path,
    role: str,
    lanes: Sequence[Mapping[str, np.ndarray]],
) -> tuple[list[dict[str, np.ndarray]], str]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [value.name for value in session.get_outputs()]
    outputs = []
    for lane in lanes:
        values = session.run(output_names, _ort_inputs(role, lane))
        outputs.append({
            name: np.ascontiguousarray(value)
            for name, value in zip(output_names, values)
        })
    return outputs, session.get_providers()[0]


def _load_runtime_output(descriptor: Mapping[str, Any]) -> np.ndarray:
    dtype = {"float32": np.float32, "int32": np.int32}.get(
        descriptor.get("dtype")
    )
    if dtype is None:
        raise AuditFailure(f"unsupported Runtime output dtype {descriptor.get('dtype')!r}")
    shape = tuple(descriptor.get("shape", ()))
    value = np.fromfile(descriptor["path"], dtype=dtype)
    if value.size != int(np.prod(shape, dtype=np.int64)):
        raise AuditFailure("Runtime output byte count does not match its shape")
    return value.reshape(shape)


def _compare_route(
    route: Mapping[str, Any],
    ort_outputs: Sequence[Mapping[str, np.ndarray]],
    graph_outputs: Mapping[str, str],
    *,
    atol: float,
    rtol: float,
) -> Mapping[str, Any]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    all_outputs_within_tolerance = True
    lane_reports = []
    for lane_index, (descriptors, reference) in enumerate(
        zip(route["outputs"], ort_outputs)
    ):
        output_reports = {}
        for semantic_name, graph_name in graph_outputs.items():
            if semantic_name not in reference or graph_name not in descriptors:
                raise AuditFailure(
                    f"lane {lane_index} output mapping {semantic_name}->{graph_name} is missing"
                )
            actual = _load_runtime_output(descriptors[graph_name])
            expected = reference[semantic_name]
            if actual.shape != expected.shape:
                raise AuditFailure(
                    f"lane {lane_index} output {semantic_name} shape differs: "
                    f"Runtime {actual.shape}, ORT {expected.shape}"
                )
            if expected.dtype.kind == "f":
                if not (
                    np.all(np.isfinite(expected))
                    and np.all(np.isfinite(actual))
                ):
                    raise AuditFailure(
                        f"lane {lane_index} output {semantic_name} is non-finite"
                    )
                absolute = np.abs(expected - actual)
                max_abs = float(np.max(absolute, initial=0.0))
                max_rel = float(np.max(
                    absolute / np.maximum(np.abs(expected), np.float32(1.0e-6)),
                    initial=0.0,
                ))
                matched = bool(np.allclose(
                    expected, actual, atol=atol, rtol=rtol
                ))
            else:
                max_abs = 0.0
                max_rel = 0.0
                matched = bool(np.array_equal(
                    expected.astype(np.int64), actual.astype(np.int64)
                ))
                if not matched:
                    raise AuditFailure(
                        f"lane {lane_index} integer output {semantic_name} "
                        "differs from independent ORT B1"
                    )
            all_outputs_within_tolerance = (
                all_outputs_within_tolerance and matched
            )
            maximum_absolute = max(maximum_absolute, max_abs)
            maximum_relative = max(maximum_relative, max_rel)
            output_reports[semantic_name] = {
                "runtimeTensor": graph_name,
                "shape": list(actual.shape),
                "runtimeDtype": str(actual.dtype),
                "ortDtype": str(expected.dtype),
                "maxAbsoluteDifference": max_abs,
                "maxRelativeDifference": max_rel,
                "runtimeSha256": descriptors[graph_name]["sha256"],
                "ortSha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                "status": "matched" if matched else "outside_tolerance",
                **(
                    {
                        "argmaxMatched": bool(np.array_equal(
                            np.argmax(expected, axis=-1),
                            np.argmax(actual, axis=-1),
                        ))
                    }
                    if semantic_name == "logits"
                    else {}
                ),
            }
        lane_reports.append({"lane": lane_index, "outputs": output_reports})
    return {
        "status": (
            "passed" if all_outputs_within_tolerance else "outside_tolerance"
        ),
        "toleranceGatePassed": all_outputs_within_tolerance,
        "reference": "independent original producer ONNX Runtime B=1 per lane",
        "absoluteTolerance": atol,
        "relativeTolerance": rtol,
        "maximumAbsoluteDifference": maximum_absolute,
        "maximumRelativeDifference": maximum_relative,
        "lanes": lane_reports,
    }


def _public_runtime_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Runtime proof without local paths or output fingerprints."""

    public = json.loads(json.dumps(report))
    for key in ("api", "wasm"):
        public.get("environment", {}).pop(key, None)
    public.get("package", {}).pop("path", None)
    for route_name in ("independent", "scheduled"):
        for lane in public.get(route_name, {}).get("outputs", []):
            for descriptor in lane.values():
                descriptor.pop("path", None)
                descriptor.pop("sha256", None)
    public["privacy"] = {
        "machinePathsIncluded": False,
        "outputDigestsIncluded": False,
        "rawOutputFilesWritten": True,
    }
    return public


def _public_parity_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep numerical gates while dropping stable per-output fingerprints."""

    return {
        key: value
        for key, value in report.items()
        if key != "lanes"
    }


def _public_fixture_summary(
    lanes: Sequence[Mapping[str, np.ndarray]], role: str,
) -> dict[str, Any]:
    return {
        "Q": QUESTION_LENGTH if role == "encoder" else None,
        "M": MEMORY_LENGTH,
        "P": PAST_LENGTH if role == "decoder" else None,
        "lanes": [
            {
                "laneId": f"lane-{lane_index + 1:02d}",
                "inputs": {
                    semantic_name: {
                        "shape": list(runtime_value.shape),
                        "dtype": str(runtime_value.dtype),
                    }
                    for semantic_name, value in lane.items()
                    for runtime_value in [_runtime_array(value)]
                },
            }
            for lane_index, lane in enumerate(lanes)
        ],
    }


def _require_ort_tolerance(
    independent: Mapping[str, Any], scheduled: Mapping[str, Any],
) -> None:
    failed = [
        name
        for name, comparison in (
            ("independent", independent),
            ("scheduled", scheduled),
        )
        if comparison.get("toleranceGatePassed") is not True
    ]
    if failed:
        raise AuditFailure(
            "ORT reference tolerance gate failed for " + ", ".join(failed)
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--role", choices=("encoder", "decoder"), required=True)
    parser.add_argument("--backend", choices=("cpu-js", "wasm", "webgpu"), required=True)
    # The JavaScript worker validates this value against the generated
    # proto-backed executionModes contract exported by the selected API.
    parser.add_argument("--mode", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--max-batch-delay-ms",
        type=float,
        default=10.0,
        help="Runtime scheduler collection delay in milliseconds (default: 10)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--api", type=Path, default=Path("dist/0.4.0/volvoxai.js"))
    parser.add_argument("--wasm", type=Path, default=Path("dist/0.4.0/volvoxai.wasm"))
    parser.add_argument(
        "--node-tool",
        type=Path,
        default=Path(__file__).with_name("benchmark_runtime_batches.mjs"),
    )
    parser.add_argument("--node", default="node")
    parser.add_argument("--deno", default="deno")
    parser.add_argument(
        "--adapter",
        choices=("default", "high-performance", "low-power"),
        default="high-performance",
    )
    parser.add_argument("--require-adapter")
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--include-private-artifacts",
        action="store_true",
        help="include local paths and stable fixture/output digests in the report",
    )
    arguments = parser.parse_args(argv)
    for name in ("concurrency", "repeat"):
        if getattr(arguments, name) < 1:
            parser.error(f"--{name} must be >= 1")
    if arguments.warmup < 0:
        parser.error("--warmup must be >= 0")
    if (not math.isfinite(arguments.max_batch_delay_ms)
            or not 0 <= arguments.max_batch_delay_ms <= 60_000):
        parser.error("--max-batch-delay-ms must be between 0 and 60000")
    if arguments.backend == "wasm" and not arguments.wasm.is_file():
        parser.error(f"--wasm does not exist: {arguments.wasm}")
    if arguments.backend == "webgpu" and not arguments.require_adapter:
        parser.error("--require-adapter is required for physical WebGPU")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    package = arguments.package.resolve()
    source = arguments.source.resolve()
    manifest_path = package / "package_manifest.json"
    manifest = _read_json(manifest_path)
    graph_contract = manifest.get("graphs", {}).get(arguments.role)
    if not isinstance(graph_contract, Mapping):
        raise AuditFailure(f"package does not expose {arguments.role}")
    graph_inputs = graph_contract.get("inputs")
    graph_outputs = graph_contract.get("outputs")
    if not isinstance(graph_inputs, Mapping) or not isinstance(graph_outputs, Mapping):
        raise AuditFailure("package graph input/output mapping is malformed")
    lanes = (
        _encoder_lanes(arguments.concurrency)
        if arguments.role == "encoder"
        else _decoder_lanes(arguments.concurrency)
    )
    model_path = _variant_onnx(manifest, source, arguments.role)

    with tempfile.TemporaryDirectory(
        prefix="volvoxai-vqa-runtime-batch-"
    ) as temporary:
        work = Path(temporary)
        input_path, _runtime_inputs = _write_fixture(
            work, lanes, graph_inputs, graph_outputs
        )
        node_report_path = work / "runtime-report.json"
        output_directory = work / "runtime-outputs"
        worker = [
            str(arguments.node_tool.resolve()),
            "--package", str(package),
            "--role", arguments.role,
            "--inputs", str(input_path),
            "--output-directory", str(output_directory),
            "--backend", arguments.backend,
            "--mode", arguments.mode,
            "--concurrency", str(arguments.concurrency),
            "--max-batch-delay-ms", str(arguments.max_batch_delay_ms),
            "--warmup", str(arguments.warmup),
            "--repeat", str(arguments.repeat),
            "--api", str(arguments.api.resolve()),
            "--report", str(node_report_path),
            # The wrapper needs temporary output locators for ORT comparison;
            # its default persisted report redacts them again below.
            "--include-private-artifacts",
        ]
        if arguments.backend == "webgpu":
            command = [
                arguments.deno,
                "run",
                "--unstable-webgpu",
                "--allow-read",
                f"--allow-write={work}",
                "--allow-env=DENO_WEBGPU_BACKEND,VOLVOXAI_ROW_DEBUG",
                "--allow-ffi",
                *worker,
                "--adapter", arguments.adapter,
                "--require-adapter", arguments.require_adapter,
            ]
        else:
            command = [arguments.node, *worker]
        if arguments.backend == "wasm":
            command.extend(["--wasm", str(arguments.wasm.resolve())])
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AuditFailure(
                "Runtime batch worker failed\n"
                f"command: {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        runtime_report = _read_json(node_report_path)
        ort_outputs, provider = _run_ort_lanes(
            model_path, arguments.role, lanes
        )
        independent_parity = _compare_route(
            runtime_report["independent"],
            ort_outputs,
            graph_outputs,
            atol=arguments.atol,
            rtol=arguments.rtol,
        )
        scheduled_parity = _compare_route(
            runtime_report["scheduled"],
            ort_outputs,
            graph_outputs,
            atol=arguments.atol,
            rtol=arguments.rtol,
        )

        private_fixture = {
            "Q": QUESTION_LENGTH if arguments.role == "encoder" else None,
            "M": MEMORY_LENGTH,
            "P": PAST_LENGTH if arguments.role == "decoder" else None,
            "laneInputs": [
                {
                    semantic_name: {
                        "shape": list(runtime_value.shape),
                        "dtype": str(runtime_value.dtype),
                        "sha256": hashlib.sha256(
                            runtime_value.tobytes()
                        ).hexdigest(),
                    }
                    for semantic_name, value in lane.items()
                    for runtime_value in [_runtime_array(value)]
                }
                for lane in lanes
            ],
        }
        persisted_runtime = (
            runtime_report
            if arguments.include_private_artifacts
            else _public_runtime_report(runtime_report)
        )
        report = {
            "schema": SCHEMA,
            "scope": (
                "encoder/decoder component proof only; TinyReceiptSplitSession "
                "remains serialized B=1"
            ),
            **({"command": command} if arguments.include_private_artifacts else {}),
            "environment": {
                "python": platform.python_version(),
                "runtime": runtime_report["environment"]["runtime"],
                "onnxruntime": ort.__version__,
                "onnxruntimeProvider": provider,
                "platform": platform.platform(),
                "affinity": sorted(__import__("os").sched_getaffinity(0)),
                "backend": arguments.backend,
            },
            "source": {
                "modelSha256": _sha256(model_path),
                **(
                    {"path": str(source), "model": str(model_path)}
                    if arguments.include_private_artifacts
                    else {}
                ),
            },
            "package": persisted_runtime["package"],
            "role": arguments.role,
            "mode": arguments.mode,
            "scheduler": runtime_report["scheduler"],
            "concurrency": arguments.concurrency,
            "distinctLaneFixture": True,
            "fixture": (
                private_fixture
                if arguments.include_private_artifacts
                else _public_fixture_summary(lanes, arguments.role)
            ),
            "runtime": persisted_runtime,
            "ortReference": {
                "execution": "independent B=1 per lane, threads=1",
                "independentRouteParity": (
                    independent_parity
                    if arguments.include_private_artifacts
                    else _public_parity_summary(independent_parity)
                ),
                "scheduledRouteParity": (
                    scheduled_parity
                    if arguments.include_private_artifacts
                    else _public_parity_summary(scheduled_parity)
                ),
            },
            "privacy": {
                "machine_paths_included": arguments.include_private_artifacts,
                "fixture_digests_included": arguments.include_private_artifacts,
                "output_digests_included": arguments.include_private_artifacts,
            },
            "status": (
                "passed"
                if independent_parity["toleranceGatePassed"]
                and scheduled_parity["toleranceGatePassed"]
                else "failed_ort_reference_tolerance"
            ),
        }
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _require_ort_tolerance(independent_parity, scheduled_parity)
    print(arguments.report.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
