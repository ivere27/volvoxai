#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_matrix  # noqa: E402


DEVICE_NAME = "NVIDIA GeForce RTX 3090"
DRIVER = "999.1"
REPO = Path(__file__).resolve().parents[3]


def case_result() -> dict:
    return {
        "record": "native-dynamic-batch-case",
        "caseId": "receipt-fp32-b4-vulkan",
        "backend": "vulkan",
        "inputMode": "receipt",
        "status": "pass",
        "logicalLanes": 4,
        "uniqueInputClasses": 4,
        "fixtureReuse": "none",
        "syntheticFixture": {"kind": "private-f32-lanes", "seed": None},
        "resolvedInputs": [
            {"name": "input0", "dtype": 18, "shape": [1, 1, 320, 672]},
        ],
        "runtime": {
            "executionMode": "scheduled",
            "cpuThreads": 1,
            "freshness": "ALL",
            "maxScheduledRequests": 4,
            "maxScheduledInputBytes": 2 * 1024**3,
            "maxUnconsumedResults": 8,
            "maxUnconsumedResultBytes": 2 * 1024**3,
            "requestedMaxBatchDelayMs": 100,
            "effectiveMaxBatchDelayMs": 100,
            "vulkanArenaMB": 1024,
        },
        "batchContract": {
            "protocol": "provider-batch-contract/v1",
            "axis": 0,
            "symbol": "batch",
            "min": 1,
            "max": 19,
            "multiple": 1,
            "proofIdentity": "typed-independent-batch-proof/v1:abc",
        },
        "strictRoute": {
            "routeAttested": True,
            "provider": "builtin:vulkan",
            "nodes": 10,
            "selected": 10,
            "fallback": 0,
            "missing": 0,
            "operatorFallbackEvidence": "operator=none",
        },
        "measurement": {
            "warmupGroups": 5,
            "repeatGroups": 30,
            "order": "alternating-direct-scheduled",
            "clock": "CLOCK_MONOTONIC",
            "timingBoundary": (
                "direct:first-vx_runtime_run-to-Nth-vx_runtime_run-owned-result;"
                "scheduled:first-vx_runtime_submit-through-all-waits-to-Nth-"
                "vx_request_result-owned-result;read-parity-release-untimed"
            ),
        },
        "direct": {
            "logicalLanes": 4,
            "physicalBatchSize": 1,
            "trueBackendInvocationsPerGroup": 4,
            "samplesMs": [4.0] * 30,
            "medianMs": 4.0,
        },
        "scheduled": {
            "logicalLanes": 4,
            "physicalBatchSize": 4,
            "trueBackendInvocationsPerGroup": 1,
            "allLanesSharedPhysicalExecutionId": True,
            "physicalExecutionIdChangesAcrossGroups": True,
            "physicalExecutionIdScope": "process-local-runtime-counter",
            "physicalExecutionIds": list(range(101, 131)),
            "samplesMs": [2.0] * 30,
            "medianMs": 2.0,
        },
        "parity": {
            "status": "pass",
            "absoluteTolerance": 0.0001,
            "relativeTolerance": 0.0001,
            "combinedAllclose": True,
            "nonfiniteRejected": True,
            "integerOutputsByteExact": True,
            "receiptDecodedEquality": True,
            "comparedElements": 176 * 4 * 35,
            "outputTensors": 140,
            "maxAbs": 0.0,
            "maxRel": 0.0,
        },
        "publicEvidence": {
            "source": "VxReport.route_evidence",
            "compiledContractExact": True,
            "scheduledLaneTokensExact": True,
            "directBatchTokensAbsent": True,
            "backendExecutionProofExact": True,
            "vulkanComputeDeviceLocal": True,
            "vulkanArenaBytes": 1024 * 1024 * 1024,
            "vulkanStagingBytes": 32 * 1024 * 1024,
            "vulkanStagingHostCoherent": True,
            "cudaGraphReplayMeasured": None,
        },
    }


def sha256(path: Path) -> str:
    return run_matrix._sha256(path)


class MatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.build = self.root / "build"
        (self.build / "native").mkdir(parents=True)
        self.harness = self.build / "native" / "native_dynamic_batch_benchmark"
        self.harness.write_bytes(b"benchmark")
        self.native_shader_compiler = self.build / "native-shader-compiler"
        self.native_shader_compiler.write_bytes(b"shader-compiler")
        self.native_shader_compiler.chmod(0o755)
        self.naga = self.build / "naga"
        self.naga.write_bytes(b"naga")
        self.naga.chmod(0o755)
        (self.build / "CMakeCache.txt").write_text("", encoding="utf-8")
        (self.build / "compile_commands.json").write_text("[]\n", encoding="utf-8")
        self.lanes = []
        for index in range(4):
            lane = self.root / f"lane-{index}.f32"
            lane.write_bytes(bytes([index + 1]) * 16)
            self.lanes.append(str(lane))
        self.packages: dict[tuple[str, int], tuple[Path, Path, Path]] = {}
        for precision in ("fp32", "int8"):
            for batch in (4, 19):
                package = self.root / f"{precision}-b{batch}"
                package.mkdir()
                manifest = package / "manifest.json"
                graph = package / "graph.json"
                weight = package / "model.safetensors"
                manifest.write_text(
                    json.dumps(
                        {
                            "format":
                                "volvoxai-receipt-digit-reader-onnx-package-v1",
                            "variant": precision,
                            "source_format": "receipt_digit_reader_onnx_v1",
                            "abi": {
                                "input": {
                                    "name": "input0",
                                    "shape": [1, 1, 320, 672],
                                },
                                "output": "slot_logits",
                                "batch": {
                                    "per_request": 1,
                                    "symbol": "batch",
                                    "min": 1,
                                    "max": batch,
                                    "multiple_of": 1,
                                }
                            },
                            "decode": {
                                "slots": 16,
                                "phone_slots": 12,
                                "street_slots": 4,
                                "blank_class": 10,
                                "num_classes": 11,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                graph.write_text(
                    json.dumps(
                        {
                            "format": "volvox-graph/v1",
                            "dimensions": {
                                "batch": {
                                    "min": 1,
                                    "max": batch,
                                    "multiple_of": 1,
                                }
                            },
                            "inputs": {
                                "input0": {
                                    "shape": ["batch", 1, 320, 672],
                                    "dtype": "float32",
                                }
                            },
                            "nodes": [
                                {
                                    "outputs": {
                                        "out": {
                                            "tensor": "slot_logits",
                                            "dtype": "float32",
                                            "shape": ["batch", 16, 11],
                                        }
                                    }
                                }
                            ],
                            "outputs": ["slot_logits"],
                        }
                    ),
                    encoding="utf-8",
                )
                weight.write_bytes(f"{precision}-{batch}".encode())
                self.packages[(precision, batch)] = (manifest, graph, weight)
        cases = []
        for backend in ("vulkan", "opengl", "cuda"):
            for precision in ("fp32", "int8"):
                for batch in (4, 19):
                    manifest, graph, weight = self.packages[(precision, 19)]
                    cases.append(
                        {
                            "id": f"receipt-{precision}-b{batch}-{backend}",
                            "artifactRole":
                                f"receipt-{precision}-maxB19",
                            "precision": precision,
                            "backend": backend,
                            "inputMode": "receipt",
                            "batchSize": batch,
                            "manifest": str(manifest),
                            "graph": str(graph),
                            "weights": [str(weight)],
                            "laneFiles": self.lanes,
                        }
                    )
        self.config = {
            "schema": "volvoxai.native-dynamic-batch-matrix-input",
            "suite": "receipt-digit-reader",
            "harness": str(self.harness),
            "measuredAtUtc": "2026-08-21T12:00:00+09:00",
            "device": {
                "expectedName": DEVICE_NAME,
                "expectedDriverVersion": DRIVER,
                "expectedTotalMemoryMiB": 24576,
            },
            "build": {
                "sourceRoot": str(REPO),
                "buildDirectory": str(self.build),
                "nativeShaderCompiler": str(self.native_shader_compiler),
                "naga": str(self.naga),
            },
            "warmupGroups": 5,
            "repeatGroups": 30,
            "maxBatchDelayMs": 100,
            "vulkanArenaMB": 1024,
            "cases": cases,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, config: dict | None = None) -> Path:
        path = self.root / "config.json"
        path.write_text(json.dumps(config or self.config), encoding="utf-8")
        return path

    def make_vqa_config(self) -> dict:
        packages: dict[
            tuple[str, str], tuple[Path, Path, Path, Path]
        ] = {}
        for precision in ("fp32", "int8"):
            package = self.root / f"vqa-{precision}"
            package.mkdir()
            manifest = package / "package_manifest.json"
            selected_graphs: dict[str, dict] = {}
            for component in ("encoder", "decoder"):
                component_directory = package / component
                component_directory.mkdir()
                graph = component_directory / "graph.json"
                weight = component_directory / "model.safetensors"
                if component == "encoder":
                    semantic_inputs = {
                        "image": "input0",
                        "question_ids": "input1",
                        "family_ids": "input2",
                        "question_position_ids": "input3",
                    }
                    graph_inputs = {
                        "input0": {
                            "shape": ["B", 1, 320, 672],
                            "dtype": "float32",
                        },
                        "input1": {
                            "shape": ["B", "Q"], "dtype": "int32"
                        },
                        "input2": {"shape": ["B"], "dtype": "int32"},
                        "input3": {
                            "shape": ["B", "Q"], "dtype": "int32"
                        },
                    }
                    dimensions = {
                        "B": {"min": 1, "max": 8},
                        "Q": {"min": 1, "max": 192},
                        "M": {"min": 211, "max": 402},
                    }
                    output_specs = {
                        "memory": (["B", "M", 320], "float32"),
                        "memory_padding_mask": (["B", "M"], "int32"),
                        "router_logits": (["B", 8], "float32"),
                        "selected_family_ids": (["B"], "int32"),
                        **{
                            f"cross_{kind}_{layer}":
                                (["B", 8, "M", 40], "float32")
                            for layer in range(4) for kind in ("k", "v")
                        },
                    }
                else:
                    semantic_names = [
                        "decoder_input_ids", "position_ids", "family_ids",
                        "memory_padding_mask", "past_padding_mask",
                        *(f"cross_{kind}_{layer}" for layer in range(4)
                          for kind in ("k", "v")),
                        *(f"past_{kind}_{layer}" for layer in range(4)
                          for kind in ("k", "v")),
                    ]
                    semantic_inputs = {
                        name: f"input{index}"
                        for index, name in enumerate(semantic_names)
                    }
                    graph_inputs = {
                        "input0": {"shape": ["B", 1], "dtype": "int32"},
                        "input1": {"shape": ["B"], "dtype": "int32"},
                        "input2": {"shape": ["B"], "dtype": "int32"},
                        "input3": {
                            "shape": ["B", "M"], "dtype": "int32"
                        },
                        "input4": {
                            "shape": ["B", "P"], "dtype": "int32"
                        },
                        **{
                            f"input{index}": {
                                "shape": ["B", 8, "M", 40],
                                "dtype": "float32",
                            }
                            for index in range(5, 13)
                        },
                        **{
                            f"input{index}": {
                                "shape": ["B", 8, "P", 40],
                                "dtype": "float32",
                            }
                            for index in range(13, 21)
                        },
                    }
                    dimensions = {
                        "B": {"min": 1, "max": 8},
                        "M": {"min": 211, "max": 402},
                        "P": {"min": 1, "max": 191},
                        "R": {"min": 2, "max": 192},
                    }
                    output_specs = {
                        "logits": (["B", 1, 1536], "float32"),
                        "present_padding_mask": (["B", "R"], "int32"),
                        **{
                            f"present_{kind}_{layer}":
                                (["B", 8, "R", 40], "float32")
                            for layer in range(4) for kind in ("k", "v")
                        },
                    }
                graph.write_text(
                    json.dumps(
                        {
                            "format": "volvox-graph/v1",
                            "dimensions": dimensions,
                            "inputs": graph_inputs,
                            "nodes": [
                                {
                                    "outputs": {
                                        f"out{index}": {
                                            "tensor": name,
                                            "shape": shape,
                                            "dtype": dtype,
                                        }
                                        for index, (name, (shape, dtype))
                                        in enumerate(output_specs.items())
                                    }
                                }
                            ],
                            "outputs": list(output_specs),
                        }
                    ),
                    encoding="utf-8",
                )
                weight.write_bytes(f"{precision}-{component}".encode())
                selected_graphs[component] = {
                    "graph": {
                        "path": f"{component}/graph.json",
                        "bytes": graph.stat().st_size,
                        "sha256": sha256(graph),
                    },
                    "weights": {
                        "path": f"{component}/model.safetensors",
                        "bytes": weight.stat().st_size,
                        "sha256": sha256(weight),
                    },
                    "inputs": semantic_inputs,
                    "outputs": {name: name for name in output_specs},
                }
                packages[(precision, component)] = (
                    manifest, graph, weight, package
                )
            manifest.write_text(
                json.dumps(
                    {
                        "format": (
                            "volvoxai-tiny-receipt-vqa-split-kv-onnx-"
                            "package-v2"
                        ),
                        "source": {
                            "format": "tiny_receipt_vqa_split_kv_onnx_v2",
                            "variant": (
                                "fp32" if precision == "fp32" else "int8-w8a8"
                            ),
                        },
                        "shape_contract": {
                            "dimensions": {
                                "B": {"min": 1, "max": 8},
                                "Q": {"min": 1, "max": 192},
                                "M": {"min": 211, "max": 402},
                                "P": {"min": 1, "max": 191},
                                "R": {"min": 2, "max": 192},
                            }
                        },
                        "graphs": selected_graphs,
                    }
                ),
                encoding="utf-8",
            )
        cases = []
        for backend in ("vulkan", "opengl", "cuda"):
            for precision in ("fp32", "int8"):
                for component in ("encoder", "decoder"):
                    manifest, graph, weight, _ = packages[(precision, component)]
                    cases.append(
                        {
                            "id": (
                                f"vqa-{precision}-{component}-b8-{backend}"
                            ),
                            "artifactRole": (
                                f"vqa-{precision}-{component}-maxB8"
                            ),
                            "precision": precision,
                            "backend": backend,
                            "inputMode": f"vqa-{component}",
                            "batchSize": 8,
                            "manifest": str(manifest),
                            "graph": str(graph),
                            "weights": [str(weight)],
                            "laneFiles": [],
                        }
                    )
        config = copy.deepcopy(self.config)
        config.update(
            {
                "suite": "tiny-receipt-vqa",
                "warmupGroups": 3,
                "repeatGroups": 5,
                "cases": cases,
            }
        )
        return config

    def test_data_declared_full_cross_product_is_required(self) -> None:
        loaded = run_matrix.load_config(self.write_config())
        self.assertEqual(len(loaded["cases"]), 12)
        incomplete = copy.deepcopy(self.config)
        incomplete["cases"].pop()
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(incomplete))
        custom_policy = copy.deepcopy(self.config)
        custom_policy.update({
            "warmupGroups": 1,
            "repeatGroups": 2,
            "maxBatchDelayMs": 777,
            "vulkanArenaMB": 6144,
        })
        custom_loaded = run_matrix.load_config(self.write_config(custom_policy))
        self.assertEqual(custom_loaded["repeatGroups"], 2)
        too_many_samples = copy.deepcopy(self.config)
        too_many_samples["repeatGroups"] = 1001
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(too_many_samples))
        historical = copy.deepcopy(self.config)
        historical["schema"] += "/v2"
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(historical))

    def test_receipt_package_max_can_serve_an_arbitrary_batch_sweep(self) -> None:
        split_packages = copy.deepcopy(self.config)
        for case in split_packages["cases"]:
            if case["batchSize"] != 4:
                continue
            manifest, graph, weight = self.packages[(case["precision"], 4)]
            case.update({
                "artifactRole": f"receipt-{case['precision']}-maxB4",
                "manifest": str(manifest),
                "graph": str(graph),
                "weights": [str(weight)],
            })
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(split_packages))

        sweep = copy.deepcopy(self.config)
        for backend in ("vulkan", "opengl", "cuda"):
            for precision in ("fp32", "int8"):
                source = next(
                    case for case in sweep["cases"]
                    if case["backend"] == backend
                    and case["precision"] == precision
                    and case["batchSize"] == 19
                )
                added = copy.deepcopy(source)
                added["id"] = f"receipt-{precision}-b16-{backend}"
                added["batchSize"] = 16
                sweep["cases"].append(added)
        loaded = run_matrix.load_config(self.write_config(sweep))
        self.assertEqual(
            sorted({case["batchSize"] for case in loaded["cases"]}),
            [4, 16, 19],
        )
        b16 = [case for case in loaded["cases"] if case["batchSize"] == 16]
        self.assertEqual(len(b16), 6)
        self.assertEqual(
            {case["artifactRole"] for case in b16},
            {"receipt-fp32-maxB19", "receipt-int8-maxB19"},
        )

        incomplete = copy.deepcopy(sweep)
        incomplete["cases"] = [
            case for case in incomplete["cases"]
            if case["id"] != "receipt-int8-b16-cuda"
        ]
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(incomplete))

        outside_package = copy.deepcopy(sweep)
        case = next(
            case for case in outside_package["cases"]
            if case["id"] == "receipt-fp32-b16-cuda"
        )
        case["id"] = "receipt-fp32-b20-cuda"
        case["batchSize"] = 20
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(outside_package))

    def test_report_records_sorted_batches_and_same_package_scope(self) -> None:
        loaded = run_matrix.load_config(self.write_config())
        build_evidence = {
            "sourceSnapshotSha256": "source-snapshot",
            "harnessSha256": "harness",
            "shaderToolchain": {
                "nativeShaderCompiler": {"binarySha256": "shader"},
                "naga": {"binarySha256": "naga"},
            },
        }

        def digest(path: Path) -> str:
            resolved = Path(path).resolve()
            if resolved == Path(loaded["harness"]).resolve():
                return "harness"
            if resolved == Path(
                loaded["build"]["nativeShaderCompiler"]
            ).resolve():
                return "shader"
            if resolved == Path(loaded["build"]["naga"]).resolve():
                return "naga"
            return "private-lane-stable"

        def measured_case(_config: dict, case: dict, _device: dict) -> dict:
            return {
                "artifactRole": case["artifactRole"],
                "artifacts": {"selection": case["artifactRole"]},
            }

        with (
            mock.patch.object(run_matrix, "verify_clean_environment"),
            mock.patch.object(run_matrix, "collect_host_evidence", return_value={}),
            mock.patch.object(
                run_matrix,
                "query_device",
                return_value={
                    "name": DEVICE_NAME,
                    "driverVersion": DRIVER,
                    "totalMemoryMiB": 24576,
                    "ordinal": 0,
                },
            ),
            mock.patch.object(
                run_matrix, "collect_build_evidence",
                return_value=build_evidence,
            ),
            mock.patch.object(run_matrix, "run_case", side_effect=measured_case),
            mock.patch.object(
                run_matrix, "_source_snapshot",
                return_value=("source-snapshot", False),
            ),
            mock.patch.object(run_matrix, "_sha256", side_effect=digest),
        ):
            report = run_matrix.build_report(loaded)
        self.assertEqual(report["matrix"]["batchSizes"], [4, 19])
        self.assertEqual(
            report["comparisonPolicy"],
            "same-package-dynamic-batch-sweep-only",
        )

    def test_malformed_case_result_fails_closed(self) -> None:
        loaded = run_matrix.load_config(self.write_config())
        configured = next(
            case for case in loaded["cases"]
            if case["id"] == "receipt-fp32-b4-vulkan"
        )
        valid = case_result()
        run_matrix.validate_case_result(valid, configured, loaded)
        malformed = copy.deepcopy(valid)
        malformed["scheduled"]["physicalExecutionIds"][1] = 101
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)
        malformed = copy.deepcopy(valid)
        malformed["batchContract"]["max"] = 4
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)
        malformed = copy.deepcopy(valid)
        malformed["batchContract"]["multiple"] = True
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)
        malformed = copy.deepcopy(valid)
        del malformed["strictRoute"]["missing"]
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)
        malformed = copy.deepcopy(valid)
        malformed["parity"]["outputTensors"] -= 1
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)
        malformed = copy.deepcopy(valid)
        malformed["publicEvidence"]["vulkanComputeDeviceLocal"] = False
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)
        malformed = copy.deepcopy(valid)
        malformed["publicEvidence"]["vulkanArenaBytes"] -= 1
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)
        malformed = copy.deepcopy(valid)
        malformed["publicEvidence"]["vulkanStagingBytes"] -= 1
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)
        malformed = copy.deepcopy(valid)
        malformed["publicEvidence"]["vulkanStagingHostCoherent"] = 1
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(malformed, configured, loaded)

        cuda_case = next(
            case for case in loaded["cases"]
            if case["id"] == "receipt-fp32-b4-cuda"
        )
        cuda_result = copy.deepcopy(valid)
        cuda_result["caseId"] = cuda_case["id"]
        cuda_result["backend"] = "cuda"
        cuda_result["strictRoute"]["provider"] = "builtin:cuda"
        cuda_result["runtime"]["vulkanArenaMB"] = None
        cuda_result["publicEvidence"].update({
            "vulkanComputeDeviceLocal": None,
            "vulkanArenaBytes": None,
            "vulkanStagingBytes": None,
            "vulkanStagingHostCoherent": None,
            "cudaGraphReplayMeasured": True,
        })
        run_matrix.validate_case_result(cuda_result, cuda_case, loaded)
        cuda_result["publicEvidence"]["cudaGraphReplayMeasured"] = False
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.validate_case_result(cuda_result, cuda_case, loaded)

        opengl_case = next(
            case for case in loaded["cases"]
            if case["id"] == "receipt-fp32-b4-opengl"
        )
        opengl_result = copy.deepcopy(cuda_result)
        opengl_result["caseId"] = opengl_case["id"]
        opengl_result["backend"] = "opengl"
        opengl_result["strictRoute"]["provider"] = "builtin:opengl"
        opengl_result["publicEvidence"]["cudaGraphReplayMeasured"] = None
        run_matrix.validate_case_result(opengl_result, opengl_case, loaded)

    def test_receipt_manifest_convention_and_abi_are_bound(self) -> None:
        loaded = run_matrix.load_config(self.write_config())
        case = next(
            case for case in loaded["cases"]
            if case["id"] == "receipt-int8-b19-cuda"
        )
        evidence = run_matrix.verify_manifest(
            loaded["suite"],
            case,
            Path(case["manifest"]),
            Path(case["graph"]),
            [Path(case["weights"][0])],
        )
        self.assertEqual(
            evidence["packageBindingVerification"],
            "package-convention+manifest-abi+graph-domain",
        )
        self.assertEqual(evidence["packageBatchMax"], 19)
        self.assertEqual(len(evidence["graphSha256"]), 64)
        broken = json.loads(Path(case["manifest"]).read_text(encoding="utf-8"))
        broken["abi"]["batch"]["max"] = 4
        Path(case["manifest"]).write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.verify_manifest(
                loaded["suite"],
                case,
                Path(case["manifest"]),
                Path(case["graph"]),
                [Path(case["weights"][0])],
            )

    def test_private_receipt_lane_identity_is_frozen_without_public_digest(
        self,
    ) -> None:
        loaded = run_matrix.load_config(self.write_config())
        identity = run_matrix._receipt_lane_identity(loaded)
        self.assertEqual(len(identity), 4)
        run_matrix._require_receipt_lane_identity(loaded, identity)
        Path(loaded["cases"][0]["laneFiles"][0]).write_bytes(b"changed")
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix._require_receipt_lane_identity(loaded, identity)
        public_attestation = {
            "privateReceiptLaneBytesStableAcrossMatrix": True
        }
        run_matrix.privacy_check(public_attestation, loaded)
        encoded = json.dumps(public_attestation).lower()
        self.assertNotIn("sha256", encoded)
        self.assertNotIn("digest", encoded)

        two_classes = copy.deepcopy(self.config)
        for case in two_classes["cases"]:
            case["laneFiles"] = case["laneFiles"][:2]
        two_loaded = run_matrix.load_config(self.write_config(two_classes))
        self.assertEqual(len(run_matrix._receipt_lane_identity(two_loaded)), 2)

        no_classes = copy.deepcopy(self.config)
        for case in no_classes["cases"]:
            case["laneFiles"] = []
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(no_classes))

    def test_vqa_manifest_package_max_and_dynamic_sweep_are_bound(self) -> None:
        config = self.make_vqa_config()
        loaded = run_matrix.load_config(self.write_config(config))
        self.assertEqual(loaded["warmupGroups"], 3)
        for case in loaded["cases"]:
            evidence = run_matrix.verify_manifest(
                loaded["suite"],
                case,
                Path(case["manifest"]),
                Path(case["graph"]),
                [Path(case["weights"][0])],
            )
            self.assertEqual(
                evidence["packageBindingVerification"],
                "manifest-selected-path+bytes+sha256+graph-domain",
            )
            self.assertEqual(evidence["packageBatchMax"], 8)

        sweep = copy.deepcopy(config)
        for source in list(sweep["cases"]):
            added = copy.deepcopy(source)
            added["batchSize"] = 4
            added["id"] = source["id"].replace("-b8-", "-b4-")
            sweep["cases"].append(added)
        swept = run_matrix.load_config(self.write_config(sweep))
        self.assertEqual(len(swept["cases"]), 24)
        self.assertEqual(
            sorted({case["batchSize"] for case in swept["cases"]}), [4, 8]
        )
        self.assertEqual(
            {case["artifactRole"] for case in swept["cases"]},
            {
                "vqa-fp32-encoder-maxB8",
                "vqa-fp32-decoder-maxB8",
                "vqa-int8-encoder-maxB8",
                "vqa-int8-decoder-maxB8",
            },
        )

        wrong_role = copy.deepcopy(config)
        wrong_role["cases"][0]["artifactRole"] = "vqa-fp32-decoder-maxB8"
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(wrong_role))
        aliased_component = copy.deepcopy(config)
        encoder_graph = next(
            case["graph"] for case in aliased_component["cases"]
            if case["precision"] == "fp32"
            and case["inputMode"] == "vqa-encoder"
        )
        for configured in aliased_component["cases"]:
            if (
                configured["precision"] == "fp32"
                and configured["inputMode"] == "vqa-decoder"
            ):
                configured["graph"] = encoder_graph
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.load_config(self.write_config(aliased_component))

        case = next(
            case for case in loaded["cases"]
            if case["id"] == "vqa-fp32-encoder-b8-vulkan"
        )
        manifest_path = Path(case["manifest"])
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        broken = copy.deepcopy(original)
        broken["graphs"]["encoder"]["graph"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.verify_manifest(
                loaded["suite"], case, manifest_path, Path(case["graph"]),
                [Path(case["weights"][0])],
            )
        broken = copy.deepcopy(original)
        broken["graphs"]["encoder"]["inputs"]["image"] = "input1"
        manifest_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.verify_manifest(
                loaded["suite"], case, manifest_path, Path(case["graph"]),
                [Path(case["weights"][0])],
            )

        graph_path = Path(case["graph"])
        original_graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph = copy.deepcopy(original_graph)
        graph["outputs"].pop()
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        broken = copy.deepcopy(original)
        broken["graphs"]["encoder"]["graph"].update(
            {
                "bytes": graph_path.stat().st_size,
                "sha256": sha256(graph_path),
            }
        )
        manifest_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.verify_manifest(
                loaded["suite"], case, manifest_path, graph_path,
                [Path(case["weights"][0])],
            )

        graph = copy.deepcopy(original_graph)
        graph["dimensions"]["B"]["max"] = 7
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        broken = copy.deepcopy(original)
        broken["graphs"]["encoder"]["graph"].update(
            {
                "bytes": graph_path.stat().st_size,
                "sha256": sha256(graph_path),
            }
        )
        manifest_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.verify_manifest(
                loaded["suite"], case, manifest_path, graph_path,
                [Path(case["weights"][0])],
            )
        broken = copy.deepcopy(original)
        broken["shape_contract"]["dimensions"]["B"]["max"] = 7
        manifest_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.verify_manifest(
                loaded["suite"], case, manifest_path, Path(case["graph"]),
                [Path(case["weights"][0])],
            )

    def test_build_evidence_is_release_sm86_strict_fma_and_pathless(self) -> None:
        compiler = Path(shutil.which("cc") or "/usr/bin/cc").resolve()
        nvcc = self.build / "fake-nvcc"
        nvcc.write_text("fixture\n", encoding="utf-8")
        cache = {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_HOME_DIRECTORY": str(REPO),
            "CMAKE_C_COMPILER": str(compiler),
            "VOLVOXAI_NVCC_EXECUTABLE": str(nvcc),
            "VOLVOXAI_ENABLE_CUDA": "ON",
            "VOLVOXAI_ENABLE_VULKAN": "ON",
            "VOLVOXAI_ENABLE_OPENGL": "ON",
            "VOLVOXAI_CUDA_ARCH": "86",
            "VOLVOXAI_CUDA_FAST_FP32": "OFF",
        }

        def write_build(fmad: str = "false") -> None:
            (self.build / "CMakeCache.txt").write_text(
                "".join(f"{key}:STRING={value}\n" for key, value in cache.items()),
                encoding="utf-8",
            )
            (self.build / "compile_commands.json").write_text(
                json.dumps(
                    [
                        {
                            "directory": str(self.build / "native"),
                            "file": str(
                                REPO / "examples/native_dynamic_batch_benchmark/main.c"
                            ),
                            "arguments": [
                                str(compiler), "-O3", "-DNDEBUG",
                                "-ffp-contract=off", "-c", "main.c",
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            generated = (
                self.build / "native" / "CMakeFiles" /
                "volvoxai_compile_shaders.dir"
            )
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "build.make").write_text(
                f"{nvcc} --ptx -O3 "
                f"--fmad={fmad} --gpu-architecture=compute_86 "
                "native/src/backends/cuda_kernels.cu\n"
                "cmake -E env VOLVOXAI_NATIVE_SHADER_COMPILER="
                f"{self.native_shader_compiler} "
                f"{REPO}/tools/compile_shaders.sh\n",
                encoding="utf-8",
            )

        def evidence(_command: list[str], label: str) -> str:
            return {
                "C compiler": "cc fixture 1.0",
                "CMake": "cmake version 3.30.0",
                "NVCC": "Cuda compilation tools, release 12.4, V12.4.0",
                "Naga": "30.0.0",
                "source commit": "1" * 40,
            }[label]

        def collect() -> dict:
            with (
                mock.patch.object(
                    run_matrix, "_run_evidence", side_effect=evidence
                ),
                mock.patch.object(
                    run_matrix.shutil, "which", return_value=str(self.naga)
                ),
            ):
                return run_matrix.collect_build_evidence(config)

        write_build()
        config = run_matrix.load_config(self.write_config())
        result = collect()
        self.assertEqual(result["configuration"], "Release")
        self.assertEqual(result["cudaArch"], "sm86")
        self.assertEqual(result["cudaFp32Contract"], "strict-no-fma")
        self.assertIs(result["nvccFmad"], False)
        self.assertIn(
            "examples/native_dynamic_batch_benchmark/**",
            result["sourceSnapshotScope"]["included"],
        )
        self.assertIn("docs/**", result["sourceSnapshotScope"]["excluded"])
        self.assertEqual(
            result["shaderToolchain"]["naga"]["version"], "30.0.0"
        )
        self.assertTrue(
            result["shaderToolchain"]
            ["cmakeCommandUsedConfiguredNativeCompiler"]
        )
        encoded = json.dumps(result)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn(str(REPO), encoded)

        write_build()
        commands_path = self.build / "compile_commands.json"
        commands = json.loads(commands_path.read_text(encoding="utf-8"))
        wrong_compiler = self.build / "wrong-compiler"
        wrong_compiler.write_text("fixture\n", encoding="utf-8")
        commands[0]["arguments"][0] = str(wrong_compiler)
        commands_path.write_text(json.dumps(commands), encoding="utf-8")
        with self.assertRaises(run_matrix.ValidationError):
            collect()

        cache["CMAKE_HOME_DIRECTORY"] = str(self.root)
        write_build()
        with self.assertRaises(run_matrix.ValidationError):
            collect()
        cache["CMAKE_HOME_DIRECTORY"] = str(REPO)

        write_build()
        commands_path = self.build / "compile_commands.json"
        commands = json.loads(commands_path.read_text(encoding="utf-8"))
        commands[0]["file"] = str(
            self.root / "other/examples/native_dynamic_batch_benchmark/main.c"
        )
        commands_path.write_text(json.dumps(commands), encoding="utf-8")
        with self.assertRaises(run_matrix.ValidationError):
            collect()

        write_build()
        generated_path = (
            self.build / "native/CMakeFiles/volvoxai_compile_shaders.dir/build.make"
        )
        generated_path.write_text(
            generated_path.read_text(encoding="utf-8").replace(
                "VOLVOXAI_NATIVE_SHADER_COMPILER=",
                "XVOLVOXAI_NATIVE_SHADER_COMPILER=",
            ),
            encoding="utf-8",
        )
        with self.assertRaises(run_matrix.ValidationError):
            collect()

        write_build()
        generated_path.write_text(
            generated_path.read_text(encoding="utf-8").replace(
                str(REPO / "tools/compile_shaders.sh"),
                "X" + str(REPO / "tools/compile_shaders.sh"),
            ),
            encoding="utf-8",
        )
        with self.assertRaises(run_matrix.ValidationError):
            collect()

        write_build()
        generated_path.write_text(
            generated_path.read_text(encoding="utf-8").replace(
                f"{nvcc} --ptx", f"X{nvcc} --ptx",
            ),
            encoding="utf-8",
        )
        with self.assertRaises(run_matrix.ValidationError):
            collect()

        write_build()
        generated_path.write_text(
            generated_path.read_text(encoding="utf-8").replace(
                str(self.native_shader_compiler),
                str(self.native_shader_compiler) + "-other",
            ),
            encoding="utf-8",
        )
        with self.assertRaises(run_matrix.ValidationError):
            collect()

        cache["VOLVOXAI_CUDA_FAST_FP32"] = "ON"
        write_build()
        with self.assertRaises(run_matrix.ValidationError):
            collect()
        cache["VOLVOXAI_CUDA_FAST_FP32"] = "OFF"
        write_build("true")
        with self.assertRaises(run_matrix.ValidationError):
            collect()

    def test_source_snapshot_ignores_reports_but_binds_runtime_sources(self) -> None:
        source = self.root / "snapshot-source"
        (source / "native/src/runtime").mkdir(parents=True)
        (source / "native/tests").mkdir(parents=True)
        (source / "docs").mkdir()
        (source / "examples/receipt_digit_reader/reports").mkdir(parents=True)
        (source / "examples/native_dynamic_batch_benchmark").mkdir(
            parents=True
        )
        runtime_source = source / "native/src/runtime/public_api.c"
        runtime_source.write_text("runtime-a\n", encoding="utf-8")
        (source / "native/tests/test.c").write_text("test-a\n", encoding="utf-8")
        documentation = source / "docs/native-runtime.md"
        documentation.write_text("docs-a\n", encoding="utf-8")
        report = source / "examples/receipt_digit_reader/reports/result.json"
        report.write_text("{}\n", encoding="utf-8")
        wrapper = source / "examples/native_dynamic_batch_benchmark/run_matrix.py"
        wrapper.write_text("wrapper-a\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "add", "."], check=True
        )
        before, _ = run_matrix._source_snapshot(source)
        documentation.write_text("docs-b\n", encoding="utf-8")
        report.write_text('{"measured":true}\n', encoding="utf-8")
        (source / "native/tests/test.c").write_text("test-b\n", encoding="utf-8")
        after_publication, _ = run_matrix._source_snapshot(source)
        self.assertEqual(after_publication, before)
        runtime_source.write_text("runtime-b\n", encoding="utf-8")
        after_runtime_change, _ = run_matrix._source_snapshot(source)
        self.assertNotEqual(after_runtime_change, before)
        runtime_source.write_text("runtime-a\n", encoding="utf-8")
        wrapper.write_text("wrapper-b\n", encoding="utf-8")
        after_wrapper_change, _ = run_matrix._source_snapshot(source)
        self.assertNotEqual(after_wrapper_change, before)

    def test_host_evidence_is_derived_and_child_affinity_is_zero(self) -> None:
        evidence = run_matrix.collect_host_evidence()
        self.assertEqual(evidence["benchmarkChildAffinity"], [0])
        self.assertGreaterEqual(evidence["logicalCpuCount"], 1)
        self.assertTrue(evidence["operatingSystem"])
        self.assertTrue(evidence["kernelRelease"])
        self.assertTrue(evidence["architecture"])
        self.assertTrue(evidence["cpuModel"])
        loaded = run_matrix.load_config(self.write_config())
        run_matrix.privacy_check({"host": evidence}, loaded)

    def test_run_case_pins_cuda_vulkan_and_locale_and_one_json(self) -> None:
        loaded = run_matrix.load_config(self.write_config())

        def raw_for(case: dict) -> dict:
            raw = case_result()
            raw["caseId"] = case["id"]
            raw["backend"] = case["backend"]
            raw["strictRoute"]["provider"] = f"builtin:{case['backend']}"
            raw["runtime"]["vulkanArenaMB"] = (
                1024 if case["backend"] == "vulkan" else None
            )
            if case["backend"] == "cuda":
                raw["publicEvidence"].update({
                    "vulkanComputeDeviceLocal": None,
                    "vulkanArenaBytes": None,
                    "vulkanStagingBytes": None,
                    "vulkanStagingHostCoherent": None,
                    "cudaGraphReplayMeasured": True,
                })
            return raw

        for backend, init_line in (
            (
                "cuda",
                f"[CUDA] device 0: {DEVICE_NAME} (compute 8.6)\n",
            ),
            (
                "vulkan",
                "[VolvoxAI GPU] Vulkan Compute initialized successfully! "
                f"Device: {DEVICE_NAME}; packed INT8 dot: enabled\n",
            ),
        ):
            case = next(
                case for case in loaded["cases"]
                if case["id"] == f"receipt-fp32-b4-{backend}"
            )
            process = SimpleNamespace(
                returncode=0,
                stdout=init_line + json.dumps(raw_for(case), separators=(",", ":"))
                + "\n",
                stderr="",
            )
            with (
                mock.patch.object(
                    run_matrix,
                    "verify_manifest",
                    return_value={
                        "graphSha256": "a" * 64,
                        "packageBatchMax": 19,
                    },
                ),
                mock.patch.object(
                    run_matrix.subprocess, "run", return_value=process
                ) as invoked,
            ):
                result = run_matrix.run_case(
                    loaded,
                    case,
                    {
                        "name": DEVICE_NAME,
                        "driverVersion": DRIVER,
                    },
                )
            self.assertEqual(result["status"], "pass")
            command = invoked.call_args.args[0]
            environment = invoked.call_args.kwargs["env"]
            self.assertEqual(environment["LC_ALL"], "C")
            self.assertEqual(environment["OMP_NUM_THREADS"], "1")
            if backend == "cuda":
                self.assertEqual(environment["VOLVOXAI_CUDA_DEVICE"], "0")
                self.assertIn("--cuda-device", command)
            else:
                self.assertEqual(environment["VOLVOX_VULKAN_MB"], "1024")
                self.assertIn("--vulkan-arena-mb", command)

        duplicate = SimpleNamespace(
            returncode=0,
            stdout=process.stdout + "  {\"other\":true}\n",
            stderr="",
        )
        with (
            mock.patch.object(
                run_matrix, "verify_manifest",
                return_value={"packageBatchMax": 19},
            ),
            mock.patch.object(run_matrix.subprocess, "run", return_value=duplicate),
        ):
            with self.assertRaises(run_matrix.ValidationError):
                run_matrix.run_case(
                    loaded,
                    case,
                    {"name": DEVICE_NAME, "driverVersion": DRIVER},
                )

    def test_canonical_output_and_privacy_are_fail_closed(self) -> None:
        loaded = run_matrix.load_config(self.write_config())
        report = {
            "schema": run_matrix.OUTPUT_SCHEMA,
            "artifact": {"graphSha256": "a" * 64},
        }
        run_matrix.privacy_check(report, loaded)
        output = self.root / "report.json"
        run_matrix.write_canonical(output, report)
        encoded = output.read_text(encoding="utf-8")
        self.assertEqual(
            encoded,
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        )
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.privacy_check(
                {"leak": loaded["cases"][0]["graph"]}, loaded
            )
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.privacy_check({"device": "GPU PCIe/SSE2"}, loaded)
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.privacy_check(
                {"toolchain": "/home/private-user/bin/cc"}, loaded
            )
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.privacy_check(
                {"maintainer": "private@example.test"}, loaded
            )

    def test_backend_identity_is_sanitized_and_exact(self) -> None:
        cuda_line = f"[CUDA] device 0: {DEVICE_NAME} (compute 8.6)\n"
        cuda = run_matrix._parse_backend_identity(
            "cuda", "", cuda_line * 3,
            DEVICE_NAME, DRIVER,
        )
        self.assertEqual(cuda["ordinal"], 0)
        self.assertEqual(cuda["observationCount"], 3)
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix._parse_backend_identity(
                "cuda", "", cuda_line +
                f"[CUDA] device 1: {DEVICE_NAME} (compute 8.6)\n",
                DEVICE_NAME, DRIVER,
            )
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix._parse_backend_identity(
                "cuda", "", cuda_line +
                f"[CUDA] device malformed: {DEVICE_NAME} (compute 8.6)\n",
                DEVICE_NAME, DRIVER,
            )
        opengl_line = (
            "[VolvoxAI GPU] OpenGL Compute initialized: NVIDIA Corporation / "
            f"{DEVICE_NAME}/PCIe/SSE2 / 4.6.0 NVIDIA {DRIVER}\n"
        )
        opengl = run_matrix._parse_backend_identity(
            "opengl", opengl_line * 2, "", DEVICE_NAME, DRIVER,
        )
        self.assertNotIn("PCI", json.dumps(opengl))
        self.assertEqual(opengl["observationCount"], 2)
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix._parse_backend_identity(
                "opengl", opengl_line + opengl_line.replace(
                    DEVICE_NAME, "NVIDIA GeForce RTX 4090"
                ), "", DEVICE_NAME, DRIVER,
            )
        vulkan_line = (
            "[VolvoxAI GPU] Vulkan Compute initialized successfully! "
            f"Device: {DEVICE_NAME}; packed INT8 dot: enabled\n"
        )
        vulkan = run_matrix._parse_backend_identity(
            "vulkan", vulkan_line * 2, "", DEVICE_NAME, DRIVER,
        )
        self.assertEqual(vulkan["observationCount"], 2)
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix._parse_backend_identity(
                "vulkan", vulkan_line + vulkan_line.replace(
                    "enabled", "unavailable"
                ), "", DEVICE_NAME, DRIVER,
            )
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix._parse_backend_identity(
                "vulkan",
                "[VolvoxAI GPU] Vulkan Compute initialized successfully! "
                "Device: llvmpipe; packed INT8 dot: unavailable\n",
                "", DEVICE_NAME, DRIVER,
            )

    def test_derived_metrics_recompute_median_and_throughput(self) -> None:
        result = case_result()
        run_matrix.add_derived_metrics(result)
        self.assertEqual(result["derived"]["direct"]["logicalRequests"], 120)
        self.assertEqual(
            result["derived"]["scheduled"]["physicalInvocations"], 30
        )
        self.assertEqual(result["derived"]["sameBackendMedianSpeedup"], 2.0)
        result = case_result()
        result["direct"]["medianMs"] = 99.0
        with self.assertRaises(run_matrix.ValidationError):
            run_matrix.add_derived_metrics(result)


if __name__ == "__main__":
    unittest.main()
