import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.tiny_receipt_vqa.tools.benchmark_explicit_kv import (
    BenchmarkFailure,
    DYNAMIC_QUALIFICATION_PREFIX,
    DYNAMIC_QUALIFICATION_SCHEMA,
    REPORT_FORMAT,
    SAMPLE_ISOLATION_SCOPE,
    USER_OWNED_PROVENANCE_EXCLUSIONS,
    WARMUP_ISOLATION_SCOPE,
    WORKTREE_EXECUTION_SOURCE_SNAPSHOT_FORMAT,
    _assert_execution_inputs_stable,
    _benchmark_child_environment,
    _counterbalanced_tier_order,
    _git_provenance,
    _host_cpu_identity,
    _inspect_ort_optimized_graph,
    _onnx_graph_summary,
    _ort_strict_session_probe,
    _parse_ort_profile_placement,
    _parse_cpuinfo,
    _threading_settings,
    _volvox_graph_summary,
    parse_native_dynamic_qualification,
    parse_native_sample,
    parse_wasm_dynamic_qualification,
    prove_dynamic_qualification_coverage,
    run_native_dynamic_qualification,
    run_ort_sample,
)


def _cpu_dynamic_qualification_output(*, threads=6, input_hash="a" * 64):
    lines = []
    for index, (logical_q, bound_q) in enumerate(((2, 2), (8, 8), (8, 192), (2, 2))):
        mode = "maximum-padded" if index == 2 else "active"
        lines.append(
            f"DYNAMIC_REBIND_RUN index={index} mode={mode} "
            f"logical_Q={logical_q} bound_Q={bound_q} "
            f"logical_M={logical_q + 210} bound_M={bound_q + 210} "
            "family_id=0 tokens=2 token_digest=9a76a600c5543d20 "
            "token_ids=4,5 seed_P=1 seed_R=2 step_P=2 step_R=3 "
            "cache_preserved=1"
        )
    lines.append(
        "DYNAMIC_REBIND_RESULT status=pass backend=cpu timed=0 "
        "same_runtime=1 same_encoder_context=1 same_decoder_context=1 "
        f"strict_no_fallback=1 cpu_threads={threads}"
    )
    route = (
        "provider=builtin:cpu;nodes=3;selected=3;fallback=0;missing=0;"
        "digest=fixture;shape_plan=hit;"
    )
    debug = []
    question_tokens = ("11,2", "11,12,13,14,15,16,17,2", "11,12,13,14,15,16,17,2", "11,2")
    for run_index in range(4):
        debug.append(f"[debug] tinyreceipt split input_f32_sha256={input_hash}")
        debug.append(
            f"[debug] tinyreceipt split question_token_ids={question_tokens[run_index]}"
        )
        debug.append(f"[debug] tinyreceipt split encoder shape {route}")
        for step, token in enumerate((4, 5)):
            debug.append(
                f"[debug] tinyreceipt explicit-kv family=phone step={step} "
                f"P={step + 1} R={step + 2} token={token} time=1.000 ms"
            )
            debug.append(f"[debug] tinyreceipt split decoder shape {route}")
    return ("\n".join(lines) + "\n").encode(), ("\n".join(debug) + "\n").encode()


def _strict_wasm_route():
    return {
        "tierFallback": False,
        "operator": {
            "attestation": "none",
            "used": False,
            "offendingNode": None,
        },
    }


def _strict_wasm_compilation():
    return {
        "requestedPolicy": {
            "mode": "require",
            "backend": "wasm",
            "operatorFallback": "forbid",
        },
        "selectedBackend": "wasm",
        "routeEvidence": _strict_wasm_route(),
        "candidates": [{
            "backend": "wasm",
            "outcome": "selected",
            "routeEvidence": _strict_wasm_route(),
        }],
    }


def _strict_wasm_execution(context_id):
    return {
        "contextId": context_id,
        "backend": "wasm",
        "outcome": "success",
        "operatorFallback": "none",
        "routeEvidence": _strict_wasm_route(),
    }


def _dynamic_qualification_runs(*, include_family):
    runs = []
    shapes = (
        ("short-before", "active", 2, 2, [11, 2]),
        ("representative-active", "active", 8, 8, [11, 12, 13, 14, 15, 16, 17, 2]),
        (
            "representative-maximum-padded",
            "maximum-padded",
            8,
            192,
            [11, 12, 13, 14, 15, 16, 17, 2],
        ),
        ("short-after", "active", 2, 2, [11, 2]),
    )
    for label, mode, logical_q, bound_q, question_ids in shapes:
        run = {
            "label": label,
            "shapeMode": mode,
            "familyId": 0,
            "tokenIds": [4, 5],
            "activeShape": {
                "B": 1,
                "Q": bound_q,
                "M": bound_q + 210,
                "T": 5,
            },
            "logicalShape": {
                "B": 1,
                "Q": logical_q,
                "M": logical_q + 210,
                "T": 5,
            },
            "cache": {
                "initialPastLength": 1,
                "finalPastLength": 3,
                "sentinelMaskValue": 1,
                "transitions": [
                    {"position": 0, "pastLength": 1, "presentLength": 2},
                    {"position": 1, "pastLength": 2, "presentLength": 3},
                ],
            },
        }
        if include_family:
            run.update({
                "family": "phone",
                "requestedFamily": "phone",
                "questionTokenIds": question_ids,
            })
        runs.append(run)
    return runs


def _wasm_dynamic_qualification_value(precision):
    runs = _dynamic_qualification_runs(include_family=True)
    return {
        "schema": DYNAMIC_QUALIFICATION_SCHEMA,
        "engine": "volvoxai-wasm",
        "backend": "wasm",
        "precision": precision,
        "prompt": "phone number last one",
        "family": "phone",
        "timed": False,
        "freshProcess": True,
        "sameSession": True,
        "sameRuntime": True,
        "sameEncoderContext": True,
        "sameDecoderContext": True,
        "strictNoFallback": True,
        "boundedDecoderMaximumNewTokens": 4,
        "maximumLegalEncoderBinding": {"B": 1, "Q": 192, "M": 402},
        "inputTensorSha256": "a" * 64,
        "checks": {
            "exactOutputShapes": True,
            "finiteOutputs": True,
            "cachePrefixPreserved": True,
            "appendedCacheRowFinite": True,
            "selectedFamilyParity": True,
            "tokenParity": True,
            "encoderGrowShrink": True,
            "decoderGrowShrink": True,
        },
        "runs": runs,
        "routeEvidence": {
            "compilation": [_strict_wasm_compilation(), _strict_wasm_compilation()],
            "execution": [{
                "label": run["label"],
                "encoder": _strict_wasm_execution("encoder-context"),
                "decoder": [
                    _strict_wasm_execution("decoder-context")
                    for _ in run["tokenIds"]
                ],
            } for run in runs],
        },
    }


def _wasm_dynamic_qualification_output(precision, value=None):
    payload = _wasm_dynamic_qualification_value(precision) if value is None else value
    return f"diagnostic\n{DYNAMIC_QUALIFICATION_PREFIX}{json.dumps(payload)}\n".encode()


class BenchmarkProvenanceTests(unittest.TestCase):
    def test_report_format_reflects_worktree_snapshot_contract(self):
        self.assertEqual(
            REPORT_FORMAT,
            "volvoxai.tiny-receipt-explicit-kv-cpu-benchmark/v1",
        )

    def test_native_cpu_dynamic_qualification_requires_exact_evidence(self):
        stdout, stderr = _cpu_dynamic_qualification_output(threads=6)
        value = parse_native_dynamic_qualification(
            stdout,
            stderr,
            precision="int8",
            backend="cpu",
            max_new=4,
            cpu_threads=6,
        )
        self.assertEqual(value["engine"], "native-c")
        self.assertEqual(value["backend"], "cpu")
        self.assertFalse(value["timed"])
        self.assertTrue(value["strictNoFallback"])
        self.assertEqual(value["runtime"], {"cpuThreads": 6})
        self.assertEqual(value["inputTensorSha256"], "a" * 64)
        self.assertEqual(
            [run["label"] for run in value["runs"]],
            [
                "short-before",
                "representative-active",
                "representative-maximum-padded",
                "short-after",
            ],
        )
        self.assertEqual([run["tokenIds"] for run in value["runs"]], [[4, 5]] * 4)
        self.assertEqual(value["runs"][2]["activeShape"], {
            "B": 1, "Q": 192, "M": 402, "T": 5,
        })
        self.assertEqual(value["runs"][0]["cache"], {
            "initialPastLength": 1,
            "finalPastLength": 3,
            "sentinelMaskValue": 1,
            "transitions": [
                {"position": 0, "pastLength": 1, "presentLength": 2},
                {"position": 1, "pastLength": 2, "presentLength": 3},
            ],
        })
        self.assertEqual(len(value["routeEvidence"]["encoder"]), 4)
        self.assertEqual(len(value["routeEvidence"]["decoder"]), 8)
        self.assertEqual(len(value["routeEvidence"]["decoderSteps"]), 4)

        with self.assertRaisesRegex(BenchmarkFailure, "unique evidence"):
            parse_native_dynamic_qualification(
                stdout,
                stderr,
                precision="int8",
                backend="cpu",
                max_new=4,
                cpu_threads=1,
            )
        fallback = stderr.replace(b"fallback=0", b"fallback=1", 1)
        with self.assertRaisesRegex(BenchmarkFailure, "strict fallback 0"):
            parse_native_dynamic_qualification(
                stdout,
                fallback,
                precision="int8",
                backend="cpu",
                max_new=4,
                cpu_threads=6,
            )
        mismatched_hash = stderr.replace(b"a" * 64, b"b" * 64, 1)
        with self.assertRaisesRegex(BenchmarkFailure, "input/step/route"):
            parse_native_dynamic_qualification(
                stdout,
                mismatched_hash,
                precision="int8",
                backend="cpu",
                max_new=4,
                cpu_threads=6,
            )
        corrupt_digest = stdout.replace(
            b"token_digest=9a76a600c5543d20",
            b"token_digest=0123456789abcdef",
            1,
        )
        with self.assertRaisesRegex(BenchmarkFailure, "run 0 is inconsistent"):
            parse_native_dynamic_qualification(
                corrupt_digest,
                stderr,
                precision="int8",
                backend="cpu",
                max_new=4,
                cpu_threads=6,
            )

    def test_native_cpu_dynamic_runner_passes_requested_thread_limit(self):
        stdout, stderr = _cpu_dynamic_qualification_output(threads=6)
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=stdout, stderr=stderr
        )
        with patch(
            "examples.tiny_receipt_vqa.tools.benchmark_explicit_kv.subprocess.run",
            return_value=completed,
        ) as mocked:
            value = run_native_dynamic_qualification(
                binary=Path("/fixture/tiny_receipt_split_w8a8"),
                package=Path("/fixture/package"),
                precision="fp32",
                image=Path("/fixture/input.pgm"),
                backend="cpu",
                max_new=4,
                threads=6,
                timeout=30.0,
                environment={"PATH": "/fixture/bin"},
            )
        self.assertEqual(value["runtime"]["cpuThreads"], 6)
        command = mocked.call_args.args[0]
        self.assertEqual(command, [
            "/fixture/tiny_receipt_split_w8a8",
            "/fixture/package",
            "--image",
            "/fixture/input.pgm",
            "--qualify-dynamic",
            "--cpu",
            "--threads",
            "6",
        ])
        self.assertEqual(mocked.call_args.kwargs["env"], {"PATH": "/fixture/bin"})
        self.assertEqual(mocked.call_args.kwargs["timeout"], 30.0)

    def test_wasm_dynamic_qualification_requires_strict_same_context_evidence(self):
        self.assertEqual(
            DYNAMIC_QUALIFICATION_SCHEMA,
            "volvoxai.tiny-receipt-dynamic-shape-qualification/v1",
        )
        stdout = _wasm_dynamic_qualification_output("int8")
        value = parse_wasm_dynamic_qualification(
            stdout,
            precision="int8",
            family="phone",
            max_new=4,
            node_version="v24.0.0",
        )
        self.assertEqual(value["engine"], "volvoxai-wasm")
        self.assertEqual(value["runtime"], {
            "name": "node",
            "version": "v24.0.0",
            "executionThreads": 1,
            "workerThreads": 0,
        })
        self.assertEqual(len(value["routeEvidence"]["compilation"]), 2)
        self.assertEqual(len(value["routeEvidence"]["execution"]), 4)
        self.assertEqual(
            value["routeEvidence"]["execution"][3]["encoder"]["contextId"],
            "encoder-context",
        )

        unknown_schema = _wasm_dynamic_qualification_value("int8")
        unknown_schema["schema"] = "invalid.dynamic-qualification-schema"
        with self.assertRaisesRegex(BenchmarkFailure, "identity is invalid"):
            parse_wasm_dynamic_qualification(
                _wasm_dynamic_qualification_output("int8", unknown_schema),
                precision="int8",
                family="phone",
                max_new=4,
                node_version="v24.0.0",
            )

        fallback = _wasm_dynamic_qualification_value("int8")
        fallback["routeEvidence"]["execution"][0]["decoder"][0][
            "routeEvidence"
        ]["tierFallback"] = True
        with self.assertRaisesRegex(BenchmarkFailure, "strict fallback 0"):
            parse_wasm_dynamic_qualification(
                _wasm_dynamic_qualification_output("int8", fallback),
                precision="int8",
                family="phone",
                max_new=4,
                node_version="v24.0.0",
            )

        wrong_context = _wasm_dynamic_qualification_value("int8")
        wrong_context["routeEvidence"]["execution"][3]["encoder"][
            "contextId"
        ] = "replacement-encoder-context"
        with self.assertRaisesRegex(BenchmarkFailure, "reuse both contexts"):
            parse_wasm_dynamic_qualification(
                _wasm_dynamic_qualification_output("int8", wrong_context),
                precision="int8",
                family="phone",
                max_new=4,
                node_version="v24.0.0",
            )

    def test_cpu_wasm_dynamic_qualification_coverage_is_exact_and_in_parity(self):
        native_stdout, native_stderr = _cpu_dynamic_qualification_output(threads=6)
        entries = [
            parse_native_dynamic_qualification(
                native_stdout,
                native_stderr,
                precision=precision,
                backend="cpu",
                max_new=4,
                cpu_threads=6,
            )
            for precision in ("fp32", "int8")
        ]
        entries.extend(
            parse_wasm_dynamic_qualification(
                _wasm_dynamic_qualification_output(precision),
                precision=precision,
                family="phone",
                max_new=4,
                node_version="v24.0.0",
            )
            for precision in ("fp32", "int8")
        )
        proof = prove_dynamic_qualification_coverage(
            entries,
            native_backends=["cpu"],
            include_webgpu=False,
            include_wasm=True,
            expected_question_token_ids=([11, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 2]),
        )
        self.assertEqual(proof["status"], "pass")
        self.assertFalse(proof["timed"])
        self.assertEqual(proof["expectedEntries"], [
            "native-c/cpu/fp32",
            "native-c/cpu/int8",
            "volvoxai-wasm/wasm/fp32",
            "volvoxai-wasm/wasm/int8",
        ])
        self.assertEqual(len(proof["entries"]), 4)
        self.assertTrue(proof["checks"]["crossBackendPrecisionParity"])

        with self.assertRaisesRegex(BenchmarkFailure, "coverage is incomplete"):
            prove_dynamic_qualification_coverage(
                entries[:-1],
                native_backends=["cpu"],
                include_webgpu=False,
                include_wasm=True,
                expected_question_token_ids=([11, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 2]),
            )
        mismatched = json.loads(json.dumps(entries))
        mismatched[-1]["runs"][3]["tokenIds"] = [4, 6]
        with self.assertRaisesRegex(BenchmarkFailure, "parity failed"):
            prove_dynamic_qualification_coverage(
                mismatched,
                native_backends=["cpu"],
                include_webgpu=False,
                include_wasm=True,
                expected_question_token_ids=([11, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 2]),
            )
        missing_question_tokens = json.loads(json.dumps(entries))
        del missing_question_tokens[0]["runs"][0]["questionTokenIds"]
        with self.assertRaisesRegex(BenchmarkFailure, "questionTokenIds"):
            prove_dynamic_qualification_coverage(
                missing_question_tokens,
                native_backends=["cpu"],
                include_webgpu=False,
                include_wasm=True,
                expected_question_token_ids=([11, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 2]),
            )
        wrong_canonical_question = json.loads(json.dumps(entries))
        for entry in wrong_canonical_question:
            entry["runs"][1]["questionTokenIds"] = [21, 22, 23, 24, 25, 26, 27, 2]
            entry["runs"][2]["questionTokenIds"] = [21, 22, 23, 24, 25, 26, 27, 2]
        with self.assertRaisesRegex(BenchmarkFailure, "exact question-token evidence"):
            prove_dynamic_qualification_coverage(
                wrong_canonical_question,
                native_backends=["cpu"],
                include_webgpu=False,
                include_wasm=True,
                expected_question_token_ids=([11, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 12, 13, 14, 15, 16, 17, 2], [11, 2]),
            )

    def test_child_environment_removes_volvox_overrides_without_values(self):
        inherited = {
            "PATH": "/usr/bin",
            "OMP_NUM_THREADS": "99",
            "VOLVOXAI_SHADER_DIR": "/private/shaders",
            "VOLVOXAI_THREADS": "999",
            "VOLVOX_VULKAN_DISABLE_DOT": "private-value",
        }
        environment, evidence = _benchmark_child_environment(inherited, 3)
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["OMP_NUM_THREADS"], "3")
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "3")
        self.assertNotIn("VOLVOXAI_SHADER_DIR", environment)
        self.assertNotIn("VOLVOXAI_THREADS", environment)
        self.assertNotIn("VOLVOX_VULKAN_DISABLE_DOT", environment)
        self.assertEqual(evidence["removedRuntimeOverrideNames"], [
            "VOLVOXAI_SHADER_DIR",
            "VOLVOXAI_THREADS",
            "VOLVOX_VULKAN_DISABLE_DOT",
        ])
        self.assertFalse(evidence["removedRuntimeOverrideValuesRecorded"])
        serialized = json.dumps(evidence, sort_keys=True)
        self.assertNotIn("/private/shaders", serialized)
        self.assertNotIn("private-value", serialized)
        self.assertNotIn('"999"', serialized)
        for invalid in (True, 0, -1, 1.0):
            with self.subTest(invalid=invalid), self.assertRaises(BenchmarkFailure):
                _benchmark_child_environment(inherited, invalid)

    def test_cpu_matrix_order_counterbalances_warmup_and_measured_indices(self):
        tiers = [
            "onnxruntime/fp32",
            "native-c/fp32",
            "volvoxai-wasm/fp32",
            "onnxruntime/int8",
            "native-c/int8",
            "volvoxai-wasm/int8",
        ]
        self.assertEqual(_counterbalanced_tier_order(tiers, 0), tiers)
        self.assertEqual(_counterbalanced_tier_order(tiers, 1), tiers[::-1])
        self.assertEqual(
            _counterbalanced_tier_order(tiers, 2), tiers[1:] + tiers[:1]
        )
        self.assertEqual(
            _counterbalanced_tier_order(tiers, 3),
            list(reversed(tiers[1:] + tiers[:1])),
        )
        for matrix_index in range(2 * len(tiers)):
            self.assertCountEqual(
                _counterbalanced_tier_order(tiers, matrix_index), tiers
            )
        for invalid_tiers, invalid_index in (([], 0), (["x", "x"], 0), (tiers, -1)):
            with self.subTest(
                tiers=invalid_tiers, index=invalid_index
            ), self.assertRaises(BenchmarkFailure):
                _counterbalanced_tier_order(invalid_tiers, invalid_index)

    def test_ort_profile_placement_is_exact_order_independent_and_closed(self):
        events = [
            {"cat": "Session", "name": "model_run"},
            {
                "cat": "Node",
                "name": "quant_kernel_time",
                "args": {
                    "node_index": "7",
                    "op_name": "QuantizeLinear",
                    "provider": "CPUExecutionProvider",
                },
            },
            {
                "cat": "Node",
                "name": "gemm_kernel_time",
                "args": {
                    "node_index": "2",
                    "op_name": "MatMul",
                    "provider": "CUDAExecutionProvider",
                },
            },
            {
                "cat": "Node",
                "name": "add_kernel_time",
                "args": {
                    "node_index": 5,
                    "op_name": "Add",
                    "provider": "CUDAExecutionProvider",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(json.dumps(events), encoding="utf-8")
            placement = _parse_ort_profile_placement(
                profile,
                role="encoder",
                allowed_providers=(
                    "CUDAExecutionProvider", "CPUExecutionProvider",
                ),
            )
            self.assertEqual(placement["executedNodeCount"], 3)
            self.assertEqual(placement["providers"], {
                "CPUExecutionProvider": 1,
                "CUDAExecutionProvider": 2,
            })
            self.assertEqual(
                placement["operatorCountsByProvider"]["CUDAExecutionProvider"],
                {"Add": 1, "MatMul": 1},
            )
            self.assertRegex(placement["nodeAssignmentSha256"], r"^[0-9a-f]{64}$")

            events[-1]["args"]["node_index"] = "2"
            profile.write_text(json.dumps(events), encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkFailure, "duplicate node placement"):
                _parse_ort_profile_placement(
                    profile,
                    role="encoder",
                    allowed_providers=(
                        "CUDAExecutionProvider", "CPUExecutionProvider",
                    ),
                )

            events[-1]["args"]["node_index"] = "5"
            events[-1]["args"]["provider"] = "UnknownExecutionProvider"
            profile.write_text(json.dumps(events), encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkFailure, "node placement"):
                _parse_ort_profile_placement(
                    profile,
                    role="encoder",
                    allowed_providers=(
                        "CUDAExecutionProvider", "CPUExecutionProvider",
                    ),
                )

    def test_ort_strict_probe_accepts_only_cpu_partition_rejection(self):
        class SessionOptions:
            def add_session_config_entry(self, name, value):
                self.entry = (name, value)

        strict_error = RuntimeError(
            "This session contains graph nodes that are assigned to the default "
            "CPU EP, but fallback to CPU EP has been explicitly disabled by the user."
        )
        fake_ort = types.SimpleNamespace(
            SessionOptions=SessionOptions,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="all"),
            InferenceSession=lambda *_args, **_kwargs: (_ for _ in ()).throw(strict_error),
        )
        result = _ort_strict_session_probe(
            fake_ort,
            Path("encoder.onnx"),
            role="encoder",
            threads=1,
            provider="CUDAExecutionProvider",
            provider_options={"device_id": "0"},
        )
        self.assertEqual(result["outcome"], "rejected")
        self.assertEqual(result["stage"], "session-create")
        self.assertTrue(result["cpuEpFallbackDisabled"])

        fake_ort.InferenceSession = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("GPU device was lost")
        )
        with self.assertRaisesRegex(BenchmarkFailure, "other than CPU partitioning"):
            _ort_strict_session_probe(
                fake_ort,
                Path("encoder.onnx"),
                role="encoder",
                threads=1,
                provider="CUDAExecutionProvider",
                provider_options={},
            )

    def test_ort_execution_warmup_reuses_sessions_and_resets_kv(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is required")

        class Output:
            def __init__(self, name):
                self.name = name

        class FakeSession:
            def __init__(self, kind, drift=False):
                self.kind = kind
                self.drift = drift
                self.calls = 0
                self.request_starts = 0
                self.names = (
                    ["memory", "memory_padding_mask", "router_logits",
                     "selected_family_ids", *(
                         name for layer in range(4)
                         for name in (f"cross_k_{layer}", f"cross_v_{layer}")
                     )]
                    if kind == "encoder"
                    else ["logits", "present_padding_mask", *(
                        name for layer in range(4)
                        for name in (f"present_k_{layer}", f"present_v_{layer}")
                    )]
                )

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def get_outputs(self):
                return [Output(name) for name in self.names]

            def run(self, _outputs, feeds):
                self.calls += 1
                if self.kind == "encoder":
                    m = 218
                    values = {
                        "memory": np.zeros((1, m, 320), np.float32),
                        "memory_padding_mask": np.zeros((1, m), np.bool_),
                        "router_logits": np.zeros((1, 8), np.float32),
                        "selected_family_ids": np.asarray([0], np.int64),
                    }
                    for layer in range(4):
                        for prefix in ("cross_k", "cross_v"):
                            values[f"{prefix}_{layer}"] = np.zeros(
                                (1, 8, m, 40), np.float32
                            )
                    return [values[name] for name in self.names]
                position = int(feeds["position_ids"][0])
                past_mask = feeds["past_padding_mask"]
                past_length = past_mask.shape[1]
                if position == 0:
                    self.request_starts += 1
                    self.assert_sentinel(feeds, past_mask)
                token = 5 if self.drift and self.request_starts == 3 \
                    and position == 0 else (4 if position == 0 else 2)
                logits = np.zeros((1, 1, 1536), np.float32)
                logits[0, 0, token] = 1
                present_mask = np.concatenate([
                    past_mask, np.asarray([[feeds["decoder_input_ids"][0, 0] == 0]])
                ], axis=1)
                values = {
                    "logits": logits,
                    "present_padding_mask": present_mask,
                }
                for layer in range(4):
                    for prefix in ("k", "v"):
                        past = feeds[f"past_{prefix}_{layer}"]
                        values[f"present_{prefix}_{layer}"] = np.concatenate([
                            past, np.zeros((1, 8, 1, 40), np.float32)
                        ], axis=2)
                return [values[name] for name in self.names]

            def assert_sentinel(self, feeds, past_mask):
                if not np.array_equal(past_mask, np.ones((1, 1), np.bool_)):
                    raise AssertionError("request did not reset the sentinel mask")
                for layer in range(4):
                    for prefix in ("k", "v"):
                        value = feeds[f"past_{prefix}_{layer}"]
                        if value.shape != (1, 8, 1, 40) or np.any(value):
                            raise AssertionError("request did not reset the KV sentinel")

        class Tokenizer:
            def encode(self, _prompt, *, add_eos, max_len):
                self.assert_options(add_eos, max_len)
                return list(range(11, 18)) + [2]

            @staticmethod
            def assert_options(add_eos, max_len):
                if add_eos is not True or max_len != 192:
                    raise AssertionError("unexpected tokenizer options")

        def run(drift=False, warmup=2):
            sessions = [FakeSession("encoder"), FakeSession("decoder", drift)]
            created = []

            class SessionOptions:
                def add_session_config_entry(self, _name, _value):
                    pass

            fake_ort = types.SimpleNamespace(
                __version__="fixture",
                SessionOptions=SessionOptions,
                ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                GraphOptimizationLevel=types.SimpleNamespace(
                    ORT_ENABLE_ALL="all"
                ),
                get_available_providers=lambda: ["CPUExecutionProvider"],
            )

            def create_session(*_args, **_kwargs):
                session = sessions[len(created)]
                created.append(session)
                return session

            fake_ort.InferenceSession = create_session
            source = {
                "selected_paths": {"encoder": Path("encoder.onnx"),
                                   "decoder": Path("decoder.onnx")},
                "tokenizer_runtime": Tokenizer(),
            }
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}), patch(
                "examples.tiny_receipt_vqa.tools.benchmark_explicit_kv._preprocess_image",
                return_value=np.zeros((1, 1, 320, 672), np.float32),
            ):
                result = run_ort_sample(
                    source, "fp32", Path("image.png"),
                    "phone number last one", "phone", 2, 1,
                    execution_warmup=warmup,
                )
            return result, sessions, created

        result, sessions, created = run()
        self.assertEqual(len(created), 2)
        self.assertEqual(sessions[0].calls, 3)
        self.assertEqual(sessions[1].calls, 6)
        self.assertEqual(sessions[1].request_starts, 3)
        self.assertEqual(result["tokenIds"], [4, 2])
        self.assertEqual(result["runtime"]["executionWarmup"], {
            "runs": 2,
            "sameSessions": True,
            "stateResetToSentinel": True,
            "tokenCacheTransitionParity": True,
        })
        with self.assertRaisesRegex(Exception, "warmup token/cache transition parity"):
            run(drift=True)
        cold, sessions, _ = run(warmup=0)
        self.assertEqual(sessions[0].calls, 1)
        self.assertEqual(sessions[1].calls, 2)
        self.assertEqual(cold["runtime"]["executionWarmup"]["runs"], 0)
        for invalid in (True, -1, 21, "1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                Exception, "integer in"
            ):
                run(warmup=invalid)

    def test_native_gpu_parser_requires_the_requested_strict_backend(self):
        stdout = (
            "VolvoxAI Native Runtime\n"
            "Backend policy: vulkan\n"
            "[VolvoxAI GPU] Vulkan Compute initialized successfully! Device: "
            "Fixture GPU; packed INT8 dot: enabled\n"
        ).encode()
        stderr = (
            "[debug] tinyreceipt split ABI=explicit-kv-v1 routing=runtime "
            "decoder_output=f32_logits shape_mode=active argmax=host-first-index\n"
            "[debug] tinyreceipt split input_f32_sha256=" + "a" * 64 + "\n"
            "[debug] tinyreceipt split router=phone selected=phone (requested) "
            "encoder=10.000 ms\n"
            "[debug] tinyreceipt split encoder shape provider=builtin:vulkan;"
            "nodes=2;selected=2;fallback=0;missing=0;digest=deadbeef;"
            "shape_plan=hit;\n"
            "[debug] tinyreceipt explicit-kv family=phone step=0 P=1 R=2 "
            "token=4 time=2.000 ms\n"
            "[debug] tinyreceipt split decoder shape provider=builtin:vulkan;"
            "nodes=3;selected=3;fallback=0;missing=0;digest=beadfeed;"
            "shape_plan=hit;\n"
            "[debug] tinyreceipt explicit-kv family=phone step=1 P=2 R=3 "
            "token=5 time=1.000 ms\n"
            "[debug] tinyreceipt split decoder shape provider=builtin:vulkan;"
            "nodes=3;selected=3;fallback=0;missing=0;digest=beadfeed;"
            "shape_plan=hit;\n"
            "[debug] tinyreceipt explicit-kv family=phone tokens=2 total=3.000 ms\n"
            "[debug] tinyreceipt split emitted_token_ids=4,5\n"
            "[debug] tinyreceipt split question_token_ids=11,12,13,14,15,16,17,2\n"
            "[debug] tinyreceipt split shape mode=active logical_Q=8 bound_Q=8 "
            "logical_M=218 bound_M=218 seed_P=1 maximum_R=3\n"
        ).encode()
        sample = parse_native_sample(
            stdout, stderr, precision="int8", family="phone", max_new=2,
            threads=1, process_wall_ms=20.0, backend="vulkan",
        )
        self.assertEqual(sample["backend"], "vulkan")
        self.assertEqual(sample["provider"], "vulkan")
        self.assertEqual(sample["runtime"]["cpuThreads"], 1)
        self.assertEqual(sample["runtime"]["device"], {
            "name": "Fixture GPU", "packedInt8Dot": True,
        })
        self.assertEqual(len(sample["routeEvidence"]["decoder"]), 2)

        warmup_stdout = stdout + (
            "WARMUP_RESULT status=pass count=1 warmup_timed=0 measured_runs=1 "
            "same_runtime=1 same_encoder_context=1 same_decoder_context=1 "
            "strict_no_fallback=1 token_parity=1 cache_parity=1 cache_reset=1 "
            "family_id=0 tokens=2 token_digest=9a76a600c5543d20 token_ids=4,5 "
            "seed_P=1 seed_R=2 last_P=2 last_R=3 cache_preserved=1\n"
        ).encode()
        warmed = parse_native_sample(
            warmup_stdout, stderr, precision="int8", family="phone", max_new=2,
            threads=1, process_wall_ms=20.0, backend="vulkan",
            execution_warmup=1,
        )
        self.assertEqual(warmed["runtime"]["executionWarmup"]["runs"], 1)
        with self.assertRaisesRegex(Exception, "warmup evidence"):
            parse_native_sample(
                stdout, stderr, precision="int8", family="phone", max_new=2,
                threads=1, process_wall_ms=20.0, backend="vulkan",
                execution_warmup=1,
            )
        with self.assertRaisesRegex(Exception, "inconsistent"):
            parse_native_sample(
                warmup_stdout.replace(b"token_ids=4,5", b"token_ids=4,6"),
                stderr, precision="int8", family="phone", max_new=2,
                threads=1, process_wall_ms=20.0, backend="vulkan",
                execution_warmup=1,
            )
        malformed_warmup = warmup_stdout + b"WARMUP_RESULT malformed\n"
        with self.assertRaisesRegex(Exception, "unique same-context warmup evidence"):
            parse_native_sample(
                malformed_warmup, stderr, precision="int8", family="phone",
                max_new=2, threads=1, process_wall_ms=20.0,
                backend="vulkan", execution_warmup=1,
            )
        with self.assertRaisesRegex(Exception, "inconsistent"):
            parse_native_sample(
                warmup_stdout.replace(
                    b"token_digest=9a76a600c5543d20",
                    b"token_digest=0123456789abcdef",
                ),
                stderr, precision="int8", family="phone", max_new=2,
                threads=1, process_wall_ms=20.0, backend="vulkan",
                execution_warmup=1,
            )
        missed_encoder = stderr.replace(b"shape_plan=hit", b"shape_plan=miss", 1)
        with self.assertRaisesRegex(Exception, "warmed vulkan shape plan hit"):
            parse_native_sample(
                warmup_stdout, missed_encoder, precision="int8", family="phone",
                max_new=2, threads=1, process_wall_ms=20.0,
                backend="vulkan", execution_warmup=1,
            )
        missed_decoder = stderr.replace(
            b"shape_plan=hit", b"shape_plan=miss", 2
        ).replace(b"shape_plan=miss", b"shape_plan=hit", 1)
        with self.assertRaisesRegex(Exception, "warmed vulkan shape plan hit"):
            parse_native_sample(
                warmup_stdout, missed_decoder, precision="int8", family="phone",
                max_new=2, threads=1, process_wall_ms=20.0,
                backend="vulkan", execution_warmup=1,
            )
        cold = parse_native_sample(
            stdout, missed_encoder, precision="int8", family="phone", max_new=2,
            threads=1, process_wall_ms=20.0, backend="vulkan",
            execution_warmup=0,
        )
        self.assertEqual(cold["runtime"]["executionWarmup"]["runs"], 0)
        for invalid in (True, -1, 21, "1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                Exception, "integer in"
            ):
                parse_native_sample(
                    stdout, stderr, precision="int8", family="phone",
                    max_new=2, threads=1, process_wall_ms=20.0,
                    backend="vulkan", execution_warmup=invalid,
                )

        cuda_stdout = stdout.replace(
            b"Backend policy: vulkan", b"Backend policy: cuda"
        )
        cuda_stderr = (
            b"[CUDA] device 0: Fixture RTX (compute 8.6)\n"
            b"[CUDA] device 0: Fixture RTX (compute 8.6)\n"
            + stderr.replace(b"builtin:vulkan", b"builtin:cuda")
        )
        cuda_sample = parse_native_sample(
            cuda_stdout, cuda_stderr, precision="int8", family="phone",
            max_new=2, threads=1, process_wall_ms=20.0, backend="cuda",
        )
        self.assertEqual(cuda_sample["runtime"]["device"], {
            "index": 0,
            "name": "Fixture RTX",
            "computeCapability": "8.6",
        })
        with self.assertRaisesRegex(Exception, "conflicting device identities"):
            parse_native_sample(
                cuda_stdout,
                cuda_stderr + b"[CUDA] device 1: Other GPU (compute 9.0)\n",
                precision="int8", family="phone", max_new=2, threads=1,
                process_wall_ms=20.0, backend="cuda",
            )

        with self.assertRaisesRegex(Exception, "opengl policy"):
            parse_native_sample(
                stdout, stderr, precision="int8", family="phone", max_new=2,
                threads=1, process_wall_ms=20.0, backend="opengl",
            )

    def test_volvox_graph_summary_records_runtime_operator_counts(self):
        with tempfile.TemporaryDirectory(prefix="tinyreceipt-volvox-graph-") as value:
            path = Path(value) / "graph.json"
            path.write_text(
                json.dumps({
                    "nodes": [
                        {"opType": "Add"},
                        {"opType": "QConv2D"},
                        {"opType": "Add"},
                    ]
                }),
                encoding="utf-8",
            )
            self.assertEqual(_volvox_graph_summary(path), {
                "nodeCount": 3,
                "operatorCounts": {"Add": 2, "QConv2D": 1},
            })

    def test_ort_graph_inspection_records_source_and_optimized_structure(self):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("ONNX graph inspection requires optional onnx/onnxruntime")

        with tempfile.TemporaryDirectory(prefix="tinyreceipt-ort-graph-") as value:
            directory = Path(value)
            source = directory / "source.onnx"
            optimized = directory / "optimized.onnx"
            tensor = onnx.helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [1]
            )
            model = onnx.helper.make_model(
                onnx.helper.make_graph(
                    [onnx.helper.make_node("Identity", ["x"], ["y"])],
                    "fixture",
                    [tensor],
                    [onnx.helper.make_tensor_value_info(
                        "y", onnx.TensorProto.FLOAT, [1]
                    )],
                ),
                opset_imports=[onnx.helper.make_opsetid("", 18)],
            )
            model.ir_version = 9
            onnx.save(model, source)
            self.assertEqual(_onnx_graph_summary(onnx, source)["nodeCount"], 1)

            inspection = _inspect_ort_optimized_graph(
                ort=ort,
                onnx_module=onnx,
                source=source,
                output=optimized,
                logical_name="fixture/optimized.onnx",
                threads=1,
            )
            self.assertEqual(inspection["source"]["nodeCount"], 1)
            self.assertEqual(
                inspection["optimized"]["name"], "fixture/optimized.onnx"
            )
            self.assertEqual(
                inspection["graphOptimizationLevel"], "ORT_ENABLE_ALL"
            )
            self.assertTrue(optimized.is_file())

    def test_worktree_snapshot_includes_untracked_and_exact_exclusions(self):
        with tempfile.TemporaryDirectory(prefix="tinyreceipt-benchmark-git-") as value:
            repository = Path(value)
            source = repository / "source.txt"
            report = repository / "reports" / "matrix.json"
            report.parent.mkdir()
            source.write_text("source-one\n", encoding="utf-8")
            report.write_text("report-one\n", encoding="utf-8")
            (repository / ".gitignore").write_text(
                "ignored.tmp\n", encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "benchmark@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Benchmark Test"],
                cwd=repository,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"], cwd=repository, check=True
            )

            untracked = repository / "untracked.txt"
            untracked.write_text("untracked-one\n", encoding="utf-8")
            ignored = repository / "ignored.tmp"
            ignored.write_text("ignored-one\n", encoding="utf-8")
            link = repository / "source-link"
            link.symlink_to("source.txt")
            for relative in USER_OWNED_PROVENANCE_EXCLUSIONS:
                planning_path = repository / relative
                planning_path.parent.mkdir(parents=True, exist_ok=True)
                planning_path.write_text("planning-one\n", encoding="utf-8")

            first = _git_provenance(repository, ["reports/matrix.json"])
            snapshot = first["worktreeExecutionSource"]
            self.assertEqual(
                snapshot["format"], WORKTREE_EXECUTION_SOURCE_SNAPSHOT_FORMAT
            )
            self.assertEqual(snapshot["trackedPathCount"], 2)
            self.assertEqual(snapshot["nonignoredUntrackedPathCount"], 2)
            self.assertEqual(snapshot["pathCount"], 4)
            self.assertEqual(snapshot["regularFileCount"], 3)
            self.assertEqual(snapshot["symlinkCount"], 1)
            self.assertEqual(snapshot["missingPathCount"], 0)
            self.assertEqual(
                snapshot["includedTrackedPaths"], [".gitignore", "source.txt"]
            )
            self.assertEqual(
                snapshot["includedNonignoredUntrackedPaths"],
                ["source-link", "untracked.txt"],
            )
            self.assertEqual(
                snapshot["excludedPublicationPaths"], ["reports/matrix.json"]
            )
            self.assertEqual(snapshot["excludedTrackedPaths"], ["reports/matrix.json"])
            self.assertEqual(
                snapshot["excludedUserOwnedPlanningPaths"],
                list(USER_OWNED_PROVENANCE_EXCLUSIONS),
            )
            self.assertEqual(
                snapshot["excludedNonignoredUntrackedPaths"],
                sorted(USER_OWNED_PROVENANCE_EXCLUSIONS),
            )

            report.write_text("report-two\n", encoding="utf-8")
            ignored.write_text("ignored-two\n", encoding="utf-8")
            for relative in USER_OWNED_PROVENANCE_EXCLUSIONS:
                (repository / relative).write_text("planning-two\n", encoding="utf-8")
            report_only = _git_provenance(repository, ["reports/matrix.json"])
            self.assertEqual(first["statusSha256"], report_only["statusSha256"])
            self.assertEqual(
                report_only["excludedStatusPaths"],
                sorted([
                    "reports/matrix.json", *USER_OWNED_PROVENANCE_EXCLUSIONS
                ]),
            )
            self.assertEqual(
                report_only["excludedPublicationStatusPaths"],
                ["reports/matrix.json"],
            )
            self.assertEqual(
                report_only["excludedUserOwnedPlanningStatusPaths"],
                list(USER_OWNED_PROVENANCE_EXCLUSIONS),
            )
            self.assertEqual(
                snapshot["sha256"],
                report_only["worktreeExecutionSource"]["sha256"],
            )

            untracked.write_text("untracked-two\n", encoding="utf-8")
            untracked_change = _git_provenance(repository, ["reports/matrix.json"])
            self.assertNotEqual(
                snapshot["sha256"],
                untracked_change["worktreeExecutionSource"]["sha256"],
            )

            untracked.write_text("untracked-one\n", encoding="utf-8")
            source.chmod(0o755)
            mode_change = _git_provenance(repository, ["reports/matrix.json"])
            self.assertNotEqual(
                snapshot["sha256"],
                mode_change["worktreeExecutionSource"]["sha256"],
            )

            source.chmod(0o644)
            link.unlink()
            link.symlink_to("reports/matrix.json")
            link_change = _git_provenance(repository, ["reports/matrix.json"])
            self.assertNotEqual(
                snapshot["sha256"],
                link_change["worktreeExecutionSource"]["sha256"],
            )

    def test_execution_input_stability_rejects_changed_or_added_inputs(self):
        before = {
            "repository": {"head": "a" * 40, "dirty": True},
            "generator": {"sha256": "b" * 64},
        }
        _assert_execution_inputs_stable(before, dict(before))
        with self.assertRaisesRegex(BenchmarkFailure, "generator"):
            _assert_execution_inputs_stable(
                before,
                {**before, "generator": {"sha256": "c" * 64}},
            )
        with self.assertRaisesRegex(BenchmarkFailure, "package"):
            _assert_execution_inputs_stable(
                before,
                {**before, "package": {"sha256": "d" * 64}},
            )

    def test_cpuinfo_parser_preserves_model_and_sorted_feature_set(self):
        model, features = _parse_cpuinfo(
            "processor: 0\n"
            "model name: Example CPU\n"
            "flags: avx2 sse2 avx2\n\n"
            "processor: 1\n"
            "model name: Example CPU\n"
            "flags: sse2 vnni\n"
        )
        self.assertEqual(model, "Example CPU")
        self.assertEqual(features, ["avx2", "sse2", "vnni"])

    def test_host_and_thread_records_make_wasm_single_thread_explicit(self):
        host = _host_cpu_identity()
        self.assertIsInstance(host["allowedCpuIds"], list)
        self.assertEqual(host["allowedCpuIds"], sorted(set(host["allowedCpuIds"])))
        self.assertEqual(
            host["allowedLogicalCpuCount"], len(host["allowedCpuIds"])
        )
        topology = host["allowedCpuTopology"]
        if topology["available"]:
            unique_cores = {
                (value["packageId"], value["coreId"])
                for value in topology["logicalCpus"]
            }
            self.assertEqual(topology["physicalCoreCount"], len(unique_cores))
            self.assertEqual(
                len(topology["logicalCpus"]), host["allowedLogicalCpuCount"]
            )
        threading = _threading_settings(6, host)
        self.assertFalse(threading["affinityPinnedByHarness"])
        self.assertEqual(threading["onnxruntime"]["intraOpThreadLimit"], 6)
        self.assertEqual(
            threading["onnxruntime"]["graphOptimizationLevel"], "ORT_ENABLE_ALL"
        )
        self.assertEqual(threading["nativeC"]["cpuThreadLimit"], 6)
        self.assertEqual(threading["volvoxaiWasm"]["executionThreads"], 1)
        self.assertEqual(threading["volvoxaiWasm"]["workerThreads"], 0)
        self.assertFalse(threading["volvoxaiWasm"]["cpuThreadLimitApplied"])
        self.assertEqual(
            threading["volvoxaiWasm"]["threadCountScope"],
            "VolvoxAI WASM engine only",
        )
        self.assertIn(
            "Node/V8 auxiliary threads",
            threading["volvoxaiWasm"]["hostRuntimeAuxiliaryThreads"],
        )
        self.assertEqual(
            threading["inheritedAffinity"]["allowedCpuIds"], host["allowedCpuIds"]
        )

    def test_isolation_scope_distinguishes_ort_sessions_from_child_processes(self):
        self.assertIn("persistent Python harness process", SAMPLE_ISOLATION_SCOPE)
        self.assertIn("fresh child process", SAMPLE_ISOLATION_SCOPE)
        self.assertIn("ONNX Runtime sessions", WARMUP_ISOLATION_SCOPE)
        self.assertIn("global state", WARMUP_ISOLATION_SCOPE)


if __name__ == "__main__":
    unittest.main()
