import json
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from examples.tiny_receipt_vqa.tools.benchmark_explicit_kv import (
    PUBLICATION_OUTPUT_PATHS,
    _ort_provider_registration,
)
from examples.tiny_receipt_vqa.tools.benchmark_explicit_kv_gpu import (
    DYNAMIC_QUALIFICATION_SCHEMA,
    REPORT_FORMAT,
    WEBGPU_RESULT_PREFIX,
    _counterbalanced_tier_order,
    _expected_tiers,
    _parse_public_arguments,
    _physical_native_gpu_device,
    _provider_options,
    parse_native_dynamic_qualification,
    parse_ort_webgpu_sample,
    parse_webgpu_dynamic_qualification,
    parse_webgpu_sample,
    prove_dynamic_qualification_coverage,
    prove_gpu_parity,
    summarize_gpu_samples,
)

QUESTION_IDS = [11, 12, 2]


def ort_webgpu_output_model_hashes(precision):
    return {
        "fp32": {
            "encoder": "0e2206c54756b15d697d2bbd8d22bb68d0d2475d0323965ca03a14fb186e6eb0",
            "decoder": "b918a9de531bad205990acff28a2a7e2244b14229c312ed6b2f75d0540d6a6a2",
        },
        "int8": {
            "encoder": "a5be30f7507f8e2988c4cf735a80fb6c09a4f1e93c9759b1fc5e12d04c4da45c",
            "decoder": "75437beed7cdac93b4a704e9a4ffbefbe0199855344a57cb8f78ae105e915419",
        },
    }[precision]


def route():
    return {
        "tierFallback": False,
        "operator": {
            "attestation": "none",
            "used": False,
            "offendingNode": None,
        },
    }


def compilation(label, adapter):
    return {
        "label": label,
        "report": {
            "requestedPolicy": {
                "mode": "require",
                "backend": "webgpu",
                "operatorFallback": "forbid",
            },
            "selectedBackend": "webgpu",
            "selectedDevice": adapter,
            "routeEvidence": route(),
            "candidates": [{
                "backend": "webgpu",
                "outcome": "selected",
                "routeEvidence": route(),
            }],
        },
    }


def execution():
    return {
        "backend": "webgpu",
        "outcome": "success",
        "operatorFallback": "none",
        "routeEvidence": route(),
    }


def volvox_webgpu_kv_evidence(token_count=2, *, qualified=False):
    return {
        "enabled": True,
        "mode": "device-qualified" if qualified else "device-resident",
        "crossCacheOutputs": 8,
        "presentCacheOutputsPerStep": 8,
        "encoderCrossCacheHandoffs": token_count * 8,
        "decoderCacheHandoffs": max(0, token_count - 1) * 8,
        "runtimeValidatedDeviceInputs": True,
        "encoderResultRetainedThroughDecode": True,
        "decoderResultRetainedUntilSuccessorExecution": True,
        "encoderMemoryReadback": qualified,
        "encoderMemoryReadbackValidated": qualified,
        "encoderCrossCacheReadbackValidated": qualified,
        "cachePrefixReadbackValidated": qualified,
        "appendedCacheReadbackValidated": qualified,
        "cacheReadbackFree": not qualified,
    }


def sample(engine, backend, precision):
    timing = {
        "encoderExecutionMs": 10.0,
        "decoderSeedMs": 2.0,
        "decoderSteadySteps": 1,
        "decoderSteadyTotalMs": 1.0,
        "decoderSteadyMeanMs": 1.0,
        "decoderSteadyTokensPerSecond": 1000.0,
        "decoderExecutionTotalMs": 3.0,
        "decoderStepMs": [2.0, 1.0],
        "processWallMs": 20.0,
    }
    value = {
        "schema": "volvoxai.tiny-receipt-explicit-kv-runtime-sample/v1",
        "engine": engine,
        "backend": backend,
        "precision": precision,
        "provider": backend,
        "strictNoFallback": True,
        "family": "phone",
        "familyId": 0,
        "requestedFamily": "phone",
        "inputTensorSha256": "a" * 64,
        "questionTokenIds": list(QUESTION_IDS),
        "tokenIds": [4, 5],
        "stoppedAtEos": False,
        "shape": {"mode": "active", "B": 1, "Q": 3, "M": 213, "T": 5},
        "cache": {
            "initialPastLength": 1,
            "finalPastLength": 3,
            "sentinelMaskValue": 1,
            "transitions": [
                {"position": 0, "pastLength": 1, "presentLength": 2},
                {"position": 1, "pastLength": 2, "presentLength": 3},
            ],
        },
        "timing": timing,
        "runtime": {
            "executionWarmup": {
                "runs": 1,
                "sameSessions": True,
                "stateResetToSentinel": True,
                "tokenCacheTransitionParity": True,
            },
        },
    }
    if engine == "native-c":
        devices = {
            "vulkan": {
                "name": "NVIDIA GeForce RTX 3090",
                "packedInt8Dot": True,
            },
            "opengl": {
                "vendor": "NVIDIA Corporation",
                "renderer": "NVIDIA GeForce RTX 3090/PCIe/SSE2",
                "version": "4.6.0 NVIDIA",
            },
            "cuda": {
                "index": 0,
                "name": "NVIDIA GeForce RTX 3090",
                "computeCapability": "8.6",
            },
        }
        value["runtime"] = {
            "cpuThreads": 1,
            "device": devices[backend],
            "executionWarmup": {
                "runs": 1,
                "sameContexts": True,
                "stateResetToSentinel": True,
                "tokenCacheTransitionParity": True,
            },
        }
    elif engine == "volvoxai-webgpu":
        value["runtime"]["executionWarmup"] = {
            "runs": 1,
            "sameSession": True,
            "stateResetToSentinel": True,
            "tokenCacheTransitionParity": True,
        }
    return value


def webgpu_output():
    adapter = {
        "backend": "webgpu",
        "device": "Fixture Discrete GPU",
        "vendor": "fixture-vendor",
        "architecture": "fixture-discrete-gpu",
    }
    cache = sample("volvoxai-webgpu", "webgpu", "int8")["cache"]
    timing = dict(sample("volvoxai-webgpu", "webgpu", "int8")["timing"])
    timing.pop("processWallMs")
    result = {
        "status": "pass",
        "adapter": adapter,
        "packedDot4": True,
        "prompt": "phone number last one",
        "family": "phone",
        "precision": "int8",
        "maxNewTokens": 4,
        "warmup": 1,
        "iterations": 1,
        "inputTensorSha256": "a" * 64,
        "runtimeInitMs": 1.0,
        "sessionLoadMs": 2.0,
        "preprocessMs": 3.0,
        "compiles": [compilation("encoder", adapter), compilation("decoder", adapter)],
        "contextReuse": {
            "sameRuntime": True,
            "sameSession": True,
            "encoderContextCount": 1,
            "decoderContextCount": 1,
        },
        "cacheQualification": {
            "source": "warmup-0",
            "encoderMemoryReadbackValidated": True,
            "encoderCrossCacheReadbackValidated": True,
            "cachePrefixReadbackValidated": True,
            "appendedCacheReadbackValidated": True,
            "tokenCacheTransitionParity": True,
        },
        "warmupRuns": [{
            "name": "warmup-0",
            "generationMs": 16.0,
            "family": "phone",
            "familyId": 0,
            "requestedFamily": "phone",
            "questionTokenIds": list(QUESTION_IDS),
            "tokenIds": [4, 5],
            "stoppedAtEos": False,
            "shape": {"mode": "active", "B": 1, "Q": 3, "M": 213, "T": 5},
            "cache": cache,
            "gpuResidentKv": volvox_webgpu_kv_evidence(qualified=True),
            "executionAttestations": {
                "encoder": execution(),
                "decoder": [execution(), execution()],
            },
        }],
        "runs": [{
            "name": "measured-0",
            "generationMs": 15.0,
            "family": "phone",
            "familyId": 0,
            "requestedFamily": "phone",
            "questionTokenIds": list(QUESTION_IDS),
            "tokenIds": [4, 5],
            "stoppedAtEos": False,
            "shape": {"mode": "active", "B": 1, "Q": 3, "M": 213, "T": 5},
            "cache": cache,
            "gpuResidentKv": volvox_webgpu_kv_evidence(),
            "timing": {
                "synchronization": "required-small-output-readback",
                "applicationValidationIncluded": False,
                "cacheQualificationReadbackIncluded": False,
                **timing,
            },
            "executionPhases": {
                "encoder": {
                    "executionMs": 10.0,
                    "specializationCacheHit": True,
                    "specializationWrites": {
                        "writeCount": 4, "writeBytes": 64,
                        "skipCount": 2, "skipBytes": 32, "contentBytes": 64,
                    },
                },
                "decoder": [{
                    "executionMs": 2.0,
                    "specializationCacheHit": True,
                    "specializationWrites": {
                        "writeCount": 8, "writeBytes": 128,
                        "skipCount": 4, "skipBytes": 64, "contentBytes": 96,
                    },
                }, {
                    "executionMs": 1.0,
                    "specializationCacheHit": True,
                    "specializationWrites": {
                        "writeCount": 10, "writeBytes": 144,
                        "skipCount": 10, "skipBytes": 192, "contentBytes": 96,
                    },
                }],
            },
            "executionAttestations": {
                "encoder": execution(),
                "decoder": [execution(), execution()],
            },
            "tactics": {"encoder": [], "decoder": [[], []]},
        }],
    }
    return f"diagnostic\n{WEBGPU_RESULT_PREFIX}{json.dumps(result)}\n".encode()


def ort_webgpu_output(precision="int8", warmup=1):
    cache = sample("volvoxai-webgpu", "webgpu", precision)["cache"]
    timing = dict(sample("volvoxai-webgpu", "webgpu", precision)["timing"])
    timing.pop("processWallMs")
    adapter = {
        "vendor": "fixture-vendor",
        "architecture": "fixture-discrete-gpu",
        "device": "",
        "description": "",
    }
    strict_error = (
        "This session contains graph nodes that are assigned to the default CPU EP, "
        "but fallback to CPU EP has been explicitly disabled by the user."
    )
    placement_counts = {
        "fp32": {
            "encoder": {"CPUExecutionProvider": 70, "WebGpuExecutionProvider": 504},
            "decoder": {"CPUExecutionProvider": 3, "WebGpuExecutionProvider": 244},
        },
        "int8": {
            "encoder": {"CPUExecutionProvider": 282, "WebGpuExecutionProvider": 1223},
            "decoder": {"CPUExecutionProvider": 133, "WebGpuExecutionProvider": 676},
        },
    }[precision]

    def placement(role):
        providers = placement_counts[role]
        return {
            "providers": providers,
            "unsupportedKernelEvents": {},
            "records": [
                {"provider": provider, "nodes": nodes}
                for provider, nodes in reversed(list(providers.items()))
            ],
        }

    def run(measured, readback):
        cross_readbacks = 8 if readback else 0
        present_readbacks = 16 if readback else 0
        return {
            "measured": measured,
            "family": "phone",
            "familyId": 0,
            "requestedFamily": "phone",
            "questionTokenIds": list(QUESTION_IDS),
            "tokenIds": [4, 5],
            "stoppedAtEos": False,
            "shape": {"mode": "active", "B": 1, "Q": 3, "M": 213, "T": 5},
            "cache": cache,
            "timing": {
                "synchronization": "session-run-required-cpu-output-materialization",
                "applicationValidationIncluded": False,
                "cacheQualificationReadbackIncluded": False,
                **timing,
            },
            "endToEndInferenceMs": 15.0,
            "publicKvBufferHandoff": {
                "crossCacheOutputsOnGpuBuffer": 8,
                "presentCacheOutputsOnGpuBufferPerStep": 8,
                "encoderCrossCacheTensorObjectHandoffs": 16,
                "decoderPresentCacheTensorObjectHandoffs": 8,
                "totalDirectTensorObjectHandoffs": 24,
                "preferredOutputLocation": "gpu-buffer",
                "crossCacheReadbackValidated": readback,
                "cachePrefixReadbackValidated": readback,
                "appendedCacheReadbackValidated": readback,
                "harnessCacheReadbacks": {
                    "scope": "application-gpu-buffer-to-host-cache-validation",
                    "crossCacheTensorReadbacks": cross_readbacks,
                    "presentCacheTensorReadbacks": present_readbacks,
                    "totalCacheTensorReadbacks": cross_readbacks + present_readbacks,
                    "includedInExecutionTiming": False,
                    "includedInRequestEndToEndInferenceMs": readback,
                },
                "measuredHarnessCacheReadbacks": 0,
                "internalEpTransfersAttested": False,
            },
        }

    result = {
        "status": "pass",
        "runtime": {"name": "onnxruntime-web", "version": "1.27.0"},
        "backend": "webgpu-cpu-fallback",
        "strictNoFallback": False,
        "adapter": adapter,
        "adapterPreflight": adapter,
        "actualAdapterObjectAttested": True,
        "precision": precision,
        "prompt": "phone number last one",
        "family": "phone",
        "maxNewTokens": 4,
        "warmup": warmup,
        "iterations": 1,
        "inputTensorSha256": "a" * 64,
        "modelSha256": ort_webgpu_output_model_hashes(precision),
        "preprocessMs": 3.0,
        "sessionLoadMs": {"encoder": 10.0, "decoder": 5.0},
        "placement": {"encoder": placement("encoder"), "decoder": placement("decoder")},
        "strictProbe": {
            "encoder": {"rejected": True, "error": strict_error},
            "decoder": {"rejected": True, "error": strict_error},
        },
        "cacheQualification": {
            "source": "warmup-0" if warmup else "untimed-correctness-request",
            "crossCacheReadbackValidated": True,
            "cachePrefixReadbackValidated": True,
            "appendedCacheReadbackValidated": True,
            "tokenCacheTransitionParity": True,
            "harnessCacheReadbacks": {
                "scope": "application-gpu-buffer-to-host-cache-validation",
                "crossCacheTensorReadbacks": 8,
                "presentCacheTensorReadbacks": 16,
                "totalCacheTensorReadbacks": 24,
                "includedInExecutionTiming": False,
                "includedInRequestEndToEndInferenceMs": True,
            },
        },
        "sessionReuse": {
            "sameRuntime": True,
            "sameSessions": True,
            "encoderSessionCount": 1,
            "decoderSessionCount": 1,
        },
        "warmupRuns": [run(False, index == 0) for index in range(warmup)],
        "runs": [run(True, False)],
    }
    return f"diagnostic\n{WEBGPU_RESULT_PREFIX}{json.dumps(result)}\n".encode()


def mutate_ort_webgpu_output(stdout, mutate):
    prefix, payload = stdout.decode().split(WEBGPU_RESULT_PREFIX, 1)
    result = json.loads(payload)
    mutate(result)
    return f"{prefix}{WEBGPU_RESULT_PREFIX}{json.dumps(result)}\n".encode()


def dynamic_runs(include_web_fields=True):
    result = []
    values = [
        ("short-before", "active", 2, 2, [11, 2]),
        ("representative-active", "active", 8, 8, [*range(11, 18), 2]),
        ("representative-maximum-padded", "maximum-padded", 8, 192,
         [*range(11, 18), 2]),
        ("short-after", "active", 2, 2, [11, 2]),
    ]
    for label, mode, logical_q, bound_q, question_ids in values:
        run = {
            "label": label,
            "shapeMode": mode,
            "familyId": 0,
            "tokenIds": [4, 5],
            "activeShape": {"B": 1, "Q": bound_q, "M": bound_q + 210, "T": 5},
            "logicalShape": {
                "B": 1, "Q": logical_q, "M": logical_q + 210, "T": 5,
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
        if include_web_fields:
            run.update({
                "family": "phone",
                "requestedFamily": "phone",
                "questionTokenIds": question_ids,
            })
        result.append(run)
    return result


def native_qualification_output(backend="vulkan", device="Fixture Discrete GPU"):
    if backend == "vulkan":
        device_line = (
            "[VolvoxAI GPU] Vulkan Compute initialized successfully! Device: "
            f"{device}; packed INT8 dot: enabled\n"
        )
    elif backend == "opengl":
        device_line = (
            "[VolvoxAI GPU] OpenGL Compute initialized: Fixture Vendor / "
            f"{device} / 4.6 Fixture\n"
        )
    else:
        device_line = ""
    lines = [device_line.rstrip("\n")]
    shapes = [(2, 2), (8, 8), (8, 192), (2, 2)]
    for index, (logical_q, bound_q) in enumerate(shapes):
        mode = "maximum-padded" if index == 2 else "active"
        digest = "9a76a600c5543d20"
        lines.append(
            f"DYNAMIC_REBIND_RUN index={index} mode={mode} "
            f"logical_Q={logical_q} bound_Q={bound_q} "
            f"logical_M={logical_q + 210} bound_M={bound_q + 210} "
            f"family_id=0 tokens=2 token_digest={digest} token_ids=4,5 "
            "seed_P=1 seed_R=2 step_P=2 step_R=3 cache_preserved=1"
        )
    lines.append(
        f"DYNAMIC_REBIND_RESULT status=pass backend={backend} timed=0 "
        "same_runtime=1 same_encoder_context=1 same_decoder_context=1 "
        "strict_no_fallback=1 cpu_threads=1"
    )
    route = (
        f"provider=builtin:{backend};nodes=3;selected=3;"
        "fallback=0;missing=0;digest=fixture"
    )
    debug = []
    question_tokens = ("11,2", "11,12,13,14,15,16,17,2", "11,12,13,14,15,16,17,2", "11,2")
    for run_index in range(4):
        debug.append(f"[debug] tinyreceipt split input_f32_sha256={'a' * 64}")
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
    if backend == "cuda":
        debug.insert(0, "[CUDA] device 0: Fixture Discrete GPU (compute 8.6)")
    return ("\n".join(lines) + "\n").encode(), ("\n".join(debug) + "\n").encode()


def webgpu_qualification_result():
    adapter = {"backend": "webgpu", "device": "Fixture Discrete GPU"}
    runs = dynamic_runs()
    return {
        "status": "pass",
        "mode": "dynamic-rebind-qualification",
        "timed": False,
        "adapter": adapter,
        "packedDot4": True,
        "prompt": "phone number last one",
        "family": "phone",
        "precision": "int8",
        "maxNewTokens": 4,
        "inputTensorSha256": "a" * 64,
        "compiles": [compilation("encoder", adapter), compilation("decoder", adapter)],
        "qualification": {
            "schema": DYNAMIC_QUALIFICATION_SCHEMA,
            "timed": False,
            "boundedDecoderMaximumNewTokens": 4,
            "maximumLegalEncoderBinding": {"B": 1, "Q": 192, "M": 402},
            "sameSession": True,
            "sameRuntime": True,
            "sameEncoderContext": True,
            "sameDecoderContext": True,
            "strictNoFallback": True,
            "deviceResidentKVQualified": True,
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
            "runs": [
                {**run, "gpuResidentKv": volvox_webgpu_kv_evidence(
                    len(run["tokenIds"]), qualified=True,
                )}
                for run in runs
            ],
            "executionAttestations": [{
                "encoder": execution(),
                "decoder": [execution() for _ in run["tokenIds"]],
            } for run in runs],
        },
    }


def webgpu_qualification_output(result=None):
    value = webgpu_qualification_result() if result is None else result
    return f"diagnostic\n{WEBGPU_RESULT_PREFIX}{json.dumps(value)}\n".encode()


def ort_cpu_fallback_sample(precision):
    value = sample("onnxruntime", "cuda-cpu-fallback", precision)
    value["provider"] = "CUDAExecutionProvider"
    value["strictNoFallback"] = False
    value["runtime"] = {
        "providerOptions": {"device_id": "0", "use_tf32": "0"},
        "providerMode": "accelerator-with-cpu-fallback",
        "providerOrder": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "registeredEncoderProviders": [
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ],
        "registeredDecoderProviders": [
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ],
        "cpuEpFallbackAllowed": True,
        "cpuEpFallbackUsage": "exact-executed-node-placement-attested",
        "cpuEpFallbackDisabled": False,
        "providerPlacement": {
            role: {
                "executedNodeCount": 3,
                "providers": {
                    "CPUExecutionProvider": 1,
                    "CUDAExecutionProvider": 2,
                },
                "operatorCountsByProvider": {
                    "CPUExecutionProvider": {"Shape": 1},
                    "CUDAExecutionProvider": {"Add": 1, "MatMul": 1},
                },
                "nodeAssignmentSha256": ("a" if role == "encoder" else "b") * 64,
            }
            for role in ("encoder", "decoder")
        },
        "providerPlacementEvidence": {
            "source": "separate-untimed-profiling-sessions",
            "executionOrder": "after-measured-request",
            "sameMeasuredSessions": False,
            "measuredSessionsProfiled": False,
            "sessionConfigurationInvariant": "same graph and provider options",
            "requests": {
                "encoder": "canonical request",
                "decoder": "blocked P=1 seed request",
            },
        },
        "strictProbe": {
            "source": "separate-untimed-session-create-probes",
            "executionOrder": "after-measured-request",
            "roles": {
                role: {
                    "stage": "session-create",
                    "outcome": "rejected",
                    "cpuEpFallbackDisabled": True,
                    "errorType": "Fail",
                    "error": (
                        "nodes assigned to the default CPU EP; fallback to CPU EP "
                        "has been explicitly disabled"
                    ),
                }
                for role in ("encoder", "decoder")
            },
        },
        "executionWarmup": {
            "runs": 1,
            "sameSessions": True,
            "stateResetToSentinel": True,
            "tokenCacheTransitionParity": True,
        },
    }
    return value


class GpuBenchmarkHarnessTests(unittest.TestCase):
    def test_native_dynamic_qualification_is_strict_unique_and_physical(self):
        for backend in ("vulkan", "opengl", "cuda"):
            accepted_stdout, accepted_stderr = native_qualification_output(backend)
            accepted = parse_native_dynamic_qualification(
                accepted_stdout,
                accepted_stderr,
                precision="int8",
                backend=backend,
                max_new=4,
            )
            self.assertEqual(accepted["backend"], backend)
            if backend == "cuda":
                device_line = next(
                    line for line in accepted_stderr.splitlines()
                    if line.startswith(b"[CUDA] device ")
                )
                duplicate_stdout = accepted_stdout
                duplicate_stderr = accepted_stderr + device_line + b"\n"
                conflicting_stdout = accepted_stdout
                conflicting_stderr = accepted_stderr + (
                    b"[CUDA] device 1: Other Physical GPU (compute 9.0)\n"
                )
            else:
                prefix = (
                    b"[VolvoxAI GPU] Vulkan Compute initialized"
                    if backend == "vulkan"
                    else b"[VolvoxAI GPU] OpenGL Compute initialized"
                )
                device_line = next(
                    line for line in accepted_stdout.splitlines()
                    if line.startswith(prefix)
                )
                duplicate_stdout = accepted_stdout + device_line + b"\n"
                duplicate_stderr = accepted_stderr
                conflicting_stderr = accepted_stderr
                conflicting_stdout = accepted_stdout + (
                    b"[VolvoxAI GPU] Vulkan Compute initialized successfully! "
                    b"Device: Other Physical GPU; packed INT8 dot: enabled\n"
                    if backend == "vulkan"
                    else b"[VolvoxAI GPU] OpenGL Compute initialized: Other Vendor / "
                    b"Other Physical GPU / 4.6 Other\n"
                )
            duplicate = parse_native_dynamic_qualification(
                duplicate_stdout,
                duplicate_stderr,
                precision="int8",
                backend=backend,
                max_new=4,
            )
            self.assertEqual(duplicate["runtime"]["device"], accepted["runtime"]["device"])
            with self.assertRaisesRegex(Exception, "conflicting devices"):
                parse_native_dynamic_qualification(
                    conflicting_stdout,
                    conflicting_stderr,
                    precision="int8",
                    backend=backend,
                    max_new=4,
                )

        stdout, stderr = native_qualification_output()
        value = parse_native_dynamic_qualification(
            stdout, stderr, precision="int8", backend="vulkan", max_new=4
        )
        self.assertFalse(value["timed"])
        self.assertTrue(value["freshProcess"])
        self.assertEqual(len(value["runs"]), 4)
        self.assertEqual(value["runs"][2]["activeShape"]["M"], 402)
        self.assertEqual(len(value["routeEvidence"]["decoder"]), 8)
        self.assertEqual(
            value["routeEvidence"]["decoderSteps"][0][1],
            {"position": 1, "pastLength": 2, "presentLength": 3, "tokenId": 5},
        )

        result_line = next(
            line for line in stdout.splitlines()
            if line.startswith(b"DYNAMIC_REBIND_RESULT")
        )
        duplicate = stdout + result_line + b"\n"
        with self.assertRaisesRegex(Exception, "unique evidence"):
            parse_native_dynamic_qualification(
                duplicate, stderr, precision="int8", backend="vulkan", max_new=4
            )

        missing = b"\n".join(
            line for line in stdout.splitlines()
            if not line.startswith(b"DYNAMIC_REBIND_RUN index=2 ")
        ) + b"\n"
        with self.assertRaisesRegex(Exception, "unique evidence"):
            parse_native_dynamic_qualification(
                missing, stderr, precision="int8", backend="vulkan", max_new=4
            )

        malformed = stdout.replace(b"step_R=3", b"step_R=4", 1)
        with self.assertRaisesRegex(Exception, "run 0 is inconsistent"):
            parse_native_dynamic_qualification(
                malformed, stderr, precision="int8", backend="vulkan", max_new=4
            )

        software, software_stderr = native_qualification_output(device="lavapipe")
        with self.assertRaisesRegex(Exception, "software adapter"):
            parse_native_dynamic_qualification(
                software,
                software_stderr,
                precision="int8",
                backend="vulkan",
                max_new=4,
            )

        fallback = stderr.replace(b"fallback=0", b"fallback=1", 1)
        with self.assertRaisesRegex(Exception, "strict fallback 0"):
            parse_native_dynamic_qualification(
                stdout, fallback, precision="int8", backend="vulkan", max_new=4
            )

        input_line = (
            b"[debug] tinyreceipt split input_f32_sha256=" + b"a" * 64 + b"\n"
        )
        missing_hash = stderr.replace(input_line, b"", 1)
        with self.assertRaisesRegex(Exception, "input/step/route"):
            parse_native_dynamic_qualification(
                stdout,
                missing_hash,
                precision="int8",
                backend="vulkan",
                max_new=4,
            )
        mismatched_hash = stderr.replace(b"a" * 64, b"b" * 64, 1)
        with self.assertRaisesRegex(Exception, "input/step/route"):
            parse_native_dynamic_qualification(
                stdout,
                mismatched_hash,
                precision="int8",
                backend="vulkan",
                max_new=4,
            )

        corrupt_step = stderr.replace(b"token=4 time=", b"token=9 time=", 1)
        with self.assertRaisesRegex(Exception, "contradicts its machine evidence"):
            parse_native_dynamic_qualification(
                stdout,
                corrupt_step,
                precision="int8",
                backend="vulkan",
                max_new=4,
            )

    def test_webgpu_dynamic_qualification_rejects_malformed_evidence(self):
        self.assertEqual(
            DYNAMIC_QUALIFICATION_SCHEMA,
            "volvoxai.tiny-receipt-dynamic-shape-qualification/v1",
        )
        value = parse_webgpu_dynamic_qualification(
            webgpu_qualification_output(),
            precision="int8",
            prompt="phone number last one",
            family="phone",
            max_new=4,
        )
        self.assertFalse(value["timed"])
        self.assertTrue(value["freshBrowser"])
        self.assertEqual(value["runs"][0]["tokenIds"], value["runs"][3]["tokenIds"])

        unknown_schema = webgpu_qualification_result()
        unknown_schema["qualification"]["schema"] = (
            "invalid.dynamic-qualification-schema"
        )
        with self.assertRaisesRegex(Exception, "dynamic qualification"):
            parse_webgpu_dynamic_qualification(
                webgpu_qualification_output(unknown_schema),
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
            )

        duplicate = webgpu_qualification_output() + webgpu_qualification_output()
        with self.assertRaisesRegex(Exception, "expected one result"):
            parse_webgpu_dynamic_qualification(
                duplicate,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
            )

        missing = webgpu_qualification_result()
        missing["qualification"]["runs"].pop()
        with self.assertRaisesRegex(Exception, "exactly four runs"):
            parse_webgpu_dynamic_qualification(
                webgpu_qualification_output(missing),
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
            )

        malformed = webgpu_qualification_result()
        malformed["qualification"]["runs"][2]["activeShape"]["M"] = 401
        with self.assertRaisesRegex(Exception, "wrong active binding"):
            parse_webgpu_dynamic_qualification(
                webgpu_qualification_output(malformed),
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
            )

        software = webgpu_qualification_output().replace(
            b"Fixture Discrete GPU", b"SwiftShader Device"
        )
        with self.assertRaisesRegex(Exception, "software adapter"):
            parse_webgpu_dynamic_qualification(
                software,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
            )

    def test_dynamic_qualification_coverage_is_complete_and_untimed(self):
        native_stdout, native_stderr = native_qualification_output()
        native_entries = []
        for precision in ("fp32", "int8"):
            native_entries.append(parse_native_dynamic_qualification(
                native_stdout,
                native_stderr,
                precision=precision,
                backend="vulkan",
                max_new=4,
            ))
        web_entries = []
        for precision in ("fp32", "int8"):
            result = webgpu_qualification_result()
            result["precision"] = precision
            web_entries.append(parse_webgpu_dynamic_qualification(
                webgpu_qualification_output(result),
                precision=precision,
                prompt="phone number last one",
                family="phone",
                max_new=4,
            ))
        entries = [*native_entries, *web_entries]
        proof = prove_dynamic_qualification_coverage(
            entries, native_backends=["vulkan"], include_webgpu=True,
            expected_question_token_ids=(
                [11, 2], [*range(11, 18), 2], [*range(11, 18), 2], [11, 2],
            ),
        )
        self.assertEqual(proof["status"], "pass")
        self.assertFalse(proof["timed"])
        self.assertEqual(len(proof["entries"]), 4)

        with self.assertRaisesRegex(Exception, "coverage is incomplete"):
            prove_dynamic_qualification_coverage(
                entries[:-1], native_backends=["vulkan"], include_webgpu=True,
                expected_question_token_ids=(
                    [11, 2], [*range(11, 18), 2], [*range(11, 18), 2], [11, 2],
                ),
            )
        with self.assertRaisesRegex(Exception, "coverage is incomplete"):
            prove_dynamic_qualification_coverage(
                [*entries, entries[0]],
                native_backends=["vulkan"],
                include_webgpu=True,
                expected_question_token_ids=(
                    [11, 2], [*range(11, 18), 2], [*range(11, 18), 2], [11, 2],
                ),
            )
        entries[-1]["inputTensorSha256"] = "b" * 64
        with self.assertRaisesRegex(Exception, "input hash parity"):
            prove_dynamic_qualification_coverage(
                entries, native_backends=["vulkan"], include_webgpu=True,
                expected_question_token_ids=(
                    [11, 2], [*range(11, 18), 2], [*range(11, 18), 2], [11, 2],
                ),
            )
        entries[-1]["inputTensorSha256"] = "a" * 64
        entries[-1]["runs"][3]["tokenIds"] = [4, 6]
        with self.assertRaisesRegex(Exception, "parity failed"):
            prove_dynamic_qualification_coverage(
                entries, native_backends=["vulkan"], include_webgpu=True,
                expected_question_token_ids=(
                    [11, 2], [*range(11, 18), 2], [*range(11, 18), 2], [11, 2],
                ),
            )
        entries[-1]["runs"][3]["tokenIds"] = [4, 1038, 5, 6]
        del entries[0]["runs"][0]["questionTokenIds"]
        with self.assertRaisesRegex(Exception, "questionTokenIds"):
            prove_dynamic_qualification_coverage(
                entries, native_backends=["vulkan"], include_webgpu=True,
                expected_question_token_ids=(
                    [11, 2], [*range(11, 18), 2], [*range(11, 18), 2], [11, 2],
                ),
            )

    def test_native_gpu_samples_require_physical_device_identity(self):
        for backend in ("vulkan", "opengl", "cuda"):
            value = sample("native-c", backend, "int8")
            self.assertEqual(
                _physical_native_gpu_device(value),
                value["runtime"]["device"],
            )

        software_devices = [
            ("vulkan", {"name": "llvmpipe (LLVM 18.1.8)"}),
            ("vulkan", {"name": "llvmpipe-compatible Lavapipe GPU"}),
            ("vulkan", {"name": "Mesa softpipe"}),
            ("opengl", {"vendor": "Google", "renderer": "SwiftShader Device"}),
            ("opengl", {"vendor": "Mesa", "renderer": "Software Rasterizer"}),
            ("opengl", {"vendor": "Fixture", "renderer": "Software Renderer"}),
        ]
        for backend, device in software_devices:
            with self.subTest(backend=backend, device=device):
                value = sample("native-c", backend, "int8")
                value["runtime"]["device"] = device
                with self.assertRaisesRegex(Exception, "software adapter"):
                    _physical_native_gpu_device(value)

        missing = sample("native-c", "vulkan", "int8")
        missing["runtime"] = {}
        with self.assertRaisesRegex(Exception, "identity is unavailable"):
            _physical_native_gpu_device(missing)
        incomplete = sample("native-c", "opengl", "int8")
        incomplete["runtime"]["device"] = {"vendor": "NVIDIA Corporation"}
        with self.assertRaisesRegex(Exception, "renderer"):
            _physical_native_gpu_device(incomplete)

    def test_webgpu_sample_requires_physical_strict_single_browser_run(self):
        value = parse_webgpu_sample(
            webgpu_output(),
            precision="int8",
            prompt="phone number last one",
            family="phone",
            max_new=4,
            process_wall_ms=30.0,
        )
        self.assertEqual(value["engine"], "volvoxai-webgpu")
        self.assertEqual(value["runtime"]["adapter"]["device"], "Fixture Discrete GPU")
        self.assertTrue(value["runtime"]["freshBrowser"])
        self.assertEqual(value["runtime"]["executionWarmup"]["runs"], 1)
        self.assertEqual(value["timing"]["processWallMs"], 30.0)
        self.assertEqual(
            value["specializationWrites"]["decoder"][-1]["skipBytes"], 192,
        )

        fake_memory_qualification = webgpu_output().replace(
            b'"encoderMemoryReadback": true, "encoderMemoryReadbackValidated": true',
            b'"encoderMemoryReadback": true, "encoderMemoryReadbackValidated": false',
            1,
        )
        with self.assertRaisesRegex(Exception, "explicit-KV handoff/readback evidence"):
            parse_webgpu_sample(
                fake_memory_qualification,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        application_timed = webgpu_output().replace(
            b'"applicationValidationIncluded": false',
            b'"applicationValidationIncluded": true',
            1,
        )
        with self.assertRaisesRegex(Exception, "required small-output readback"):
            parse_webgpu_sample(
                application_timed,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        warmup_token_mismatch = webgpu_output().replace(
            b'"name": "warmup-0"', b'"name": "warmup-0"', 1
        ).replace(b'"tokenIds": [4, 5]', b'"tokenIds": [4, 6]', 1)
        with self.assertRaisesRegex(Exception, "warmup run 0 failed token/cache parity"):
            parse_webgpu_sample(
                warmup_token_mismatch,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        cold_measured = webgpu_output().replace(
            b'"specializationCacheHit": true',
            b'"specializationCacheHit": false',
            1,
        )
        with self.assertRaisesRegex(Exception, "warmed shape specialization"):
            parse_webgpu_sample(
                cold_measured,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        warmup_fallback = webgpu_output().replace(
            b'"tierFallback": false', b'"tierFallback": true', 7
        )
        with self.assertRaisesRegex(Exception, "fallback 0"):
            parse_webgpu_sample(
                warmup_fallback,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

    def test_ort_webgpu_sample_is_pinned_partitioned_and_gpu_resident(self):
        value = parse_ort_webgpu_sample(
            ort_webgpu_output(),
            precision="int8",
            prompt="phone number last one",
            family="phone",
            max_new=4,
            process_wall_ms=30.0,
            execution_warmup=1,
            runtime_identity={"version": "1.27.0"},
        )
        self.assertEqual(value["engine"], "onnxruntime-web")
        self.assertEqual(value["backend"], "webgpu-cpu-fallback")
        self.assertFalse(value["strictNoFallback"])
        self.assertEqual(
            value["runtime"]["placement"]["encoder"]["providers"]
            ["WebGpuExecutionProvider"],
            1223,
        )
        self.assertEqual(
            value["runtime"]["publicKvBufferHandoff"][
                "measuredHarnessCacheReadbacks"
            ],
            0,
        )
        self.assertEqual(
            {
                name: value["runtime"]["publicKvBufferHandoff"][name]
                for name in (
                    "encoderCrossCacheTensorObjectHandoffs",
                    "decoderPresentCacheTensorObjectHandoffs",
                    "totalDirectTensorObjectHandoffs",
                )
            },
            {
                "encoderCrossCacheTensorObjectHandoffs": 16,
                "decoderPresentCacheTensorObjectHandoffs": 8,
                "totalDirectTensorObjectHandoffs": 24,
            },
        )
        self.assertEqual(
            value["runtime"]["publicKvBufferHandoff"]["harnessCacheReadbacks"],
            {
                "scope": "application-gpu-buffer-to-host-cache-validation",
                "crossCacheTensorReadbacks": 0,
                "presentCacheTensorReadbacks": 0,
                "totalCacheTensorReadbacks": 0,
                "includedInExecutionTiming": False,
                "includedInRequestEndToEndInferenceMs": False,
            },
        )
        self.assertEqual(
            value["runtime"]["cacheQualification"]["harnessCacheReadbacks"],
            {
                "scope": "application-gpu-buffer-to-host-cache-validation",
                "crossCacheTensorReadbacks": 8,
                "presentCacheTensorReadbacks": 16,
                "totalCacheTensorReadbacks": 24,
                "includedInExecutionTiming": False,
                "includedInRequestEndToEndInferenceMs": True,
            },
        )
        self.assertFalse(
            value["runtime"]["publicKvBufferHandoff"][
                "internalEpTransfersAttested"
            ]
        )
        self.assertEqual(
            value["runtime"]["placementEvidence"]["modelSha256"],
            ort_webgpu_output_model_hashes("int8"),
        )

        cold = parse_ort_webgpu_sample(
            ort_webgpu_output(warmup=0),
            precision="int8",
            prompt="phone number last one",
            family="phone",
            max_new=4,
            process_wall_ms=30.0,
            execution_warmup=0,
        )
        self.assertEqual(cold["runtime"]["executionWarmup"]["runs"], 0)
        self.assertEqual(
            cold["runtime"]["cacheQualification"]["source"],
            "untimed-correctness-request",
        )
        self.assertEqual(
            cold["runtime"]["cacheQualification"]["harnessCacheReadbacks"]
            ["presentCacheTensorReadbacks"],
            16,
        )

        bad_placement = ort_webgpu_output().replace(
            b'"WebGpuExecutionProvider": 1223',
            b'"WebGpuExecutionProvider": 1222',
            1,
        )
        with self.assertRaisesRegex(Exception, "pinned graph partition"):
            parse_ort_webgpu_sample(
                bad_placement,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        wrong_model = ort_webgpu_output().replace(
            b'a5be30f7507f8e2988c4cf735a80fb6c09a4f1e93c9759b1fc5e12d04c4da45c',
            b'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            1,
        )
        with self.assertRaisesRegex(Exception, "producer model SHA"):
            parse_ort_webgpu_sample(
                wrong_model,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        wrong_strict_reason = ort_webgpu_output().replace(
            b"assigned to the default CPU EP", b"GPU device was lost", 1,
        )
        with self.assertRaisesRegex(Exception, "strict probe"):
            parse_ort_webgpu_sample(
                wrong_strict_reason,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        software = ort_webgpu_output().replace(
            b"fixture-discrete-gpu", b"swiftshader-device"
        )
        with self.assertRaisesRegex(Exception, "software adapter"):
            parse_ort_webgpu_sample(
                software,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        measured_readback = ort_webgpu_output().replace(
            b'"cachePrefixReadbackValidated": false',
            b'"cachePrefixReadbackValidated": true',
            1,
        )
        with self.assertRaisesRegex(Exception, "public gpu-buffer"):
            parse_ort_webgpu_sample(
                measured_readback,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        software = webgpu_output().replace(b"Fixture Discrete GPU", b"SwiftShader GPU")
        with self.assertRaisesRegex(Exception, "software adapter"):
            parse_webgpu_sample(
                software,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        fallback = webgpu_output().replace(
            b'"tierFallback": false', b'"tierFallback": true', 1
        )
        with self.assertRaisesRegex(Exception, "fallback 0"):
            parse_webgpu_sample(
                fallback,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

    def test_ort_webgpu_readback_evidence_is_phase_scoped_and_exact(self):
        def add_nonvalidating_warmup_readback(result):
            evidence = result["warmupRuns"][1]["publicKvBufferHandoff"]
            evidence["harnessCacheReadbacks"].update({
                "crossCacheTensorReadbacks": 8,
                "totalCacheTensorReadbacks": 8,
                "includedInRequestEndToEndInferenceMs": True,
            })

        false_warmup_readback = mutate_ort_webgpu_output(
            ort_webgpu_output(warmup=2), add_nonvalidating_warmup_readback,
        )
        with self.assertRaisesRegex(
            Exception, "warmup run 1 public gpu-buffer handoff/readback evidence",
        ):
            parse_ort_webgpu_sample(
                false_warmup_readback,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
                execution_warmup=2,
            )

        def add_measured_readback(result):
            evidence = result["runs"][0]["publicKvBufferHandoff"]
            evidence["harnessCacheReadbacks"].update({
                "presentCacheTensorReadbacks": 8,
                "totalCacheTensorReadbacks": 8,
                "includedInRequestEndToEndInferenceMs": True,
            })

        measured_readback = mutate_ort_webgpu_output(
            ort_webgpu_output(), add_measured_readback,
        )
        with self.assertRaisesRegex(
            Exception, "measured run public gpu-buffer handoff/readback evidence",
        ):
            parse_ort_webgpu_sample(
                measured_readback,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        def omit_cross_cache_handoffs(result):
            evidence = result["runs"][0]["publicKvBufferHandoff"]
            evidence["encoderCrossCacheTensorObjectHandoffs"] = 8
            evidence["totalDirectTensorObjectHandoffs"] = 16

        incomplete_handoffs = mutate_ort_webgpu_output(
            ort_webgpu_output(), omit_cross_cache_handoffs,
        )
        with self.assertRaisesRegex(
            Exception, "measured run public gpu-buffer handoff/readback evidence",
        ):
            parse_ort_webgpu_sample(
                incomplete_handoffs,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
            )

        def corrupt_cold_qualification_total(result):
            result["cacheQualification"]["harnessCacheReadbacks"][
                "totalCacheTensorReadbacks"
            ] = 16

        bad_cold_qualification = mutate_ort_webgpu_output(
            ort_webgpu_output(warmup=0), corrupt_cold_qualification_total,
        )
        with self.assertRaisesRegex(Exception, "untimed cache-prefix/readback"):
            parse_ort_webgpu_sample(
                bad_cold_qualification,
                precision="int8",
                prompt="phone number last one",
                family="phone",
                max_new=4,
                process_wall_ms=30.0,
                execution_warmup=0,
            )

    def test_matrix_parity_includes_backend_precision_hash_tokens_and_cache(self):
        values = [
            sample("onnxruntime", "cpu", "fp32"),
            sample("native-c", "cuda", "fp32"),
            sample("volvoxai-webgpu", "webgpu", "fp32"),
        ]
        tiers = [
            "onnxruntime/cpu/fp32",
            "native-c/cuda/fp32",
            "volvoxai-webgpu/webgpu/fp32",
        ]
        parity = prove_gpu_parity(
            values, expected_tiers=tiers, repeat=1, max_new=4,
            expected_execution_warmup=1,
            expected_question_token_ids=QUESTION_IDS,
        )
        self.assertTrue(parity["exact"])
        self.assertIn("active-shape-binding", parity["exactScope"])
        self.assertIn("not a corpus-level", parity["accuracyEvidence"])
        self.assertEqual(parity["comparisons"]["fp32"]["cuda"]["status"], "not-measured")
        self.assertEqual(
            parity["comparisons"]["fp32"]["vulkan"]["status"],
            "not-available",
        )
        summaries = summarize_gpu_samples(values)
        self.assertEqual(set(summaries), set(tiers))
        self.assertEqual(
            summaries["onnxruntime/cpu/fp32"]["componentTotal"]["medianMs"],
            13.0,
        )
        self.assertTrue(
            summaries["onnxruntime/cpu/fp32"]["strictNoFallback"]
        )
        values[1]["cache"]["transitions"][1]["pastLength"] = 1
        with self.assertRaisesRegex(Exception, "cache transition|parity"):
            prove_gpu_parity(
                values, expected_tiers=tiers, repeat=1, max_new=4,
                expected_question_token_ids=QUESTION_IDS,
            )
        values[1]["cache"]["transitions"][1]["pastLength"] = 2
        values[1]["runtime"]["executionWarmup"]["runs"] = 0
        with self.assertRaisesRegex(Exception, "execution-warmup evidence"):
            prove_gpu_parity(
                values, expected_tiers=tiers, repeat=1, max_new=4,
                expected_execution_warmup=1,
                expected_question_token_ids=QUESTION_IDS,
            )
        values[1]["runtime"]["executionWarmup"]["runs"] = 1
        values[1]["runtime"]["device"]["name"] = "lavapipe"
        with self.assertRaisesRegex(Exception, "software adapter"):
            prove_gpu_parity(
                values, expected_tiers=tiers, repeat=1, max_new=4,
                expected_question_token_ids=QUESTION_IDS,
            )

        values[1]["runtime"]["device"]["name"] = "NVIDIA GeForce RTX 3090"
        values[0]["questionTokenIds"] = [11, 13, 2]
        with self.assertRaisesRegex(Exception, "canonical question-token parity"):
            prove_gpu_parity(
                values, expected_tiers=tiers, repeat=1, max_new=4,
                expected_question_token_ids=QUESTION_IDS,
            )

    def test_direct_webgpu_comparison_requires_matching_adapter_class(self):
        ort = parse_ort_webgpu_sample(
            ort_webgpu_output(precision="fp32"),
            precision="fp32",
            prompt="phone number last one",
            family="phone",
            max_new=4,
            process_wall_ms=30.0,
        )
        volvox = sample("volvoxai-webgpu", "webgpu", "fp32")
        volvox["runtime"]["adapter"] = {
            "vendor": "fixture-vendor",
            "architecture": "fixture-discrete-gpu",
        }
        tiers = [
            "onnxruntime-web/webgpu-cpu-fallback/fp32",
            "volvoxai-webgpu/webgpu/fp32",
        ]
        comparison = prove_gpu_parity(
            [ort, volvox],
            expected_tiers=tiers,
            repeat=1,
            max_new=4,
            expected_execution_warmup=1,
            expected_question_token_ids=QUESTION_IDS,
        )["comparisons"]["fp32"]["webgpu"]
        self.assertEqual(comparison["status"], "direct")
        self.assertTrue(comparison["adapterClassIdentityMatch"])
        self.assertFalse(comparison["samePhysicalAdapterAttested"])

        volvox["runtime"]["adapter"]["architecture"] = "different-gpu"
        with self.assertRaisesRegex(Exception, "adapter classes differ"):
            prove_gpu_parity(
                [ort, volvox],
                expected_tiers=tiers,
                repeat=1,
                max_new=4,
                expected_execution_warmup=1,
                expected_question_token_ids=QUESTION_IDS,
            )
    def test_provider_options_and_matrix_defaults_are_deterministic(self):
        self.assertEqual(
            REPORT_FORMAT,
            "volvoxai.tiny-receipt-explicit-kv-gpu-benchmark/v1",
        )
        self.assertEqual(
            _provider_options("CUDAExecutionProvider", []),
            {"device_id": "0", "use_tf32": "0"},
        )
        self.assertEqual(
            _provider_options("CPUExecutionProvider", ["arena_extend_strategy=1"]),
            {"arena_extend_strategy": "1"},
        )
        with self.assertRaisesRegex(Exception, "duplicate"):
            _provider_options("CPUExecutionProvider", ["x=1", "x=2"])

        args = _parse_public_arguments([
            "--source", "source",
            "--fp32-package", "fp32",
            "--int8-package", "int8",
        ])
        self.assertEqual(args.native_backend, ["vulkan", "opengl", "cuda"])
        self.assertEqual(args.ort_provider, "CPUExecutionProvider")
        self.assertIsNone(args.ort_web_root)
        self.assertFalse(args.allow_ort_cpu_fallback)
        self.assertFalse(args.no_webgpu)
        self.assertEqual(args.warmup, 1)
        self.assertEqual(args.repeat, 5)
        tiers = ["ort/fp32", "native/fp32", "ort/int8", "native/int8"]
        self.assertEqual(_counterbalanced_tier_order(tiers, 0), tiers)
        self.assertEqual(_counterbalanced_tier_order(tiers, 1), tiers[::-1])
        self.assertEqual(
            _counterbalanced_tier_order(tiers, 2),
            ["native/fp32", "ort/int8", "native/int8", "ort/fp32"],
        )
        self.assertEqual(
            _counterbalanced_tier_order(tiers, 3),
            ["ort/fp32", "native/int8", "ort/int8", "native/fp32"],
        )
        for sample_index in range(8):
            self.assertCountEqual(
                _counterbalanced_tier_order(tiers, sample_index), tiers
            )
        ort_web_args = _parse_public_arguments([
            "--source", "source",
            "--fp32-package", "fp32",
            "--int8-package", "int8",
            "--ort-web-root", "external-ort-web",
        ])
        self.assertEqual(ort_web_args.ort_web_root.name, "external-ort-web")
        self.assertEqual(
            _expected_tiers(
                ort_web_args.ort_provider,
                ["vulkan"],
                include_webgpu=False,
                use_ort_webgpu=True,
            ),
            [
                "onnxruntime-web/webgpu-cpu-fallback/fp32",
                "native-c/vulkan/fp32",
                "onnxruntime-web/webgpu-cpu-fallback/int8",
                "native-c/vulkan/int8",
            ],
        )
        for warmup in (0, 20):
            bounded = _parse_public_arguments([
                "--source", "source",
                "--fp32-package", "fp32",
                "--int8-package", "int8",
                "--warmup", str(warmup),
            ])
            self.assertEqual(bounded.warmup, warmup)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parse_public_arguments([
                    "--source", "source",
                    "--fp32-package", "fp32",
                    "--int8-package", "int8",
                    "--warmup", "21",
                ])
            with self.assertRaises(SystemExit):
                _parse_public_arguments([
                    "--source", "source",
                    "--fp32-package", "fp32",
                    "--int8-package", "int8",
                    "--ort-web-root", "external-ort-web",
                    "--ort-threads", "2",
                ])

    def test_optional_ort_cpu_fallback_is_accelerator_first_and_honestly_labeled(self):
        self.assertEqual(
            _ort_provider_registration(
                "CUDAExecutionProvider", {"device_id": "0"}, True
            ),
            [
                ("CUDAExecutionProvider", {"device_id": "0"}),
                "CPUExecutionProvider",
            ],
        )
        with self.assertRaisesRegex(Exception, "non-CPU primary"):
            _ort_provider_registration("CPUExecutionProvider", {}, True)

        args = _parse_public_arguments([
            "--source", "source",
            "--fp32-package", "fp32",
            "--int8-package", "int8",
            "--ort-provider", "CUDAExecutionProvider",
            "--allow-ort-cpu-fallback",
            "--no-webgpu",
            "--native-backend", "cuda",
        ])
        self.assertTrue(args.allow_ort_cpu_fallback)
        self.assertTrue(args.no_webgpu)
        self.assertEqual(args.native_backend, ["cuda"])
        tiers = _expected_tiers(
            args.ort_provider,
            args.native_backend,
            include_webgpu=not args.no_webgpu,
            allow_ort_cpu_fallback=args.allow_ort_cpu_fallback,
        )
        self.assertEqual(tiers, [
            "onnxruntime/cuda-cpu-fallback/fp32",
            "native-c/cuda/fp32",
            "onnxruntime/cuda-cpu-fallback/int8",
            "native-c/cuda/int8",
        ])

        values = [
            ort_cpu_fallback_sample("fp32"),
            sample("native-c", "cuda", "fp32"),
            ort_cpu_fallback_sample("int8"),
            sample("native-c", "cuda", "int8"),
        ]
        parity = prove_gpu_parity(
            values,
            expected_tiers=tiers,
            repeat=1,
            max_new=4,
            allow_ort_cpu_fallback=True,
            expected_execution_warmup=1,
            expected_question_token_ids=QUESTION_IDS,
        )
        self.assertTrue(parity["exact"])
        self.assertEqual(
            parity["comparisons"]["fp32"]["cuda"]["status"], "direct"
        )
        self.assertFalse(
            parity["comparisons"]["fp32"]["cuda"][
                "physicalDeviceUuidAttested"
            ]
        )
        self.assertFalse(
            summarize_gpu_samples(values)[
                "onnxruntime/cuda-cpu-fallback/fp32"
            ]["strictNoFallback"]
        )
        values[0]["runtime"]["providerPlacement"]["encoder"][
            "executedNodeCount"
        ] = 2
        with self.assertRaisesRegex(Exception, "executed-node placement"):
            prove_gpu_parity(
                values,
                expected_tiers=tiers,
                repeat=1,
                max_new=4,
                allow_ort_cpu_fallback=True,
                expected_question_token_ids=QUESTION_IDS,
            )
        values[0]["runtime"]["providerPlacement"]["encoder"][
            "executedNodeCount"
        ] = 3
        values[0]["runtime"]["registeredEncoderProviders"].reverse()
        with self.assertRaisesRegex(Exception, "accelerator-first attestation"):
            prove_gpu_parity(
                values,
                expected_tiers=tiers,
                repeat=1,
                max_new=4,
                allow_ort_cpu_fallback=True,
                expected_question_token_ids=QUESTION_IDS,
            )

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parse_public_arguments([
                    "--source", "source",
                    "--fp32-package", "fp32",
                    "--int8-package", "int8",
                    "--allow-ort-cpu-fallback",
                ])

    def test_gpu_provenance_excludes_benchmark_publication_files(self):
        self.assertIn(
            "docs/tiny-receipt-vqa-bpe1536-benchmark.md",
            PUBLICATION_OUTPUT_PATHS,
        )
        self.assertIn(
            "examples/tiny_receipt_vqa/reports/explicit_kv_v1_gpu_matrix.json",
            PUBLICATION_OUTPUT_PATHS,
        )
        self.assertIn(
            "examples/tiny_receipt_vqa/reports/explicit_kv_v1_cuda_matrix.json",
            PUBLICATION_OUTPUT_PATHS,
        )
        self.assertIn(
            "examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix.json",
            PUBLICATION_OUTPUT_PATHS,
        )

    def test_cuda_report_uses_current_format_and_passed(self):
        path = Path(__file__).resolve().parents[1] / "reports" / (
            "explicit_kv_v1_cuda_matrix.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["format"], REPORT_FORMAT)
        self.assertEqual(report["status"], "pass")


if __name__ == "__main__":
    unittest.main()
