#!/usr/bin/env python3
"""Offline prepared-sensor replay through the generated native public API.

The fixture exercises tensor snapshots, LATEST stateless scheduling and ALL
state feedback. It is a numerical/ownership capture, not a robot policy model
or a latency benchmark. Source clocks, synchronization and freshness belong to
the caller; the engine only receives named tensors.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx

pb = vx.pb
POLICY_PATH = ROOT / "tools/llm_baseline/multimodal_sensor.json"


def policy() -> dict[str, Any]:
    return json.loads(POLICY_PATH.read_text())["sensor"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Observation:
    index: int
    source_ns: int
    tensors: dict[str, np.ndarray]


def observations(spec: dict[str, Any]) -> list[Observation]:
    random = np.random.default_rng(spec["seed"])
    result = []
    for index in range(spec["samples"]):
        source_ns = index * spec["period_ns"]
        tensors = {name: random.uniform(-1, 1, size=shape).astype(np.float32)
                   for name, shape in (("image_features", (1, 4)), ("imu", (1, 6)),
                                       ("proprioception", (1, 3)))}
        tensors["observation_time"] = np.array([[source_ns / 1e9]], dtype=np.float32)
        tensors["sample_index"] = np.array([index], dtype=np.int32)
        result.append(Observation(index, source_ns, tensors))
    return result


def latest_fresh(records: list[Observation], *, now_ns: int, max_age_ns: int) -> Observation | None:
    """Caller policy: reject stale/future clocks, choose time then sample index."""
    eligible = [record for record in records if 0 <= now_ns - record.source_ns <= max_age_ns]
    return max(eligible, key=lambda record: (record.source_ns, record.index)) if eligible else None


def fixture_weights() -> dict[str, np.ndarray]:
    # Explicit small coefficients make the NumPy oracle independent of the
    # engine graph builder, and exercise every prepared feature family.
    return {"image_weight": (np.arange(12, dtype=np.float32).reshape(4, 3) - 5) / 32,
            "imu_weight": (np.arange(18, dtype=np.float32).reshape(6, 3) - 8) / 64,
            "proprio_weight": np.array([[.25, -.125, .0625], [-.25, .5, .125],
                                        [.125, -.0625, .25]], dtype=np.float32),
            "time_weight": np.array([[.125, -.25, .5]], dtype=np.float32),
            "bias": np.array([.0625, -.125, .25], dtype=np.float32)}


def reference(record: Observation, state: np.ndarray | None = None) -> np.ndarray:
    values, weights = record.tensors, fixture_weights()
    result = (((values["image_features"] @ weights["image_weight"] + weights["bias"])
              + values["imu"] @ weights["imu_weight"])
              + values["proprioception"] @ weights["proprio_weight"])
    result = result + values["observation_time"] @ weights["time_weight"]
    return result if state is None else result + state


def write_fixture(directory: Path, *, stateful: bool) -> Path:
    from safetensors.numpy import save_file
    directory.mkdir(parents=True, exist_ok=True)
    inputs = {name: {"dtype": "float32", "shape": shape} for name, shape in
              (("image_features", [1, 4]), ("imu", [1, 6]), ("proprioception", [1, 3]),
               ("observation_time", [1, 1]))}
    inputs["sample_index"] = {"dtype": "int32", "shape": [1]}
    if stateful:
        inputs["state"] = {"dtype": "float32", "shape": [1, 3]}
        inputs["position"] = {"dtype": "int32", "shape": [1]}
    nodes = []

    def node(name: str, op: str, bindings: dict[str, str], *, shape=(1, 3),
             dtype="float32", params=None) -> None:
        nodes.append({"id": name, "opType": op, "inputs": bindings,
            "outputs": {"out": {"tensor": name, "dtype": dtype, "shape": list(shape)}},
            "params": params or {}})

    for name, source in (("image", "image_features"), ("imu", "imu"),
                          ("proprio", "proprioception"), ("time", "observation_time")):
        bindings = {"input": source, "weight": name + "_weight"}
        if name == "image":
            bindings["bias"] = "bias"
        node(name + "_projection", "Linear", bindings, params={"weight_layout": "din_dout"})
    node("sum_image_imu", "Add", {"a": "image_projection", "b": "imu_projection"})
    node("sum_features", "Add", {"a": "sum_image_imu", "b": "proprio_projection"})
    node("feature_output", "Add", {"a": "sum_features", "b": "time_projection"})
    if stateful:
        node("state_next", "Add", {"a": "feature_output", "b": "state"})
    node("echo_time", "Identity", {"input": "observation_time"}, shape=(1, 1))
    node("echo_sample_index", "Identity", {"input": "sample_index"}, shape=(1,), dtype="int32")
    outputs = ["state_next" if stateful else "feature_output", "echo_time", "echo_sample_index"]
    if stateful:
        node("echo_position", "Identity", {"input": "position"}, shape=(1,), dtype="int32")
        outputs.append("echo_position")
    entries = [("initialize", 2), ("transition", 3)] if stateful else [("evaluate", 1)]
    execution = {"entrypoints": [{"name": name, "effect": effect, "allowed_ranges": [1],
                                  "inputs": list(inputs), "outputs": outputs} for name, effect in entries]}
    if stateful:
        execution["tensor_feedback"] = {"position_input": "position",
            "bindings": [{"input_name": "state", "output_name": "state_next"}]}
    else:
        execution["stateless"] = {}
    graph = {"format": "volvox-graph/v1", "dimensions": {}, "inputs": inputs,
             "nodes": nodes, "outputs": outputs, "execution": execution}
    path = directory / "graph.json"
    path.write_text(json.dumps(graph, indent=2) + "\n")
    save_file(fixture_weights(), directory / "model.safetensors")
    return path


def _input_tensors(values: dict[str, np.ndarray]) -> list[Any]:
    result = []
    for name, value in values.items():
        if not value.flags.c_contiguous or value.dtype not in (np.dtype("float32"), np.dtype("int32")):
            raise ValueError("replay requires contiguous F32/I32 arrays")
        result.append(pb.Tensor(name=name, shape=list(value.shape),
            dtype=pb.DataType.DATA_TYPE_F32 if value.dtype == np.float32 else pb.DataType.DATA_TYPE_I32,
            borrowed=pb.BorrowedBuffer(resource=pb.NativeResource(
                kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST,
                handle=value.ctypes.data, size_bytes=value.nbytes), length_bytes=value.nbytes)))
    return result


def _execute(context: int, values: dict[str, np.ndarray], entrypoint: str, *, feedback=False,
             reuse=()) -> Any:
    return pb.ExecuteRequest(context_id=context,
        inputs=pb.InputBindings(values=_input_tensors(values), reuse=list(reuse), feedback=[
            pb.TensorFeedback(input_name="state", output_name="state_next")] if feedback else []),
        scope=pb.ExecutionScope(entrypoint=entrypoint, full=pb.Empty()),
        budget=pb.ExecutionBudget(max_output_bytes=4096),
        outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER))


def _copy_packet(buffers: Any, packet: Any) -> dict[str, np.ndarray]:
    arrays, destinations = {}, []
    for tensor in packet.outputs:
        dtype = {pb.DataType.DATA_TYPE_F32: np.float32, pb.DataType.DATA_TYPE_I32: np.int32}[tensor.dtype]
        value = np.empty(tuple(tensor.shape), dtype=dtype)
        if (not tensor.HasField("buffer") or tensor.buffer.buffer_id <= 0 or
                tensor.buffer.length_bytes != value.nbytes):
            raise ValueError("packet is not an independently owned dense snapshot")
        arrays[tensor.name] = value
        destinations.append(pb.BorrowedBuffer(resource=pb.NativeResource(
            kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST,
            handle=value.ctypes.data, size_bytes=value.nbytes), length_bytes=value.nbytes))
    if not destinations:
        return arrays
    copied = buffers.copy_tensors(pb.CopyTensorsRequest(sources=packet.outputs, into=destinations))
    if len(copied.outputs) != len(arrays):
        raise ValueError("copy receipt count differs from the output selection")
    for tensor, (name, value) in zip(copied.outputs, arrays.items()):
        if (tensor.name != name or tensor.borrowed is None or
                tensor.borrowed.resource.handle != value.ctypes.data or
                tensor.borrowed.length_bytes != value.nbytes or tuple(tensor.shape) != value.shape):
            raise ValueError("copy receipt differs from its exact destination")
    return arrays


def _release_packet(buffers: Any, packet: Any) -> None:
    owned = [tensor.buffer.buffer_id for tensor in packet.outputs if tensor.HasField("buffer")]
    if owned:
        buffers.release_buffers(pb.BufferRefs(buffer_ids=owned))


def _assert_output(actual: dict[str, np.ndarray], expected: np.ndarray, record: Observation,
                   spec: dict[str, Any], *, position=None) -> None:
    name = "state_next" if position is not None else "feature_output"
    np.testing.assert_allclose(actual[name], expected, atol=spec["tolerance"]["absolute"],
                               rtol=spec["tolerance"]["relative"])
    np.testing.assert_array_equal(actual["echo_time"], record.tensors["observation_time"])
    np.testing.assert_array_equal(actual["echo_sample_index"], record.tensors["sample_index"])
    if position is not None:
        np.testing.assert_array_equal(actual["echo_position"], np.array([position], dtype=np.int32))


def replay(directory: Path, *, backend="cpu") -> dict[str, Any]:
    spec = policy()
    records = observations(spec)
    from safetensors.numpy import save_file
    prepared_path = directory / "prepared_observations.safetensors"
    directory.mkdir(parents=True, exist_ok=True)
    prepared = {name: np.concatenate([record.tensors[name] for record in records], axis=0)
                for name in records[0].tensors}
    prepared["source_ns"] = np.array([record.source_ns for record in records], dtype=np.int64)
    save_file(prepared, str(prepared_path))
    paths = {name: write_fixture(directory / name, stateful=name == "stateful")
             for name in ("stateless", "stateful")}
    rows, latest_rows = [], []
    with vx.open_library() as host:
        inference = vx.VxInferenceServiceClient(host)
        scheduler = vx.VxSchedulerServiceClient(host)
        buffers = vx.VxBufferServiceClient(host)
        platform_info = vx.VxPlatformServiceClient(host).get_platform_info(pb.Empty())
        if backend not in platform_info.compiled_backends:
            raise ValueError(f"selected library does not compile backend {backend}")
        backend_policy = pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
            backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)

        def setup(name: str, delay: int) -> tuple[int, int, int]:
            runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=spec["cpu_threads"],
                execution_mode=pb.ExecutionMode.EXECUTION_MODE_SCHEDULED,
                budget=pb.RuntimeBudget(max_scheduled_requests=spec["max_backlog"],
                    max_scheduled_input_bytes=spec["max_scheduled_input_bytes"], max_batch_delay_ns=delay,
                    max_unconsumed_results=spec["max_backlog"],
                    max_unconsumed_result_bytes=spec["max_unconsumed_result_bytes"])))
            model = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                graph_path=str(paths[name]), weight_paths=[str(paths[name].parent / "model.safetensors")]))
            compiled = inference.compile_model(pb.CompileModelRequest(model_id=model.model_id, policy=backend_policy))
            return runtime.runtime_id, model.model_id, compiled.compiled_model_id

        def context(compiled: int) -> int:
            return inference.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled, graph=pb.GraphExecution())).context_id

        def wait_take(handle: Any) -> Any:
            ref = pb.RequestRef(request_id=handle.request_id)
            info = scheduler.wait_request(ref, timeout=10)
            if info.state != pb.RequestState.REQUEST_STATE_SUCCEEDED:
                raise AssertionError(f"replay request failed: {info}")
            packet = scheduler.take_request_result(ref)
            scheduler.release_request(ref)
            return packet

        state_runtime, state_model, state_compiled = setup("stateful", 0)
        state_context = context(state_compiled)
        initial = {**records[0].tensors, "state": np.zeros((1, 3), dtype=np.float32),
                   "position": np.array([0], dtype=np.int32)}
        # Freshness never mutates a stateful context. Refusal must leave its
        # explicit initialize entrypoint available for the subsequent ALL call.
        try:
            scheduler.submit(pb.SubmitRequest(execute=_execute(state_context, initial, "initialize"),
                options=pb.SubmitOptions(freshness=pb.RequestFreshness.REQUEST_FRESHNESS_LATEST, stream_key=71)))
        except vx.VolvoxAIError as error:
            if error.report.status != pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT:
                raise
        else:
            raise AssertionError("LATEST admitted a stateful initialization")
        state = np.zeros((1, 3), dtype=np.float32)
        retained = None
        for step, record in enumerate(records):
            frozen = Observation(record.index, record.source_ns,
                {name: value.copy() for name, value in record.tensors.items()})
            values = {name: value.copy() for name, value in record.tensors.items()}
            values["position"] = np.array([step], dtype=np.int32)
            if step == 0:
                values["state"] = state.copy()
            accepted = scheduler.submit(pb.SubmitRequest(execute=_execute(state_context, values,
                "initialize" if step == 0 else "transition", feedback=step > 0),
                options=pb.SubmitOptions(freshness=pb.RequestFreshness.REQUEST_FRESHNESS_ALL)))
            # Submit owns a snapshot now, even while the producer reuses memory.
            for value in values.values():
                value.fill(99)
            packet = wait_take(accepted)
            actual = _copy_packet(buffers, packet)
            state = reference(frozen, state)
            _assert_output(actual, state, frozen, spec, position=step)
            rows.append({"sample_index": step, "state": actual["state_next"].tolist()})
            actual["state_next"].fill(-999)
            if retained is None:
                retained = packet
            else:
                _release_packet(buffers, packet)
        inference.release_execution_context(pb.ExecutionContextRef(context_id=state_context))
        first = _copy_packet(buffers, retained)
        _assert_output(first, reference(records[0], np.zeros((1, 3), dtype=np.float32)), records[0], spec, position=0)
        _release_packet(buffers, retained)
        inference.release_compiled_model(pb.CompiledModelRef(compiled_model_id=state_compiled))
        inference.release_model(pb.ModelRef(model_id=state_model))
        inference.release_runtime(pb.RuntimeRef(runtime_id=state_runtime))

        latest_runtime, latest_model, latest_compiled = setup("stateless", spec["latest_coalescing_ns"])
        invalid_context = context(latest_compiled)
        incomplete = {name: value.copy() for name, value in records[0].tensors.items()
                      if name != "image_features"}
        try:
            scheduler.submit(pb.SubmitRequest(execute=_execute(invalid_context, incomplete, "evaluate",
                reuse=("image_features",)), options=pb.SubmitOptions(
                    freshness=pb.RequestFreshness.REQUEST_FRESHNESS_LATEST, stream_key=71)))
        except vx.VolvoxAIError as error:
            if error.report.status != pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT:
                raise
        else:
            raise AssertionError("LATEST admitted reused inputs")
        # The rejected request never reserves or poisons this stateless context.
        accepted = scheduler.submit(pb.SubmitRequest(execute=_execute(invalid_context, records[0].tensors, "evaluate"),
            options=pb.SubmitOptions(freshness=pb.RequestFreshness.REQUEST_FRESHNESS_LATEST, stream_key=71)))
        packet = wait_take(accepted)
        _assert_output(_copy_packet(buffers, packet), reference(records[0]), records[0], spec)
        _release_packet(buffers, packet)
        inference.release_execution_context(pb.ExecutionContextRef(context_id=invalid_context))
        stale_drops = 0
        for start in range(0, len(records), spec["latest_burst_size"]):
            burst = records[start:start + spec["latest_burst_size"]]
            now_ns = burst[-1].source_ns + spec["period_ns"]
            stale = Observation(-1, now_ns - 2 * spec["freshness_max_age_ns"], burst[0].tensors)
            selected = latest_fresh([stale, *burst], now_ns=now_ns, max_age_ns=spec["freshness_max_age_ns"])
            if selected is not burst[-1]:
                raise AssertionError("caller freshness selected the wrong source observation")
            stale_drops += 1
            contexts, requests = [], []
            for record in burst:
                current = context(latest_compiled)
                contexts.append(current)
                values = {name: value.copy() for name, value in record.tensors.items()}
                requests.append(scheduler.submit(pb.SubmitRequest(execute=_execute(current, values, "evaluate"),
                    options=pb.SubmitOptions(freshness=pb.RequestFreshness.REQUEST_FRESHNESS_LATEST, stream_key=71))))
                for value in values.values():
                    value.fill(99)
            packet = wait_take(requests[-1])
            actual = _copy_packet(buffers, packet)
            _assert_output(actual, reference(burst[-1]), burst[-1], spec)
            _release_packet(buffers, packet)
            superseded, completed_before_replacement = 0, 0
            for previous, previous_record in zip(requests[:-1], burst[:-1]):
                ref = pb.RequestRef(request_id=previous.request_id)
                info = scheduler.wait_request(ref, timeout=10)
                if info.state == pb.RequestState.REQUEST_STATE_SUPERSEDED:
                    superseded += 1
                    try:
                        scheduler.take_request_result(ref)
                    except vx.VolvoxAIError as error:
                        if error.report.status != pb.NativeStatus.NATIVE_STATUS_SUPERSEDED:
                            raise
                    else:
                        raise AssertionError("superseded request published an output")
                elif info.state == pb.RequestState.REQUEST_STATE_SUCCEEDED:
                    # A tiny request can finish before the next submission.
                    # LATEST does not retroactively invalidate published work.
                    previous_packet = scheduler.take_request_result(ref)
                    previous_output = _copy_packet(buffers, previous_packet)
                    _assert_output(previous_output, reference(previous_record), previous_record, spec)
                    _release_packet(buffers, previous_packet)
                    completed_before_replacement += 1
                else:
                    raise AssertionError(f"unexpected LATEST predecessor state: {info}")
                scheduler.release_request(ref)
            for current in contexts:
                inference.release_execution_context(pb.ExecutionContextRef(context_id=current))
            latest_rows.append({"sample_index": burst[-1].index, "superseded": superseded,
                                "completed_before_replacement": completed_before_replacement,
                                "features": actual["feature_output"].tolist()})
        inference.release_compiled_model(pb.CompiledModelRef(compiled_model_id=latest_compiled))
        inference.release_model(pb.ModelRef(model_id=latest_model))
        inference.release_runtime(pb.RuntimeRef(runtime_id=latest_runtime))
    library = vx.find_library().resolve()
    cpu_model = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None
    cpu_info = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    cpu_flags = next((line.partition(":")[2].strip().split() for line in cpu_info.splitlines()
                      if line.startswith("flags")), [])
    memory_info = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
    ram_bytes = next((int(line.split()[1]) * 1024 for line in memory_info.splitlines()
                      if line.startswith("MemTotal:")), None)
    report = {"format": "volvox-native-sensor-replay/v1", "status": "pass", "backend": backend,
        "build_profile": int(platform_info.build_profile), "library_version": platform_info.library_version,
        "library_path": str(library), "library_sha256": sha256_file(library),
        "harness_sha256": sha256_file(Path(__file__)), "policy_sha256": sha256_file(POLICY_PATH),
        "device": {"os": platform.platform(), "machine": platform.machine(), "cpu_model": cpu_model,
                   "pointer_bits": np.dtype(np.uintp).itemsize * 8, "byte_order": sys.byteorder,
                   "libc": list(platform.libc_ver()), "cpu_flags": cpu_flags, "ram_bytes": ram_bytes},
        "fixture_hashes": {str(path.relative_to(directory)): sha256_file(path)
            for graph in paths.values() for path in (graph, graph.parent / "model.safetensors")},
        "prepared_observations": {"path": prepared_path.name, "sha256": sha256_file(prepared_path),
            "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                "bytes": value.nbytes, "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                for name, value in prepared.items()}},
        "admission_refusals": ["stateful LATEST initialization", "stateless LATEST reused inputs"],
        "qualification": spec, "stateful_all": rows, "stateless_latest": latest_rows,
        "caller_stale_drops": stale_drops, "input_snapshot": "producer reused after acceptance; preserved",
        "output_independence": "first native snapshot retained across 12 steps and context release",
        "timing": False, "physical_robot": False}
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = replay(args.output, backend=args.backend)
    print(json.dumps({"status": result["status"], "backend": args.backend,
        "all_steps": len(result["stateful_all"]), "latest_results": len(result["stateless_latest"]),
        "superseded": sum(row["superseded"] for row in result["stateless_latest"]),
        "report": str(args.output / "result.json")}, indent=2))


if __name__ == "__main__":
    main()
