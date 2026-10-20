#!/usr/bin/env python3
"""Opt-in paced prepared-sensor CPU replay; observed workload, not a performance gate.

Preparation, warmup and NumPy assertions are separate from the paced interval.
The generated public scheduler supplies ALL state feedback or independent
stateless LATEST contexts. Source clocks and dropping remain caller policy.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import native_sensor_replay as capture

vx, pb = capture.vx, capture.pb
POLICY_PATH = ROOT / "tools/llm_baseline/native_sensor_paced.json"
TERMINAL = {pb.RequestState.REQUEST_STATE_SUCCEEDED, pb.RequestState.REQUEST_STATE_CANCELLED,
            pb.RequestState.REQUEST_STATE_FAILED, pb.RequestState.REQUEST_STATE_SUPERSEDED}


def measurement_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text())
    positive = ("period_ns", "samples", "duration_ns", "caller_queue_limit", "freshness_max_age_ns",
                "deadline_ns", "max_wait_ns", "poll_interval_ns", "rss_sample_period_ns",
                "inventory_every_completed", "cpu_threads")
    if any(type(value.get(name)) is not int or value[name] <= 0 for name in positive):
        raise ValueError("paced policy requires positive integer bounds")
    if value["duration_ns"] != value["samples"] * value["period_ns"]:
        raise ValueError("paced duration must equal samples times period")
    if value["input_rate_hz"] * value["period_ns"] != 1_000_000_000:
        raise ValueError("paced rate and period disagree")
    return value


def freshness_decision(source_ns: int, now_ns: int, maximum_age: int) -> str | None:
    if source_ns > now_ns:
        return "caller_future_drop"
    if now_ns - source_ns > maximum_age:
        return "caller_stale_drop"
    return None


def percentiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    q = np.quantile(np.asarray(values, dtype=np.float64), [.50, .95, .99], method="linear")
    return {"count": len(values), "p50": float(q[0]), "p95": float(q[1]),
            "p99": float(q[2]), "max": max(values)}


def poll_request(scheduler: Any, ref: Any) -> Any:
    """The SDK raises for BUSY and failed terminal reports; preserve typed RequestInfo."""
    try:
        return scheduler.poll_request(ref)
    except vx.VolvoxAIError as error:
        info = error.response
        if (not isinstance(info, pb.RequestInfo) or info.request_id != ref.request_id or
                info.report is None or error.report is None or info.status != error.report.status):
            raise
        if (info.state in (pb.RequestState.REQUEST_STATE_QUEUED, pb.RequestState.REQUEST_STATE_RUNNING) and
                info.status == pb.NativeStatus.NATIVE_STATUS_BUSY and
                info.report.code == pb.OperationCode.OPERATION_CODE_BUSY):
            return info
        if info.state in TERMINAL and info.state != pb.RequestState.REQUEST_STATE_SUCCEEDED:
            return info
        raise


@dataclass
class Arrival:
    record: capture.Observation
    source_ns: int
    enqueue_ns: int
    row: dict[str, Any]


class CallerQueue:
    """Bound queued + one currently consumed observation before engine admission."""
    def __init__(self, limit: int, maximum_age: int):
        self.limit, self.maximum_age = limit, maximum_age
        self.condition = threading.Condition()
        self.pending: deque[Arrival] = deque()
        self.processing = 0
        self.done = False
        self.rows: list[dict[str, Any]] = []
        self.backlog: list[dict[str, int]] = []

    def _sample(self, now_ns: int) -> None:
        self.backlog.append({"monotonic_ns": now_ns, "queued": len(self.pending),
                             "processing": self.processing, "total": len(self.pending) + self.processing})

    def emit(self, record: capture.Observation, source_ns: int, enqueue_ns: int) -> None:
        with self.condition:
            row = {"sample_index": record.index, "source_planned_ns": source_ns,
                   "enqueue_ns": enqueue_ns, "timer_overshoot_ns": enqueue_ns - source_ns,
                   "admitted": False, "decision": "pending"}
            self.rows.append(row)
            if len(self.pending) + self.processing >= self.limit:
                row["decision"] = "caller_backlog_drop"
            else:
                self.pending.append(Arrival(record, source_ns, enqueue_ns, row))
            self._sample(enqueue_ns)
            self.condition.notify_all()

    def pop(self, now_ns: int, *, latest: bool) -> Arrival | None:
        with self.condition:
            eligible = []
            while self.pending:
                item = self.pending[0]
                reason = freshness_decision(item.source_ns, now_ns, self.maximum_age)
                if reason:
                    self.pending.popleft().row.update(decision=reason, decision_ns=now_ns)
                    continue
                eligible.append(self.pending.popleft())
                if not latest:
                    break
            if not eligible:
                self._sample(now_ns)
                return None
            selected = max(eligible, key=lambda item: (item.source_ns, item.record.index))
            for previous in eligible:
                if previous is not selected:
                    previous.row.update(decision="caller_coalesced_drop", decision_ns=now_ns)
            self.processing = 1
            self._sample(now_ns)
            return selected

    def finish(self, now_ns: int) -> None:
        with self.condition:
            self.processing = 0
            self._sample(now_ns)
            self.condition.notify_all()

    def cancel_pending(self, now_ns: int) -> None:
        with self.condition:
            while self.pending:
                self.pending.popleft().row.update(decision="caller_cancel_drop", decision_ns=now_ns)
            self._sample(now_ns)
            self.condition.notify_all()


def accounting(rows: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    counts = {name: 0 for name in ("completed", "cancelled", "failed", "superseded",
        "caller_stale_drop", "caller_future_drop", "caller_backlog_drop", "caller_coalesced_drop",
        "caller_cancel_drop")}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    if counts.get("pending", 0) or counts.get("admitted", 0) or len({row["sample_index"] for row in rows}) != len(rows):
        raise AssertionError("replay observations are not uniquely settled")
    accepted = sum(bool(row["admitted"]) for row in rows)
    accepted_terminal = sum(counts.get(name, 0) for name in ("completed", "cancelled", "failed", "superseded"))
    dropped = sum(count for name, count in counts.items() if name.startswith("caller_"))
    if accepted != accepted_terminal or accepted + dropped != len(rows) or len(rows) > planned:
        raise AssertionError("accepted/terminal/caller-drop accounting differs")
    return {"planned": planned, "released_by_producer": len(rows),
            "not_released_after_cancel": planned - len(rows), "accepted": accepted,
            "terminal": accepted_terminal, "caller_dropped": dropped, "decisions": counts,
            "balanced": True}


def process_memory() -> dict[str, Any]:
    now_ns = time.monotonic_ns()
    path = Path("/proc/self/status")
    if not path.exists():
        return {"monotonic_ns": now_ns, "status": "unsupported", "source": str(path)}
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        return {"monotonic_ns": now_ns, "status": "failed", "source": str(path), "error": str(error)}
    fields = {line.partition(":")[0]: line.partition(":")[2].strip().split() for line in lines}
    values = {key: int(fields[key][0]) * 1024 if key in fields else None for key in ("VmRSS", "VmHWM")}
    return {"monotonic_ns": now_ns, "status": "available", "source": "/proc/self/status",
            "resident_bytes": values["VmRSS"], "process_lifetime_high_water_bytes": values["VmHWM"]}


def inventory(profiling: Any, *, runtime_id: int | None, label: str) -> dict[str, Any]:
    request = (pb.GetResourceSnapshotRequest(runtime_id=runtime_id) if runtime_id is not None else
               pb.GetResourceSnapshotRequest(module=pb.Empty()))
    snapshot = profiling.get_resource_snapshot(request).snapshot
    memory = snapshot.memory
    return {"label": label, "observation_start_ns": snapshot.observation_start_ns,
            "observation_end_ns": snapshot.observation_end_ns, "inventory": int(memory.inventory),
            "truncated": memory.truncated, "known_allocation_count": len(memory.allocations),
            "known_capacity_bytes": sum(allocation.capacity_bytes for allocation in memory.allocations),
            "unconsumed_result_bytes": memory.unconsumed_result_bytes,
            "retained_result_capacity_bytes": memory.retained_result_capacity_bytes,
            "idle_result_capacity_bytes": memory.idle_result_capacity_bytes,
            "coverage": "known scoped engine backing capacities; not process RSS or exact whole-engine peak"}


def run_measurement(directory: Path, *, library: Path, artifacts: Path,
                    policy_path: Path = POLICY_PATH, mode: str | None = None,
                    stop: threading.Event | None = None) -> dict[str, Any]:
    spec = measurement_policy(policy_path)
    mode = mode or spec["default_mode"]
    if mode not in ("stateful-all", "stateless-latest"):
        raise ValueError("unknown paced sensor mode")
    stateful = mode == "stateful-all"
    fixture_spec = capture.policy()
    if capture.sha256_file(capture.POLICY_PATH) != spec["base_policy_sha256"]:
        raise ValueError("prepared sensor policy differs from the prospective policy")
    library = library.resolve()
    library_hash = capture.sha256_file(library)
    pins = json.loads(artifacts.read_text())
    selected = [name for name in ("native_full", "native_inference")
                if pins.get(name, {}).get("sha256") == library_hash and
                Path(pins[name]["path"]).resolve() == library]
    if len(selected) != 1:
        raise ValueError("library does not match one accepted CPU artifact pin")
    # Exclusive path and prospective policy are committed before loading or observing results.
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {"format": "volvox-native-sensor-paced-prospective/v1", "mode": mode,
                  "policy": spec, "policy_sha256": capture.sha256_file(policy_path),
                  "harness_sha256": capture.sha256_file(Path(__file__)),
                  "capture_harness_sha256": capture.sha256_file(Path(capture.__file__)),
                  "base_policy_sha256": capture.sha256_file(capture.POLICY_PATH),
                  "artifact": {"profile": selected[0], "path": str(library), "sha256": library_hash},
                  "accepted_artifacts_path": str(artifacts.resolve()),
                  "accepted_artifacts_sha256": capture.sha256_file(artifacts), "results_observed": False}
    (directory / "prospective.json").write_text(json.dumps(provenance, indent=2) + "\n")
    records = capture.observations({**fixture_spec, "samples": spec["samples"], "period_ns": spec["period_ns"]})
    prepared = {name: np.concatenate([record.tensors[name] for record in records], axis=0)
                for name in records[0].tensors}
    prepared["source_relative_ns"] = np.asarray([record.source_ns for record in records], dtype=np.int64)
    from safetensors.numpy import save_file
    prepared_path = directory / "prepared_observations.safetensors"
    save_file(prepared, prepared_path)
    graph = capture.write_fixture(directory / "fixture", stateful=stateful)
    queue = CallerQueue(spec["caller_queue_limit"], spec["freshness_max_age_ns"])
    stop = stop if stop is not None else threading.Event()
    rss_samples = [{**process_memory(), "phase": "before-loading"}]
    inventory_samples, copied_outputs, requests, contexts = [], [], [], []
    counts = {name: 0 for name in ("requests_created", "requests_released", "contexts_created",
               "contexts_released", "owned_buffers_created", "owned_buffers_released",
               "peak_live_owned_buffers", "warmup_executions")}
    producer = None
    retained = None
    failure = None
    profile = None
    start_ns = end_ns = 0
    cpu_start_ns = cpu_end_ns = 0
    runtime_id = model_id = compiled_id = None
    producer_errors: list[str] = []
    route_samples = []
    numerical_max_absolute = 0.0
    numerical_comparisons = 0

    with vx.open_library(library) as host:
        inference = vx.VxInferenceServiceClient(host)
        scheduler = vx.VxSchedulerServiceClient(host)
        buffers = vx.VxBufferServiceClient(host)
        profiling = vx.VxProfilingServiceClient(host)
        native_platform = vx.VxPlatformServiceClient(host)
        platform_info = native_platform.get_platform_info(pb.Empty())
        profile = int(platform_info.build_profile)
        if "cpu" not in platform_info.compiled_backends:
            raise ValueError("accepted library does not compile CPU")
        clock_before = time.monotonic_ns()
        native_ns = native_platform.get_monotonic_time(pb.Empty()).nanoseconds
        clock_after = time.monotonic_ns()
        native_offset = native_ns - (clock_before + clock_after) // 2
        clock_evidence = {"python": time.get_clock_info("monotonic").implementation,
                          "resolution_seconds": time.get_clock_info("monotonic").resolution,
                          "native_minus_python_midpoint_ns": native_offset,
                          "calibration_bracket_ns": clock_after - clock_before,
                          "drift": "unmeasured; local CLOCK_MONOTONIC mapping only"}

        def create_context() -> int:
            context_id = inference.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled_id, graph=pb.GraphExecution())).context_id
            contexts.append(context_id)
            counts["contexts_created"] += 1
            return context_id

        def release_context(context_id: int) -> None:
            inference.release_execution_context(pb.ExecutionContextRef(context_id=context_id))
            contexts.remove(context_id)
            counts["contexts_released"] += 1

        def count_packet(packet: Any) -> None:
            identifiers = [tensor.buffer.buffer_id for tensor in packet.outputs if tensor.HasField("buffer")]
            if len(identifiers) != len(set(identifiers)):
                raise AssertionError("fixture outputs unexpectedly share public buffer handles")
            owned = len(identifiers)
            counts["owned_buffers_created"] += owned
            counts["peak_live_owned_buffers"] = max(counts["peak_live_owned_buffers"],
                counts["owned_buffers_created"] - counts["owned_buffers_released"])

        def release_packet(packet: Any) -> None:
            capture._release_packet(buffers, packet)
            counts["owned_buffers_released"] += sum(tensor.HasField("buffer") for tensor in packet.outputs)

        def take_request(ref: Any) -> Any:
            packet = scheduler.take_request_result(ref)
            count_packet(packet)
            return packet

        def release_request(ref: Any) -> None:
            scheduler.release_request(ref)
            requests.remove(ref)
            counts["requests_released"] += 1

        def check_route(packet: Any) -> None:
            route = packet.report.route
            if (route is None or route.provider != "cpu" or not route.builtin or not route.attested or
                    route.missing_nodes or route.selected_nodes != route.active_nodes):
                raise AssertionError("paced result lacks an attested required CPU route")
            route_samples.append({"provider": route.provider, "active_nodes": route.active_nodes,
                                  "selected_nodes": route.selected_nodes, "missing_nodes": route.missing_nodes})

        def submit(context_id: int, record: capture.Observation, ordinal: int, deadline: int = 0) -> Any:
            values = {name: value.copy() for name, value in record.tensors.items()}
            if stateful:
                values["position"] = np.array([ordinal], dtype=np.int32)
                if ordinal == 0:
                    values["state"] = np.zeros((1, 3), dtype=np.float32)
            options = pb.SubmitOptions(freshness=(pb.RequestFreshness.REQUEST_FRESHNESS_ALL if stateful else
                pb.RequestFreshness.REQUEST_FRESHNESS_LATEST), stream_key=0 if stateful else 71,
                deadline_monotonic_ns=deadline)
            handle = scheduler.submit(pb.SubmitRequest(execute=capture._execute(context_id, values,
                "initialize" if stateful and ordinal == 0 else "transition" if stateful else "evaluate",
                feedback=stateful and ordinal > 0), options=options))
            ref = pb.RequestRef(request_id=handle.request_id)
            requests.append(ref)
            counts["requests_created"] += 1
            # Accepted inputs are independently owned even if the producer reuses this memory.
            for value in values.values():
                value.fill(99)
            return ref

        def await_terminal(ref: Any, *, cancellable: bool) -> tuple[Any, int, int | None, int]:
            began = time.monotonic_ns()
            last_nonterminal = None
            polls = 0
            cancelled_at = None
            while True:
                info = poll_request(scheduler, ref)
                observed = time.monotonic_ns()
                polls += 1
                if info.state in TERMINAL:
                    return info, observed, last_nonterminal, polls
                last_nonterminal = observed
                if cancelled_at is None and ((cancellable and stop.is_set()) or observed - began > spec["max_wait_ns"]):
                    scheduler.cancel_request(ref)
                    cancelled_at = time.monotonic_ns()
                elif cancelled_at is not None and observed - cancelled_at > spec["max_wait_ns"]:
                    raise TimeoutError("native request did not settle within the declared cancellation drain")
                # Native poll observation adds this interval to caller completion precision.
                time.sleep(spec["poll_interval_ns"] / 1e9)

        try:
            runtime_id = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=spec["cpu_threads"],
                execution_mode=pb.ExecutionMode.EXECUTION_MODE_SCHEDULED,
                budget=pb.RuntimeBudget(max_scheduled_requests=spec["runtime_max_scheduled_requests"],
                    max_scheduled_input_bytes=spec["runtime_max_scheduled_input_bytes"], max_batch_delay_ns=0,
                    max_unconsumed_results=spec["runtime_max_unconsumed_results"],
                    max_unconsumed_result_bytes=spec["runtime_max_unconsumed_result_bytes"]))).runtime_id
            model_id = inference.load_model(pb.LoadModelRequest(runtime_id=runtime_id, graph_path=str(graph),
                weight_paths=[str(graph.parent / "model.safetensors")])).model_id
            compiled = inference.compile_model(pb.CompileModelRequest(model_id=model_id,
                policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=["cpu"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            compiled_id = compiled.compiled_model_id
            warm_context = create_context()
            warm_state = np.zeros((1, 3), dtype=np.float32)
            for ordinal in range(spec["warmup_samples"]):
                record = records[ordinal % len(records)]
                ref = submit(warm_context, record, ordinal)
                info, _, _, _ = await_terminal(ref, cancellable=False)
                if info.state != pb.RequestState.REQUEST_STATE_SUCCEEDED:
                    raise AssertionError("separate warmup did not succeed")
                packet = take_request(ref)
                try:
                    check_route(packet)
                    actual = capture._copy_packet(buffers, packet)
                    expected = capture.reference(record, warm_state if stateful else None)
                    capture._assert_output(actual, expected, record, fixture_spec,
                                           position=ordinal if stateful else None)
                    warm_state = expected
                finally:
                    release_packet(packet)
                    release_request(ref)
                counts["warmup_executions"] += 1
            release_context(warm_context)
            current_context = create_context() if stateful else None
            inventory_samples.append(inventory(profiling, runtime_id=runtime_id, label="after-warmup-before-pacing"))
            rss_samples.append({**process_memory(), "phase": "after-warmup-before-pacing"})
            start_ns = time.monotonic_ns() + 10_000_000
            planned_end_ns = start_ns + spec["duration_ns"]
            cpu_start_ns = time.process_time_ns()

            def produce() -> None:
                next_rss = start_ns
                try:
                    for record in records:
                        source = start_ns + record.source_ns
                        while source > time.monotonic_ns() and not stop.is_set():
                            stop.wait((source - time.monotonic_ns()) / 1e9)
                        if stop.is_set():
                            break
                        enqueued = time.monotonic_ns()
                        queue.emit(record, source, enqueued)
                        if enqueued >= next_rss:
                            rss_samples.append({**process_memory(), "phase": "paced"})
                            next_rss = enqueued + spec["rss_sample_period_ns"]
                    # Last release is at duration-period; include the final period in the paced window.
                    if not stop.is_set():
                        stop.wait(max(0, planned_end_ns - time.monotonic_ns()) / 1e9)
                except BaseException as error:
                    producer_errors.append(f"{type(error).__name__}: {error}")
                    stop.set()
                finally:
                    with queue.condition:
                        queue.done = True
                        queue.condition.notify_all()

            producer = threading.Thread(target=produce, name="prepared-sensor-producer", daemon=False)
            producer.start()
            ordinal = 0
            while True:
                if stop.is_set():
                    queue.cancel_pending(time.monotonic_ns())
                item = queue.pop(time.monotonic_ns(), latest=not stateful) if not stop.is_set() else None
                if item is None:
                    with queue.condition:
                        if queue.done and not queue.pending:
                            break
                        queue.condition.wait(spec["poll_interval_ns"] / 1e9)
                    continue
                packet = None
                ref = None
                context_id = current_context if stateful else create_context()
                try:
                    submit_start = time.monotonic_ns()
                    reason = freshness_decision(item.source_ns, submit_start, spec["freshness_max_age_ns"])
                    if reason:
                        item.row.update(decision=reason, decision_ns=submit_start)
                        continue
                    ref = submit(context_id, item.record, ordinal, item.source_ns + spec["deadline_ns"] + native_offset)
                    item.row.update(admitted=True, decision="admitted", ordinal=ordinal,
                                    submit_start_ns=submit_start, accepted_return_ns=time.monotonic_ns())
                    info, completed_ns, last_nonterminal_ns, polls = await_terminal(ref, cancellable=True)
                    item.row.update(completion_observed_ns=completed_ns, last_nonterminal_observed_ns=last_nonterminal_ns,
                        native_completion_observation_bracket_ns=[last_nonterminal_ns or submit_start, completed_ns],
                        poll_count=polls, engine_state=int(info.state), engine_status=int(info.status),
                        engine_deadline_missed=info.deadline_missed,
                        caller_deadline_missed=completed_ns > item.source_ns + spec["deadline_ns"],
                        enqueue_to_completion_ns=completed_ns - item.enqueue_ns,
                        source_to_completion_ns=completed_ns - item.source_ns,
                        submit_to_completion_ns=completed_ns - submit_start)
                    if info.state == pb.RequestState.REQUEST_STATE_SUCCEEDED:
                        item.row["decision"] = "completed"
                        packet = take_request(ref)
                        check_route(packet)
                        release_request(ref)
                        ref = None
                        # Output copy/receipt validation is deliberately after completion measurement.
                        actual = capture._copy_packet(buffers, packet)
                        copied_outputs.append((item.record, ordinal, actual))
                        if retained is None:
                            retained, packet = packet, None
                        ordinal += 1
                    else:
                        item.row["decision"] = {pb.RequestState.REQUEST_STATE_CANCELLED: "cancelled",
                            pb.RequestState.REQUEST_STATE_FAILED: "failed", pb.RequestState.REQUEST_STATE_SUPERSEDED: "superseded"}[info.state]
                        stop.set()
                        if info.state != pb.RequestState.REQUEST_STATE_CANCELLED:
                            raise AssertionError(f"unexpected paced terminal state: {info}")
                    if ordinal and ordinal % spec["inventory_every_completed"] == 0:
                        inventory_samples.append(inventory(profiling, runtime_id=runtime_id, label=f"after-{ordinal}-completed"))
                finally:
                    if packet is not None:
                        release_packet(packet)
                    if ref is not None:
                        release_request(ref)
                    if not stateful:
                        release_context(context_id)
                    item.row["consumed_release_ns"] = time.monotonic_ns()
                    queue.finish(item.row["consumed_release_ns"])
            producer.join(spec["max_wait_ns"] / 1e9)
            if producer.is_alive():
                raise TimeoutError("sensor producer did not stop")
            if producer_errors:
                raise RuntimeError("producer failed: " + "; ".join(producer_errors))
            end_ns, cpu_end_ns = time.monotonic_ns(), time.process_time_ns()
            rss_samples.append({**process_memory(), "phase": "after-paced"})
            # No NumPy correctness assertion runs inside the paced source interval.
            state = np.zeros((1, 3), dtype=np.float32)
            for record, position, actual in copied_outputs:
                expected = capture.reference(record, state if stateful else None)
                capture._assert_output(actual, expected, record, fixture_spec, position=position if stateful else None)
                numerical_comparisons += 1
                numerical_max_absolute = max(numerical_max_absolute, float(np.max(np.abs(
                    actual["state_next" if stateful else "feature_output"].astype(np.float64) -
                    expected.astype(np.float64)))))
                state = expected
            for context_id in list(contexts):
                release_context(context_id)
            if retained is not None:
                first_record, first_position, _ = copied_outputs[0]
                first = capture._copy_packet(buffers, retained)
                capture._assert_output(first, capture.reference(first_record,
                    np.zeros((1, 3), dtype=np.float32) if stateful else None), first_record,
                    fixture_spec, position=first_position if stateful else None)
                release_packet(retained)
                retained = None
            final_inventory = inventory(profiling, runtime_id=runtime_id, label="after-all-outputs-and-contexts-released")
            inventory_samples.append(final_inventory)
            if final_inventory["unconsumed_result_bytes"] != 0:
                raise AssertionError("runtime retains unconsumed output bytes after cleanup")
        except BaseException as error:
            stop.set()
            failure = {"type": type(error).__name__, "message": str(error)}
            if start_ns and not end_ns:
                end_ns, cpu_end_ns = time.monotonic_ns(), time.process_time_ns()
        finally:
            if producer is not None:
                stop.set()
                producer.join(spec["max_wait_ns"] / 1e9)
                queue.cancel_pending(time.monotonic_ns())
            for ref in list(requests):
                try:
                    scheduler.cancel_request(ref)
                    await_terminal(ref, cancellable=False)
                finally:
                    release_request(ref)
            for context_id in list(contexts):
                release_context(context_id)
            if retained is not None:
                release_packet(retained)
            if compiled_id is not None:
                inference.release_compiled_model(pb.CompiledModelRef(compiled_model_id=compiled_id))
            if model_id is not None:
                inference.release_model(pb.ModelRef(model_id=model_id))
            if runtime_id is not None:
                inference.release_runtime(pb.RuntimeRef(runtime_id=runtime_id))
            inventory_samples.append(inventory(profiling, runtime_id=None, label="module-after-owner-release"))
    rss_samples.append({**process_memory(), "phase": "after-host-close"})
    settled = accounting(queue.rows, spec["samples"]) if failure is None else None
    completed = [row for row in queue.rows if row["decision"] == "completed"]
    cleanup_balanced = (counts["requests_created"] == counts["requests_released"] and
        counts["contexts_created"] == counts["contexts_released"] and
        counts["owned_buffers_created"] == counts["owned_buffers_released"])
    if not cleanup_balanced and failure is None:
        failure = {"type": "AssertionError", "message": "created/released native handle accounting differs"}
    library_hash_after = capture.sha256_file(library)
    if library_hash_after != library_hash and failure is None:
        failure = {"type": "AssertionError", "message": "accepted CPU artifact changed during replay"}
    captured_path = None
    if copied_outputs:
        captured_path = directory / "completed_outputs.safetensors"
        save_file({name: np.concatenate([actual[name] for _, _, actual in copied_outputs], axis=0)
                   for name in copied_outputs[0][2]}, captured_path)
    paced_rss = [row for row in rss_samples if row.get("phase") == "paced"]
    cpu_text = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    mem_text = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
    report = {"format": "volvox-native-sensor-paced/v1", "status": "failed" if failure else "cancelled" if
              settled["not_released_after_cancel"] or settled["decisions"].get("cancelled", 0) else "observed",
        "failure": failure, "prospective": provenance, "build_profile": profile,
        "library_sha256_after": library_hash_after,
        "device": {"os": platform.platform(), "machine": platform.machine(), "libc": list(platform.libc_ver()),
                   "pointer_bits": np.dtype(np.uintp).itemsize * 8, "byte_order": sys.byteorder,
                   "cpu_model": next((line.partition(":")[2].strip() for line in cpu_text.splitlines()
                                      if line.startswith("model name")), None),
                   "cpu_flags": next((line.partition(":")[2].strip().split() for line in cpu_text.splitlines()
                                      if line.startswith("flags")), []),
                   "ram_bytes": next((int(line.split()[1]) * 1024 for line in mem_text.splitlines()
                                      if line.startswith("MemTotal:")), None)},
        "clock": clock_evidence, "prepared_sha256": capture.sha256_file(prepared_path),
        "fixture_hashes": {str(path.relative_to(directory)): capture.sha256_file(path)
                           for path in (graph, graph.parent / "model.safetensors")},
        "window": {"origin_monotonic_ns": start_ns, "end_monotonic_ns": end_ns,
                   "actual_elapsed_ns": max(0, end_ns - start_ns), "planned_duration_ns": spec["duration_ns"],
                   "planned_end_monotonic_ns": start_ns + spec["duration_ns"] if start_ns else None,
                   "last_completion_observed_ns": max((row["completion_observed_ns"] for row in completed), default=None),
                   "drain_after_planned_end_ns": max(0, end_ns - start_ns - spec["duration_ns"]) if end_ns else None,
                   "process_cpu_time_ns": max(0, cpu_end_ns - cpu_start_ns),
                   "observed_successful_requests_per_second": len(completed) * 1e9 / (end_ns - start_ns)
                       if end_ns > start_ns else None},
        "accounting": settled, "samples": queue.rows,
        "latency_ns": {name: percentiles([row[name] for row in completed]) for name in
            ("enqueue_to_completion_ns", "submit_to_completion_ns", "source_to_completion_ns")},
        "producer_timer_overshoot_ns": percentiles([row["timer_overshoot_ns"] for row in queue.rows]),
        "deadline_misses": {"caller_source_target": sum(row["caller_deadline_missed"] for row in completed),
                            "native_scheduler_target": sum(row["engine_deadline_missed"] for row in completed)},
        "backlog": {"samples": queue.backlog, "maximum_queued": max((row["queued"] for row in queue.backlog), default=0),
                    "maximum_queued_plus_processing": max((row["total"] for row in queue.backlog), default=0),
                    "maximum_native_inflight": 1 if any(row["admitted"] for row in queue.rows) else 0},
        "memory": {"coverage": spec["memory_scope"], "process_samples": rss_samples,
                   "maximum_sampled_resident_bytes": max((row["resident_bytes"] for row in rss_samples if row.get("resident_bytes") is not None), default=None),
                   "maximum_paced_sampled_resident_bytes": max((row["resident_bytes"] for row in rss_samples if row.get("phase") == "paced" and row.get("resident_bytes") is not None), default=None),
                   "maximum_os_process_lifetime_high_water_bytes": max((row["process_lifetime_high_water_bytes"] for row in rss_samples if row.get("process_lifetime_high_water_bytes") is not None), default=None),
                   "paced_rss_sampling_interval_ns": percentiles([b["monotonic_ns"] - a["monotonic_ns"]
                                                                 for a, b in zip(paced_rss, paced_rss[1:])]),
                   "engine_inventory_samples": inventory_samples, "exact_interval_peak": "unknown"},
        "cleanup": {**counts, "balanced": cleanup_balanced, "module_host_closed": True,
                    "producer_joined": producer is None or not producer.is_alive()},
        "correctness": {"deferred_numpy_comparisons": numerical_comparisons,
                        "captured_completed_outputs": len(copied_outputs), "tolerance": fixture_spec["tolerance"],
                        "maximum_absolute_error": numerical_max_absolute if numerical_comparisons else None,
                        "captured_outputs": {"path": captured_path.name, "sha256": capture.sha256_file(captured_path)}
                            if captured_path is not None else None,
                        "owned_snapshot_checked_after_context_release": bool(copied_outputs) and failure is None,
                        "fast_capture_is_separate": True},
        "cpu_route_observations_including_separate_warmup": route_samples,
        "timing_scope": "integrated caller workflow: output copy/owned receipt and inventory occur between paced requests; enqueue-to-terminal latency excludes those operations, later queuing/backlog and process CPU/RSS include them",
        "comparison_performance_pass": False, "peak_memory_nonregression": "unknown",
        "hard_realtime": False, "physical_robot": False, "CUDA": False, "ARM64": False}
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if failure is not None:
        raise RuntimeError(f"paced replay failed; see {directory / 'result.json'}: {failure}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--mode", choices=("stateful-all", "stateless-latest"))
    args = parser.parse_args()
    report = run_measurement(args.output, library=args.library, artifacts=args.artifacts,
                             policy_path=args.policy, mode=args.mode)
    print(json.dumps({"status": report["status"], "accounting": report["accounting"],
                      "latency_ns": report["latency_ns"], "cleanup": report["cleanup"],
                      "report": str(args.output / "result.json")}, indent=2))


if __name__ == "__main__":
    main()
