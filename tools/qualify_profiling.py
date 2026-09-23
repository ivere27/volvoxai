#!/usr/bin/env python3
"""Qualify native profiling on one required backend, with no operator fallback.

Run separately for cpu, cuda, opengl and vulkan on an idle physical GPU. CPU
uses both profiles; native GPUs require full. Evidence uses relative names.
"""
import argparse
import array
import hashlib
import json
from pathlib import Path
import sys
import time
import struct
import ctypes
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx  # noqa: E402

p = vx.pb
WIDTH = 256
OUTPUT = 'out"한글'
GRAPH = {
    "format": "volvox-graph/v1", "dimensions": {},
    "inputs": {"x": {"dtype": "float32", "shape": [WIDTH // 2, 2]}},
    "nodes": [
        {"id": "first", "opType": "Linear", "inputs": {"input": "x", "weight": "weight", "bias": "bias"},
         "outputs": {"out": {"tensor": "middle", "dtype": "float32", "shape": [WIDTH // 2, 2]}},
         "params": {"weight_layout": "dout_din"}},
        {"id": "second", "opType": "Linear", "inputs": {"input": "middle", "weight": "weight", "bias": "bias"},
         "outputs": {"out": {"tensor": OUTPUT, "dtype": "float32", "shape": [WIDTH // 2, 2]}},
         "params": {"weight_layout": "dout_din"}},
    ], "outputs": [OUTPUT],
}
HEADER = json.dumps({"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
                     "bias": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]}}).encode()
HEADER += b" " * (-len(HEADER) % 8)
WEIGHTS = struct.pack("<Q", len(HEADER)) + HEADER + struct.pack("<6f", 1, 2, -3, 4, .25, -.5)


def export_trace(profiling, trace_id):
    chunks, offset = [], 0
    while True:
        chunk = profiling.export_chrome_trace(p.ExportChromeTraceRequest(
            trace_id=trace_id, offset=offset, limit=128))
        chunks.append(chunk.data)
        if chunk.eof:
            return b"".join(chunks)
        offset = chunk.next_offset


def check_trace_memory(events, info):
    live, seen, totals = {}, set(), {}
    memory = [e.memory for e in events if e.HasField("memory")]
    for event in memory:
        row = totals.setdefault(event.allocator, dict(existing_bytes=0, allocated_bytes=0,
            freed_bytes=0, live_bytes=0, peak_bytes=0))
        assert event.allocation_id > 0
        if event.action == p.TraceMemoryAction.TRACE_MEMORY_ACTION_FREE:
            assert live.pop(event.allocation_id) == (event.allocator, event.bytes)
            row["freed_bytes"] += event.bytes
            row["live_bytes"] -= event.bytes
        else:
            assert event.allocation_id not in seen
            seen.add(event.allocation_id)
            live[event.allocation_id] = (event.allocator, event.bytes)
            row["existing_bytes" if event.action == p.TraceMemoryAction.TRACE_MEMORY_ACTION_EXISTING else "allocated_bytes"] += event.bytes
            row["live_bytes"] += event.bytes
        row["peak_bytes"] = max(row["peak_bytes"], row["live_bytes"])
        assert event.live_bytes == row["live_bytes"]
    assert len(totals) == len(info.allocators)
    for allocator in info.allocators:
        assert allocator.accounting_complete and allocator.dropped_events == 0
        assert allocator.inventory == p.MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL
        assert allocator.scope == p.MemoryOwnerKind.MEMORY_OWNER_KIND_RUNTIME
        for key, value in totals[allocator.allocator].items():
            assert getattr(allocator, key) == value, (allocator.allocator, key)
    assert memory
    return [{"allocator": a.allocator, "scope": int(a.scope), "accountingComplete": a.accounting_complete,
             **{key: getattr(a, key) for key in totals[a.allocator]}} for a in info.allocators]


def qualify(backend, library, output, probe=False):
    counter_names = ["graphLaunch", "graphCapture", "timingEventCreate", "elapsedTime",
                     "streamSynchronize", "eventSynchronize", "contextSynchronize", "externalRecord", "glCreateQuery", "glTimestamp",
                     "glReadQuery", "glFinish", "vkCreateQuery", "vkTimestamp", "vkReadQuery", "vkWaitFence", "eglProc", "vkProc", "glCalibrate", "vkCalibrate"]
    counter = ctypes.CDLL(None).vx_test_gpu_counter if probe else None
    if counter:
        counter.argtypes, counter.restype = [ctypes.c_uint], ctypes.c_uint64
    driver = {"on": [], "off": []}
    with vx.open_library(ROOT / library) as host:
        inference = vx.VxInferenceServiceClient(host)
        profiling = vx.VxProfilingServiceClient(host)
        runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
        model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=p.ModelPackage(graph_document=json.dumps(GRAPH).encode(), weight_shards=[WEIGHTS])))
        compiled = inference.compile_model(p.CompileModelRequest(model_id=model.model_id,
            policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=[backend], operator_fallback=p.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        assert compiled.report.backend == backend and compiled.report.route.attested
        context = inference.create_execution_context(p.CreateExecutionContextRequest(
            compiled_model_id=compiled.compiled_model_id))
        execution_count = 0
        collecting = False
        timing_enabled = False

        def run():
            nonlocal execution_count
            # Change every input so replay/stale-output bugs cannot pass.
            values = [(index % 17 - 8 + execution_count % 5) / 4 for index in range(WIDTH)]
            execution_count += 1
            before = [counter(i) for i in range(len(counter_names))] if counter else []
            result = inference.execute(p.ExecuteRequest(context_id=context.context_id, inputs=[p.Tensor(
                name="x", dtype=p.DataType.DATA_TYPE_F32, shape=[WIDTH // 2, 2],
                inline=array.array("f", values).tobytes())]))
            assert result.report.backend == backend and result.report.route.attested
            deadline = time.monotonic() + 30
            while True:
                info = inference.get_result(p.ResultRef(result_id=result.result_id))
                if info.state != p.ResultState.RESULT_STATE_PENDING:
                    assert info.state == p.ResultState.RESULT_STATE_READY
                    break
                assert time.monotonic() < deadline, "GPU completion timed out"
                time.sleep(.001)
            tensor = inference.read_output(p.ReadOutputRequest(result_id=result.result_id, name=OUTPUT)).tensor
            actual = array.array("f")
            actual.frombytes(tensor.inline)
            expected = []
            for a, b in zip(values[::2], values[1::2]):
                first, second = a + 2 * b + .25, -3 * a + 4 * b - .5
                expected.extend([first + 2 * second + .25, -3 * first + 4 * second - .5])
            assert list(actual) == expected
            assert list(tensor.shape) == [WIDTH // 2, 2]
            inference.release_result(p.ResultRef(result_id=result.result_id))
            if counter:
                calls = dict(zip(counter_names, [counter(i) - n for i, n in enumerate(before)]))
                if not collecting or not timing_enabled:
                    assert calls["timingEventCreate"] == calls["elapsedTime"] == 0, calls
                assert calls["eventSynchronize"] == calls["contextSynchronize"] == 0, calls
                if not collecting or not timing_enabled:
                    assert all(calls[key] == 0 for key in ["externalRecord", "glCreateQuery", "glTimestamp", "glReadQuery",
                        "vkCreateQuery", "vkTimestamp", "vkReadQuery", "glCalibrate", "vkCalibrate"]), calls
                driver["on" if collecting else "off"].append(calls)
            return result.execution_id

        def start(capacity=4 * 1024 * 1024, device_timing=True, detail=p.TraceDetail.TRACE_DETAIL_NODES):
            return profiling.start_trace(p.StartTraceRequest(runtime_id=runtime.runtime_id,
                device_timing=device_timing, detail=detail, capacity_bytes=capacity, memory=True))

        def capture(count, capacity=4 * 1024 * 1024, device_timing=True, detail=p.TraceDetail.TRACE_DETAIL_NODES):
            nonlocal collecting, timing_enabled
            trace = start(capacity, device_timing, detail)
            timing_enabled = device_timing
            collecting = True
            identities = [run() for _ in range(count)]
            ref = p.TraceRef(trace_id=trace.trace_id)
            stopped = profiling.stop_trace(ref)
            collecting = False
            assert stopped.state == p.TraceState.TRACE_STATE_READY
            # Inactive execution must not append, and stopping is repeatable.
            run()
            assert profiling.get_trace(ref).event_count == stopped.event_count
            return trace, stopped, identities

        def inspect(capture, name, overflow=False):
            trace, stopped, identities = capture
            events, memory, offset = [], [], 0
            while True:
                request = p.ReadTraceRequest(trace_id=trace.trace_id, offset=offset, limit=3)
                page = profiling.read_trace(request)
                repeat = profiling.read_trace(request)
                assert [e.sequence for e in page.events] == [e.sequence for e in repeat.events]
                if offset:
                    assert not page.process_memory
                events.extend(page.events)
                memory.extend(page.process_memory)
                if page.eof:
                    break
                assert page.next_offset == offset + len(page.events)
                offset = page.next_offset
            assert len(events) == stopped.event_count
            assert len(memory) == 2
            assert all(event.lineage.execution_id in identities for event in events if not event.HasField("memory"))
            operations = [e for e in events if e.HasField("host") and not e.HasField("node") and e.activity == p.TraceActivity.TRACE_ACTIVITY_WORK]
            nodes = [e for e in events if e.HasField("host") and e.HasField("node") and e.activity == p.TraceActivity.TRACE_ACTIVITY_WORK]
            device = [e for e in events if e.HasField("device")]
            programs = [e for e in device if e.HasField("program")]
            device_scopes = [e for e in device if not e.HasField("program") and e.activity == p.TraceActivity.TRACE_ACTIVITY_WORK]
            activities = [e for e in events if e.HasField("host") and e.activity != p.TraceActivity.TRACE_ACTIVITY_WORK]
            copies = [e for e in device if e.activity == p.TraceActivity.TRACE_ACTIVITY_COPY]
            if overflow:
                assert stopped.dropped_events > 0
            else:
                assert stopped.dropped_events == 0
                check_trace_memory(events, stopped)
                if backend != "cpu":
                    assert any(a.allocator == backend + ".graph" for a in stopped.allocators)
                assert len(operations) == len(identities)
                assert len(nodes) == (2 * len(identities) if stopped.detail == p.TraceDetail.TRACE_DETAIL_NODES else 0)
                assert all(e.backend == backend for e in events if not e.HasField("memory"))
                enabled = backend != "cpu" and stopped.device_timing
                node_timing = stopped.detail == p.TraceDetail.TRACE_DETAIL_NODES
                assert bool(device) == enabled
                if device:
                    assert len(device_scopes) == len(identities) * (3 if node_timing else 1)
                    assert bool(programs) == node_timing
                    assert all(e.phase == p.TracePhase.TRACE_PHASE_FORWARD and e.HasField("node") and e.program.entry_point for e in programs)
                    assert len([e for e in device_scopes if e.HasField("node")]) == (len(identities) * 2 if node_timing else 0)
                    assert all(e.device.elapsed_ns > 0 for e in device)
                assert any(d.support == p.TraceSupport.TRACE_SUPPORT_AVAILABLE for d in stopped.devices) == bool(device)
                assert len(stopped.devices) == int(enabled or (backend != "cpu" and node_timing))
                if enabled:
                    coverage = stopped.devices[0]
                    assert coverage.program_timing_available
                    assert coverage.program_intervals == len(programs)
                    assert coverage.node_timing_available
                    assert coverage.pass_intervals == len([e for e in device_scopes if not e.HasField("node")])
                    assert coverage.node_intervals == len([e for e in device_scopes if e.HasField("node")])
                    assert coverage.failed_intervals == coverage.unavailable_passes == 0
                    assert not coverage.splits_passes
                    assert coverage.adds_barriers == (backend == "vulkan" and node_timing)
                if backend != "cpu" and node_timing:
                    assert any(e.activity == p.TraceActivity.TRACE_ACTIVITY_COPY for e in activities)
                    assert any(e.activity == p.TraceActivity.TRACE_ACTIVITY_WAIT for e in activities)
                    assert any(e.activity == p.TraceActivity.TRACE_ACTIVITY_SUBMIT for e in activities)
                    assert all(e.HasField("queue") and e.queue.device_id and e.queue.queue_id for e in activities)
                    assert all(e.HasField("copy") and e.copy.bytes > 0 for e in activities if e.activity == p.TraceActivity.TRACE_ACTIVITY_COPY)
                for e in device:
                    assert e.HasField("queue") and e.queue.device_id and e.queue.queue_id
                    assert e.device.HasField("correlation"), (backend, name, e)
                    c = e.device.correlation
                    assert c.earliest_start_ns <= c.latest_start_ns
                    expected_method = p.TraceClockMethod.TRACE_CLOCK_METHOD_BOUNDED if backend == "cuda" else p.TraceClockMethod.TRACE_CLOCK_METHOD_CALIBRATED
                    assert c.method == expected_method, (backend, name, e)
                if backend in ("opengl", "vulkan"):
                    for parent in [e for e in device_scopes if not e.HasField("node")]:
                        batch = [e for e in device if e.queue.queue_id == parent.queue.queue_id and
                                 e.queue.submission_id == parent.queue.submission_id]
                        window = parent.device.correlation
                        uncertainty = window.latest_start_ns - window.earliest_start_ns
                        for child in batch:
                            c = child.device.correlation
                            assert c.latest_start_ns - c.earliest_start_ns == uncertainty
                            assert c.earliest_start_ns >= window.earliest_start_ns
                            assert c.latest_start_ns + child.device.elapsed_ns <= window.latest_start_ns + parent.device.elapsed_ns
                        # Sequential node positions must stay sequential even
                        # when calibration uncertainty clips the pass envelope.
                        sequence = sorted([e for e in batch if e.HasField("node") and not e.HasField("program")],
                                          key=lambda e: e.node.schedule_index)
                        for previous, following in zip(sequence, sequence[1:]):
                            assert previous.device.correlation.latest_start_ns + previous.device.elapsed_ns <= following.device.correlation.latest_start_ns
                if enabled:
                    coverage = stopped.devices[0]
                    assert coverage.copy_intervals == len(copies)
                    assert coverage.calibrated_intervals + coverage.bounded_intervals == len(device)
            chunks, offset = [], 0
            while True:
                request = p.ExportChromeTraceRequest(trace_id=trace.trace_id, offset=offset, limit=3)
                chunk = profiling.export_chrome_trace(request)
                assert chunk.data == profiling.export_chrome_trace(request).data
                # Offsets count source events; fragments form one JSON document.
                chunks.append(chunk.data)
                assert chunk.next_offset == min(offset + 3, stopped.event_count)
                offset = chunk.next_offset
                if chunk.eof:
                    assert offset == stopped.event_count
                    break
            data = b"".join(chunks)
            exported = json.loads(data)
            assert exported["otherData"]["format"] == "volvoxai-trace/v6"
            assert exported["otherData"]["detail"] == ("nodes" if stopped.detail == p.TraceDetail.TRACE_DETAIL_NODES else "basic")
            assert exported["otherData"]["deviceTiming"] == stopped.device_timing
            assert exported["otherData"]["memory"] is True
            assert [d["nodeTimingAvailable"] for d in exported["otherData"]["devices"]] == [d.node_timing_available for d in stopped.devices]
            assert int(exported["otherData"]["droppedEvents"]) == stopped.dropped_events
            slices = [e for e in exported["traceEvents"] if e["ph"] == "X"]
            intervals = [e for e in exported["traceEvents"] if "deviceDurationNs" in e.get("args", {})]
            assert len(slices) == len(operations) + len(nodes) + len(activities) + sum(e["args"]["clockAligned"] for e in intervals)
            assert len(intervals) == len(device)
            assert all(e["ph"] == ("X" if e["args"]["clockAligned"] else "i") for e in intervals)
            profile = "lite" if "-lite" in library else "full"
            filename = f"native-{backend}-{profile}-{name}.trace.json"
            (output / filename).write_bytes(data)
            profiling.release_trace(p.TraceRef(trace_id=trace.trace_id))
            return {"phase": name, "memoryEvents": sum(e.HasField("memory") for e in events),
                    "allocators": check_trace_memory(events, stopped) if not overflow else [], "detail": int(stopped.detail), "deviceTiming": stopped.device_timing, "devices": [{"backend": d.backend, "support": int(d.support), "nodeTimingAvailable": d.node_timing_available, "passIntervals": d.pass_intervals, "nodeIntervals": d.node_intervals, "programIntervals": d.program_intervals, "failedIntervals": d.failed_intervals, "copyIntervals": d.copy_intervals, "calibratedIntervals": d.calibrated_intervals, "boundedIntervals": d.bounded_intervals, "hostCopyCalls": d.host_copy_calls, "hostWaitCalls": d.host_wait_calls, "hostSubmitCalls": d.host_submit_calls, "hostAwaits": d.host_awaits} for d in stopped.devices], "executions": len(identities), "hostOperations": len(operations),
                    "hostNodes": len(nodes), "deviceIntervals": len(device),
                    "deviceDurationNs": [e.device.elapsed_ns for e in device],
                    "droppedEvents": stopped.dropped_events, "trace": filename,
                    "traceSha256": hashlib.sha256(data).hexdigest()}

        cold = capture(4)  # Observe, capture and replay while collection is active.
        cases = [inspect(cold, "cold")]
        for _ in range(8):
            run()
        warm = capture(10)  # Attach to an already warmed CUDA replay plan.
        cases.append(inspect(capture(20, 4096), "overflow", overflow=True))
        for _ in range(8):
            run()
        cases.append(inspect(capture(3, device_timing=False), "nodes-host-only"))
        cases.append(inspect(capture(3, detail=p.TraceDetail.TRACE_DETAIL_BASIC), "basic-device"))
        cases.append(inspect(capture(3, device_timing=False, detail=p.TraceDetail.TRACE_DETAIL_BASIC), "basic-host-only"))
        snapshot = profiling.get_memory_snapshot(p.GetMemorySnapshotRequest(
            context_id=context.context_id, include_process=True)).snapshot
        assert snapshot.counters[0].name == "host_arena"
        assert len(snapshot.envelopes) == 2
        assert all(e.value_relation != p.MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE
                   and e.bytes.bytes > 0 for e in snapshot.envelopes)
        # Retained output aliases keep storage live; cached storage is reusable
        # without allocation/free events, and context retirement frees it once.
        lifetime_trace = start(device_timing=False)
        buffers = vx.VxBufferServiceClient(host)
        request = p.ExecuteTensorsRequest(context_id=context.context_id, inputs=[p.Tensor(
            name="x", dtype=p.DataType.DATA_TYPE_F32, shape=[WIDTH // 2, 2],
            inline=array.array("f", [1, 2] * (WIDTH // 2)).tobytes())])
        retained = inference.execute_tensors(request)
        alias = buffers.retain_buffers(p.BufferRefs(buffer_ids=[retained.outputs[0].buffer.buffer_id]))
        buffers.release_buffers(p.BufferRefs(buffer_ids=[retained.outputs[0].buffer.buffer_id]))
        second = inference.execute_tensors(request)
        buffers.release_buffers(p.BufferRefs(buffer_ids=[second.outputs[0].buffer.buffer_id]))
        reused = inference.execute_tensors(request)
        copied = buffers.copy_tensors(p.CopyTensorsRequest(sources=reused.outputs, inline_result=True))
        observed_values = array.array("f")
        observed_values.frombytes(copied.outputs[0].inline)
        assert list(observed_values) == [14.5, 1.75] * (WIDTH // 2)
        execution_count += 3
        buffers.release_buffers(p.BufferRefs(buffer_ids=[reused.outputs[0].buffer.buffer_id]))
        snapshot = profiling.get_memory_snapshot(p.GetMemorySnapshotRequest(context_id=context.context_id)).snapshot
        capacities = {counter.name: counter.measurement.bytes.bytes for counter in snapshot.counters}
        assert capacities["retained_result_capacity"] == 2 * WIDTH * 4
        assert capacities["idle_result_capacity"] == WIDTH * 4
        lifetime_ids = [r.report.lineage.execution_id for r in (retained, second, reused)]
        inference.release_execution_context(p.ExecutionContextRef(context_id=context.context_id))
        pool_name = "host.result" if backend == "cpu" else backend + ".result"
        retained_info = profiling.get_trace(p.TraceRef(trace_id=lifetime_trace.trace_id))
        assert next(a for a in retained_info.allocators if a.allocator == pool_name).live_bytes == WIDTH * 4
        buffers.release_buffers(p.BufferRefs(buffer_ids=[alias.buffers[0].buffer_id]))
        inference.release_compiled_model(p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        inference.release_model(p.ModelRef(model_id=model.model_id))
        lifetime_stopped = profiling.stop_trace(p.TraceRef(trace_id=lifetime_trace.trace_id))
        pool_name = "host.result" if backend == "cpu" else backend + ".result"
        result_capacity = next(a for a in lifetime_stopped.allocators if a.allocator == pool_name)
        assert result_capacity.allocated_bytes == result_capacity.freed_bytes == 2 * WIDTH * 4
        assert result_capacity.live_bytes == 0 and result_capacity.peak_bytes == 2 * WIDTH * 4
        assert all(a.live_bytes == 0 for a in lifetime_stopped.allocators), lifetime_stopped.allocators
        cases.append(inspect((lifetime_trace, lifetime_stopped, lifetime_ids), "memory-lifetime"))
        inference.release_runtime(p.RuntimeRef(runtime_id=runtime.runtime_id))
        cases.append(inspect(warm, "warm-after-retirement"))
        if counter and backend == "cuda":
            # Each cache observes once, captures once, then replays. Compare
            # warmed submissions only; overflow can retire the timed cache.
            replayed_on = [c for c in driver["on"] if c["graphLaunch"] == 1]
            replayed_off = [c for c in driver["off"] if c["graphLaunch"] == 1]
            assert len(replayed_on) >= 10 and len(replayed_off) >= 10
            on_waits = {c["streamSynchronize"] for c in replayed_on}
            off_waits = {c["streamSynchronize"] for c in replayed_off}
            assert len(on_waits) == 1 and on_waits == off_waits, (on_waits, off_waits)
            assert all(c["elapsedTime"] >= 5 for c in driver["on"][:14])
        if counter and backend in ("opengl", "vulkan"):
            wait = "glFinish" if backend == "opengl" else "vkWaitFence"
            assert {c[wait] for c in driver["on"][1:]} == {c[wait] for c in driver["off"]}, (wait, [c[wait] for c in driver["on"]], [c[wait] for c in driver["off"]])
            write = "glTimestamp" if backend == "opengl" else "vkTimestamp"
            assert all(c[write] >= 10 for c in driver["on"][:14])
        return {"backend": backend, "library": library,
                "librarySha256": hashlib.sha256((ROOT / library).read_bytes()).hexdigest(),
                "executionsChecked": execution_count, "cases": cases,
                **({"gpuDriverCalls": driver} if counter else {})}


def qualify_training(backend, output):
    """Check a one-step SGD update against a closed-form oracle, off/on/off."""
    graph = {"format": "volvox-graph/v1", "dimensions": {},
        "inputs": {"x": {"dtype": "float32", "shape": [1, 2]}},
        "nodes": [{"id": "projection", "opType": "Linear",
            "inputs": {"input": "x", "weight": "parameter"},
            "outputs": {"out": {"tensor": "logits", "dtype": "float32", "shape": [1, 2]}},
            "params": {"weight_layout": "din_dout"}}], "outputs": ["logits"]}
    header = json.dumps({"parameter": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
    header += b" " * (-len(header) % 8)
    weights = struct.pack("<Q", len(header)) + header + struct.pack("<4f", .2, -.4, .1, .3)
    probability = math.exp(.2) / (math.exp(.2) + math.exp(-.4))
    expected = [.2 + .1 * (1 - probability), -.4 - .1 * (1 - probability), .1, .3]
    with vx.open_library(ROOT / "native/libvolvoxai.so.0.6.0") as host:
        inference = vx.VxInferenceServiceClient(host)
        training = vx.VxTrainingServiceClient(host)
        profiling = vx.VxProfilingServiceClient(host)
        runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
        model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=p.ModelPackage(graph_document=json.dumps(graph).encode(), weight_shards=[weights])))
        trainer = training.create_trainer(p.CreateTrainerRequest(model_id=model.model_id, backend=backend, rng_seed=7))
        request = p.TrainStepRequest(trainer_id=trainer.trainer_id,
            inputs=[p.Tensor(name="x", shape=[1, 2], dtype=p.DataType.DATA_TYPE_F32, inline=struct.pack("<2f", 1, 0))],
            losses=[p.CrossEntropyLoss(name="classification", logits_name="logits", targets=p.Tensor(
                shape=[1], dtype=p.DataType.DATA_TYPE_I32, inline=struct.pack("<i", 0)))],
            trainable_names=["parameter"], optimizer=p.TrainerOptimizerOptions(
                kind=p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD, learning_rate=.1))

        def step():
            result = training.train_step(request)
            assert result.backend == backend and result.state == p.ResultState.RESULT_STATE_READY
            assert result.update_applied and abs(result.loss + math.log(probability)) < 1e-6
            shard = training.export_trainer_weights(p.ExportTrainerWeightsRequest(trainer_id=trainer.trainer_id)).shards[0]
            size, = struct.unpack_from("<Q", shard)
            metadata = json.loads(shard[8:8 + size])
            offset = metadata["parameter"]["data_offsets"][0]
            actual = struct.unpack_from("<4f", shard, 8 + size + offset)
            assert all(abs(a - b) < 1e-6 for a, b in zip(actual, expected)), (actual, expected)
            training.rollback_trainer(p.TrainerRef(trainer_id=trainer.trainer_id))

        step()
        trace = profiling.start_trace(p.StartTraceRequest(runtime_id=runtime.runtime_id,
            device_timing=True, detail=p.TraceDetail.TRACE_DETAIL_NODES))
        step()
        step()
        stopped = profiling.stop_trace(p.TraceRef(trace_id=trace.trace_id))
        step()
        assert profiling.stop_trace(p.TraceRef(trace_id=trace.trace_id)).event_count == stopped.event_count
        page = profiling.read_trace(p.ReadTraceRequest(trace_id=trace.trace_id))
        assert page.eof and stopped.dropped_events == 0
        assert len([e for e in page.events if e.name == "TrainStep"]) == 2
        device = [e for e in page.events if e.HasField("device")]
        assert bool(device) == (backend != "cpu")
        assert all(e.device.elapsed_ns > 0 for e in device)
        phases = [p.TracePhase.TRACE_PHASE_FORWARD, p.TracePhase.TRACE_PHASE_LOSS,
                  p.TracePhase.TRACE_PHASE_BACKWARD, p.TracePhase.TRACE_PHASE_GRADIENT,
                  p.TracePhase.TRACE_PHASE_OPTIMIZER]
        assert all(any(e.phase == phase for e in page.events) for phase in phases)
        programs = [e for e in device if e.HasField("program")]
        expected_device_phases = phases if backend == "cuda" else [phases[0], phases[2]]
        if backend != "cpu":
            assert all(any(e.phase == phase for e in programs) for phase in expected_device_phases)
            assert all(e.HasField("node") for e in programs if e.phase == p.TracePhase.TRACE_PHASE_BACKWARD)
            assert stopped.devices[0].program_intervals == len(programs)
        if backend in ("cpu", "opengl", "vulkan"):
            assert all(e.backend == "cpu" and e.HasField("host") for e in page.events
                       if e.phase in (p.TracePhase.TRACE_PHASE_LOSS, p.TracePhase.TRACE_PHASE_OPTIMIZER))
        assert any(e.tensor_name == "parameter" and e.phase == p.TracePhase.TRACE_PHASE_OPTIMIZER for e in page.events)
        data = export_trace(profiling, trace.trace_id)
        json.loads(data)
        filename = f"native-{backend}-training.trace.json"
        (output / filename).write_bytes(data)
        profiling.release_trace(p.TraceRef(trace_id=trace.trace_id))
        # Exercise the new phase instrumentation while updates accumulate and
        # clip. A separate CPU trainer is the oracle for both optimizer states.
        reference = training.create_trainer(p.CreateTrainerRequest(model_id=model.model_id, backend="cpu", rng_seed=7))
        candidate = training.create_trainer(p.CreateTrainerRequest(model_id=model.model_id, backend=backend, rng_seed=7))
        detail = profiling.start_trace(p.StartTraceRequest(runtime_id=runtime.runtime_id,
            device_timing=True, detail=p.TraceDetail.TRACE_DETAIL_NODES))
        optimizer_cases = []
        for kind, accumulation, clip in [(1, 1, 0), (1, 1, .05), (0, 1, .05), (0, 2, .05), (0, 2, .05)]:
            results, parameters = [], []
            for owner in [reference, candidate]:
                result = training.train_step(p.TrainStepRequest(trainer_id=owner.trainer_id,
                    inputs=request.inputs, losses=[p.CrossEntropyLoss(name="classification", logits_name="logits",
                        targets=request.losses[0].targets, normalizer=accumulation)], trainable_names=["parameter"],
                    accumulation_steps=accumulation, optimizer=p.TrainerOptimizerOptions(
                        kind=p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD if kind else p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW,
                        learning_rate=.03, weight_decay=.02, max_gradient_norm=clip)))
                results.append(result)
                if not result.update_applied:
                    continue
                shard = training.export_trainer_weights(p.ExportTrainerWeightsRequest(trainer_id=owner.trainer_id)).shards[0]
                size, = struct.unpack_from("<Q", shard)
                offset = json.loads(shard[8:8 + size])["parameter"]["data_offsets"][0]
                parameters.append(struct.unpack_from("<4f", shard, 8 + size + offset))
            expected_result, actual_result = results
            assert actual_result.backend == backend
            assert actual_result.update_applied == expected_result.update_applied
            assert actual_result.optimizer_step == expected_result.optimizer_step
            assert abs(actual_result.loss - expected_result.loss) < 2e-6
            maximum = max((abs(a - b) for a, b in zip(*parameters)), default=0)
            assert maximum < 2e-6, (backend, kind, maximum)
            optimizer_cases.append(dict(optimizer="sgd" if kind else "adamw", accumulation=accumulation,
                                        clip=clip, updateApplied=actual_result.update_applied, maxWeightError=maximum))
        training.rollback_trainer(p.TrainerRef(trainer_id=candidate.trainer_id))
        optimizer_info = profiling.stop_trace(p.TraceRef(trace_id=detail.trace_id))
        assert optimizer_info.dropped_events == 0
        optimizer_events = profiling.read_trace(p.ReadTraceRequest(trace_id=detail.trace_id, limit=4096))
        assert optimizer_events.eof
        if backend == "cuda":
            kernels = [e for e in optimizer_events.events if e.HasField("program") and e.HasField("device")]
            assert any("adamw" in e.program.name and e.phase == p.TracePhase.TRACE_PHASE_OPTIMIZER for e in kernels)
            assert all(e.phase != p.TracePhase.TRACE_PHASE_UNSPECIFIED for e in kernels)
        optimizer_export = export_trace(profiling, detail.trace_id)
        optimizer_filename = f"native-{backend}-training-optimizer.trace.json"
        (output / optimizer_filename).write_bytes(optimizer_export)
        profiling.release_trace(p.TraceRef(trace_id=detail.trace_id))
        report = {"backend": backend, "trainingUpdatesChecked": 4,
                  "trace": filename, "deviceIntervals": len(device), "droppedEvents": stopped.dropped_events,
                  "optimizerCases": optimizer_cases, "optimizerTrace": optimizer_filename,
                  "programs": [{"name": e.program.name, "entryPoint": e.program.entry_point,
                                "phase": int(e.phase), "node": e.node.schedule_index if e.HasField("node") else None,
                                "tensor": e.tensor_name} for e in programs]}
        (output / f"native-{backend}-training.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def qualify_queues(backend, output):
    """Two traced contexts share a runtime; an untraced runtime runs beside them."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    with vx.open_library(ROOT / "native/libvolvoxai.so.0.6.0") as host:
        inference, profiling = vx.VxInferenceServiceClient(host), vx.VxProfilingServiceClient(host)
        runtimes, compiled_models, contexts = [], [], []
        for _ in range(2):
            runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
            runtimes.append(runtime)
            model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=p.ModelPackage(graph_document=json.dumps(GRAPH).encode(), weight_shards=[WEIGHTS])))
            compiled = inference.compile_model(p.CompileModelRequest(model_id=model.model_id,
                policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=[backend], operator_fallback=p.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            compiled_models.append(compiled)
        for compiled in [compiled_models[0], compiled_models[0], compiled_models[1]]:
            contexts.append(inference.create_execution_context(p.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id)))

        def execute(context, value):
            result = inference.execute(p.ExecuteRequest(context_id=context.context_id, inputs=[p.Tensor(
                name="x", dtype=p.DataType.DATA_TYPE_F32, shape=[WIDTH // 2, 2],
                inline=array.array("f", [value, 2] * (WIDTH // 2)).tobytes())]))
            assert result.report.route.attested and result.report.backend == backend
            deadline = time.monotonic() + 30
            while inference.get_result(p.ResultRef(result_id=result.result_id)).state == p.ResultState.RESULT_STATE_PENDING:
                assert time.monotonic() < deadline
                time.sleep(.001)
            tensor = inference.read_output(p.ReadOutputRequest(result_id=result.result_id, name=OUTPUT)).tensor
            actual = array.array("f"); actual.frombytes(tensor.inline)
            first, second = value + 4.25, -3 * value + 7.5
            assert list(actual) == [first + 2 * second + .25, -3 * first + 4 * second - .5] * (WIDTH // 2)
            inference.release_result(p.ResultRef(result_id=result.result_id))

        for context in contexts:
            for _ in range(3): execute(context, 1)
        trace = profiling.start_trace(p.StartTraceRequest(runtime_id=runtimes[0].runtime_id,
            detail=p.TraceDetail.TRACE_DETAIL_NODES, device_timing=True, capacity_bytes=8 * 1024 * 1024))
        barrier = Barrier(3, timeout=30)

        def worker(index):
            try:
                for step in range(8):
                    barrier.wait(); execute(contexts[index], step + index)
            finally:
                if barrier.n_waiting: barrier.abort()

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(worker, range(3)))
        inference.release_execution_context(p.ExecutionContextRef(context_id=contexts[0].context_id))
        replacement = inference.create_execution_context(p.CreateExecutionContextRequest(
            compiled_model_id=compiled_models[0].compiled_model_id))
        execute(replacement, 4)
        ref = p.TraceRef(trace_id=trace.trace_id)
        stopped = profiling.stop_trace(ref)
        assert stopped.state == p.TraceState.TRACE_STATE_READY and stopped.dropped_events == 0
        events = profiling.read_trace(p.ReadTraceRequest(trace_id=trace.trace_id, limit=4096))
        assert events.eof
        queues = {}
        for e in events.events:
            assert e.lineage.runtime_id == runtimes[0].runtime_id
            assert e.lineage.context_id != contexts[2].context_id
            if e.HasField("device"):
                assert e.queue.device_id and e.queue.queue_id
                queues.setdefault(e.lineage.context_id, set()).add(e.queue.queue_id)
        assert set(queues) == {contexts[0].context_id, contexts[1].context_id, replacement.context_id}
        assert all(len(ids) == 1 for ids in queues.values())
        unique = set.union(*queues.values())
        assert len(unique) == (3 if backend == "cuda" else 1), queues
        chunks, offset = [], 0
        while True:
            chunk = profiling.export_chrome_trace(p.ExportChromeTraceRequest(trace_id=trace.trace_id,
                offset=offset, limit=1024))
            chunks.append(chunk.data)
            if chunk.eof: break
            offset = chunk.next_offset
        filename = f"native-{backend}-queues.trace.json"
        (output / filename).write_bytes(b"".join(chunks))
        profiling.release_trace(ref)
        report = dict(backend=backend, numericalExecutions=34, concurrentExecutions=24,
            untracedRuntimeExcluded=True, contextRecreation=True, uniqueQueues=len(unique), trace=filename)
        (output / f"native-{backend}-queues.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "opengl", "vulkan"])
    parser.add_argument("--output", type=Path, default=ROOT / "build/profiling-hardware")
    parser.add_argument("--gpu-probe", action="store_true", help="require the test-only Driver shim in LD_PRELOAD")
    parser.add_argument("--training", action="store_true", help="qualify full-profile SGD instead of inference")
    parser.add_argument("--queues", action="store_true", help="qualify concurrent native GPU queues and context recreation")
    args = parser.parse_args()
    if args.gpu_probe and args.backend not in ("cuda", "opengl", "vulkan"):
        parser.error("--gpu-probe requires a native GPU backend")
    if args.gpu_probe and args.training:
        parser.error("--gpu-probe qualifies inference; run --training separately")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.queues:
        if args.backend == "cpu" or args.training or args.gpu_probe:
            parser.error("--queues requires a native GPU backend and its own run")
        print(json.dumps(qualify_queues(args.backend, args.output)), flush=True)
        return
    if args.training:
        print(json.dumps(qualify_training(args.backend, args.output)), flush=True)
        return
    reports = []
    libraries = ["native/libvolvoxai.so.0.6.0"]
    if args.backend == "cpu":
        libraries.insert(0, "native/libvolvoxai-lite.so.0.6.0")
    for library in libraries:
        report = qualify(args.backend, library, args.output, args.gpu_probe)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    (args.output / f"native-{args.backend}.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
