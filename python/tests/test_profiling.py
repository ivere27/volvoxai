"""The profiling workflow through public Python clients on both native profiles."""
import array
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph, feedback_graph

import volvoxai as vx  # noqa: E402

p = vx.pb
GRAPH = {
    "format": "volvox-graph/v1", "dimensions": {},
    "inputs": {"x": {"dtype": "float32", "shape": [4]}},
    "nodes": [{"id": "add", "opType": "Add", "inputs": {"a": "x", "b": "x"},
               "outputs": {"out": {"tensor": "sum", "dtype": "float32", "shape": [4]}}, "params": {}}],
    "outputs": ["sum"],
}


class ProfilingTest(unittest.TestCase):
    def test_native_profiles_export_after_parent_retirement(self):
        for library in ("libvolvoxai-lite.so", "libvolvoxai.so"):
            with self.subTest(library=library), vx.open_library(ROOT / "native" / library) as host:
                inference = vx.VxInferenceServiceClient(host)
                profiling = vx.VxProfilingServiceClient(host)
                runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
                trace = profiling.start_trace(p.StartTraceRequest(runtime_id=runtime.runtime_id, options=p.TraceOptions(
                    detail=p.TraceDetail.TRACE_DETAIL_NODES, memory=True, utilization=True, execution_plans=True)))
                self.assertFalse(trace.options.device_timing)
                self.assertEqual(trace.options.capacity_bytes, 4 * 1024 * 1024)
                with self.assertRaises(vx.VolvoxAIError) as caught:
                    profiling.list_trace_events(p.ListTraceEventsRequest(trace_id=trace.trace_id))
                self.assertEqual(caught.exception.status, p.NativeStatus.NATIVE_STATUS_BUSY)
                model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=p.ModelPackage(graph_document=json.dumps(stateless_graph(GRAPH)).encode())))
                compiled = inference.compile_model(p.CompileModelRequest(model_id=model.model_id))
                self.assertTrue(compiled.memory_bounds.bounds)
                self.assertGreaterEqual(compiled.metrics.host_time_ns, 0)
                context = inference.create_execution_context(p.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id, graph=p.GraphExecution()))
                platform = vx.VxPlatformServiceClient(host)
                start = platform.get_monotonic_time(p.Empty()).nanoseconds
                result = inference.execute(p.ExecuteRequest(context_id=context.context_id, inputs=p.InputBindings(values=[p.Tensor(
                    name="x", dtype=p.DataType.DATA_TYPE_F32, shape=[4], inline=array.array("f", [1, -2, 3, 4]).tobytes())]), scope=p.ExecutionScope(entrypoint="evaluate", full=p.Empty()), outputs=p.OutputOptions(storage=p.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)))
                end = platform.get_monotonic_time(p.Empty()).nanoseconds
                profiling.annotate_trace(p.AnnotateTraceRequest(trace_id=trace.trace_id, name="frame 1", start_ns=start, end_ns=end))
                buffers = vx.VxBufferServiceClient(host)
                output = buffers.copy_tensors(p.CopyTensorsRequest(sources=result.outputs, inline_result=True)).outputs[0]
                values = array.array("f")
                values.frombytes(output.inline)
                self.assertEqual(list(values), [2, -4, 6, 8])
                self.assertEqual(result.metrics.output_bytes, 16)
                observed = profiling.get_resource_snapshot(p.GetResourceSnapshotRequest(runtime_id=runtime.runtime_id))
                self.assertEqual(observed.snapshot.memory.unconsumed_result_bytes, 16)
                self.assertGreaterEqual(observed.snapshot.observation_end_ns, observed.snapshot.observation_start_ns)
                self.assertEqual(observed.snapshot.cpu.status, p.ObservationStatus.OBSERVATION_STATUS_AVAILABLE)
                self.assertGreater(observed.snapshot.process.resident.bytes, 0)
                module = profiling.get_resource_snapshot(p.GetResourceSnapshotRequest(module=p.Empty()))
                self.assertEqual(module.snapshot.memory.subject.kind, p.MemoryOwnerKind.MEMORY_OWNER_KIND_MODULE)
                ref = p.TraceRef(trace_id=trace.trace_id)
                stopped = profiling.stop_trace(ref)  # Replies when READY.
                self.assertEqual(stopped.state, p.TraceState.TRACE_STATE_READY)
                buffers.release_buffers(p.BufferRefs(buffer_ids=[tensor.buffer.buffer_id for tensor in result.outputs]))
                inference.release_execution_context(p.ExecutionContextRef(context_id=context.context_id))
                inference.release_compiled_model(p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                inference.release_model(p.ModelRef(model_id=model.model_id))
                inference.release_runtime(p.RuntimeRef(runtime_id=runtime.runtime_id))
                page = profiling.list_trace_events(p.ListTraceEventsRequest(trace_id=trace.trace_id))
                self.assertEqual(page.next_page_token, "")
                self.assertEqual(page.total_size, stopped.events.count)
                self.assertEqual({event.name for event in page.events
                                  if event.HasField("host") and not event.HasField("node")
                                  and event.activity == p.TraceActivity.TRACE_ACTIVITY_COMPUTE},
                                 {"LoadModel", "CompileModel", "Execute"})
                annotation = [event for event in page.events if event.activity == p.TraceActivity.TRACE_ACTIVITY_ANNOTATION]
                self.assertEqual([event.name for event in annotation], ["frame 1"])
                self.assertEqual(annotation[0].host.duration_ns, end - start)
                first = profiling.list_trace_events(p.ListTraceEventsRequest(trace_id=trace.trace_id, page_size=1))
                self.assertEqual(len(first.events), 1)
                second = profiling.list_trace_events(p.ListTraceEventsRequest(
                    trace_id=trace.trace_id, page_size=1, page_token=first.next_page_token))
                self.assertEqual(second.events[0].sequence, page.events[1].sequence)
                with self.assertRaises(vx.VolvoxAIError):
                    profiling.list_trace_events(p.ListTraceEventsRequest(trace_id=trace.trace_id, page_token="bogus"))
                resources = profiling.list_trace_resource_snapshots(p.ListTraceResourceSnapshotsRequest(trace_id=trace.trace_id))
                self.assertGreaterEqual(len(resources.resource_snapshots), 2)
                self.assertTrue(all(s.cpu.status == p.ObservationStatus.OBSERVATION_STATUS_AVAILABLE for s in resources.resource_snapshots))
                self.assertEqual(stopped.plans.count, 1)
                plan = profiling.get_trace_plan(p.GetTracePlanRequest(trace_id=trace.trace_id, plan_id=1)).plan
                self.assertEqual(plan.steps[0].source_node_ids, ["add"])
                self.assertTrue(plan.source_mapping_complete)
                self.assertTrue(all(e.plan_id == plan.plan_id for e in page.events if e.lineage.execution_id))
                with self.assertRaises(vx.VolvoxAIError) as missing:
                    profiling.get_trace_plan(p.GetTracePlanRequest(trace_id=trace.trace_id, plan_id=2))
                self.assertEqual(missing.exception.status, p.NativeStatus.NATIVE_STATUS_NOT_FOUND)
                request, chunks = p.ExportChromeTraceRequest(trace_id=trace.trace_id, page_size=3), []
                while True:
                    chunk = profiling.export_chrome_trace(request)
                    chunks.append(chunk.data)
                    if not chunk.next_page_token:
                        break
                    request.page_token = chunk.next_page_token
                exported = json.loads(b"".join(chunks))
                self.assertEqual(exported["otherData"]["format"], "volvoxai-trace/v8")
                self.assertEqual(exported["otherData"]["detail"], "nodes")
                self.assertIs(exported["otherData"]["deviceTiming"], False)
                self.assertEqual(len([event for event in exported["traceEvents"] if event["ph"] == "X"]),
                                 len([event for event in page.events if event.HasField("host")]))
                self.assertTrue(any(event.get("cat") == "user.annotation" for event in exported["traceEvents"]))
                self.assertTrue(any(event["ph"] == "C" for event in exported["traceEvents"]))
                profiling.release_trace(ref)
                profiling.release_trace(ref)
                with self.assertRaises(vx.VolvoxAIError):
                    profiling.stop_trace(ref)

    def test_session_trace_and_debug_scopes(self):
        import tempfile
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps(stateless_graph(GRAPH)))
            with vx.InferenceSession(path) as session:
                with session.trace(detail="nodes", execution_plans=True) as trace:
                    with trace.annotate("preprocess"):
                        data = np.array([1, -2, 3, 4], dtype=np.float32)
                    session.run(data)
                self.assertEqual(trace.info.state, vx.pb.TraceState.TRACE_STATE_READY)
                self.assertIn("preprocess", {event.name for event in trace.events})
                self.assertEqual(len(trace.plans), 1)
                chrome = Path(directory) / "trace.json"
                trace.export_chrome_trace(chrome)
                self.assertIn("traceEvents", json.loads(chrome.read_text()))
                with session.debug(np.array([1, -2, np.nan, 4], dtype=np.float32)) as debug:
                    self.assertEqual(debug.stop_reason, vx.pb.DebugStopReason.DEBUG_STOP_REASON_ENTRY)
                    debug.continue_(break_on_nonfinite=True)
                    self.assertEqual(debug.state, vx.pb.DebugState.DEBUG_STATE_COMPLETED)
                    events = debug.events()
                    output = events[-1].snapshots[0]
                    self.assertEqual(output.statistics.nan_count, 1)
                    values = debug.tensor(output)
                    self.assertEqual(values.tolist()[:2], [2, -4])
                    with debug.tensor_buffer(output) as retained:
                        self.assertEqual(np.from_dlpack(retained).tolist()[:2], [2, -4])
                    self.assertIsNone(debug.failure)


def library_available() -> bool:
    try:
        vx.find_library()
    except vx.VolvoxAIError:
        return False
    return True


def write_package(directory: Path) -> Path:
    graph = {
        "format": "volvox-graph/v1", "dimensions": {}, "inputs": {"x": {"dtype": "float32", "shape": [2, 3]}},
        "nodes": [
            {"id": "fc", "opType": "Linear", "inputs": {"input": "x", "weight": "fc.w", "bias": "fc.b"},
             "outputs": {"out": {"tensor": "h", "dtype": "float32", "shape": [2, 4]}},
             "params": {"weight_layout": "dout_din"}},
            {"id": "act", "opType": "GELU", "inputs": {"input": "h"},
             "outputs": {"out": {"tensor": "y", "dtype": "float32", "shape": [2, 4]}}, "params": {}},
        ],
        "outputs": ["y"],
    }
    package = directory / "package"
    package.mkdir()
    (package / "graph.json").write_text(json.dumps(stateless_graph(graph)), encoding="utf-8")
    save_file({"fc.w": np.sin(np.arange(12, dtype=np.float32)).reshape(4, 3),
               "fc.b": np.array([.1, -.2, .3, -.4], dtype=np.float32)}, str(package / "model.safetensors"))
    return package


def statuses(info) -> dict[str, tuple[int, int]]:
    return {row.mechanism: (row.status, row.ranges) for row in info.external}


@unittest.skipUnless(library_available(), "libvolvoxai.so is not built")
class ProfilingExternalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-profiling-external-"))
        cls.package = write_package(cls.directory)
        cls.x = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_cpu_summary_and_nvtx_ranges(self) -> None:
        with vx.InferenceSession(self.package) as session:
            with session.trace(detail="nodes", execution_plans=True, external_annotations=True) as trace:
                for _ in range(3):
                    session.run(self.x)
            available = vx.pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE
            nvtx = statuses(trace.info)["nvtx"]
            self.assertEqual(nvtx[0], available)
            self.assertGreaterEqual(nvtx[1], 3 * 3)
            summary = trace.summary("node")
            linear = next(row for row in summary.rows if row.name == "Linear")
            self.assertEqual(linear.count, 3)
            self.assertEqual(linear.cost_per_call.multiply_accumulates, 24)
            self.assertEqual(list(linear.source_node_ids), ["fc"])
            table = trace.table("operator")
            self.assertIn("Linear", table)
            self.assertIn("GELU", table)

    def device_labels(self, backend: str, mechanism: str) -> None:
        try:
            session = vx.InferenceSession(self.package, backend=backend)
        except vx.VolvoxAIError as error:
            self.skipTest(f"{backend} is not usable here: {error}")
        with session:
            with session.trace(detail="nodes", external_annotations=True) as trace:
                session.run(self.x)
            status, ranges = statuses(trace.info)[mechanism]
            if status == vx.pb.ObservationStatus.OBSERVATION_STATUS_UNSUPPORTED:
                self.skipTest(f"{mechanism} is not offered by this driver")
            self.assertEqual(status, vx.pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE)
            self.assertEqual(ranges, 2)

    def test_vulkan_debug_utils_labels(self) -> None:
        self.device_labels("vulkan", "vk_debug_utils")

    def test_opengl_khr_debug_groups(self) -> None:
        self.device_labels("opengl", "khr_debug")


if __name__ == "__main__":
    unittest.main()
