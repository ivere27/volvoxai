"""The profiling workflow through public Python clients on both native profiles."""
import array
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
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
                trace = profiling.start_trace(p.StartTraceRequest(runtime_id=runtime.runtime_id,
                    detail=p.TraceDetail.TRACE_DETAIL_NODES, memory=True))
                self.assertFalse(trace.device_timing)
                with self.assertRaises(vx.VolvoxAIError) as caught:
                    profiling.read_trace(p.ReadTraceRequest(trace_id=trace.trace_id))
                self.assertEqual(caught.exception.status, p.NativeStatus.NATIVE_STATUS_BUSY)
                model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=p.ModelPackage(graph_document=json.dumps(GRAPH).encode())))
                compiled = inference.compile_model(p.CompileModelRequest(model_id=model.model_id))
                self.assertTrue(compiled.memory_bounds.bounds)
                context = inference.create_execution_context(p.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id))
                result = inference.execute(p.ExecuteRequest(context_id=context.context_id, inputs=[p.Tensor(
                    name="x", dtype=p.DataType.DATA_TYPE_F32, shape=[4], inline=array.array("f", [1, -2, 3, 4]).tobytes())]))
                output = inference.read_output(p.ReadOutputRequest(result_id=result.result_id, name="sum"))
                values = array.array("f")
                values.frombytes(output.tensor.inline)
                self.assertEqual(list(values), [2, -4, 6, 8])
                self.assertEqual(result.metrics.output_bytes, 16)
                observed = profiling.get_memory_snapshot(p.GetMemorySnapshotRequest(runtime_id=runtime.runtime_id, include_process=True))
                self.assertEqual(observed.snapshot.counters[0].name, "unconsumed_result_budget")
                self.assertGreaterEqual(observed.snapshot.observation_end_ns, observed.snapshot.observation_start_ns)
                self.assertEqual(len(observed.snapshot.envelopes), 2)
                ref = p.TraceRef(trace_id=trace.trace_id)
                stopped = profiling.stop_trace(ref)
                self.assertEqual(stopped.state, p.TraceState.TRACE_STATE_READY)
                inference.release_result(p.ResultRef(result_id=result.result_id))
                inference.release_execution_context(p.ExecutionContextRef(context_id=context.context_id))
                inference.release_compiled_model(p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                inference.release_model(p.ModelRef(model_id=model.model_id))
                inference.release_runtime(p.RuntimeRef(runtime_id=runtime.runtime_id))
                page = profiling.read_trace(p.ReadTraceRequest(trace_id=trace.trace_id))
                self.assertEqual({event.name for event in page.events if event.HasField("host") and not event.HasField("node")},
                                 {"LoadModel", "CompileModel", "Execute"})
                self.assertEqual(len(page.process_memory), 2)
                offset, chunks = 0, []
                while True:
                    chunk = profiling.export_chrome_trace(p.ExportChromeTraceRequest(trace_id=trace.trace_id,
                        offset=offset, limit=3))
                    chunks.append(chunk.data)
                    if chunk.eof:
                        break
                    offset = chunk.next_offset
                exported = json.loads(b"".join(chunks))
                self.assertEqual(exported["otherData"]["format"], "volvoxai-trace/v6")
                self.assertEqual(exported["otherData"]["detail"], "nodes")
                self.assertIs(exported["otherData"]["deviceTiming"], False)
                self.assertEqual(len([event for event in exported["traceEvents"] if event["ph"] == "X"]), len([event for event in page.events if event.HasField("host")]))
                self.assertTrue(any(event["ph"] == "C" for event in exported["traceEvents"]))
                profiling.release_trace(ref)
                profiling.release_trace(ref)
                with self.assertRaises(vx.VolvoxAIError):
                    profiling.stop_trace(ref)


if __name__ == "__main__":
    unittest.main()
