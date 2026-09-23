#!/usr/bin/env python3
"""Paired native release benchmark using each release's generated Python codec.

Pass repository/archive root and native library path. An archive must contain
python/volvoxai, runtime/generated/python and package.json as well as the module.
"""
import argparse
import array
import json
import math
from pathlib import Path
import sys
import time
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("library", type=Path)
parser.add_argument("--on", action="store_true", help="measure active node collection separately")
parser.add_argument("--paired", action="store_true", help="accept sample/next commands on stdin")
parser.add_argument("--backend", choices=["cpu", "cuda", "opengl", "vulkan"], default="cpu")
parser.add_argument("--warmup", type=int, default=2000)
parser.add_argument("--batch", type=int, default=50)
args = parser.parse_args()
sys.path.insert(0, str(args.root.resolve() / "python"))
import volvoxai as vx  # noqa: E402

p = vx.pb
results = []


def summary(values):
    ordered = sorted(values)
    return {"p50Ms": ordered[math.ceil(len(values) * .5) - 1],
            "p95Ms": ordered[math.ceil(len(values) * .95) - 1], "samplesMs": values}


with vx.open_library(args.library.resolve(), loader=Path("native/libsynurang_module_host.so").resolve()) as host:
    inference = vx.VxInferenceServiceClient(host)
    runtime = inference.create_runtime(p.CreateRuntimeRequest(cpu_threads=1))
    workloads = [(8, 64), (128, 64), (64, 8192)] if args.backend == "cpu" else [(8, 64), (64, 64), (32, 8192)]
    for nodes, width in workloads:
        document = {
            "format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"dtype": "float32", "shape": [width]}},
            "nodes": [{"id": f"add{i}", "opType": "Add",
                       "inputs": {"a": f"out{i-1}" if i else "x", "b": "x"},
                       "outputs": {"out": {"tensor": f"out{i}", "dtype": "float32", "shape": [width]}},
                       "params": {}} for i in range(nodes)],
            "outputs": [f"out{nodes-1}"],
        }
        shape, shards = [width], []
        if args.backend != "cpu":
            # Native GPU admission proves dense layers. Identity weights plus
            # unit bias preserve the independent nodes+1 numerical oracle.
            shape = [width // 64, 64]
            document["inputs"]["x"]["shape"] = shape
            for i, node in enumerate(document["nodes"]):
                node.update(opType="Linear", inputs={"input": f"out{i-1}" if i else "x", "weight": "w", "bias": "b"},
                            params={"weight_layout": "dout_din"})
                node["outputs"]["out"]["shape"] = shape
            weights = array.array("f", [float(i == j) for i in range(64) for j in range(64)] + [1.] * 64).tobytes()
            header = json.dumps({"w": {"dtype": "F32", "shape": [64, 64], "data_offsets": [0, 16384]},
                                 "b": {"dtype": "F32", "shape": [64], "data_offsets": [16384, 16640]}}).encode()
            header += b" " * (-len(header) % 8)
            shards = [struct.pack("<Q", len(header)) + header + weights]
        model = inference.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
            package=p.ModelPackage(graph_document=json.dumps(document).encode(), weight_shards=shards)))
        compiled = inference.compile_model(p.CompileModelRequest(model_id=model.model_id,
            policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=[args.backend], operator_fallback=p.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        assert compiled.report.backend == args.backend and compiled.report.route.attested
        context = inference.create_execution_context(p.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id))
        request = p.ExecuteRequest(context_id=context.context_id, inputs=[p.Tensor(name="x",
            dtype=p.DataType.DATA_TYPE_F32, shape=shape, inline=array.array("f", [1] * width).tobytes())])

        def run(check=False):
            result = inference.execute(request)
            if check:
                output = inference.read_output(p.ReadOutputRequest(result_id=result.result_id, name=document["outputs"][0]))
                values = array.array("f")
                values.frombytes(output.tensor.inline)
                assert len(values) == width and all(value == nodes + 1 for value in values)
            inference.release_result(p.ResultRef(result_id=result.result_id))
            if hasattr(result, "metrics"):
                return result.metrics.host_time_ns / 1e6 if hasattr(result.metrics, "host_time_ns") else result.metrics.host_time_ms
            return result.report.timings.execution_time_ms

        for i in range(args.warmup):
            run(i == args.warmup - 1)
        profiling = vx.VxProfilingServiceClient(host) if args.on else None
        trace = profiling.start_trace(p.StartTraceRequest(runtime_id=runtime.runtime_id,
            detail=p.TraceDetail.TRACE_DETAIL_NODES, device_timing=True, capacity_bytes=64 * 1024 * 1024)) if profiling else None
        wall, engine = [], []
        if args.paired:
            print(json.dumps({"ready": {"nodes": nodes, "width": width}}), flush=True)
        for sample in range(21):
            if args.paired:
                assert input() == "sample"
            elapsed = 0
            start = time.perf_counter()
            for i in range(args.batch):
                elapsed += run()
            wall.append((time.perf_counter() - start) * 1000 / args.batch)
            engine.append(elapsed / args.batch)
            if args.paired:
                print(json.dumps({"sample": sample, "wallMs": wall[-1], "engineMs": engine[-1]}), flush=True)
        if args.paired:
            assert input() == "next"
        captured = None
        if trace:
            info = profiling.stop_trace(p.TraceRef(trace_id=trace.trace_id))
            assert info.state == p.TraceState.TRACE_STATE_READY and info.dropped_events == 0, info
            captured = {"eventCount": info.event_count, "droppedEvents": info.dropped_events}
            profiling.release_trace(p.TraceRef(trace_id=trace.trace_id))
        results.append({"nodes": nodes, "width": width, "wall": summary(wall), "engine": summary(engine),
                        **({"capture": captured} if captured is not None else {})})
        inference.release_execution_context(p.ExecutionContextRef(context_id=context.context_id))
        inference.release_compiled_model(p.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        inference.release_model(p.ModelRef(model_id=model.model_id))
print(json.dumps({"library": str(args.library), "backend": args.backend, "profiling": args.on,
                  "warmup": args.warmup, "samples": 21, "batch": args.batch, "cases": results}))
