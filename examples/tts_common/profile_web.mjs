#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { TtsSession, waveformDifference, tensorValues } from "./TtsSession.js";
import { loadRuntimeRelease, releaseUrls } from "../common/RuntimeRelease.js";
import { collectProfile } from "../../tools/profile_report.mjs";
// Capture live requests through the same shared-host session as benchmark_web.
const args = globalThis.Deno?.args ?? process.argv.slice(2);
const options = { backend: "webgpu", warmups: 2 };
const usage = "Expected --manifest FILE --out FILE.json [--backend wasm|webgpu --warmups 2]";
for (let i = 0; i < args.length; i += 2) {
  if (!["--manifest", "--out", "--backend", "--warmups"].includes(args[i]) || !args[i + 1]) {
    throw Error(usage);
  }
  options[args[i].slice(2)] = args[i + 1];
}
if (!options.manifest || !options.out || !["wasm", "webgpu"].includes(options.backend)) {
  throw Error(usage);
}
options.warmups = Number(options.warmups);
if (!Number.isSafeInteger(options.warmups) || options.warmups < 0) {
  throw Error("--warmups must be a nonnegative integer");
}
if (!options.out.endsWith(".json")) {
  throw Error("--out must end in .json so the waveform files have distinct paths");
}
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const originalFetch = globalThis.fetch;
globalThis.fetch = async (source, init) => {
  const url = typeof source === "string" ? source : source instanceof URL ? source.href : source?.url;
  return url?.startsWith("file:") ? new Response(await fs.readFile(fileURLToPath(url))) : originalFetch(source, init);
};
const report = {
  format: "volvoxai-tts-web-profile/v1",
  backend: options.backend,
  stages: {},
  stage_ms: {},
  api_counts: {},
  transfer_note: "mapAsync completion includes preceding GPU execution and scheduling; it is not isolated transfer bandwidth. GPU writes are input/parameter uploads, with bytes counted at the API.",
  trace_overhead: "Node timing splits passes; use benchmark_web for latency. Timestamp capacity and dropped intervals are in each trace info.",
};
let session;
const traces = [];
const restores = [];
let collecting = false;
try {
  if (options.backend === "webgpu") {
    const gpu = navigator.gpu;
    const request = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = async (...adapterArgs) => {
      const adapter = await request(...adapterArgs);
      if (!adapter) return adapter;
      const requestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async (...deviceArgs) => {
        const device = await requestDevice(...deviceArgs);
        const countCall = (name, fn, callArgs, bytes = 0) => {
          if (!collecting) return fn(...callArgs);
          const row = report.api_counts[name] ??= { count: 0, host_ms: 0, bytes: 0 };
          const start = performance.now();
          const value = fn(...callArgs);
          const elapsed = performance.now() - start;
          row.host_ms += elapsed;
          row.count++;
          row.bytes += bytes;
          if (name === "writeBuffer" || name === "createBuffer") {
            const label = callArgs[0].label || "(unlabeled)";
            row.labels ??= {};
            const item = row.labels[label] ??= { count: 0, bytes: 0, host_ms: 0 };
            item.count++;
            item.bytes += bytes;
            item.host_ms += elapsed;
          }
          return value;
        };
        for (const name of ["createBuffer", "createBindGroup", "createComputePipeline", "createShaderModule", "createCommandEncoder"]) {
          const original = device[name].bind(device);
          device[name] = (...callArgs) => {
            const bytes = name === "createBuffer" ? Number(callArgs[0].size) : 0;
            const value = countCall(name, original, callArgs, bytes);
            if (name === "createBuffer") {
              const map = value.mapAsync.bind(value);
              value.mapAsync = (...mapArgs) => {
                if (!collecting) return map(...mapArgs);
                const row = report.api_counts.mapAsync ??= { count: 0, completion_ms: 0, bytes: 0 };
                const start = performance.now();
                row.count++;
                row.bytes += Number(mapArgs[2] ?? value.size - (mapArgs[1] ?? 0));
                return map(...mapArgs).finally(() => {
                  row.completion_ms += performance.now() - start;
                });
              };
            }
            return value;
          };
        }
        for (const name of ["submit", "writeBuffer"]) {
          const original = device.queue[name].bind(device.queue);
          device.queue[name] = (...callArgs) => {
            // Typed-array offsets and sizes use elements; ArrayBuffer uses bytes.
            const data = callArgs[2];
            const unit = data?.BYTES_PER_ELEMENT ?? 1;
            let bytes = 0;
            if (name === "writeBuffer") {
              bytes = callArgs[4] === undefined
                ? data.byteLength - (callArgs[3] ?? 0) * unit
                : callArgs[4] * unit;
            }
            return countCall(name, original, callArgs, bytes);
          };
        }
        return device;
      };
      return adapter;
    };
    restores.push(() => gpu.requestAdapter = request);
  }
  const manifestUrl = pathToFileURL(path.resolve(options.manifest));
  const manifestBytes = await fs.readFile(manifestUrl);
  report.manifest_sha256 = hash(manifestBytes);
  report.caller_sha256 = hash(await fs.readFile(fileURLToPath(import.meta.url)));
  const runtime = await loadRuntimeRelease("full");
  const { pb } = runtime.api;
  const urls = releaseUrls("full");
  report.artifacts = {
    js_sha256: hash(await fs.readFile(urls.jsUrl)),
    wasm_sha256: hash(await fs.readFile(urls.wasmUrl)),
  };
  session = await TtsSession.open({ manifestUrl, ...runtime, backend: options.backend });
  const fixture = session.manifest.cases[0];
  const input = await session.prepare(fixture);
  // Count the final ordinary request, then capture separate node/basic traces.
  let ordinary;
  for (let i = 0; i < 1 + options.warmups; i++) {
    collecting = i === options.warmups;
    ordinary = await session.execute(input);
    collecting = false;
  }
  report.untraced_api_counts = report.api_counts;
  report.api_counts = {};
  report.untraced_ms = ordinary.elapsed_ms;
  await fs.mkdir(path.dirname(path.resolve(options.out)), { recursive: true });
  const ordinaryBytes = new Uint8Array(ordinary.audio.buffer, ordinary.audio.byteOffset, ordinary.audio.byteLength);
  report.untraced_output_sha256 = hash(ordinaryBytes);
  await fs.writeFile(options.out.replace(/\.json$/, ".untraced.audio.f32"), ordinaryBytes);
  const descriptor = fixture.output_files.find((x) => x.name === (fixture.audio_output ?? "audio"));
  const reference = tensorValues(await session.readBytes(descriptor.path), descriptor);
  report.validation = waveformDifference(ordinary.audio, reference, { waveform: fixture.validation?.waveform });
  const profiler = new runtime.api.VxProfilingServiceClient(session.host);
  for (const { name, session: stage } of session.stages) {
    const runtimeId = stage.resources.find((x) => x[1] === "RuntimeRef")[2].runtimeId;
    const trace = await profiler.startTrace(new pb.StartTraceRequest({
      runtimeId,
      options: new pb.TraceOptions({
        detail: pb.TraceDetail.TRACE_DETAIL_NODES,
        deviceTiming: options.backend === "webgpu",
        memory: true,
        utilization: true,
        executionPlans: true,
        capacityBytes: 67108864n,
      }),
    }));
    if (trace.report?.status) throw Error(trace.report.message);
    traces.push({ name, id: trace.traceId });
    const execute = stage.executeTensors.bind(stage);
    stage.executeTensors = async (request) => {
      const start = performance.now();
      const result = await execute(request);
      report.stage_ms[name] = performance.now() - start;
      return result;
    };
  }
  collecting = true;
  const result = await session.execute(input);
  collecting = false;
  report.request_ms = result.elapsed_ms;
  report.route = result.report.route;
  const outputBytes = new Uint8Array(result.audio.buffer, result.audio.byteOffset, result.audio.byteLength);
  report.output_sha256 = hash(outputBytes);
  await fs.mkdir(path.dirname(path.resolve(options.out)), { recursive: true });
  await fs.writeFile(options.out.replace(/\.json$/, ".audio.f32"), outputBytes);
  for (const { name, id } of traces) {
    await profiler.stopTrace(new pb.TraceRef({ traceId: id }));
    report.stages[name] = await collectProfile(profiler, pb, id);
    await profiler.releaseTrace(new pb.TraceRef({ traceId: id }));
  }
  report.node_api_counts = report.api_counts;
  report.api_counts = {};
  report.node_stage_ms = report.stage_ms;
  report.stage_ms = {};
  for (const { name, session: stage } of session.stages) {
    const runtimeId = stage.resources.find((x) => x[1] === "RuntimeRef")[2].runtimeId;
    const trace = await profiler.startTrace(new pb.StartTraceRequest({
      runtimeId,
      options: new pb.TraceOptions({
        detail: pb.TraceDetail.TRACE_DETAIL_BASIC,
        deviceTiming: options.backend === "webgpu",
        capacityBytes: 4194304n,
      }),
    }));
    if (trace.report?.status) throw Error(trace.report.message);
    traces.find((x) => x.name === name).basicId = trace.traceId;
  }
  collecting = true;
  const basic = await session.execute(input);
  collecting = false;
  report.basic_request_ms = basic.elapsed_ms;
  report.basic_api_counts = report.api_counts;
  delete report.api_counts;
  report.basic_stage_ms = report.stage_ms;
  delete report.stage_ms;
  for (const { name, basicId } of traces) {
    await profiler.stopTrace(new pb.TraceRef({ traceId: basicId }));
    report.stages[name + "_basic"] = await collectProfile(profiler, pb, basicId);
    await profiler.releaseTrace(new pb.TraceRef({ traceId: basicId }));
  }
  report.status = "captured";
} catch (error) {
  report.status = "failed";
  report.error = String(error);
  throw error;
} finally {
  collecting = false;
  try {
    // Closing the owning host also retires any trace left by an earlier error.
    await session?.close();
  } finally {
    for (const restore of restores) restore();
    globalThis.fetch = originalFetch;
    await fs.mkdir(path.dirname(path.resolve(options.out)), { recursive: true });
    const json = JSON.stringify(report, (_key, value) => (
      typeof value === "bigint" ? String(value) : value
    ), 2);
    await fs.writeFile(options.out, json + "\n");
  }
}
