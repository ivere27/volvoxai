import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {fixture, p, ok, safetensors, tensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxDebugServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';
import {collectDebugSession, compareDebugSessions, ptqActivationName} from '../tools/debug_report.mjs';

// Node debugging is a full-profile service.
const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.wasm`, import.meta.url).pathname;
const graph = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [4]}},
  nodes: [
    {id: 'relu', opType: 'ReLU', inputs: {input: 'x'}, outputs: {out: {tensor: 'a', dtype: 'float32', shape: [4]}}, params: {}},
    {id: 'double', opType: 'Add', inputs: {a: 'a', b: 'a'}, outputs: {out: {tensor: 'b', dtype: 'float32', shape: [4]}}, params: {}},
  ], outputs: ['b']};

async function session(f, debug, data, shape = [4]) {
  let info = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({forward: new p.DebugForward({
    compiledModelId: f.compiled.compiledModelId, inputs: tensors({x: {data, shape}})}), capture: new p.DebugCapture({values: true})})));
  info = ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision})));
  assert.equal(info.state, p.DebugState.DEBUG_STATE_COMPLETED);
  return collectDebugSession(debug, p, info.debugSessionId);
}

test('two sessions align by source node and report the first observable mismatch', async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const debug = new VxDebugServiceClient(reportTransport(f.host));
  f.compiled = await f.compile(await f.load(graph));
  const reference = await session(f, debug, Float32Array.of(-1, 2, -3, 4));
  const same = compareDebugSessions(reference, await session(f, debug, Float32Array.of(-1, 2, -3, 4)));
  assert.equal(same.comparable, true);
  assert.equal(same.mode, 'exact');
  assert.equal(same.firstMismatch, null);
  assert.ok(same.matched >= 4);
  assert.equal(same.activationCoverage.ratio, 1);
  const changed = compareDebugSessions(reference, await session(f, debug, Float32Array.of(-1, 2, -3, 4.5)));
  assert.equal(changed.firstMismatch.node, 'relu');
  assert.equal(changed.firstMismatch.point, 'DEBUG_POINT_BEFORE');
  assert.equal(changed.firstMismatch.maxAbsError, .5);
  const output = changed.rows.find(row => row.node === 'double' && row.point === 'DEBUG_POINT_AFTER');
  assert.equal(output.maxAbsError, 1);
});

// A float model and the W8A8 package C PTQ authoring writes for it. Authoring
// keeps node IDs but renames every quantized activation, and adds Q/DQ
// boundary nodes; the comparison must still align each layer's value.
const D = 8, H = 16;
const ramp = (length, scale, offset = 0) => Float32Array.from({length}, (_, i) => Math.sin(i * 1.7 + offset) * scale);
const mlp = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [1, D]}},
  nodes: [
    {id: 'fc1', opType: 'Linear', inputs: {input: 'x', weight: 'fc1.w', bias: 'fc1.b'},
      outputs: {out: {tensor: 'h1', dtype: 'float32', shape: [1, H]}}, params: {weight_layout: 'dout_din'}},
    {id: 'act', opType: 'GELU', inputs: {input: 'h1'},
      outputs: {out: {tensor: 'h2', dtype: 'float32', shape: [1, H]}}, params: {}},
    {id: 'fc2', opType: 'Linear', inputs: {input: 'h2', weight: 'fc2.w', bias: 'fc2.b'},
      outputs: {out: {tensor: 'y', dtype: 'float32', shape: [1, D]}}, params: {weight_layout: 'dout_din'}},
  ], outputs: ['y']};
const mlpWeights = safetensors([
  {name: 'fc1.w', shape: [H, D], data: ramp(H * D, .5)}, {name: 'fc1.b', shape: [H], data: ramp(H, .1, 1)},
  {name: 'fc2.w', shape: [D, H], data: ramp(D * H, .5, 2)}, {name: 'fc2.b', shape: [D], data: ramp(D, .1, 3)}]);

async function preserved(f, model) {
  return ok(await f.inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
    preserveNodeBoundaries: true, policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: ['wasm']})})));
}

test('a float model aligns with its C-authored PTQ package and ranks layers by SQNR', async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const debug = new VxDebugServiceClient(reportTransport(f.host));
  const graphBytes = new TextEncoder().encode(JSON.stringify(mlp));
  const floatModel = await f.load(mlp, mlpWeights);
  const authored = ok(await f.quantization.authorPtqTemplate(new p.AuthorPtqTemplateRequest({
    sourceGraph: graphBytes, weightShards: [mlpWeights], config: new p.PtqAuthoringConfig({
      activationDtype: p.DataType.DATA_TYPE_I8, activationScheme: p.PtqScheme.PTQ_SCHEME_ASYMMETRIC})})));
  // The comparison's alias derivation is the C authoring derivation.
  for (const observer of authored.observers)
    assert.equal(observer.quantizedTensorName, ptqActivationName(observer.tensorName));
  const plan = ok(await f.quantization.createPtqPlan(new p.CreatePtqPlanRequest({modelId: floatModel.modelId,
    templateGraph: authored.templateGraph, observers: authored.observers, layers: authored.layers, profileNames: ['p']})));
  for (let sample = 0; sample < 8; sample++)
    ok(await f.quantization.calibratePtqPlan(new p.CalibratePtqPlanRequest({ptqPlanId: plan.ptqPlanId, profileName: 'p',
      sampleName: `s${sample}`, sampleCount: 1n, inputs: tensors({x: {data: ramp(D, 1, sample), shape: [1, D]}})})));
  const packed = ok(await f.quantization.writePtqPackage(new p.WritePtqPackageRequest({ptqPlanId: plan.ptqPlanId})));
  const quantizedModel = ok(await f.inference.loadModel(new p.LoadModelRequest({runtimeId: f.runtime.runtimeId,
    package: new p.ModelPackage({graphDocument: packed.graph, weightShards: [packed.weights]})})));
  const input = ramp(D, 1, .5);
  f.compiled = await preserved(f, floatModel);
  const reference = await session(f, debug, input, [1, D]);
  f.compiled = await preserved(f, quantizedModel);
  const candidate = await session(f, debug, input, [1, D]);

  const report = compareDebugSessions(reference, candidate);
  assert.equal(report.comparable, true);
  assert.equal(report.mode, 'quantization');
  assert.ok(report.aliases.ptqActivations > 0, 'renamed activations are recognized');
  assert.deepEqual(report.activationCoverage, {matched: report.activationCoverage.total,
    total: report.activationCoverage.total, ratio: 1});
  // Only float weights, which the package replaces with packed storage, stay unmatched.
  assert.ok(report.unmatchedObservations.every(observation => observation.kind === 'constant'));
  for (const tensor of ['h1', 'h2', 'y']) {
    const row = report.rows.find(row => row.tensor === tensor && row.point === 'DEBUG_POINT_AFTER');
    assert.ok(row, `${tensor} is aligned`);
    assert.ok(row.sqnrDb > 25 && row.sqnrDb < 80, `${tensor} SQNR ${row.sqnrDb}`);
    assert.ok(row.cosine > .999 && row.rmseOverScale > 0);
  }
  assert.deepEqual(report.worst.map(row => row.tensor).sort(), ['h1', 'h2', 'y']);
  assert.ok(report.worst[0].sqnrDb <= report.worst.at(-1).sqnrDb);
  assert.equal(report.firstMismatch, null);
  const strict = compareDebugSessions(reference, candidate, {minSqnrDb: 90});
  assert.equal(strict.firstMismatch.tensor, 'x');
  assert.equal(compareDebugSessions(reference, candidate, {mode: 'exact'}).firstMismatch.tensor, 'x');
});

test('sessions without a common observation are reported as not comparable', async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const debug = new VxDebugServiceClient(reportTransport(f.host));
  f.compiled = await f.compile(await f.load(graph));
  const reference = await session(f, debug, Float32Array.of(-1, 2, -3, 4));
  const renamed = structuredClone(reference);
  for (const tensor of renamed.plan.tensors) tensor.name = `other.${tensor.name}`;
  const report = compareDebugSessions(reference, renamed);
  assert.equal(report.comparable, false);
  assert.equal(report.matched, 0);
  assert.equal(report.firstMismatch, null);
  assert.match(report.reason, /no observation aligned/);

  const directory = mkdtempSync(join(tmpdir(), 'debug-report-'));
  t.after(() => rmSync(directory, {recursive: true, force: true}));
  writeFileSync(join(directory, 'a.json'), JSON.stringify(reference));
  writeFileSync(join(directory, 'b.json'), JSON.stringify(renamed));
  const cli = spawnSync(process.execPath, [new URL('../tools/debug_report.mjs', import.meta.url).pathname,
    'compare', join(directory, 'a.json'), join(directory, 'b.json'), join(directory, 'report.json')], {encoding: 'utf8'});
  assert.equal(cli.status, 2, cli.stderr);
  assert.match(cli.stderr, /not comparable/);
});

test('constants: false leaves weights out of the capture', async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const debug = new VxDebugServiceClient(reportTransport(f.host));
  const compiled = await preserved(f, await f.load(mlp, mlpWeights));
  const capture = async constants => {
    let info = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({forward: new p.DebugForward({
      compiledModelId: compiled.compiledModelId, inputs: tensors({x: {data: ramp(D, 1), shape: [1, D]}})}),
      capture: new p.DebugCapture({values: true, constants})})));
    info = ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId,
      expectedRevision: info.revision})));
    const artifact = await collectDebugSession(debug, p, info.debugSessionId);
    return artifact.events.flatMap(event => event.snapshots.map(s => artifact.plan.tensors[s.tensorId ?? 0].name));
  };
  const all = await capture(undefined), activations = await capture(false);
  assert.ok(all.includes('fc1.w') && all.includes('fc2.b'));
  assert.deepEqual(activations.filter(name => name.startsWith('fc')), []);
  assert.deepEqual([...new Set(activations)].sort(), ['h1', 'h2', 'x', 'y']);
});

// Shared by the provenance and isolated-error tests below.
async function floatAndQuantized(f) {
  const graphBytes = new TextEncoder().encode(JSON.stringify(mlp));
  const floatModel = await f.load(mlp, mlpWeights);
  const authored = ok(await f.quantization.authorPtqTemplate(new p.AuthorPtqTemplateRequest({
    sourceGraph: graphBytes, weightShards: [mlpWeights], config: new p.PtqAuthoringConfig({
      activationDtype: p.DataType.DATA_TYPE_I8, activationScheme: p.PtqScheme.PTQ_SCHEME_ASYMMETRIC})})));
  const plan = ok(await f.quantization.createPtqPlan(new p.CreatePtqPlanRequest({modelId: floatModel.modelId,
    templateGraph: authored.templateGraph, observers: authored.observers, layers: authored.layers, profileNames: ['p']})));
  for (let sample = 0; sample < 8; sample++)
    ok(await f.quantization.calibratePtqPlan(new p.CalibratePtqPlanRequest({ptqPlanId: plan.ptqPlanId, profileName: 'p',
      sampleName: `s${sample}`, sampleCount: 1n, inputs: tensors({x: {data: ramp(D, 1, sample), shape: [1, D]}})})));
  const packed = ok(await f.quantization.writePtqPackage(new p.WritePtqPackageRequest({ptqPlanId: plan.ptqPlanId})));
  const quantizedModel = ok(await f.inference.loadModel(new p.LoadModelRequest({runtimeId: f.runtime.runtimeId,
    package: new p.ModelPackage({graphDocument: packed.graph, weightShards: [packed.weights]})})));
  return {floatCompiled: await preserved(f, floatModel), quantizedCompiled: await preserved(f, quantizedModel)};
}

test('a PTQ package names the float tensor behind each renamed activation', async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const debug = new VxDebugServiceClient(reportTransport(f.host));
  const {quantizedCompiled} = await floatAndQuantized(f);
  f.compiled = quantizedCompiled;
  const artifact = await session(f, debug, ramp(D, 1, .5), [1, D]);
  const sources = Object.fromEntries(artifact.plan.tensors.filter(t => t.sourceTensorName)
    .map(t => [t.sourceTensorName, t.name]));
  assert.deepEqual(Object.keys(sources).sort(), ['h1', 'h2', 'x', 'y']);
  for (const [source, name] of Object.entries(sources)) assert.equal(name, ptqActivationName(source));
});

test('SetDebugTensor feeds a layer the float reference to measure its own error', async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const debug = new VxDebugServiceClient(reportTransport(f.host));
  const {floatCompiled, quantizedCompiled} = await floatAndQuantized(f);
  const input = ramp(D, 1, .5);
  f.compiled = floatCompiled;
  const reference = await session(f, debug, input, [1, D]);
  f.compiled = quantizedCompiled;
  const accumulated = compareDebugSessions(reference, await session(f, debug, input, [1, D]));

  let info = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({forward: new p.DebugForward({
    compiledModelId: quantizedCompiled.compiledModelId, inputs: tensors({x: {data: input, shape: [1, D]}})}),
    capture: new p.DebugCapture({values: true})})));
  info = ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision, breakBeforeNodes: ['fc2']})));
  const {plan} = ok(await debug.getDebugPlan(new p.DebugSessionRef(info)));
  const inputId = plan.steps[info.nextStep].inputs.find(id => plan.tensors[id].sourceTensorName === 'h2');
  // The activation's affine, from the snapshot the previous step published.
  const events = ok(await debug.listDebugEvents(new p.ListDebugEventsRequest({debugSessionId: info.debugSessionId}))).events;
  const produced = events.flatMap(e => e.snapshots).find(s => (s.tensorId ?? 0) === inputId);
  const {scale, zeroPoint = 0} = produced.quantization.perTensor;
  const floatH2 = reference.events.flatMap(e => e.snapshots.map(s => ({s, name: reference.plan.tensors[s.tensorId ?? 0].name})))
    .find(({name, s}) => name === 'h2' && s.valuesBase64).s;
  const bytes = Buffer.from(floatH2.valuesBase64, 'base64');
  const h2 = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const q = Int8Array.from(h2, v => Math.max(-128, Math.min(127, Math.round(v / scale) + zeroPoint)));
  const value = new p.Tensor({dtype: p.DataType.DATA_TYPE_I8, shape: [1n, BigInt(H)], inline: new Uint8Array(q.buffer)});

  // Refusals: a stale revision, and a constant.
  const stale = await debug.setDebugTensor(new p.SetDebugTensorRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision - 1n, tensorId: inputId, value}));
  assert.equal(stale.report.status, p.NativeStatus.NATIVE_STATUS_REVISION_CONFLICT);
  const weightId = plan.steps[info.nextStep].inputs.find(id => id !== inputId);
  const constant = await debug.setDebugTensor(new p.SetDebugTensorRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision, tensorId: weightId, value}));
  assert.equal(constant.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);

  info = ok(await debug.setDebugTensor(new p.SetDebugTensorRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision, tensorId: inputId, value})));
  assert.equal(info.modified, true);
  info = ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision})));
  assert.equal(info.state, p.DebugState.DEBUG_STATE_COMPLETED);
  const isolated = compareDebugSessions(reference, await collectDebugSession(debug, p, info.debugSessionId));
  const row = (report, tensor, point) => report.rows.find(r => r.node === 'fc2' && r.tensor === tensor && r.point === point);
  // fc2 saw the float reference, off only by rounding to its input affine,
  // instead of the value its quantized predecessors produced.
  assert.ok(row(isolated, 'h2', 'DEBUG_POINT_BEFORE').maxAbsError <= scale / 2 * (1 + 1e-6));
  assert.ok(row(accumulated, 'h2', 'DEBUG_POINT_BEFORE').maxAbsError > row(isolated, 'h2', 'DEBUG_POINT_BEFORE').maxAbsError);
  // The output now measures fc2's own error; either may be larger, since
  // upstream errors can partly cancel.
  assert.ok(Number.isFinite(row(isolated, 'y', 'DEBUG_POINT_AFTER').sqnrDb));
});
