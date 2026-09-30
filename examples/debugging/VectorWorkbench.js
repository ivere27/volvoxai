/** A second, non-image model using the same workbench with no receipt extensions. */
// Node debugging is a full-profile service.
import {loadRuntimeRelease} from '../common/RuntimeRelease.js';
import {ModelWorkbench} from '../common/workbench/ModelWorkbench.js';
import {EngineDebugSession} from '../common/EngineDebugSession.js';
import {startEngineTrace} from '../common/EngineTrace.js';
import {formatMs} from '../common/workbench/View.js';

const {api, wasmUrl} = await loadRuntimeRelease();
const p = api.pb, host = new api.FullEngineHost({wasmUrl}), inference = new api.VxInferenceServiceClient(host);
const graph = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [4]}},
  nodes: [{id: 'positive', opType: 'ReLU', inputs: {input: 'x'}, outputs: {out: {tensor: 'a', dtype: 'float32', shape: [4]}}, params: {}},
    {id: 'double', opType: 'Add', inputs: {a: 'a', b: 'a'}, outputs: {out: {tensor: 'y', dtype: 'float32', shape: [4]}}, params: {}}], outputs: ['y']};
let runtime, model, compiled, pageBusy = true, debugBusy = false;
const checked = response => { const report = response.report ?? response; if (report.status !== 0) throw new Error(report.message); return response; };
const source = () => ({label: `x = [${values().join(', ')}]`, variant: 'FP32', backend: 'wasm', graph});
const workbench = new ModelWorkbench({root: document.getElementById('workbench'), pb: p, title: 'Vector transform',
  subtitle: 'A minimal model adapter · y = 2 × ReLU(x)', onRecord: () => run(true),
  debug: {getSource: source, createSession: async ({preserveNodeBoundaries, capture, limits}) => {
    const prepared = checked(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId, preserveNodeBoundaries})));
    try { return await EngineDebugSession.create(api, host, {forward: new p.DebugForward({compiledModelId: prepared.compiledModelId, inputs: [input()]}), capture: new p.DebugCapture(capture), limits: new p.DebugLimits(limits)}); }
    finally { checked(await inference.releaseCompiledModel(new p.CompiledModelRef(prepared))); }
  }}, onBusyChange: value => { debugBusy = value; sync(); }});
workbench.controlsRoot.innerHTML = '<span class="badge">WASM · CPU</span><span class="badge">FP32</span><button id="vectorRun" class="primary" disabled>Run inference</button>';
workbench.inputRoot.innerHTML = `<div class="task-intro"><h2>A small graph, a complete workflow</h2><p class="hint">The debugger and profiler are the same components used by the receipt model. This model needs no image or attention view.</p></div>
  <div class="input-layout"><section class="card"><div class="card-heading"><h2>Input tensor x</h2><span class="badge">F32 [4]</span></div><div class="card-body"><label for="vectorInput">Four finite numbers, separated by commas</label><input id="vectorInput" class="input-text" type="text" value="-1, 2, -3, 4"><p class="hint">ReLU clamps negative values to zero; Add doubles each result.</p></div></section>
  <aside class="input-results"><section class="card"><div class="card-heading"><h2>Output tensor y</h2><span id="vectorTime" class="badge">—</span></div><div class="card-body"><pre id="vectorOutput">No inference yet.</pre></div><div class="workflow-links"><button id="vectorDebug">Debug tensors →</button><button id="vectorProfile">Measure performance →</button></div></section><div class="notice">Open Debug and Start session. Step through ReLU and Add, then use Tensor values to inspect their snapshots.</div></aside></div>`;
const inputField = workbench.inputRoot.querySelector('#vectorInput'), runButton = workbench.controlsRoot.querySelector('#vectorRun');
function values() {
  const items = inputField.value.split(',').map(v => v.trim());
  if (items.length !== 4 || items.some(v => !v || !Number.isFinite(Number(v)))) throw new Error('Enter four finite numbers separated by commas.');
  const numbers = Array.from(Float32Array.from(items.map(Number)));
  if (numbers.some(v => !Number.isFinite(v))) throw new Error('Input values must fit finite float32 numbers.');
  return numbers;
}
function input() { return new p.Tensor({name: 'x', dtype: p.DataType.DATA_TYPE_F32, shape: [4n], inline: new Uint8Array(Float32Array.from(values()).buffer)}); }
function sync() {
  inputField.disabled = runButton.disabled = pageBusy || debugBusy || !compiled;
  workbench.debugger.setAvailable(!pageBusy && !!compiled); workbench.profiler.setAvailable(!!compiled); workbench.profiler.setBusy(pageBusy || debugBusy);
}
async function run(profiled = false) {
  if (pageBusy || debugBusy || !compiled) return;
  pageBusy = true; sync(); let capture;
  try {
    workbench.setStatus(profiled ? 'Recording inference…' : 'Running…');
    const tensor = input(); if (profiled) capture = await startEngineTrace(api, host, runtime.runtimeId);
    const started = performance.now();
    const result = checked(await inference.run(new p.RunRequest({compiledModelId: compiled.compiledModelId, inputs: [tensor]})));
    let output;
    try { output = checked(await inference.readOutput(new p.ReadOutputRequest({resultId: result.resultId, name: 'y'}))).tensor; }
    finally { checked(await inference.releaseResult(new p.ResultRef(result))); }
    const elapsed = performance.now() - started;
    workbench.inputRoot.querySelector('#vectorOutput').textContent = JSON.stringify([...new Float32Array(output.inline.slice().buffer)]);
    workbench.inputRoot.querySelector('#vectorTime').textContent = formatMs(elapsed);
    if (capture) workbench.profiler.setTrace({trace: await capture.finish(), source: source(), wallMs: elapsed});
    workbench.setStatus('Ready');
  } catch (error) { workbench.setStatus(error.message, {error: true}); if (profiled) workbench.profiler.error(error.message); }
  finally { await capture?.release(); pageBusy = false; sync(); }
}
runButton.onclick = () => run();
workbench.inputRoot.querySelector('#vectorDebug').onclick = () => workbench.show('debug');
workbench.inputRoot.querySelector('#vectorProfile').onclick = () => workbench.show('profile');
inputField.onchange = async () => {
  pageBusy = true; sync();
  try {
    await workbench.debugger.reset(); workbench.profiler.reset();
    workbench.inputRoot.querySelector('#vectorOutput').textContent = 'Input changed · run again.';
    values(); workbench.setStatus('Ready');
  } catch (error) { workbench.setStatus(error.message, {error: true}); }
  finally { pageBusy = false; sync(); }
};
async function init() {
  try {
    runtime = checked(await inference.createRuntime(new p.CreateRuntimeRequest()));
    model = checked(await inference.loadModel(new p.LoadModelRequest({...runtime, package: new p.ModelPackage({graphDocument: new TextEncoder().encode(JSON.stringify(graph))})})));
    compiled = checked(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId})));
    workbench.setStatus('Ready');
  } catch (error) { workbench.setStatus(error.message, {error: true}); }
  finally { pageBusy = false; sync(); }
  await run();
}
addEventListener('pagehide', event => { if (!event.persisted) void host.close(); });
void init();
