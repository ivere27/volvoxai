import {loadRuntimeRelease} from '../common/RuntimeRelease.js';
import {ModelWorkbench} from '../common/workbench/ModelWorkbench.js';
import {formatMs} from '../common/workbench/View.js';
import {ReceiptDigitSession} from './ReceiptDigitSession.js';
import {ReceiptSpatialView} from './ReceiptSpatialView.js';
import {answerFromRecord} from './questionRouter.js';

const {api: volvoxai, wasmUrl} = await loadRuntimeRelease();

const packages = {fp32: new URL('../../build/receipt-digit-reader-fp32/', import.meta.url), int8: new URL('../../build/receipt-digit-reader-int8/', import.meta.url)};
const samples = {
  receipt_ko: {label: 'Korean receipt', phone: '5008936', street: '699', question: '가게 전화번호의 뒤에서 2번째 숫자는 무엇입니까?'},
  receipt_en: {label: 'English receipt', phone: '4234929', street: '732', question: "What is the first number of the store's phone number?"},
};
let session = null, graph = null, manifest = null, pageBusy = true, debugBusy = false, hasImage = false, sampleKey = 'receipt_ko';
const source = () => ({label: ui.sceneDescription.textContent, backend: session?.backend ?? ui.backend.value, variant: ui.variant.value,
  graph, manifest, getImage: currentImage});
const workbench = new ModelWorkbench({root: document.getElementById('workbench'), pb: volvoxai.pb,
  title: 'Receipt digit reader', subtitle: 'Inspect model behavior. Trace execution. Measure resources.',
  debug: {getSource: source, createSession: options => session.createDebugSession(currentImage(), options), tensorViews: [{
    id: 'spatial', label: 'Spatial / image', create: ({capture, select, ...options}) => new ReceiptSpatialView({...options,
      captureAttention: entries => capture({tensorNames: entries.map(e => e.name)}),
      selectAttention: entry => entry && select(entry.step, entry.tensorId)}),
  }]}, onRecord: () => run(true), onBusyChange: busy => { debugBusy = busy; syncControls(); }});

workbench.controlsRoot.innerHTML = `<label>Backend <select id="backend"><option value="wasm">WASM · CPU</option><option value="webgpu">WebGPU</option></select></label><label>Model <select id="variant"><option value="fp32">FP32</option><option value="int8">INT8 · mixed</option></select></label><button id="run" class="primary" disabled>Run inference</button>`;
workbench.inputRoot.innerHTML = `
  <div class="task-intro"><h2>Input &amp; model output</h2><p class="hint">Run a receipt, then inspect tensors or record a profile using the same model and input.</p></div>
  <div class="input-layout">
    <section class="input-main">
      <div class="card"><div class="card-heading"><h2>Input image</h2><span id="canvasDim" class="badge">672 × 320</span></div>
        <div class="input-toolbar"><button class="sample-chip" data-sample="receipt_ko" aria-pressed="true">Korean sample</button><button class="sample-chip" data-sample="receipt_en" aria-pressed="false">English sample</button><label class="hint">Upload image <input id="file" type="file" accept="image/png,image/jpeg" aria-label="Upload receipt image"></label></div>
        <div class="input-viewport"><canvas id="canvas" width="672" height="320" aria-label="Receipt model input"></canvas></div>
        <div class="input-caption"><span id="sceneDescription">Korean receipt</span><span>Grayscale · normalized to [−1, 1]</span></div>
      </div>
      <div class="card"><div class="card-heading"><h2>Model context</h2><span id="modelFormat" class="badge">FP32</span></div><div class="card-body"><dl class="model-properties">
        <dt>Input tensor</dt><dd id="inputShape" class="mono">—</dd><dt>Output tensor</dt><dd class="mono">slot_logits [1, 16, 11]</dd><dt>Decode</dt><dd>12 phone slots + 4 street slots · 10 digits + blank</dd>
        <dt>Compiled route</dt><dd id="routeInfo">—</dd><dt>Package</dt><dd id="pkgPath" class="mono">—</dd>
      </dl></div></div>
    </section>
    <aside class="input-results">
      <div class="card"><div class="card-heading"><h2>Decoded result</h2><span id="recordBadge" class="badge">No result</span></div><div class="result-fields"><div class="result-field"><span>Phone number</span><strong id="phone">—</strong></div><div class="result-field"><span>Street number</span><strong id="street">—</strong></div></div>
        <p id="expectedRecord" class="hint card-body">Sample reference: phone 5008936 · street 699</p><div class="workflow-links"><button id="openDebug">Debug tensors →</button><button id="openProfile">Measure performance →</button></div>
      </div>
      <div class="metric-grid"><div class="metric"><span>Last inference</span><strong id="metricTime">—</strong><small id="timingKind">End-to-end · one run</small></div><div class="metric"><span>Load &amp; compile</span><strong id="metricCompile">—</strong><small>Model preparation wall time</small></div></div>
      <div class="notice">Debug inspects tensor values. Performance records timings and memory. Both capture only when you request them.</div>
      <details class="card host-question"><summary>Host-side question demo</summary><div><p class="hint">A text rule uses the decoded numbers; the model receives only the image.</p><label for="question">Question</label><input id="question" type="text"><div class="row"><button id="askPhone">Last phone digit</button><button id="askStreet">Street number</button></div><strong id="answerValue" class="host-answer">—</strong></div></details>
    </aside>
  </div>`;
const ui = Object.fromEntries(['backend', 'variant', 'run', 'canvas', 'file', 'sceneDescription', 'canvasDim', 'modelFormat', 'inputShape', 'routeInfo', 'pkgPath',
  'recordBadge', 'phone', 'street', 'expectedRecord', 'openDebug', 'openProfile', 'metricTime', 'metricCompile', 'timingKind', 'question', 'askPhone', 'askStreet', 'answerValue']
  .map(id => [id, workbench.inputRoot.querySelector(`#${id}`) ?? workbench.controlsRoot.querySelector(`#${id}`)]));
const context = ui.canvas.getContext('2d', {willReadFrequently: true});
const sampleButtons = workbench.inputRoot.querySelectorAll('[data-sample]');
if (!navigator.gpu) { ui.backend.querySelector('[value="webgpu"]').disabled = true; }

function syncControls() {
  const busy = pageBusy || debugBusy;
  for (const control of [ui.backend, ui.variant, ui.file, ...sampleButtons]) control.disabled = busy;
  ui.run.disabled = busy || !session || !hasImage;
  workbench.debugger.setAvailable(!pageBusy && !!session && hasImage);
  workbench.profiler.setAvailable(!!session && hasImage); workbench.profiler.setBusy(busy);
}
function currentImage() {
  const frame = context.getImageData(0, 0, ui.canvas.width, ui.canvas.height);
  return {data: frame.data, width: frame.width, height: frame.height, channels: 4};
}
function clearResult() {
  ui.phone.textContent = ui.street.textContent = ui.answerValue.textContent = ui.metricTime.textContent = '—'; ui.recordBadge.textContent = 'No result';
}
function updateAnswer() { ui.answerValue.textContent = answerFromRecord(ui.question.value, ui.phone.textContent, ui.street.textContent) || '—'; }
function showResult(record, elapsed, profiled) {
  ui.phone.textContent = record.phone || '—'; ui.street.textContent = record.street || '—';
  ui.metricTime.textContent = formatMs(elapsed); ui.timingKind.textContent = profiled ? 'Instrumented inference · one run' : 'End-to-end · one run';
  const reference = samples[sampleKey];
  ui.recordBadge.textContent = reference ? record.phone === reference.phone && record.street === reference.street ? 'Matches sample' : 'Differs from sample' : 'Decoded';
  updateAnswer();
}

async function openModel() {
  await workbench.debugger.reset({clearCaptureSelection: true}); workbench.profiler.reset(); clearResult();
  await session?.close(); session = null; graph = null; manifest = null;
  const pkg = packages[ui.variant.value]; ui.pkgPath.textContent = `build/receipt-digit-reader-${ui.variant.value}`;
  const [manifestResponse, graphResponse] = await Promise.all([fetch(new URL('manifest.json', pkg)), fetch(new URL('graph.json', pkg))]);
  if (!manifestResponse.ok || !graphResponse.ok) throw new Error(`Model package unavailable: ${ui.pkgPath.textContent}`);
  manifest = await manifestResponse.json(); graph = await graphResponse.json();
  const started = performance.now();
  session = await ReceiptDigitSession.open({manifest, graphUrl: new URL('graph.json', pkg).href, weightsUrl: new URL('model.safetensors', pkg).href,
    backend: ui.backend.value, wasmUrl, api: volvoxai});
  ui.metricCompile.textContent = formatMs(performance.now() - started); ui.modelFormat.textContent = ui.variant.value.toUpperCase();
  ui.inputShape.textContent = `${manifest.abi.input.name} [${manifest.abi.input.shape.join(', ')}] · F32`;
  const report = session.compileReport;
  ui.routeInfo.textContent = `${session.backend} · ${report.route?.activeNodes ?? '?'} active / ${report.route?.selectedNodes ?? '?'} selected nodes · ${report.route?.attested ? 'attested' : 'unattested'} · ${report.fallback?.operatorFallbackUsed ? 'fallback used' : 'no operator fallback'}`;
}

async function setImage(blob, label) {
  const bitmap = await createImageBitmap(blob);
  try {
    ui.canvas.width = manifest.preprocess.width; ui.canvas.height = manifest.preprocess.height;
    context.drawImage(bitmap, 0, 0, ui.canvas.width, ui.canvas.height); hasImage = true;
  } finally { bitmap.close(); }
  ui.sceneDescription.textContent = label; ui.canvasDim.textContent = `${ui.canvas.width} × ${ui.canvas.height}`;
}
async function loadSample(key) {
  const response = await fetch(new URL(`./benchmarks/work/source/examples/${key}.jpg`, import.meta.url));
  if (!response.ok) throw new Error(`Sample image unavailable: ${key}`);
  await setImage(await response.blob(), samples[key].label); sampleKey = key;
  for (const button of sampleButtons) button.setAttribute('aria-pressed', String(button.dataset.sample === key));
  ui.question.value = samples[key].question;
  ui.expectedRecord.textContent = `Sample reference: phone ${samples[key].phone} · street ${samples[key].street}`;
}

async function run(profiled = false) {
  if (pageBusy || debugBusy || !session || !hasImage) return;
  pageBusy = true; syncControls(); workbench.setStatus(profiled ? 'Recording inference…' : 'Running inference…');
  try {
    const image = currentImage(), started = performance.now();
    if (profiled) {
      const measured = await session.readProfiled(image); showResult(measured.record, measured.wallMs, true);
      const {label, backend, variant} = source(); workbench.profiler.setTrace({trace: measured.trace, wallMs: measured.wallMs, source: {label, backend, variant}});
    } else showResult(await session.read(image), performance.now() - started, false);
    workbench.setStatus('Ready');
  } catch (error) { workbench.setStatus(error.message, {error: true}); if (profiled) workbench.profiler.error(error.message); }
  finally { pageBusy = false; syncControls(); }
}

async function changeInput(action, status) {
  if (pageBusy || debugBusy) return;
  pageBusy = true; syncControls(); workbench.setStatus(status); let changed = false;
  try {
    await workbench.debugger.reset(); workbench.profiler.reset(); clearResult(); await action(); changed = true; workbench.setStatus('Ready');
  } catch (error) { workbench.setStatus(error.message, {error: true}); }
  finally { pageBusy = false; syncControls(); }
  if (changed && session && hasImage) await run();
}
ui.backend.onchange = ui.variant.onchange = () => changeInput(openModel, 'Loading & compiling…');
ui.run.onclick = () => run(); ui.openDebug.onclick = () => workbench.show('debug'); ui.openProfile.onclick = () => workbench.show('profile');
ui.question.oninput = updateAnswer;
ui.askPhone.onclick = () => { ui.question.value = '가게 전화번호의 뒤에서 1번째 숫자는 무엇입니까?'; updateAnswer(); };
ui.askStreet.onclick = () => { ui.question.value = 'What is the street number in the store address?'; updateAnswer(); };
for (const button of sampleButtons) button.onclick = () => changeInput(() => loadSample(button.dataset.sample), 'Loading sample…');
ui.file.onchange = () => {
  const file = ui.file.files?.[0]; if (!file) return;
  void changeInput(async () => {
    await setImage(file, file.name); sampleKey = null;
    for (const button of sampleButtons) button.setAttribute('aria-pressed', 'false');
    ui.expectedRecord.textContent = 'Custom input · no reference labels';
  }, 'Loading image…');
};

async function init() {
  syncControls();
  try { await openModel(); await loadSample('receipt_ko'); workbench.setStatus('Ready'); }
  catch (error) { workbench.setStatus(error.message, {error: true}); }
  finally { pageBusy = false; syncControls(); }
  await run();
}
addEventListener('pagehide', event => { if (!event.persisted) void session?.close(); });
void init();
