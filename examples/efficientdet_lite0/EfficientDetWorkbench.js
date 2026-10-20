import {loadRuntimeRelease} from '../common/RuntimeRelease.js';
import {ModelWorkbench} from '../common/workbench/ModelWorkbench.js';
import {formatMs} from '../common/workbench/View.js';
import {EfficientDetSession} from './EfficientDetSession.js';
import {decodeDetections, classColor} from './Detection.js';

const {api, wasmUrl} = await loadRuntimeRelease();
const samples = {
  cat: {label: 'Domestic cat', description: 'Cat sample', file: 'cat.jpg'},
  dog: {label: 'White dog', description: 'Dog sample', file: 'dog.jpg'},
  street: {label: 'Street scene', description: 'Street scene · cars, bicycles and people', file: 'street.jpg'},
  fruits: {label: 'Kitchen counter', description: 'Kitchen counter · fruit and household objects', file: 'fruits.jpg'},
};
const params = new URLSearchParams(location.search);
let session = null, graph = null, labels = [], pageBusy = true, debugBusy = false, hasImage = false;
let live = false, loopPromise = null, closed = false, detections = [], raw = null, hovered = -1;
const original = document.createElement('canvas'), originalContext = original.getContext('2d', {willReadFrequently: true});
const input = document.createElement('canvas'); input.width = input.height = 320;
const inputContext = input.getContext('2d', {willReadFrequently: true});
const source = () => ({label: ui.sceneDescription.textContent, backend: session?.backend ?? ui.backend.value,
  variant: ui.precision.value, graph, details: {phase: 'forward', imageWidth: original.width,
    imageHeight: original.height, input: session?.input, outputs: ['scores', 'boxes'],
    scoreThreshold: Number(ui.scoreThreshold.value), nmsThreshold: Number(ui.nmsThreshold.value),
    maxDetections: Number(ui.maxDets.value)}});
const workbench = new ModelWorkbench({root: document.getElementById('workbench'), pb: api.pb,
  title: 'EfficientDet Lite0', subtitle: 'Detect objects. Inspect tensors. Measure execution and memory.',
  debug: {getSource: source, createSession: options => session.createDebugSession(currentPixels(), options)},
  onRecord: () => run(true), onBusyChange: busy => {debugBusy = busy; syncControls();}});
const stylesheet = document.createElement('link'); stylesheet.rel = 'stylesheet';
stylesheet.href = new URL('./EfficientDetWorkbench.css', import.meta.url); workbench.inputRoot.getRootNode().append(stylesheet);

workbench.controlsRoot.innerHTML = `<label>Backend <select id="backend"><option value="wasm">WASM · CPU</option><option value="webgpu">WebGPU</option></select></label><label>Model <select id="precision"><option value="int8">INT8</option><option value="fp16">FP16</option><option value="fp32">FP32</option></select></label><button id="run" class="primary" disabled>Run detection</button>`;
workbench.inputRoot.innerHTML = `
  <div class="detector-layout">
    <section class="detector-image card">
      <div class="card-heading"><h2>Image &amp; detections</h2><span id="canvasDim" class="badge">—</span></div>
      <div class="input-toolbar"><div class="sample-options">${Object.entries(samples).map(([key, sample]) => `<button class="sample-chip" data-sample="${key}" aria-pressed="false">${sample.label}</button>`).join('')}</div>
        <div class="image-actions"><label class="upload-label">Upload <input id="file" type="file" accept="image/png,image/jpeg" aria-label="Upload detection image"></label><button id="camera">Camera</button><button id="stop" disabled>Stop &amp; keep frame</button></div></div>
      <div class="detector-viewport"><canvas id="canvas" width="640" height="480" aria-label="Input image with detected bounding boxes"></canvas><video id="video" autoplay playsinline muted hidden></video></div>
      <div class="input-caption"><span id="sceneDescription">Loading cat sample…</span><span>RGB · stretched to 320 × 320 for inference</span></div>
    </section>
    <aside class="detector-results">
      <section class="card detections-card"><div class="card-heading"><h2>Detected objects</h2><span id="summaryBadge" class="badge">No result</span></div>
        <div class="detector-policy"><label>Score ≥ <output id="scoreThresholdVal">0.20</output><input id="scoreThreshold" type="range" min="0.05" max="0.95" step="0.05" value="0.20"></label>
          <label>NMS IoU <output id="nmsThresholdVal">0.45</output><input id="nmsThreshold" type="range" min="0.10" max="0.90" step="0.05" value="0.45"></label>
          <label>Max objects <select id="maxDets"><option>6</option><option selected>12</option><option>25</option><option>50</option></select></label>
          <label>Class filter <input id="classFilter" type="search" placeholder="e.g. cat, person"></label></div>
        <div class="detections-table"><table><thead><tr><th>#</th><th>Score</th><th>Class</th><th>ID</th><th>Box · xyxy</th></tr></thead><tbody id="rows"></tbody></table></div>
        <p id="detectionNote" class="hint detection-note">Hover a result to highlight its box. Coordinates are normalized to the source image.</p>
        <div class="workflow-links"><button id="openDebug">Debug tensors →</button><button id="openProfile">Measure performance →</button></div>
      </section>
      <div class="metric-grid detector-metrics"><div class="metric"><span>Last forward</span><strong id="metric-time">—</strong><small id="timingKind">Ordinary detection · one run</small></div><div class="metric"><span>Load &amp; compile</span><strong id="metricCompile">—</strong><small>Model preparation wall time</small></div><div class="metric"><span>Preprocess</span><strong id="metric-preprocess">—</strong><small>Resize + input conversion</small></div><div class="metric"><span>Postprocess</span><strong id="metric-postprocess">—</strong><small>Anchor decode + class-aware NMS</small></div></div>
      <p id="cameraHint" class="notice">Debug runs the current image independently. Performance records an ordinary forward pass. Stop the camera to capture a stable frame.</p>
    </aside>
    <section class="card detector-context"><div class="card-heading"><h2>Model context</h2><span id="modelFormat" class="badge">INT8</span></div><div class="card-body"><dl class="model-properties">
      <dt>Input</dt><dd id="inputShape" class="mono">—</dd><dt>Outputs</dt><dd class="mono">scores [1, 19206, 90] · boxes [1, 19206, 4]</dd><dt>Compiled route</dt><dd id="routeInfo">—</dd><dt>Package</dt><dd id="pkgPath" class="mono">—</dd><dt>Last throughput</dt><dd id="metric-fps">—</dd>
      <dt>Selected runtime</dt><dd><span id="metric-backend">—</span> · <span id="metric-precision">—</span></dd></dl><p id="routeNote" class="hint" hidden></p></div></section>
  </div><pre id="out" hidden></pre>`;
const ids = ['backend', 'precision', 'run', 'file', 'camera', 'stop', 'canvas', 'video', 'canvasDim', 'sceneDescription',
  'summaryBadge', 'scoreThreshold', 'scoreThresholdVal', 'nmsThreshold', 'nmsThresholdVal', 'maxDets', 'classFilter', 'rows',
  'detectionNote', 'openDebug', 'openProfile', 'metric-time', 'metricCompile', 'metric-preprocess', 'metric-postprocess',
  'timingKind', 'modelFormat', 'inputShape', 'routeInfo', 'pkgPath', 'metric-fps', 'metric-backend', 'metric-precision', 'routeNote', 'out'];
const ui = Object.fromEntries(ids.map(id => [id, workbench.inputRoot.querySelector(`#${id}`) ?? workbench.controlsRoot.querySelector(`#${id}`)]));
const context = ui.canvas.getContext('2d', {willReadFrequently: true}), sampleButtons = [...workbench.inputRoot.querySelectorAll('[data-sample]')];
ui.backend.value = ['wasm', 'webgpu'].includes(params.get('backend')) ? params.get('backend') : 'wasm';
ui.precision.value = ['int8', 'fp16', 'fp32'].includes(params.get('precision')) ? params.get('precision') : 'int8';
if (params.has('thresh') && Number.isFinite(Number(params.get('thresh')))) ui.scoreThreshold.value = params.get('thresh');
if (!navigator.gpu) ui.backend.querySelector('[value="webgpu"]').disabled = true;
if (!navigator.mediaDevices?.getUserMedia) ui.camera.title = 'Camera access requires a secure browser context';

function syncControls() {
  const busy = pageBusy || debugBusy || closed, locked = busy || live;
  for (const control of [ui.backend, ui.precision, ui.file, ...sampleButtons]) control.disabled = locked;
  ui.run.disabled = locked || !session || !hasImage; ui.camera.disabled = locked || !session || !navigator.mediaDevices?.getUserMedia;
  ui.stop.disabled = !live; ui.openDebug.disabled = ui.openProfile.disabled = locked || !session || !hasImage;
  workbench.debugger.setAvailable(!pageBusy && !live && !closed && !!session && hasImage);
  workbench.profiler.setAvailable(!live && !closed && !!session && hasImage); workbench.profiler.setBusy(busy || live);
}

function currentPixels() {
  if (!hasImage) throw new Error('Select an image first');
  inputContext.drawImage(original, 0, 0, 320, 320);
  return inputContext.getImageData(0, 0, 320, 320).data;
}

function drawSource() {
  if (!hasImage) return;
  const width = Math.min(960, Math.max(320, original.width)), height = Math.round(width * original.height / original.width);
  if (ui.canvas.width !== width || ui.canvas.height !== height) {ui.canvas.width = width; ui.canvas.height = height;}
  context.drawImage(original, 0, 0, width, height); ui.canvasDim.textContent = `${original.width} × ${original.height} px`;
  for (const d of filteredDetections()) {
    const x = d.xmin * width, y = d.ymin * height, w = (d.xmax - d.xmin) * width, h = (d.ymax - d.ymin) * height;
    if (w < 2 || h < 2) continue;
    const color = classColor(d.cls), highlighted = hovered === d.index;
    context.save(); context.lineWidth = highlighted ? 4 : 2; context.strokeStyle = color;
    if (highlighted) {context.fillStyle = `${color}22`; context.fillRect(x, y, w, h);}
    context.strokeRect(x, y, w, h);
    const text = `${d.label} ${(d.score * 100).toFixed(0)}%`, tagHeight = Math.max(20, Math.round(width / 30));
    context.font = `600 ${Math.max(12, Math.round(width / 44))}px system-ui, sans-serif`;
    const tagWidth = context.measureText(text).width + 10, tagY = Math.max(0, y - tagHeight);
    context.fillStyle = color; context.fillRect(x, tagY, tagWidth, tagHeight);
    context.fillStyle = '#fff'; context.fillText(text, x + 5, tagY + tagHeight - 5); context.restore();
  }
}

function filteredDetections() {
  const filter = ui.classFilter.value.trim().toLowerCase();
  return filter ? detections.filter(d => d.label.toLowerCase().includes(filter)) : detections;
}

function renderRows() {
  const filtered = filteredDetections(); ui.rows.replaceChildren();
  ui.summaryBadge.textContent = `${filtered.length} object${filtered.length === 1 ? '' : 's'}`;
  for (const [index, d] of filtered.entries()) {
    const row = document.createElement('tr'), box = [d.xmin, d.ymin, d.xmax, d.ymax].map(value => value.toFixed(3)).join(', ');
    for (const value of [index + 1, d.score.toFixed(3), d.label, d.cls, box]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    row.children[2].style.color = classColor(d.cls); row.lastElementChild.className = 'mono box-coordinates';
    row.lastElementChild.title = box; row.onmouseenter = () => {hovered = d.index; drawSource();};
    row.onmouseleave = () => {hovered = -1; drawSource();}; ui.rows.append(row);
  }
  const below = detections.length && detections.every(d => d.score < Number(ui.scoreThreshold.value));
  ui.detectionNote.textContent = below ? 'No scores meet the threshold. Showing the highest-scoring candidates, as in the original detector.' :
    'Hover a result to highlight its box. Coordinates are normalized to the source image.';
}

function postprocess() {
  if (!raw) return;
  const started = performance.now();
  detections = decodeDetections(raw.scores, raw.boxes, labels, {scoreThreshold: Number(ui.scoreThreshold.value),
    nmsThreshold: Number(ui.nmsThreshold.value), maxDetections: Number(ui.maxDets.value)});
  ui['metric-postprocess'].textContent = formatMs(performance.now() - started); hovered = -1; renderRows(); drawSource();
  ui.out.textContent = `DONE backend=${session.backend} ms=${raw.executionMs.toFixed(2)} rows=${detections.length}\n` +
    detections.map((d, i) => `${i + 1}\t${d.index}\t${d.score.toFixed(6)}\t${d.cls}\t${d.label}`).join('\n');
}

function clearResult() {
  raw = null; detections = []; hovered = -1; ui.rows.replaceChildren(); ui.summaryBadge.textContent = 'No result'; ui.out.textContent = '';
  for (const id of ['metric-time', 'metric-preprocess', 'metric-postprocess', 'metric-fps']) ui[id].textContent = '—';
  drawSource();
}

async function openModel() {
  await workbench.debugger.reset({clearCaptureSelection: true}); workbench.profiler.reset(); clearResult();
  await session?.close(); session = null; graph = null;
  const pkg = new URL(`../../models/efficientdet_lite0_${ui.precision.value}/`, import.meta.url);
  ui.pkgPath.textContent = `models/efficientdet_lite0_${ui.precision.value}`;
  const [graphResponse, labelResponse] = await Promise.all([fetch(new URL('graph.json', pkg)), fetch(new URL('labels.txt', pkg))]);
  if (!graphResponse.ok || !labelResponse.ok) throw new Error(`Model package unavailable: ${ui.pkgPath.textContent}`);
  graph = await graphResponse.json(); labels = (await labelResponse.text()).split(/\r?\n/).map(label => label.trim());
  const started = performance.now();
  const opened = await EfficientDetSession.open({api, wasmUrl, backend: ui.backend.value,
    graphUrl: new URL('graph.json', pkg), weightsUrl: new URL('model.safetensors', pkg)});
  if (closed) {await opened.close(); throw new Error('Workbench closed while loading the model');}
  session = opened;
  ui.metricCompile.textContent = formatMs(performance.now() - started); ui.backend.value = session.backend;
  ui.modelFormat.textContent = ui.precision.value.toUpperCase(); ui['metric-precision'].textContent = ui.precision.value.toUpperCase();
  ui['metric-backend'].textContent = session.backend; ui.inputShape.textContent = `input0 [${session.input.shape.join(', ')}] · ${session.input.dtype}`;
  const report = session.compileReport;
  ui.routeInfo.textContent = `${session.backend} · ${report.route?.activeNodes ?? '?'} active / ${report.route?.selectedNodes ?? '?'} selected nodes · ${report.route?.attested ? 'attested' : 'unattested'} · ${report.fallback?.operatorFallbackUsed ? 'operator fallback used' : 'no operator fallback'}`;
  ui.routeNote.hidden = !session.fallbackReason; ui.routeNote.textContent = session.fallbackReason;
}

async function setImage(blob, label) {
  const bitmap = await createImageBitmap(blob);
  try {
    if (closed) return;
    original.width = bitmap.width; original.height = bitmap.height; originalContext.drawImage(bitmap, 0, 0);
  }
  finally {bitmap.close();}
  hasImage = true; ui.sceneDescription.textContent = label; drawSource();
}

async function loadSample(key) {
  const sample = samples[key], response = await fetch(new URL(`./assets/${sample.file}`, import.meta.url));
  if (!response.ok) throw new Error(`Sample image unavailable: ${sample.label}`);
  await setImage(await response.blob(), sample.description);
  for (const button of sampleButtons) button.setAttribute('aria-pressed', String(button.dataset.sample === key));
}

async function run(profiled = false, cameraFrame = false) {
  if (pageBusy || debugBusy || !session || !hasImage || closed || (live && !cameraFrame)) return;
  pageBusy = true; syncControls(); workbench.setStatus(profiled ? 'Recording detection…' : live ? 'Live detection…' : 'Running detection…');
  try {
    const started = performance.now(), resizedAt = performance.now(), pixels = currentPixels();
    const resizeMs = performance.now() - resizedAt;
    if (profiled) await workbench.debugger.reset();
    raw = profiled ? await session.detectProfiled(pixels) : await session.detect(pixels);
    raw.preprocessMs += resizeMs; postprocess();
    ui['metric-time'].textContent = formatMs(raw.executionMs); ui['metric-preprocess'].textContent = formatMs(raw.preprocessMs);
    ui['metric-fps'].textContent = `${(1000 / Math.max(1, performance.now() - started)).toFixed(1)} fps · one run`;
    ui.timingKind.textContent = profiled ? 'Instrumented forward · one run' : 'Ordinary forward · one run';
    if (profiled) {
      const {label, backend, variant, details} = source();
      workbench.profiler.setTrace({trace: raw.trace, wallMs: raw.wallMs, source: {label, backend, variant, details,
        phaseRecords: [{phase: 'forward', contextId: raw.contextId, modelId: raw.modelId,
          executionId: raw.executionId, compiledModelId: raw.compiledModelId, resultId: raw.resultId}]}});
    }
    workbench.setStatus(live ? `Live · ${raw.executionMs.toFixed(0)} ms forward` : 'Ready');
  } catch (error) {
    workbench.setStatus(error.message, {error: true}); ui.out.textContent = `FATAL ${error.message}`;
    if (profiled) workbench.profiler.error(error.message);
    if (live) {live = false; stopTracks();}
  } finally {pageBusy = false; syncControls();}
}

async function changeInput(action, status) {
  if (pageBusy || debugBusy || live || closed) return;
  pageBusy = true; syncControls(); workbench.setStatus(status); let changed = false;
  try {await workbench.debugger.reset(); workbench.profiler.reset(); clearResult(); await action(); changed = true; workbench.setStatus('Ready');}
  catch (error) {workbench.setStatus(error.message, {error: true});}
  finally {pageBusy = false; syncControls();}
  if (changed && session && hasImage) await run();
}

function stopTracks() {
  for (const track of ui.video.srcObject?.getTracks() ?? []) track.stop();
  ui.video.srcObject = null;
}

function cameraImage() {
  original.width = ui.video.videoWidth; original.height = ui.video.videoHeight;
  originalContext.drawImage(ui.video, 0, 0); hasImage = true; drawSource();
}

async function startCamera() {
  if (pageBusy || debugBusy || live || closed) return;
  pageBusy = true; syncControls(); workbench.setStatus('Starting camera…');
  try {
    await workbench.debugger.reset(); workbench.profiler.reset(); clearResult();
    ui.video.srcObject = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'environment'}, audio: false});
    await ui.video.play(); cameraImage(); ui.sceneDescription.textContent = 'Camera frame';
    for (const button of sampleButtons) button.setAttribute('aria-pressed', 'false'); live = true;
  } catch (error) {stopTracks(); workbench.setStatus(error.message, {error: true});}
  finally {pageBusy = false; syncControls();}
  if (live) loopPromise = (async () => {
    while (live && !closed) {
      if (ui.video.readyState >= 2) {cameraImage(); await run(false, true);}
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  })();
}

async function stopCamera() {
  live = false; stopTracks(); syncControls(); await loopPromise; loopPromise = null;
  ui.sceneDescription.textContent = 'Captured camera frame'; workbench.setStatus('Frame ready'); syncControls();
}

ui.run.onclick = () => run(); ui.backend.onchange = ui.precision.onchange = () => changeInput(openModel, 'Loading & compiling…');
ui.openDebug.onclick = () => workbench.show('debug'); ui.openProfile.onclick = () => workbench.show('profile');
ui.camera.onclick = () => startCamera(); ui.stop.onclick = () => stopCamera();
for (const button of sampleButtons) button.onclick = () => changeInput(() => loadSample(button.dataset.sample), 'Loading sample…');
ui.file.onchange = () => {
  const file = ui.file.files?.[0]; if (!file) return;
  void changeInput(async () => {await setImage(file, file.name); for (const button of sampleButtons) button.setAttribute('aria-pressed', 'false');}, 'Loading image…');
};
for (const [slider, value] of [[ui.scoreThreshold, ui.scoreThresholdVal], [ui.nmsThreshold, ui.nmsThresholdVal]]) {
  slider.oninput = () => {value.textContent = Number(slider.value).toFixed(2); postprocess();};
  value.textContent = Number(slider.value).toFixed(2);
}
ui.maxDets.onchange = postprocess; ui.classFilter.oninput = () => {renderRows(); drawSource();};

async function init() {
  syncControls();
  try {
    await openModel();
    if (params.get('image')) {
      const response = await fetch(new URL(params.get('image'), location.href));
      if (!response.ok) throw new Error(`Image unavailable: ${params.get('image')}`);
      await setImage(await response.blob(), 'Image URL');
    } else if (params.get('autorun') === 'zero') {
      original.width = original.height = 320; originalContext.fillStyle = '#000'; originalContext.fillRect(0, 0, 320, 320);
      hasImage = true; ui.sceneDescription.textContent = 'Zero image'; drawSource();
    } else await loadSample('cat');
    workbench.setStatus('Ready');
  } catch (error) {workbench.setStatus(error.message, {error: true}); ui.out.textContent = `FATAL ${error.message}`;}
  finally {pageBusy = false; syncControls();}
  await run();
}
addEventListener('pagehide', event => {
  if (event.persisted) return;
  closed = true; live = false; stopTracks();
  void (async () => {await loopPromise; await workbench.dispose(); await session?.close();})().catch(console.error);
});
void init();
