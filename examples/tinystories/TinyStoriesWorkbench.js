import {loadRuntimeRelease} from '../common/RuntimeRelease.js';
import {ModelWorkbench} from '../common/workbench/ModelWorkbench.js';
import {DecodeStateView} from '../common/workbench/DecodeStateView.js';
import {download, element, formatMs, jsonBlob} from '../common/workbench/View.js';
import {Tokenizer} from './Tokenizer.js';
import {TinyStoriesSession} from './TinyStoriesSession.js';

const {api, wasmUrl} = await loadRuntimeRelease();
const query = new URLSearchParams(location.search);
const packageUrl = new URL(query.get('model') || '../../models/tinystories_1m/', import.meta.url);
if (!packageUrl.pathname.endsWith('/')) packageUrl.pathname += '/';
let session = null, prepared = null, lastRun = null, selectedPrediction = 0, pageBusy = true, debugBusy = false;
let stopRequested = false, operation = null, unloading = false;
const workbench = new ModelWorkbench({root: document.getElementById('workbench'), pb: api.pb,
  title: 'TinyStories · GPT-Neo', subtitle: 'Inspect token predictions, incremental execution and KV state.',
  debug: {getSource: source, createSession: options => session.createDebugSession(settings(), {...options, prediction: selectedPrediction, run: lastRun}),
    onProfileInspect: async (output, row) => {
      const lineage = row?.lineage;
      const phase = lastRun?.phaseRecords.find(value => String(value.executionId) === String(lineage?.executionId) && String(value.contextId) === String(lineage?.contextId));
      if (phase) { await workbench.debugger.reset(); selectedPrediction = phase.index; renderTokens(); renderSelection(); }
    }, tensorViews: [{id: 'decode', label: 'Decode / KV', create: options => new DecodeStateView(options)}]},
  onRecord: () => generate(true), onBusyChange: busy => { debugBusy = busy; syncControls(); }});
const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = new URL('TinyStories.css', import.meta.url);
workbench.inputRoot.getRootNode().append(css);
workbench.controlsRoot.innerHTML = `<label>Backend <select id="backendSelect"><option value="wasm">WASM · CPU</option><option value="webgpu">WebGPU</option></select></label><span class="badge">TinyStories-1M · FP32</span><button id="generateBtn" class="primary" disabled>Generate</button><button id="stopBtn" disabled>Stop</button>`;
workbench.inputRoot.innerHTML = `
  <div class="tiny-workspace"><section class="tiny-generation">
    <div class="card tiny-prompt"><div class="card-heading"><h2>Prompt &amp; sampling</h2><span id="promptCount" class="badge">Loading tokenizer…</span></div><div class="tiny-prompt-body">
      <label for="promptInput" class="hint">Prompt text · byte-level BPE</label><textarea id="promptInput" rows="3" spellcheck="false">Once upon a time, Lily</textarea>
      <div class="tiny-parameters"><label>New tokens<input id="tokensInput" type="number" min="1" max="256" value="32"></label><label>Seed<input id="seedInput" type="number" min="0" max="4294967295" value="42"></label><label>Temperature<input id="tempInput" type="number" min="0" max="5" step="0.05" value="0.7"></label><label>Top-K<input id="topkInput" type="number" min="1" max="50257" value="40"></label><button id="randomSeedBtn" title="Randomize sampling seed">New seed</button></div>
      <div class="tiny-caption"><span id="contextUsage">—</span><span>Temperature 0 → greedy</span></div></div></div>
    <div class="card tiny-story"><div class="card-heading"><h2>Generated continuation</h2><span id="runState" class="badge">No run</span></div><div id="story" class="tiny-story-text" aria-live="polite">Load the model to generate a continuation.</div><div class="tiny-story-footer"><span class="hint">Select a predicted token to inspect the operation that produced its logits.</span><button id="exportRun" disabled>Export run JSON</button></div></div>
    <div class="tiny-metrics"><div class="metric"><span>First token</span><strong id="ttft">—</strong><small>Generation start → first sample</small></div><div class="metric"><span>Decode mean</span><strong id="decodeLatency">—</strong><small>RPC + wait + logits read</small></div><div class="metric"><span>Generation wall</span><strong id="totalTime">—</strong><small id="timingKind">Ordinary run · includes sampling/UI</small></div><div class="metric"><span>Output rate</span><strong id="tps">—</strong><small id="tokenProgress">No generated tokens</small></div></div>
  </section><aside class="tiny-inspection">
    <div class="card tiny-tokens"><div class="card-heading"><h2>Prediction navigator</h2><span id="predictionCount" class="badge">0 predictions</span></div><div class="tiny-token-toolbar"><button id="previousToken" aria-label="Previous token">←</button><select id="predictionSelect" aria-label="Prediction operation"><option value="0">Prefill · first prediction</option></select><button id="nextToken" aria-label="Next token">→</button></div><div id="tokenList" class="tiny-token-list" aria-label="Generated token predictions"></div>
      <div class="tiny-selected"><div class="tiny-selection-heading"><strong id="selectedToken">Prefill · first prediction</strong><span id="selectedId" class="badge">—</span></div><p id="selectedContext" class="hint">The prompt is executed in an isolated decode context.</p><div class="tiny-inspect-actions"><button id="debugToken" class="primary">Debug prediction</button><button id="openProfile">Record generation</button></div><div class="tiny-candidate-heading"><span>Candidate token</span><span>Top-K probability</span></div><div id="candidates" class="tiny-candidates"></div></div>
    </div>
    <details class="card tiny-model" open><summary>Model &amp; execution provenance</summary><dl class="model-properties"><dt>Package</dt><dd id="pkgPath" class="mono"></dd><dt>Compiled route</dt><dd id="routeInfo">—</dd><dt>Preparation</dt><dd id="loadTime">—</dd><dt>Inputs</dt><dd class="mono">tokens, positions · I32 [1,S]</dd><dt>Logits</dt><dd id="outputShape" class="mono">—</dd><dt>Prompt IDs</dt><dd id="promptIds" class="mono">—</dd><dt>Capacity</dt><dd id="capacityInfo">—</dd></dl></details>
  </aside></div><pre id="out" hidden></pre>`;
const ui = Object.fromEntries(['backendSelect', 'generateBtn', 'stopBtn', 'promptInput', 'tokensInput', 'seedInput', 'tempInput', 'topkInput', 'randomSeedBtn', 'promptCount', 'contextUsage', 'runState', 'story', 'exportRun', 'ttft', 'decodeLatency', 'totalTime', 'tps', 'tokenProgress', 'timingKind', 'predictionCount', 'previousToken', 'nextToken', 'predictionSelect', 'tokenList', 'selectedToken', 'selectedId', 'selectedContext', 'debugToken', 'openProfile', 'candidates', 'pkgPath', 'routeInfo', 'loadTime', 'outputShape', 'promptIds', 'capacityInfo', 'out'].map(id => [id, workbench.inputRoot.querySelector(`#${id}`) ?? workbench.controlsRoot.querySelector(`#${id}`)]));
for (const [parameter, id] of [['backend', 'backendSelect'], ['seed', 'seedInput'], ['temp', 'tempInput'], ['topk', 'topkInput'], ['n', 'tokensInput'], ['prompt', 'promptInput']]) if (query.has(parameter)) ui[id].value = query.get(parameter);
if (!navigator.gpu) ui.backendSelect.querySelector('[value="webgpu"]').disabled = true;

function settings() { return {prompt: ui.promptInput.value, requestedNewTokens: Number(ui.tokensInput.value), seed: Number(ui.seedInput.value), temperature: Number(ui.tempInput.value), topK: Number(ui.topkInput.value)}; }
function source() {
  const data = lastRun ?? prepared, index = selectedPrediction, position = (data?.promptLength ?? data?.state?.promptLength ?? 1) - 1 + index;
  return {label: `${index === 0 ? 'Prefill' : `Decode prediction ${index}`} · position ${position} · seed ${data?.seed ?? ui.seedInput.value}`, backend: session?.backend ?? ui.backendSelect.value, variant: 'FP32', graph: session?.graph, decodeTarget: true,
    details: {phase: index === 0 ? 'prefill' : 'decode', generatedIndex: index, position, modelUrl: session?.modelUrl, sequenceCapacity: data?.sequenceCapacity ?? data?.state?.sequenceCapacity, prompt: data?.prompt, promptTokenIds: data?.promptTokenIds,
      prefixTokenIds: data ? [...data.promptTokenIds, ...(lastRun?.generatedTokenIds.slice(0, index) ?? [])] : [], seed: data?.seed, temperature: data?.temperature, topK: data?.topK}};
}
function syncControls() {
  const busy = pageBusy || debugBusy, available = !!session && !!prepared;
  for (const field of [ui.backendSelect, ui.promptInput, ui.tokensInput, ui.seedInput, ui.tempInput, ui.topkInput, ui.randomSeedBtn, ui.predictionSelect, ...ui.tokenList.querySelectorAll('button')]) field.disabled = busy;
  ui.generateBtn.disabled = busy || !available; ui.stopBtn.disabled = !pageBusy || !operation;
  ui.debugToken.disabled = ui.openProfile.disabled = busy || !available; ui.exportRun.disabled = busy || !lastRun;
  ui.previousToken.disabled = busy || selectedPrediction <= 0; ui.nextToken.disabled = busy || selectedPrediction >= (lastRun?.records.length ?? 1) - 1;
  workbench.debugger.setAvailable(!pageBusy && available); workbench.profiler.setAvailable(available); workbench.profiler.setBusy(busy);
}
function prepareInput() {
  try { prepared = session.prepare(settings()); ui.promptCount.textContent = `${prepared.state.promptLength} prompt tokens`; ui.promptIds.textContent = prepared.promptTokenIds.join(', '); ui.capacityInfo.textContent = `${prepared.state.sequenceCapacity} allocated · ${session.contract.maximumSequenceCapacity} maximum`; ui.contextUsage.textContent = `${prepared.state.promptLength} prompt + up to ${prepared.state.generationLimit} new / ${prepared.state.sequenceCapacity} rows`; return true; }
  catch (error) { prepared = null; ui.promptCount.textContent = 'Invalid input'; workbench.setStatus(error.message, {error: true}); return false; }
}
function clearRun() { lastRun = null; selectedPrediction = 0; for (const key of ['ttft', 'decodeLatency', 'totalTime', 'tps']) ui[key].textContent = '—'; ui.runState.textContent = 'No run'; ui.tokenProgress.textContent = 'No generated tokens'; ui.story.textContent = ui.promptInput.value; renderTokens(); renderSelection(); }
function renderTokens() {
  const records = lastRun?.records ?? []; ui.predictionCount.textContent = `${records.length} predictions`; ui.predictionSelect.replaceChildren(); ui.tokenList.replaceChildren();
  if (!records.length) { const option = element('option', '', 'Prefill · first prediction'); option.value = '0'; ui.predictionSelect.append(option); ui.tokenList.append(element('p', 'hint', 'Generate a continuation to inspect incremental decode. Prefill is ready now.')); return; }
  for (const record of records) {
    const text = record.eos ? '<EOS>' : record.tokenText.replace(/\n/g, '↵'), option = element('option', '', `${record.index} · ${record.phase} · ${text}`); option.value = record.index; ui.predictionSelect.append(option);
    const button = element('button', 'tiny-token'); button.dataset.prediction = record.index; button.setAttribute('aria-pressed', String(record.index === selectedPrediction)); button.append(element('span', 'tiny-token-index', record.index), element('span', 'tiny-token-text', text || '∅'), element('span', 'tiny-token-id', record.tokenId)); button.title = `${record.phase} at row ${record.position}; token ${record.tokenId}; ${formatMs(record.wallMs)}`; button.onclick = () => selectPrediction(record.index); ui.tokenList.append(button);
  }
  ui.predictionSelect.value = String(selectedPrediction);
}
function renderSelection() {
  const record = lastRun?.records[selectedPrediction], data = source().details; ui.selectedToken.textContent = record ? `${record.phase === 'prefill' ? 'Prefill' : 'Decode'} ${record.index} · ${record.eos ? '<EOS>' : JSON.stringify(record.tokenText)}` : 'Prefill · first prediction'; ui.selectedId.textContent = record ? `token ${record.tokenId}` : 'No sample yet'; ui.selectedContext.textContent = `Logits row ${data.position} · ${data.prefixTokenIds.length} prefix tokens · isolated context replay`; ui.candidates.replaceChildren();
  if (!record) { ui.candidates.append(element('p', 'hint', 'Candidate logits and sampling probabilities appear after generation.')); return; }
  for (const value of record.candidates.slice(0, 8)) { const row = element('div', 'tiny-candidate'); row.dataset.selected = String(value.tokenId === record.tokenId); row.append(element('span', 'mono', `${value.tokenId} ${JSON.stringify(session.tokenizer.decodeToken(value.tokenId))}`), element('span', 'mono', `${(value.probability * 100).toFixed(2)}%`)); row.title = `logit ${value.logit} · temperature ${lastRun.temperature} · top-K ${lastRun.topK}`; ui.candidates.append(row); }
}
async function selectPrediction(index) {
  if (pageBusy || debugBusy) return; pageBusy = true; syncControls();
  try { await workbench.debugger.reset(); selectedPrediction = index; renderTokens(); renderSelection(); }
  catch (error) { workbench.setStatus(error.message, {error: true}); } finally { pageBusy = false; syncControls(); }
}
function showProgress(run) {
  lastRun = run; selectedPrediction = Math.min(selectedPrediction, Math.max(0, run.records.length - 1)); ui.story.textContent = run.text; ui.ttft.textContent = formatMs(run.firstTokenMs); ui.decodeLatency.textContent = formatMs(run.decodeMs); ui.totalTime.textContent = formatMs(run.wallMs); ui.tps.textContent = run.wallMs > 0 ? `${(run.generatedTokenIds.length * 1000 / run.wallMs).toFixed(1)} tok/s` : '—'; ui.tokenProgress.textContent = `${run.generatedTokenIds.length} / ${run.generationLimit} generated`; ui.contextUsage.textContent = `${run.promptLength} prompt + ${run.generatedTokenIds.length} new / ${run.sequenceCapacity} allocated rows`; renderTokens(); renderSelection(); syncControls();
}
async function generate(profiled = false) {
  if (pageBusy || debugBusy || !session || !prepareInput()) return;
  pageBusy = true; stopRequested = false; syncControls(); workbench.setStatus(profiled ? 'Recording generation…' : 'Generating…');
  operation = (async () => {
    try {
      await workbench.debugger.reset(); workbench.profiler.reset(); clearRun(); ui.timingKind.textContent = profiled ? 'Instrumented run · includes sampling/UI' : 'Ordinary run · includes sampling/UI';
      const result = await session.generate(settings(), {profiled, shouldStop: () => stopRequested || unloading, onToken: run => { showProgress(run); ui.runState.textContent = 'Generating'; }});
      showProgress(result); ui.runState.textContent = result.stopReason === 'eos' ? 'EOS' : result.stopReason === 'stopped' ? 'Stopped' : 'Length limit';
      if (profiled) workbench.profiler.setTrace({trace: result.trace, wallMs: result.wallMs, source: {...source(), label: `Generation · ${result.promptLength} prompt + ${result.generatedTokenIds.length} output tokens`, phaseRecords: result.phaseRecords}});
      workbench.setStatus(`Ready · ${result.generatedTokenIds.length} tokens`); ui.out.textContent = `DONE backend=${result.backend} tps=${result.generatedTokenIds.length * 1000 / result.wallMs}\n${result.text}`;
    } catch (error) {
      ui.runState.textContent = 'Failed'; ui.out.textContent = `FATAL ${error.message}`;
      if (lastRun) { lastRun.stopReason = 'failed'; lastRun.error = error.message; }
      workbench.setStatus(error.message, {error: true}); if (profiled) workbench.profiler.error(error.message);
    }
    finally { operation = null; pageBusy = false; syncControls(); }
  })(); syncControls(); await operation;
}
async function changeSettings() {
  if (pageBusy || debugBusy || !session) return; pageBusy = true; syncControls();
  try { await workbench.debugger.reset(); workbench.profiler.reset(); clearRun(); if (prepareInput()) workbench.setStatus('Ready'); }
  catch (error) { workbench.setStatus(error.message, {error: true}); } finally { pageBusy = false; syncControls(); }
}
async function loadModel() {
  await workbench.debugger.reset({clearCaptureSelection: true}); workbench.profiler.reset(); await session?.close(); session = null; prepared = null; clearRun(); ui.pkgPath.textContent = packageUrl.pathname;
  const response = await fetch(new URL('graph.json', packageUrl)); if (!response.ok) throw new Error(`Model graph unavailable (${response.status}). Run make models_tinystories.`);
  const graph = await response.json(), tokenizer = new Tokenizer(), start = performance.now(); await tokenizer.load(new URL('vocab.bin', packageUrl), new URL('merges.txt', packageUrl));
  session = await TinyStoriesSession.open({api, wasmUrl, graph, tokenizer, graphUrl: new URL('graph.json', packageUrl), weightsUrl: new URL('model.safetensors', packageUrl), backend: ui.backendSelect.value});
  ui.loadTime.textContent = formatMs(performance.now() - start); const route = session.compileReport.route; ui.routeInfo.textContent = `${session.backend} · ${route?.activeNodes ?? '?'} active / ${route?.selectedNodes ?? '?'} selected · ${route?.attested ? 'attested' : 'unattested'} · required, no operator fallback`; ui.outputShape.textContent = `${session.contract.outputName} · F32 [1,S,${session.contract.vocabularySize}]`; if (prepareInput()) workbench.setStatus('Ready'); renderSelection();
}
ui.generateBtn.onclick = () => generate(); ui.stopBtn.onclick = () => { stopRequested = true; workbench.setStatus('Stopping after current token…'); };
ui.debugToken.onclick = () => workbench.show('debug'); ui.openProfile.onclick = () => workbench.show('profile'); ui.predictionSelect.onchange = () => selectPrediction(Number(ui.predictionSelect.value)); ui.previousToken.onclick = () => selectPrediction(selectedPrediction - 1); ui.nextToken.onclick = () => selectPrediction(selectedPrediction + 1);
ui.exportRun.onclick = () => { const {trace, state, ...run} = lastRun; download(jsonBlob({format: 'volvoxai-tinystories-run/v1', ...run, source: source()}), 'tinystories-run.json'); };
ui.randomSeedBtn.onclick = () => { ui.seedInput.value = crypto.getRandomValues(new Uint32Array(1))[0]; void changeSettings(); };
for (const input of [ui.promptInput, ui.tokensInput, ui.seedInput, ui.tempInput, ui.topkInput]) input.onchange = () => changeSettings();
ui.backendSelect.onchange = async () => { if (pageBusy || debugBusy) return; pageBusy = true; syncControls(); workbench.setStatus('Loading & compiling…'); try { await loadModel(); } catch (error) { workbench.setStatus(error.message, {error: true}); } finally { pageBusy = false; syncControls(); } };
addEventListener('pagehide', event => { if (event.persisted) return; unloading = stopRequested = true; void Promise.resolve(operation).finally(async () => { await workbench.dispose(); await session?.close(); }); });
try { syncControls(); await loadModel(); } catch (error) { workbench.setStatus(error.message, {error: true}); } finally { pageBusy = false; syncControls(); }
if (session && !['0', 'false'].includes(query.get('auto'))) await generate();
