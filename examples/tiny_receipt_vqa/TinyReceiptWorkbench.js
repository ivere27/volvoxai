import {loadRuntimeRelease} from '../common/RuntimeRelease.js';
import {ModelWorkbench} from '../common/workbench/ModelWorkbench.js';
import {formatMs, element} from '../common/workbench/View.js';
import {TinyReceiptSession} from './TinyReceiptSession.js';

const {api, wasmUrl} = await loadRuntimeRelease();
const packages = {fp32: new URL('../../build/tiny-receipt-kv-f32/', import.meta.url), int8: new URL('../../build/tiny-receipt-kv-int8/', import.meta.url)};
const samples = {
  receipt_ko: {label: 'Korean receipt', question: '가게 전화번호는 무엇입니까?', reference: '착한 도매슈퍼마켓 · 911-3125 · 엄치길 875', questions: [
    '가게 전화번호는 무엇입니까?', '영수증의 가게 위치에서 도로명 뒤 숫자는 무엇입니까?', '가게 이름은 무엇입니까?',
    '1번째 품목명은 무엇입니까?', '품목 총합을 모두 더하면 얼마입니까?', '과일류 개수는 몇 개입니까?', '1번째 품목의 가격 곱하기 개수는 얼마입니까?']},
  receipt_en: {label: 'English receipt', question: 'What is the store phone number?', reference: 'Best price Mall · 653-3831 · St. 837', questions: [
    'What is the store phone number?', 'What is the street number in the store address?', 'What is the store name?',
    'What is the name of item 1?', 'What is the sum of item totals?', 'What is the quantity of Packed Yogurt?', 'Calculate item 1 unit price multiplied by quantity.']},
};
let session = null, pageBusy = true, debugBusy = false, hasImage = false;
const options = () => ({questionCapacity: Number(ui.questionCapacity.value), maxTokens: Number(ui.maxTokens.value), family: Number(ui.family.value),
  sampling: ui.sampling.value, temperature: Number(ui.temperature.value), topK: Number(ui.topK.value), seed: Number(ui.seed.value)});
function source() {
  const target = ui.debugTarget.value, position = target === 'encoder' ? null : target === 'prefill' ? 0 : Number(ui.debugPosition.value);
  const run = session?.lastRun, row = run?.tokenRows[position], role = target === 'encoder' ? 'encoder' : 'decoder';
  return {label: `${ui.sceneDescription.textContent} · ${role}${position === null ? '' : ` / token ${position}`}`,
    backend: session?.backend ?? ui.backend.value, variant: ui.variant.value, graph: session?.graphs[role], manifest: session?.manifest, decodeTarget: false,
    details: {target, role, position, image: ui.sceneDescription.textContent, question: ui.question.value.trim(), ...options(),
      questionTokens: run?.questionTokens, memoryTokens: run?.memoryTokens, inputToken: row?.inputToken, outputToken: row?.outputToken,
      pastLength: row?.pastLength, prefixTokens: position > 0 ? run?.tokens.slice(0, position) : [], cacheContract: session?.manifest.cache_contract}};
}
const workbench = new ModelWorkbench({root: document.getElementById('workbench'), pb: api.pb,
  title: 'Tiny Receipt VQA', subtitle: 'Encoder / explicit KV decoder · inspect a real autoregressive invocation',
  debug: {getSource: source, createSession: debugOptions => session.createDebugSession(currentImage(), ui.question.value.trim(), {
    ...options(), ...debugOptions, target: ui.debugTarget.value, position: Number(ui.debugPosition.value)}),
    onProfileInspect: async (_output, row) => {
      const phase = session.phaseForProfile(row); ui.debugTarget.value = phase.target; ui.debugPosition.value = Math.max(1, phase.position);
      await workbench.debugger.reset(); syncControls();
    }}, onRecord: () => run(true), onBusyChange: busy => { debugBusy = busy; syncControls(); }});
workbench.controlsRoot.innerHTML = `<label>Backend <select id="backend" aria-label="Backend"><option value="wasm">WASM</option><option value="webgpu">WebGPU</option></select></label>
  <label>Model <select id="variant" aria-label="Model precision"><option value="fp32">FP32</option><option value="int8">INT8</option></select></label>
  <label>Debug <select id="debugTarget" aria-label="Debug target"><option value="encoder">Encoder</option><option value="prefill">Decoder prefill</option><option value="decode">Decoder token</option></select></label>
  <label>Token <input id="debugPosition" aria-label="Decoder token position" type="number" value="1" min="1" max="190" step="1" disabled></label><button id="run" class="primary" disabled>Generate</button>`;
workbench.inputRoot.innerHTML = `<div class="vqa-layout">
  <section class="vqa-column vqa-image-column">
    <div class="card"><div class="card-heading"><h2>Receipt image</h2><span id="canvasDim" class="badge">672 × 320</span></div>
      <div class="vqa-image-toolbar"><button data-sample="receipt_ko" aria-pressed="true">Korean</button><button data-sample="receipt_en" aria-pressed="false">English</button>
        <label class="vqa-upload">Upload <input id="file" type="file" accept="image/png,image/jpeg" aria-label="Upload receipt image"></label></div>
      <div class="vqa-image"><canvas id="canvas" width="672" height="320" aria-label="Receipt model input"></canvas></div>
      <div class="vqa-caption"><strong id="sceneDescription">Korean receipt</strong><span>Grayscale · [−1, 1]</span></div><p id="sampleReference" class="hint vqa-reference">Sample reference</p></div>
    <div class="card"><div class="card-heading"><h2>Model context</h2><span id="modelFormat" class="badge">FP32</span></div>
      <dl class="vqa-properties"><dt>Package</dt><dd id="pkgPath" class="mono">—</dd><dt>Image</dt><dd id="inputShape" class="mono">—</dd>
        <dt>Question / memory</dt><dd id="shapeState" class="mono">Q ≤ 192 · M = Q + 210</dd><dt>Cache</dt><dd id="cacheGeometry" class="mono">4 × K/V · [1, 8, P, 40]</dd>
        <dt>First call</dt><dd>P=1 masked zero sentinel + BOS</dd><dt>Each token</dt><dd>Present K/V → next past K/V · R=P+1</dd><dt>Tokenizer</dt><dd>1,536 byte-fallback BPE · NFC</dd></dl>
      <div class="vqa-flow"><span>Image + question</span><b>→</b><span>Encoder</span><b>→</b><span>Cross K/V</span><b>→</b><span>Decoder × tokens</span></div></div>
  </section>
  <section class="vqa-column vqa-request-column">
    <div class="card"><div class="card-heading"><h2>Question &amp; generation</h2><span id="familyBadge" class="badge">Auto router</span></div>
      <div class="vqa-request-body"><label for="question">Question</label><textarea id="question" rows="3" placeholder="Ask about the receipt" spellcheck="false" aria-describedby="questionHint"></textarea>
        <p id="questionHint" class="hint" role="status" hidden>Enter a question to enable generation and debugging.</p>
        <div id="questionChips" class="vqa-question-chips" aria-label="Sample questions"></div>
        <div class="vqa-options"><label>Family <select id="family"><option value="-1">Auto router</option></select></label>
          <label>Question limit <input id="questionCapacity" type="number" min="1" max="192" value="192" step="1"></label>
          <label>New tokens <input id="maxTokens" type="number" min="1" max="191" value="191" step="1"></label>
          <label>Sampling <select id="sampling"><option value="greedy">Greedy</option><option value="sample">Temperature</option></select></label>
          <label>Temperature <input id="temperature" type="number" min="0.01" max="5" value="1" step="0.05"></label>
          <label>Top K <input id="topK" type="number" min="0" max="1536" value="40" step="1" title="0 selects the whole vocabulary"></label>
          <label>Seed <input id="seed" type="number" min="0" max="4294967295" value="1" step="1"></label></div>
        <p class="hint vqa-policy">Limits bind to the package domain. Greedy uses the first-index tie rule. Sampling uses a reproducible seed.</p></div></div>
    <div class="card"><div class="card-heading"><h2>Compiled execution</h2><span id="routeBadge" class="badge">Loading</span></div>
      <dl class="vqa-properties"><dt>Encoder</dt><dd id="encoderRoute">—</dd><dt>Decoder</dt><dd id="decoderRoute">—</dd><dt>Preparation</dt><dd id="metricCompile">—</dd></dl>
      <p class="hint vqa-policy">Both graphs require the selected backend and forbid operator fallback. Debug creates an independent invocation.</p>
      <div class="vqa-actions"><button id="openDebug">Debug selected phase →</button><button id="openProfile">Performance &amp; memory →</button></div></div>
  </section>
  <section class="vqa-column vqa-result-column">
    <div class="card vqa-answer-card"><div class="card-heading"><h2>Generated answer</h2><span id="eosBadge" class="badge">No run</span></div>
      <strong id="answerValue" class="vqa-answer" aria-live="polite">—</strong><dl class="vqa-fields"><dt>Field</dt><dd id="outField">—</dd><dt>Value</dt><dd id="outValue">—</dd><dt>Operation</dt><dd id="outOp">—</dd></dl>
      <div class="vqa-metrics"><div><span>Encoder</span><strong id="metricEncoder">—</strong></div><div><span>Decoder</span><strong id="metricDecoder">—</strong></div>
        <div><span>Generation</span><strong id="metricWall">—</strong></div><div><span>Tokens / rate</span><strong id="metricTokens">—</strong></div></div>
      <p id="timingKind" class="hint vqa-policy">One generation · end-to-end wall time</p><progress id="progress" max="191" value="0" aria-label="Generation progress"></progress></div>
    <div class="card vqa-token-card"><div class="card-heading"><h2>Autoregressive sequence</h2><span id="tokenCount" class="badge">0 tokens</span></div>
      <pre id="rawOutput" class="vqa-raw">Run inference to inspect the structured output.</pre><div class="vqa-token-scroll"><table class="vqa-token-table"><thead><tr><th>Call</th><th>Output token</th><th>K/V P→R</th><th class="numeric">Wall</th></tr></thead><tbody id="tokenRows"></tbody></table></div>
      <p class="hint vqa-policy">Select a token call to debug its input and cache prefix. Prefill is position 0; later calls consume the preceding output token.</p></div>
  </section></div>`;
const style = document.createElement('link'); style.rel = 'stylesheet'; style.href = new URL('./TinyReceiptWorkbench.css', import.meta.url); workbench.inputRoot.prepend(style);
const ids = ['backend', 'variant', 'debugTarget', 'debugPosition', 'run', 'canvas', 'canvasDim', 'file', 'sceneDescription', 'sampleReference', 'modelFormat', 'pkgPath', 'inputShape',
  'shapeState', 'cacheGeometry', 'question', 'questionHint', 'questionChips', 'questionCapacity', 'maxTokens', 'family', 'sampling', 'temperature', 'topK', 'seed', 'familyBadge', 'encoderRoute', 'decoderRoute',
  'routeBadge', 'metricCompile', 'openDebug', 'openProfile', 'eosBadge', 'answerValue', 'outField', 'outValue', 'outOp', 'metricEncoder', 'metricDecoder', 'metricWall', 'metricTokens',
  'timingKind', 'progress', 'tokenCount', 'rawOutput', 'tokenRows'];
const ui = Object.fromEntries(ids.map(id => [id, workbench.inputRoot.querySelector(`#${id}`) ?? workbench.controlsRoot.querySelector(`#${id}`)]));
const context = ui.canvas.getContext('2d', {willReadFrequently: true}), sampleButtons = workbench.inputRoot.querySelectorAll('[data-sample]');
if (!navigator.gpu) ui.backend.querySelector('[value="webgpu"]').disabled = true;
function currentImage() {
  const image = context.getImageData(0, 0, ui.canvas.width, ui.canvas.height); return {data: image.data, width: image.width, height: image.height, channels: 4};
}
function syncControls() {
  const busy = pageBusy || debugBusy, ready = !!session && hasImage && !!ui.question.value.trim();
  for (const control of [ui.backend, ui.variant, ui.debugTarget, ui.file, ui.question, ui.questionCapacity, ui.maxTokens, ui.family, ui.sampling, ...sampleButtons,
    ...ui.questionChips.querySelectorAll('button'), ...ui.tokenRows.querySelectorAll('button')]) control.disabled = busy;
  for (const control of [ui.temperature, ui.topK, ui.seed]) control.disabled = busy || ui.sampling.value === 'greedy';
  ui.run.disabled = busy || !ready;
  const steps = session?.lastRun?.tokens.length ?? 0; ui.debugTarget.querySelector('[value="decode"]').disabled = steps < 2;
  ui.debugPosition.disabled = busy || ui.debugTarget.value !== 'decode'; ui.debugPosition.max = Math.max(1, steps - 1);
  const debugReady = ready && (ui.debugTarget.value !== 'decode' || steps > 1);
  workbench.debugger.setAvailable(!pageBusy && debugReady); workbench.profiler.setAvailable(ready); workbench.profiler.setBusy(busy);
  ui.openDebug.disabled = busy || !debugReady; ui.openProfile.disabled = busy || !ready;
  ui.questionHint.hidden = !!ui.question.value.trim();
}
function clearResult() {
  session?.clearGeneration(); for (const name of ['answerValue', 'outField', 'outValue', 'outOp', 'metricEncoder', 'metricDecoder', 'metricWall', 'metricTokens']) ui[name].textContent = '—';
  ui.eosBadge.textContent = 'No run'; delete ui.eosBadge.dataset.complete; ui.tokenCount.textContent = '0 tokens'; ui.rawOutput.textContent = 'Run inference to inspect the structured output.';
  ui.tokenRows.replaceChildren(); ui.progress.value = 0; ui.familyBadge.textContent = 'Auto router'; if (ui.debugTarget.value === 'decode') ui.debugTarget.value = 'prefill';
  ui.shapeState.textContent = `Q ≤ ${ui.questionCapacity.value} · M = Q + ${session?.manifest.shape_contract.fixed_geometry.image_tokens ?? 210}`;
}
function routeText(report) { return `${report.backend} · ${report.route.activeNodes}/${report.route.selectedNodes} nodes · attested · 0 fallback`; }
async function openModel() {
  await session?.close(); session = null; ui.pkgPath.textContent = `build/tiny-receipt-kv-${ui.variant.value === 'fp32' ? 'f32' : 'int8'}`;
  const started = performance.now(); session = await TinyReceiptSession.open({api, wasmUrl, packageUrl: packages[ui.variant.value], backend: ui.backend.value});
  ui.metricCompile.textContent = formatMs(performance.now() - started); ui.modelFormat.textContent = ui.variant.value.toUpperCase();
  ui.encoderRoute.textContent = routeText(session.compileReports.encoder); ui.decoderRoute.textContent = routeText(session.compileReports.decoder); ui.routeBadge.textContent = 'REQUIRE · FORBID';
  const geo = session.manifest.shape_contract.fixed_geometry; ui.inputShape.textContent = `image [${geo.image.join(', ')}] · F32`;
  ui.cacheGeometry.textContent = `${geo.decoder_layers} × K/V · [1, ${geo.attention_heads}, P, ${geo.attention_head_width}] · F32`;
  ui.questionCapacity.max = session.questionCapacity; ui.questionCapacity.value = Math.min(Number(ui.questionCapacity.value), session.questionCapacity);
  ui.maxTokens.max = session.maxTokens; ui.maxTokens.value = Math.min(Number(ui.maxTokens.value), session.maxTokens); ui.progress.max = Number(ui.maxTokens.value);
  ui.family.replaceChildren(new Option('Auto router', '-1')); session.families.forEach((name, index) => ui.family.append(new Option(name, String(index))));
}
async function setImage(blob, label) {
  const bitmap = await createImageBitmap(blob);
  try { const geo = session.manifest.shape_contract.fixed_geometry.image; ui.canvas.width = geo[3]; ui.canvas.height = geo[2];
    context.drawImage(bitmap, 0, 0, ui.canvas.width, ui.canvas.height); hasImage = true;
  } finally { bitmap.close(); }
  ui.sceneDescription.textContent = label; ui.canvasDim.textContent = `${ui.canvas.width} × ${ui.canvas.height}`;
}
async function loadSample(key) {
  const response = await fetch(new URL(`./benchmarks/work/source/examples/${key}.jpg`, import.meta.url)); if (!response.ok) throw new Error(`Sample image unavailable: ${key}`);
  await setImage(await response.blob(), samples[key].label); ui.question.value = samples[key].question; ui.sampleReference.textContent = `Reference: ${samples[key].reference}`;
  for (const button of sampleButtons) button.setAttribute('aria-pressed', String(button.dataset.sample === key)); ui.questionChips.replaceChildren();
  for (const question of samples[key].questions) { const button = element('button', 'vqa-question-chip', question);
    button.onclick = async () => { await changeInput(() => { ui.question.value = question; }, 'Question updated'); await run(); }; ui.questionChips.append(button); }
}
function showResult(result, profiled) {
  ui.answerValue.textContent = result.parsed.answer ?? result.parsed.raw ?? '—';
  ui.outField.textContent = result.parsed.field ?? '—'; ui.outValue.textContent = result.parsed.value ?? '—'; ui.outOp.textContent = result.parsed.op ?? '—';
  ui.eosBadge.textContent = result.reachedEos ? 'EOS reached' : 'Token limit reached'; ui.eosBadge.dataset.complete = String(result.reachedEos);
  ui.familyBadge.textContent = session.families[result.family] ?? `Family ${result.family}`;
  ui.metricEncoder.textContent = formatMs(result.encoderMs); ui.metricDecoder.textContent = formatMs(result.decoderMs); ui.metricWall.textContent = formatMs(result.wallMs);
  ui.metricTokens.textContent = `${result.tokens.length} · ${(result.tokens.length * 1000 / Math.max(result.decoderMs, .001)).toFixed(1)}/s`;
  ui.timingKind.textContent = profiled ? 'Instrumented generation · one run' : 'One generation · end-to-end wall time';
  ui.shapeState.textContent = `Q=${result.questionTokens} · M=${result.memoryTokens} · P=1…${result.tokens.length}`;
  ui.tokenCount.textContent = `${result.tokens.length} tokens`; ui.rawOutput.textContent = result.parsed.raw || '(No printable output)'; ui.tokenRows.replaceChildren();
  for (const row of result.tokenRows) {
    const tr = element('tr'), cell = element('td'), button = element('button', 'vqa-token-call', row.position === 0 ? '0 · prefill' : `${row.position} · decode`);
    button.dataset.position = row.position; button.title = `Input ${row.inputToken} → output ${row.outputToken}. Debug this invocation.`;
    button.onclick = async () => { ui.debugTarget.value = row.position === 0 ? 'prefill' : 'decode'; ui.debugPosition.value = Math.max(1, row.position);
      await workbench.debugger.reset(); syncControls(); workbench.show('debug'); }; cell.append(button); tr.append(cell);
    tr.append(element('td', 'vqa-token-literal', `${row.literal} · #${row.outputToken}`), element('td', 'mono', `${row.pastLength}→${row.presentLength}`),
      element('td', 'numeric mono', formatMs(row.wallMs))); ui.tokenRows.append(tr);
  }
  ui.progress.value = result.tokens.length; ui.debugPosition.value = Math.min(Math.max(1, Number(ui.debugPosition.value)), Math.max(1, result.tokens.length - 1));
}
async function run(profiled = false) {
  if (pageBusy || debugBusy || !session || !hasImage) return; pageBusy = true; syncControls(); workbench.setStatus(profiled ? 'Recording generation…' : 'Generating…');
  try {
    await workbench.debugger.reset(); workbench.profiler.reset(); ui.answerValue.textContent = 'Generating…'; ui.progress.value = 0; ui.progress.max = Number(ui.maxTokens.value);
    const result = await session.run(currentImage(), ui.question.value.trim(), {...options(), profile: profiled,
      onProgress: event => { ui.progress.value = event.position; workbench.setStatus(event.phase === 'encoder' ? 'Encoder…' : `${event.phase} · token ${event.position}`); }});
    showResult(result, profiled);
    if (profiled) workbench.profiler.setTrace({trace: result.trace, wallMs: result.wallMs, source: {
      label: `${ui.sceneDescription.textContent} · ${ui.question.value.trim()}`, backend: session.backend, variant: ui.variant.value,
      phaseRecords: result.phases.map(phase => ({...phase, label: phase.role === 'encoder' ? `Encoder · Q=${result.questionTokens}` :
        `${phase.phase === 'prefill' ? 'Prefill' : 'Decode'} ${phase.position} · ${session.vocab.itos[phase.inputToken]} → ${session.vocab.itos[phase.outputToken]}`})),
      details: {image: ui.sceneDescription.textContent, family: result.family, questionTokens: result.questionTokens, memoryTokens: result.memoryTokens,
        tokens: result.tokens, ...result.provenance}}}); workbench.setStatus('Ready');
  } catch (error) { workbench.setStatus(error.message, {error: true}); ui.answerValue.textContent = '—'; if (profiled) workbench.profiler.error(error.message); }
  finally { pageBusy = false; syncControls(); }
}
async function changeInput(action, status) {
  if (pageBusy || debugBusy) return; pageBusy = true; syncControls(); workbench.setStatus(status);
  try { await workbench.debugger.reset({clearCaptureSelection: true}); workbench.profiler.reset(); clearResult(); await action(); workbench.setStatus('Ready'); }
  catch (error) { workbench.setStatus(error.message, {error: true}); } finally { pageBusy = false; syncControls(); }
}
ui.run.onclick = () => run(); ui.backend.onchange = ui.variant.onchange = () => changeInput(openModel, 'Loading & compiling…');
ui.openDebug.onclick = () => workbench.show('debug'); ui.openProfile.onclick = () => workbench.show('profile');
ui.debugTarget.onchange = ui.debugPosition.onchange = async () => { await workbench.debugger.reset({clearCaptureSelection: true}); syncControls(); };
ui.question.oninput = () => { session?.clearGeneration(); if (ui.debugTarget.value === 'decode') ui.debugTarget.value = 'prefill'; syncControls(); };
for (const name of ['question', 'questionCapacity', 'maxTokens', 'family', 'sampling', 'temperature', 'topK', 'seed']) ui[name].onchange = () => changeInput(() => {}, 'Input updated');
for (const button of sampleButtons) button.onclick = () => changeInput(() => loadSample(button.dataset.sample), 'Loading image…');
ui.file.onchange = () => {
  const file = ui.file.files?.[0]; if (!file) return;
  void changeInput(async () => { await setImage(file, file.name); ui.sampleReference.textContent = 'Custom image · no sample reference';
    for (const button of sampleButtons) button.setAttribute('aria-pressed', 'false'); }, 'Loading image…');
};
async function init() {
  syncControls(); try { await openModel(); await loadSample('receipt_ko'); workbench.setStatus('Ready'); }
  catch (error) { workbench.setStatus(error.message, {error: true}); } finally { pageBusy = false; syncControls(); }
}
addEventListener('pagehide', event => { if (!event.persisted) void session?.close(); }); void init();
