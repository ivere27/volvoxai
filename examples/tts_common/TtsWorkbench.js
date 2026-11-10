import {loadRuntimeRelease} from '../common/RuntimeRelease.js';
import {TtsSession, validateManifest, tensorValues, pcm16Wav, benchmarkPrepared} from './TtsSession.js';

const family = document.body.dataset.model;
const root = document.querySelector('main');
root.innerHTML = `<h1>${family === 'piper' ? 'Piper' : 'Kokoro'} speech example</h1>
  <p>Generate a prepared utterance entirely with VolvoxAI and compare it with the saved ONNX reference.</p>
  <form id="load"><label>Prepared manifest <input id="manifest" required></label><button>Load example</button></form>
  <p id="help">Prepare the model first using <a href="./${family}/README.md">the example instructions</a>.
  Model weights stay on your server. Inference runs in this browser.</p>
  <p id="status" role="status" aria-live="polite"></p>
  <section hidden id="example"><h2 id="model"></h2><p id="scope"></p><p id="text"></p>
    <label>Prepared request <select id="request"></select></label>
    <h3>ONNX reference</h3><audio id="reference" controls></audio>
    <p id="availability"></p>
    <form id="run"><label>VolvoxAI backend <select id="backend"><option value="wasm">WASM CPU</option><option value="webgpu">WebGPU</option></select></label>
      <button id="synthesize">Run and compare</button></form>
    <p>One first execution, two warmups and five measured executions. Timings include input binding,
    device completion and host output; model loading and audio playback are separate.</p>
    <audio id="candidate" controls hidden></audio><pre id="results"></pre>
  </section>`;
const element = id => document.getElementById(id);
element('manifest').value = new URLSearchParams(location.search).get('manifest') ?? `../build/tts-examples/${family === 'piper' ? 'piper-full' : 'kokoro-complete'}/manifest.json`;
let manifest, manifestUrl, busy = false;
const objectUrls = new Map();
function audio(id, data, rate) {
  if (objectUrls.has(id)) URL.revokeObjectURL(objectUrls.get(id));
  const url = URL.createObjectURL(new Blob([pcm16Wav(data, rate)], {type: 'audio/wav'}));
  objectUrls.set(id, url); element(id).src = url; element(id).hidden = false;
}
function lock(value) {
  busy = value;
  for (const control of root.querySelectorAll('button,input,select')) control.disabled = value;
  if (!value && manifest) element('synthesize').disabled = manifest.volvox.status !== 'ready';
}
async function readBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url.pathname} (HTTP ${response.status}). Run the preparation command first.`);
  return new Uint8Array(await response.arrayBuffer());
}
async function selectRequest() {
  const fixture = manifest.cases[Number(element('request').value)];
  element('text').textContent = fixture.text ?? manifest.text ?? fixture.phonemes ?? fixture.id;
  const descriptor = fixture.output_files?.find(value => value.name === (fixture.audio_output ?? 'audio'));
  if (!descriptor) throw new Error('The manifest has no saved ONNX waveform. Re-run preparation.');
  audio('reference', tensorValues(await readBytes(new URL(descriptor.path, manifestUrl)), descriptor), fixture.sample_rate ?? manifest.sample_rate);
  element('candidate').hidden = true; element('results').textContent = '';
}
element('load').addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return; lock(true); element('status').textContent = 'Loading prepared example…';
  element('example').hidden = true;
  try {
    manifestUrl = new URL(element('manifest').value, location.href);
    manifest = validateManifest(JSON.parse(new TextDecoder().decode(await readBytes(manifestUrl))));
    element('model').textContent = typeof manifest.model === 'string' ? manifest.model : manifest.model?.name ?? family;
    element('scope').textContent = `Measured computation: ${manifest.scope ?? 'whole model'}.`;
    element('availability').textContent = manifest.volvox.status === 'ready'
      ? 'Every model stage runs in VolvoxAI on the selected backend. The prepared package fixes token count and audio length.'
      : `This full model currently runs with ONNX Runtime only. VolvoxAI export: ${manifest.volvox.reason}`;
    element('request').replaceChildren(...manifest.cases.map((fixture, i) => new Option(fixture.id, String(i))));
    await selectRequest(); element('example').hidden = false; element('status').textContent = 'Prepared audio loaded.';
  } catch (error) { manifest = undefined; element('status').textContent = error.message; }
  finally { lock(false); }
});
element('request').addEventListener('change', async () => {
  lock(true); try { await selectRequest(); } catch (error) { element('status').textContent = error.message; } finally { lock(false); }
});
element('run').addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return; lock(true); let session;
  element('status').textContent = 'Loading and compiling the model…'; element('results').textContent = '';
  try {
    const runtime = await loadRuntimeRelease('full'), start = performance.now();
    session = await TtsSession.open({manifestUrl, ...runtime, backend: element('backend').value});
    const loadCompileMs = performance.now() - start, fixture = manifest.cases[Number(element('request').value)];
    const {last, summary} = await benchmarkPrepared(session, fixture, {reportWaveformDifferences: true, onProgress: () => {
      element('status').textContent = 'Warming up and measuring completed executions…';
    }});
    audio('candidate', last.audio, fixture.sample_rate ?? manifest.sample_rate);
    element('results').textContent = JSON.stringify({backend: session.backend, load_compile_ms: loadCompileMs,
      first_ms: summary.first_ms, timing: summary.timing, comparison: summary.comparison}, null, 2);
    element('status').textContent = summary.comparison.pass
      ? 'Complete. Every waveform satisfies the declared comparison criterion.'
      : 'Synthesis complete. The waveform differs from ONNX beyond the elementwise tolerance; errors are shown below.';
  } catch (error) { element('status').textContent = error.message; }
  finally { try { if (session) await session.close(); } finally { lock(false); } }
});
window.addEventListener('pagehide', () => { for (const url of objectUrls.values()) URL.revokeObjectURL(url); });
