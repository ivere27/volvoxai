import {element, download, jsonBlob, formatBytes} from './View.js';
import {tensorDecoder, formatTensorValue} from '../TensorValues.js';

/** Inspect the engine's decode state at a stop. Live cache bytes are read only on request. */
export class DecodeStateView {
  #root; #pb; #run; #source = null; #state = null; #busy = false; #e;
  constructor({root, pb, run}) {
    this.#root = root; this.#pb = pb; this.#run = run;
    root.innerHTML = `<div class="decode-state-view">
      <div class="row"><h3>Decode cursors &amp; KV caches</h3><span class="spacer"></span><button data-action="refresh">Refresh state</button><button data-action="export">State JSON ↓</button></div>
      <p class="hint">Engine cache state at the current stop. A pending writer leaves the previous contents in its token rows.</p>
      <p data-view="status" class="notice" role="status">Start a decode session to inspect its state.</p>
      <section class="card"><div class="card-heading"><h2>Slot cursors</h2></div><div class="decode-table"><table><thead><tr><th>Slot</th><th>Length before → after</th><th>Write range</th><th>State</th></tr></thead><tbody data-view="slots"></tbody></table></div></section>
      <section class="card"><div class="card-heading"><h2>Self-attention caches</h2></div><div class="decode-table"><table><thead><tr><th>Cache / tensor</th><th>Role / dtype</th><th>Token shape</th><th>Capacity / pages</th><th>Writer</th></tr></thead><tbody data-view="caches"></tbody></table></div></section>
      <section class="card"><div class="card-heading"><h2>Read live cache values</h2><span class="hint">Raw storage units</span></div><div class="card-body">
        <div class="row"><label>Cache <select data-view="cache" aria-label="KV cache"></select></label><label>Slot <select data-view="slot" aria-label="Decode slot"></select></label><label>Token row <input data-view="token" aria-label="Logical token row" type="number" min="0" value="0"></label><button data-action="read">Read row</button><button data-action="download">Cache bytes ↓</button></div>
        <p data-view="readInfo" class="hint">Read one logical token row, or explicitly download all readable rows in this cache.</p><pre data-view="values" tabindex="0"></pre>
      </div></section>
    </div>`;
    const style = element('style'); style.textContent = `.decode-state-view {display:grid; gap:14px; min-width:0} .decode-table {overflow:auto} .decode-state-view label {display:flex; align-items:center; gap:5px; font-size:11px} .decode-state-view select {max-width:200px} .decode-state-view input {width:80px; border:1px solid var(--line); border-radius:5px; background:var(--panel); padding:6px} .decode-state-view pre {max-height:280px; margin-top:12px; line-height:1.7} .decode-state-view td {font-size:11px} .decode-state-view .card {border-radius:7px}`;
    root.prepend(style);
    this.#e = Object.fromEntries([...root.querySelectorAll('[data-view]')].map(item => [item.dataset.view, item]));
    root.querySelector('[data-action="refresh"]').onclick = () => run(() => this.#refresh());
    root.querySelector('[data-action="read"]').onclick = () => run(() => this.#read());
    root.querySelector('[data-action="download"]').onclick = () => run(() => this.#download());
    root.querySelector('[data-action="export"]').onclick = () => download(jsonBlob(this.toJson()), 'volvoxai-decode-state.json');
    this.#e.cache.onchange = this.#e.slot.onchange = () => { this.#clearValues(); this.#controls(); };
    this.#e.token.oninput = () => this.#clearValues();
    this.#controls();
  }

  setContext(source) { this.#source = source; this.#state = null; this.#clearValues(); }
  async show() { await this.#refresh(); }
  clearSelection() { this.#clearValues(); }
  reset() {
    this.#source = this.#state = null;
    for (const key of ['slots', 'caches', 'cache', 'slot']) this.#e[key].replaceChildren();
    this.#clearValues(); this.#e.status.textContent = 'Start a decode session to inspect its state.'; this.#controls();
  }
  setBusy(value) { this.#busy = value; this.#controls(); }
  toJson() { return {format: 'volvoxai-decode-state/v1', details: this.#source?.details, state: this.#state?.toJson(), liveValuesIncluded: false}; }

  #clearValues() { this.#e.values.textContent = ''; this.#e.readInfo.textContent = 'Reads observe live storage at this stop; stepping can change these rows.'; }
  #controls() {
    const active = !!this.#source?.decodeTarget && !!this.#source?.session;
    for (const control of this.#root.querySelectorAll('button, input, select')) control.disabled = this.#busy || !active;
    for (const action of ['read', 'download']) this.#root.querySelector(`[data-action="${action}"]`).disabled = this.#busy || !this.#state?.caches.length || !this.#state?.slots.length;
    this.#root.querySelector('[data-action="export"]').disabled = this.#busy || !this.#state;
  }
  async #refresh() {
    if (!this.#source?.decodeTarget || !this.#source?.session) {
      this.#e.status.textContent = 'This execution has no engine decode target. Inspect explicit past/present tensors in Tensor values.';
      this.#controls(); return;
    }
    const cache = this.#e.cache.value, slot = this.#e.slot.value;
    this.#state = await this.#source.session.decodeState();
    const e = this.#e, state = this.#state;
    e.status.textContent = `${this.#pb.DebugTarget[state.target].replace('DEBUG_TARGET_', '')} · ${state.slots.length} slots · ${state.caches.length} cache operands · stop #${this.#source.session.info.nextStep}`;
    e.slots.replaceChildren(); e.caches.replaceChildren(); e.cache.replaceChildren(); e.slot.replaceChildren();
    for (const item of state.slots) {
      const row = element('tr');
      for (const value of [item.slot, `${item.lengthBefore} → ${item.lengthAfter}`, `[${item.writeBegin}, ${item.writeEnd})`, item.empty ? 'Empty' : 'Active']) row.append(element('td', '', value));
      e.slots.append(row); e.slot.append(new Option(String(item.slot), String(item.slot)));
    }
    for (const item of state.caches) {
      const tensor = this.#source.session.plan.tensors.find(t => t.tensorId === item.tensorId), name = tensor?.name ?? `tensor ${item.tensorId}`;
      const row = element('tr');
      for (const value of [`${item.cacheId} · ${name}`, `${this.#pb.DebugKVRole[item.role].replace('DEBUG_KV_ROLE_', '')} / ${this.#pb.DataType[item.dtype].replace('DATA_TYPE_', '')}`,
        `[${item.tokenShape.join(', ')}]`, `${item.tokenCapacity}${item.pageTokens ? ` / ${item.pageTokens} tokens per page` : ' / contiguous'}`,
        item.writerStep === undefined ? 'No writer in this operation' : `#${item.writerStep} · ${item.written ? 'written' : 'pending'}`]) row.append(element('td', '', value));
      e.caches.append(row); e.cache.append(new Option(`${item.cacheId} · ${name}`, String(item.cacheId)));
    }
    if ([...e.cache.options].some(option => option.value === cache)) e.cache.value = cache;
    if ([...e.slot.options].some(option => option.value === slot)) e.slot.value = slot;
    this.#clearValues(); this.#controls();
  }
  async #read() {
    const offset = Number(this.#e.token.value);
    if (!Number.isSafeInteger(offset) || offset < 0) throw new Error('Token row must be a non-negative integer.');
    const page = await this.#source.session.readKVCachePage(Number(this.#e.cache.value), Number(this.#e.slot.value), offset, 1);
    const {width, read} = tensorDecoder(page.dtype, this.#pb), view = new DataView(page.data.buffer, page.data.byteOffset, page.data.byteLength);
    const count = Math.min(256, page.data.byteLength / width);
    const values = Array.from({length: count}, (_, index) => `${index}: ${formatTensorValue(read(view, index * width))}`);
    this.#e.values.textContent = values.join('\n');
    this.#e.readInfo.textContent = `Row ${offset} / ${page.tokenCount} readable tokens · shape [${page.shape.join(', ')}] · ${formatBytes(page.data.byteLength)} · first ${count} elements${page.quantization ? ' · quantized storage units' : ''}`;
  }
  async #download() {
    const cacheId = Number(this.#e.cache.value), slot = Number(this.#e.slot.value);
    const cache = this.#state.caches.find(item => item.cacheId === cacheId), cursor = this.#state.slots.find(item => item.slot === slot);
    const {width} = tensorDecoder(cache.dtype, this.#pb);
    const bytes = BigInt(cursor.lengthAfter) * cache.tokenShape.reduce((count, extent) => count * extent, 1n) * BigInt(width);
    if (bytes > 64n * 1024n * 1024n) throw new Error('This cache exceeds the 64 MiB browser download limit. Read bounded rows instead.');
    const value = await this.#source.session.readKVCache(cacheId, slot);
    download(new Blob([value.data], {type: 'application/octet-stream'}), `volvoxai-kv-${cacheId}-slot-${slot}.bin`);
    download(jsonBlob({format: 'volvoxai-kv-cache/v1', cacheId, slot, details: this.#source.details,
      dtype: this.#pb.DataType[value.dtype], shape: value.shape, quantization: value.quantization?.toJson()}), `volvoxai-kv-${cacheId}-slot-${slot}.json`);
  }
}
