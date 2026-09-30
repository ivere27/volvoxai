import {mountView, element, download, jsonBlob, formatBytes, formatMs, tabNavigation} from './View.js';
import {summarizeProfile, observationLabel, profileEvidence} from './ProfileData.js';
import {openPerfetto} from '../Perfetto.js';

const template = `
  <section class="profile-workspace">
    <header class="profile-toolbar"><div><h2>Performance &amp; memory</h2><p class="hint" id="profileSource">Record an inference to inspect measured work and resource use.</p></div><span class="spacer"></span><button id="profileRecord" class="primary" disabled>● Record inference</button><button id="profileBaseline" disabled>Pin baseline</button><button id="profileJson" disabled>Analysis JSON ↓</button><button id="profileTrace" disabled>Chrome trace ↓</button><button id="profilePerfetto" disabled>Open Perfetto ↗</button></header>
    <p id="profileStatus" class="notice" role="status">Capturing is explicit. Opening this view does not start a profiler.</p>
    <div id="profileEmpty" class="empty"><span class="profile-empty-icon" aria-hidden="true">▥</span><h2>Measure before optimizing</h2><p>Record the current input to see expensive operators, memory ownership and CPU/GPU observations.</p><p class="hint">Host duration, GPU elapsed time and hardware utilization are different measurements.</p></div>
    <div id="profileContent" hidden>
      <div id="profileMetrics" class="metric-grid"></div><div id="profileComparison" class="notice" hidden></div>
      <div class="tabs" role="tablist" aria-label="Profile analysis"><button id="profileOperatorsTab" data-profile-tab="operators" role="tab" aria-selected="true" aria-controls="profileOperators">Operators</button><button id="profileMemoryTab" data-profile-tab="memory" role="tab" aria-selected="false" tabindex="-1" aria-controls="profileMemory">Memory &amp; resources</button></div>
      <section id="profileOperators" role="tabpanel" aria-labelledby="profileOperatorsTab" class="profile-operators">
        <div class="profile-results"><div class="profile-filter row"><input id="profileSearch" type="search" placeholder="Find operator, tensor, source…" aria-label="Search profiled operators"><label>Sort <select id="profileSort"><option value="host">Host duration</option><option value="device">GPU elapsed</option><option value="step">Execution order</option><option value="bytes">Tensor size</option></select></label><span id="profileCount" class="hint"></span></div>
          <div class="profile-timeline"><div class="row"><strong>Host node timeline</strong><span class="hint">Click a span to inspect</span></div><svg id="profileTimeline" role="img" aria-label="Measured host node intervals"></svg><p class="hint">GPU intervals retain their own clocks. Open Perfetto for full tracks and clock bounds.</p></div>
          <div class="profile-table"><table><thead><tr><th>Step / operator</th><th>Output</th><th class="numeric">Host</th><th class="numeric">GPU</th><th class="numeric">Host share</th></tr></thead><tbody id="profileRows"></tbody></table></div>
          <p class="profile-footnote hint">Shares use measured host-node durations. Nested program timings are excluded; fused work is not divided among source nodes.</p>
        </div>
        <aside id="profileDetail" class="profile-detail"><div class="empty"><h3>Select an operator</h3><p>Inspect measured durations, tensor placement and the corresponding debug node.</p></div></aside>
      </section>
      <section id="profileMemory" role="tabpanel" aria-labelledby="profileMemoryTab" class="profile-memory" hidden>
        <div class="card"><div class="card-heading"><h2>WASM module heap</h2><span class="hint">Module scope · not process RSS</span></div><div id="profileHeap" class="card-body"></div></div>
        <div class="card"><div class="card-heading"><h2>Allocator ownership</h2><span class="hint">Observed requested capacities</span></div><div class="profile-table"><table><thead><tr><th>Allocator / scope</th><th class="numeric">Live</th><th class="numeric">Observed peak</th><th>Coverage</th></tr></thead><tbody id="profileAllocators"></tbody></table></div><p class="profile-footnote hint">These allocations overlap the WASM heap or device storage. Do not add them to RSS or module memory. Driver-private storage may be absent.</p></div>
        <div class="card"><div class="card-heading"><h2>CPU / GPU observations</h2><span class="hint">Actual observation scope</span></div><div id="profileResources" class="card-body"></div></div>
      </section>
    </div>
  </section>`;

export class ProfileView {
  #root; #e; #pb; #record; #inspect; #trace = null; #summary = null; #source; #wallMs; #baseline = null;
  #busy = false; #available = false; #selected = null;
  constructor({root, pb, onRecord, onInspect}) {
    this.#root = mountView(root, template, ['Profile.css']); this.#pb = pb; this.#record = onRecord; this.#inspect = onInspect;
    this.#e = Object.fromEntries(['Source', 'Record', 'Baseline', 'Json', 'Trace', 'Perfetto', 'Status', 'Empty', 'Content', 'Metrics', 'Comparison',
      'Search', 'Sort', 'Count', 'Timeline', 'Rows', 'Detail', 'Heap', 'Allocators', 'Resources'].map(key => [key, this.#root.getElementById(`profile${key}`)]));
    const e = this.#e;
    e.Record.onclick = () => this.#record();
    e.Search.oninput = e.Sort.onchange = () => this.#renderRows();
    e.Json.onclick = () => download(jsonBlob(profileEvidence(this.#trace, this.#source, this.#wallMs)), 'volvoxai-profile.json');
    e.Trace.onclick = () => download(this.#trace.blob, 'volvoxai-trace.json');
    e.Perfetto.onclick = async () => {
      try { await openPerfetto(this.#trace.blob, 'volvoxai-trace.json'); }
      catch (error) { this.error(error.message); }
    };
    e.Baseline.onclick = () => {
      this.#baseline = {source: this.#source, wallMs: this.#wallMs, heap: this.#summary.wasm?.linearBytes ?? null}; this.#compare();
    };
    const tabs = this.#root.querySelectorAll('[data-profile-tab]');
    for (const button of tabs) button.onclick = () => {
      for (const tab of tabs) {
        const active = button === tab; tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1;
        this.#root.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
      }
    };
    tabNavigation(tabs); this.#controls();
  }

  setAvailable(value) { this.#available = value; this.#controls(); }
  setBusy(value) { this.#busy = value; this.#controls(); }
  error(message) { this.#e.Status.textContent = message; this.#e.Status.dataset.error = 'true'; }
  reset() {
    this.#trace = this.#summary = null; this.#selected = null;
    this.#e.Content.hidden = true; this.#e.Empty.hidden = false; this.#e.Search.value = '';
    this.#e.Status.textContent = 'Record the current input to collect a new profile. The pinned baseline, if any, is retained for comparison.';
    this.#e.Status.dataset.error = 'false'; this.#e.Source.textContent = 'Current model and input';
    for (const key of ['Rows', 'Timeline', 'Metrics', 'Heap', 'Allocators', 'Resources', 'Detail']) this.#e[key].replaceChildren();
    this.#controls();
  }
  setTrace({trace, source, wallMs}) {
    const summary = summarizeProfile(trace, this.#pb);
    this.#trace = trace; this.#source = {...source, capturedAt: new Date().toISOString()}; this.#wallMs = wallMs;
    this.#summary = summary; this.#selected = null;
    const e = this.#e, s = this.#summary, loss = Object.values(s.loss).some(count => count > 0n);
    e.Content.hidden = false; e.Empty.hidden = true;
    e.Source.textContent = `${source.variant} / ${source.backend} · ${source.label}`;
    e.Status.dataset.error = String(loss);
    e.Status.textContent = `${loss ? 'Incomplete capture' : 'Capture ready'} · ${trace.info.events?.count ?? 0n} events · ${trace.resources.length} resource samples · ${formatBytes(trace.info.collectorBytes)} collector · dropped ${s.loss.events} events / ${s.loss.resources} samples / ${s.loss.plans} plans. Single instrumented run.`;
    e.Metrics.replaceChildren();
    for (const [label, value, hint] of [
      ['Inference wall time', formatMs(wallMs), 'Includes host / transport; profiled run'],
      ['Measured host nodes', formatMs(s.hostMs), `${s.nodes.length} executable nodes · not CPU utilization`],
      ['WASM linear memory', formatBytes(s.wasm?.linearBytes), `Sampled peak ${formatBytes(s.sampledWasmPeak)} · module scope`],
      ['Process CPU load', s.cpuLabel, `GPU telemetry: ${observationLabel(s.latest?.gpu, this.#pb)} · see Resources`],
    ]) {
      const metric = element('div', 'metric'); metric.append(element('span', '', label), element('strong', '', value), element('small', '', hint)); e.Metrics.append(metric);
    }
    this.#renderRows(); this.#renderTimeline(); this.#renderMemory(); this.#compare();
    const slowest = s.nodes.reduce((best, row) => !best || (row.hostMs ?? -1) > (best.hostMs ?? -1) ? row : best, null);
    if (slowest) this.#select(slowest);
    else e.Detail.replaceChildren(element('p', 'empty', 'No complete node measurements in this capture.'));
    this.#controls();
  }

  #controls() {
    this.#e.Record.disabled = this.#busy || !this.#available;
    for (const key of ['Baseline', 'Json', 'Trace', 'Perfetto']) this.#e[key].disabled = this.#busy || !this.#trace;
    const inspect = this.#e.Detail.querySelector('button');
    if (inspect) inspect.disabled = this.#busy || !this.#available || !this.#inspect;
  }

  #compare() {
    const e = this.#e, base = this.#baseline; e.Comparison.hidden = !base || !this.#summary;
    if (!base || !this.#summary) return;
    e.Comparison.replaceChildren(element('strong', '', `Baseline: ${base.source.variant} / ${base.source.backend} · ${base.source.label}`),
      element('p', '', `Wall time ${formatMs(base.wallMs)} → ${formatMs(this.#wallMs)} (${this.#wallMs - base.wallMs >= 0 ? '+' : ''}${(this.#wallMs - base.wallMs).toFixed(2)} ms). Module heap ${formatBytes(base.heap)} → ${formatBytes(this.#summary.wasm?.linearBytes)}.`));
    e.Comparison.append(element('p', 'hint', 'One capture per result; compare matching inputs and capture settings. This is not a repeated-run benchmark.'));
    const clear = element('button', '', 'Clear baseline'); clear.onclick = () => { this.#baseline = null; this.#compare(); }; e.Comparison.append(clear);
  }

  #renderRows() {
    if (!this.#summary) return;
    const e = this.#e, query = e.Search.value.toLowerCase().trim(), sort = e.Sort.value;
    const rows = this.#summary.nodes.filter(row => [row.name, row.output, ...row.sources].join(' ').toLowerCase().includes(query));
    rows.sort((a, b) => sort === 'step' ? a.step - b.step : sort === 'bytes' ? Number(b.tensor?.logicalBytes ?? -1n) - Number(a.tensor?.logicalBytes ?? -1n) :
      sort === 'device' ? (b.deviceMs ?? -1) - (a.deviceMs ?? -1) : (b.hostMs ?? -1) - (a.hostMs ?? -1));
    e.Count.textContent = `${rows.length} / ${this.#summary.nodes.length} nodes`; e.Rows.replaceChildren();
    for (const row of rows) {
      const tr = element('tr'); tr.dataset.key = row.key; tr.classList.toggle('selected', this.#selected === row.key);
      const identity = element('td'), button = element('button', 'profile-node', `#${row.step} ${row.name}${row.fused ? ' · fused' : ''}`);
      button.onclick = () => this.#select(row); identity.append(button, element('div', 'hint', row.backend));
      tr.append(identity, element('td', 'mono', row.output), element('td', 'numeric', formatMs(row.hostMs)), element('td', 'numeric', formatMs(row.deviceMs)));
      const share = element('td', 'numeric', row.share === null ? '—' : `${(row.share * 100).toFixed(1)}%`);
      const bar = element('div', 'profile-share'); bar.style.width = `${(row.share ?? 0) * 100}%`; share.append(bar); tr.append(share); e.Rows.append(tr);
    }
    if (!rows.length) { const tr = element('tr'), td = element('td', 'empty', 'No matching measured nodes.'); td.colSpan = 5; tr.append(td); e.Rows.append(tr); }
  }

  #select(row) {
    this.#selected = row.key; this.#renderRows(); const detail = this.#e.Detail;
    detail.replaceChildren(element('span', 'eyebrow', `Step ${row.step} · ${row.backend}`), element('h3', '', row.name), element('p', 'mono', row.output));
    const dl = element('dl', 'profile-properties');
    for (const [label, value] of [['Host duration', formatMs(row.hostMs)], ['GPU elapsed', formatMs(row.deviceMs)],
      ['GPU observations', row.deviceIntervals ? `${row.deviceIntervals} ${row.deviceBasis}` : 'Not captured'], ['Host calls', row.calls],
      ['Sources', row.sources.join(', ') || 'Unknown'], ['Shape', row.tensor ? `[${row.tensor.shape.join(', ')}]` : 'Not captured'],
      ['Logical tensor bytes', formatBytes(row.tensor?.logicalBytes)], ['Allocation offset', formatBytes(row.tensor?.offsetBytes)],
      ['Planned lifetime', row.tensor?.firstStep !== undefined ? `steps ${row.tensor.firstStep} – ${row.tensor.lastStep}` : 'Unknown']])
      dl.append(element('dt', '', label), element('dd', '', value));
    detail.append(dl, element('p', 'hint', 'Logical tensor bytes can alias arena storage. GPU sums cover only recorded intervals; missing intervals are not zero. Durations do not measure utilization.'));
    const inspect = element('button', 'primary', 'Inspect tensor in debugger →'); inspect.disabled = !this.#inspect || this.#busy || !this.#available;
    inspect.onclick = () => { if (!this.#busy && this.#available) void this.#inspect(row.output); }; detail.append(inspect);
  }

  #renderTimeline() {
    const svg = this.#e.Timeline, ns = 'http://www.w3.org/2000/svg', intervals = this.#summary.timeline;
    svg.replaceChildren();
    if (!intervals.length) return;
    const end = intervals.reduce((value, i) => Math.max(value, i.start + i.duration), 0) || 1;
    const tracks = [...new Set(intervals.map(i => i.track))], height = tracks.length * 30 + 25;
    svg.setAttribute('viewBox', `0 0 1000 ${height}`); svg.style.height = `${height}px`;
    for (const item of intervals) {
      const rect = document.createElementNS(ns, 'rect'); rect.setAttribute('x', String(item.start / end * 1000)); rect.setAttribute('y', String(tracks.indexOf(item.track) * 30 + 2));
      rect.setAttribute('width', String(Math.max(.8, item.duration / end * 1000))); rect.setAttribute('height', '26'); rect.setAttribute('rx', '2');
      const title = document.createElementNS(ns, 'title'); title.textContent = `Host track ${item.track} · ${item.name} · ${item.output} · ${formatMs(item.duration)} @ ${formatMs(item.start)}`; rect.append(title);
      rect.onclick = () => this.#select(this.#summary.nodes.find(n => n.key === item.key)); svg.append(rect);
    }
    for (let i = 0; i <= 4; i++) {
      const text = document.createElementNS(ns, 'text'); text.setAttribute('x', String(i * 249)); text.setAttribute('y', String(height - 5));
      text.setAttribute('text-anchor', i === 4 ? 'end' : 'start'); text.textContent = `${(i / 4 * end).toFixed(1)} ms`; svg.append(text);
    }
  }

  #renderMemory() {
    const e = this.#e, s = this.#summary; e.Heap.replaceChildren();
    if (s.wasm) {
      const entries = [['Allocated', s.wasm.allocatedBytes], ['Reusable free', s.wasm.freeBytes], ['Allocator metadata', s.wasm.allocatorMetadataBytes],
        ['Module prefix', s.wasm.modulePrefixBytes], ['Page slack', s.wasm.pageSlackBytes], ['Untracked', s.wasm.untrackedBytes]];
      const bar = element('div', 'heap-bar');
      entries.forEach(([label, bytes], i) => { const part = element('span'); part.style.flex = String(bytes); part.style.background = `var(--heap-${i})`; part.title = `${label}: ${formatBytes(bytes)}`; bar.append(part); });
      const list = element('div', 'heap-legend');
      entries.forEach(([label, bytes], i) => { const item = element('div'); const dot = element('i'); dot.style.background = `var(--heap-${i})`; item.append(dot, element('span', '', label), element('strong', '', formatBytes(bytes))); list.append(item); });
      e.Heap.append(bar, list, element('p', 'hint', `Linear memory ${formatBytes(s.wasm.linearBytes)} · largest free block ${formatBytes(s.wasm.largestFreeBlockBytes)}. Free allocator blocks and unused space inside tensor arenas are different. Linear memory does not shrink on free.`));
    } else e.Heap.append(element('p', 'hint', observationLabel(s.latest?.wasm, this.#pb)));
    e.Allocators.replaceChildren();
    for (const a of s.allocators) {
      const tr = element('tr'); tr.append(element('td', '', `${a.allocator} · ${this.#pb.MemoryOwnerKind[a.scope]?.replace('MEMORY_OWNER_KIND_', '') ?? 'unknown scope'}`),
        element('td', 'numeric', formatBytes(a.liveBytes)), element('td', 'numeric', formatBytes(a.peakBytes)), element('td', '', `${a.accountingComplete ? 'Tracked identities complete' : 'Partial accounting'} · ${a.droppedEvents} dropped events`)); e.Allocators.append(tr);
    }
    e.Resources.replaceChildren(element('p', '', `Process RSS: ${s.rss === null ? observationLabel(s.latest?.process?.resident, this.#pb) : formatBytes(s.rss)}`),
      element('p', '', `Process CPU: ${s.cpuLabel}`),
      element('p', 'hint', `CPU-time delta / elapsed observation time across all process threads; 1.0 means one occupied core. Source: ${s.latest?.cpu?.source || 'unreported'}.`));
    if (s.latest?.memory) e.Resources.append(element('p', 'hint', `Allocation inventory: ${this.#pb.MemoryInventoryKind[s.latest.memory.inventory]?.replace('MEMORY_INVENTORY_KIND_', '') ?? 'unknown'}${s.latest.memory.truncated ? ' · truncated' : ''}. Inventory coverage and allocator event accounting are separate.`));
    if (!s.gpu.length) e.Resources.append(element('p', '', `GPU utilization: ${observationLabel(s.latest?.gpu, this.#pb)}`));
    for (const gpu of s.gpu) {
      const hasUtilization = gpu.utilizationStatus === this.#pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE;
      e.Resources.append(element('h3', '', gpu.name), element('p', '', `Device activity: ${hasUtilization && gpu.gpuUtilization !== undefined ? `${(gpu.gpuUtilization * 100).toFixed(1)}%` : 'Unavailable'} · Memory interface: ${hasUtilization && gpu.memoryUtilization !== undefined ? `${(gpu.memoryUtilization * 100).toFixed(1)}%` : 'Unavailable'}`),
        element('p', '', `Device memory: ${this.#observedBytes(gpu.memoryUsed)} used / ${this.#observedBytes(gpu.memoryTotal)} total`),
        element('p', 'hint', `${gpu.source} · device scope, includes other applications · sample window ${gpu.samplingWindowNs === undefined ? 'unreported' : formatMs(nsToMs(gpu.samplingWindowNs))}`));
    }
    e.Resources.append(element('p', 'hint', 'Browser runtimes cannot report process CPU load or hardware GPU utilization through these APIs. Unsupported is never displayed as zero. No utilization is inferred from operator duration.'));
  }

  #observedBytes(value) {
    return value?.status === this.#pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE && value.bytes !== undefined ? formatBytes(value.bytes) : observationLabel(value, this.#pb);
  }
}

const nsToMs = value => Number(value) / 1e6;
