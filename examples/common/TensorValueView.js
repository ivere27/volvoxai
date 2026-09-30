import {tensorLayout, tensorIndex, readTensorPage, formatTensorValue, summarizeTensorPage} from './TensorValues.js';

/** A bounded, on-demand view of one captured storage version. */
export class TensorValueView {
  #pb;
  #recapture;
  #elements;
  #session;
  #snapshot;
  #layout;
  #offset = 0n;
  #rows = [];
  #busy = false;
  #highlight = null;

  constructor({root, pb, run, recapture}) {
    this.#pb = pb; this.#recapture = recapture;
    this.#elements = Object.fromEntries(['Title', 'Message', 'Range', 'First', 'Previous', 'Next', 'Last',
      'Offset', 'Go', 'Count', 'Index', 'Jump', 'Rows', 'Shade', 'Csv', 'Recapture', 'RecaptureNote',
      'Mode', 'Table', 'MapPanel', 'Map', 'MapReadout', 'MapMin', 'MapMax', 'Distribution', 'Histogram', 'HistogramCaption']
      .map(key => [key, root.querySelector(`#tensorValue${key}`)]));
    const e = this.#elements;
    const navigate = action => run(async () => {
      try { await action(); }
      catch (error) { e.Message.textContent = error.message; e.Message.dataset.error = 'true'; }
    });
    e.First.onclick = () => navigate(() => this.#load(0n));
    e.Previous.onclick = () => navigate(() => this.#load(this.#offset > this.#pageSize() ? this.#offset - this.#pageSize() : 0n));
    e.Next.onclick = () => navigate(() => this.#load(this.#offset + BigInt(this.#rows.length)));
    e.Last.onclick = () => navigate(() => this.#load((this.#layout.count - 1n) / this.#pageSize() * this.#pageSize()));
    e.Go.onclick = () => navigate(async () => {
      if (!/^\d+$/.test(e.Offset.value.trim())) throw new Error('Start element must be a non-negative integer');
      await this.#load(BigInt(e.Offset.value.trim()));
    });
    e.Jump.onclick = () => navigate(async () => {
      const index = tensorIndex(this.#layout.shape, e.Index.value);
      await this.#load(index / this.#pageSize() * this.#pageSize(), index);
    });
    e.Count.onchange = () => navigate(() => this.#load(this.#offset));
    e.Shade.onchange = () => this.#renderRows();
    e.Mode.onchange = () => this.#renderRows();
    e.Map.onpointermove = event => {
      if (!this.#rows.length) return;
      const bounds = e.Map.getBoundingClientRect(), columns = Math.ceil(Math.sqrt(this.#rows.length));
      const index = Math.floor((event.clientY - bounds.top) / bounds.height * columns) * columns +
        Math.floor((event.clientX - bounds.left) / bounds.width * columns);
      const item = this.#rows[index];
      e.MapReadout.textContent = item ? `#${item.index} [${item.coordinates.join(', ')}] = ${formatTensorValue(item.value)}` +
        (item.dequantized === undefined ? '' : ` · dequantized ${formatTensorValue(item.dequantized)}`) : 'Unused cell';
    };
    e.Csv.onclick = () => this.#downloadPage();
    e.Recapture.onclick = () => run(() => this.#recapture(this.#snapshot));
    for (const [field, action] of [[e.Offset, e.Go], [e.Index, e.Jump]]) {
      field.onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); action.click(); } };
    }
    this.reset();
  }

  #pageSize() { return BigInt(this.#elements.Count.value); }

  setBusy(busy) { this.#busy = busy; this.#controls(); }

  reset() {
    this.#session = this.#snapshot = this.#layout = null;
    this.#offset = 0n; this.#rows = []; this.#highlight = null;
    const e = this.#elements;
    e.Title.textContent = 'Tensor values';
    e.Message.textContent = 'Select a captured input or output tensor above to browse its actual values.';
    e.Message.dataset.error = 'false';
    e.Range.textContent = ''; e.Rows.replaceChildren();
    e.Distribution.hidden = e.MapPanel.hidden = true; e.Table.hidden = false;
    e.Map.getContext('2d').clearRect(0, 0, e.Map.width, e.Map.height);
    e.Histogram.replaceChildren(); e.MapReadout.textContent = 'Hover a cell to inspect its value.';
    e.MapMin.textContent = e.MapMax.textContent = '—';
    e.Offset.value = '0'; e.Index.value = '';
    this.#controls();
  }

  async show(session, snapshot, name) {
    this.reset(); this.#session = session; this.#snapshot = snapshot;
    const e = this.#elements, p = this.#pb;
    e.Title.textContent = `${name} · ${p.DataType[snapshot.dtype].replace('DATA_TYPE_', '')} [${snapshot.shape.join(', ')}]`;
    const reasons = new Map([
      [p.DebugTensorStatus.DEBUG_TENSOR_STATUS_STATISTICS_ONLY, 'Only statistics were saved. Recapture this tensor to see its values.'],
      [p.DebugTensorStatus.DEBUG_TENSOR_STATUS_OPTIMIZED_OUT, 'This tensor was optimized out. Recapture it with original node boundaries.'],
      [p.DebugTensorStatus.DEBUG_TENSOR_STATUS_BUDGET_EXCEEDED, 'This tensor did not fit the capture budget. Recapture only this tensor, or increase the budget.'],
      [p.DebugTensorStatus.DEBUG_TENSOR_STATUS_UNSUPPORTED, 'The backend did not expose values for this tensor.'],
      [p.DebugTensorStatus.DEBUG_TENSOR_STATUS_FAILED, 'Tensor capture failed. See the session error.'],
    ]);
    if (snapshot.status !== p.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE) {
      e.Message.textContent = reasons.get(snapshot.status) ?? 'No values were captured.';
      this.#controls(); return;
    }
    this.#layout = tensorLayout(snapshot, p);
    e.Index.placeholder = `[${this.#layout.shape.map(() => '0').join(', ')}]`;
    await this.#load(0n);
  }

  #controls() {
    const e = this.#elements, ready = !!this.#layout?.count && !this.#busy;
    for (const control of [e.Offset, e.Go, e.Index, e.Jump, e.Count, e.Shade, e.Mode]) control.disabled = !ready;
    e.First.disabled = e.Previous.disabled = !ready || this.#offset === 0n;
    e.Next.disabled = e.Last.disabled = !ready || this.#offset + BigInt(this.#rows.length) >= this.#layout.count;
    e.Csv.disabled = this.#busy || !this.#rows.length;
    const canRecapture = this.#snapshot && [this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_STATISTICS_ONLY,
      this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_OPTIMIZED_OUT,
      this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_BUDGET_EXCEEDED].includes(this.#snapshot.status);
    e.Recapture.hidden = e.RecaptureNote.hidden = !canRecapture;
    e.Recapture.disabled = !canRecapture || this.#busy;
  }

  async #load(offset, highlight = null) {
    const rows = await readTensorPage(this.#session, this.#snapshot, this.#pb, offset, Number(this.#pageSize()));
    this.#offset = offset; this.#rows = rows; this.#highlight = highlight;
    this.#elements.Message.textContent = 'Exact stored values. Dequantized = (stored − zero point) × scale. Colors and distribution describe this page only.';
    this.#elements.Message.dataset.error = 'false';
    this.#elements.Offset.value = String(offset);
    this.#elements.Range.textContent = rows.length ?
      `${offset.toLocaleString()}–${(offset + BigInt(rows.length) - 1n).toLocaleString()} / ${this.#layout.count.toLocaleString()}` : 'Empty tensor';
    this.#elements.Range.title = `${rows.length * this.#layout.width} bytes read. Indices are zero-based.`;
    this.#renderRows(); this.#controls();
  }

  #renderRows() {
    const e = this.#elements;
    e.Rows.replaceChildren();
    const summary = summarizeTensorPage(this.#rows), {min, max} = summary;
    e.Table.hidden = e.Mode.value === 'map' && !!this.#rows.length;
    e.MapPanel.hidden = !e.Table.hidden;
    e.Distribution.hidden = !this.#rows.length;
    for (const item of this.#rows) {
      const row = document.createElement('tr'); row.dataset.index = String(item.index);
      if (item.index === this.#highlight) row.className = 'tensor-value-highlight';
      for (const text of [String(item.index), `[${item.coordinates.join(', ')}]`, formatTensorValue(item.value),
        item.dequantized === undefined ? '—' : formatTensorValue(item.dequantized)]) {
        const cell = document.createElement('td'); cell.textContent = text; row.append(cell);
      }
      if (e.Shade.checked && Number.isFinite(item.value)) {
        const ratio = max > min ? (item.value - min) / (max - min) : 0.5;
        row.children[2].style.background = `hsla(${220 * (1 - ratio)}, 70%, 50%, .16)`;
      }
      if (!Number.isFinite(item.value)) row.children[2].className = 'tensor-value-nonfinite';
      e.Rows.append(row);
    }
    const container = e.Rows.closest('.debug-table-wrap');
    container.scrollTop = 0;
    const selected = e.Rows.querySelector('.tensor-value-highlight');
    if (selected) container.scrollTop += selected.getBoundingClientRect().top - container.getBoundingClientRect().top - 35;
    this.#renderPlots(summary);
  }

  #renderPlots({min, max, bins, finite, nonfinite}) {
    const e = this.#elements, canvas = e.Map, ctx = canvas.getContext('2d');
    const columns = Math.ceil(Math.sqrt(this.#rows.length)), cell = canvas.width / columns;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.#rows.forEach((item, index) => {
      const ratio = max > min ? (item.value - min) / (max - min) : 0.5;
      ctx.fillStyle = Number.isFinite(item.value) ? `hsl(${220 * (1 - ratio)} 70% 50%)` : '#df3ac4';
      ctx.fillRect(index % columns * cell, Math.floor(index / columns) * cell, cell, cell);
      if (item.index === this.#highlight) {
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
        ctx.strokeRect(index % columns * cell + 1, Math.floor(index / columns) * cell + 1, cell - 2, cell - 2);
      }
    });
    e.MapReadout.textContent = 'Hover a cell to inspect its value.';
    e.MapMin.textContent = min === undefined ? 'No finite values' : formatTensorValue(min);
    e.MapMax.textContent = max === undefined ? '—' : formatTensorValue(max);
    e.Histogram.replaceChildren();
    const peak = Math.max(1, ...bins);
    bins.forEach((count, index) => {
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      const height = count / peak * 54;
      for (const [key, value] of Object.entries({x: index * 320 / bins.length, y: 58 - height, width: 320 / bins.length - 2, height}))
        rect.setAttribute(key, value);
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `${count} values`; rect.append(title); e.Histogram.append(rect);
    });
    e.HistogramCaption.textContent = `Current page distribution · ${finite} finite${nonfinite ? ` · ${nonfinite} nonfinite excluded` : ''} · ${bins.length} bins`;
  }

  #downloadPage() {
    const rows = [['flat_index', 'coordinates', 'stored_value', 'dequantized_value'], ...this.#rows.map(item =>
      [String(item.index), `[${item.coordinates.join(', ')}]`, formatTensorValue(item.value),
        item.dequantized === undefined ? '' : formatTensorValue(item.dequantized)])];
    const csv = rows.map(row => row.map(value => `"${value.replaceAll('"', '""')}"`).join(',')).join('\r\n') + '\r\n';
    const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
    const link = document.createElement('a'); link.href = url;
    link.download = `receipt-debug-${this.#session.info.debugSessionId}-tensor-${this.#snapshot.snapshotId}-from-${this.#offset}.csv`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
