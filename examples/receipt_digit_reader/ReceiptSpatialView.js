import {receiptSpatialPlan} from './ReceiptSpatialPlan.js';
import {readSpatialPlane, spatialCell, spatialCoordinates, spatialColor} from '../common/SpatialTensor.js';
import {formatTensorValue} from '../common/TensorValues.js';

const size = bytes => `${(bytes / 1024).toLocaleString(undefined, {maximumFractionDigits: 1})} KiB`;
const brief = value => value === null ? 'no finite values' : Number(value).toPrecision(5);
function option(value, label) { const item = document.createElement('option'); item.value = value; item.textContent = label; return item; }

/** Lazy presentation of immutable snapshots. At most a current plane and one pinned plane. */
export class ReceiptSpatialView {
  #pb; #e; #context = null; #mapping = null; #entry = null; #descriptor = null; #step = null;
  #plane = null; #pin = null; #image = null; #busy = false; #range = null; #slot = 0;

  constructor({root, pb, run, recapture, captureAttention, selectAttention}) {
    this.#pb = pb;
    root.innerHTML = `
      <link rel="stylesheet" href="${new URL('./ReceiptSpatial.css', import.meta.url)}">
      <div class="spatial-workspace">
      <aside class="spatial-settings">
      <div id="spatialDisplayControls" class="spatial-display-controls">
        <div class="spatial-row"><label>Display<select id="spatialMode"><option value="overlay">Receipt overlay</option><option value="map">Feature map</option><option value="image">Input image</option></select></label>
          <label class="spatial-opacity">Opacity <input id="spatialOpacity" type="range" min="0" max="1" step="0.05" value="0.65"></label></div>
      </div>
      <div class="spatial-attention">
        <strong>Digit attention</strong>
        <div class="spatial-row"><select id="spatialStage" aria-label="Attention pass"></select><button id="spatialAttention" class="secondary">Capture attention</button></div>
        <p class="debug-note">Capture starts a new session with only attention outputs.</p>
      </div>
      <p id="spatialMessage" class="debug-capture-note" role="status"></p>
      <button id="spatialRecapture" class="secondary" hidden>Capture this tensor · new session</button>
      <div id="spatialControls" hidden>
        <div class="spatial-row"><label><span id="spatialSelectorLabel">Channel</span><select id="spatialSelector"></select></label>
          <label>Value domain<select id="spatialDomain"><option value="stored">Stored values</option><option value="dequantized">Dequantized</option></select></label></div>
      </div>
      <div id="spatialComparisonControls" hidden>
        <div class="spatial-row"><label class="debug-checkbox"><input id="spatialLock" type="checkbox"> Lock color scale</label><button id="spatialPin" class="secondary">Pin for comparison</button><button id="spatialExport" class="secondary">Map JSON ↓</button></div>
      </div>
      </aside>
      <div class="spatial-images">
      <div id="spatialEmpty" class="empty"><h3>See features on the input</h3><p>Capture a selected image tensor, or capture attention to explore digit slots. The visualization appears here.</p></div>
      <figure id="spatialCurrent" class="spatial-figure" hidden><figcaption id="spatialTitle"></figcaption><canvas id="spatialCanvas" tabindex="0" aria-label="Spatial tensor map. Arrow keys move between grid cells."></canvas>
        <div class="spatial-legend"><span id="spatialMin"></span><span class="spatial-gradient"></span><span id="spatialMax"></span></div></figure>
      <p id="spatialReadout" class="spatial-readout" aria-live="polite">Hover a map or use its arrow keys to inspect coordinates and values.</p>
      <figure id="spatialPinned" class="spatial-figure" hidden><figcaption><span id="spatialPinnedTitle"></span><button id="spatialUnpin" class="secondary">Unpin</button></figcaption><canvas id="spatialPinnedCanvas" tabindex="0" aria-label="Pinned spatial tensor map. Arrow keys move between grid cells."></canvas><p id="spatialPinnedScale" class="debug-note"></p></figure>
      <details class="spatial-explanation"><summary>Map geometry &amp; interpretation</summary><p id="spatialGeometry" class="debug-note"></p><p id="spatialMemory" class="debug-note"></p><p class="debug-note">Colors show activations or attention weights, not causal attribution. Overlays use nearest grid centers in the displayed input image. GroupNorm and attention can depend on the whole image; a cell is not a receptive-field boundary.</p></details>
      </div></div>`;
    this.#e = Object.fromEntries(['Stage', 'Attention', 'Message', 'Recapture', 'Controls', 'ComparisonControls', 'SelectorLabel', 'Selector', 'Domain', 'Mode',
      'Opacity', 'Lock', 'Pin', 'Export', 'Current', 'Title', 'Canvas', 'Min', 'Max', 'Readout', 'Pinned', 'PinnedTitle', 'Unpin',
      'PinnedCanvas', 'PinnedScale', 'Geometry', 'Memory', 'Empty'].map(key => [key, root.querySelector(`#spatial${key}`)]));
    const e = this.#e;
    e.Recapture.onclick = () => run(() => recapture());
    e.Attention.onclick = () => run(() => captureAttention(this.attention));
    e.Stage.onchange = () => { if (e.Stage.value !== '') run(() => selectAttention(this.attention[Number(e.Stage.value)])); };
    e.Selector.onchange = () => run(async () => { if (this.#descriptor.axes.includes('s')) this.#slot = Number(e.Selector.value); await this.#load(); });
    e.Domain.onchange = () => run(() => this.#load());
    e.Mode.onchange = e.Opacity.oninput = () => this.#render();
    e.Lock.onchange = () => { this.#range = e.Lock.checked && this.#plane ? [this.#plane.min, this.#plane.max] : null; this.#render(); };
    e.Pin.onclick = () => { this.#pin = this.#plane; this.#render(); this.#controls(); };
    e.Unpin.onclick = () => { this.#pin = null; this.#render(); this.#controls(); };
    e.Export.onclick = () => {
      const blob = new Blob([JSON.stringify({...this.toJson(), values: Array.from(this.#plane.values, formatTensorValue)}, null, 2)], {type: 'application/json'});
      const link = document.createElement('a'), url = URL.createObjectURL(blob);
      link.href = url; link.download = `receipt-spatial-${this.#plane.name}-${this.#plane.index}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    for (const [canvas, getPlane] of [[e.Canvas, () => this.#plane], [e.PinnedCanvas, () => this.#pin]]) {
      let position = {x: 0, y: 0};
      canvas.onpointermove = event => {
        const plane = getPlane(); if (!plane || !this.#image) return;
        const bounds = canvas.getBoundingClientRect();
        const x = Math.floor((event.clientX - bounds.left) / bounds.width * canvas.width), y = Math.floor((event.clientY - bounds.top) / bounds.height * canvas.height);
        position = e.Mode.value === 'map' ? {x, y} : spatialCell(plane.descriptor, x, y, canvas.width, canvas.height);
        this.#readout(plane, position);
      };
      canvas.onkeydown = event => {
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
        const plane = getPlane(); if (!plane) return;
        event.preventDefault();
        position.x += event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : 0;
        position.y += event.key === 'ArrowUp' ? -1 : event.key === 'ArrowDown' ? 1 : 0;
        const g = plane.descriptor.grid;
        position = {x: Math.max(0, Math.min(g.width - 1, position.x)), y: Math.max(0, Math.min(g.height - 1, position.y))};
        this.#readout(plane, position);
      };
    }
    this.reset();
  }

  get attention() { this.#prepareMapping(); return this.#mapping?.attention ?? []; }
  setBusy(busy) { this.#busy = busy; this.#controls(); }
  setContext(context) { this.#context = context; }

  reset({keepPin = false} = {}) {
    this.#context = this.#mapping = null;
    if (!keepPin || !this.#pin) this.#image = null;
    if (!keepPin) { this.#pin = this.#range = null; this.#e.Lock.checked = false; this.#slot = 0; }
    this.#e.Stage.replaceChildren(option('', 'Attention pass'));
    this.clearSelection();
  }

  clearSelection() {
    this.#entry = this.#descriptor = this.#plane = null; this.#step = null;
    this.#e.Message.textContent = this.#context ? 'Select a tensor to explore its image coordinates.' :
      'Start a debug session, then capture a tensor or use Capture attention to view the receipt overlay.';
    this.#e.Message.dataset.error = 'false';
    this.#e.Readout.textContent = 'Hover a map or use its arrow keys to inspect coordinates and values.';
    this.#e.Geometry.textContent = ''; this.#render(); this.#controls();
  }

  #prepareMapping() {
    if (!this.#mapping && this.#context) {
      const {graph, session, manifest} = this.#context;
      this.#mapping = receiptSpatialPlan(graph, session.plan, manifest);
      this.#e.Stage.replaceChildren(option('', 'Jump to attention pass…'), ...this.#mapping.attention.map((a, i) => option(i, `Pass ${i + 1} · #${a.step} ${a.name}`)));
    }
  }

  async show(entry, step) {
    this.#prepareMapping();
    if (!entry || !this.#context || this.#entry === entry) return;
    this.#entry = entry; this.#step = step; this.#plane = null;
    const e = this.#e, descriptor = this.#mapping.maps.get(entry.tensor.name);
    this.#descriptor = descriptor;
    const stage = this.attention.findIndex(a => a.name === entry.tensor.name);
    e.Stage.value = stage < 0 ? '' : String(stage);
    if (!descriptor) { e.Message.textContent = this.#mapping.reason; this.#render(); this.#controls(); return; }
    const slots = descriptor.axes.includes('s'), count = descriptor.shape[descriptor.axes.indexOf(slots ? 's' : 'c')];
    e.SelectorLabel.textContent = slots ? 'Digit slot' : 'Channel';
    const phone = this.#context.manifest.decode.phone_slots;
    e.Selector.replaceChildren(...Array.from({length: count}, (_, i) => option(i, slots ? `${i < phone ? `Phone ${i + 1}` : `Street ${i - phone + 1}`} · slot ${i}` : `Channel ${i}`)));
    if (!slots) e.Selector.append(option('mean-abs', 'Mean |activation| · all channels'));
    e.Selector.value = String(slots ? Math.min(this.#slot, count - 1) : 0);
    e.Domain.value = entry.snapshot?.quantization ? 'dequantized' : 'stored';
    e.Domain.querySelector('[value="dequantized"]').disabled = !entry.snapshot?.quantization;
    if (entry.snapshot?.status === this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE) await this.#load();
    else {
      e.Message.textContent = 'This tensor has image coordinates but no retained raw values. Capture it in a new session to draw the map. Check the capture budget if values were dropped.';
      this.#render(); this.#controls();
    }
  }

  async #load() {
    const e = this.#e;
    this.#plane = null; this.#render(); e.Message.dataset.error = 'false';
    e.Message.textContent = 'Reading the selected map in chunks of at most 64 KiB…';
    try {
      const mean = e.Selector.value === 'mean-abs';
      const plane = await readSpatialPlane(this.#context.session, this.#entry.snapshot, this.#pb, this.#descriptor,
        {index: mean ? 0 : Number(e.Selector.value), reduction: mean ? 'mean-abs' : 'channel', domain: e.Domain.value});
      this.#image ??= this.#context.getImage();
      plane.name = this.#entry.tensor.name; plane.step = this.#step; plane.phase = this.#entry.phase;
      plane.dtype = this.#pb.DataType[this.#entry.snapshot.dtype].replace('DATA_TYPE_', '');
      plane.quantization = this.#entry.snapshot.quantization?.toJson();
      plane.sessionId = String(this.#context.session.info.debugSessionId); plane.snapshotId = String(this.#entry.snapshot.snapshotId);
      plane.source = this.#context.label; plane.selector = e.Selector.selectedOptions[0].textContent;
      this.#plane = plane;
      if (e.Lock.checked && !this.#range) this.#range = [plane.min, plane.max];
      e.Message.textContent = `${this.#descriptor.kind === 'attention' ? 'Attention weights across spatial positions' : 'Captured activations'} · ${plane.domain} · ${plane.nonfinite} nonfinite cells (magenta).`;
    } catch (error) { e.Message.textContent = error.message; e.Message.dataset.error = 'true'; }
    this.#render(); this.#controls();
  }

  #controls() {
    const e = this.#e, available = this.#entry?.snapshot?.status === this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE;
    e.Attention.disabled = this.#busy || !this.#mapping?.attention.length;
    e.Stage.disabled = this.#busy || !this.#mapping?.attention.length;
    e.Recapture.hidden = !this.#descriptor || !!available; e.Recapture.disabled = this.#busy || !this.#entry;
    e.Controls.hidden = e.ComparisonControls.hidden = !this.#plane;
    for (const key of ['Selector', 'Domain', 'Mode', 'Opacity', 'Lock', 'Pin', 'Export']) e[key].disabled = this.#busy || !this.#plane;
    e.Lock.disabled ||= this.#plane?.min === null;
    e.Unpin.disabled = this.#busy;
  }

  #label(plane) { return `#${plane.step} ${plane.name} · ${plane.phase} · ${plane.selector} · ${plane.domain}`; }
  #scale(plane) { return this.#range ?? [plane.min, plane.max]; }

  #render() {
    const e = this.#e, plane = this.#plane, pin = this.#pin;
    e.Empty.hidden = !!plane;
    e.Current.hidden = !plane || !this.#image; e.Pinned.hidden = !pin || !this.#image;
    if (!plane) e.Canvas.width = e.Canvas.height = 1;
    if (!pin) e.PinnedCanvas.width = e.PinnedCanvas.height = 1;
    if (plane) {
      e.Title.textContent = this.#label(plane); e.Title.title = `${plane.source} · session ${plane.sessionId} · snapshot ${plane.snapshotId}`; this.#draw(e.Canvas, plane);
      const [min, max] = this.#scale(plane); e.Min.textContent = brief(min); e.Max.textContent = brief(max);
      const g = plane.descriptor.grid;
      e.Geometry.textContent = `${g.width} × ${g.height} grid · axes ${plane.descriptor.axes.join(', ')} · input centers x=${g.originX}+${g.stepX}·col, y=${g.originY}+${g.stepY}·row. ${plane.descriptor.globalMixing ? 'Includes operations with whole-image dependencies.' : ''}`;
    }
    if (pin && this.#image) {
      e.PinnedTitle.textContent = `Pinned · ${this.#label(pin)}`; e.PinnedTitle.title = `${pin.source} · session ${pin.sessionId} · snapshot ${pin.snapshotId}`; this.#draw(e.PinnedCanvas, pin);
      e.PinnedScale.textContent = `${this.#range ? 'Shared locked' : 'Independent auto'} scale: ${this.#scale(pin).map(brief).join(' → ')}. ${this.#range ? 'Compare only maps with compatible units.' : 'Lock the color scale to compare magnitudes.'}`;
    }
    const retained = (plane?.values.byteLength ?? 0) + (pin && pin !== plane ? pin.values.byteLength : 0);
    e.Memory.textContent = `${plane ? `Read ${size(plane.bytesRead)}; ` : ''}map arrays ${size(retained)} (current + one pin). Image and canvas memory are separate. ${plane && plane.bytesRead > plane.values.byteLength ? 'Interleaved channels require scanning the enclosing tensor span.' : ''}`;
  }

  #draw(canvas, plane) {
    if (!this.#image) return;
    const e = this.#e, mapOnly = e.Mode.value === 'map', g = plane.descriptor.grid, image = this.#image;
    canvas.width = mapOnly ? g.width : image.width; canvas.height = mapOnly ? g.height : image.height;
    const ctx = canvas.getContext('2d'), frame = ctx.createImageData(canvas.width, canvas.height), [min, max] = this.#scale(plane);
    const alpha = mapOnly ? 1 : e.Mode.value === 'image' ? 0 : Number(e.Opacity.value);
    for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
      const cell = mapOnly ? {x, y} : spatialCell(plane.descriptor, x, y, image.width, image.height);
      const color = spatialColor(plane.values[cell.y * g.width + cell.x], min, max), offset = (y * canvas.width + x) * 4;
      for (let c = 0; c < 3; c++) frame.data[offset + c] = color[c] * alpha + (mapOnly ? 0 : image.data[offset + c] * (1 - alpha));
      frame.data[offset + 3] = 255;
    }
    ctx.putImageData(frame, 0, 0);
  }

  #readout(plane, {x, y}) {
    const g = plane.descriptor.grid;
    if (x < 0 || x >= g.width || y < 0 || y >= g.height) return;
    const coords = spatialCoordinates(plane, x, y);
    if (plane.reduction === 'mean-abs') coords[plane.descriptor.axes.indexOf('c')] = '*';
    this.#e.Readout.textContent = `${plane.name} [${coords.join(', ')}] = ${formatTensorValue(plane.values[y * g.width + x])} (${plane.domain}${plane.reduction === 'mean-abs' ? ', mean absolute' : ''}) · grid (${x}, ${y}) · input center (${g.originX + x * g.stepX}, ${g.originY + y * g.stepY})`;
  }

  toJson() {
    if (!this.#plane) return null;
    const {values, ...metadata} = this.#plane;
    return {format: 'volvoxai-receipt-spatial/v1', ...metadata, displayRange: this.#scale(this.#plane),
      valueEncoding: 'row-major y,x; decimal strings preserve nonfinite values and negative zero',
      interpretation: 'Activations or spatial attention weights; not causal attribution or receptive-field bounds.'};
  }
}
