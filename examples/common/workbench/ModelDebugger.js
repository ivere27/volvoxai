/** Model-independent debugger UI. Session creation and model policy belong to its adapter. */
import {TensorValueView} from '../TensorValueView.js';
import {DebugGraphView} from '../DebugGraphView.js';
import {mountView, tabNavigation, formatBytes as bytes, element as node, download} from './View.js';
import {debugTemplate} from './DebugTemplate.js';
const number = value => value === undefined ? '—' : Number(value).toLocaleString(undefined, {maximumSignificantDigits: 6});

export class ModelDebugger {
  #pb; #source; #onBusy; #elements; #valueView; #graphView; #root; #createSession; #views = new Map();
  #session = null; #provenance = null; #available = false; #busy = false; #cancelling = false; #executing = false;
  #error = null; #breakpoints = new Set(); #issues = new Set(); #observations = new Map();
  #selectedStep = null; #selectedEntry = null; #valueSnapshot = null; #tab = 'graph';

  constructor({root, pb, getSource, createSession, tensorViews = [], onBusyChange = () => {}}) {
    this.#root = mountView(root, debugTemplate, ['Debugger.css']); this.#createSession = createSession;
    this.#pb = pb; this.#source = getSource; this.#onBusy = onBusyChange;
    this.#elements = Object.fromEntries(['Workspace', 'Start', 'Step', 'Continue', 'Cancel', 'Release', 'Export',
      'Graph', 'Values', 'Filter', 'Budget', 'Status', 'State', 'Progress', 'Memory', 'MemoryBar', 'Source',
      'Nodes', 'Tensors', 'Selection', 'NodeInfo', 'Preview', 'Download', 'Options', 'OptionsToggle',
      'Search', 'NodeFilter', 'NodeCount', 'NodesEmpty', 'ClearBreakpoints', 'BreakCount', 'Current', 'RunTo',
      'TensorSummary', 'ShowValues', 'SnapshotLabel', 'CaptureMode', 'GraphEmpty', 'GraphFit', 'GraphFocus',
      'ZoomIn', 'ZoomOut', 'ValuesMissing', 'CaptureSelected'].map(key => [key, this.#root.getElementById(`debug${key}`)]));
    const e = this.#elements;
    this.#valueView = new TensorValueView({root: this.#root.getElementById('tensorValueView'), pb,
      run: action => this.#act(action), recapture: snapshot => this.#recapture(snapshot)});
    for (const definition of tensorViews) {
      if (!/^[a-z][a-z0-9-]*$/.test(definition.id) || ['graph', 'values'].includes(definition.id) || this.#views.has(definition.id))
        throw new Error('Tensor views require unique lowercase IDs');
      const id = definition.id, suffix = id[0].toUpperCase() + id.slice(1);
      const button = node('button', '', definition.label), panel = node('section', 'debug-tab-panel debug-analysis-scroll');
      button.id = `debug${suffix}Tab`; button.dataset.debugTab = id; button.tabIndex = -1;
      button.setAttribute('role', 'tab'); button.setAttribute('aria-selected', 'false'); button.setAttribute('aria-controls', `debug${suffix}Panel`);
      panel.id = `debug${suffix}Panel`; panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', button.id); panel.hidden = true;
      this.#root.getElementById('debugAnalysisTabs').append(button); this.#root.getElementById('debugAnalysisPanels').append(panel);
      this.#views.set(id, definition.create({root: panel, pb, run: action => this.#act(action),
        recapture: () => this.#recaptureEntry(this.#selectedEntry, this.#selectedStep),
        capture: selection => this.#captureTensors(selection),
        select: (step, tensorId) => this.#selectStep(step, {tensorId, phase: 'output'})}));
    }
    this.#graphView = new DebugGraphView({root: this.#root.getElementById('debugGraphCanvas'),
      zoomLabel: this.#root.getElementById('debugGraphZoom'),
      onSelect: step => this.#act(() => this.#selectStep(step)),
      onTensor: (step, tensorId) => this.#act(() => this.#selectStep(step, {tensorId, phase: 'input'})),
      onBreakpoint: step => this.#toggleBreakpoint(step)});
    e.Start.onclick = () => this.#act(() => this.#start());
    e.Step.onclick = () => this.#act(() => this.#execute('step'));
    e.Continue.onclick = () => this.#act(() => this.#execute('continue', [...this.#breakpoints]));
    e.RunTo.onclick = () => this.#act(() => this.#execute('continue', [...new Set([...this.#breakpoints, this.#selectedStep])]));
    e.Current.onclick = () => this.#act(() => this.#selectStep(Math.min(this.#session.info.nextStep, this.#session.info.stepCount - 1)));
    e.Cancel.onclick = () => { void this.#cancel(); };
    e.Release.onclick = () => this.#act(() => this.reset());
    e.OptionsToggle.onclick = () => this.#showOptions(e.Options.hidden);
    e.Search.oninput = e.NodeFilter.onchange = () => this.#filterNodes();
    e.ClearBreakpoints.onclick = () => { this.#breakpoints.clear(); this.#renderExecution(); };
    e.GraphFit.onclick = () => this.#graphView.fit();
    e.GraphFocus.onclick = () => this.#graphView.focus(this.#selectedStep);
    e.ZoomIn.onclick = () => this.#graphView.zoom(1.25);
    e.ZoomOut.onclick = () => this.#graphView.zoom(0.8);
    e.CaptureSelected.onclick = () => this.#act(() => this.#recaptureEntry(this.#selectedEntry, this.#selectedStep));
    e.ShowValues.onclick = () => this.#act(() => this.#setTab('values'));
    e.Export.onclick = () => download(new Blob([JSON.stringify({format: 'volvoxai-debug-observations/v1',
      source: this.#provenance, tensorViews: Object.fromEntries([...this.#views].map(([id, view]) => [id, view.toJson()])), rawValuesIncluded: false, ...this.#session.toJson()}, null, 2)], {type: 'application/json'}),
      `volvoxai-debug-${this.#session.info.debugSessionId}.json`);
    e.Download.onclick = () => this.#act(async () => {
      const snapshot = this.#selectedEntry.snapshot;
      download(await this.#session.tensorBlob(snapshot), `volvoxai-debug-${this.#session.info.debugSessionId}-tensor-${snapshot.snapshotId}.bin`);
    });
    const tabs = e.Workspace.querySelectorAll('[data-debug-tab]');
    for (const button of tabs) button.onclick = () => this.#act(() => this.#setTab(button.dataset.debugTab));
    tabNavigation(tabs);
    const inspectorTabs = e.Workspace.querySelectorAll('[data-inspector-tab]');
    for (const button of inspectorTabs) button.onclick = () => {
      for (const tab of inspectorTabs) {
        const selected = tab === button; tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
        this.#root.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
      }
    };
    tabNavigation(inspectorTabs);
    for (const button of e.Workspace.querySelectorAll('[data-debug-pane]'))
      button.onclick = () => this.#setPane(button.dataset.debugPane);
    e.Workspace.addEventListener('keydown', event => this.#keyboard(event));
    for (const separator of e.Workspace.querySelectorAll('[data-resize]')) this.#resizePane(separator);
    this.#clearInspector(); this.#renderExecution();
  }

  setAvailable(available) { this.#available = available; this.#controls(); }

  async reset({preserveComparisons = false, clearCaptureSelection = false} = {}) {
    await this.#session?.release();
    this.#session = this.#provenance = null; this.#selectedStep = null; this.#error = null;
    this.#breakpoints.clear(); this.#issues.clear(); this.#observations.clear();
    this.#elements.Nodes.replaceChildren(); this.#elements.Search.value = ''; this.#elements.NodeFilter.value = 'all';
    if (clearCaptureSelection) this.#elements.Filter.value = '';
    for (const view of this.#views.values()) view.reset({keepPin: preserveComparisons});
    this.#graphView.clear(); this.#clearInspector(); this.#renderExecution();
  }

  #controls() {
    const e = this.#elements, state = this.#session?.info.state, busy = this.#busy || this.#cancelling || !this.#available;
    const paused = state === this.#pb.DebugState.DEBUG_STATE_PAUSED;
    e.Start.disabled = busy || !this.#available;
    e.Step.disabled = e.Continue.disabled = busy || !this.#available || !paused;
    e.RunTo.disabled = busy || !this.#available || !paused || this.#selectedStep <= this.#session?.info.nextStep;
    e.Current.disabled = busy || !this.#session;
    e.Cancel.disabled = !this.#available || this.#cancelling || !this.#session || (this.#busy && !this.#executing) ||
      (!paused && state !== this.#pb.DebugState.DEBUG_STATE_RUNNING);
    e.Release.disabled = e.Export.disabled = busy || !this.#session;
    e.Download.disabled = busy || this.#selectedEntry?.snapshot?.status !== this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE;
    e.ShowValues.disabled = e.CaptureSelected.disabled = busy || !this.#selectedEntry;
    e.ClearBreakpoints.disabled = busy || !this.#breakpoints.size;
    for (const control of [e.Graph, e.Values, e.Filter, e.Budget, ...e.Nodes.querySelectorAll('button'),
      ...e.Tensors.querySelectorAll('button'), ...e.Workspace.querySelectorAll('[data-debug-tab]')]) control.disabled = busy;
    this.#graphView.setDisabled(busy); this.#valueView.setBusy(busy); for (const view of this.#views.values()) view.setBusy(busy);
  }

  async #act(action) {
    if (this.#busy || this.#cancelling || !this.#available) return;
    this.#busy = true; this.#error = null; this.#onBusy(true); this.#controls();
    try { await action(); }
    catch (error) { this.#error = error.message; if (!this.#session) this.#showOptions(true); }
    finally { this.#busy = false; this.#onBusy(this.#cancelling); this.#renderStatus(); this.#controls(); }
  }

  async #cancel() {
    if (this.#elements.Cancel.disabled) return;
    this.#cancelling = true; this.#onBusy(true); this.#controls();
    try {
      await this.#session.cancel();
      // An in-flight execution owns its metadata refresh; idle cancellation does its own.
      if (!this.#executing) { await this.#session.refresh(); this.#indexObservations(); this.#renderExecution(); }
    } catch (error) { this.#error = error.message; }
    finally { this.#cancelling = false; this.#onBusy(this.#busy); this.#renderStatus(); this.#controls(); }
  }

  async #start(selection = {}) {
    const oldPlan = this.#session?.plan, breaks = [...this.#breakpoints];
    await this.reset({preserveComparisons: true});
    const source = this.#source(), e = this.#elements;
    if (!this.#available) throw new Error('Load a model and input first');
    e.Status.textContent = 'Preparing an independent execution of the current input…';
    const preserveNodeBoundaries = e.Graph.value === 'source';
    const tensorNames = e.Filter.value.split(',').map(value => value.trim()).filter(Boolean);
    const capture = {values: e.Values.checked, tensorNames, ...selection};
    this.#session = await this.#createSession({preserveNodeBoundaries, capture,
      limits: {maxBytes: BigInt(e.Budget.value) * 1048576n}});
    this.#provenance = {input: source.label, backend: source.backend, variant: source.variant,
      createdAt: new Date().toISOString(), preserveNodeBoundaries, tensorNames: capture.tensorNames, values: e.Values.checked, capture};
    for (const view of this.#views.values()) view.setContext({...source, session: this.#session});
    const signature = plan => JSON.stringify(plan.steps.map(step => [step.operatorName, step.sourceNodeIds]));
    if (oldPlan && signature(oldPlan) === signature(this.#session.plan)) this.#breakpoints = new Set(breaks);
    this.#buildNodes(); this.#graphView.setPlan(this.#session.plan); this.#showOptions(false);
    await this.#selectStep(0, {showPane: false});
  }

  async #execute(command, breaks = []) {
    this.#executing = true; this.#controls();
    this.#elements.Status.textContent = command === 'step' ? 'Executing one node…' : 'Running to the next breakpoint or end of graph…';
    this.#elements.State.textContent = 'RUNNING'; this.#elements.State.dataset.state = 'running';
    try {
      if (command === 'step') await this.#session.step();
      else await this.#session.continue({breakBeforeNodes: this.#nodeIds(breaks)});
      this.#indexObservations();
      const {info} = this.#session;
      const selected = command === 'continue' && info.state === this.#pb.DebugState.DEBUG_STATE_PAUSED ? info.nextStep : Math.max(0, info.nextStep - 1);
      await this.#selectStep(selected, {showPane: false});
    } finally { this.#executing = false; }
  }

  /** Breakpoints and capture selections name source nodes, not schedule indices. */
  #nodeIds(steps) {
    return steps.map(index => this.#session.plan.steps[index]?.sourceNodeIds[0]).filter(Boolean);
  }

  #indexObservations() {
    this.#observations.clear(); this.#issues.clear();
    for (const event of this.#session.events) for (const snapshot of event.snapshots) {
      const entries = this.#observations.get(event.step) ?? [];
      entries.push({snapshot, phase: event.point === this.#pb.DebugPoint.DEBUG_POINT_BEFORE ? 'input' : 'output'});
      this.#observations.set(event.step, entries);
      const stats = snapshot.statistics, s = this.#pb.DebugTensorStatus;
      if ([s.DEBUG_TENSOR_STATUS_BUDGET_EXCEEDED, s.DEBUG_TENSOR_STATUS_UNSUPPORTED, s.DEBUG_TENSOR_STATUS_FAILED].includes(snapshot.status) ||
          (stats && (stats.nanCount + stats.positiveInfinityCount + stats.negativeInfinityCount > 0n))) this.#issues.add(event.step);
    }
  }

  #buildNodes() {
    const e = this.#elements, {plan} = this.#session;
    e.Nodes.replaceChildren();
    for (const step of plan.steps) {
      const index = step.scheduleIndex, row = node('div', 'debug-node-row'); row.dataset.step = index;
      row.dataset.search = [index, step.operatorName, ...step.sourceNodeIds,
        ...[...step.inputs, ...step.outputs].map(id => plan.tensors[id].name)].join(' ').toLowerCase();
      const breakpoint = node('button', 'debug-breakpoint'); breakpoint.type = 'button'; breakpoint.tabIndex = -1;
      breakpoint.setAttribute('aria-label', `Toggle breakpoint before step ${index}`);
      breakpoint.title = `Break before step ${index} (F9)`; breakpoint.onclick = () => this.#toggleBreakpoint(index);
      const select = node('button', 'debug-node'); select.type = 'button'; select.tabIndex = -1;
      const top = node('span', 'debug-node-top');
      top.append(node('span', 'debug-node-number', index), node('span', 'debug-node-op', step.operatorName), node('span', 'debug-node-status'));
      select.append(top, node('span', 'debug-node-sub', `${step.sourceNodeIds.join(', ') || 'unmapped'}${step.skipped ? ' · skipped' : step.fused ? ' · fused' : ''}`));
      select.onclick = () => this.#act(() => this.#selectStep(index));
      row.append(breakpoint, select); e.Nodes.append(row);
    }
  }

  #toggleBreakpoint(step) {
    if (this.#busy || this.#cancelling || !this.#available || !this.#session) return;
    if (this.#breakpoints.has(step)) this.#breakpoints.delete(step); else this.#breakpoints.add(step);
    this.#renderExecution();
  }

  #filterNodes() {
    const e = this.#elements, query = e.Search.value.trim().toLowerCase(), mode = e.NodeFilter.value;
    const visible = [];
    for (const row of e.Nodes.children) {
      const step = Number(row.dataset.step);
      row.hidden = !row.dataset.search.includes(query) || (mode === 'breakpoints' && !this.#breakpoints.has(step)) || (mode === 'issues' && !this.#issues.has(step));
      row.querySelector('.debug-node').tabIndex = -1;
      if (!row.hidden) visible.push(row);
    }
    const active = visible.find(row => Number(row.dataset.step) === this.#selectedStep) ?? visible[0];
    if (active) active.querySelector('.debug-node').tabIndex = 0;
    e.NodeCount.textContent = this.#session ? `${visible.length} / ${this.#session.plan.steps.length}` : 'No plan';
    e.NodesEmpty.hidden = !!visible.length;
    e.NodesEmpty.textContent = this.#session ? 'No nodes match this filter.' : 'Start a session to explore the graph.';
  }

  #renderStatus() {
    const e = this.#elements, info = this.#session?.info;
    const state = info ? this.#pb.DebugState[info.state].replace('DEBUG_STATE_', '') : 'IDLE';
    e.State.textContent = state; e.State.dataset.state = state.toLowerCase();
    const failed = info?.state === this.#pb.DebugState.DEBUG_STATE_FAILED;
    e.Status.dataset.error = String(!!this.#error || failed);
    e.Status.textContent = this.#error || (info ? `${state} · ${info.nextStep}/${info.stepCount} steps · ` +
      (failed ? info.failure?.message ?? 'Model execution failed' : info.nextStep < info.stepCount ? `next: #${info.nextStep} ${this.#session.plan.steps[info.nextStep].operatorName}` : 'end of graph') :
      'Start a session to inspect the current input. No capture is active.');
    e.Start.textContent = info ? '↻ Restart' : 'Start session';
    e.Progress.max = info?.stepCount || 1; e.Progress.value = info?.nextStep ?? 0;
    e.Source.textContent = this.#provenance ? `${(this.#provenance.variant ?? 'model').toUpperCase()} / ${this.#provenance.backend} · ${this.#provenance.input}` : 'Current input · independent debug execution';
    e.Source.title = e.Source.textContent;
    e.CaptureMode.textContent = this.#provenance ? `${this.#provenance.values ? 'Values + statistics' : 'Statistics'} · ${this.#provenance.preserveNodeBoundaries ? 'original graph' : 'optimized graph'}` : 'Statistics · original graph';
    if (this.#provenance?.capture.nodeIds?.length) e.CaptureMode.textContent += ` · node ${this.#provenance.capture.nodeIds.join(', ')} only`;
    e.Memory.textContent = info ? `${bytes(info.retainedBytes)} / ${bytes(info.maxBytes)} capture · ${info.snapshots.count} snapshots${info.captureComplete ? '' : ' · incomplete'}` : 'Capture store: —';
    e.Memory.title = info ? `Capture peak ${bytes(info.peakBytes)}. Dropped events: ${info.events.dropped}; snapshots: ${info.snapshots.dropped}. Execution buffers and model weights are separate. Missing tensor values are reported in the inspector.` : '';
    e.MemoryBar.style.width = info ? `${Math.min(100, Number(info.retainedBytes) / Number(info.maxBytes) * 100)}%` : '0%';
    e.MemoryBar.dataset.loss = String(info ? !info.captureComplete : false);
    e.BreakCount.textContent = `${this.#breakpoints.size} breakpoint${this.#breakpoints.size === 1 ? '' : 's'}`;
  }

  #renderExecution() {
    const e = this.#elements, info = this.#session?.info;
    this.#renderStatus();
    e.GraphEmpty.hidden = !!info;
    for (const row of e.Nodes.children) {
      const step = Number(row.dataset.step), next = info && step === info.nextStep;
      row.classList.toggle('is-selected', step === this.#selectedStep);
      row.classList.toggle('is-next', !!next); row.classList.toggle('is-done', !!info && step < info.nextStep);
      row.classList.toggle('has-issue', this.#issues.has(step));
      row.querySelector('.debug-breakpoint').setAttribute('aria-pressed', String(this.#breakpoints.has(step)));
      row.querySelector('.debug-node').setAttribute('aria-current', next ? 'step' : 'false');
      row.querySelector('.debug-node-status').textContent = this.#issues.has(step) ? '!' : next ? 'next' : info && step < info.nextStep ? '✓' : '';
    }
    this.#filterNodes();
    this.#graphView.update({selected: this.#selectedStep, next: info?.nextStep ?? 0, breakpoints: this.#breakpoints,
      issues: this.#issues, completed: info?.nextStep === info?.stepCount});
    this.#controls();
  }

  #clearInspector() {
    const e = this.#elements;
    this.#selectedEntry = this.#valueSnapshot = null; this.#valueView.reset();
    this.#elements.ValuesMissing.hidden = false; this.#root.getElementById('tensorValueView').hidden = true;
    for (const view of this.#views.values()) view.clearSelection();
    e.Tensors.replaceChildren(); e.TensorSummary.replaceChildren(node('div', 'debug-empty', 'Choose a node, then an input or output tensor.'));
    e.Selection.textContent = 'Select a node'; e.NodeInfo.textContent = 'Inspect inputs, outputs and their captured values.';
    e.Preview.textContent = 'Select a tensor.'; e.SnapshotLabel.textContent = 'No tensor selected';
  }

  async #selectStep(stepIndex, {tensorId, phase, showPane = true} = {}) {
    const step = this.#session.plan.steps[stepIndex];
    if (!step) return;
    this.#selectedStep = stepIndex; this.#clearInspector();
    const e = this.#elements, observed = this.#observations.get(stepIndex) ?? [];
    e.Selection.textContent = `#${stepIndex} · ${step.operatorName}`;
    e.NodeInfo.textContent = `${step.sourceNodeIds.join(', ') || 'Unmapped source'} · ${step.skipped ? 'optimized / skipped' : step.fused ? 'fused entry' : 'original node'} · ${stepIndex < this.#session.info.nextStep ? 'executed' : 'not executed yet'}`;
    const entries = [];
    for (const [side, ids] of [['input', step.inputs], ['output', step.outputs]]) {
      for (const id of new Set(ids)) {
        const snapshot = observed.find(item => item.phase === side && item.snapshot.tensorId === id)?.snapshot;
        const entry = {phase: side, tensorId: id, tensor: this.#session.plan.tensors[id], snapshot}; entries.push(entry);
        const button = node('button', 'debug-tensor'); button.dataset.tensorId = id; button.dataset.phase = side;
        const title = node('span', 'debug-tensor-name'); title.append(node('span', '', entry.tensor.name), node('span', 'debug-tensor-side', side));
        const dtype = this.#pb.DataType[(snapshot ?? entry.tensor).dtype].replace('DATA_TYPE_', '');
        button.append(title, node('span', 'debug-tensor-meta', `${dtype} [${(snapshot ?? entry.tensor).shape.join(' × ')}] · ${this.#captureLabel(snapshot)}`));
        button.title = `${entry.tensor.name} · ${side} · ${this.#captureLabel(snapshot)}`;
        button.onclick = () => this.#act(() => this.#inspect(entry)); e.Tensors.append(button);
      }
    }
    this.#renderExecution(); this.#graphView.focus(stepIndex);
    const selected = e.Nodes.querySelector(`[data-step="${stepIndex}"]`);
    if (selected && !selected.hidden) {
      const y = selected.offsetTop - e.Nodes.offsetTop;
      if (y < e.Nodes.scrollTop || y + selected.offsetHeight > e.Nodes.scrollTop + e.Nodes.clientHeight)
        e.Nodes.scrollTop = y - e.Nodes.clientHeight / 3;
    }
    if (showPane) this.#setPane('analysis');
    const entry = entries.find(value => value.tensorId === tensorId && (!phase || value.phase === phase)) ??
      entries.find(value => value.phase === 'output' && value.snapshot) ?? entries.find(value => value.snapshot) ?? entries.find(value => value.phase === 'output') ?? entries[0];
    if (entry) await this.#inspect(entry);
  }

  #captureLabel(snapshot) {
    if (!snapshot) return this.#selectedStep < this.#session.info.nextStep ? 'Not captured' : 'Not executed';
    const s = this.#pb.DebugTensorStatus;
    return new Map([[s.DEBUG_TENSOR_STATUS_AVAILABLE, 'Values captured'], [s.DEBUG_TENSOR_STATUS_STATISTICS_ONLY, 'Statistics only'],
      [s.DEBUG_TENSOR_STATUS_OPTIMIZED_OUT, 'Optimized out'], [s.DEBUG_TENSOR_STATUS_BUDGET_EXCEEDED, 'Budget exceeded'],
      [s.DEBUG_TENSOR_STATUS_UNSUPPORTED, 'Unavailable'], [s.DEBUG_TENSOR_STATUS_FAILED, 'Capture failed']]).get(snapshot.status);
  }

  async #inspect(entry) {
    const e = this.#elements, {tensor, snapshot, phase, tensorId} = entry, data = snapshot ?? tensor;
    if (this.#selectedEntry !== entry) {
      this.#valueView.reset(); this.#valueSnapshot = null;
      for (const view of this.#views.values()) view.clearSelection();
      e.Workspace.querySelector('.debug-inspector-body').scrollTop = 0;
    }
    this.#selectedEntry = entry;
    for (const button of e.Tensors.children) button.setAttribute('aria-pressed', String(Number(button.dataset.tensorId) === tensorId && button.dataset.phase === phase));
    e.Preview.textContent = JSON.stringify({name: tensor.name, phase, ...data.toJson(),
      dtype: this.#pb.DataType[data.dtype], status: snapshot ? this.#pb.DebugTensorStatus[snapshot.status] : 'NOT_CAPTURED'}, null, 2);
    e.SnapshotLabel.textContent = snapshot ? `Snapshot ${snapshot.snapshotId} · ${phase === 'input' ? 'before' : 'after'} #${this.#selectedStep}` : `${phase} · plan metadata`;
    this.#renderTensorSummary();
    if (this.#tab === 'values') await this.#showValues();
    await this.#views.get(this.#tab)?.show(entry, this.#selectedStep);
    this.#controls();
  }

  #renderTensorSummary() {
    const e = this.#elements, {tensor, snapshot, phase} = this.#selectedEntry, data = snapshot ?? tensor;
    const identity = node('div', ''), badges = node('div', 'debug-tensor-identity');
    identity.append(node('h4', '', tensor.name));
    badges.append(node('span', 'debug-badge', this.#pb.DataType[data.dtype].replace('DATA_TYPE_', '')),
      node('span', 'debug-shape', `[${data.shape.join(' × ')}]`), node('span', 'debug-badge', bytes(data.logicalBytes)));
    identity.append(badges); e.TensorSummary.replaceChildren(identity);
    const status = this.#captureLabel(snapshot), note = node('p', 'debug-capture-note');
    note.textContent = `${status}. ` + (snapshot ? `${phase === 'input' ? 'Input before' : 'Output after'} step ${this.#selectedStep}. ` +
      (snapshot.status === this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_STATISTICS_ONLY ? 'Open Values to recapture just this tensor.' : 'Captured snapshots are immutable.') :
      this.#selectedStep < this.#session.info.nextStep ? 'This tensor was outside the capture selection or its record was dropped. Adjust capture settings and restart.' :
      'Run to this node, then Step to capture its inputs and outputs.');
    e.TensorSummary.append(note);
    const stats = snapshot?.statistics;
    if (stats) {
      const metrics = node('div', 'debug-metric-grid');
      for (const [label, value] of [['Minimum', stats.minimum], ['Maximum', stats.maximum], ['Mean', stats.mean],
        ['Std. deviation', stats.variance === undefined ? undefined : Math.sqrt(stats.variance)]]) {
        const metric = node('div', 'debug-metric'); metric.append(node('span', '', label), node('strong', '', number(value)));
        metric.title = value === undefined ? 'No finite elements' : String(value); metrics.append(metric);
      }
      const counts = node('div', 'debug-nonfinite');
      for (const [label, value] of [['Finite', stats.finiteCount], ['NaN', stats.nanCount], ['+∞', stats.positiveInfinityCount], ['−∞', stats.negativeInfinityCount]])
        counts.append(node('span', label !== 'Finite' && value > 0n ? 'has-issue' : '', `${label} ${value.toLocaleString()}`));
      e.TensorSummary.append(metrics, counts, node('p', 'debug-note', 'Statistics describe all captured stored values; quantized tensors use the integer storage domain.'));
    }
    if (snapshot?.quantization) {
      const q = snapshot.quantization;
      e.TensorSummary.append(node('p', 'debug-capture-note', q.perTensor ?
        `Affine quantization · scale ${q.perTensor.scale} · zero point ${q.perTensor.zeroPoint}. Values shows stored and dequantized numbers side by side.` :
        `Per-axis affine quantization · axis ${q.perAxis.axis} · ${q.perAxis.scales.length} scales. See Metadata for the complete parameters.`));
    }
    const connections = node('div', 'debug-connections'), steps = this.#session.plan.steps;
    let neighbors;
    if (phase === 'input') neighbors = steps.filter(step => step.scheduleIndex < this.#selectedStep && step.outputs.includes(tensor.tensorId)).slice(-1);
    else neighbors = steps.filter(step => step.scheduleIndex > this.#selectedStep && step.inputs.includes(tensor.tensorId));
    connections.append(node('span', '', phase === 'input' ? 'Produced by' : 'Consumed by'));
    if (!neighbors.length) connections.append(node('span', '', phase === 'input' ? 'Model / input binding' : 'No downstream consumer in this plan'));
    for (const step of neighbors) {
      const button = node('button', 'secondary', `#${step.scheduleIndex} ${step.operatorName}`);
      button.onclick = () => this.#act(() => this.#selectStep(step.scheduleIndex)); connections.append(button);
    }
    e.TensorSummary.append(connections);
  }

  async #showValues() {
    const entry = this.#selectedEntry;
    this.#elements.ValuesMissing.hidden = !!entry?.snapshot;
    this.#root.getElementById('tensorValueView').hidden = !entry?.snapshot;
    if (!entry?.snapshot) return;
    if (entry.snapshot.snapshotId === this.#valueSnapshot) return;
    await this.#valueView.show(this.#session, entry.snapshot, entry.tensor.name);
    this.#valueSnapshot = entry.snapshot.snapshotId;
  }

  async #setTab(tab) {
    if (tab !== this.#tab) this.#elements.Workspace.querySelector('.debug-inspector-body').scrollTop = 0;
    this.#tab = tab;
    for (const button of this.#elements.Workspace.querySelectorAll('[data-debug-tab]')) {
      const selected = button.dataset.debugTab === tab;
      button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
      this.#root.getElementById(button.getAttribute('aria-controls')).hidden = !selected;
    }
    if (tab === 'values') await this.#showValues();
    await this.#views.get(tab)?.show(this.#selectedEntry, this.#selectedStep);
    if (tab === 'graph') this.#graphView.focus(this.#selectedStep);
  }

  async #recapture(snapshot) {
    const event = this.#session.events.find(value => value.snapshots.some(item => item.snapshotId === snapshot.snapshotId));
    await this.#recaptureEntry({tensor: this.#session.plan.tensors[snapshot.tensorId], snapshot,
      phase: event.point === this.#pb.DebugPoint.DEBUG_POINT_BEFORE ? 'input' : 'output'}, event.step);
  }

  async #recaptureEntry(entry, stepIndex) {
    const {name} = entry.tensor, e = this.#elements, before = entry.phase === 'input', tab = this.#tab;
    const original = this.#provenance.preserveNodeBoundaries || this.#views.has(tab) ||
      entry.snapshot?.status === this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_OPTIMIZED_OUT;
    let nodeId = this.#session.plan.steps[stepIndex].sourceNodeIds[0];
    if (original && !this.#provenance.preserveNodeBoundaries) {
      const ids = this.#session.plan.steps[stepIndex].sourceNodeIds;
      stepIndex = this.#source().graph?.nodes.findIndex(node => ids.includes(node.id) &&
        (before ? Object.values(node.inputs).includes(name) : Object.values(node.outputs).some(output => output.tensor === name)));
      if (!(stepIndex >= 0)) throw new Error('Cannot locate this tensor in the original graph. Select an original-graph node first.');
      nodeId = this.#source().graph.nodes[stepIndex].id;
    }
    e.Graph.value = original ? 'source' : 'optimized';
    e.Values.checked = true; e.Filter.value = name;
    await this.#start({nodeIds: [nodeId], tensorNames: [name], inputs: before, outputs: !before});
    const next = stepIndex + 1;
    await this.#execute('continue', next < this.#session.info.stepCount ? [next] : []);
    const tensorId = this.#session.plan.tensors.find(tensor => tensor.name === name).tensorId;
    await this.#selectStep(stepIndex, {tensorId, phase: before ? 'input' : 'output'});
    await this.#setTab(tab);
  }

  async #captureTensors({tensorNames, targetName = tensorNames.at(-1)}) {
    if (!tensorNames.length) return;
    const e = this.#elements, tab = this.#tab;
    e.Graph.value = 'source'; e.Values.checked = true; e.Filter.value = tensorNames.join(', ');
    await this.#start({tensorNames, inputs: false, outputs: true});
    await this.#execute('continue');
    await this.#selectTensor(targetName); await this.#setTab(tab);
  }

  async #selectTensor(name) {
    const tensor = this.#session.plan.tensors.find(t => t.name === name);
    const step = tensor && this.#session.plan.steps.find(s => s.outputs.includes(tensor.tensorId));
    if (!step) throw new Error(`Tensor ${name} is absent from this execution plan`);
    await this.#selectStep(step.scheduleIndex, {tensorId: tensor.tensorId, phase: 'output'});
  }

  async inspectTensor(name) {
    return this.#act(async () => {
      if (!this.#session) await this.#start();
      await this.#selectTensor(name);
    });
  }

  async dispose() { await this.reset(); this.#graphView.dispose(); this.#root.replaceChildren(); }

  #showOptions(show) {
    this.#elements.Options.hidden = !show; this.#elements.OptionsToggle.setAttribute('aria-expanded', String(show));
  }

  #setPane(pane) {
    this.#elements.Workspace.dataset.pane = pane;
    for (const button of this.#elements.Workspace.querySelectorAll('[data-debug-pane]'))
      button.setAttribute('aria-pressed', String(button.dataset.debugPane === pane));
    if (pane === 'analysis' && this.#tab === 'graph') this.#graphView.focus(this.#selectedStep);
  }

  #keyboard(event) {
    if (event.defaultPrevented) return;
    const editing = event.target.matches('input, textarea, select, [contenteditable="true"]');
    if ((event.key === '/' && !editing) || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'f')) {
      event.preventDefault(); this.#setPane('execution'); this.#elements.Search.focus(); return;
    }
    if (editing || event.ctrlKey || event.metaKey || event.altKey) return;
    const actions = {F8: this.#elements.Continue, F10: this.#elements.Step};
    if (actions[event.key]) { event.preventDefault(); actions[event.key].click(); }
    else if (event.key === 'F9' && this.#selectedStep !== null) { event.preventDefault(); this.#toggleBreakpoint(this.#selectedStep); }
    else if (['ArrowUp', 'ArrowDown'].includes(event.key) && event.target.closest('#debugNodes')) {
      event.preventDefault();
      const rows = [...this.#elements.Nodes.children].filter(row => !row.hidden), index = rows.findIndex(row => Number(row.dataset.step) === this.#selectedStep);
      const row = rows[Math.max(0, Math.min(rows.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))];
      if (row) { row.querySelector('.debug-node').focus(); void this.#act(() => this.#selectStep(Number(row.dataset.step), {showPane: false})); }
    }
  }

  #resizePane(separator) {
    const workspace = this.#elements.Workspace, side = separator.dataset.resize;
    const property = side === 'outline' ? '--debug-outline-width' : '--debug-inspector-width';
    const pane = workspace.querySelector(side === 'outline' ? '.debug-outline' : '.debug-inspector');
    let start;
    const resize = width => {
      const other = workspace.querySelector(side === 'outline' ? '.debug-inspector' : '.debug-outline').getBoundingClientRect().width;
      const min = side === 'outline' ? 180 : 300, max = workspace.clientWidth - other - 270;
      width = Math.max(min, Math.min(max, width)); workspace.style.setProperty(property, `${width}px`);
      separator.setAttribute('aria-valuenow', Math.round(width)); separator.setAttribute('aria-valuemin', min); separator.setAttribute('aria-valuemax', Math.round(max));
    };
    separator.onpointerdown = event => { start = {x: event.clientX, width: pane.getBoundingClientRect().width}; separator.setPointerCapture(event.pointerId); event.preventDefault(); };
    separator.onpointermove = event => { if (start) resize(start.width + (event.clientX - start.x) * (side === 'outline' ? 1 : -1)); };
    separator.onpointerup = separator.onpointercancel = () => { start = null; };
    separator.onkeydown = event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault(); resize(pane.getBoundingClientRect().width + (event.key === 'ArrowRight' ? 20 : -20) * (side === 'outline' ? 1 : -1));
    };
  }
}
