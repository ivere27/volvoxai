const SVG = 'http://www.w3.org/2000/svg';
let nextGraphId = 0;
function element(tag, attributes = {}, text) {
  const value = document.createElementNS(SVG, tag);
  for (const [name, content] of Object.entries(attributes)) value.setAttribute(name, content);
  if (text !== undefined) value.textContent = text;
  return value;
}

/** Edges follow tensor producers, not adjacent entries in the execution order. */
export function layoutDebugGraph(plan) {
  const producers = new Map(), levels = new Map(), nodes = [], edges = [];
  for (const step of plan.steps) {
    const dependencies = step.inputs.filter(id => producers.has(id));
    const level = dependencies.reduce((depth, id) => Math.max(depth, producers.get(id).level + 1), 0);
    const peers = levels.get(level) ?? [];
    const node = {step, level, column: peers.length, x: 0, y: 36 + level * 112};
    peers.push(node); levels.set(level, peers); nodes.push(node);
    for (const tensorId of new Set(dependencies))
      edges.push({from: producers.get(tensorId), to: node, tensorId});
    for (const id of step.outputs) producers.set(id, node);
  }
  const columns = Math.max(1, ...[...levels.values()].map(items => items.length));
  for (const peers of levels.values())
    for (const node of peers) node.x = 36 + ((columns - peers.length) / 2 + node.column) * 232;
  return {nodes, edges, width: columns * 232 + 130, height: levels.size * 112 + 40};
}

/** SVG navigation only. No engine state is inferred from visual position. */
export class DebugGraphView {
  #root; #svg; #world; #layout; #onSelect; #onTensor; #onBreakpoint;
  #nodeElements = new Map(); #edgeElements = [];
  #selected = null; #scale = 1; #x = 0; #y = 0; #drag = null; #moved = false; #disabled = false;
  #zoomLabel; #observer; #markerId = `debugGraphArrow-${++nextGraphId}`;

  constructor({root, zoomLabel, onSelect, onTensor, onBreakpoint}) {
    this.#root = root; this.#zoomLabel = zoomLabel;
    this.#onSelect = onSelect; this.#onTensor = onTensor; this.#onBreakpoint = onBreakpoint;
    this.#svg = element('svg', {role: 'img', 'aria-label': 'Model graph. Select an operator or use the execution list.'});
    this.#world = element('g');
    const defs = element('defs'), marker = element('marker', {id: this.#markerId, viewBox: '0 0 10 10',
      refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse'});
    marker.append(element('path', {d: 'M 0 0 L 10 5 L 0 10 z', fill: 'context-stroke'}));
    defs.append(marker); this.#svg.append(defs, this.#world); root.append(this.#svg);
    this.#svg.addEventListener('wheel', event => {
      event.preventDefault();
      if (event.ctrlKey || event.metaKey) this.zoom(Math.exp(-event.deltaY * 0.008), event.clientX, event.clientY);
      else { this.#x -= event.deltaX; this.#y -= event.deltaY; this.#transform(); }
    }, {passive: false});
    this.#svg.onpointerdown = event => {
      if (event.button !== 0) return;
      this.#moved = false;
      if (event.target.closest('[data-step], [data-tensor]')) return;
      this.#drag = {x: event.clientX, y: event.clientY}; this.#moved = false;
      this.#svg.setPointerCapture(event.pointerId);
    };
    this.#svg.onpointermove = event => {
      if (!this.#drag) return;
      const x = event.clientX - this.#drag.x, y = event.clientY - this.#drag.y;
      if (Math.abs(x) + Math.abs(y) > 2) this.#moved = true;
      this.#x += x; this.#y += y; this.#drag = {x: event.clientX, y: event.clientY}; this.#transform();
    };
    this.#svg.onpointerup = event => {
      this.#drag = null;
      if (this.#svg.hasPointerCapture(event.pointerId)) this.#svg.releasePointerCapture(event.pointerId);
    };
    this.#svg.onpointercancel = () => { this.#drag = null; this.#moved = false; };
    this.#observer = new ResizeObserver(() => { if (this.#layout && this.#selected !== null) this.focus(this.#selected, false); });
    this.#observer.observe(root);
    this.clear();
  }

  clear() {
    this.#layout = null; this.#selected = null; this.#world.replaceChildren();
    this.#nodeElements.clear(); this.#edgeElements = [];
    this.#scale = 1; this.#x = this.#y = 0; this.#transform();
  }

  dispose() { this.#observer.disconnect(); this.clear(); this.#svg.remove(); }

  setPlan(plan) {
    this.clear(); this.#layout = layoutDebugGraph(plan);
    for (const edge of this.#layout.edges) {
      const {from, to} = edge;
      const x1 = from.x + 96, y1 = from.y + 68, x2 = to.x + 96, y2 = to.y;
      const lane = Math.max(from.x, to.x) + 260 + edge.tensorId % 3 * 18;
      const route = to.level - from.level > 1 ?
        `M${from.x + 192},${from.y + 34} C${lane},${from.y + 34} ${lane},${to.y + 34} ${to.x + 192},${to.y + 34}` :
        `M${x1},${y1} C${x1},${y1 + 24} ${x2},${y2 - 24} ${x2},${y2}`;
      const path = element('path', {d: route,
        class: 'debug-graph-edge', 'marker-end': `url(#${this.#markerId})`, 'data-tensor': edge.tensorId});
      const tensor = plan.tensors[edge.tensorId];
      path.append(element('title', {}, `${tensor.name} [${tensor.shape.join(', ')}] → step ${to.step.scheduleIndex}`));
      path.onclick = event => {
        event.stopPropagation();
        if (!this.#disabled && !this.#moved) this.#onTensor(to.step.scheduleIndex, edge.tensorId);
      };
      this.#world.append(path); this.#edgeElements.push({edge, path});
    }
    for (const node of this.#layout.nodes) {
      const {step, x, y} = node, id = step.scheduleIndex;
      const group = element('g', {transform: `translate(${x} ${y})`, class: 'debug-graph-node', 'data-step': id});
      group.append(element('rect', {width: 192, height: 68, rx: 9}));
      group.append(element('text', {x: 14, y: 25, class: 'debug-graph-op'}, step.operatorName));
      group.append(element('text', {x: 14, y: 47, class: 'debug-graph-source'},
        `#${id} · ${(step.sourceNodeIds.join(', ') || 'unmapped').slice(0, 24)}`));
      const dot = element('circle', {cx: 179, cy: 12, r: 4, class: 'debug-graph-breakpoint'});
      group.append(dot, element('title', {}, `${step.operatorName} · step ${id}\n${step.sourceNodeIds.join(', ')}\nDouble-click to toggle a breakpoint.`));
      group.onclick = () => { if (!this.#disabled && !this.#moved) this.#onSelect(id); };
      group.ondblclick = event => { event.preventDefault(); if (!this.#disabled) this.#onBreakpoint(id); };
      group.classList.toggle('is-skipped', step.skipped);
      this.#world.append(group); this.#nodeElements.set(id, group);
    }
    this.focus(plan.steps[0]?.scheduleIndex ?? 0);
  }

  setDisabled(disabled) { this.#disabled = disabled; }

  update({selected, next, breakpoints, issues, completed}) {
    this.#selected = selected;
    for (const [id, group] of this.#nodeElements) {
      group.classList.toggle('is-selected', id === selected);
      group.classList.toggle('is-next', id === next && !completed);
      group.classList.toggle('is-done', id < next);
      group.classList.toggle('has-breakpoint', breakpoints.has(id));
      group.classList.toggle('has-issue', issues.has(id));
    }
    for (const {edge, path} of this.#edgeElements)
      path.classList.toggle('is-connected', edge.from.step.scheduleIndex === selected || edge.to.step.scheduleIndex === selected);
  }

  focus(step = this.#selected, resetZoom = true) {
    const node = this.#layout?.nodes.find(item => item.step.scheduleIndex === step);
    if (!node || !this.#root.clientWidth || !this.#root.clientHeight) return;
    if (resetZoom) this.#scale = 1;
    this.#x = this.#root.clientWidth / 2 - (node.x + 96) * this.#scale;
    this.#y = Math.min(30, this.#root.clientHeight * 0.42 - (node.y + 34) * this.#scale);
    this.#transform();
  }

  fit() {
    if (!this.#layout) return;
    this.#scale = Math.min(1, this.#root.clientWidth / this.#layout.width, this.#root.clientHeight / this.#layout.height) * 0.92;
    this.#x = (this.#root.clientWidth - this.#layout.width * this.#scale) / 2;
    this.#y = (this.#root.clientHeight - this.#layout.height * this.#scale) / 2;
    this.#transform();
  }

  zoom(factor, clientX, clientY) {
    if (!this.#layout) return;
    const bounds = this.#root.getBoundingClientRect();
    const x = clientX === undefined ? bounds.width / 2 : clientX - bounds.left;
    const y = clientY === undefined ? bounds.height / 2 : clientY - bounds.top;
    const next = Math.min(2.5, Math.max(0.025, this.#scale * factor));
    this.#x = x - (x - this.#x) * next / this.#scale;
    this.#y = y - (y - this.#y) * next / this.#scale;
    this.#scale = next; this.#transform();
  }

  #transform() {
    this.#world.setAttribute('transform', `translate(${this.#x} ${this.#y}) scale(${this.#scale})`);
    this.#zoomLabel.textContent = `${Math.round(this.#scale * 100)}%`;
  }
}
