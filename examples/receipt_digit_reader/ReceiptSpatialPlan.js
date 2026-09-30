/** Receipt display coordinates, derived from graph operations, never from element count alone.
 * Grid centers describe sampling geometry, not causal influence or receptive-field bounds.
 * This deliberately small interpreter rejects transformations it cannot prove.
 */
const equal = (a, b) => a?.length === b?.length && a.every((v, i) => v === b[i]);
const concrete = shape => shape?.length && shape.every(n => Number.isSafeInteger(Number(n)) && Number(n) > 0) ? shape.map(Number) : null;
const geometry = d => JSON.stringify([d.axes, d.grid]);
const preservesGrid = new Set(['GroupNorm', 'Sigmoid', 'Relu', 'SiLU', 'Add', 'Sub', 'Mul', 'Div', 'QuantizeLinear', 'DequantizeLinear']);

export function receiptSpatialPlan(graph, plan, manifest) {
  const tensors = new Map(plan.tensors.map(t => [t.name, t]));
  const shapes = new Map(plan.tensors.map(t => [t.name, concrete(t.shape)]));
  const maps = new Map(), attention = [];
  const inputName = manifest.abi.input.name, input = shapes.get(inputName);
  if (!graph || !input || input.length !== 4 || input[0] !== 1 || input[1] !== 1 ||
      input[2] !== manifest.preprocess.height || input[3] !== manifest.preprocess.width)
    return {maps, attention, reason: 'This receipt viewer requires the declared single-image NCHW input.'};
  maps.set(inputName, {shape: input, axes: ['b', 'c', 'y', 'x'], kind: 'feature',
    grid: {height: input[2], width: input[3], originX: 0, originY: 0, stepX: 1, stepY: 1},
    inputWidth: input[3], inputHeight: input[2], globalMixing: false});
  for (const n of graph.nodes) {
    const outputs = Object.values(n.outputs);
    if (outputs.length !== 1) continue;
    const name = outputs[0].tensor, shape = shapes.get(name), p = n.params ?? {};
    if (!shape) continue;
    const inputs = Object.values(n.inputs), mapped = inputs.map(name => maps.get(name)).filter(Boolean);
    const source = maps.get(n.inputs.input), op = n.opType;
    let d;
    if (preservesGrid.has(op)) {
      const match = mapped.find(m => equal(m.shape, shape));
      if (match && mapped.every(m => equal(m.shape, shape) && geometry(m) === geometry(match)))
        d = {...match, globalMixing: mapped.some(m => m.globalMixing) || op === 'GroupNorm'};
    } else if ((op === 'Conv2D' || op === 'QConv2D') && source) {
      const axes = p.data_layout === 'NHWC' ? ['b', 'y', 'x', 'c'] : p.data_layout === 'NCHW' ? ['b', 'c', 'y', 'x'] : [];
      const w = shapes.get(n.inputs.weight), layout = p.weight_layout;
      const kh = w?.[layout === 'HWIO' ? 0 : layout === 'OHWI' ? 1 : 2];
      const kw = w?.[layout === 'HWIO' ? 1 : layout === 'OHWI' ? 2 : 3];
      const stride = p.stride ?? [1, 1], dilation = p.dilation ?? [1, 1];
      const pads = p.pads ?? (p.padding ? [...p.padding, ...p.padding] : [0, 0, 0, 0]);
      if (equal(source.axes, axes) && ['HWIO', 'OHWI', 'OIHW'].includes(layout) && kh > 0 && kw > 0 &&
          stride.length === 2 && stride.every(v => v > 0) && dilation.length === 2 && dilation.every(v => v > 0) && pads.length === 4) {
        const g = source.grid, height = shape[axes.indexOf('y')], width = shape[axes.indexOf('x')];
        if (height === Math.floor((g.height + pads[0] + pads[2] - dilation[0] * (kh - 1) - 1) / stride[0] + 1) &&
            width === Math.floor((g.width + pads[1] + pads[3] - dilation[1] * (kw - 1) - 1) / stride[1] + 1))
          d = {...source, axes, grid: {height, width,
            originY: g.originY + ((kh - 1) * dilation[0] / 2 - pads[0]) * g.stepY,
            originX: g.originX + ((kw - 1) * dilation[1] / 2 - pads[1]) * g.stepX,
            stepY: g.stepY * stride[0], stepX: g.stepX * stride[1]}};
      }
    } else if (op === 'Transpose' && source) {
      const perm = p.perm;
      if (perm?.length === source.axes.length && new Set(perm).size === perm.length &&
          perm.every(i => Number.isInteger(i) && i >= 0 && i < perm.length) && equal(shape, perm.map(i => source.shape[i])))
        d = {...source, axes: perm.map(i => source.axes[i])};
    } else if (op === 'Reshape' && source) {
      if (equal(shape, source.shape)) d = {...source};
      // The input's unit channel can move without changing row-major pixel order.
      else if (equal(source.axes, ['b', 'c', 'y', 'x']) && source.shape[1] === 1 &&
          equal(shape, [1, source.grid.height, source.grid.width, 1])) d = {...source, axes: ['b', 'y', 'x', 'c']};
      else {
        const y = source.axes.indexOf('y');
        if (y >= 0 && source.axes[y + 1] === 'x') {
          const flatShape = [...source.shape], axes = [...source.axes];
          flatShape.splice(y, 2, source.grid.height * source.grid.width); axes.splice(y, 2, 'p');
          if (equal(shape, flatShape)) d = {...source, axes};
        }
      }
    } else if (op === 'BatchMatMul') {
      const right = maps.get(n.inputs.b), leftShape = shapes.get(n.inputs.a);
      if (right && equal(right.axes, ['b', 'c', 'p']) && leftShape?.length === 3 &&
          leftShape[0] === 1 && leftShape[1] === manifest.decode.slots && leftShape[2] === right.shape[1] &&
          equal(shape, [1, leftShape[1], right.shape[2]]) && !p.transpose_a && !p.transpose_b)
        d = {...right, axes: ['b', 's', 'p'], kind: 'slot scores', globalMixing: true};
    } else if (op === 'Softmax' && source?.kind === 'slot scores') {
      const axis = (p.axis ?? -1) < 0 ? source.axes.length + (p.axis ?? -1) : p.axis;
      if (source.axes[axis] === 'p' && equal(shape, source.shape)) d = {...source, kind: 'attention'};
    }
    if (!d) continue;
    d.shape = shape; maps.set(name, d);
    if (d.kind === 'attention' && op === 'Softmax') {
      const step = plan.steps.find(s => s.sourceNodeIds.includes(n.id) && s.outputs.includes(tensors.get(name).tensorId));
      if (step) attention.push({name, step: step.scheduleIndex, tensorId: tensors.get(name).tensorId, descriptor: d});
    }
  }
  return {maps, attention, reason: 'No verified image-coordinate mapping for this tensor. Weights, pooled features and logits are not image maps.'};
}
