/** Explicit execution contracts and requests used by parity fixtures. */
export function declareIncremental(graph, {slots = 1, updates = Object.keys(graph.inputs), generation} = {}) {
  graph.inputs.active_lengths = {dtype: 'int32', shape: [slots]};
  const inputs = Object.keys(graph.inputs), outputs = graph.outputs;
  graph.execution = {incremental: {active_lengths_input: 'active_lengths', sequence_axis: 1, update_inputs: updates},
    invariant_inputs: ['active_lengths'], entrypoints: [
      {name: 'evaluate', effect: 'EXECUTION_EFFECT_EVALUATE', allowed_ranges: ['EXECUTION_RANGE_KIND_FULL', 'EXECUTION_RANGE_KIND_PREFIX_ROWS'], inputs, outputs},
      {name: 'initialize', effect: 'EXECUTION_EFFECT_INITIALIZE', allowed_ranges: ['EXECUTION_RANGE_KIND_FULL'], inputs, outputs},
      {name: 'transition', effect: 'EXECUTION_EFFECT_TRANSITION', allowed_ranges: ['EXECUTION_RANGE_KIND_SLOT_ROWS', 'EXECUTION_RANGE_KIND_DEPENDENCIES'], inputs, outputs},
    ], ...(generation ? {generation} : {})};
  return graph;
}
export function declareStateless(graph) {
  graph.execution = {stateless: {}, invariant_inputs: [], entrypoints: [
    {name: 'evaluate', effect: 'EXECUTION_EFFECT_EVALUATE', allowed_ranges: ['EXECUTION_RANGE_KIND_FULL'],
      inputs: Object.keys(graph.inputs), outputs: graph.outputs},
  ]};
  return graph;
}
export function activeLengths(p, lengths) {
  return new p.Tensor({name: 'active_lengths', dtype: p.DataType.DATA_TYPE_I32,
    shape: [BigInt(lengths.length)], inline: new Uint8Array(Int32Array.from(lengths).buffer)});
}
export function graphRequest(p, graph, contextId, values, {entrypoint = 'evaluate', lengths, positions, roots, outputs} = {}) {
  values = values.filter(value => value.name !== 'active_lengths');
  if (entrypoint !== 'transition' && graph.inputs.active_lengths)
    values.push(activeLengths(p, lengths ?? Array(graph.inputs.active_lengths.shape[0]).fill(1)));
  const supplied = new Set(values.map(value => value.name));
  const reuse = entrypoint === 'transition' ? Object.keys(graph.inputs).filter(name => !supplied.has(name)) : [];
  const scope = {entrypoint};
  if (roots) scope.dependencies = new p.DependencyClosure({roots});
  else if (positions) scope.slotRows = new p.SlotRows({slots: positions.map(position => new p.SlotRow(
    position === null || position === 'empty' ? {empty: new p.Empty()} : position === 'recompute'
      ? {recompute: new p.Empty()} : {position}))});
  else scope.full = new p.Empty();
  return new p.ExecuteRequest({contextId, inputs: new p.InputBindings({values, reuse}), scope: new p.ExecutionScope(scope), outputs});
}
export function graphContext(p, compiledModelId, {rowMode, slots = 1, cache} = {}) {
  return new p.CreateExecutionContextRequest({compiledModelId,
    graph: new p.GraphExecution({slots, ...(rowMode === undefined ? {} : {rowMode, requireIncremental: rowMode === p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED})}),
    ...(cache ? {limits: new p.ExecutionLimits({cache: new p.ContextCacheOptions(cache)})} : {})});
}
export function inlineOutput(packet, name, Type = Float32Array) {
  const tensor = packet.outputs.find(value => value.name === name);
  if (!tensor?.inline) throw new Error(`missing inline output ${name}`);
  return new Type(tensor.inline.slice().buffer);
}

export function statelessDefinition(p, fields) {
  return {...fields, execution: new p.ModelExecutionContract({stateless: new p.Empty(),
    entrypoints: [new p.ExecutionEntrypoint({name: 'evaluate', effect: p.ExecutionEffect.EXECUTION_EFFECT_EVALUATE,
      allowedRanges: [p.ExecutionRangeKind.EXECUTION_RANGE_KIND_FULL],
      inputs: fields.inputs.map(input => input.name), outputs: fields.outputs})]})};
}
