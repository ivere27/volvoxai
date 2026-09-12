/** Index origins and rejection before the first GPU command/input mutation. */
import assert from 'node:assert/strict';
import path from 'node:path';
import { CASES, graphOf, runGraph, compare, requirePhysicalDevice } from './webgpu_device_bridge.mjs';
const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
const api = await import(`file://${path.resolve(args.get('--bundle'))}`), p = api.pb;
const { device, description } = await requirePhysicalDevice();
const options = { wasm: path.resolve(args.get('--wasm')), device };
const records = [];
const selected = name => !args.has('--only') || name.toLowerCase().includes(args.get('--only').toLowerCase());
const tensor = (name, shape, data) => ({ name, shape, data });
const node = (id, opType, inputs, shape, dtype = 'int32', params = {}) => ({ id, opType, inputs, params,
  outputs: { out: { tensor: id, shape, dtype } } });
const check = async (name, graph, batches, expected, extra = {}) => {
  if (!selected(name)) return;
  const cpu = await runGraph(api, { ...options, bankResidency: extra.bankResidency }, graph, batches, 'wasm');
  const gpu = await runGraph(api, { ...options, ...extra }, graph, batches, 'webgpu');
  for (let round = 0; round < gpu.rounds.length; round++) for (const [output, values] of cpu.rounds[round]) {
    compare(`${name}/${round}`, gpu.rounds[round].get(output), values, 1e-4);
    if (expected) compare(`${name}/${round}/oracle`, gpu.rounds[round].get(output), expected[round], 1e-4);
  }
  assert.ok(gpu.encodes); assert.equal(gpu.route.fallbackNodes, 0);
  records.push({ name, dispatches: gpu.encodes }); console.log(`value preflight ${name}: PASS`);
};
const encoded = values => values.map(t => new p.Tensor({ name: t.name, shape: t.shape.map(BigInt),
  dtype: t.data instanceof Int32Array ? p.DataType.DATA_TYPE_I32 : t.data instanceof Int8Array ? p.DataType.DATA_TYPE_I8 : p.DataType.DATA_TYPE_F32,
  inline: new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength) }));
const rejectBetweenRuns = invalid => async ({ inference, context, result, outputs, encodes }) => {
  for (const values of invalid) {
    const before = encodes();
    const rejected = await inference.execute(new p.ExecuteRequest({ contextId: context.contextId, inputs: encoded(values) }));
    assert.equal(rejected.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
      JSON.stringify(rejected.report, (_, value) => typeof value === 'bigint' ? String(value) : value));
    assert.equal(encodes(), before, 'invalid input reached a GPU command');
    for (const [name, values] of outputs) {
      const read = await inference.readOutput(new p.ReadOutputRequest({ resultId: result.resultId, name }));
      assert.equal(read.report.status, 0);
      assert.deepEqual([...new values.constructor(read.tensor.inline.slice().buffer)], [...values], 'invalid execution changed a retained result');
    }
  }
};
try {
  const bank = { shape: [4], data: Float32Array.of(10, 20, 30, 40) };
  for (const op of ['Clip', 'Clip-tensor', 'Equal', 'GreaterOrEqual', 'Not', 'Cast']) {
    const clip = op.startsWith('Clip'), comparison = op === 'Equal' || op === 'GreaterOrEqual';
    const graph = graphOf([
      node('ids', clip ? 'Clip' : op, comparison ? { a: 'raw', b: 'other' } : { input: 'raw' }, [4], 'int32',
        clip ? { min: -4, max: 3 } : op === 'Cast' ? { to: 'int32' } : {}),
      node('out', 'Gather', { input: 'bank', indices: 'ids' }, [4], 'float32', { axis: 0 })
    ], { raw: { shape: [4], dtype: 'int32' }, ...(comparison ? { other: { shape: [4], dtype: 'int32' } } : {}) }, ['out'], { bank });
    if (op === 'Clip-tensor') {
      graph.nodes[0].inputs.min = 'min'; graph.nodes[0].inputs.max = 'max';
      graph.nodes[0].params = {};
      graph.__weights.min = { shape: [], data: Int32Array.of(-4) }; graph.__weights.max = { shape: [], data: Int32Array.of(3) };
    }
    const raw = clip ? [Int32Array.of(-100, 2, 100, -1), Int32Array.of(3, -3, 0, 50)] :
      [Int32Array.of(0, 2, 1, 3), Int32Array.of(3, 1, 0, 2)];
    const other = Int32Array.of(1, 2, 3, 0);
    const batches = raw.map(values => [tensor('raw', [4], values), ...(comparison ? [tensor('other', [4], other)] : [])]);
    const expected = raw.map(values => Float32Array.from(values, (v, i) => {
      const id = clip ? Math.max(-4, Math.min(3, v)) : op === 'Equal' ? +(v === other[i]) :
        op === 'GreaterOrEqual' ? +(v >= other[i]) : op === 'Not' ? +(v === 0) : v;
      return bank.data[id < 0 ? id + 4 : id];
    }));
    await check(`${op}-to-Gather`, graph, batches, expected);
  }
  {
    const graph = graphOf([
      node('ids', 'ArgMax', { input: 'scores' }, [], 'int32', { axis: 0, keepdims: false }),
      node('out', 'Gather', { input: 'bank', indices: 'ids' }, [], 'float32')
    ], { scores: { shape: [4], dtype: 'float32' } }, ['out'], { bank });
    await check('ArgMax-scalar-to-Gather', graph,
      [[tensor('scores', [4], Float32Array.of(0, 1, 9, 3))], [tensor('scores', [4], Float32Array.of(9, 1, 2, 3))]],
      [Float32Array.of(30), Float32Array.of(10)]);
  }
  {
    const graph = graphOf([
      node('ids', 'ArgMax', { input: 'scores' }, [2, 1], 'int32', { axis: 1, keepdims: true }),
      node('out', 'GatherElements', { input: 'bank', indices: 'ids' }, [2, 1], 'float32', { axis: 1 })
    ], { scores: { shape: [2, 4], dtype: 'float32' } }, ['out'],
      { bank: { shape: [2, 4], data: Float32Array.of(10, 20, 30, 40, 50, 60, 70, 80) } });
    await check('ArgMax-to-GatherElements', graph,
      [[tensor('scores', [2, 4], Float32Array.of(0, 1, 9, 3, 9, 1, 2, 3))],
       [tensor('scores', [2, 4], Float32Array.of(0, 9, 1, 3, 0, 1, 2, 9))]],
      [Float32Array.of(30, 50), Float32Array.of(20, 80)]);
  }
  for (const quantized of [false, true]) for (const constant of [false, true]) {
    const Array_=quantized?Int8Array:Float32Array, dtype=quantized?'int8':'float32';
    const values=Array_.of(0,9,1,3,9,1,2,3), scores={shape:[4,2],data:values};
    const graph=graphOf([
      node('ids',quantized?'QArgMax':'ArgMax',{input:'scores'},[2],'int32',
        quantized?{axis:0}:{axis:0,keepdims:false}),
      node('out','Gather',{input:'bank',indices:'ids'},[2],'float32')
    ],constant?{}:{scores:{shape:[4,2],dtype}},['out'],{bank,...(constant?{scores}:{})});
    if(quantized) {
      graph.__weights.scale={shape:[1],data:Float32Array.of(.03)};
      graph.__weights.zero={shape:[1],data:Int8Array.of(0)};
      graph.quantization={format:'volvox-affine-safetensors/v1',tensors:{scores:{scheme:'per_tensor',scale_tensor:'scale',zero_point_tensor:'zero'}}};
    }
    await check(`${quantized?'QArgMax':'ArgMax'}-axis-zero-${constant?'constant':'public'}-to-Gather`,graph,
      [constant?[]:[tensor('scores',[4,2],values)]],[Float32Array.of(30,10)]);
  }
  for (const [dtype,Array_,domain] of [['int8',Int8Array,128],['uint8',Uint8Array,256]]) {
    const graph=graphOf([
      node('ids','Cast',{input:'raw'},[4],'int32',{to:'int32'}),
      node('out','Gather',{input:'bank',indices:'ids'},[4],'float32')
    ],{raw:{shape:[4],dtype}},['out'],{bank:{shape:[domain],data:Float32Array.from({length:domain},(_,i)=>i+.25)}});
    const values=Array_.of(-128,-1,0,127),expected=Float32Array.from(values,id=>(id<0?id+domain:id)+.25);
    await check(`Cast-${dtype}-to-Gather`,graph,[[tensor('raw',[4],values)]],[expected]);
  }
  for (const preserve of [false, true]) {
    const graph = graphOf([
      ...(preserve ? [node('ids', 'Identity', { input: 'raw' }, [2])] : []),
      node('out', 'Gather', { input: 'data', indices: preserve ? 'ids' : 'raw' }, ['B', 2], 'float32', { axis: -1 })
    ], { data: { shape: ['B', 4], dtype: 'float32' }, raw: { shape: [2], dtype: 'int32' } }, ['out']);
    graph.dimensions = { B: { min: 1, max: 3 } };
    const batches = [1, 3, 2].map(b => [tensor('data', [b, 4], Float32Array.from({ length: b * 4 }, (_, i) => i + b)), tensor('raw', [2], Int32Array.of(-1, 0))]);
    const invalid = [-5, 4, 2147483647].map(id => [tensor('data', [1, 4], Float32Array.of(999, 999, 999, 999)), tensor('raw', [2], Int32Array.of(id, 0))]);
    await check(`public-axis-minus-one${preserve ? '-preserved' : ''}`, graph, batches, null, { afterRead: rejectBetweenRuns(invalid) });
  }
  {
    const [name, original, source] = CASES.find(([name]) => name === 'QLayerNorm');
    const graph = structuredClone(original), tensors = [...source];
    for (const role of ['gamma', 'beta']) {
      const t = graph.__weights[role]; graph.inputs[role] = { shape: t.shape, dtype: 'float32' };
      tensors.push(tensor(role, t.shape, t.data)); delete graph.__weights[role];
    }
    const invalid = [NaN, Infinity, -Infinity].map(value => tensors.map(t => t.name === 'gamma' ? { ...t, data: Float32Array.of(value, 1) } : t));
    await check(`${name}-public-finite`, graph, [tensors, tensors], null, { afterRead: rejectBetweenRuns(invalid) });
  }
  for (const slots of [[1, 3], [1, 4]]) {
    const graph = graphOf([node('out', 'Gather', { input: 'bank', indices: 'raw' }, [2, 2], 'float32')],
      { raw: { shape: [2], dtype: 'int32' } }, ['out'],
      { bank: { shape: [5, 2], data: Float32Array.of(10, 11, 20, 21, 30, 31, 40, 41, 50, 51) } });
    graph.banks = { bank: 'E' }; graph.dimensions = { E: { min: 1, max: 8 } };
    const positive = slots, negative = slots.map(value => value - 5);
    const batches = [positive, negative].map(ids => [tensor('raw', [2], Int32Array.from(ids))]);
    const expected = Float32Array.from(slots.flatMap(slot => [10 * (slot + 1), 10 * (slot + 1) + 1]));
    const invalid = [0, 2, 5, ...(!slots.includes(4) ? [-1] : [])].map(id => [tensor('raw', [2], Int32Array.of(id, slots[0]))]);
    await check(`Gather-partial-bank-${slots.join('-')}`, graph, batches, [expected, expected],
      { bankResidency: [{ bank: 'bank', slots }], afterRead: rejectBetweenRuns(invalid) });
  }
  for (const partial of [false, true]) {
    const graph = graphOf([node('out', 'MoELinear', {
      input: 'x', expert_weight: 'experts', route_indices: 'ids', route_weights: 'gates'
    }, [2, 2], 'float32')], Object.fromEntries(['x', 'ids', 'gates'].map(name => [name, { shape: [2, 2], dtype: 'float32' }])),
      ['out'], { experts: { shape: [5, 2, 2], data: Float32Array.from({ length: 20 }, (_, i) => i % 4 === 0 || i % 4 === 3 ? Math.floor(i / 4) + 1 : 0) } });
    graph.banks = { experts: 'E' }; graph.dimensions = { E: { min: 1, max: 8 } };
    const inputs = [tensor('x', [2, 2], Float32Array.of(1, 2, 3, 4)), tensor('ids', [2, 2], Float32Array.of(1, 3, 1, 3)),
      tensor('gates', [2, 2], Float32Array.of(.25, .75, .5, .5))];
    const invalid = [];
    for (const id of [-1, 1.5, 5, Infinity, NaN, ...(partial ? [0, 4] : [])])
      invalid.push(inputs.map(t => t.name === 'ids' ? { ...t, data: Float32Array.of(id, 3, 1, 3) } : t));
    for (const gate of [Infinity, -Infinity, NaN])
      invalid.push(inputs.map(t => t.name === 'gates' ? { ...t, data: Float32Array.of(gate, 0, 0, 0) } : t));
    await check(`MoELinear-${partial ? 'partial' : 'full'}-bank`, graph, [inputs, inputs],
      [Float32Array.of(3.5, 7, 9, 12), Float32Array.of(3.5, 7, 9, 12)],
      { bankResidency: partial ? [{ bank: 'experts', slots: [1, 3] }] : [], afterRead: rejectBetweenRuns(invalid) });
  }
  assert.ok(records.length);
  console.log(`value preflight: ${records.length} cases passed`);
  if (args.has('--out')) await Deno.writeTextFile(args.get('--out'), JSON.stringify({ adapter: description, records }, null, 2));
} finally { device.destroy(); await device.lost; }
