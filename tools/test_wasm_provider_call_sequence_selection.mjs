import assert from 'node:assert/strict';
import path from 'node:path';
import { fixture, p, ok, tensors } from './proto_fixture.mjs';
import { wasmProfile } from './release_profiles.mjs';
const artifact = process.argv[2];
if (!artifact) throw Error('Pass an inference or full WASM artifact.');
const extraInputNames = Object.freeze(Array.from({ length: 7 }, (_, index) => `extra${index}`));
const shape = Object.freeze([1, 'S', 2]);
const graph = {
  format: 'volvox-graph/v1',
  dimensions: { S: { min: 2, max: 13, multiple_of: 1 } },
  inputs: {
    left: { dtype: 'float32', shape },
    right: { dtype: 'float32', shape },
    ...Object.fromEntries(extraInputNames.map((name) => [
      name,
      { dtype: 'float32', shape },
    ])),
  },
  nodes: [{
    id: 'add',
    opType: 'Add',
    inputs: { a: 'left', b: 'right' },
    outputs: { out: { tensor: 'out', dtype: 'float32', shape } },
    params: {},
  }],
  outputs: ['out'],
};


const f = await fixture({wasmUrl: path.resolve(artifact),
  full: wasmProfile(path.basename(artifact)).id === 'full'});
try {
  const model = await f.load(graph), compiled = await f.compile(model);
  const context = ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest({
    compiledModelId: compiled.compiledModelId, requireIncremental: true,
    decodeRowMode: p.DecodeRowMode.DECODE_ROW_MODE_AUTO,
  })));
  const inputs = Object.fromEntries(['left','right',...extraInputNames].map(name =>
    [name, {data: new Float32Array(26), shape: [1,13,2]}]));
  inputs.left.data.set([1,2]); inputs.right.data.set([10,20]);
  const initial = ok(await f.inference.decodePrefill(new p.DecodePrefillRequest({
    contextId: context.contextId, inputs: tensors(inputs), position: 0})));
  ok(await f.inference.releaseResult(new p.ResultRef(initial)));
  const expected = new Float32Array(26); expected.set([11,22]);
  const selections = [['left','right'],['right','left'],['left'],['right'],
    ...extraInputNames.map(name => [name]), [extraInputNames.at(-1)]];
  for (const [index, names] of selections.entries()) {
    const position = index + 1;
    for (const name of names) inputs[name].data.set([position * 3, position * 7], position * 2);
    if (names.includes('left') || names.includes('right')) {
      expected[position*2] = inputs.left.data[position*2] + inputs.right.data[position*2];
      expected[position*2+1] = inputs.left.data[position*2+1] + inputs.right.data[position*2+1];
    }
    const result = ok(await f.inference.decodeStep(new p.DecodeStepRequest({contextId: context.contextId,
      position, inputs: tensors(Object.fromEntries(names.map(name => [name, inputs[name]])))})));
    assert.deepEqual(await f.read(result, 'out'), expected);
    ok(await f.inference.releaseResult(new p.ResultRef(result)));
  }
  console.log('C decode changed-input selection: PASS (reordered, subset and irrelevant inputs).');
} finally { await f.close(); }
