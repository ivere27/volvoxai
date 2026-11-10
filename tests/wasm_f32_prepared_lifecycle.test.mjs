import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {fixture, p, ok, tensors, safetensors, statelessGraph} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxProfilingServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';

const {version} = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
for (const profile of ['inference', 'full']) {
  if (process.env.VOLVOXAI_F32_PROFILE && process.env.VOLVOXAI_F32_PROFILE !== profile) continue;
  test(`${profile}: prepared F32 packs are inventoried and independently released with contexts`, async t => {
    const stem = profile === 'full' ? 'volvoxai' : 'volvoxai.lite';
    const wasmUrl = process.env[`VOLVOXAI_F32_${profile.toUpperCase()}_WASM`] ??
      new URL(`../dist/${version}/${stem}.wasm`, import.meta.url).pathname;
    const f = await fixture({wasmUrl, full: profile === 'full'});
    t.after(() => f.close());
    const profiling = new VxProfilingServiceClient(reportTransport(f.host));
    const k = 17, n = 9;
    const graph = statelessGraph({format: 'volvox-graph/v1', dimensions: {},
      inputs: {x: {dtype: 'float32', shape: [1, k]}}, outputs: ['y'],
      nodes: [{id: 'dense', opType: 'Linear', inputs: {input: 'x', weight: 'w'},
        outputs: {out: {tensor: 'y', dtype: 'float32', shape: [1, n]}},
        params: {weight_layout: 'dout_din'}}]});
    const model = await f.load(graph, safetensors([{name: 'w', shape: [n, k],
      data: new Float32Array(n * k).fill(1)}]));
    const compiled = await f.compile(model);
    const first = await f.context(compiled), second = await f.context(compiled);
    const packs = async scope => ok(await profiling.getResourceSnapshot(
      new p.GetResourceSnapshotRequest(scope))).snapshot.memory.allocations
        .filter(allocation => allocation.name === 'prepared F32 weight');
    assert.deepEqual((await packs(first)).map(x => x.capacityBytes), [1088n]);
    assert.deepEqual((await packs(second)).map(x => x.capacityBytes), [1088n]);
    assert.equal((await packs(f.runtime)).length, 2);
    const run = async (context, value) => {
      const packet = ok(await f.inference.execute(new p.ExecuteRequest({contextId: context.contextId,
        inputs: new p.InputBindings({values: tensors({x: {shape: [1, k],
          data: new Float32Array(k).fill(value)}})}),
        scope: new p.ExecutionScope({entrypoint: 'evaluate', full: new p.Empty()}),
        outputs: new p.OutputOptions({storage: p.OutputStorage.OUTPUT_STORAGE_INLINE})})));
      assert.deepEqual([...await f.read(packet, 'y')], Array(n).fill(k * value));
    };
    await run(first, 1); await run(second, 2); await run(first, -1);
    ok(await f.inference.releaseExecutionContext(new p.ExecutionContextRef(first)));
    assert.equal((await packs(f.runtime)).length, 1);
    await run(second, 3);
    ok(await f.inference.releaseExecutionContext(new p.ExecutionContextRef(second)));
    assert.equal((await packs(f.runtime)).length, 0);
    const third = await f.context(compiled);
    await run(third, 4);
    assert.equal((await packs(f.runtime)).length, 1);
    ok(await f.inference.releaseExecutionContext(new p.ExecutionContextRef(third)));
    assert.equal((await packs(f.runtime)).length, 0);
    ok(await f.inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
    ok(await f.inference.releaseModel(new p.ModelRef(model)));
    ok(await f.inference.releaseRuntime(new p.RuntimeRef(f.runtime)));
  });
}
