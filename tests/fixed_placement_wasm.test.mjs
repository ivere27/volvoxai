import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {fixture, p, ok, safetensors, statelessGraph} from '../tools/proto_fixture.mjs';

const {version} = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
for (const full of [false, true]) {
  test(`${full ? 'full' : 'inference'} WASM validates fixed placement without ignoring it`, async t => {
    const f = await fixture({full,
      wasmUrl: new URL(`../dist/${version}/volvoxai${full ? '' : '.lite'}.wasm`, import.meta.url)});
    t.after(() => f.close());
    const model = await f.load(statelessGraph({format: 'volvox-graph/v1', dimensions: {},
      inputs: {x: {dtype: 'float32', shape: [1, 2]}},
      nodes: [{id: 'projection', opType: 'Linear', inputs: {input: 'x', weight: 'w'},
        outputs: {out: {tensor: 'y', dtype: 'float32', shape: [1, 2]}}, params: {}}], outputs: ['y'],
    }), safetensors([{name: 'w', shape: [2, 2], data: Float32Array.of(1, 0, 0, 1)}]));
    const policy = new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: ['wasm'], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID});
    const ordinary = ok(await f.inference.compileModel(new p.CompileModelRequest({modelId: model.modelId, policy})));
    ok(await f.inference.releaseCompiledModel(new p.CompiledModelRef(ordinary)));
    for (const [cpuNodes, status, code] of [
      [['projection'], p.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED, p.OperationCode.OPERATION_CODE_BACKEND_UNSUPPORTED],
      [[''], p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT, p.OperationCode.OPERATION_CODE_INVALID_POLICY],
      [['projection\0ignored'], p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT, p.OperationCode.OPERATION_CODE_INVALID_POLICY],
      [['x'.repeat(128)], p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT, p.OperationCode.OPERATION_CODE_INVALID_POLICY],
    ]) {
      const response = await f.inference.compileModel(new p.CompileModelRequest({modelId: model.modelId, policy, cpuNodes}));
      assert.equal(response.report.status, status);
      assert.equal(response.report.code, code);
      assert.equal(response.report.stage, p.OperationStage.OPERATION_STAGE_COMPILE);
      assert.equal(response.compiledModelId, 0n);
    }
  });
}
