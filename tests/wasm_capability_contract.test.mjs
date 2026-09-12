import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadWasmReleaseModule } from '../ts/core/WasmReleaseModule.js';
import { EngineHost } from '../ts/host/EngineHost.js';
import { VxInferenceServiceClient } from '../runtime/generated/typescript/inference/volvoxai_ffi.js';
import * as pb from '../runtime/generated/typescript/inference/volvoxai_lite.js';

test('invalid modules fail at transport initialization and do not create handles', async () => {
  const host=new EngineHost({wasmUrl:'data:application/wasm;base64,bm90LXdhc20='});
  const api=new VxInferenceServiceClient(reportTransport(host));
  try { await assert.rejects((api.createRuntime(new pb.CreateRuntimeRequest())),/WebAssembly|magic|module/i); }
  finally { await host.close(); }
});

test('a failed module compilation is not retained by the shared cache', async () => {
  const url='data:application/wasm;base64,bm90LXdhc20tbW9kdWxl';
  const errors=[];
  for(let attempt=0;attempt<2;attempt++) {
    try {await loadWasmReleaseModule(url); assert.fail('invalid module was accepted');}
    catch(error){errors.push(error);}
  }
  assert.ok(errors.every(e=>e instanceof WebAssembly.CompileError));
  assert.notEqual(errors[0],errors[1], 'a corrected source can be compiled on the next load');
});
