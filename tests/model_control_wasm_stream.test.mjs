import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {loadModelControlWasmDispatchFactory, ModelSourceLengthError} from '../ts/core/ModelControlWasm.js';
import {VxPlatformServiceClient} from '../runtime/generated/typescript/inference/volvoxai_ffi.js';
import {Empty} from '../runtime/generated/typescript/inference/volvoxai_lite.js';
import {RpcError} from '../runtime/generated/typescript/inference/synurang_runtime.js';

const {version}=JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
const wasmUrl=new URL(`../dist/${version}/volvoxai.lite.wasm`,import.meta.url);
const factory=await loadModelControlWasmDispatchFactory(wasmUrl);

test('manifest stream uses bounded writes while ordinary RPCs remain available',async()=>{
  const owner=factory.create();const path='/__volvoxai_host/model-stream/direct';
  let resume,controller,entered;const reading=new Promise(resolve=>{entered=resolve;});
  const source=new Uint8Array(2*1024*1024+7);source.fill(17);
  const stream=new ReadableStream({start(value){controller=value;value.enqueue(source);},
    pull(){entered();return new Promise(resolve=>{resume=resolve;});}}, {highWaterMark:0});
  try {
    const mounting=owner.mountFileStream(path,stream,source.byteLength+3);
    await reading;
    const info=await new VxPlatformServiceClient(owner).getPlatformInfo(new Empty());
    assert.ok(info,'awaiting I/O does not hold the synchronous C mailbox guard');
    await assert.rejects(owner.mountFileStream('/other',new ReadableStream(),4),error=>error instanceof RpcError&&error.code===8);
    controller.enqueue(Uint8Array.of(1,2,3));controller.close();resume();await mounting;
    owner.unmountFile(path);
  } finally {await owner.close();}
});

test('stream length and read failures abort unpublished VFS storage',async()=>{
  const owner=factory.create();const path='/__volvoxai_host/model-stream/failure';
  try {
    for(const bytes of [Uint8Array.of(1),Uint8Array.of(1,2,3)]) {
      const stream=new ReadableStream({start(controller){controller.enqueue(bytes);controller.close();}});
      await assert.rejects(owner.mountFileStream(path,stream,2),ModelSourceLengthError);
      owner.mountFile(path,Uint8Array.of(8,9));owner.unmountFile(path);
    }
    const broken=new ReadableStream({pull(controller){controller.error(new Error('reader failed'));}});
    await assert.rejects(owner.mountFileStream(path,broken,2),/reader failed/);
    owner.mountFile(path,Uint8Array.of(8,9));owner.unmountFile(path);
  } finally {await owner.close();}
});

test('cancellation aborts an active upload while retaining a usable owner',async()=>{
  const owner=factory.create();const path='/__volvoxai_host/model-stream/cancel';
  const controller=new AbortController();let entered,cancelled=false;
  const reading=new Promise(resolve=>{entered=resolve;});
  const stream=new ReadableStream({pull(){entered();return new Promise(()=>{});},cancel(){cancelled=true;}});
  try {
    const mounting=owner.mountFileStream(path,stream,4,controller.signal);
    const rejected=assert.rejects(mounting,error=>error instanceof RpcError&&error.code===1);
    await reading;controller.abort(new RpcError(1,'cancelled'));await rejected;
    assert.equal(cancelled,true);
    owner.mountFile(path,Uint8Array.of(1,2,3,4));owner.unmountFile(path);
    assert.ok(await new VxPlatformServiceClient(owner).getPlatformInfo(new Empty()));
  } finally {await owner.close();}
});

test('owner close drains upload ABORT even when transport cancel never settles',async()=>{
  const owner=factory.create();let entered,cancelled=false;
  const reading=new Promise(resolve=>{entered=resolve;});
  const stream=new ReadableStream({pull(){entered();return new Promise(()=>{});},cancel(){cancelled=true;return new Promise(()=>{});}});
  const mounting=owner.mountFileStream('/__volvoxai_host/model-stream/close',stream,4);
  const rejected=assert.rejects(mounting,error=>error instanceof RpcError&&error.code===1);
  await reading;await owner.close();await rejected;
  assert.equal(cancelled,true);assert.equal(owner.closed,true);
});
