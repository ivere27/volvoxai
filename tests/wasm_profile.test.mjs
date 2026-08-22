import test from 'node:test';
import assert from 'node:assert/strict';

import { Runtime } from '../ts/core/ContextRuntime.js';
import { WasmBackendProvider } from '../ts/backends/WasmBackendProvider.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';
import { createRuntime } from '../ts/WasmProfile.js';
import * as wasmEntry from '../ts/wasm.js';

function typedFailure(code, message) {
  return (error) => error?.name === 'VolvoxAIError' && error.code === code &&
    error.phase === 'initialization' && error.backend === 'wasm' &&
    message.test(error.message);
}

test('strict WASM runtime validates its public options before initialization', async () => {
  for (const options of [null, [], 1, 'wasm']) {
    await assert.rejects(
      createRuntime(options),
      typedFailure('INVALID_ARGUMENT', /options must be an object/),
    );
  }
  for (const wasmUrl of ['', 7, {}]) {
    await assert.rejects(
      createRuntime({ wasmUrl }),
      typedFailure('INVALID_ARGUMENT', /wasmUrl must be a non-empty string or URL/),
    );
  }
  await assert.rejects(
    createRuntime({ onDiagnostic: 'verbose' }),
    typedFailure('INVALID_ARGUMENT', /onDiagnostic must be a function or null/),
  );
});

test('strict WASM runtime reports thrown and null initialization failures with stable codes',
  { concurrency: false }, async (t) => {
    const originalInit = WasmEngine.init;
    t.after(() => { WasmEngine.init = originalInit; });

    const cause = new Error('fixture initialization failure');
    WasmEngine.init = async () => { throw cause; };
    await assert.rejects(
      createRuntime({ wasmUrl: 'fixture.wasm' }),
      (error) => typedFailure('BACKEND_UNAVAILABLE', /could not load 'fixture\.wasm'/)(error) &&
        error.cause === cause,
    );

    WasmEngine.init = async () => null;
    await assert.rejects(
      createRuntime({ wasmUrl: 'missing.wasm' }),
      typedFailure('BACKEND_UNAVAILABLE', /could not load 'missing\.wasm'/),
    );
  });

function disposableWasmEngine(onDispose) {
  const engine = Object.create(WasmEngine.prototype);
  Object.defineProperty(engine, 'dispose', { value: onDispose });
  return engine;
}

test('strict WASM runtime forwards RuntimeOptions and owns its initialized engine once',
  { concurrency: false }, async (t) => {
    const originalInit = WasmEngine.init;
    t.after(() => { WasmEngine.init = originalInit; });

    let disposeCalls = 0;
    WasmEngine.init = async () => disposableWasmEngine(() => { disposeCalls++; });
    const runtime = await createRuntime({
      wasmUrl: 'fixture.wasm',
      execution: {
        mode: 'scheduled',
        scheduler: {
          maxRequests: 7,
          maxInputBytes: 4096,
          maxBatchSize: 3,
          maxBatchDelayMs: 2,
          maxPriorityBurst: 4,
        },
        results: {
          maxRetainedResults: 5,
          maxRetainedOutputBytes: 2048,
        },
      },
    });
    assert.deepEqual(runtime.listBackends(), ['wasm']);
    assert.deepEqual(runtime.inspectExecution(), {
      defaultMode: 'scheduled',
      schedulerAllocated: false,
      admittedRequests: 0,
      admittedInputBytes: 0,
      admittingInputBytes: 0,
      stagingInputBytes: 0,
      totalInputBytes: 0,
      inputSnapshotCopies: 0,
      retainedResults: 0,
      retainedOutputBytes: 0,
      maxRetainedResults: 5,
      maxRetainedOutputBytes: 2048,
    });
    await runtime.close();
    await runtime.close();
    assert.equal(disposeCalls, 1);
  });

test('strict WASM runtime disposes post-init Runtime and provider failures exactly once',
  { concurrency: false }, async (t) => {
    const originalInit = WasmEngine.init;
    t.after(() => { WasmEngine.init = originalInit; });

    let runtimeFailureDisposals = 0;
    WasmEngine.init = async () => disposableWasmEngine(() => {
      runtimeFailureDisposals++;
    });
    await assert.rejects(
      createRuntime({ wasmUrl: 'fixture.wasm', execution: { mode: 'invalid' } }),
      (error) => error?.code === 'INVALID_ARGUMENT' &&
        /execution mode is invalid/.test(error.message),
    );
    assert.equal(runtimeFailureDisposals, 1);

    let memoryFailureDisposals = 0;
    WasmEngine.init = async () => disposableWasmEngine(() => {
      memoryFailureDisposals++;
    });
    await assert.rejects(
      createRuntime({ wasmUrl: 'fixture.wasm', memoryCapture: { protocol: 'obsolete' } }),
      (error) => error?.code === 'INVALID_ARGUMENT' &&
        /memoryCapture\.protocol/.test(error.message),
    );
    assert.equal(memoryFailureDisposals, 1);

    let providerFailureDisposals = 0;
    WasmEngine.init = async () => ({
      dispose() { providerFailureDisposals++; },
    });
    await assert.rejects(
      createRuntime({ wasmUrl: 'fixture.wasm' }),
      typedFailure('INVALID_ARGUMENT', /provider requires a WasmEngine/),
    );
    assert.equal(providerFailureDisposals, 1);
  });

test('strict WASM construction preserves its failure when partial-registration cleanup throws',
  { concurrency: false }, async (t) => {
    const originalInit = WasmEngine.init;
    const originalAddProvider = Runtime.prototype._addProvider;
    const originalProviderClose = WasmBackendProvider.prototype.close;
    t.after(() => {
      WasmEngine.init = originalInit;
      Runtime.prototype._addProvider = originalAddProvider;
      WasmBackendProvider.prototype.close = originalProviderClose;
    });

    const constructionFailure = new Error('fixture failed after provider registration');
    let providerCloseCalls = 0;
    let engineDisposeCalls = 0;
    WasmEngine.init = async () => disposableWasmEngine(() => {
      engineDisposeCalls++;
      throw new Error('fixture engine cleanup failed');
    });
    Runtime.prototype._addProvider = function addThenFail(name, provider) {
      originalAddProvider.call(this, name, provider);
      throw constructionFailure;
    };
    WasmBackendProvider.prototype.close = function countedClose() {
      providerCloseCalls++;
      return originalProviderClose.call(this);
    };

    await assert.rejects(
      createRuntime({ wasmUrl: 'fixture.wasm' }),
      (error) => error === constructionFailure,
    );
    assert.equal(providerCloseCalls, 1);
    assert.equal(engineDisposeCalls, 1);
  });

test('strict WASM entry exposes the main inference scheduling surface', () => {
  assert.equal(wasmEntry.ExecutionMode.Direct, 0);
  assert.equal(wasmEntry.ExecutionMode.Scheduled, 1);
  assert.deepEqual(wasmEntry.executionModes, ['direct', 'scheduled']);
  assert.equal(typeof wasmEntry.RuntimeRequestHandle, 'function');
});
