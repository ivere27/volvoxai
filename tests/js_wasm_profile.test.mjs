import test from 'node:test';
import assert from 'node:assert/strict';

import { WasmEngine } from '../ts/backends/WasmEngine.js';
import { createRuntime } from '../ts/WasmProfile.js';

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
