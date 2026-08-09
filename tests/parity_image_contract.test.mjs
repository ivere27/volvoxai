import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { concreteExecutionInputs } from './parity/lib/runmodel.mjs';

test('image parity binds input storage with the package public descriptor shape', async () => {
  const data = new Uint8Array(1 * 2 * 3 * 4);
  const snapshot = {
    inputNames: ['input0'],
    staticShapePlan: {
      tensors: {
        input0: { dtype: 'uint8', shape: [1, 2, 3, 4] },
      },
    },
  };
  assert.deepEqual(concreteExecutionInputs(snapshot, { input0: data }), {
    input0: { data, shape: [1, 2, 3, 4] },
  });

  const source = await readFile(new URL('./parity/image_check.mjs', import.meta.url), 'utf8');
  assert.match(
    source,
    /context\.execute\(concreteExecutionInputs\(snapshot, \{ input0: input \}\)\)/,
  );
  assert.doesNotMatch(source, /context\.execute\(\{ input0: input \}\)/);
});
