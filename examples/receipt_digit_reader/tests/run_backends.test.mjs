import test from 'node:test';
import assert from 'node:assert/strict';

import { summarizeBackendRecords } from '../tools/run_backends.mjs';

const PRIVATE_RECORDS = [
  {
    backend: 'cpu-js', phone: 'private phone', street: 'private street',
    runs: 2, median_ms: 1,
  },
  {
    backend: 'wasm', phone: 'private phone', street: 'private street',
    runs: 2, median_ms: 2,
  },
];

test('backend agreement summaries omit decoded receipt values by default', () => {
  const summary = summarizeBackendRecords(PRIVATE_RECORDS);
  assert.deepEqual(summary, [
    {
      backend: 'cpu-js', runs: 2, median_ms: 1, decodedRecordStableAcrossRuns: true,
      decodedRecordMatchesFirstBackend: true,
    },
    {
      backend: 'wasm', runs: 2, median_ms: 2, decodedRecordStableAcrossRuns: true,
      decodedRecordMatchesFirstBackend: true,
    },
  ]);
  assert.doesNotMatch(JSON.stringify(summary), /private/);
});

test('decoded receipt values require the explicit private-report option', () => {
  const summary = summarizeBackendRecords(PRIVATE_RECORDS, true);
  assert.equal(summary[0].phone, 'private phone');
  assert.equal(summary[0].street, 'private street');
});
