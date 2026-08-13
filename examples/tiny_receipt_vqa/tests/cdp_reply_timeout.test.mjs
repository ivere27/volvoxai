import assert from 'node:assert/strict';
import test from 'node:test';

import { cdpReplyTimeoutMs } from '../tools/cdp_reply_timeout.mjs';

test('Runtime.evaluate consumes the remaining overall benchmark deadline', () => {
  assert.equal(cdpReplyTimeoutMs('Runtime.evaluate', 60_000), 60_000);
  assert.equal(cdpReplyTimeoutMs('Runtime.evaluate', 73), 73);
});

test('CDP control commands retain the short watchdog', () => {
  assert.equal(cdpReplyTimeoutMs('Runtime.enable', 60_000), 5000);
  assert.equal(cdpReplyTimeoutMs('Log.enable', 73), 73);
});
