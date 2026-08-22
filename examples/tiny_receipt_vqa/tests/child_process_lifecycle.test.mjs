import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import test from 'node:test';

import {
  childProcessHasSettled,
  observeChildProcessExit,
} from '../tools/child_process_lifecycle.mjs';

class FixtureChild extends EventEmitter {
  constructor({ exitCode = null, signalCode = null } = {}) {
    super();
    this.exitCode = exitCode;
    this.signalCode = signalCode;
  }
}

test('already-signaled child is settled without waiting for a past exit event', async () => {
  const child = new FixtureChild({ signalCode: 'SIGKILL' });
  assert.equal(childProcessHasSettled(child), true);
  await Promise.race([
    observeChildProcessExit(child),
    new Promise((_, reject) => setTimeout(
      () => reject(new Error('already-signaled child did not settle')),
      100,
    )),
  ]);
  assert.equal(child.listenerCount('exit'), 0);
});

test('live child observer settles on exit and spawn error', async () => {
  for (const event of ['exit', 'error']) {
    const child = new FixtureChild();
    const settled = observeChildProcessExit(child);
    child.emit(event, event === 'error' ? new Error('fixture spawn failure') : 0);
    await settled;
    assert.equal(child.listenerCount('exit'), 0);
    assert.equal(child.listenerCount('error'), 0);
  }
});
