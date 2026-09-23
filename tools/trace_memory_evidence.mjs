/** Validate a complete public allocation history against its reported counters. */
import assert from 'node:assert/strict';
export function checkTraceMemory(events, info, p) {
  const live = new Map(), seen = new Set(), totals = new Map();
  const memory = events.filter(event => event.memory);
  for (const {memory: event} of memory) {
    const row = totals.get(event.allocator) ?? {existingBytes: 0n, allocatedBytes: 0n,
      freedBytes: 0n, liveBytes: 0n, peakBytes: 0n};
    totals.set(event.allocator, row);
    assert.ok(event.allocationId > 0n);
    if (event.action === p.TraceMemoryAction.TRACE_MEMORY_ACTION_FREE) {
      const allocation = live.get(event.allocationId);
      assert.deepEqual(allocation, [event.allocator, event.bytes]);
      live.delete(event.allocationId);
      row.freedBytes += event.bytes; row.liveBytes -= event.bytes;
    } else {
      assert.ok(!seen.has(event.allocationId), 'allocation identities must not be reused');
      seen.add(event.allocationId);
      live.set(event.allocationId, [event.allocator, event.bytes]);
      row[event.action === p.TraceMemoryAction.TRACE_MEMORY_ACTION_EXISTING ? 'existingBytes' : 'allocatedBytes'] += event.bytes;
      row.liveBytes += event.bytes;
    }
    row.peakBytes = row.liveBytes > row.peakBytes ? row.liveBytes : row.peakBytes;
    assert.equal(event.liveBytes, row.liveBytes);
  }
  assert.equal(totals.size, info.allocators.length);
  for (const allocator of info.allocators) {
    assert.equal(allocator.accountingComplete, true);
    assert.equal(allocator.droppedEvents, 0n);
    assert.equal(allocator.inventory, p.MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL);
    assert.equal(allocator.scope, allocator.allocator === 'webgpu.buffer'
      ? p.MemoryOwnerKind.MEMORY_OWNER_KIND_BACKEND_SHARED : p.MemoryOwnerKind.MEMORY_OWNER_KIND_RUNTIME);
    for (const [key, value] of Object.entries(totals.get(allocator.allocator)))
      assert.equal(allocator[key], value, `${allocator.allocator}.${key}`);
  }
  return memory;
}
