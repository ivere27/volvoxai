/** Exercise the actual allocator, including gaps occupied by raw kernel arenas. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('WASM allocation, coalescing and realloc preserve live payloads and kernel arenas', async () => {
  const temporary = mkdtempSync(path.join(os.tmpdir(), 'volvoxai-allocator-'));
  try {
    const output = path.join(temporary, 'allocator.wasm');
    execFileSync(process.env.WASM_CC ?? 'clang', ['--target=wasm32', '-O2', '-fno-builtin', '-nostdlib',
      '-Inative/src/runtime/wasm_freestanding/include', '-DVOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC=1',
      'native/src/runtime/wasm_libc.c', 'tests/contracts/wasm_allocator_heap.c',
      '-Wl,--no-entry,--export=malloc,--export=calloc,--export=realloc,--export=free,--export=alloc_bytes,--export-memory,--initial-memory=16777216,--max-memory=67108864',
      '-o', output], { cwd: root, stdio: 'pipe' });
    const module = new WebAssembly.Module(readFileSync(output));
    assert.deepEqual(WebAssembly.Module.imports(module), []);
    const { exports: heap } = new WebAssembly.Instance(module);
    const bytes = (pointer, size) => new Uint8Array(heap.memory.buffer, pointer, size);
    const aligned = pointer => { assert.ok(pointer); assert.equal(pointer % 16, 0); return pointer; };

    const first = aligned(heap.malloc(16)), gap = aligned(heap.alloc_bytes(1024));
    const next = aligned(heap.malloc(4096));
    bytes(first, 16).fill(19); bytes(gap, 1024).fill(227);
    heap.free(next);
    const grown = aligned(heap.realloc(first, 2048));
    assert.notEqual(grown, first, 'realloc must not merge across the raw kernel arena');
    assert.ok(bytes(grown, 16).every(value => value === 19));
    bytes(grown, 2048).fill(53);
    assert.ok(bytes(gap, 1024).every(value => value === 227));
    heap.free(grown);

    const adjacent = new WebAssembly.Instance(module).exports;
    const left = aligned(adjacent.malloc(1024)), right = aligned(adjacent.malloc(4096));
    const guard = aligned(adjacent.malloc(64));
    new Uint8Array(adjacent.memory.buffer, left, 1024).fill(91); adjacent.free(right);
    const contiguous = aligned(adjacent.realloc(left, 3072));
    assert.equal(contiguous, left, 'adjacent free storage grows in place');
    assert.ok(new Uint8Array(adjacent.memory.buffer, contiguous, 1024).every(value => value === 91));
    adjacent.free(contiguous); adjacent.free(guard);

    let seed = 20260913;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
    const slots = Array.from({ length: 4096 }, (_, index) => {
      const size = 16 + index % 769, pointer = aligned(heap.malloc(size)), tag = index % 251;
      bytes(pointer, size).fill(tag); return { pointer, size, tag };
    });
    const check = slot => assert.ok(bytes(slot.pointer, slot.size).every(value => value === slot.tag));
    for (let iteration = 0; iteration < 20000; iteration++) {
      const index = random() % slots.length, old = slots[index], size = random() % 4096 + 1;
      check(old);
      let pointer;
      if (iteration % 3 === 0) {
        pointer = aligned(heap.realloc(old.pointer, size));
        assert.ok(bytes(pointer, Math.min(old.size, size)).every(value => value === old.tag));
      } else {
        heap.free(old.pointer);
        pointer = aligned(heap.calloc(size, 1));
        assert.ok(bytes(pointer, size).every(value => value === 0));
      }
      const tag = iteration % 251;
      bytes(pointer, size).fill(tag); slots[index] = { pointer, size, tag };
    }
    for (const slot of slots) { check(slot); heap.free(slot.pointer); }
    const reusable = aligned(heap.malloc(8 * 1024 * 1024));
    heap.free(reusable);
    const before = heap.memory.buffer.byteLength;
    for (let repeat = 0; repeat < 16; repeat++) {
      const whole = aligned(heap.malloc(8 * 1024 * 1024));
      bytes(whole, 8 * 1024 * 1024).fill(repeat); heap.free(whole);
    }
    assert.equal(heap.memory.buffer.byteLength, before, 'coalesced free storage is reused');
    assert.equal(heap.calloc(0x40000000, 16), 0, 'overflow fails without changing live storage');
    assert.ok(bytes(gap, 1024).every(value => value === 227));
  } finally { rmSync(temporary, { recursive: true, force: true }); }
});
