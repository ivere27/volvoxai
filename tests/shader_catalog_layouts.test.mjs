import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import test from 'node:test';

import * as full from '../ts/generated/shaderCatalog.js';
import * as inference from '../ts/generated/shaderCatalogInference.js';

for (const [profile, catalog, header] of [
  ['full', full, 'shader_catalog.h'],
  ['inference', inference, 'shader_catalog_inference.h'],
]) {
  test(`${profile} shared shader layouts retain every native binding contract`, async () => {
    const native = await fs.readFile(new URL(`../native/src/backends/${header}`, import.meta.url), 'utf8');
    assert.equal(catalog.SHADER_CATALOG_HASH, native.match(/VX_SHADER_CATALOG_HASH "([a-f0-9]+)"/)[1]);
    const table = native.match(/vx_shader_interface\[VX_SHADER_COUNT\] = \{([\s\S]*?)\n\};/)[1];
    const rows = [...table.matchAll(/\{ \{([\d, ]+)\}, (\d+), (\d+), (\d+), \d+, \{[\d, ]+\}, \d+ \}, \/\* (.+) \*\//g)];
    assert.equal(rows.length, catalog.SHADER_COUNT);
    assert.equal(catalog.SHADER_LAYOUTS.length, catalog.SHADER_COUNT);
    assert.ok(Object.isFrozen(catalog.SHADER_LAYOUTS));
    const shared = new Map();
    for (const [id, row] of rows.entries()) {
      const [, slots, count, writesMask, paramsBinding, name] = row;
      const expected = {
        slots: slots.split(',').slice(0, Number(count)).map(Number),
        writesMask: Number(writesMask), paramsBinding: Number(paramsBinding),
      };
      const layout = catalog.SHADER_LAYOUTS[id];
      assert.equal(catalog.SHADER_NAMES[id], name);
      assert.deepEqual(layout, expected, name);
      assert.ok(Object.isFrozen(layout), name);
      assert.ok(Object.isFrozen(layout.slots), name);
      const key = JSON.stringify(expected);
      if (shared.has(key)) assert.equal(layout, shared.get(key), name);
      else shared.set(key, layout);
    }
    assert.ok(shared.size < catalog.SHADER_COUNT);
    assert.throws(() => { catalog.SHADER_LAYOUTS[0].writesMask = 0; }, TypeError);
    assert.throws(() => { catalog.SHADER_LAYOUTS[0].slots[0] = 255; }, TypeError);
  });
}
