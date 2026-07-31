#!/usr/bin/env node
/**
 * Per-operator timing for one split graph on the WASM backend.
 *
 * Node counts and element traffic say what a graph *asks* the runtime to do;
 * they say nothing about whether each operator lands on a fast kernel. Those
 * two came apart badly here: an encoder carrying a third of the imported
 * package's traffic ran 2.7x slower, because its 13 `QConv2D` nodes were
 * emitted without a bias and `packedEligible` in `WasmEngine.ts` silently
 * dropped them from the packed im2col SIMD path onto the scalar
 * `qconv2d_i8u8`. Same operator, same count, 10.4x the time — invisible to any
 * inspection of the graph, obvious in one profile.
 *
 * So: measure before optimizing, and measure again after, on the backend that
 * ships. Two hypotheses about which kernel dominated were wrong before this
 * script settled it.
 *
 *   node --import tsx examples/tiny_receipt_vqa/tools/profile_wasm_ops.mjs \
 *     --package build/tiny-receipt-w8a8 [--graph encoder] \
 *     [--fixtures build/tiny-receipt-e2e-artifacts/native-cpu]
 */

import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { Graph, GraphLoader, VolvoxAI } from '../../../ts/index.js';

function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index++) {
    const name = values[index].replace(/^--/, '');
    result[name] = values[index + 1]?.startsWith('--') ? true : values[++index];
  }
  return result;
}

async function loadGraph(packageDir, kind) {
  const document = JSON.parse(
    await readFile(join(packageDir, kind, 'graph.json'), 'utf8'),
  );
  const weights = await readFile(join(packageDir, kind, 'model.safetensors'));
  const graph = new Graph();
  await GraphLoader.load(graph, 'model.safetensors', {
    graphUrl: 'graph.json',
    fetch: async (url) => url === 'graph.json'
      ? { ok: true, json: async () => document }
      : {
        ok: true,
        arrayBuffer: async () => weights.buffer.slice(
          weights.byteOffset, weights.byteOffset + weights.byteLength,
        ),
      },
  });
  return { graph, document };
}

async function fixtureInputs(fixtures, document, kind) {
  const files = kind === 'encoder'
    ? { image: 'image.f32', question_ids: 'question_ids.i32', family_ids: 'family_ids.i32' }
    : {
      decoder_input_ids: 'decoder_input_ids.i32',
      memory: 'memory.f32',
      memory_padding_mask: 'memory_padding_mask.i32',
      family_ids: 'family_ids.i32',
    };
  const inputs = {};
  const derived = [];
  for (const [name, descriptor] of Object.entries(document.inputs || {})) {
    const file = files[descriptor.source_name || name];
    if (!file) {
      // A hoisted keep mask has no fixture of its own; the caller derives it.
      derived.push([name, descriptor]);
      continue;
    }
    const raw = await readFile(join(fixtures, file));
    const Kind = file.endsWith('.i32') ? Int32Array : Float32Array;
    inputs[name] = new Kind(
      raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
    );
  }
  for (const [name, descriptor] of derived) {
    const width = (descriptor.shape || []).reduce((a, b) => a * b, 1);
    const donor = Object.values(inputs).find(
      (values) => values instanceof Int32Array && values.length === width,
    );
    if (!donor) throw new Error(`no fixture for graph input '${name}'`);
    inputs[name] = Int32Array.from(donor, (value) => (value !== 0 ? 1 : 0));
  }
  return inputs;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (!options.package) throw new Error('pass --package <dir>');
  const kind = options.graph ?? 'encoder';
  const fixtures = options.fixtures
    ?? 'build/tiny-receipt-e2e-artifacts/native-cpu';

  const { graph, document } = await loadGraph(options.package, kind);
  const inputs = await fixtureInputs(fixtures, document, kind);
  const runtime = await VolvoxAI.createRuntime({
    backends: ['wasm'],
    wasmUrl: pathToFileURL(resolve('dist/0.3.0/volvoxai.wasm')),
  });
  const model = runtime.createModel(graph);
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();

  const profile = new Map();
  globalThis.__VOLVOX_WASM_PROFILE = profile;
  const started = performance.now();
  const result = await context.execute(inputs);
  const total = performance.now() - started;
  for (const name of graph.outputNames) await result.output(name).read();
  globalThis.__VOLVOX_WASM_PROFILE = undefined;

  console.log(`\n=== ${options.package} / ${kind} — ${total.toFixed(0)} ms ===`);
  console.log('operator              count        ms      %');
  const rows = [...profile.entries()].sort((a, b) => b[1].ms - a[1].ms);
  for (const [op, { ms, count }] of rows) {
    console.log(`${op.padEnd(21)} ${String(count).padStart(5)} ${ms.toFixed(0).padStart(9)}`
      + ` ${(100 * ms / total).toFixed(1).padStart(6)}`);
  }

  await result.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
