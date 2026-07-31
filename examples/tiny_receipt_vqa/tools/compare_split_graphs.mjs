#!/usr/bin/env node
/**
 * Numerically compare two split packages on identical inputs.
 *
 * The optimizer's attention passes (SDPA fusion, head-layout normalization,
 * additive -> I32 keep mask) claim to be semantics-preserving. This checks that
 * claim the only way that counts: run the graph before and after, on the same
 * deterministic inputs, and diff the outputs elementwise.
 *
 *   node examples/tiny_receipt_vqa/tools/compare_split_graphs.mjs \
 *     --a build/tiny-receipt-hf-fp32 --b build/tiny-receipt-fp32-fused \
 *     [--graph encoder] [--tolerance 1e-4]
 */

import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { Graph, GraphLoader, VolvoxAI } from '../../../ts/index.js';

function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index++) {
    const name = values[index].replace(/^--/, '');
    result[name] = values[index + 1]?.startsWith('--') ? true : values[++index];
  }
  return result;
}

function packageFetch(graphDocument, weightsBuffer) {
  return async (url) => url === 'graph.json'
    ? { ok: true, json: async () => graphDocument }
    : url === 'model.safetensors'
      ? { ok: true, arrayBuffer: async () => weightsBuffer.slice(0) }
      : { ok: false, statusText: `unexpected source ${url}` };
}

async function loadGraph(packageDir, kind) {
  const graphDocument = JSON.parse(
    await readFile(join(packageDir, kind, 'graph.json'), 'utf8'),
  );
  const weights = await readFile(join(packageDir, kind, 'model.safetensors'));
  const graph = new Graph();
  await GraphLoader.load(graph, 'model.safetensors', {
    graphUrl: 'graph.json',
    fetch: packageFetch(graphDocument, weights.buffer.slice(
      weights.byteOffset, weights.byteOffset + weights.byteLength,
    )),
  });
  return { graph, document: graphDocument };
}

/** Deterministic pseudo-random inputs, identical for both packages. */
function makeInputs(document, seed = 12345) {
  let state = seed >>> 0;
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const inputs = {};
  for (const [name, descriptor] of Object.entries(document.inputs || {})) {
    const count = (descriptor.shape || []).reduce((a, b) => a * b, 1);
    if (descriptor.dtype === 'int32') {
      const values = new Int32Array(count);
      // Token-ish ids and 0/1 masks both live in a small non-negative range.
      for (let i = 0; i < count; i++) values[i] = Math.floor(next() * 8);
      inputs[name] = values;
    } else {
      const values = new Float32Array(count);
      for (let i = 0; i < count; i++) values[i] = next() * 2 - 1;
      inputs[name] = values;
    }
  }
  return inputs;
}

async function run(packageDir, kind, inputs) {
  const { graph } = await loadGraph(packageDir, kind);
  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const model = runtime.createModel(graph);
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const result = await context.execute(inputs);
  const outputs = {};
  for (const name of graph.outputNames) {
    outputs[name] = Float32Array.from(await result.output(name).read());
  }
  await result.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
  return outputs;
}

function compare(a, b, tolerance) {
  const report = [];
  for (const name of Object.keys(a)) {
    const left = a[name];
    const right = b[name];
    if (!right || left.length !== right.length) {
      report.push({ name, ok: false, reason: 'missing or length mismatch' });
      continue;
    }
    let maxAbsolute = 0;
    let maxRelative = 0;
    for (let i = 0; i < left.length; i++) {
      const absolute = Math.abs(left[i] - right[i]);
      if (absolute > maxAbsolute) maxAbsolute = absolute;
      const scale = Math.max(Math.abs(left[i]), Math.abs(right[i]), 1e-6);
      if (absolute / scale > maxRelative) maxRelative = absolute / scale;
    }
    report.push({
      name,
      elements: left.length,
      maxAbsolute,
      maxRelative,
      ok: maxAbsolute <= tolerance,
    });
  }
  return report;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const a = options.a;
  const b = options.b;
  if (!a || !b) throw new Error('pass --a <package> --b <package>');
  const kinds = options.graph ? [options.graph] : ['encoder', 'decoder'];
  const tolerance = Number(options.tolerance ?? 1e-4);

  let failed = false;
  for (const kind of kinds) {
    const { document } = await loadGraph(a, kind);
    const inputs = makeInputs(document);
    const left = await run(a, kind, inputs);
    const right = await run(b, kind, inputs);
    console.log(`--- ${kind} ---`);
    for (const row of compare(left, right, tolerance)) {
      const status = row.ok ? 'MATCH' : 'DIFFER';
      console.log(
        `  ${status}  ${row.name}  n=${row.elements}  `
        + `maxAbs=${row.maxAbsolute?.toExponential(3)}  `
        + `maxRel=${row.maxRelative?.toExponential(3)}`,
      );
      if (!row.ok) failed = true;
    }
  }
  console.log(failed ? '\nRESULT: outputs differ' : '\nRESULT: outputs match');
  process.exitCode = failed ? 1 : 0;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
