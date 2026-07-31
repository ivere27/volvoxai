#!/usr/bin/env node
/**
 * Run one or two split packages on the real receipt fixture and report what
 * actually matters after quantization: the router's family choice and the
 * decoder's argmax tokens.
 *
 * `compare_split_graphs.mjs` diffs two packages elementwise on synthetic
 * inputs, which is the right check for a semantics-preserving rewrite but the
 * wrong one for quantization — a W8A8 package is *supposed* to differ from
 * float32 numerically. What must not differ is the decision: the same router
 * family and the same tokens. This runs the real fixture and compares those.
 *
 * Execution uses `operatorFallback: 'forbid'`, so a package that needs an
 * operator the backend does not implement fails here rather than silently
 * falling back — which makes this a runnability check as well as an accuracy one.
 *
 *   node --import tsx examples/tiny_receipt_vqa/tools/verify_split_package.mjs \
 *     --package build/tr-w8a8 \
 *     [--reference build/tiny-receipt-hf-fp32] \
 *     [--fixtures build/tiny-receipt-e2e-artifacts/native-cpu] [--tokens 8]
 */

import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { Graph, GraphLoader, VolvoxAI } from '../../../ts/index.js';
import {
  nextDecoderToken,
  resolveDecoderInputs,
  resolveDecoderOutput,
} from './benchmark_heldout.mjs';

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

async function readTyped(path, Kind) {
  const raw = await readFile(path);
  return new Kind(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}

/** Resolve stable source semantics to their possibly generated runtime names. */
export function resolveFixtureInputSemantics(document, kind) {
  if (!['encoder', 'decoder'].includes(kind)) {
    throw new Error("fixture input kind must be 'encoder' or 'decoder'");
  }
  if (!document?.inputs || typeof document.inputs !== 'object'
      || Array.isArray(document.inputs)) {
    throw new Error(`current ${kind} graph must declare input descriptors`);
  }
  const allowed = kind === 'encoder'
    ? new Set(['image', 'question_ids', 'family_ids'])
    : new Set([
      'decoder_input_ids', 'memory', 'memory_padding_mask', 'family_ids', 'v4_keep',
    ]);
  const resolved = {};
  for (const [name, descriptor] of Object.entries(document.inputs)) {
    const semantic = descriptor?.source_name ?? name;
    if (typeof semantic !== 'string' || semantic.length === 0 || !allowed.has(semantic)) {
      throw new Error(`current ${kind} ABI has no input semantic '${semantic}'`);
    }
    if (resolved[semantic] !== undefined) {
      throw new Error(`current ${kind} ABI declares input semantic '${semantic}' more than once`);
    }
    resolved[semantic] = name;
  }
  return Object.freeze(resolved);
}

async function fixtureInputs(fixtures, document, kind) {
  const files = kind === 'encoder'
    ? { image: 'image.f32', question_ids: 'question_ids.i32', family_ids: 'family_ids.i32' }
    : {
      decoder_input_ids: 'decoder_input_ids.i32',
      memory: 'memory.f32',
      memory_padding_mask: 'memory_padding_mask.i32',
      family_ids: 'decoder_family_ids.i32',
    };
  const semantics = resolveFixtureInputSemantics(document, kind);
  const inputs = {};
  for (const [semantic, name] of Object.entries(semantics)) {
    const file = files[semantic];
    if (!file && semantic === 'v4_keep' && kind === 'decoder') {
      continue;
    }
    if (!file) throw new Error(`current ${kind} ABI has no input semantic '${semantic}'`);
    const Kind = file.endsWith('.i32') ? Int32Array : Float32Array;
    inputs[name] = await readTyped(join(fixtures, file), Kind);
  }
  if (semantics.v4_keep) {
    const name = semantics.v4_keep;
    const descriptor = document.inputs[name];
    const width = (descriptor.shape || []).reduce((a, b) => a * b, 1);
    const idsName = semantics.decoder_input_ids;
    const donor = idsName ? inputs[idsName] : null;
    if (!donor) throw new Error(`no fixture for graph input '${name}'`);
    if (!(donor instanceof Int32Array) || donor.length !== width) {
      throw new Error(`decoder_input_ids cannot derive graph input '${name}'`);
    }
    inputs[name] = Int32Array.from(donor, (value) => (value !== 0 ? 1 : 0));
  }
  return inputs;
}

async function createRunner(packageDir, kind, fixtures, backend = 'cpu') {
  const { graph, document } = await loadGraph(packageDir, kind);
  const inputs = await fixtureInputs(fixtures, document, kind);
  const runtime = await VolvoxAI.createRuntime({
    backends: [backend],
    ...(backend === 'wasm'
      ? { wasmUrl: pathToFileURL(resolve('dist/0.3.0/volvoxai.wasm')) }
      : {}),
  });
  const model = runtime.createModel(graph);
  const compiled = await model.compile({
    backend: { mode: 'require', backend, operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  return {
    document,
    inputs,
    async execute(executionInputs = inputs) {
      const started = performance.now();
      const result = await context.execute(executionInputs);
      try {
        const outputs = {};
        for (const name of graph.outputNames) {
          outputs[name] = await result.output(name).read();
        }
        return { outputs, elapsed: performance.now() - started };
      } finally {
        await result.close();
      }
    },
    async close() {
      await context.close();
      await compiled.close();
      await model.close();
      await runtime.close();
    },
  };
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index++) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}

/** Execute one ordinary full decoder forward for every generated prefix. */
export async function generateAutoregressiveTokens({
  contract,
  decoderInputs,
  baseInputs,
  execute,
  tokenCount,
  padTokenId,
  bosTokenId,
  eosTokenId,
}) {
  if (!Number.isSafeInteger(tokenCount) || tokenCount < 1 || tokenCount > 191) {
    throw new Error('token count must be an integer in [1, 191]');
  }
  if (typeof execute !== 'function') throw new Error('decoder execute callback is required');
  const decoderIds = new Int32Array(192).fill(padTokenId);
  decoderIds[0] = bosTokenId;
  const executionInputs = {
    ...baseInputs,
    [decoderInputs.decoder_input_ids]: decoderIds,
  };
  const decoderKeep = decoderInputs.v4_keep ? new Int32Array(192) : null;
  if (decoderKeep) {
    decoderKeep[0] = 1;
    executionInputs[decoderInputs.v4_keep] = decoderKeep;
  }
  const tokens = [];
  let elapsed = 0;
  for (let step = 0; step < tokenCount; step++) {
    const execution = await execute(executionInputs);
    elapsed += execution.elapsed ?? 0;
    const token = nextDecoderToken(contract, execution.outputs[contract.name], step);
    tokens.push(token);
    if (token === eosTokenId) break;
    decoderIds[step + 1] = token;
    if (decoderKeep) decoderKeep[step + 1] = 1;
  }
  return Object.freeze({ tokens: Object.freeze(tokens), elapsed });
}

async function describe(packageDir, fixtures, tokenCount, backend) {
  const encoder = await createRunner(packageDir, 'encoder', fixtures, backend);
  const decoder = await createRunner(packageDir, 'decoder', fixtures, backend);
  try {
    const manifest = JSON.parse(
      await readFile(join(packageDir, 'package_manifest.json'), 'utf8'),
    );
    const vocabulary = JSON.parse(await readFile(join(packageDir, 'vocab.json'), 'utf8'))
      ?.itos?.length;
    const contract = resolveDecoderOutput(decoder.document, manifest, vocabulary);
    const decoderInputs = resolveDecoderInputs(decoder.document, contract.kind);
    const tokenIds = manifest?.tokenizer?.token_ids;
    for (const name of ['pad', 'bos', 'eos']) {
      if (!Number.isSafeInteger(tokenIds?.[name])) {
        throw new Error(`package tokenizer must declare integer ${name} token ID`);
      }
    }
    const encoded = await encoder.execute();
    const routerLogits = encoded.outputs.router_logits;
    const generated = await generateAutoregressiveTokens({
      contract,
      decoderInputs,
      baseInputs: decoder.inputs,
      execute: (inputs) => decoder.execute(inputs),
      tokenCount,
      padTokenId: tokenIds.pad,
      bosTokenId: tokenIds.bos,
      eosTokenId: tokenIds.eos,
    });
    return {
      family: routerLogits ? argmax(routerLogits) : null,
      routerLogits: routerLogits ? Array.from(routerLogits.slice(0, 8)) : null,
      tokens: [...generated.tokens],
      encoderMs: encoded.elapsed,
      decoderMs: generated.elapsed,
    };
  } finally {
    await decoder.close();
    await encoder.close();
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (!options.package) throw new Error('pass --package <dir>');
  const fixtures = options.fixtures
    ?? 'build/tiny-receipt-e2e-artifacts/native-cpu';
  const tokenCount = Number(options.tokens ?? 8);
  const backend = options.backend ?? 'cpu';

  const actual = await describe(options.package, fixtures, tokenCount, backend);
  console.log(`package  ${options.package}  [${backend}]`);
  console.log(`  router family ${actual.family}  tokens ${JSON.stringify(actual.tokens)}`);
  console.log(`  encoder ${actual.encoderMs.toFixed(0)} ms   decoder ${actual.decoderMs.toFixed(0)} ms`);

  if (!options.reference) return;
  const expected = await describe(options.reference, fixtures, tokenCount, backend);
  console.log(`reference ${options.reference}`);
  console.log(`  router family ${expected.family}  tokens ${JSON.stringify(expected.tokens)}`);
  console.log(`  encoder ${expected.encoderMs.toFixed(0)} ms   decoder ${expected.decoderMs.toFixed(0)} ms`);

  const sameFamily = actual.family === expected.family;
  const sameTokens = JSON.stringify(actual.tokens) === JSON.stringify(expected.tokens);
  console.log(`\nrouter family ${sameFamily ? 'MATCH' : 'DIFFERS'} | `
    + `autoregressive tokens ${sameTokens ? 'MATCH' : 'DIFFER'}`);
  console.log(`speedup: encoder ${(expected.encoderMs / actual.encoderMs).toFixed(2)}x, `
    + `decoder ${(expected.decoderMs / actual.decoderMs).toFixed(2)}x`);
  if (!sameFamily || !sameTokens) process.exitCode = 1;
}

if (process.argv[1]
    && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
