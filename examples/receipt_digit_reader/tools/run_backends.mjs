/*
 * Read one receipt on each requested backend and compare the records.
 *
 * The point is not the record itself but the agreement: a package that decodes
 * differently across requested implementations has a backend problem, not a
 * model problem.
 * WebGPU is deliberately absent here -- it needs a browser, so it lives in
 * receipt_digit_reader.html.
 *
 *   node examples/receipt_digit_reader/tools/run_backends.mjs \
 *     --package build/receipt-digit-reader-fp32 \
 *     --raw receipt.f32 --backend wasm
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';

import { ReceiptDigitSession } from '../ReceiptDigitSession.js';

function fail(message) {
  throw new Error(`[receipt_digit_reader/run_backends] ${message}`);
}

function parseArguments(argv) {
  const options = {
    package: null, raw: null, backends: [], wasmUrl: null, repeat: 0, warmup: 3, json: false,
    includePrivateRecords: false,
  };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    const value = () => {
      const next = argv[++index];
      if (next === undefined) fail(`${flag} needs a value`);
      return next;
    };
    if (flag === '--package') options.package = value();
    else if (flag === '--raw') options.raw = value();
    else if (flag === '--wasm-url') options.wasmUrl = value();
    else if (flag === '--backend') options.backends.push(value());
    else if (flag === '--repeat') options.repeat = Number(value());
    else if (flag === '--warmup') options.warmup = Number(value());
    else if (flag === '--api') options.api = value();
    else if (flag === '--json') options.json = true;
    else if (flag === '--include-private-records') options.includePrivateRecords = true;
    else fail(`unknown option ${flag}`);
  }
  if (!options.package || !options.raw) fail('--package and --raw are required');
  if (options.backends.length === 0) options.backends = ['wasm'];
  if (!Number.isInteger(options.repeat) || options.repeat < 0
      || !Number.isInteger(options.warmup) || options.warmup < 0) {
    fail('--repeat and --warmup must be non-negative integers');
  }
  return options;
}

export function summarizeBackendRecords(records, includePrivateRecords = false) {
  const reference = records[0];
  return Object.freeze(records.map(({ phone, street, ...measurement }) => Object.freeze({
    ...measurement,
    decodedRecordStableAcrossRuns: Number.isSafeInteger(measurement.runs) ? true : null,
    decodedRecordMatchesFirstBackend:
      phone === reference.phone && street === reference.street,
    ...(includePrivateRecords ? { phone, street } : {}),
  })));
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

const fileFetch = async (source) => {
  const buffer = readFileSync(source.startsWith('file:') ? fileURLToPath(source) : source);
  return {
    ok: true,
    text: async () => buffer.toString('utf8'),
    json: async () => JSON.parse(buffer.toString('utf8')),
    arrayBuffer: async () => buffer.buffer.slice(
      buffer.byteOffset, buffer.byteOffset + buffer.byteLength,
    ),
  };
};

export async function runBackends(options) {
  const manifestBytes = readFileSync(`${options.package}/manifest.json`);
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const buffer = readFileSync(options.raw);
  const pixels = new Float32Array(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
  // Deno cannot resolve the repository TypeScript entry, so an explicit --api
  // points at the published bundle instead.
  const api = options.api
    ? await import(pathToFileURL(options.api).href)
    : undefined;
  const records = [];
  for (const backend of options.backends) {
    const started = performance.now();
    const session = await ReceiptDigitSession.open({
      manifest,
      graphUrl: pathToFileURL(`${options.package}/graph.json`).href,
      weightsUrl: pathToFileURL(`${options.package}/model.safetensors`).href,
      backend,
      fetch: fileFetch,
      ...(api ? { api } : {}),
      ...(options.wasmUrl ? { wasmUrl: pathToFileURL(options.wasmUrl).href } : {}),
    });
    try {
      const record = await session.read(pixels);
      const entry = {
        backend,
        phone: record.phone,
        street: record.street,
        ms: performance.now() - started,
      };
      if (options.repeat > 0) {
        // Session setup is deliberately outside the loop: folding model load,
        // compilation, and context creation into every sample would measure
        // the lifecycle, not the inference.
        for (let index = 0; index < options.warmup; index++) await session.read(pixels);
        const samples = [];
        for (let index = 0; index < options.repeat; index++) {
          const { record: measured, executionMs } = await session.readForBenchmark(pixels);
          samples.push(executionMs);
          // A route that stops producing the same record is not a benchmark
          // result; timing a wrong answer measures nothing worth comparing.
          if (measured.phone !== record.phone || measured.street !== record.street) {
            fail(`${backend} record changed between runs`);
          }
        }
        entry.runs = options.repeat;
        entry.median_ms = median(samples);
        entry.min_ms = Math.min(...samples);
        entry.max_ms = Math.max(...samples);
        entry.samples = samples;
      }
      records.push(entry);
    } finally {
      await session.close();
    }
  }
  const first = records[0];
  const agree = records.every(
    (entry) => entry.phone === first.phone && entry.street === first.street,
  );
  return Object.freeze({
    packageManifestSha256: createHash('sha256').update(manifestBytes).digest('hex'),
    variant: manifest.variant,
    records: summarizeBackendRecords(records, options.includePrivateRecords === true),
    agree,
    privacy: Object.freeze({
      decodedRecordsIncluded: options.includePrivateRecords === true,
      inputPathIncluded: false,
    }),
  });
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const options = parseArguments(process.argv.slice(2));
  const report = await runBackends(options);
  if (options.json) {
    process.stdout.write(`${JSON.stringify(report)}\n`);
  } else {
    for (const entry of report.records) {
      const timing = entry.median_ms === undefined
        ? `${entry.ms.toFixed(1)} ms`
        : `median ${entry.median_ms.toFixed(2)} ms over ${entry.runs} runs `
          + `(min ${entry.min_ms.toFixed(2)}, max ${entry.max_ms.toFixed(2)})`;
      const record = options.includePrivateRecords
        ? `phone=${entry.phone.padEnd(12)} street=${entry.street.padEnd(6)}`
        : `decoded-record=${entry.decodedRecordMatchesFirstBackend ? 'matches' : 'differs'}`;
      process.stdout.write(`${entry.backend.padEnd(8)} ${record} ${timing}\n`);
    }
    process.stdout.write(report.agree ? 'backends agree\n' : 'BACKENDS DISAGREE\n');
  }
  if (!report.agree) process.exitCode = 1;
}
