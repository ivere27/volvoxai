import { Tensor } from '../ts/core/Tensor.js';
import { _cpuConv2D } from '../ts/ops/conv2D.js';
import { _cpuMatMul } from '../ts/ops/matMul.js';
import { _cpuQConv2D } from '../ts/ops/qConv2D.js';
import { _cpuQLinear } from '../ts/ops/qLinear.js';

// This intentionally measures direct portable CPU(JS) kernels. It can be
// imported by a browser module, while the CLI below is useful in Node.
export const DEFAULT_OPTIONS = Object.freeze({
  op: 'all', warmup: 3, iterations: 20, seed: 0x564f4c58, json: false, dryRun: false,
  linear: Object.freeze({ rows: 8, inputFeatures: 256, outputFeatures: 256 }),
  conv: Object.freeze({ batch: 1, height: 16, width: 16, inputChannels: 16, outputChannels: 32, kernel: 3 }),
});

const MAX_TENSOR_ELEMENTS = 64 * 1024 * 1024;

function integer(value, label, minimum) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum) throw new Error(`${label} must be an integer >= ${minimum}.`);
  return parsed;
}

function elements(shape, label) {
  return shape.reduce((total, dimension) => {
    const next = total * dimension;
    if (!Number.isSafeInteger(next) || next > MAX_TENSOR_ELEMENTS) {
      throw new Error(`${label} exceeds the ${MAX_TENSOR_ELEMENTS}-element benchmark limit.`);
    }
    return next;
  }, 1);
}

function normalizeOp(value) {
  const op = String(value).toLowerCase();
  if (op === 'all') return op;
  if (op === 'linear' || op === 'qlinear') return 'qlinear';
  if (op === 'conv' || op === 'qconv' || op === 'qconv2d') return 'qconv';
  throw new Error(`--op must be all, qlinear, or qconv (received '${value}').`);
}

function normalizeOptions(source = {}) {
  const options = {
    ...DEFAULT_OPTIONS,
    ...source,
    linear: { ...DEFAULT_OPTIONS.linear, ...(source.linear || {}) },
    conv: { ...DEFAULT_OPTIONS.conv, ...(source.conv || {}) },
  };
  options.op = normalizeOp(options.op);
  options.warmup = integer(options.warmup, 'warmup', 0);
  options.iterations = integer(options.iterations, 'iterations', 1);
  options.seed = integer(options.seed, 'seed', 0) >>> 0;
  options.json = Boolean(options.json);
  options.dryRun = Boolean(options.dryRun);
  for (const [key, label] of [['rows', 'linear rows'], ['inputFeatures', 'linear input features'], ['outputFeatures', 'linear output features']]) {
    options.linear[key] = integer(options.linear[key], label, 1);
  }
  for (const [key, label] of [
    ['batch', 'conv batch'], ['height', 'conv height'], ['width', 'conv width'],
    ['inputChannels', 'conv input channels'], ['outputChannels', 'conv output channels'], ['kernel', 'conv kernel'],
  ]) options.conv[key] = integer(options.conv[key], label, 1);
  if (options.conv.kernel % 2 === 0) throw new Error('conv kernel must be odd so FP32 has the same output shape.');
  elements([options.linear.rows, options.linear.inputFeatures], 'QLinear input');
  elements([options.linear.outputFeatures, options.linear.inputFeatures], 'QLinear weight');
  elements([options.linear.rows, options.linear.outputFeatures], 'QLinear output');
  elements([options.conv.batch, options.conv.height, options.conv.width, options.conv.inputChannels], 'QConv2D input');
  elements([options.conv.outputChannels, options.conv.kernel, options.conv.kernel, options.conv.inputChannels], 'QConv2D weight');
  elements([options.conv.batch, options.conv.height, options.conv.width, options.conv.outputChannels], 'QConv2D output');
  return options;
}

function parseDimensions(value, length, label) {
  const fields = String(value).split(',').map((field) => field.trim());
  if (fields.length !== length || fields.some((field) => field === '')) {
    throw new Error(`${label} expects ${length} comma-separated positive integers.`);
  }
  return fields.map((field, index) => integer(field, `${label} field ${index + 1}`, 1));
}

export function parseBenchmarkArguments(argv = []) {
  const options = { linear: {}, conv: {} };
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    const next = () => {
      const value = argv[++index];
      if (value == null || value.startsWith('--')) throw new Error(`${argument} requires a value.`);
      return value;
    };
    if (argument === '--op') options.op = next();
    else if (argument === '--warmup') options.warmup = next();
    else if (argument === '--iterations') options.iterations = next();
    else if (argument === '--seed') options.seed = next();
    else if (argument === '--linear') {
      const [rows, inputFeatures, outputFeatures] = parseDimensions(next(), 3, '--linear');
      options.linear = { rows, inputFeatures, outputFeatures };
    } else if (argument === '--conv') {
      const [batch, height, width, inputChannels, outputChannels, kernel] = parseDimensions(next(), 6, '--conv');
      options.conv = { batch, height, width, inputChannels, outputChannels, kernel };
    } else if (argument === '--json') options.json = true;
    else if (argument === '--dry-run') options.dryRun = true;
    else if (argument === '--help' || argument === '-h') options.help = true;
    else throw new Error(`Unknown argument '${argument}'. Use --help for usage.`);
  }
  return options.help ? options : normalizeOptions(options);
}

function prng(seed) {
  let state = seed || 0x6d2b79f5;
  return () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0);
}

function fillI8(length, random, magnitude) {
  const values = new Int8Array(length);
  for (let index = 0; index < length; index++) values[index] = (random() % (magnitude * 2 + 1)) - magnitude;
  return values;
}

function tensor(name, shape, dtype, buffer, quantization = null) {
  const value = new Tensor(name, shape, dtype, false, { quantization });
  value.buffer = buffer;
  return value;
}

function weightQuantization(channels) {
  return {
    scheme: 'per_axis', axis: 0,
    scales: Array.from({ length: channels }, (_, index) => Math.fround(1 / (64 + (index % 5) * 8))),
    zero_points: Array.from({ length: channels }, (_, index) => (index % 7) - 3),
  };
}

function dequantizeInput(values, scale, zeroPoint) {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index++) output[index] = Math.fround((values[index] - zeroPoint) * scale);
  return output;
}

function dequantizeOutputMajor(values, outputChannels, inner, quantization) {
  const output = new Float32Array(values.length);
  for (let channel = 0; channel < outputChannels; channel++) {
    const { scales, zero_points: zeroPoints } = quantization;
    for (let index = 0; index < inner; index++) {
      const offset = channel * inner + index;
      output[offset] = Math.fround((values[offset] - zeroPoints[channel]) * scales[channel]);
    }
  }
  return output;
}

function dequantizeBias(values, inputScale, quantization) {
  return Float32Array.from(values, (value, channel) => Math.fround(value * inputScale * quantization.scales[channel]));
}

function footprint(input, weight, bias, output, sidecars = []) {
  const sidecarBytes = sidecars.reduce((total, value) => total + value.sizeBytes, 0);
  const inputBytes = input.sizeBytes;
  const weightBytes = weight.sizeBytes;
  const biasBytes = bias?.sizeBytes || 0;
  const outputBytes = output.sizeBytes;
  return {
    input_bytes: inputBytes, weight_bytes: weightBytes, bias_bytes: biasBytes, output_bytes: outputBytes,
    sidecar_parameter_bytes: sidecarBytes,
    activation_bytes: inputBytes + outputBytes,
    model_parameter_bytes: weightBytes + biasBytes + sidecarBytes,
    live_tensor_bytes: inputBytes + weightBytes + biasBytes + outputBytes + sidecarBytes,
    accounting: 'Raw typed tensor buffers only; canonical W8A8 graph descriptors are excluded.',
  };
}

function variant(id, label, dtypePath, execute, output, storage) {
  return { id, label, dtype_path: dtypePath, execute, output, storage };
}

function makeLinear(options) {
  const { rows, inputFeatures: inputFeatures, outputFeatures: outputFeatures } = options.linear;
  const random = prng(options.seed ^ 0x4c494e45);
  const inputScale = Math.fround(1 / 32);
  const inputZeroPoint = -3;
  const outputQuantization = { scheme: 'per_tensor', scale: Math.fround(1 / 16), zero_point: -2 };
  const inputQuantization = { scheme: 'per_tensor', scale: inputScale, zero_point: inputZeroPoint };
  const quantization = weightQuantization(outputFeatures);
  const inputValues = fillI8(rows * inputFeatures, random, 16);
  const weightValues = fillI8(outputFeatures * inputFeatures, random, 8);
  const biasValues = Int32Array.from({ length: outputFeatures }, () => (random() % 129) - 64);

  const qInput = tensor('qlinear_input_i8', [rows, inputFeatures], 'int8', inputValues, inputQuantization);
  const qWeight = tensor('qlinear_weight_i8', [outputFeatures, inputFeatures], 'int8', weightValues, quantization);
  const qBias = tensor('qlinear_bias_i32', [outputFeatures], 'int32', biasValues);
  const qOutput = tensor('qlinear_output_i8', [rows, outputFeatures], 'int8', new Int8Array(rows * outputFeatures), outputQuantization);
  const qNode = { id: 'benchmark_qlinear_w8a8', opType: 'QLinear', inputs: { input: qInput, weight: qWeight, bias: qBias }, outputs: { out: qOutput }, params: {} };

  const floatInputValues = dequantizeInput(inputValues, inputScale, inputZeroPoint);
  const floatWeightValues = dequantizeOutputMajor(weightValues, outputFeatures, inputFeatures, quantization);
  const floatBiasValues = dequantizeBias(biasValues, inputScale, quantization);
  const f32Input = tensor('linear_input_f32', [rows, inputFeatures], 'float32', floatInputValues);
  const f32Weight = tensor('linear_weight_f32', [outputFeatures, inputFeatures], 'float32', floatWeightValues);
  const f32Bias = tensor('linear_bias_f32', [outputFeatures], 'float32', floatBiasValues);
  const f32Output = tensor('linear_output_f32', [rows, outputFeatures], 'float32', new Float32Array(rows * outputFeatures));
  const f32Node = { id: 'benchmark_linear_fp32', opType: 'MatMul', inputs: { input: f32Input, weight: f32Weight, bias: f32Bias }, outputs: { out: f32Output }, wLayout: 'dout', params: {} };

  const w8a32Input = tensor('w8a32_input_f32', [rows, inputFeatures], 'float32', floatInputValues.slice());
  const w8a32Weight = tensor('w8a32_weight_i8', [outputFeatures, inputFeatures], 'int8', weightValues.slice());
  const w8a32Scale = tensor('w8a32_weight_scale', [outputFeatures], 'float32', Float32Array.from(quantization.scales));
  const w8a32ZeroPoint = tensor('w8a32_weight_zero_point', [outputFeatures], 'int32', Int32Array.from(quantization.zero_points));
  const w8a32Bias = tensor('w8a32_bias_f32', [outputFeatures], 'float32', floatBiasValues.slice());
  const w8a32Output = tensor('w8a32_output_f32', [rows, outputFeatures], 'float32', new Float32Array(rows * outputFeatures));
  const w8a32Node = {
    id: 'benchmark_linear_w8a32', opType: 'MatMul', wLayout: 'dout', params: {}, outputs: { out: w8a32Output },
    inputs: { input: w8a32Input, weight: w8a32Weight, weight_scale: w8a32Scale, weight_zero_point: w8a32ZeroPoint, bias: w8a32Bias },
  };

  return {
    op: 'QLinear', shape: { rows, input_features: inputFeatures, output_features: outputFeatures },
    macs_per_iteration: rows * inputFeatures * outputFeatures,
    comparison: 'Equivalent dequantized inputs, weights, and bias. W8A8 requantizes output to I8; W8A32 and FP32 produce F32 output.',
    variants: [
      variant('canonical_w8a8', 'Canonical W8A8 QLinear (I8 activations/weights/output, I32 bias)', 'I8 × I8 -> I32 -> I8', () => _cpuQLinear(qNode), qOutput, footprint(qInput, qWeight, qBias, qOutput)),
      variant('w8a32_reference', 'Existing W8A32 MatMul reference (F32 activations/output, I8 weights)', 'F32 × I8 -> F32', () => _cpuMatMul(w8a32Node), w8a32Output, footprint(w8a32Input, w8a32Weight, w8a32Bias, w8a32Output, [w8a32Scale, w8a32ZeroPoint])),
      variant('fp32_reference', 'FP32 MatMul reference', 'F32 × F32 -> F32', () => _cpuMatMul(f32Node), f32Output, footprint(f32Input, f32Weight, f32Bias, f32Output)),
    ],
  };
}

function convFloatWeight(values, outputChannels, kernel, inputChannels, quantization) {
  const output = new Float32Array(values.length);
  for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
    for (let kernelY = 0; kernelY < kernel; kernelY++) for (let kernelX = 0; kernelX < kernel; kernelX++) {
      for (let inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
        const source = (((outputChannel * kernel + kernelY) * kernel + kernelX) * inputChannels) + inputChannel;
        const target = (((kernelY * kernel + kernelX) * inputChannels + inputChannel) * outputChannels) + outputChannel;
        output[target] = Math.fround((values[source] - quantization.zero_points[outputChannel]) * quantization.scales[outputChannel]);
      }
    }
  }
  return output;
}

function makeConv(options) {
  const { batch, height, width, inputChannels, outputChannels, kernel } = options.conv;
  const random = prng(options.seed ^ 0x434f4e56);
  const inputScale = Math.fround(1 / 32);
  const inputZeroPoint = -3;
  const inputQuantization = { scheme: 'per_tensor', scale: inputScale, zero_point: inputZeroPoint };
  const outputQuantization = { scheme: 'per_tensor', scale: Math.fround(1 / 16), zero_point: -2 };
  const quantization = weightQuantization(outputChannels);
  const inputValues = fillI8(batch * height * width * inputChannels, random, 16);
  const weightValues = fillI8(outputChannels * kernel * kernel * inputChannels, random, 8);
  const biasValues = Int32Array.from({ length: outputChannels }, () => (random() % 129) - 64);
  const outputElements = batch * height * width * outputChannels;
  const padding = Math.floor(kernel / 2);
  const pads = [padding, padding, padding, padding];

  const qInput = tensor('qconv_input_i8', [batch, height, width, inputChannels], 'int8', inputValues, inputQuantization);
  const qWeight = tensor('qconv_weight_i8', [outputChannels, kernel, kernel, inputChannels], 'int8', weightValues, quantization);
  const qBias = tensor('qconv_bias_i32', [outputChannels], 'int32', biasValues);
  const qOutput = tensor('qconv_output_i8', [batch, height, width, outputChannels], 'int8', new Int8Array(outputElements), outputQuantization);
  const qNode = {
    id: 'benchmark_qconv_w8a8', opType: 'QConv2D', inputs: { input: qInput, weight: qWeight, bias: qBias }, outputs: { out: qOutput },
    params: { data_layout: 'NHWC', weight_layout: 'OHWI', groups: 1, stride: 1, dilation: 1, padding, pads },
  };

  const f32Input = tensor('conv_input_f32', [batch, height, width, inputChannels], 'float32', dequantizeInput(inputValues, inputScale, inputZeroPoint));
  const f32Weight = tensor('conv_weight_f32', [kernel, kernel, inputChannels, outputChannels], 'float32', convFloatWeight(weightValues, outputChannels, kernel, inputChannels, quantization));
  const f32Bias = tensor('conv_bias_f32', [outputChannels], 'float32', dequantizeBias(biasValues, inputScale, quantization));
  const f32Output = tensor('conv_output_f32', [batch, height, width, outputChannels], 'float32', new Float32Array(outputElements));
  const f32Node = {
    id: 'benchmark_conv_fp32', opType: 'Conv2D', inputs: { input: f32Input, weight: f32Weight, bias: f32Bias }, outputs: { out: f32Output },
    params: { groups: 1, stride: 1, dilation: 1, padding, pads },
  };

  return {
    op: 'QConv2D',
    shape: { batch, height, width, input_channels: inputChannels, output_channels: outputChannels, kernel: [kernel, kernel], groups: 1 },
    macs_per_iteration: batch * height * width * outputChannels * kernel * kernel * inputChannels,
    comparison: 'Equivalent dequantized inputs, OHWI/HWIO-transposed weights, and bias. W8A8 requantizes output to I8.',
    unavailable_variants: [{
      id: 'w8a32_reference',
      reason: 'CPU(JS) implements W8A32 for MatMul/Linear, but has no corresponding W8A32 Conv2D reference kernel.',
    }],
    variants: [
      variant('canonical_w8a8', 'Canonical W8A8 QConv2D (NHWC/OHWI I8, I32 bias)', 'I8 × I8 -> I32 -> I8', () => _cpuQConv2D(qNode), qOutput, footprint(qInput, qWeight, qBias, qOutput)),
      variant('fp32_reference', 'FP32 Conv2D reference', 'F32 × F32 -> F32', () => _cpuConv2D(f32Node), f32Output, footprint(f32Input, f32Weight, f32Bias, f32Output)),
    ],
  };
}

function workloads(options) {
  const result = [];
  if (options.op === 'all' || options.op === 'qlinear') result.push(makeLinear(options));
  if (options.op === 'all' || options.op === 'qconv') result.push(makeConv(options));
  return result;
}

function checksum(output) {
  const bytes = new Uint8Array(output.buffer.buffer, output.buffer.byteOffset, output.buffer.byteLength);
  let hash = 0x811c9dc5;
  for (const value of bytes) hash = Math.imul(hash ^ value, 0x01000193) >>> 0;
  return `0x${hash.toString(16).padStart(8, '0')}`;
}

function timeVariant(item, macs, warmup, iterations) {
  const now = () => globalThis.performance?.now?.() ?? Date.now();
  let warmupTotal = 0;
  if (warmup) {
    const start = now();
    for (let iteration = 0; iteration < warmup; iteration++) item.execute();
    warmupTotal = now() - start;
  }
  const samples = [];
  const start = now();
  for (let iteration = 0; iteration < iterations; iteration++) {
    const iterationStart = now();
    item.execute();
    samples.push(now() - iterationStart);
  }
  const total = now() - start;
  const mean = total / iterations;
  return {
    warmup_iterations: warmup, warmup_total_ms: warmupTotal, iterations,
    total_ms: total, mean_ms: mean, min_ms: Math.min(...samples), max_ms: Math.max(...samples),
    throughput_gmac_per_s: mean > 0 ? macs / (mean * 1e6) : null,
  };
}

function header(options, dryRun) {
  return {
    schema: 'volvoxai.w8a8-benchmark/v1', dry_run: dryRun,
    execution: {
      backend: 'CPU(JS) portable reference kernels',
      scope: 'Direct operator calls only; excludes graph construction/loading, CPUEngine scheduling, WASM, WebGPU, native backends, and model I/O.',
      clock: 'performance.now() when available',
    },
    options: {
      op: options.op, warmup: options.warmup, iterations: options.iterations, seed: options.seed,
      linear: { rows: options.linear.rows, input_features: options.linear.inputFeatures, output_features: options.linear.outputFeatures },
      conv: { batch: options.conv.batch, height: options.conv.height, width: options.conv.width, input_channels: options.conv.inputChannels, output_channels: options.conv.outputChannels, kernel: options.conv.kernel },
    },
    limitations: [
      'This is a CPU(JS) reference-kernel benchmark, not a WASM, WebGPU, Vulkan, OpenGL, Metal, or native CPU performance claim.',
      'QLinear compares canonical W8A8, existing W8A32 MatMul, and FP32 MatMul. QConv2D compares canonical W8A8 with FP32 only because no W8A32 CPU(JS) Conv2D reference exists.',
      'Storage counts raw typed tensor buffers used by one invocation. Canonical W8A8 quantization descriptors are graph metadata and are excluded from raw-payload totals.',
    ],
  };
}

function reportWorkload(workload, timings = null) {
  const report = {
    op: workload.op, shape: workload.shape, macs_per_iteration: workload.macs_per_iteration,
    comparison: workload.comparison,
    variants: workload.variants.map((item) => {
      const result = { id: item.id, label: item.label, dtype_path: item.dtype_path, storage: item.storage };
      const timing = timings?.get(item.id);
      if (timing) Object.assign(result, { timing_ms: timing, result_checksum: checksum(item.output) });
      return result;
    }),
  };
  if (workload.unavailable_variants) report.unavailable_variants = workload.unavailable_variants;
  return report;
}

export function createBenchmarkPlan(source = {}) {
  const options = normalizeOptions(source);
  return { ...header(options, true), workloads: workloads(options).map((workload) => reportWorkload(workload)) };
}

export function runBenchmark(source = {}) {
  const options = normalizeOptions(source);
  if (options.dryRun) return createBenchmarkPlan(options);
  return {
    ...header(options, false),
    workloads: workloads(options).map((workload) => {
      const timings = new Map(workload.variants.map((item) => [
        item.id, timeVariant(item, workload.macs_per_iteration, options.warmup, options.iterations),
      ]));
      return reportWorkload(workload, timings);
    }),
  };
}

function bytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(2)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MiB`;
}

export function formatBenchmarkReport(report) {
  const lines = ['VolvoxAI W8A8 benchmark', `Backend: ${report.execution.backend}`, `Scope: ${report.execution.scope}`];
  if (report.dry_run) lines.push('Dry run: no kernels executed.');
  for (const workload of report.workloads) {
    const shape = Object.entries(workload.shape).map(([key, value]) => `${key}=${Array.isArray(value) ? value.join('x') : value}`).join(', ');
    lines.push('', `${workload.op} (${shape}; ${workload.macs_per_iteration} MACs/iteration)`, `  Comparison: ${workload.comparison}`);
    for (const item of workload.variants) {
      const storage = item.storage;
      lines.push(`  ${item.id}: ${item.dtype_path}; live buffers ${bytes(storage.live_tensor_bytes)} (activations ${bytes(storage.activation_bytes)}, parameters ${bytes(storage.model_parameter_bytes)})`);
      if (item.timing_ms) {
        const timing = item.timing_ms;
        const throughput = Number.isFinite(timing.throughput_gmac_per_s) ? `${timing.throughput_gmac_per_s.toFixed(3)} GMAC/s` : 'n/a GMAC/s';
        lines.push(`    warmup ${timing.warmup_iterations} in ${timing.warmup_total_ms.toFixed(3)} ms; ${timing.iterations} iterations: mean ${timing.mean_ms.toFixed(3)} ms, min ${timing.min_ms.toFixed(3)} ms, max ${timing.max_ms.toFixed(3)} ms, ${throughput}; checksum ${item.result_checksum}`);
      }
    }
    for (const unavailable of workload.unavailable_variants || []) lines.push(`  ${unavailable.id}: unavailable — ${unavailable.reason}`);
  }
  lines.push('', 'Limitations:', ...report.limitations.map((value) => `  - ${value}`));
  return lines.join('\n');
}

export function helpText() {
  return [
    'Usage: npx tsx tools/benchmark_w8a8.mjs [options]', '',
    'Measures direct CPU(JS) portable kernels; it does not measure WASM, WebGPU, or native backends.', '',
    '  --op all|qlinear|qconv', '  --warmup N', '  --iterations N', '  --seed N',
    '  --linear rows,input,out', '  --conv batch,height,width,in,out,kernel  (groups=1, stride=1)',
    '  --dry-run  deterministic plan and storage only', '  --json', '  --help, -h', '',
    'Examples:', '  npx tsx tools/benchmark_w8a8.mjs --op qlinear --warmup 5 --iterations 50',
    '  npx tsx tools/benchmark_w8a8.mjs --dry-run --json',
  ].join('\n');
}

export function main(argv = (typeof process !== 'undefined' ? process.argv.slice(2) : [])) {
  const options = parseBenchmarkArguments(argv);
  if (options.help) {
    console.log(helpText());
    return null;
  }
  const report = options.dryRun ? createBenchmarkPlan(options) : runBenchmark(options);
  console.log(options.json ? JSON.stringify(report, null, 2) : formatBenchmarkReport(report));
  return report;
}

if (typeof process !== 'undefined' && typeof process.argv?.[1] === 'string' &&
    process.argv[1].replace(/\\/g, '/').endsWith('/tools/benchmark_w8a8.mjs')) {
  try {
    main();
  } catch (error) {
    console.error(`benchmark_w8a8: ${error.message}`);
    process.exitCode = 1;
  }
}
