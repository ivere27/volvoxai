#!/usr/bin/env node
/**
 * The operator dispatch table the WebGPU backend plans from.
 *
 * Naming a shader, binding a node's operands in its order, filling its uniform
 * words and choosing a grid is the same shape of work for every operator, and
 * writing it as a C function per operator meant writing the same thing 22
 * times -- once per operator, and again in `WebGPUPhysicalDomain.ts`, and
 * again in `WebGPUGraphCompiler.ts`. This declares it once.
 *
 * What stays in C is admission: which shapes a shader is *correct* for. That
 * is a claim about correctness rather than a transcription of an interface,
 * and a table that guessed it wrong would produce answers instead of refusals.
 *
 * The shader's own half -- which slot each buffer occupies, which of them the
 * dispatch writes, how many uniform bytes it reads -- is not here either. It
 * is read from the WGSL by `generate_shader_catalog.mjs` and checked against
 * every plan, so this table cannot disagree with the shader it names.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runtimeOperatorCConstants, runtimePortCConstants } from './generated/volvoxaiGraphOperators.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = path.join(ROOT, 'native/src/backends/webgpu_dispatch.inc');

/* Where a uniform word comes from. One entry per kind, and no nesting: a word
 * is either a constant, something a tensor already knows, or a node parameter. */
const VALUE = Object.freeze({
  CONST: 'VX_WEBGPU_VALUE_CONST',
  CONST_F32: 'VX_WEBGPU_VALUE_CONST_F32',
  ELEMENTS: 'VX_WEBGPU_VALUE_ELEMENTS',
  STORAGE_WORDS: 'VX_WEBGPU_VALUE_STORAGE_WORDS',
  LASTDIM: 'VX_WEBGPU_VALUE_LASTDIM',
  ROWS: 'VX_WEBGPU_VALUE_ROWS',
  AXIS: 'VX_WEBGPU_VALUE_AXIS',
  DTYPE: 'VX_WEBGPU_VALUE_DTYPE',
  PARAM_I32: 'VX_WEBGPU_VALUE_PARAM_I32',
  PARAM_F32: 'VX_WEBGPU_VALUE_PARAM_F32',
  INV_LASTDIM: 'VX_WEBGPU_VALUE_INV_LASTDIM',
  GRID_STRIDE: 'VX_WEBGPU_VALUE_GRID_STRIDE',
  APPROX_TANH: 'VX_WEBGPU_VALUE_APPROX_TANH',
});

/* How the grid is derived. The shaders index themselves in exactly these
 * three ways; a fourth would be a new shader convention, not a new option. */
const GRID = Object.freeze({
  LINEAR_1D: 'VX_WEBGPU_GRID_LINEAR_1D',
  LINEAR_2D: 'VX_WEBGPU_GRID_LINEAR_2D',
});

/* What an operand is to the node. */
const OPERAND = Object.freeze({
  OUTPUT: 'VX_WEBGPU_OPERAND_OUTPUT',
  INPUT: 'VX_WEBGPU_OPERAND_INPUT',
  PORT: 'VX_WEBGPU_OPERAND_PORT',
});

/* Admission, by name. Each is a C predicate; the table says which one guards
 * the operator, never what it checks. */
const ADMIT = Object.freeze({
  EQUAL_ELEMENTS: 'VX_WEBGPU_ADMIT_EQUAL_ELEMENTS',
  EQUAL_ELEMENTS_LAST_AXIS: 'VX_WEBGPU_ADMIT_EQUAL_ELEMENTS_LAST_AXIS',
  EQUAL_ELEMENTS_DISTINCT: 'VX_WEBGPU_ADMIT_EQUAL_ELEMENTS_DISTINCT',
  ROW_REDUCTION: 'VX_WEBGPU_ADMIT_ROW_REDUCTION',
  SAME_DTYPE_ELEMENTS_DISTINCT: 'VX_WEBGPU_ADMIT_SAME_DTYPE_ELEMENTS_DISTINCT',
  PORTABLE_CAST: 'VX_WEBGPU_ADMIT_PORTABLE_CAST',
});

const F32 = 'T_F32';
const I32 = 'T_I32';
const ANY = '0';

/*
 * How an operand is sized against the output.
 *
 * FULL is one element per output element; LANE is one per channel, which is
 * what a weight, bias or slope carries. Declaring it here is what keeps a
 * short operand from being read past its end on the device -- the shader has
 * no bounds to check against.
 */
const SIZE = Object.freeze({
  ANY: 'VX_WEBGPU_SIZE_ANY',
  FULL: 'VX_WEBGPU_SIZE_FULL',
  LANE: 'VX_WEBGPU_SIZE_LANE',
});

const input = (dtype = F32, sizing = SIZE.FULL) =>
  ({ kind: OPERAND.INPUT, dtype, sizing });
const output = (dtype = F32) =>
  ({ kind: OPERAND.OUTPUT, dtype, sizing: SIZE.ANY });
const port = (name, dtype = F32, sizing = SIZE.LANE, alias = null) =>
  ({ kind: OPERAND.PORT, port: name, alias, dtype, sizing });

/*
 * Every value names its operand.
 *
 * A default here would be positional, and a position means different things
 * to a two-operand operator and a three-operand one -- which is a wrong
 * uniform word rather than a refusal.
 */
const constant = (value) => ({ kind: VALUE.CONST, a: value });
const elements = (operand) => ({ kind: VALUE.ELEMENTS, operand });
const lastdim = (operand) => ({ kind: VALUE.LASTDIM, operand });
const rows = (operand) => ({ kind: VALUE.ROWS, operand });
const dtypeOf = (operand) => ({ kind: VALUE.DTYPE, operand });
const gridStride = () => ({ kind: VALUE.GRID_STRIDE });

/*
 * The table.
 *
 * `operands` are bound in this order, which is the order the shader declares
 * its storage bindings. `params` are the uniform words, in order. Nothing here
 * repeats a fact the WGSL already states.
 */
const OPERATORS = [
  ...['ReLU', 'Sigmoid', 'SiLU', 'Tanh'].map((op) => ({
    op,
    shader: `INFERENCE_${op === 'ReLU' ? 'RE_LU' : op === 'SiLU' ? 'SI_LU' : op.toUpperCase()}`,
    operands: [input(), output()],
    params: [elements(1)],
    grid: { kind: GRID.LINEAR_2D, value: elements(1), divisor: 64 },
    admit: ADMIT.EQUAL_ELEMENTS,
  })),
  ...[['HardSigmoid', 'HARD_SIGMOID'], ['HardSwish', 'HARD_SWISH']].map(([op, shader]) => ({
    op, shader: `INFERENCE_${shader}`,
    operands: [input(), output()],
    params: [elements(1)],
    grid: { kind: GRID.LINEAR_2D, value: elements(1), divisor: 64 },
    admit: ADMIT.EQUAL_ELEMENTS,
  })),
  {
    op: 'LeakyReLU', shader: 'INFERENCE_LEAKY_RE_LU',
    operands: [input(), output()],
    params: [elements(1), { kind: VALUE.PARAM_F32, a: 'VX_NODE_PARAM_ALPHA', fallback: 0.01 }],
    grid: { kind: GRID.LINEAR_2D, value: elements(1), divisor: 64 },
    admit: ADMIT.EQUAL_ELEMENTS,
  },
  {
    op: 'GELU',
    shader: 'INFERENCE_G_ELU',
    operands: [input(), output()],
    params: [elements(1), { kind: VALUE.APPROX_TANH }, constant(0), constant(0)],
    grid: { kind: GRID.LINEAR_1D, value: elements(1), divisor: 64 },
    admit: ADMIT.EQUAL_ELEMENTS,
  },
  {
    op: 'Not',
    shader: 'INFERENCE_NOT_I32',
    operands: [input(I32), output(I32)],
    params: [elements(1), constant(0), constant(0), constant(0)],
    grid: { kind: GRID.LINEAR_1D, value: elements(1), divisor: 64 },
    admit: ADMIT.EQUAL_ELEMENTS,
  },
  {
    op: 'Cast',
    shader: 'INFERENCE_CAST',
    operands: [input(ANY), output(ANY)],
    params: [elements(1), dtypeOf(0), dtypeOf(1), constant(0)],
    grid: { kind: GRID.LINEAR_1D, value: { kind: VALUE.STORAGE_WORDS, operand: 1 }, divisor: 64 },
    admit: ADMIT.PORTABLE_CAST,
  },
  ...['Softmax', 'LogSoftmax'].map((op) => ({
    op,
    shader: op === 'Softmax' ? 'INFERENCE_SOFTMAX' : 'INFERENCE_LOG_SOFTMAX',
    operands: [input(F32, SIZE.ANY), output()],
    params: [rows(0), lastdim(0)],
    grid: { kind: GRID.LINEAR_1D, value: rows(0), divisor: 64 },
    admit: ADMIT.EQUAL_ELEMENTS_LAST_AXIS,
  })),
  ...['ReduceSum', 'ReduceMean'].map((op) => ({
    op,
    shader: 'INFERENCE_REDUCE',
    operands: [input(F32, SIZE.ANY), output()],
    params: [
      rows(0),
      lastdim(0),
      op === 'ReduceMean'
        ? { kind: VALUE.INV_LASTDIM, operand: 0 }
        : { kind: VALUE.CONST_F32, fallback: 1 },
    ],
    grid: { kind: GRID.LINEAR_1D, value: rows(0), divisor: 64 },
    admit: ADMIT.ROW_REDUCTION,
  })),
  ...['Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Identity', 'Dropout'].map((op) => ({
    op,
    shader: 'INFERENCE_COPY32',
    operands: [input(ANY), output(ANY)],
    params: [elements(1), gridStride(), constant(0), constant(0)],
    grid: { kind: GRID.LINEAR_2D, value: elements(1), divisor: 64 },
    admit: ADMIT.SAME_DTYPE_ELEMENTS_DISTINCT,
  })),
  {
    op: 'PReLU',
    shader: 'INFERENCE_P_RE_LU',
    operands: [input(), port('slope', F32, SIZE.LANE, 'weight'), output()],
    params: [elements(2), lastdim(0), elements(1), constant(0)],
    grid: { kind: GRID.LINEAR_1D, value: elements(2), divisor: 64 },
    admit: ADMIT.EQUAL_ELEMENTS,
  },
];

/* A float-valued word is emitted as its bits, whether or not the number
 * happens to be a whole one: 1.0f is 0x3f800000, never 1. */
const FLOAT_KINDS = new Set([VALUE.CONST_F32, VALUE.PARAM_F32]);

function renderValue(value) {
  const operand = value.operand ?? 0;
  const a = value.a ?? 0;
  const fallback = value.fallback ?? 0;
  const word = FLOAT_KINDS.has(value.kind)
    ? `0x${new Uint32Array(new Float32Array([fallback]).buffer)[0]
        .toString(16)}u`
    : `${fallback}`;
  return `{ ${value.kind}, ${operand}, ${a}, ${word} }`;
}

function renderOperand(operand) {
  const port = (name) => {
    if (name == null) return 'VX_PORT_UNSPECIFIED';
    if (!Object.hasOwn(runtimePortCConstants, name)) {
      throw new Error(`Unknown proto port: ${name}`);
    }
    return runtimePortCConstants[name];
  };
  return `{ ${operand.kind}, ${port(operand.port)}, ` +
    `${port(operand.alias)}, ${operand.dtype}, ${operand.sizing} }`;
}

function render() {
  for (const entry of OPERATORS) {
    if (!Object.hasOwn(runtimeOperatorCConstants, entry.op)) {
      throw new Error(`Unknown proto operator: ${entry.op}`);
    }
  }
  const blocks = OPERATORS.map((entry, index) => {
    const operands = entry.operands.map(renderOperand).join(',\n    ');
    const params = entry.params.map(renderValue).join(',\n    ');
    return `static const VxWebGpuOperand vx_webgpu_operands_${index}[] = {\n` +
      `    ${operands},\n};\n` +
      `static const VxWebGpuValue vx_webgpu_params_${index}[] = {\n` +
      `    ${params},\n};`;
  }).join('\n');

  const table = OPERATORS.map((entry, index) =>
    `    { ${runtimeOperatorCConstants[entry.op]}, VX_SHADER_${entry.shader},\n` +
    `      vx_webgpu_operands_${index}, ${entry.operands.length},\n` +
    `      vx_webgpu_params_${index}, ${entry.params.length},\n` +
    `      ${entry.grid.kind}, ${renderValue(entry.grid.value)}, ${entry.grid.divisor},\n` +
    `      ${entry.admit} },`).join('\n');

  return `/* DO NOT EDIT: generated by tools/generate_webgpu_dispatch.mjs. */

/*
 * The operator dispatch table.
 *
 * One row per operator: which shader, which operands in which order, which
 * uniform words, which grid, and which admission predicate guards it. The
 * shader's own interface -- slots, which bindings are written, how many
 * uniform bytes -- is not restated here; it comes from the generated shader
 * catalogue and is checked against every plan this table produces.
 */

${blocks}

static const VxWebGpuOperatorPlan vx_webgpu_operator_table[] = {
${table}
};

#define VX_WEBGPU_OPERATOR_COUNT \\
    (sizeof(vx_webgpu_operator_table) / sizeof(vx_webgpu_operator_table[0]))
`;
}

async function main() {
  const text = render();
  const check = process.argv.includes('--check');
  if (check) {
    const current = await fs.readFile(OUTPUT, 'utf8').catch(() => null);
    if (current !== text) {
      console.error(
        `${path.relative(ROOT, OUTPUT)} is stale; run ` +
        'node tools/generate_webgpu_dispatch.mjs');
      process.exitCode = 1;
      return;
    }
    console.log(
      `Verified WebGPU dispatch table (${OPERATORS.length} operators).`);
    return;
  }
  await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
  await fs.writeFile(OUTPUT, text);
  console.log(
    `Generated WebGPU dispatch table: ${OPERATORS.length} operators.`);
}

await main();
