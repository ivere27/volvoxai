#!/usr/bin/env node
/**
 * One generated authority for WGSL shader identity and storage.
 *
 * Two things needed the same table. The engine has to name a shader when it
 * plans a node, and it cannot hold WGSL text -- an id is the whole of what
 * crosses the device bridge. Separately, 165 shaders reach the browser as
 * JavaScript string literals, uncompressed, because nothing indexed them.
 *
 * An id table answers both. `shader_id` is an index here, and the same index
 * addresses a compressed block that is decoded once, lazily, when a WebGPU
 * device is actually initialised. Doing these apart would have meant designing
 * the same offset table twice.
 *
 * Text is compacted with the rule the release build already applies to `.wgsl`
 * inputs, so a decoded shader is byte-identical to what ships today. The
 * generator verifies that rather than assuming it.
 */

import { createHash } from 'node:crypto';
import { deflateSync } from 'node:zlib';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SHADER_ROOT = path.join(ROOT, 'shaders');
const SCOPES = Object.freeze(['inference', 'training']);

/** The exact rule the release build applies to a `.wgsl` input. */
function compactWgsl(source) {
  return source
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('//'))
    .join('\n');
}

/** `qConv2DInt8DotTiled.wgsl` -> `QCONV2D_INT8_DOT_TILED`. */
function constantName(scope, stem) {
  const words = stem
    .replace(/([a-z0-9])([A-Z])/gu, '$1_$2')
    .replace(/([A-Z]+)([A-Z][a-z])/gu, '$1_$2')
    .toUpperCase()
    .replace(/[^A-Z0-9]+/gu, '_')
    .replace(/^_+|_+$/gu, '');
  return `${scope.toUpperCase()}_${words}`;
}

/*
 * One source, several shaders.
 *
 * A catalogue entry is something the engine can name and a host can compile.
 * A file with a placeholder is neither: it is valid WGSL with the operator
 * missing, so compiling it succeeds and every output is written zero. Nothing
 * fails, and the wrong answer looks like an answer.
 *
 * Such a file is expanded here into the shaders it stands for and is never
 * published itself, so an engine cannot name the template even by mistake.
 */
const EXPANSIONS = Object.freeze({
  'inference/broadcastBinary': {
    placeholder: '//__BINOP__',
    variants: Object.freeze({
      broadcastAdd: 'out_val = av + bv;',
      broadcastAddReLU:
        'out_val = av + bv; if (out_val < 0.0) { out_val = 0.0; }',
      broadcastAddReLU6:
        'out_val = av + bv; if (out_val < 0.0) { out_val = 0.0; }' +
        ' if (out_val > 6.0) { out_val = 6.0; }',
      broadcastSub: 'out_val = av - bv;',
      broadcastMul: 'out_val = av * bv;',
      broadcastDiv: 'out_val = av / bv;',
    }),
  },
});

/** Anything still spelled `__NAME__` after expansion is an unfinished shader. */
const PLACEHOLDER = /__[A-Z0-9_]+__/u;

/* WGSL layout: size and alignment of the scalar and vector types the uniform
 * blocks in this repository use. */
const WGSL_TYPES = Object.freeze({
  u32: { size: 4, align: 4 }, i32: { size: 4, align: 4 }, f32: { size: 4, align: 4 },
  'vec2<u32>': { size: 8, align: 8 }, 'vec2<i32>': { size: 8, align: 8 },
  'vec2<f32>': { size: 8, align: 8 },
  'vec3<u32>': { size: 12, align: 16 }, 'vec3<i32>': { size: 12, align: 16 },
  'vec3<f32>': { size: 12, align: 16 },
  'vec4<u32>': { size: 16, align: 16 }, 'vec4<i32>': { size: 16, align: 16 },
  'vec4<f32>': { size: 16, align: 16 },
});

/**
 * What a shader declares about how it must be called.
 *
 * Bindings, their access, the workgroup size and the uniform block's size are
 * all written in the WGSL already. Reading them here rather than restating
 * them in the engine is what keeps the two from disagreeing: `read_write` is
 * exactly the "this dispatch writes the span" fact the device bridge needs,
 * and a hand-copied version of it is a silent wrong answer waiting to happen.
 */
function reachableIdentifiers(source, entryPoint) {
  const text = source.replace(/\/\*[\s\S]*?\*\//gu, '').replace(/\/\/[^\n]*/gu, '');
  const functions = new Map();
  for (const match of text.matchAll(/\bfn\s+(\w+)\s*\(/gu)) {
    const start = text.indexOf('{', match.index);
    let end = start + 1, depth = 1;
    while (end < text.length && depth) {
      if (text[end] === '{') depth++;
      if (text[end] === '}') depth--;
      end++;
    }
    if (start < 0 || depth) throw new Error(`Unclosed WGSL function ${match[1]}`);
    functions.set(match[1], text.slice(start + 1, end - 1));
  }
  const identifiers = new Set(), visited = new Set();
  const visit = name => {
    if (visited.has(name)) return;
    visited.add(name);
    const body = functions.get(name);
    if (body == null) throw new Error(`Missing WGSL entry ${name}`);
    for (const token of body.matchAll(/\b[A-Za-z_]\w*\b/gu)) {
      identifiers.add(token[0]);
      if (functions.has(token[0])) visit(token[0]);
    }
  };
  visit(entryPoint);
  return identifiers;
}

function readInterface(relative, source, entryPoint = null) {
  const storage = [];
  let uniforms = 0;
  // Multi-entry training modules share declarations, but each pipeline only
  // binds resources reachable from its entry. This also permits two operands
  // to share a gradient when separate backward entries write them in order.
  const used = relative.startsWith('shaders/training/') && entryPoint
    ? reachableIdentifiers(source, entryPoint) : null;
  const bindings = [...source.matchAll(
    /@binding\((\d+)\)\s*var<(storage,\s*(read_write|read)|uniform)>\s*(\w+)/gu)];
  for (const [, index, kind, access, name] of bindings) {
    if (used && !used.has(name)) continue;
    if (kind === 'uniform') { uniforms++; continue; }
    storage.push({ index: Number(index), writes: access === 'read_write' });
  }
  storage.sort((left, right) => left.index - right.index);
  if (uniforms > 1) throw new Error(`${relative} declares ${uniforms} uniform blocks.`);
  /* The catalogue records what a shader declares; how many of them the engine
   * can plan for is the engine's own bound, checked where the plan is built. */
  if (storage.length > 16) {
    throw new Error(
      `${relative} declares ${storage.length} storage bindings; the descriptor ` +
      'carries at most 16.');
  }
  /* The slot a buffer occupies is not its position in the engine's bind order:
   * a shader may put its uniform in the middle. Carrying the declared index
   * means the descriptor names the slot the shader actually reads. */
  const uniform = uniforms === 1
    ? /@binding\((\d+)\)\s*var<uniform>\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*([A-Za-z_][A-Za-z0-9_]*)/u
        .exec(source)
    : null;
  if (uniforms === 1 && !uniform) {
    throw new Error(`${relative} declares a uniform this parser cannot read.`);
  }
  const uniformBinding = uniform ? Number(uniform[1]) : 255;

  const group = entryPoint
    ? new RegExp(`@compute\\s*@workgroup_size\\(([^)]*)\\)\\s*fn\\s+${entryPoint}\\b`, 'u').exec(source)
    : /@compute\s*@workgroup_size\(([^)]*)\)/u.exec(source);
  if (!group) throw new Error(`${relative} declares no workgroup size.`);
  const workgroup = group[1].split(',').map((part) => Number(part.trim()));
  while (workgroup.length < 3) workgroup.push(1);
  if (workgroup.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error(`${relative} has a non-constant workgroup size.`);
  }

  /* The uniform block's size, laid out by the WGSL rules, is the exact byte
   * count the engine must supply -- fewer is a binding a device may refuse. */
  let paramsBytes = 0;
  if (uniforms === 1) {
    /* The block is found by the type the binding names, not by a fixed
     * spelling: shaders call it Params, Config, Uniforms and more. */
    const block = new RegExp(
      `struct\\s+${uniform[2]}\\s*\\{([^}]*)\\}`, 'u').exec(source);
    if (!block) {
      throw new Error(`${relative} binds uniform type ${uniform[2]} with no struct.`);
    }
    let offset = 0;
    let alignment = 4;
    /* Comments come off before the fields are split, not after: a comment
     * carries commas of its own, and splitting first turns one vec4 field
     * into several unreadable fragments -- which reads as a smaller uniform
     * block than the shader actually declares. */
    const fields = block[1].replace(/\/\/[^\n]*/gu, '').split(',');
    for (const line of fields) {
      const field = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z0-9_<>]+)\s*$/u
        .exec(line.trim());
      if (!field) continue;
      const type = WGSL_TYPES[field[2]];
      if (!type) throw new Error(`${relative} uses unmapped uniform type ${field[2]}.`);
      offset = Math.ceil(offset / type.align) * type.align + type.size;
      alignment = Math.max(alignment, type.align);
    }
    paramsBytes = Math.ceil(offset / alignment) * alignment;
  }
  let workgroupBytes = 0;
  for (const [, declaration] of source.replace(/\/\/[^\n]*/gu, '').matchAll(
    /var\s*<\s*workgroup\s*>\s*\w+\s*:\s*([^;]+);/gu)) {
    const typeName = declaration.replace(/\s+/gu, '');
    const array = /^array<(.+),(\d+)u?>$/u.exec(typeName);
    const type = WGSL_TYPES[array ? array[1] : typeName];
    if (!type) throw new Error(`${relative} uses unmapped workgroup type ${typeName}.`);
    const size = array
      ? Math.ceil(type.size / type.align) * type.align * Number(array[2]) : type.size;
    // Conservative per-variable 16-byte allocation; includes all declarations.
    // https://gpuweb.github.io/gpuweb/#dom-gpusupportedlimits-maxcomputeworkgroupstoragesize
    workgroupBytes += Math.ceil(size / 16) * 16;
  }
  return {
    slots: storage.map((binding) => binding.index),
    writesMask: storage.reduce(
      (mask, binding, position) => mask | (binding.writes ? 1 << position : 0), 0),
    paramsBinding: uniformBinding,
    paramsBytes,
    workgroup,
    workgroupBytes,
  };
}

async function collect() {
  const shaders = [];
  for (const scope of SCOPES) {
    const directory = path.join(SHADER_ROOT, scope);
    const names = (await fs.readdir(directory)).filter((n) => n.endsWith('.wgsl')).sort();
    for (const name of names) {
      const stem = name.slice(0, -'.wgsl'.length);
      const source = await fs.readFile(path.join(directory, name), 'utf8');
      const relative = `shaders/${scope}/${name}`;
      const expansion = EXPANSIONS[`${scope}/${stem}`];
      if (expansion) {
        if (!source.includes(expansion.placeholder)) {
          throw new Error(
            `${relative} is declared as a template but has no ` +
            `${expansion.placeholder} to expand.`);
        }
        for (const [variant, body] of Object.entries(expansion.variants)) {
          const expanded = source.replace(expansion.placeholder, body);
          shaders.push({
            scope,
            stem: variant,
            relative: `${relative}#${variant}`,
            constant: constantName(scope, variant),
            text: compactWgsl(expanded),
            interface: readInterface(`${relative}#${variant}`, expanded),
          });
        }
        continue;
      }
      const entries = [...source.matchAll(/@compute\s*@workgroup_size\([^)]*\)\s*fn\s+(\w+)/gu)]
        .map((match) => match[1]);
      for (const entryPoint of entries) {
        const variant = entries.length === 1 && entryPoint === 'main'
          ? stem : `${stem}_${entryPoint}`;
        shaders.push({
          scope, stem: variant, sourceStem: stem, entryPoint,
          relative: variant === stem ? relative : `${relative}#${entryPoint}`,
          constant: constantName(scope, variant), text: compactWgsl(source),
          interface: readInterface(relative, source, entryPoint),
        });
      }
    }
  }
  for (const shader of shaders) {
    const found = PLACEHOLDER.exec(shader.text);
    if (found) {
      throw new Error(
        `${shader.relative} still carries the placeholder ${found[0]}; a ` +
        'published shader must be compilable as it stands.');
    }
  }
  shaders.sort((left, right) =>
    left.scope === right.scope
      ? (left.stem < right.stem ? -1 : left.stem > right.stem ? 1 : 0)
      : (left.scope < right.scope ? -1 : 1));
  /* Ids are assigned over the sorted scope/name order so a rebuild on the same
   * inputs produces the same table. IDs may change. The bridge verifies the generated catalogue hash before
   * accepting a device call from a compiled module. */
  return shaders.map((shader, index) => ({ ...shader, id: index }));
}

function buildPack(shaders) {
  /* One deflate block, not one per shader: WGSL repeats itself heavily across
   * variants and separate blocks would throw that away. Offsets address the
   * decoded bytes. */
  const encoder = new TextEncoder();
  const parts = [];
  const offsets = [];
  let cursor = 0;
  for (const shader of shaders) {
    const bytes = encoder.encode(shader.text);
    offsets.push([cursor, bytes.length]);
    parts.push(bytes);
    cursor += bytes.length;
  }
  const plain = new Uint8Array(cursor);
  let at = 0;
  for (const part of parts) { plain.set(part, at); at += part.length; }
  /* zlib-wrapped deflate, not raw: the fixed Node 20 toolchain has no
   * `DecompressionStream('deflate-raw')`, and the six-byte wrapper is noise
   * beside what the block saves. */
  const packed = deflateSync(plain, { level: 9 });
  return { plain, packed, offsets };
}

function renderTs(shaders, pack) {
  const names = shaders.map((s) => `  ${JSON.stringify(s.relative)},`).join('\n');
  const constants = shaders
    .map((s) => `export const SHADER_${s.constant} = ${s.id};`).join('\n');
  const offsets = pack.offsets.map(([start, length]) => `  ${start},${length},`).join('\n');
  return `// DO NOT EDIT: generated by tools/generate_shader_catalog.mjs.
// Shader source SHA-256: ${pack.sourceSha256}
/**
 * WGSL identity and storage.
 *
 * The block below is every shader, compacted and deflated once. It is decoded
 * on the first \`loadShaderPack()\`, which the WebGPU compiler awaits before it
 * needs a shader and which nothing else calls -- a build that never reaches a
 * device never pays for the text.
 */

export const SHADER_PACK_ENCODING = 'deflate' as const;
export const SHADER_COUNT = ${shaders.length};
export const SHADER_DECODED_BYTES = ${pack.plain.length};

/** Index is the shader id; the value is its repository path. */
export const SHADER_CATALOG_HASH = ${JSON.stringify(pack.catalogHash)};

export const SHADER_NAMES = /* @__PURE__ */ Object.freeze([
${names}
] as const);

export const SHADER_ENTRY_POINTS = /* @__PURE__ */ Object.freeze(${JSON.stringify(shaders.map(s => s.entryPoint ?? 'main'))});
export const SHADER_LAYOUTS = /* @__PURE__ */ Object.freeze(${JSON.stringify(shaders.map(s => ({ slots: s.interface.slots, writesMask: s.interface.writesMask, paramsBinding: s.interface.paramsBinding })))});

/** Flat \`[start, length]\` pairs into the decoded block, indexed by id. */
const SHADER_SPANS = /* @__PURE__ */ Object.freeze([
${offsets}
] as const);

const SHADER_PACK_BASE64 =
${JSON.stringify(Buffer.from(pack.packed).toString('base64')).replace(/(.{110})/gu, '$1" +\n  "')};

${constants}

function base64ToBytes(value: string): Uint8Array {
  if (typeof atob === 'function') {
    const binary = atob(value);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  }
  /* Node before a DOM shim. Buffer is a Uint8Array, so this is a view, not a
   * second copy. */
  const nodeBuffer = (globalThis as unknown as {
    Buffer?: { from(input: string, encoding: string): Uint8Array };
  }).Buffer;
  if (!nodeBuffer) throw new Error('No base64 decoder is available.');
  return nodeBuffer.from(value, 'base64');
}

async function inflate(packed: Uint8Array): Promise<Uint8Array> {
  const Decompression = (globalThis as unknown as {
    DecompressionStream?: new (format: string) => {
      readable: ReadableStream<Uint8Array>;
      writable: WritableStream<Uint8Array>;
    };
  }).DecompressionStream;
  if (!Decompression) {
    throw new Error(
      "This runtime has no DecompressionStream('deflate'); WGSL shaders " +
      'cannot be decoded. Every engine that reaches WebGPU has one.',
    );
  }
  const stream = new Decompression(SHADER_PACK_ENCODING);
  const source = new Blob([packed as BlobPart]).stream() as unknown as
    ReadableStream<Uint8Array>;
  const decoded = source.pipeThrough(
    stream as unknown as ReadableWritablePair<Uint8Array, Uint8Array>,
  );
  const chunks: Uint8Array[] = [];
  let total = 0;
  const reader = decoded.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  const bytes = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length; }
  return bytes;
}

let pending: Promise<readonly string[]> | null = null;

/**
 * Decode every shader once. Repeated calls share one decode; a caller that
 * never reaches a device never triggers it.
 */
export function loadShaderPack(): Promise<readonly string[]> {
  if (pending) return pending;
  pending = (async () => {
    const bytes = await inflate(base64ToBytes(SHADER_PACK_BASE64));
    if (bytes.length !== SHADER_DECODED_BYTES) {
      throw new Error(
        \`WGSL pack decoded \${bytes.length} bytes; expected \${SHADER_DECODED_BYTES}.\`,
      );
    }
    const decoder = new TextDecoder();
    const shaders: string[] = new Array(SHADER_COUNT);
    for (let id = 0; id < SHADER_COUNT; id++) {
      const start = SHADER_SPANS[id * 2]!;
      const length = SHADER_SPANS[id * 2 + 1]!;
      shaders[id] = decoder.decode(bytes.subarray(start, start + length));
    }
    return Object.freeze(shaders);
  })();
  return pending;
}
`;
}

function renderC(shaders, pack) {
  const constants = shaders
    .map((s) => `#define VX_SHADER_${s.constant} UINT32_C(${s.id})`).join('\n');
  const interfaces = shaders.map((shader) => {
    const { slots, writesMask, paramsBinding, paramsBytes, workgroup, workgroupBytes } = shader.interface;
    const padded = [...slots];
    while (padded.length < 16) padded.push(255);
    return `    { {${padded.join(', ')}}, ${slots.length}, ${writesMask}, ` +
      `${paramsBinding}, ${paramsBytes}, {${workgroup.join(', ')}}, ${workgroupBytes} }, ` +
      `/* ${shader.relative} */`;
  }).join('\n');
  return `/* DO NOT EDIT: generated by tools/generate_shader_catalog.mjs. */
/* Shader source SHA-256: ${pack.sourceSha256} */

/*
 * Shader identity for the engine.
 *
 * The engine names a shader when it plans a node; it never holds WGSL text.
 * These ids are what \`VxGpuVariant.shader_id\` carries across the device
 * bridge, and the host resolves them through the matching generated table.
 */

#ifndef VOLVOXAI_BACKENDS_SHADER_CATALOG_GENERATED_H
#define VOLVOXAI_BACKENDS_SHADER_CATALOG_GENERATED_H

#include <stdint.h>
#define VX_SHADER_CATALOG_HASH "${pack.catalogHash}"

#define VX_SHADER_COUNT UINT32_C(${shaders.length})

/*
 * What a shader declares about how it must be called.
 *
 * All of it is read from the WGSL: which slot each buffer occupies, which of
 * them the dispatch writes, where the uniform block sits and how large it is,
 * and the workgroup size. The engine builds its descriptor from this rather
 * than restating it, because a restatement can disagree -- and the way it
 * disagrees is a dispatch that runs, succeeds, and writes the wrong bytes.
 *
 * A storage_count above VX_SHADER_MAX_PLANNED_BINDINGS means this engine
 * cannot describe the shader; a params_slot of 255 means it binds no uniform.
 */
#define VX_SHADER_MAX_PLANNED_BINDINGS 16

typedef struct {
    uint8_t slots[VX_SHADER_MAX_PLANNED_BINDINGS];
    uint8_t storage_count;
    uint16_t writes_mask;
    uint8_t params_slot;
    uint16_t params_bytes;
    uint16_t workgroup[3];
    uint32_t workgroup_bytes;
} VxShaderInterface;

/* Copied into opt-in traces only; IDs still use the canonical catalogue. */
static const struct { const char* name; const char* entry; }
vx_shader_programs[VX_SHADER_COUNT] = {
${shaders.map(s => `    { "${s.sourceStem ?? s.stem}", "${s.entryPoint ?? 'main'}" },`).join('\n')}
};

static const VxShaderInterface vx_shader_interface[VX_SHADER_COUNT] = {
${interfaces}
};

#if VOLVOXAI_ENABLE_TRAINING && VOLVOXAI_ENABLE_WEBGPU
static const struct { const char* name; const char* entry; uint32_t id; }
vx_training_shader_entries[] = {
${shaders.filter(s => s.scope === 'training').map(s => `    { "${s.sourceStem ?? s.stem}", "${s.entryPoint ?? 'main'}", ${s.id}u },`).join('\n')}
};
#endif

${constants}

#endif
`;
}

async function main() {
  const check = process.argv.includes('--check');
  const allShaders = await collect();
  for (const profile of ['inference', 'full']) {
  const shaders = allShaders.filter(s => profile === 'full' || s.scope === 'inference')
    .map((shader, id) => ({ ...shader, id }));
  const seen = new Set();
  for (const shader of shaders) {
    if (seen.has(shader.constant)) {
      throw new Error(`Shader constant ${shader.constant} is not unique.`);
    }
    seen.add(shader.constant);
  }
  const pack = buildPack(shaders);
  pack.sourceSha256 = createHash('sha256').update(pack.plain).digest('hex');
  pack.catalogHash = createHash('sha256').update(JSON.stringify(shaders.map(
    ({ id, relative, interface: layout }) => ({ id, relative, layout }),
  ))).update(pack.plain).digest('hex');

  /* Prove the round trip here rather than trusting it: a decoded span must be
   * the exact bytes the release build would have embedded. */
  const { inflateSync } = await import('node:zlib');
  const restored = inflateSync(pack.packed);
  const decoder = new TextDecoder();
  for (const [index, shader] of shaders.entries()) {
    const [start, length] = pack.offsets[index];
    if (decoder.decode(restored.subarray(start, start + length)) !== shader.text) {
      throw new Error(`Shader ${shader.relative} does not survive the pack round trip.`);
    }
  }

  const outputs = [
    [path.join(ROOT, 'ts/generated', profile === 'full' ? 'shaderCatalog.ts' : 'shaderCatalogInference.ts'), renderTs(shaders, pack)],
    [path.join(ROOT, 'native/src/backends', profile === 'full' ? 'shader_catalog.h' : 'shader_catalog_inference.h'), renderC(shaders, pack)],
  ];
  const stale = [];
  for (const [file, contents] of outputs) {
    const current = await fs.readFile(file, 'utf8').catch(() => null);
    if (current === contents) continue;
    if (check) { stale.push(path.relative(ROOT, file)); continue; }
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.writeFile(file, contents);
  }
  if (check && stale.length > 0) {
    throw new Error(
      `Stale generated shader catalogue: ${stale.join(', ')}. ` +
      'Run node tools/generate_shader_catalog.mjs.',
    );
  }
  const ratio = (pack.plain.length / pack.packed.length).toFixed(1);
  console.log(
    `${check ? 'Verified' : 'Generated'} ${profile} shader catalogue: ${shaders.length} shaders, ` +
    `${pack.plain.length} B compacted -> ${pack.packed.length} B deflated (${ratio}:1).`,
  );
  }
}

await main();
