// Author a minimal single-node Volvox package (graph.json + model.safetensors)
// so one op can be run through the real engine on every backend. Weightless ops
// use an empty safetensors; weighted ops (MatMul/LayerNorm, later) add tensors.
import fs from 'fs';
import path from 'path';

const DT_ST = { f32: 'F32', i32: 'I32', u8: 'U8', i8: 'I8' };

export function writeEmptySafetensors(file) {
  const header = Buffer.from('{}');
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.length));
  fs.writeFileSync(file, Buffer.concat([prefix, header]));
}

// tensors: { name: { dtype:'f32', shape:[...], data:Float32Array } }
export function writeSafetensors(file, tensors) {
  const entries = Object.entries(tensors);
  if (entries.length === 0) return writeEmptySafetensors(file);
  const header = {};
  const blobs = [];
  let offset = 0;
  for (const [name, t] of entries) {
    const buf = Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength);
    header[name] = { dtype: DT_ST[t.dtype], shape: t.shape, data_offsets: [offset, offset + buf.length] };
    blobs.push(buf);
    offset += buf.length;
  }
  const headerBuf = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBuf.length));
  fs.writeFileSync(file, Buffer.concat([prefix, headerBuf, ...blobs]));
}

export function writeGraph(file, graph) {
  fs.writeFileSync(file, JSON.stringify(graph, null, 1) + '\n');
}

// Write a single-node graph for one op.
export function singleNodeGraph({ opType, inputs, params, outShape, outDtype = 'float32' }) {
  const cfgInputs = {};
  for (const name of Object.values(inputs)) cfgInputs[name] = { shape: outShape, dtype: 'float32' };
  const node = {
    opType,
    inputs,
    outputs: { out: 'y' },
    outputs_shape: { out: outShape },
    outputs_dtype: { out: outDtype },
  };
  if (params && Object.keys(params).length) node.params = params;
  return { format: 'volvox-graph/v1', inputs: cfgInputs, nodes: [node], outputs: ['y'] };
}

export function ensureDir(dir) { fs.mkdirSync(dir, { recursive: true }); return dir; }
export { path };
