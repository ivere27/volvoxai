// Cooperative weight-only INT8/U8 Linear for multi-row matrices. Activations,
// accumulation, and output remain F32 (W8A32). An 8x8 workgroup computes a
// 16x16 output tile; the scalar linearInt8 kernel remains the M=1 path.
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight_packed : array<u32>;
@group(0) @binding(2) var<storage, read> weight_scales : array<f32>;
@group(0) @binding(3) var<storage, read> weight_zero_points : array<u32>;
@group(0) @binding(4) var<storage, read> bias : array<f32>;
@group(0) @binding(5) var<storage, read_write> output : array<f32>;

struct Params {
  seq_len : u32,
  d_in : u32,
  d_out : u32,
  weight_dtype : u32,
  scale_elements : u32,
  zero_point_dtype : u32,
  zero_point_elements : u32,
  _reserved : u32,
}
@group(0) @binding(6) var<uniform> params : Params;

var<workgroup> input_tile : array<f32, 256>;
var<workgroup> weight_tile : array<i32, 256>;

fn raw_i8(word : u32, byte_index : u32) -> i32 {
  return extractBits(bitcast<i32>(word), byte_index * 8u, 8u);
}

fn raw_u8(word : u32, byte_index : u32) -> i32 {
  return i32(extractBits(word, byte_index * 8u, 8u));
}

fn weight_value(byte_offset : u32) -> i32 {
  let word = weight_packed[byte_offset / 4u];
  let byte_index = byte_offset % 4u;
  if (params.weight_dtype == 2u) { return raw_i8(word, byte_index); }
  return raw_u8(word, byte_index);
}

fn weight_zero_point(column : u32) -> i32 {
  if (params.zero_point_elements == 0u) { return 0; }
  let index = select(0u, column, params.zero_point_elements != 1u);
  if (params.zero_point_dtype == 2u) {
    return raw_i8(weight_zero_points[index / 4u], index % 4u);
  }
  if (params.zero_point_dtype == 3u) {
    return raw_u8(weight_zero_points[index / 4u], index % 4u);
  }
  return bitcast<i32>(weight_zero_points[index]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) group_id : vec3<u32>,
        @builtin(local_invocation_id) local_id : vec3<u32>) {
  let row_base = group_id.y * 16u;
  let column_base = group_id.x * 16u;
  var sum00 = 0.0;
  var sum01 = 0.0;
  var sum10 = 0.0;
  var sum11 = 0.0;

  for (var k_base = 0u; k_base < params.d_in; k_base = k_base + 16u) {
    let row0 = row_base + local_id.y;
    let row1 = row0 + 8u;
    let k0 = k_base + local_id.x;
    let k1 = k0 + 8u;
    var input00 = 0.0;
    var input01 = 0.0;
    var input10 = 0.0;
    var input11 = 0.0;
    if (row0 < params.seq_len && k0 < params.d_in) { input00 = input[row0 * params.d_in + k0]; }
    if (row0 < params.seq_len && k1 < params.d_in) { input01 = input[row0 * params.d_in + k1]; }
    if (row1 < params.seq_len && k0 < params.d_in) { input10 = input[row1 * params.d_in + k0]; }
    if (row1 < params.seq_len && k1 < params.d_in) { input11 = input[row1 * params.d_in + k1]; }
    input_tile[local_id.y * 16u + local_id.x] = input00;
    input_tile[local_id.y * 16u + local_id.x + 8u] = input01;
    input_tile[(local_id.y + 8u) * 16u + local_id.x] = input10;
    input_tile[(local_id.y + 8u) * 16u + local_id.x + 8u] = input11;

    let column0 = column_base + local_id.y;
    let column1 = column0 + 8u;
    var weight00 = 0;
    var weight01 = 0;
    var weight10 = 0;
    var weight11 = 0;
    if (column0 < params.d_out && k0 < params.d_in) {
      weight00 = weight_value(column0 * params.d_in + k0) - weight_zero_point(column0);
    }
    if (column0 < params.d_out && k1 < params.d_in) {
      weight01 = weight_value(column0 * params.d_in + k1) - weight_zero_point(column0);
    }
    if (column1 < params.d_out && k0 < params.d_in) {
      weight10 = weight_value(column1 * params.d_in + k0) - weight_zero_point(column1);
    }
    if (column1 < params.d_out && k1 < params.d_in) {
      weight11 = weight_value(column1 * params.d_in + k1) - weight_zero_point(column1);
    }
    weight_tile[local_id.x * 16u + local_id.y] = weight00;
    weight_tile[(local_id.x + 8u) * 16u + local_id.y] = weight01;
    weight_tile[local_id.x * 16u + local_id.y + 8u] = weight10;
    weight_tile[(local_id.x + 8u) * 16u + local_id.y + 8u] = weight11;
    workgroupBarrier();

    for (var k = 0u; k < 16u; k = k + 1u) {
      let input0 = input_tile[local_id.y * 16u + k];
      let input1 = input_tile[(local_id.y + 8u) * 16u + k];
      let weight0 = f32(weight_tile[k * 16u + local_id.x]);
      let weight1 = f32(weight_tile[k * 16u + local_id.x + 8u]);
      sum00 = sum00 + input0 * weight0;
      sum01 = sum01 + input0 * weight1;
      sum10 = sum10 + input1 * weight0;
      sum11 = sum11 + input1 * weight1;
    }
    workgroupBarrier();
  }

  let row0 = row_base + local_id.y;
  let row1 = row0 + 8u;
  let column0 = column_base + local_id.x;
  let column1 = column0 + 8u;
  if (row0 < params.seq_len && column0 < params.d_out) {
    let scale = weight_scales[select(0u, column0, params.scale_elements != 1u)];
    output[row0 * params.d_out + column0] = sum00 * scale + bias[column0];
  }
  if (row0 < params.seq_len && column1 < params.d_out) {
    let scale = weight_scales[select(0u, column1, params.scale_elements != 1u)];
    output[row0 * params.d_out + column1] = sum01 * scale + bias[column1];
  }
  if (row1 < params.seq_len && column0 < params.d_out) {
    let scale = weight_scales[select(0u, column0, params.scale_elements != 1u)];
    output[row1 * params.d_out + column0] = sum10 * scale + bias[column0];
  }
  if (row1 < params.seq_len && column1 < params.d_out) {
    let scale = weight_scales[select(0u, column1, params.scale_elements != 1u)];
    output[row1 * params.d_out + column1] = sum11 * scale + bias[column1];
  }
}
