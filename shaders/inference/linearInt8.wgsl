// Weight-only INT8/U8 Linear. Activations, accumulation, and output remain
// F32 (W8A32); bytes are addressed globally so rows need not be 4-byte padded.
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
  weight_dtype : u32, // canonical protobuf dtype: 6=I8, 5=U8
  scale_elements : u32,
  zero_point_dtype : u32, // 0=UNSPECIFIED, 16=I32, 6=I8, 5=U8
  zero_point_elements : u32,
  _reserved : u32,
}
@group(0) @binding(6) var<uniform> params : Params;

fn raw_i8(word : u32, byte_index : u32) -> i32 {
  return extractBits(bitcast<i32>(word), byte_index * 8u, 8u);
}

fn raw_u8(word : u32, byte_index : u32) -> i32 {
  return i32(extractBits(word, byte_index * 8u, 8u));
}

fn weight_value(byte_offset : u32) -> i32 {
  let word = weight_packed[byte_offset / 4u];
  let byte_index = byte_offset % 4u;
  if (params.weight_dtype == 6u) {
    return raw_i8(word, byte_index);
  }
  return raw_u8(word, byte_index);
}

fn weight_zero_point(column : u32) -> i32 {
  if (params.zero_point_elements == 0u) {
    return 0;
  }
  let index = select(0u, column, params.zero_point_elements != 1u);
  if (params.zero_point_dtype == 6u) {
    let word = weight_zero_points[index / 4u];
    return raw_i8(word, index % 4u);
  }
  if (params.zero_point_dtype == 5u) {
    let word = weight_zero_points[index / 4u];
    return raw_u8(word, index % 4u);
  }
  return bitcast<i32>(weight_zero_points[index]);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let row = global_id.y;
  let column = global_id.x;
  if (row >= params.seq_len || column >= params.d_out) {
    return;
  }

  let weight_base = column * params.d_in;
  let input_base = row * params.d_in;
  let zero_point = weight_zero_point(column);
  var sum : f32 = 0.0;
  for (var k = 0u; k < params.d_in; k = k + 1u) {
    sum = sum + input[input_base + k] * f32(weight_value(weight_base + k) - zero_point);
  }
  let scale_index = select(0u, column, params.scale_elements != 1u);
  output[row * params.d_out + column] = sum * weight_scales[scale_index] + bias[column];
}
