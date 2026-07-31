// QLayerNorm statistics pass. Each workgroup owns one logical [..., D] row,
// reducing byte-domain values to mean_raw and an inverse standard deviation in
// real-value units. This avoids materializing an F32 activation tensor.
//
// Params is the fixed 48-byte cross-backend ABI shared with qLayerNormApply:
//   u32 rows, d_model, input_type, output_type
//   i32 input_zero_point, output_zero_point, pad, pad
//   f32 input_scale, output_scale, epsilon, pad
struct Params {
  rows : u32,
  d_model : u32,
  input_type : u32,
  output_type : u32,
  input_zero_point : i32,
  output_zero_point : i32,
  _pad0 : i32,
  _pad1 : i32,
  input_scale : f32,
  output_scale : f32,
  epsilon : f32,
  _pad2 : f32,
}

@group(0) @binding(0) var<storage, read> input_words : array<u32>;
// Two F32 scalars per row: mean_raw, then inverse standard deviation in real
// value units. This is graph-lifetime scratch rather than an activation.
@group(0) @binding(1) var<storage, read_write> row_stats : array<f32>;
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> partial_sums : array<f32, 64>;
var<workgroup> partial_square_sums : array<f32, 64>;
var<workgroup> row_mean_raw : f32;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn input_value(index : u32) -> i32 {
  let byte = word_byte(input_words[index / 4u], index);
  if (params.input_type == 6u) { return signed_byte(byte); }
  return i32(byte);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
  @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
) {
  let row = workgroup_id.x;
  let lane = local_invocation_id.x;
  let valid_row = row < params.rows;
  var sum : f32 = 0.0;

  if (valid_row) {
    let offset = row * params.d_model;
    for (var channel = lane; channel < params.d_model; channel = channel + 64u) {
      let raw = f32(input_value(offset + channel) - params.input_zero_point);
      sum = sum + raw;
    }
  }

  partial_sums[lane] = sum;
  workgroupBarrier();
  for (var stride : u32 = 32u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partial_sums[lane] = partial_sums[lane] + partial_sums[lane + stride];
    }
    workgroupBarrier();
  }

  if (lane == 0u) {
    row_mean_raw = 0.0;
    if (valid_row) {
      row_mean_raw = partial_sums[0] / f32(params.d_model);
    }
  }
  workgroupBarrier();

  var square_sum : f32 = 0.0;
  if (valid_row) {
    let offset = row * params.d_model;
    for (var channel = lane; channel < params.d_model; channel = channel + 64u) {
      let raw = f32(input_value(offset + channel) - params.input_zero_point);
      let centered = raw - row_mean_raw;
      square_sum = square_sum + centered * centered;
    }
  }

  partial_square_sums[lane] = square_sum;
  workgroupBarrier();
  for (var stride : u32 = 32u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partial_square_sums[lane] = partial_square_sums[lane] +
        partial_square_sums[lane + stride];
    }
    workgroupBarrier();
  }

  if (lane == 0u && valid_row) {
    let raw_variance = max(0.0, partial_square_sums[0] / f32(params.d_model));
    let variance = max(0.0, raw_variance * params.input_scale * params.input_scale);
    let stats_offset = row * 2u;
    row_stats[stats_offset] = row_mean_raw;
    row_stats[stats_offset + 1u] = inverseSqrt(variance + params.epsilon);
  }
}
