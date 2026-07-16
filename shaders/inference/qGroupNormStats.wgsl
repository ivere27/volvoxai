// QGroupNorm is deliberately split into statistics and apply passes. A group
// may begin or end in the middle of a packed u32 word (for example C/G = 3),
// so direct per-group packed output writes would race on byte lanes.
//
// Params is the fixed 64-byte cross-backend ABI shared by both passes. Host
// code must upload exactly this field order and dispatch one workgroup per
// flat [batch, group] for this statistics pass.
struct Params {
  batch : u32,
  height : u32,
  width : u32,
  channels : u32,
  groups : u32,
  input_type : u32,
  output_type : u32,
  _pad0 : u32,
  input_zero_point : i32,
  output_zero_point : i32,
  _pad1 : i32,
  _pad2 : i32,
  input_scale : f32,
  output_scale : f32,
  epsilon : f32,
  _pad3 : f32,
}

@group(0) @binding(0) var<storage, read> input_words : array<u32>;
// Two F32 values per flat group: mean_raw, then inverse standard deviation in
// real-value units. This buffer is graph-local reusable scratch, not an F32
// activation tensor.
@group(0) @binding(1) var<storage, read_write> group_stats : array<f32>;
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> partial_sums : array<f32, 64>;
var<workgroup> partial_square_sums : array<f32, 64>;
var<workgroup> group_mean_raw : f32;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn input_value(index : u32) -> i32 {
  let byte = word_byte(input_words[index / 4u], index);
  if (params.input_type == 2u) { return signed_byte(byte); }
  return i32(byte);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
  @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
) {
  let flat_group = workgroup_id.x;
  let lane = local_invocation_id.x;
  let total_groups = params.batch * params.groups;
  let valid_group = flat_group < total_groups;
  var sum : f32 = 0.0;

  if (valid_group) {
    let batch_index = flat_group / params.groups;
    let group = flat_group % params.groups;
    let channels_per_group = params.channels / params.groups;
    let channel_start = group * channels_per_group;
    let area = params.height * params.width;
    let values_per_group = area * channels_per_group;
    for (var element = lane; element < values_per_group; element = element + 64u) {
      let spatial = element / channels_per_group;
      let local_channel = element % channels_per_group;
      let logical_index = ((batch_index * area + spatial) * params.channels) +
        channel_start + local_channel;
      let raw = f32(input_value(logical_index) - params.input_zero_point);
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
    group_mean_raw = 0.0;
    if (valid_group) {
      let channels_per_group = params.channels / params.groups;
      let values_per_group = params.height * params.width * channels_per_group;
      group_mean_raw = partial_sums[0] / f32(values_per_group);
    }
  }
  workgroupBarrier();

  var square_sum : f32 = 0.0;
  if (valid_group) {
    let batch_index = flat_group / params.groups;
    let group = flat_group % params.groups;
    let channels_per_group = params.channels / params.groups;
    let channel_start = group * channels_per_group;
    let area = params.height * params.width;
    let values_per_group = area * channels_per_group;
    for (var element = lane; element < values_per_group; element = element + 64u) {
      let spatial = element / channels_per_group;
      let local_channel = element % channels_per_group;
      let logical_index = ((batch_index * area + spatial) * params.channels) +
        channel_start + local_channel;
      let raw = f32(input_value(logical_index) - params.input_zero_point);
      let centered = raw - group_mean_raw;
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

  if (lane == 0u && valid_group) {
    let channels_per_group = params.channels / params.groups;
    let values_per_group = params.height * params.width * channels_per_group;
    let count = f32(values_per_group);
    let raw_variance = max(0.0, partial_square_sums[0] / count);
    let variance = max(0.0, raw_variance * params.input_scale * params.input_scale);
    let stats_offset = flat_group * 2u;
    group_stats[stats_offset] = group_mean_raw;
    group_stats[stats_offset + 1u] = inverseSqrt(variance + params.epsilon);
  }
}
