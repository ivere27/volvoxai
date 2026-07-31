// Canonical byte-domain QSDPA. One 32-lane workgroup owns one
// [batch, query, head] result. Q/K raw bytes form exact I32 head-local dots;
// each lane builds an online-softmax partial and reduction combines only
// initialized states. The private workgroup F32 state is not a graph-visible
// activation or score matrix.
//
// Fixed 80-byte ABI, shared by browser and native GPU backends:
//   u32 seq_q, seq_kv, d_model, heads
//   u32 batch, mask_mode, causal, dtypes
//   i32 q_zero_point, k_zero_point, v_zero_point, output_zero_point
//   f32 q_scale, k_scale, v_scale, output_scale
//   f32 attention_scale, pad, pad, pad
// dtypes packs canonical protobuf values q | k<<8 | v<<16 | output<<24,
// where I8=6 and U8=5. mask_mode is 0 none, 1 [K], 2 [B,K], 3 [Q,K], 4 [B,Q,K].
struct Params {
  seq_q : u32,
  seq_kv : u32,
  d_model : u32,
  heads : u32,
  batch : u32,
  mask_mode : u32,
  causal : u32,
  dtypes : u32,
  q_zero_point : i32,
  k_zero_point : i32,
  v_zero_point : i32,
  output_zero_point : i32,
  q_scale : f32,
  k_scale : f32,
  v_scale : f32,
  output_scale : f32,
  attention_scale : f32,
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
}

@group(0) @binding(0) var<storage, read> q_words : array<u32>;
@group(0) @binding(1) var<storage, read> k_words : array<u32>;
@group(0) @binding(2) var<storage, read> v_words : array<u32>;
// Bound to a four-byte dummy buffer when mask_mode is zero.
@group(0) @binding(3) var<storage, read> keep_mask : array<i32>;
@group(0) @binding(4) var<storage, read_write> output_words : array<u32>;
@group(0) @binding(5) var<uniform> params : Params;

var<workgroup> partial_max : array<f32, 32>;
var<workgroup> partial_sum : array<f32, 32>;
var<workgroup> partial_has : array<u32, 32>;
// Flat form deliberately avoids nested workgroup arrays for Naga/GLSL/MSL
// portability. Element lane * 64 + channel belongs exclusively to that lane.
var<workgroup> partial_out : array<f32, 2048>;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn dtype_code(shift : u32) -> u32 {
  return (params.dtypes >> shift) & 255u;
}

fn q_value(index : u32, dtype : u32) -> i32 {
  let byte = word_byte(q_words[index / 4u], index);
  if (dtype == 6u) { return signed_byte(byte); }
  return i32(byte);
}

fn k_value(index : u32, dtype : u32) -> i32 {
  let byte = word_byte(k_words[index / 4u], index);
  if (dtype == 6u) { return signed_byte(byte); }
  return i32(byte);
}

fn v_value(index : u32, dtype : u32) -> i32 {
  let byte = word_byte(v_words[index / 4u], index);
  if (dtype == 6u) { return signed_byte(byte); }
  return i32(byte);
}

fn key_allowed(batch_index : u32, query : u32, key : u32) -> bool {
  if (params.causal != 0u && key > query) { return false; }
  if (params.mask_mode == 0u) { return true; }
  var index = key;
  if (params.mask_mode == 2u) {
    index = batch_index * params.seq_kv + key;
  } else if (params.mask_mode == 3u) {
    index = query * params.seq_kv + key;
  } else if (params.mask_mode == 4u) {
    index = (batch_index * params.seq_q + query) * params.seq_kv + key;
  }
  return keep_mask[index] != 0;
}

fn round_even(value : f32) -> i32 {
  let lower = floor(value);
  let fraction = value - lower;
  if (fraction < 0.5) { return i32(lower); }
  if (fraction > 0.5) { return i32(lower + 1.0); }
  let lower_i = i32(lower);
  if ((lower_i & 1) == 0) { return lower_i; }
  return lower_i + 1;
}

fn output_byte(value : i32) -> u32 {
  return bitcast<u32>(value) & 255u;
}

fn requantized_output(channel : u32) -> u32 {
  if (partial_has[0] == 0u || partial_sum[0] == 0.0) {
    return output_byte(params.output_zero_point);
  }
  let value = partial_out[channel] / partial_sum[0];
  let transformed = value / params.output_scale + f32(params.output_zero_point);
  let output_type = dtype_code(24u);
  let minimum : i32 = select(0, -128, output_type == 6u);
  let maximum : i32 = select(255, 127, output_type == 6u);
  if (transformed != transformed) { return output_byte(params.output_zero_point); }
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index : u32,
) {
  let query = workgroup_id.x;
  let head = workgroup_id.y;
  let batch_index = workgroup_id.z;
  let lane = local_invocation_index;
  if (query >= params.seq_q || head >= params.heads || batch_index >= params.batch) { return; }

  let head_dim = params.d_model / params.heads;
  let head_offset = head * head_dim;
  let q_offset = (batch_index * params.seq_q + query) * params.d_model + head_offset;
  let lane_offset = lane * 64u;
  let q_type = dtype_code(0u);
  let k_type = dtype_code(8u);
  let v_type = dtype_code(16u);
  let score_multiplier = params.q_scale * params.k_scale * params.attention_scale;
  partial_max[lane] = 0.0;
  partial_sum[lane] = 0.0;
  partial_has[lane] = 0u;
  for (var channel = 0u; channel < head_dim; channel = channel + 1u) {
    partial_out[lane_offset + channel] = 0.0;
  }

  // Each lane owns a strided subset of keys and forms an online state in its
  // own local maximum coordinate. The first valid key initializes it, which
  // avoids every -inf/-inf subtraction for empty assignments.
  for (var key = lane; key < params.seq_kv; key = key + 32u) {
    if (!key_allowed(batch_index, query, key)) { continue; }
    let kv_offset = (batch_index * params.seq_kv + key) * params.d_model + head_offset;
    var raw_dot : i32 = 0;
    for (var channel = 0u; channel < head_dim; channel = channel + 1u) {
      let q_raw = q_value(q_offset + channel, q_type) - params.q_zero_point;
      let k_raw = k_value(kv_offset + channel, k_type) - params.k_zero_point;
      raw_dot = raw_dot + q_raw * k_raw;
    }
    let score = f32(raw_dot) * score_multiplier;
    if (partial_has[lane] == 0u) {
      partial_has[lane] = 1u;
      partial_max[lane] = score;
      partial_sum[lane] = 1.0;
      for (var channel = 0u; channel < head_dim; channel = channel + 1u) {
        let raw = v_value(kv_offset + channel, v_type) - params.v_zero_point;
        partial_out[lane_offset + channel] = f32(raw) * params.v_scale;
      }
    } else if (score > partial_max[lane]) {
      let previous_weight = exp(partial_max[lane] - score);
      partial_sum[lane] = partial_sum[lane] * previous_weight + 1.0;
      for (var channel = 0u; channel < head_dim; channel = channel + 1u) {
        let raw = v_value(kv_offset + channel, v_type) - params.v_zero_point;
        let value = f32(raw) * params.v_scale;
        partial_out[lane_offset + channel] = partial_out[lane_offset + channel] * previous_weight + value;
      }
      partial_max[lane] = score;
    } else {
      let current_weight = select(exp(score - partial_max[lane]), 1.0, score == partial_max[lane]);
      partial_sum[lane] = partial_sum[lane] + current_weight;
      for (var channel = 0u; channel < head_dim; channel = channel + 1u) {
        let raw = v_value(kv_offset + channel, v_type) - params.v_zero_point;
        let value = f32(raw) * params.v_scale;
        partial_out[lane_offset + channel] = partial_out[lane_offset + channel] + current_weight * value;
      }
    }
  }

  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      let other = lane + stride;
      let has_a = partial_has[lane] != 0u;
      let has_b = partial_has[other] != 0u;
      if (!has_a && has_b) {
        partial_has[lane] = 1u;
        partial_max[lane] = partial_max[other];
        partial_sum[lane] = partial_sum[other];
        for (var channel = 0u; channel < head_dim; channel = channel + 1u) {
          partial_out[lane_offset + channel] = partial_out[other * 64u + channel];
        }
      } else if (has_a && has_b) {
        let max_a = partial_max[lane];
        let max_b = partial_max[other];
        let combined_max = max(max_a, max_b);
        let weight_a = select(exp(max_a - combined_max), 1.0, max_a == combined_max);
        let weight_b = select(exp(max_b - combined_max), 1.0, max_b == combined_max);
        partial_sum[lane] = partial_sum[lane] * weight_a + partial_sum[other] * weight_b;
        for (var channel = 0u; channel < head_dim; channel = channel + 1u) {
          partial_out[lane_offset + channel] = partial_out[lane_offset + channel] * weight_a +
            partial_out[other * 64u + channel] * weight_b;
        }
        partial_max[lane] = combined_max;
      }
    }
    workgroupBarrier();
  }

  // head_dim is divisible by four, and each head begins at a four-byte
  // boundary. Lane zero therefore exclusively owns every packed output word
  // it writes; no adjacent head or query can race a byte lane.
  if (lane == 0u) {
    let output_offset = (batch_index * params.seq_q + query) * params.d_model + head_offset;
    for (var channel = 0u; channel < head_dim; channel = channel + 4u) {
      var packed = requantized_output(channel);
      packed = packed | (requantized_output(channel + 1u) << 8u);
      packed = packed | (requantized_output(channel + 2u) << 16u);
      packed = packed | (requantized_output(channel + 3u) << 24u);
      output_words[(output_offset + channel) / 4u] = packed;
    }
  }
}
