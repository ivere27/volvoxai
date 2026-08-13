@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> output: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_bias: array<f32>;
struct Params { shape: vec4<u32>, config: vec4<u32>, group: vec4<u32> }
@group(0) @binding(7) var<uniform> params: Params;

// NLC activations [batch, l, c]; WIO weights [k, in_per_group, out_c].
// shape = (in_c, in_l, out_c, k); config = (stride, pad, relu, batch);
// group = (groups, in_per_group, out_l, _).
fn out_index(b: u32, oc: u32, ox: u32) -> u32 { return (b * params.group.z + ox) * params.shape.z + oc; }
fn in_index(b: u32, ic: u32, x: u32) -> u32 { return (b * params.shape.y + x) * params.shape.x + ic; }
fn weight_index(kk: u32, local_ic: u32, oc: u32) -> u32 { return (kk * params.group.y + local_ic) * params.shape.z + oc; }
fn gated_gradient(b: u32, oc: u32, ox: u32) -> f32 {
  let index = out_index(b, oc, ox);
  if (params.config.z != 0u && output[index] <= 0.0) { return 0.0; }
  return grad_output[index];
}

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x; let input_elements = params.config.w * params.shape.x * params.shape.y;
  if (index >= input_elements) { return; }
  let channel = index % params.shape.x; let x = (index / params.shape.x) % params.shape.y; let batch = index / (params.shape.x * params.shape.y);
  let group = channel / params.group.y; let local_channel = channel % params.group.y; let group_out = params.shape.z / params.group.x;
  var sum = 0.0;
  for (var local_oc = 0u; local_oc < group_out; local_oc = local_oc + 1u) {
    let oc = group * group_out + local_oc;
    for (var ox = 0u; ox < params.group.z; ox = ox + 1u) {
      for (var kk = 0u; kk < params.shape.w; kk = kk + 1u) {
        if (i32(ox * params.config.x + kk) - i32(params.config.y) == i32(x)) {
          sum = sum + gated_gradient(batch, oc, ox) * weight[weight_index(kk, local_channel, oc)];
        }
      }
    }
  }
  grad_input[index] = grad_input[index] + sum;
}

@compute @workgroup_size(64)
fn weight_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x; let weight_elements = params.shape.z * params.group.y * params.shape.w;
  if (index >= weight_elements) { return; }
  let oc = index % params.shape.z; let local_channel = (index / params.shape.z) % params.group.y; let kk = index / (params.shape.z * params.group.y);
  let group_out = params.shape.z / params.group.x; let group = oc / group_out; let channel = group * params.group.y + local_channel;
  var sum = 0.0;
  for (var batch = 0u; batch < params.config.w; batch = batch + 1u) {
    for (var ox = 0u; ox < params.group.z; ox = ox + 1u) {
      let x = i32(ox * params.config.x + kk) - i32(params.config.y);
      if (x >= 0 && x < i32(params.shape.y)) { sum = sum + gated_gradient(batch, oc, ox) * input[in_index(batch, channel, u32(x))]; }
    }
  }
  grad_weight[index] = grad_weight[index] + sum;
}

@compute @workgroup_size(64)
fn bias_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let oc = gid.x; if (oc >= params.shape.z) { return; } var sum = 0.0;
  for (var batch = 0u; batch < params.config.w; batch = batch + 1u) {
    for (var ox = 0u; ox < params.group.z; ox = ox + 1u) { sum = sum + gated_gradient(batch, oc, ox); }
  }
  grad_bias[oc] = grad_bias[oc] + sum;
}
