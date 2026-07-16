@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> running_mean : array<f32>;
@group(0) @binding(3) var<storage, read> running_var : array<f32>;
@group(0) @binding(4) var<storage, read> grad_output : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_weight : array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_bias : array<f32>;

struct Params {
    length : u32,
    channels : u32,
    epsilon : f32,
    _pad : u32,
}
@group(0) @binding(8) var<uniform> params : Params;

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.length) { return; }
    let channel = index % params.channels;
    let inv = inverseSqrt(running_var[channel] + params.epsilon);
    grad_input[index] = grad_input[index] + grad_output[index] * weight[channel] * inv;
}

@compute @workgroup_size(64)
fn param_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let channel = gid.x;
    if (channel >= params.channels) { return; }
    let inv = inverseSqrt(running_var[channel] + params.epsilon);
    var weight_sum = 0.0;
    var bias_sum = 0.0;
    for (var index = channel; index < params.length; index = index + params.channels) {
        weight_sum = weight_sum + grad_output[index] * (input[index] - running_mean[channel]) * inv;
        bias_sum = bias_sum + grad_output[index];
    }
    grad_weight[channel] = grad_weight[channel] + weight_sum;
    grad_bias[channel] = grad_bias[channel] + bias_sum;
}
