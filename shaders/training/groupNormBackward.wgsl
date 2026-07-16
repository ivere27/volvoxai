@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_weight : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_bias : array<f32>;

struct Params {
    batch : u32,
    height : u32,
    width : u32,
    channels : u32,
    groups : u32,
    has_bias : u32,
    epsilon : f32,
    _pad : u32,
}
@group(0) @binding(6) var<uniform> params : Params;

fn group_stats(batch_index : u32, group : u32) -> vec2<f32> {
    let channels_per_group = params.channels / params.groups;
    let channel_start = group * channels_per_group;
    let area = params.height * params.width;
    let count = area * channels_per_group;
    var sum = 0.0;
    var square_sum = 0.0;
    for (var spatial = 0u; spatial < area; spatial = spatial + 1u) {
        let base = (batch_index * area + spatial) * params.channels + channel_start;
        for (var local_channel = 0u; local_channel < channels_per_group; local_channel = local_channel + 1u) {
            let value = input[base + local_channel];
            sum = sum + value;
            square_sum = square_sum + value * value;
        }
    }
    let mean = sum / f32(count);
    let variance = max(0.0, square_sum / f32(count) - mean * mean);
    return vec2<f32>(mean, inverseSqrt(variance + params.epsilon));
}

// One invocation owns a complete sample/group. This avoids floating-point
// atomics while keeping the expensive group reductions linear in tensor size.
@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat_group = gid.x;
    let total_groups = params.batch * params.groups;
    if (flat_group >= total_groups) { return; }
    let batch_index = flat_group / params.groups;
    let group = flat_group - batch_index * params.groups;
    let channels_per_group = params.channels / params.groups;
    let channel_start = group * channels_per_group;
    let area = params.height * params.width;
    let count = area * channels_per_group;
    let stats = group_stats(batch_index, group);

    var sum_scaled_grad = 0.0;
    var sum_scaled_grad_xhat = 0.0;
    for (var spatial = 0u; spatial < area; spatial = spatial + 1u) {
        let base = (batch_index * area + spatial) * params.channels + channel_start;
        for (var local_channel = 0u; local_channel < channels_per_group; local_channel = local_channel + 1u) {
            let channel = channel_start + local_channel;
            let index = base + local_channel;
            let xhat = (input[index] - stats.x) * stats.y;
            let scaled_grad = grad_output[index] * weight[channel];
            sum_scaled_grad = sum_scaled_grad + scaled_grad;
            sum_scaled_grad_xhat = sum_scaled_grad_xhat + scaled_grad * xhat;
        }
    }

    for (var spatial = 0u; spatial < area; spatial = spatial + 1u) {
        let base = (batch_index * area + spatial) * params.channels + channel_start;
        for (var local_channel = 0u; local_channel < channels_per_group; local_channel = local_channel + 1u) {
            let channel = channel_start + local_channel;
            let index = base + local_channel;
            let xhat = (input[index] - stats.x) * stats.y;
            let scaled_grad = grad_output[index] * weight[channel];
            let dx = stats.y / f32(count) *
                (f32(count) * scaled_grad - sum_scaled_grad - xhat * sum_scaled_grad_xhat);
            grad_input[index] = grad_input[index] + dx;
        }
    }
}

// A channel invocation owns its parameter gradients. Statistics are recomputed
// per sample to avoid non-portable float atomics and cross-workgroup scratch.
@compute @workgroup_size(64)
fn param_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let channel = gid.x;
    if (channel >= params.channels) { return; }
    let group = channel / (params.channels / params.groups);
    let area = params.height * params.width;
    var weight_sum = 0.0;
    var bias_sum = 0.0;
    for (var batch_index = 0u; batch_index < params.batch; batch_index = batch_index + 1u) {
        let stats = group_stats(batch_index, group);
        for (var spatial = 0u; spatial < area; spatial = spatial + 1u) {
            let index = (batch_index * area + spatial) * params.channels + channel;
            weight_sum = weight_sum + grad_output[index] * (input[index] - stats.x) * stats.y;
            bias_sum = bias_sum + grad_output[index];
        }
    }
    grad_weight[channel] = grad_weight[channel] + weight_sum;
    if (params.has_bias != 0u) {
        grad_bias[channel] = grad_bias[channel] + bias_sum;
    }
}
