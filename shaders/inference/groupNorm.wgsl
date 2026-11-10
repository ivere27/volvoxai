@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

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
@group(0) @binding(4) var<uniform> params : Params;

// Compensate long spatial reductions without requiring device F64. Computing
// E[x*x] - mean*mean loses the variance of nearly constant, offset channels.
fn compensated_add(state : vec2<f32>, value : f32) -> vec2<f32> {
    let corrected = value - state.y;
    let total = state.x + corrected;
    return vec2<f32>(total, (total - state.x) - corrected);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat_group = gid.x;
    let total_groups = params.batch * params.groups;
    if (flat_group >= total_groups) { return; }

    let batch_index = flat_group / params.groups;
    let group = flat_group - batch_index * params.groups;
    let channels_per_group = params.channels / params.groups;
    let channel_start = group * channels_per_group;
    let area = params.height * params.width;
    let count = area * channels_per_group;

    var sum = vec2<f32>(0.0);
    for (var spatial = 0u; spatial < area; spatial = spatial + 1u) {
        let base = (batch_index * area + spatial) * params.channels + channel_start;
        for (var local_channel = 0u; local_channel < channels_per_group; local_channel = local_channel + 1u) {
            let value = input[base + local_channel];
            sum = compensated_add(sum, value);
        }
    }
    let mean = sum.x / f32(count);
    var centered_square_sum = vec2<f32>(0.0);
    for (var spatial = 0u; spatial < area; spatial = spatial + 1u) {
        let base = (batch_index * area + spatial) * params.channels + channel_start;
        for (var local_channel = 0u; local_channel < channels_per_group; local_channel = local_channel + 1u) {
            let centered = input[base + local_channel] - mean;
            centered_square_sum = compensated_add(centered_square_sum, centered * centered);
        }
    }
    let variance = centered_square_sum.x / f32(count);
    let inverse_stddev = inverseSqrt(variance + params.epsilon);

    for (var spatial = 0u; spatial < area; spatial = spatial + 1u) {
        let base = (batch_index * area + spatial) * params.channels + channel_start;
        for (var local_channel = 0u; local_channel < channels_per_group; local_channel = local_channel + 1u) {
            let channel = channel_start + local_channel;
            let index = base + local_channel;
            let normalized = (input[index] - mean) * inverse_stddev;
            let offset = select(0.0, bias[channel], params.has_bias != 0u);
            output[index] = normalized * weight[channel] + offset;
        }
    }
}
