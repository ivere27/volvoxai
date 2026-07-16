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
