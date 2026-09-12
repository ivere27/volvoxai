@group(0) @binding(0) var<storage, read> gradient : array<f32>;
@group(0) @binding(1) var<storage, read_write> accumulated : array<f32>;
// [valid, gradient_scale, norm partials..., per-target metric records...]
@group(0) @binding(2) var<storage, read_write> statistics : array<f32>;
struct Params {
    length : u32, partial_index : u32, partial_count : u32, metric_rows : u32,
    max_norm : f32, pad0 : u32, pad1 : u32, pad2 : u32,
}
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn accumulate_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x < params.length) { accumulated[gid.x] += gradient[gid.x]; }
}

@compute @workgroup_size(1)
fn norm_main() {
    var sum = 0.0;
    var valid = true;
    for (var i = 0u; i < params.length; i++) {
        let value = accumulated[i];
        valid = valid && abs(value) <= 3.402823e38;
        sum += value * value;
    }
    statistics[2u + params.partial_index] = select(-1.0, sum, valid && sum <= 3.402823e38);
}

@compute @workgroup_size(1)
fn finalize_main() {
    var sum = 0.0;
    var valid = true;
    for (var i = 0u; i < params.partial_count; i++) {
        let partial = statistics[2u + i];
        valid = valid && partial >= 0.0;
        sum += partial;
    }
    for (var r = 0u; r < params.metric_rows; r++) {
        let at = 2u + params.partial_count + r * 4u;
        valid = valid && statistics[at + 2u] >= 0.0 && abs(statistics[at]) <= 3.402823e38;
    }
    valid = valid && sum <= 3.402823e38;
    statistics[0] = select(0.0, 1.0, valid);
    var scale = 1.0;
    if (params.max_norm > 0.0 && sum > params.max_norm * params.max_norm) {
        scale = params.max_norm / (sqrt(sum) + 1.0e-12);
    }
    statistics[1] = scale;
}
