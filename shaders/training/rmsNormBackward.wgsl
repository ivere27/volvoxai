@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_weight : array<f32>;

struct Params {
    rows : u32,
    d_model : u32,
    epsilon : f32,
    _pad : u32,
}
@group(0) @binding(5) var<uniform> params : Params;

fn inverse_rms(row : u32) -> f32 {
    let offset = row * params.d_model;
    var square_sum = 0.0;
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        let x = input[offset + d];
        square_sum = square_sum + x * x;
    }
    return inverseSqrt(square_sum / f32(params.d_model) + params.epsilon);
}

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    let total = params.rows * params.d_model;
    if (flat >= total) { return; }
    let row = flat / params.d_model;
    let feature = flat - row * params.d_model;
    let offset = row * params.d_model;
    let inv = inverse_rms(row);
    var dot = 0.0;
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        dot = dot + grad_output[offset + d] * weight[d] * input[offset + d];
    }
    let dx = grad_output[flat] * weight[feature] * inv -
        input[flat] * inv * inv * inv * dot / f32(params.d_model);
    grad_input[flat] = grad_input[flat] + dx;
}

@compute @workgroup_size(64)
fn weight_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let feature = gid.x;
    if (feature >= params.d_model) { return; }
    var sum = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        let offset = row * params.d_model + feature;
        sum = sum + grad_output[offset] * input[offset] * inverse_rms(row);
    }
    grad_weight[feature] = grad_weight[feature] + sum;
}
