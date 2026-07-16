@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_weight : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_bias : array<f32>;

struct Params {
    rows : u32,
    d_model : u32,
    epsilon : f32,
    has_bias : u32,
}
@group(0) @binding(6) var<uniform> params : Params;

fn row_stats(row : u32) -> vec2<f32> {
    let offset = row * params.d_model;
    var sum = 0.0;
    var square_sum = 0.0;
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        let x = input[offset + d];
        sum = sum + x;
        square_sum = square_sum + x * x;
    }
    let mean = sum / f32(params.d_model);
    let variance = max(0.0, square_sum / f32(params.d_model) - mean * mean);
    return vec2<f32>(mean, inverseSqrt(variance + params.epsilon));
}

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    let total = params.rows * params.d_model;
    if (flat >= total) { return; }
    let row = flat / params.d_model;
    let feature = flat - row * params.d_model;
    let stats = row_stats(row);
    let offset = row * params.d_model;
    var sum_dy = 0.0;
    var sum_dy_xhat = 0.0;
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        let xhat = (input[offset + d] - stats.x) * stats.y;
        let scaled_grad = grad_output[offset + d] * weight[d];
        sum_dy = sum_dy + scaled_grad;
        sum_dy_xhat = sum_dy_xhat + scaled_grad * xhat;
    }
    let xhat = (input[flat] - stats.x) * stats.y;
    let dx = stats.y / f32(params.d_model) *
        (f32(params.d_model) * grad_output[flat] * weight[feature] - sum_dy - xhat * sum_dy_xhat);
    grad_input[flat] = grad_input[flat] + dx;
}

@compute @workgroup_size(64)
fn param_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let feature = gid.x;
    if (feature >= params.d_model) { return; }
    var dw = 0.0;
    var db = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        let stats = row_stats(row);
        let offset = row * params.d_model + feature;
        dw = dw + grad_output[offset] * (input[offset] - stats.x) * stats.y;
        db = db + grad_output[offset];
    }
    grad_weight[feature] = grad_weight[feature] + dw;
    if (params.has_bias != 0u) { grad_bias[feature] = grad_bias[feature] + db; }
}
