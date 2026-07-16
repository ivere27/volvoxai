@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> expert_weight : array<f32>;
@group(0) @binding(2) var<storage, read> expert_bias : array<f32>;
@group(0) @binding(3) var<storage, read> route_indices : array<f32>;
@group(0) @binding(4) var<storage, read> route_weights : array<f32>;
@group(0) @binding(5) var<storage, read> grad_output : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_expert_weight : array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_expert_bias : array<f32>;
@group(0) @binding(9) var<storage, read_write> grad_route_weights : array<f32>;

struct Params {
    rows : u32,
    d_in : u32,
    d_out : u32,
    num_experts : u32,
    top_k : u32,
    has_bias : u32,
}
@group(0) @binding(10) var<uniform> params : Params;

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.rows * params.d_in) { return; }
    let row = flat / params.d_in;
    let d = flat - row * params.d_in;
    var sum = 0.0;
    for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
        let route = row * params.top_k + slot;
        let expert = u32(route_indices[route]);
        if (expert >= params.num_experts) { continue; }
        let gate = route_weights[route];
        let base = expert * params.d_in * params.d_out;
        for (var col = 0u; col < params.d_out; col = col + 1u) {
            sum = sum + gate * grad_output[row * params.d_out + col] *
                expert_weight[base + d * params.d_out + col];
        }
    }
    grad_input[flat] = grad_input[flat] + sum;
}

@compute @workgroup_size(64)
fn weight_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    let count = params.num_experts * params.d_in * params.d_out;
    if (flat >= count) { return; }
    var tmp = flat;
    let col = tmp % params.d_out; tmp = tmp / params.d_out;
    let d = tmp % params.d_in; let expert = tmp / params.d_in;
    var sum = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
            let route = row * params.top_k + slot;
            if (u32(route_indices[route]) == expert) {
                sum = sum + route_weights[route] * input[row * params.d_in + d] *
                    grad_output[row * params.d_out + col];
            }
        }
    }
    grad_expert_weight[flat] = grad_expert_weight[flat] + sum;
}

@compute @workgroup_size(64)
fn bias_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.num_experts * params.d_out || params.has_bias == 0u) { return; }
    let expert = flat / params.d_out;
    let col = flat - expert * params.d_out;
    var sum = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
            let route = row * params.top_k + slot;
            if (u32(route_indices[route]) == expert) {
                sum = sum + route_weights[route] * grad_output[row * params.d_out + col];
            }
        }
    }
    grad_expert_bias[flat] = grad_expert_bias[flat] + sum;
}

@compute @workgroup_size(64)
fn route_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let route = gid.x;
    if (route >= params.rows * params.top_k) { return; }
    let row = route / params.top_k;
    let expert = u32(route_indices[route]);
    if (expert >= params.num_experts) { return; }
    let base = expert * params.d_in * params.d_out;
    var sum = 0.0;
    for (var col = 0u; col < params.d_out; col = col + 1u) {
        var value = 0.0;
        if (params.has_bias != 0u) { value = expert_bias[expert * params.d_out + col]; }
        for (var d = 0u; d < params.d_in; d = d + 1u) {
            value = value + input[row * params.d_in + d] *
                expert_weight[base + d * params.d_out + col];
        }
        sum = sum + grad_output[row * params.d_out + col] * value;
    }
    grad_route_weights[route] = grad_route_weights[route] + sum;
}
