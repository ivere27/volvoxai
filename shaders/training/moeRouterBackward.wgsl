@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> router_weight : array<f32>;
@group(0) @binding(2) var<storage, read> router_bias : array<f32>;
@group(0) @binding(3) var<storage, read> route_indices : array<f32>;
@group(0) @binding(4) var<storage, read> route_weights : array<f32>;
@group(0) @binding(5) var<storage, read> grad_route_weights : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_logits : array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_router_weight : array<f32>;
@group(0) @binding(9) var<storage, read_write> grad_router_bias : array<f32>;

struct Params {
    rows : u32,
    d_model : u32,
    num_experts : u32,
    top_k : u32,
    normalize : u32,
    has_bias : u32,
    temperature : f32,
}
@group(0) @binding(10) var<uniform> params : Params;

fn router_logit(row : u32, expert : u32) -> f32 {
    var value = 0.0;
    if (params.has_bias != 0u) { value = router_bias[expert]; }
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        value = value + input[row * params.d_model + d] *
            router_weight[d * params.num_experts + expert];
    }
    return value / params.temperature;
}

fn selected_slot(row : u32, expert : u32) -> i32 {
    for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
        if (u32(route_indices[row * params.top_k + slot]) == expert) { return i32(slot); }
    }
    return -1;
}

fn selected_expectation(row : u32) -> f32 {
    var expectation = 0.0;
    for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
        let offset = row * params.top_k + slot;
        expectation = expectation + route_weights[offset] * grad_route_weights[offset];
    }
    return expectation;
}

fn full_probability(row : u32, expert : u32) -> f32 {
    var maximum = -3.402823e38;
    for (var candidate = 0u; candidate < params.num_experts; candidate = candidate + 1u) {
        maximum = max(maximum, router_logit(row, candidate));
    }
    var denominator = 0.0;
    for (var candidate = 0u; candidate < params.num_experts; candidate = candidate + 1u) {
        denominator = denominator + exp(router_logit(row, candidate) - maximum);
    }
    return exp(router_logit(row, expert) - maximum) / denominator;
}

// Materialize dL/d(unscaled router logits) once. In the full-softmax routing
// mode, unselected experts receive the negative denominator term as required
// by the softmax Jacobian. Top-k selection itself remains non-differentiable.
@compute @workgroup_size(64)
fn logit_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.rows * params.num_experts) { return; }
    let row = flat / params.num_experts;
    let expert = flat - row * params.num_experts;
    let slot = selected_slot(row, expert);
    let expectation = selected_expectation(row);
    var gradient = 0.0;
    if (params.normalize != 0u) {
        if (slot >= 0) {
            let offset = row * params.top_k + u32(slot);
            gradient = route_weights[offset] * (grad_route_weights[offset] - expectation);
        }
    } else {
        var direct = 0.0;
        if (slot >= 0) { direct = grad_route_weights[row * params.top_k + u32(slot)]; }
        gradient = full_probability(row, expert) * (direct - expectation);
    }
    grad_logits[flat] = gradient / params.temperature;
}

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.rows * params.d_model) { return; }
    let row = flat / params.d_model;
    let d = flat - row * params.d_model;
    var sum = 0.0;
    for (var expert = 0u; expert < params.num_experts; expert = expert + 1u) {
        sum = sum + grad_logits[row * params.num_experts + expert] *
            router_weight[d * params.num_experts + expert];
    }
    grad_input[flat] = grad_input[flat] + sum;
}

@compute @workgroup_size(64)
fn weight_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.d_model * params.num_experts) { return; }
    let d = flat / params.num_experts;
    let expert = flat - d * params.num_experts;
    var sum = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        sum = sum + input[row * params.d_model + d] *
            grad_logits[row * params.num_experts + expert];
    }
    grad_router_weight[flat] = grad_router_weight[flat] + sum;
}

@compute @workgroup_size(64)
fn bias_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let expert = gid.x;
    if (expert >= params.num_experts || params.has_bias == 0u) { return; }
    var sum = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        sum = sum + grad_logits[row * params.num_experts + expert];
    }
    grad_router_bias[expert] = grad_router_bias[expert] + sum;
}
