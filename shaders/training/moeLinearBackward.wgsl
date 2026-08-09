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

// Partially resident expert bank. `slot_rows` maps a global slot id to its
// staged row (VX_MOE_SLOT_ABSENT when the context did not materialize it) and
// `row_slots` is the inverse, needed by the entries that iterate staged rows.
// A zero slot_domain means the bank is fully resident and ids are already rows.
@group(0) @binding(11) var<storage, read> slot_rows : array<u32>;
@group(0) @binding(12) var<storage, read> row_slots : array<u32>;

struct Params {
    rows : u32,
    d_in : u32,
    d_out : u32,
    num_experts : u32,
    top_k : u32,
    has_bias : u32,
    slot_domain : u32,
}
@group(0) @binding(10) var<uniform> params : Params;

const VX_MOE_SLOT_ABSENT : u32 = 0xffffffffu;

/// Global slot id -> staged row, or VX_MOE_SLOT_ABSENT.
fn staged_row(routed : u32) -> u32 {
    if (params.slot_domain == 0u) { return routed; }
    if (routed >= params.slot_domain) { return VX_MOE_SLOT_ABSENT; }
    return slot_rows[routed];
}

/// Staged row -> global slot id, for entries that iterate rows.
fn global_slot(row : u32) -> u32 {
    if (params.slot_domain == 0u) { return row; }
    if (row >= params.num_experts) { return VX_MOE_SLOT_ABSENT; }
    return row_slots[row];
}

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.rows * params.d_in) { return; }
    let row = flat / params.d_in;
    let d = flat - row * params.d_in;
    var sum = 0.0;
    for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
        let route = row * params.top_k + slot;
        let expert = staged_row(u32(route_indices[route]));
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
    let routed_id = global_slot(expert);
    var sum = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
            let route = row * params.top_k + slot;
            if (u32(route_indices[route]) == routed_id) {
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
    let routed_id = global_slot(expert);
    var sum = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
            let route = row * params.top_k + slot;
            if (u32(route_indices[route]) == routed_id) {
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
    let expert = staged_row(u32(route_indices[route]));
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
