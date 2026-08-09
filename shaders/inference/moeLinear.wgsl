@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> expert_weight : array<f32>;
@group(0) @binding(2) var<storage, read> expert_bias : array<f32>;
@group(0) @binding(3) var<storage, read> route_indices : array<f32>;
@group(0) @binding(4) var<storage, read> route_weights : array<f32>;
@group(0) @binding(5) var<storage, read_write> output : array<f32>;

// Global-slot -> staged-row table for a partially resident expert bank. Public
// routes are host-preflighted and invariant routes are checked at compile time.
// Entries
// outside the resident set hold VX_MOE_SLOT_ABSENT (0xffffffff). When
// slot_domain is zero the bank is fully resident and route ids are already rows.
@group(0) @binding(7) var<storage, read> slot_rows : array<u32>;

struct Params {
    rows : u32,
    d_in : u32,
    d_out : u32,
    num_experts : u32,
    top_k : u32,
    has_bias : u32,
    slot_domain : u32,
}
@group(0) @binding(6) var<uniform> params : Params;

const VX_MOE_SLOT_ABSENT : u32 = 0xffffffffu;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.rows || col >= params.d_out) { return; }
    var sum = 0.0;
    for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
        let route_offset = row * params.top_k + slot;
        let routed = u32(route_indices[route_offset]);
        var expert = routed;
        if (params.slot_domain != 0u) {
            if (routed >= params.slot_domain) { continue; }
            expert = slot_rows[routed];
            if (expert == VX_MOE_SLOT_ABSENT) { continue; }
        }
        if (expert >= params.num_experts) { continue; }
        var expert_value = 0.0;
        if (params.has_bias != 0u) {
            expert_value = expert_bias[expert * params.d_out + col];
        }
        let expert_base = expert * params.d_in * params.d_out;
        for (var d = 0u; d < params.d_in; d = d + 1u) {
            expert_value = expert_value + input[row * params.d_in + d] *
                expert_weight[expert_base + d * params.d_out + col];
        }
        sum = sum + route_weights[route_offset] * expert_value;
    }
    output[row * params.d_out + col] = sum;
}
