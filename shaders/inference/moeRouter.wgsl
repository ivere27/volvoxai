@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> route_indices : array<f32>;
@group(0) @binding(4) var<storage, read_write> route_weights : array<f32>;

struct Params {
    rows : u32,
    d_model : u32,
    num_experts : u32,
    top_k : u32,
    normalize : u32,
    has_bias : u32,
    temperature : f32,
}
@group(0) @binding(5) var<uniform> params : Params;

const MAX_TOP_K : u32 = 8u;

fn router_logit(row : u32, expert : u32) -> f32 {
    var value = 0.0;
    if (params.has_bias != 0u) { value = bias[expert]; }
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        value = value + input[row * params.d_model + d] *
                        weight[d * params.num_experts + expert];
    }
    return value / params.temperature;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows || params.top_k == 0u || params.top_k > MAX_TOP_K) { return; }

    var selected : array<u32, 8>;
    var selected_values : array<f32, 8>;
    for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
        var best = 0u;
        var best_value = -3.402823e38;
        for (var expert = 0u; expert < params.num_experts; expert = expert + 1u) {
            var used = false;
            for (var prior = 0u; prior < slot; prior = prior + 1u) {
                if (selected[prior] == expert) { used = true; }
            }
            let value = router_logit(row, expert);
            if (!used && (value > best_value || (value == best_value && expert < best))) {
                best = expert;
                best_value = value;
            }
        }
        selected[slot] = best;
        selected_values[slot] = best_value;
    }

    var max_value = -3.402823e38;
    for (var expert = 0u; expert < params.num_experts; expert = expert + 1u) {
        max_value = max(max_value, router_logit(row, expert));
    }
    var denominator = 0.0;
    if (params.normalize != 0u) {
        for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
            denominator = denominator + exp(selected_values[slot] - max_value);
        }
    } else {
        for (var expert = 0u; expert < params.num_experts; expert = expert + 1u) {
            denominator = denominator + exp(router_logit(row, expert) - max_value);
        }
    }
    for (var slot = 0u; slot < params.top_k; slot = slot + 1u) {
        let offset = row * params.top_k + slot;
        route_indices[offset] = f32(selected[slot]);
        route_weights[offset] = exp(selected_values[slot] - max_value) / denominator;
    }
}
