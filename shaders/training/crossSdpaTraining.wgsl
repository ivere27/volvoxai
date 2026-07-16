@group(0) @binding(0) var<storage, read> q_input : array<f32>;
@group(0) @binding(1) var<storage, read> k_input : array<f32>;
@group(0) @binding(2) var<storage, read> v_input : array<f32>;
@group(0) @binding(3) var<storage, read> mask : array<i32>;
@group(0) @binding(4) var<storage, read_write> output : array<f32>;

struct Params {
    seq_q : u32,
    seq_kv : u32,
    d_model : u32,
    num_heads : u32,
    head_dim : u32,
    batch : u32,
    scale : f32,
    causal : u32,
    mask_mode : u32,
    dropout_threshold : u32,
    dropout_seed : u32,
    dropout_counter : u32,
    dropout_scale : f32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
}
@group(0) @binding(5) var<uniform> params : Params;

fn key_allowed(batch_idx : u32, query : u32, key : u32) -> bool {
    if (params.causal != 0u && key > query) { return false; }
    if (params.mask_mode == 0u) { return true; }
    var index = key;
    if (params.mask_mode == 2u) {
        index = batch_idx * params.seq_kv + key;
    } else if (params.mask_mode == 3u) {
        index = query * params.seq_kv + key;
    } else if (params.mask_mode == 4u) {
        index = (batch_idx * params.seq_q + query) * params.seq_kv + key;
    }
    return mask[index] != 0;
}

fn probability_index(batch_idx : u32, head : u32, query : u32, key : u32) -> u32 {
    return (((batch_idx * params.num_heads + head) * params.seq_q + query) *
        params.seq_kv + key);
}

fn random_bits(index : u32) -> u32 {
    var value = params.dropout_seed ^ index ^ (params.dropout_counter * 0x9e3779b9u);
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

fn dropout_multiplier(batch_idx : u32, head : u32, query : u32, key : u32) -> f32 {
    if (random_bits(probability_index(batch_idx, head, query, key)) >=
        params.dropout_threshold) {
        return params.dropout_scale;
    }
    return 0.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let query = global_id.x;
    let head = global_id.y;
    let batch_idx = global_id.z;
    if (query >= params.seq_q || head >= params.num_heads || batch_idx >= params.batch) { return; }

    let q_base = batch_idx * params.seq_q * params.d_model;
    let kv_base = batch_idx * params.seq_kv * params.d_model;
    var q_cache : array<f32, 64>;
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        q_cache[dimension] = q_input[q_base + query * params.d_model +
            head * params.head_dim + dimension];
    }

    var maximum = -1e38;
    var valid_keys = 0u;
    for (var key = 0u; key < params.seq_kv; key = key + 1u) {
        if (!key_allowed(batch_idx, query, key)) { continue; }
        var score = 0.0;
        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
            score = score + q_cache[dimension] *
                k_input[kv_base + key * params.d_model + head * params.head_dim + dimension];
        }
        maximum = max(maximum, score * params.scale);
        valid_keys = valid_keys + 1u;
    }

    if (valid_keys == 0u) {
        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
            output[q_base + query * params.d_model + head * params.head_dim + dimension] = 0.0;
        }
        return;
    }

    var denominator = 0.0;
    for (var key = 0u; key < params.seq_kv; key = key + 1u) {
        if (!key_allowed(batch_idx, query, key)) { continue; }
        var score = 0.0;
        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
            score = score + q_cache[dimension] *
                k_input[kv_base + key * params.d_model + head * params.head_dim + dimension];
        }
        denominator = denominator + exp(score * params.scale - maximum);
    }

    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        var value = 0.0;
        for (var key = 0u; key < params.seq_kv; key = key + 1u) {
            if (!key_allowed(batch_idx, query, key)) { continue; }
            var score = 0.0;
            for (var inner = 0u; inner < params.head_dim; inner = inner + 1u) {
                score = score + q_cache[inner] *
                    k_input[kv_base + key * params.d_model + head * params.head_dim + inner];
            }
            let probability = exp(score * params.scale - maximum) / denominator;
            value = value + probability * dropout_multiplier(batch_idx, head, query, key) *
                v_input[kv_base + key * params.d_model + head * params.head_dim + dimension];
        }
        output[q_base + query * params.d_model + head * params.head_dim + dimension] = value;
    }
}
