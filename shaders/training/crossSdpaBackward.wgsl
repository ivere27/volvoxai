@group(0) @binding(0) var<storage, read> q_input : array<f32>;
@group(0) @binding(1) var<storage, read> k_input : array<f32>;
@group(0) @binding(2) var<storage, read> v_input : array<f32>;
@group(0) @binding(3) var<storage, read> mask : array<i32>;
@group(0) @binding(4) var<storage, read> grad_output : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_q : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_k : array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_v : array<f32>;

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
@group(0) @binding(8) var<uniform> params : Params;

fn key_allowed(batch : u32, query : u32, key : u32) -> bool {
    if (params.causal != 0u && key > query) { return false; }
    if (params.mask_mode == 0u) { return true; }
    var index = key;
    if (params.mask_mode == 2u) {
        index = batch * params.seq_kv + key;
    } else if (params.mask_mode == 3u) {
        index = query * params.seq_kv + key;
    } else if (params.mask_mode == 4u) {
        index = (batch * params.seq_q + query) * params.seq_kv + key;
    }
    return mask[index] != 0;
}

fn probability_index(batch : u32, head : u32, query : u32, key : u32) -> u32 {
    return (((batch * params.num_heads + head) * params.seq_q + query) *
        params.seq_kv + key);
}

fn random_bits(index : u32) -> u32 {
    var value = params.dropout_seed ^ index ^ (params.dropout_counter * 0x9e3779b9u);
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

fn dropout_multiplier(batch : u32, head : u32, query : u32, key : u32) -> f32 {
    if (random_bits(probability_index(batch, head, query, key)) >=
        params.dropout_threshold) {
        return params.dropout_scale;
    }
    return 0.0;
}

fn q_value(batch : u32, query : u32, head : u32, dimension : u32) -> f32 {
    let base = batch * params.seq_q * params.d_model;
    return q_input[base + query * params.d_model + head * params.head_dim + dimension];
}

fn k_value(batch : u32, key : u32, head : u32, dimension : u32) -> f32 {
    let base = batch * params.seq_kv * params.d_model;
    return k_input[base + key * params.d_model + head * params.head_dim + dimension];
}

fn v_value(batch : u32, key : u32, head : u32, dimension : u32) -> f32 {
    let base = batch * params.seq_kv * params.d_model;
    return v_input[base + key * params.d_model + head * params.head_dim + dimension];
}

fn score(batch : u32, query : u32, key : u32, head : u32) -> f32 {
    var value = 0.0;
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        value = value + q_value(batch, query, head, dimension) *
            k_value(batch, key, head, dimension);
    }
    return value * params.scale;
}

fn probability(batch : u32, query : u32, key : u32, head : u32) -> f32 {
    if (!key_allowed(batch, query, key)) { return 0.0; }
    var maximum = -3.402823e38;
    for (var candidate = 0u; candidate < params.seq_kv; candidate = candidate + 1u) {
        if (!key_allowed(batch, query, candidate)) { continue; }
        maximum = max(maximum, score(batch, query, candidate, head));
    }
    var denominator = 0.0;
    for (var candidate = 0u; candidate < params.seq_kv; candidate = candidate + 1u) {
        if (!key_allowed(batch, query, candidate)) { continue; }
        denominator = denominator + exp(score(batch, query, candidate, head) - maximum);
    }
    if (denominator == 0.0) { return 0.0; }
    return exp(score(batch, query, key, head) - maximum) / denominator;
}

fn probability_gradient(batch : u32, query : u32, key : u32, head : u32) -> f32 {
    if (!key_allowed(batch, query, key)) { return 0.0; }
    let output_offset = batch * params.seq_q * params.d_model +
        query * params.d_model + head * params.head_dim;
    var direct = 0.0;
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        direct = direct + grad_output[output_offset + dimension] *
            v_value(batch, key, head, dimension);
    }
    direct = direct * dropout_multiplier(batch, head, query, key);
    var expectation = 0.0;
    for (var candidate = 0u; candidate < params.seq_kv; candidate = candidate + 1u) {
        if (!key_allowed(batch, query, candidate)) { continue; }
        var candidate_gradient = 0.0;
        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
            candidate_gradient = candidate_gradient + grad_output[output_offset + dimension] *
                v_value(batch, candidate, head, dimension);
        }
        candidate_gradient = candidate_gradient *
            dropout_multiplier(batch, head, query, candidate);
        expectation = expectation + probability(batch, query, candidate, head) * candidate_gradient;
    }
    return probability(batch, query, key, head) * (direct - expectation);
}

@compute @workgroup_size(1, 1, 1)
fn q_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let query = gid.x;
    let head = gid.y;
    let batch = gid.z;
    if (query >= params.seq_q || head >= params.num_heads || batch >= params.batch) { return; }
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        var gradient = 0.0;
        for (var key = 0u; key < params.seq_kv; key = key + 1u) {
            gradient = gradient + probability_gradient(batch, query, key, head) *
                k_value(batch, key, head, dimension) * params.scale;
        }
        let offset = batch * params.seq_q * params.d_model +
            query * params.d_model + head * params.head_dim + dimension;
        grad_q[offset] = grad_q[offset] + gradient;
    }
}

@compute @workgroup_size(1, 1, 1)
fn k_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let key = gid.x;
    let head = gid.y;
    let batch = gid.z;
    if (key >= params.seq_kv || head >= params.num_heads || batch >= params.batch) { return; }
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        var gradient = 0.0;
        for (var query = 0u; query < params.seq_q; query = query + 1u) {
            gradient = gradient + probability_gradient(batch, query, key, head) *
                q_value(batch, query, head, dimension) * params.scale;
        }
        let offset = batch * params.seq_kv * params.d_model +
            key * params.d_model + head * params.head_dim + dimension;
        grad_k[offset] = grad_k[offset] + gradient;
    }
}

@compute @workgroup_size(1, 1, 1)
fn v_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let key = gid.x;
    let head = gid.y;
    let batch = gid.z;
    if (key >= params.seq_kv || head >= params.num_heads || batch >= params.batch) { return; }
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        var gradient = 0.0;
        for (var query = 0u; query < params.seq_q; query = query + 1u) {
            gradient = gradient + probability(batch, query, key, head) *
                dropout_multiplier(batch, head, query, key) *
                grad_output[batch * params.seq_q * params.d_model +
                    query * params.d_model + head * params.head_dim + dimension];
        }
        let offset = batch * params.seq_kv * params.d_model +
            key * params.d_model + head * params.head_dim + dimension;
        grad_v[offset] = grad_v[offset] + gradient;
    }
}
