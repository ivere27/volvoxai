@group(0) @binding(0) var<storage, read> qkv : array<f32>;
@group(0) @binding(1) var<storage, read> mask : array<i32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_qkv : array<f32>;

struct Params {
    seq_len : u32,
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
}
@group(0) @binding(4) var<uniform> params : Params;

fn key_allowed(batch : u32, query : u32, key : u32) -> bool {
    if (params.causal != 0u && key > query) { return false; }
    if (params.mask_mode == 0u) { return true; }
    var index = key;
    if (params.mask_mode == 2u) {
        index = batch * params.seq_len + key;
    } else if (params.mask_mode == 3u) {
        index = query * params.seq_len + key;
    } else if (params.mask_mode == 4u) {
        index = (batch * params.seq_len + query) * params.seq_len + key;
    }
    return mask[index] != 0;
}

fn probability_index(batch : u32, head : u32, query : u32, key : u32) -> u32 {
    return (((batch * params.num_heads + head) * params.seq_len + query) *
        params.seq_len + key);
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

fn q_value(batch : u32, q : u32, h : u32, d : u32) -> f32 {
    let base = batch * params.seq_len * params.d_model * 3u;
    return qkv[base + q * (params.d_model * 3u) + h * params.head_dim + d];
}

fn k_value(batch : u32, k : u32, h : u32, d : u32) -> f32 {
    let base = batch * params.seq_len * params.d_model * 3u;
    return qkv[base + k * (params.d_model * 3u) + params.d_model + h * params.head_dim + d];
}

fn v_value(batch : u32, k : u32, h : u32, d : u32) -> f32 {
    let base = batch * params.seq_len * params.d_model * 3u;
    return qkv[base + k * (params.d_model * 3u) + params.d_model * 2u + h * params.head_dim + d];
}

fn score(batch : u32, q : u32, k : u32, h : u32) -> f32 {
    var value = 0.0;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        value = value + q_value(batch, q, h, d) * k_value(batch, k, h, d);
    }
    return value * params.scale;
}

fn probability(batch : u32, q : u32, k : u32, h : u32) -> f32 {
    if (!key_allowed(batch, q, k)) { return 0.0; }
    var maximum = -3.402823e38;
    for (var key = 0u; key < params.seq_len; key = key + 1u) {
        if (!key_allowed(batch, q, key)) { continue; }
        maximum = max(maximum, score(batch, q, key, h));
    }
    var denominator = 0.0;
    for (var key = 0u; key < params.seq_len; key = key + 1u) {
        if (!key_allowed(batch, q, key)) { continue; }
        denominator = denominator + exp(score(batch, q, key, h) - maximum);
    }
    if (denominator == 0.0) { return 0.0; }
    return exp(score(batch, q, k, h) - maximum) / denominator;
}

fn probability_gradient(batch : u32, q : u32, k : u32, h : u32) -> f32 {
    if (!key_allowed(batch, q, k)) { return 0.0; }
    let output_base = batch * params.seq_len * params.d_model;
    var direct = 0.0;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        direct = direct + grad_output[output_base + q * params.d_model + h * params.head_dim + d] *
            v_value(batch, k, h, d);
    }
    direct = direct * dropout_multiplier(batch, h, q, k);
    var expectation = 0.0;
    for (var key = 0u; key < params.seq_len; key = key + 1u) {
        if (!key_allowed(batch, q, key)) { continue; }
        var d_probability = 0.0;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            d_probability = d_probability +
                grad_output[output_base + q * params.d_model + h * params.head_dim + d] *
                v_value(batch, key, h, d);
        }
        d_probability = d_probability * dropout_multiplier(batch, h, q, key);
        expectation = expectation + probability(batch, q, key, h) * d_probability;
    }
    return probability(batch, q, k, h) * (direct - expectation);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let position = gid.x;
    let head = gid.y;
    let batch = gid.z / 3u;
    let component = gid.z % 3u; // 0=Q, 1=K, 2=V
    if (position >= params.seq_len || head >= params.num_heads || batch >= params.batch) { return; }
    let qkv_base = batch * params.seq_len * params.d_model * 3u;
    let output_base = batch * params.seq_len * params.d_model;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        var gradient = 0.0;
        if (component == 0u) {
            for (var key = 0u; key < params.seq_len; key = key + 1u) {
                gradient = gradient + probability_gradient(batch, position, key, head) *
                    k_value(batch, key, head, d) * params.scale;
            }
        } else if (component == 1u) {
            for (var query = 0u; query < params.seq_len; query = query + 1u) {
                gradient = gradient + probability_gradient(batch, query, position, head) *
                    q_value(batch, query, head, d) * params.scale;
            }
        } else {
            for (var query = 0u; query < params.seq_len; query = query + 1u) {
                gradient = gradient + probability(batch, query, position, head) *
                    dropout_multiplier(batch, head, query, position) *
                    grad_output[output_base + query * params.d_model + head * params.head_dim + d];
            }
        }
        let offset = qkv_base + position * (params.d_model * 3u) + component * params.d_model +
            head * params.head_dim + d;
        grad_qkv[offset] = grad_qkv[offset] + gradient;
    }
}
