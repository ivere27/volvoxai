// CrossAttention has one scalar-reference entry point per differentiated
// tensor. Each entry owns its output element, so repeated attention uses can
// be reduced without floating-point atomics.
@group(0) @binding(0) var<storage, read> q_input : array<f32>;
@group(0) @binding(1) var<storage, read> kv_input : array<f32>;
@group(0) @binding(2) var<storage, read> projection_weight : array<f32>;
@group(0) @binding(3) var<storage, read> projection_scale : array<f32>;
@group(0) @binding(4) var<storage, read> projection_bias : array<f32>;
@group(0) @binding(5) var<storage, read> grad_output : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_q : array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_kv : array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_weight : array<f32>;
@group(0) @binding(9) var<storage, read_write> grad_scale : array<f32>;
@group(0) @binding(10) var<storage, read_write> grad_bias : array<f32>;

struct Params {
    seq_q : u32,
    seq_kv : u32,
    d_model : u32,
    num_heads : u32,
    head_dim : u32,
    batch : u32,
    q_elements : u32,
    kv_elements : u32,
    weight_elements : u32,
    affine_elements : u32,
    has_scale : u32,
    has_bias : u32,
    attention_scale : f32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
}
@group(0) @binding(11) var<uniform> params : Params;

fn q_offset(batch : u32, query : u32, dimension : u32) -> u32 {
    return (batch * params.seq_q + query) * params.d_model + dimension;
}

fn kv_offset(batch : u32, key : u32, dimension : u32) -> u32 {
    return (batch * params.seq_kv + key) * params.d_model + dimension;
}

fn affine_scale(column : u32) -> f32 {
    if (params.has_scale != 0u) {
        return projection_scale[column];
    }
    return 1.0;
}

fn affine_bias(column : u32) -> f32 {
    if (params.has_bias != 0u) {
        return projection_bias[column];
    }
    return 0.0;
}

fn q_raw_projection(batch : u32, query : u32, column : u32) -> f32 {
    var value = 0.0;
    let weight_offset = column * params.d_model;
    for (var input_dimension = 0u; input_dimension < params.d_model;
         input_dimension = input_dimension + 1u) {
        value = value + q_input[q_offset(batch, query, input_dimension)] *
            projection_weight[weight_offset + input_dimension];
    }
    return value;
}

fn kv_raw_projection(batch : u32, key : u32, column : u32) -> f32 {
    var value = 0.0;
    let weight_offset = column * params.d_model;
    for (var input_dimension = 0u; input_dimension < params.d_model;
         input_dimension = input_dimension + 1u) {
        value = value + kv_input[kv_offset(batch, key, input_dimension)] *
            projection_weight[weight_offset + input_dimension];
    }
    return value;
}

fn q_projection(batch : u32, query : u32, head : u32, dimension : u32) -> f32 {
    let column = head * params.head_dim + dimension;
    return q_raw_projection(batch, query, column) * affine_scale(column) + affine_bias(column);
}

fn k_projection(batch : u32, key : u32, head : u32, dimension : u32) -> f32 {
    let column = params.d_model + head * params.head_dim + dimension;
    return kv_raw_projection(batch, key, column) * affine_scale(column) + affine_bias(column);
}

fn v_projection(batch : u32, key : u32, head : u32, dimension : u32) -> f32 {
    let column = 2u * params.d_model + head * params.head_dim + dimension;
    return kv_raw_projection(batch, key, column) * affine_scale(column) + affine_bias(column);
}

fn attention_score(batch : u32, query : u32, key : u32, head : u32) -> f32 {
    var value = 0.0;
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        value = value + q_projection(batch, query, head, dimension) *
            k_projection(batch, key, head, dimension);
    }
    return value * params.attention_scale;
}

fn score_maximum(batch : u32, query : u32, head : u32) -> f32 {
    var maximum = -3.402823e38;
    for (var key = 0u; key < params.seq_kv; key = key + 1u) {
        maximum = max(maximum, attention_score(batch, query, key, head));
    }
    return maximum;
}

fn score_denominator(batch : u32, query : u32, head : u32, maximum : f32) -> f32 {
    var denominator = 0.0;
    for (var key = 0u; key < params.seq_kv; key = key + 1u) {
        denominator = denominator + exp(attention_score(batch, query, key, head) - maximum);
    }
    return denominator;
}

fn attention_probability(batch : u32, query : u32, key : u32, head : u32,
                         maximum : f32, denominator : f32) -> f32 {
    return exp(attention_score(batch, query, key, head) - maximum) / denominator;
}

fn output_gradient(batch : u32, query : u32, head : u32, dimension : u32) -> f32 {
    return grad_output[q_offset(batch, query, head * params.head_dim + dimension)];
}

fn probability_gradient(batch : u32, query : u32, key : u32, head : u32) -> f32 {
    var value = 0.0;
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        value = value + output_gradient(batch, query, head, dimension) *
            v_projection(batch, key, head, dimension);
    }
    return value;
}

fn probability_gradient_dot(batch : u32, query : u32, head : u32,
                            maximum : f32, denominator : f32) -> f32 {
    var value = 0.0;
    for (var key = 0u; key < params.seq_kv; key = key + 1u) {
        value = value + attention_probability(batch, query, key, head, maximum, denominator) *
            probability_gradient(batch, query, key, head);
    }
    return value;
}

fn score_gradient(batch : u32, query : u32, key : u32, head : u32,
                  maximum : f32, denominator : f32, probability_dot : f32) -> f32 {
    let probability = attention_probability(batch, query, key, head, maximum, denominator);
    return probability * (probability_gradient(batch, query, key, head) - probability_dot);
}

fn q_projection_gradient(batch : u32, query : u32, head : u32, dimension : u32) -> f32 {
    let maximum = score_maximum(batch, query, head);
    let denominator = score_denominator(batch, query, head, maximum);
    let probability_dot = probability_gradient_dot(batch, query, head, maximum, denominator);
    var value = 0.0;
    for (var key = 0u; key < params.seq_kv; key = key + 1u) {
        value = value + score_gradient(batch, query, key, head, maximum, denominator, probability_dot) *
            params.attention_scale * k_projection(batch, key, head, dimension);
    }
    return value;
}

fn k_projection_gradient(batch : u32, key : u32, head : u32, dimension : u32) -> f32 {
    var value = 0.0;
    for (var query = 0u; query < params.seq_q; query = query + 1u) {
        let maximum = score_maximum(batch, query, head);
        let denominator = score_denominator(batch, query, head, maximum);
        let probability_dot = probability_gradient_dot(batch, query, head, maximum, denominator);
        value = value + score_gradient(batch, query, key, head, maximum, denominator, probability_dot) *
            params.attention_scale * q_projection(batch, query, head, dimension);
    }
    return value;
}

fn v_projection_gradient(batch : u32, key : u32, head : u32, dimension : u32) -> f32 {
    var value = 0.0;
    for (var query = 0u; query < params.seq_q; query = query + 1u) {
        let maximum = score_maximum(batch, query, head);
        let denominator = score_denominator(batch, query, head, maximum);
        value = value + attention_probability(batch, query, key, head, maximum, denominator) *
            output_gradient(batch, query, head, dimension);
    }
    return value;
}

@compute @workgroup_size(64, 1, 1)
fn q_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.q_elements) { return; }
    let input_dimension = flat % params.d_model;
    let row = flat / params.d_model;
    let query = row % params.seq_q;
    let batch = row / params.seq_q;
    var value = 0.0;
    for (var head = 0u; head < params.num_heads; head = head + 1u) {
        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
            let column = head * params.head_dim + dimension;
            value = value + q_projection_gradient(batch, query, head, dimension) *
                affine_scale(column) * projection_weight[column * params.d_model + input_dimension];
        }
    }
    grad_q[flat] = grad_q[flat] + value;
}

@compute @workgroup_size(64, 1, 1)
fn kv_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.kv_elements) { return; }
    let input_dimension = flat % params.d_model;
    let row = flat / params.d_model;
    let key = row % params.seq_kv;
    let batch = row / params.seq_kv;
    var value = 0.0;
    for (var head = 0u; head < params.num_heads; head = head + 1u) {
        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
            let local_column = head * params.head_dim + dimension;
            let k_column = params.d_model + local_column;
            let v_column = 2u * params.d_model + local_column;
            value = value + k_projection_gradient(batch, key, head, dimension) *
                affine_scale(k_column) * projection_weight[k_column * params.d_model + input_dimension];
            value = value + v_projection_gradient(batch, key, head, dimension) *
                affine_scale(v_column) * projection_weight[v_column * params.d_model + input_dimension];
        }
    }
    grad_kv[flat] = grad_kv[flat] + value;
}

@compute @workgroup_size(64, 1, 1)
fn weight_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.weight_elements) { return; }
    let column = flat / params.d_model;
    let input_dimension = flat % params.d_model;
    var value = 0.0;
    if (column < params.d_model) {
        let head = column / params.head_dim;
        let dimension = column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var query = 0u; query < params.seq_q; query = query + 1u) {
                value = value + q_projection_gradient(batch, query, head, dimension) *
                    affine_scale(column) * q_input[q_offset(batch, query, input_dimension)];
            }
        }
    } else if (column < 2u * params.d_model) {
        let local_column = column - params.d_model;
        let head = local_column / params.head_dim;
        let dimension = local_column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var key = 0u; key < params.seq_kv; key = key + 1u) {
                value = value + k_projection_gradient(batch, key, head, dimension) *
                    affine_scale(column) * kv_input[kv_offset(batch, key, input_dimension)];
            }
        }
    } else {
        let local_column = column - 2u * params.d_model;
        let head = local_column / params.head_dim;
        let dimension = local_column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var key = 0u; key < params.seq_kv; key = key + 1u) {
                value = value + v_projection_gradient(batch, key, head, dimension) *
                    affine_scale(column) * kv_input[kv_offset(batch, key, input_dimension)];
            }
        }
    }
    grad_weight[flat] = grad_weight[flat] + value;
}

@compute @workgroup_size(64, 1, 1)
fn scale_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let column = gid.x;
    if (column >= params.affine_elements || params.has_scale == 0u) { return; }
    var value = 0.0;
    if (column < params.d_model) {
        let head = column / params.head_dim;
        let dimension = column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var query = 0u; query < params.seq_q; query = query + 1u) {
                value = value + q_projection_gradient(batch, query, head, dimension) *
                    q_raw_projection(batch, query, column);
            }
        }
    } else if (column < 2u * params.d_model) {
        let local_column = column - params.d_model;
        let head = local_column / params.head_dim;
        let dimension = local_column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var key = 0u; key < params.seq_kv; key = key + 1u) {
                value = value + k_projection_gradient(batch, key, head, dimension) *
                    kv_raw_projection(batch, key, column);
            }
        }
    } else {
        let local_column = column - 2u * params.d_model;
        let head = local_column / params.head_dim;
        let dimension = local_column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var key = 0u; key < params.seq_kv; key = key + 1u) {
                value = value + v_projection_gradient(batch, key, head, dimension) *
                    kv_raw_projection(batch, key, column);
            }
        }
    }
    grad_scale[column] = grad_scale[column] + value;
}

@compute @workgroup_size(64, 1, 1)
fn bias_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let column = gid.x;
    if (column >= params.affine_elements || params.has_bias == 0u) { return; }
    var value = 0.0;
    if (column < params.d_model) {
        let head = column / params.head_dim;
        let dimension = column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var query = 0u; query < params.seq_q; query = query + 1u) {
                value = value + q_projection_gradient(batch, query, head, dimension);
            }
        }
    } else if (column < 2u * params.d_model) {
        let local_column = column - params.d_model;
        let head = local_column / params.head_dim;
        let dimension = local_column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var key = 0u; key < params.seq_kv; key = key + 1u) {
                value = value + k_projection_gradient(batch, key, head, dimension);
            }
        }
    } else {
        let local_column = column - 2u * params.d_model;
        let head = local_column / params.head_dim;
        let dimension = local_column % params.head_dim;
        for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
            for (var key = 0u; key < params.seq_kv; key = key + 1u) {
                value = value + v_projection_gradient(batch, key, head, dimension);
            }
        }
    }
    grad_bias[column] = grad_bias[column] + value;
}
