@group(0) @binding(0) var<storage, read> q_in : array<f32>;
@group(0) @binding(1) var<storage, read> kv_in : array<f32>;
@group(0) @binding(2) var<storage, read> weight : array<f32>;
@group(0) @binding(3) var<storage, read> w_scale : array<f32>;
@group(0) @binding(4) var<storage, read> bias : array<f32>;
@group(0) @binding(5) var<storage, read_write> output : array<f32>;

struct Params {
    seq_len_q : u32,
    seq_len_kv : u32,
    d_model : u32,
    num_heads : u32,
    head_dim : u32,
    scale_factor : f32,
    has_scale : u32,
    has_bias : u32,
}
@group(0) @binding(6) var<uniform> params : Params;

fn project(src : ptr<function, array<f32, 64>>, row : u32, out_col : u32) -> f32 {
    var sum = 0.0;
    let w_base = out_col * params.d_model;
    for (var i = 0u; i < params.d_model; i = i + 1u) {
        sum = sum + (*src)[i] * weight[w_base + i];
    }
    if (params.has_scale > 0u) { sum = sum * w_scale[out_col]; }
    if (params.has_bias > 0u) { sum = sum + bias[out_col]; }
    return sum;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let q_idx = gid.x;
    let h_idx = gid.y;
    if (q_idx >= params.seq_len_q || h_idx >= params.num_heads) { return; }
    if (params.d_model > 64u || params.head_dim > 64u) { return; }

    var q_src : array<f32, 64>;
    for (var i = 0u; i < params.d_model; i = i + 1u) {
        q_src[i] = q_in[q_idx * params.d_model + i];
    }

    var q_proj : array<f32, 64>;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        let out_col = h_idx * params.head_dim + d;
        q_proj[d] = project(&q_src, q_idx, out_col);
    }

    var max_logit = -1e38;
    for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
        var kv_src : array<f32, 64>;
        for (var i = 0u; i < params.d_model; i = i + 1u) {
            kv_src[i] = kv_in[k_idx * params.d_model + i];
        }
        var score = 0.0;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            let out_col = params.d_model + h_idx * params.head_dim + d;
            score = score + q_proj[d] * project(&kv_src, k_idx, out_col);
        }
        score = score * params.scale_factor;
        if (score > max_logit) { max_logit = score; }
    }

    var sum_exp = 0.0;
    for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
        var kv_src : array<f32, 64>;
        for (var i = 0u; i < params.d_model; i = i + 1u) {
            kv_src[i] = kv_in[k_idx * params.d_model + i];
        }
        var score = 0.0;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            let out_col = params.d_model + h_idx * params.head_dim + d;
            score = score + q_proj[d] * project(&kv_src, k_idx, out_col);
        }
        score = score * params.scale_factor;
        sum_exp = sum_exp + exp(score - max_logit);
    }

    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        var out_val = 0.0;
        for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
            var kv_src : array<f32, 64>;
            for (var i = 0u; i < params.d_model; i = i + 1u) {
                kv_src[i] = kv_in[k_idx * params.d_model + i];
            }
            var score = 0.0;
            for (var kd = 0u; kd < params.head_dim; kd = kd + 1u) {
                let k_col = params.d_model + h_idx * params.head_dim + kd;
                score = score + q_proj[kd] * project(&kv_src, k_idx, k_col);
            }
            score = score * params.scale_factor;
            let attn = exp(score - max_logit) / sum_exp;
            let v_col = params.d_model * 2u + h_idx * params.head_dim + d;
            out_val = out_val + attn * project(&kv_src, k_idx, v_col);
        }
        output[q_idx * params.d_model + h_idx * params.head_dim + d] = out_val;
    }
}
