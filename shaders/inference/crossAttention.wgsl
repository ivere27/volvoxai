@group(0) @binding(0) var<storage, read> q_in : array<f32>;
            @group(0) @binding(1) var<storage, read> kv_in : array<f32>;
            @group(0) @binding(2) var<storage, read> weight_int8_packed : array<u32>;
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
                batch : u32,
                _pad0 : u32,
                _pad1 : u32,
                _pad2 : u32,
            }
            @group(0) @binding(6) var<uniform> params : Params;
            
            @compute @workgroup_size(64, 1, 1)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let q_idx = global_id.x; 
                let h_idx = global_id.y; 
                let b_idx = global_id.z;
                
                if (q_idx >= params.seq_len_q || h_idx >= params.num_heads || b_idx >= params.batch) { return; }
                
                let head_dim = params.head_dim;
                let d_model = params.d_model;
                let d_model_4 = d_model / 4u;
                let q_base = b_idx * params.seq_len_q * d_model;
                let kv_base = b_idx * params.seq_len_kv * d_model;
                
                // Cache Q for this head and this q_idx
                var q_cache : array<f32, 64>; // max head_dim 64
                for (var d = 0u; d < head_dim; d = d + 1u) {
                    var sum = 0.0;
                    let out_col = h_idx * head_dim + d;
                    for (var i = 0u; i < d_model_4; i = i + 1u) {
                        let w_packed = weight_int8_packed[out_col * d_model_4 + i];
                        let in_base = q_base + q_idx * d_model + i * 4u;
                        sum = sum + q_in[in_base + 0u] * f32(extractBits(i32(w_packed), 0u, 8u));
                        sum = sum + q_in[in_base + 1u] * f32(extractBits(i32(w_packed), 8u, 8u));
                        sum = sum + q_in[in_base + 2u] * f32(extractBits(i32(w_packed), 16u, 8u));
                        sum = sum + q_in[in_base + 3u] * f32(extractBits(i32(w_packed), 24u, 8u));
                    }
                    if (params.has_scale > 0u) { sum = sum * w_scale[out_col]; }
                    if (params.has_bias > 0u) { sum = sum + bias[out_col]; }
                    q_cache[d] = sum;
                }
                
                var max_logit : f32 = -1e38;
                
                // Pass 1: find max
                for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
                    var score : f32 = 0.0;
                    for (var d = 0u; d < head_dim; d = d + 1u) {
                        var k_val = 0.0;
                        let out_col = d_model + h_idx * head_dim + d;
                        for (var i = 0u; i < d_model_4; i = i + 1u) {
                            let w_packed = weight_int8_packed[out_col * d_model_4 + i];
                            let in_base = kv_base + k_idx * d_model + i * 4u;
                            k_val = k_val + kv_in[in_base + 0u] * f32(extractBits(i32(w_packed), 0u, 8u));
                            k_val = k_val + kv_in[in_base + 1u] * f32(extractBits(i32(w_packed), 8u, 8u));
                            k_val = k_val + kv_in[in_base + 2u] * f32(extractBits(i32(w_packed), 16u, 8u));
                            k_val = k_val + kv_in[in_base + 3u] * f32(extractBits(i32(w_packed), 24u, 8u));
                        }
                        if (params.has_scale > 0u) { k_val = k_val * w_scale[out_col]; }
                        if (params.has_bias > 0u) { k_val = k_val + bias[out_col]; }
                        
                        score = score + (q_cache[d] * k_val);
                    }
                    score = score * params.scale_factor;
                    if (score > max_logit) { max_logit = score; }
                }
                
                // Pass 2: sum exp
                var sum_exp : f32 = 0.0;
                for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
                    var score : f32 = 0.0;
                    for (var d = 0u; d < head_dim; d = d + 1u) {
                        var k_val = 0.0;
                        let out_col = d_model + h_idx * head_dim + d;
                        for (var i = 0u; i < d_model_4; i = i + 1u) {
                            let w_packed = weight_int8_packed[out_col * d_model_4 + i];
                            let in_base = kv_base + k_idx * d_model + i * 4u;
                            k_val = k_val + kv_in[in_base + 0u] * f32(extractBits(i32(w_packed), 0u, 8u));
                            k_val = k_val + kv_in[in_base + 1u] * f32(extractBits(i32(w_packed), 8u, 8u));
                            k_val = k_val + kv_in[in_base + 2u] * f32(extractBits(i32(w_packed), 16u, 8u));
                            k_val = k_val + kv_in[in_base + 3u] * f32(extractBits(i32(w_packed), 24u, 8u));
                        }
                        if (params.has_scale > 0u) { k_val = k_val * w_scale[out_col]; }
                        if (params.has_bias > 0u) { k_val = k_val + bias[out_col]; }
                        
                        score = score + (q_cache[d] * k_val);
                    }
                    score = score * params.scale_factor;
                    sum_exp = sum_exp + exp(score - max_logit);
                }
                
                // Pass 3: output
                for (var d = 0u; d < head_dim; d = d + 1u) {
                    var out_val : f32 = 0.0;
                    for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
                        var score : f32 = 0.0;
                        for (var kd = 0u; kd < head_dim; kd = kd + 1u) {
                            var k_val = 0.0;
                            let out_col = d_model + h_idx * head_dim + kd;
                            for (var i = 0u; i < d_model_4; i = i + 1u) {
                                let w_packed = weight_int8_packed[out_col * d_model_4 + i];
                                let in_base = kv_base + k_idx * d_model + i * 4u;
                                k_val = k_val + kv_in[in_base + 0u] * f32(extractBits(i32(w_packed), 0u, 8u));
                                k_val = k_val + kv_in[in_base + 1u] * f32(extractBits(i32(w_packed), 8u, 8u));
                                k_val = k_val + kv_in[in_base + 2u] * f32(extractBits(i32(w_packed), 16u, 8u));
                                k_val = k_val + kv_in[in_base + 3u] * f32(extractBits(i32(w_packed), 24u, 8u));
                            }
                            if (params.has_scale > 0u) { k_val = k_val * w_scale[out_col]; }
                            if (params.has_bias > 0u) { k_val = k_val + bias[out_col]; }
                            
                            score = score + (q_cache[kd] * k_val);
                        }
                        score = score * params.scale_factor;
                        
                        let w = exp(score - max_logit) / sum_exp;
                        
                        var v_val = 0.0;
                        let v_col = d_model * 2u + h_idx * head_dim + d;
                        for (var i = 0u; i < d_model_4; i = i + 1u) {
                            let w_packed = weight_int8_packed[v_col * d_model_4 + i];
                            let in_base = kv_base + k_idx * d_model + i * 4u;
                            v_val = v_val + kv_in[in_base + 0u] * f32(extractBits(i32(w_packed), 0u, 8u));
                            v_val = v_val + kv_in[in_base + 1u] * f32(extractBits(i32(w_packed), 8u, 8u));
                            v_val = v_val + kv_in[in_base + 2u] * f32(extractBits(i32(w_packed), 16u, 8u));
                            v_val = v_val + kv_in[in_base + 3u] * f32(extractBits(i32(w_packed), 24u, 8u));
                        }
                        if (params.has_scale > 0u) { v_val = v_val * w_scale[v_col]; }
                        if (params.has_bias > 0u) { v_val = v_val + bias[v_col]; }
                        
                        out_val = out_val + (w * v_val);
                    }
                    output[q_base + q_idx * d_model + (h_idx * head_dim) + d] = out_val;
                }
            }
