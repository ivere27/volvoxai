@group(0) @binding(0) var<storage, read> qkv : array<f32>;
            @group(0) @binding(1) var<storage, read> mask : array<i32>;
            @group(0) @binding(2) var<storage, read_write> output : array<f32>;
            
            struct Params {
                seq_len : u32,
                d_model : u32,
                num_heads : u32,
                head_dim : u32,
                batch : u32,
                scale : f32,
                causal : u32,
                mask_mode : u32,
            }
            @group(0) @binding(3) var<uniform> params : Params;

            fn key_allowed(batch_idx : u32, query : u32, key : u32) -> bool {
                if (params.causal != 0u && key > query) { return false; }
                if (params.mask_mode == 0u) { return true; }
                var index = key;
                if (params.mask_mode == 2u) {
                    index = batch_idx * params.seq_len + key;
                } else if (params.mask_mode == 3u) {
                    index = query * params.seq_len + key;
                } else if (params.mask_mode == 4u) {
                    index = (batch_idx * params.seq_len + query) * params.seq_len + key;
                }
                return mask[index] != 0;
            }
            
            @compute @workgroup_size(64, 1, 1)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let q_idx = global_id.x; 
                let h_idx = global_id.y; 
                let batch_idx = global_id.z;
                
                if (q_idx >= params.seq_len || h_idx >= params.num_heads || batch_idx >= params.batch) { return; }
                
                let head_dim = params.head_dim;
                let d_model = params.d_model;
                let qkv_base = batch_idx * params.seq_len * d_model * 3u;
                let output_base = batch_idx * params.seq_len * d_model;
                
                var max_logit : f32 = -1e38;
                
                // Cache Q for this head and this q_idx
                var q_cache : array<f32, 64>; // max head_dim 64
                for (var d = 0u; d < head_dim; d = d + 1u) {
                    q_cache[d] = qkv[qkv_base + q_idx * (d_model * 3u) + (h_idx * head_dim) + d];
                }
                
                // Pass 1: find max
                var valid_keys = 0u;
                for (var k_idx = 0u; k_idx < params.seq_len; k_idx = k_idx + 1u) {
                    if (!key_allowed(batch_idx, q_idx, k_idx)) { continue; }
                    var score : f32 = 0.0;
                    for (var d = 0u; d < head_dim; d = d + 1u) {
                        let k_val = qkv[qkv_base + k_idx * (d_model * 3u) + d_model + (h_idx * head_dim) + d];
                        score = score + (q_cache[d] * k_val);
                    }
                    score = score * params.scale;
                    if (score > max_logit) { max_logit = score; }
                    valid_keys = valid_keys + 1u;
                }

                if (valid_keys == 0u) {
                    for (var d = 0u; d < head_dim; d = d + 1u) {
                        output[output_base + q_idx * d_model + (h_idx * head_dim) + d] = 0.0;
                    }
                    return;
                }
                
                // Pass 2: sum exp
                var sum_exp : f32 = 0.0;
                for (var k_idx = 0u; k_idx < params.seq_len; k_idx = k_idx + 1u) {
                    if (!key_allowed(batch_idx, q_idx, k_idx)) { continue; }
                    var score : f32 = 0.0;
                    for (var d = 0u; d < head_dim; d = d + 1u) {
                        let k_val = qkv[qkv_base + k_idx * (d_model * 3u) + d_model + (h_idx * head_dim) + d];
                        score = score + (q_cache[d] * k_val);
                    }
                    score = score * params.scale;
                    sum_exp = sum_exp + exp(score - max_logit);
                }
                
                // Pass 3: output
                for (var d = 0u; d < head_dim; d = d + 1u) {
                    var out_val : f32 = 0.0;
                    for (var k_idx = 0u; k_idx < params.seq_len; k_idx = k_idx + 1u) {
                        if (!key_allowed(batch_idx, q_idx, k_idx)) { continue; }
                        var score : f32 = 0.0;
                        for (var kd = 0u; kd < head_dim; kd = kd + 1u) {
                            let k_val = qkv[qkv_base + k_idx * (d_model * 3u) + d_model + (h_idx * head_dim) + kd];
                            score = score + (q_cache[kd] * k_val);
                        }
                        score = score * params.scale;
                        
                        let w = exp(score - max_logit) / sum_exp;
                        let v_val = qkv[qkv_base + k_idx * (d_model * 3u) + d_model * 2u + (h_idx * head_dim) + d];
                        out_val = out_val + (w * v_val);
                    }
                    output[output_base + q_idx * d_model + (h_idx * head_dim) + d] = out_val;
                }
            }
