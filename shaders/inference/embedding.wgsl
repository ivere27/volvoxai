@group(0) @binding(0) var<storage, read> tokens : array<i32>;
            @group(0) @binding(1) var<storage, read> weight : array<f32>;
            @group(0) @binding(2) var<storage, read_write> output : array<f32>;
            
            struct Params {
                seq_len : u32,
                d_model : u32,
                vocab_size : u32,
                _pad : u32,
            }
            @group(0) @binding(3) var<uniform> params : Params;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let token_idx = global_id.x;
                if (token_idx >= params.seq_len) { return; }
                
                let d_model = params.d_model;
                let out_offset = token_idx * d_model;
                let signed_token_id = tokens[token_idx];
                if (signed_token_id < 0 || u32(signed_token_id) >= params.vocab_size) {
                    for (var i = 0u; i < d_model; i = i + 1u) {
                        output[out_offset + i] = 0.0;
                    }
                    return;
                }
                let in_offset = u32(signed_token_id) * d_model;
                
                for (var i = 0u; i < d_model; i = i + 1u) {
                    output[out_offset + i] = weight[in_offset + i];
                }
            }
