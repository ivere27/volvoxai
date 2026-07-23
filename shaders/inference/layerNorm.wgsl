@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read> weight : array<f32>;
            @group(0) @binding(2) var<storage, read> bias : array<f32>;
            @group(0) @binding(3) var<storage, read_write> output : array<f32>;
            
            struct Params { rows : u32, d_model : u32, eps : f32, _pad : u32 }
            @group(0) @binding(4) var<uniform> params : Params;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let row = global_id.x;
                if (row >= params.rows) { return; }
                let d_model = params.d_model;
                let offset = row * d_model;
                
                var sum : f32 = 0.0;
                
                for (var i = 0u; i < d_model; i = i + 1u) {
                    sum = sum + input[offset + i];
                }
                
                let mean = sum / f32(d_model);
                var variance_sum : f32 = 0.0;
                for (var i = 0u; i < d_model; i = i + 1u) {
                    let centered = input[offset + i] - mean;
                    variance_sum = variance_sum + (centered * centered);
                }
                let variance = variance_sum / f32(d_model);
                let inv_std = inverseSqrt(variance + params.eps);
                
                for (var i = 0u; i < d_model; i = i + 1u) {
                    let norm_val = (input[offset + i] - mean) * inv_std;
                    output[offset + i] = norm_val * weight[i] + bias[i];
                }
            }
