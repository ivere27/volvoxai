@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read_write> output : array<f32>;
            
            struct Params { in_h : u32, in_w : u32, in_c : u32 }
            @group(0) @binding(2) var<uniform> params : Params;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let x = global_id.x;
                let c = global_id.y;
                
                if (x >= params.in_w || c >= params.in_c) { return; }
                
                var max_val = -1e38;
                var sum_val = 0.0;
                
                for (var y = 0u; y < params.in_h; y = y + 1u) {
                    let val = input[(y * params.in_w + x) * params.in_c + c];
                    if (val > max_val) { max_val = val; }
                    sum_val = sum_val + val;
                }
                
                let out_max_idx = c * params.in_w + x;
                let out_mean_idx = (c + params.in_c) * params.in_w + x;
                
                output[out_max_idx] = max_val;
                output[out_mean_idx] = sum_val / f32(params.in_h);
            }
