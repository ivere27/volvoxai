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
                for (var y = 0u; y < params.in_h; y = y + 1u) {
                    let val = input[(y * params.in_w + x) * params.in_c + c];
                    if (val > max_val) { max_val = val; }
                }
                
                var denom = 0.0;
                var weighted = 0.0;
                for (var y = 0u; y < params.in_h; y = y + 1u) {
                    let val = input[(y * params.in_w + x) * params.in_c + c];
                    let ev = exp(val - max_val);
                    denom = denom + ev;
                    weighted = weighted + ev * (f32(y) + 0.5) / f32(params.in_h);
                }
                
                var out_val = 0.0;
                if (denom > 0.0) { out_val = weighted / denom; }
                output[c * params.in_w + x] = out_val;
            }
