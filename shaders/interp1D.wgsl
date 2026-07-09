@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read_write> output : array<f32>;
            
            struct Params { in_c : u32, in_l : u32, out_l : u32 }
            @group(0) @binding(2) var<uniform> params : Params;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let i = global_id.x;
                let c = global_id.y;
                
                if (i >= params.out_l || c >= params.in_c) { return; }
                
                let scale = f32(params.in_l) / f32(params.out_l);
                var src = (f32(i) + 0.5) * scale - 0.5;
                if (src < 0.0) { src = 0.0; }
                if (src > f32(params.in_l - 1u)) { src = f32(params.in_l - 1u); }
                
                let lo = u32(src);
                var hi = lo + 1u;
                if (hi >= params.in_l) { hi = params.in_l - 1u; }
                let t = src - f32(lo);
                
                let in_base = c * params.in_l;
                let out_base = c * params.out_l;
                
                output[out_base + i] = input[in_base + lo] * (1.0 - t) + input[in_base + hi] * t;
            }
