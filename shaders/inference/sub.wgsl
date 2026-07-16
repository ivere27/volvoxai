@group(0) @binding(0) var<storage, read> a : array<f32>;
        @group(0) @binding(1) var<storage, read> b : array<f32>;
        @group(0) @binding(2) var<storage, read_write> output : array<f32>;
        
        struct Params { size : u32, is_b_scalar : u32, b_size : u32, a_size: u32, is_a_scalar: u32 }
        @group(0) @binding(3) var<uniform> params : Params;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let idx = global_id.x;
            if (idx >= params.size) { return; }
            
            var a_val : f32 = 0.0;
            if (params.is_a_scalar == 1u) {
                a_val = a[0];
            } else if (params.a_size < params.size && params.a_size > 0u) {
                a_val = a[idx % params.a_size];
            } else {
                a_val = a[idx];
            }
            
            var b_val : f32 = 0.0;
            if (params.is_b_scalar == 1u) {
                b_val = b[0];
            } else if (params.b_size < params.size && params.b_size > 0u) {
                b_val = b[idx % params.b_size];
            } else {
                b_val = b[idx];
            }
            
            var out_val : f32 = 0.0;
            out_val = a_val - b_val;
            output[idx] = out_val;
        }
