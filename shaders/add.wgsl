@group(0) @binding(0) var<storage, read> a : array<f32>;
            @group(0) @binding(1) var<storage, read> b : array<f32>;
            @group(0) @binding(2) var<storage, read_write> output : array<f32>;
            
            struct Params { size : u32 }
            @group(0) @binding(3) var<uniform> params : Params;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let idx = global_id.x;
                if (idx >= params.size) { return; }
                output[idx] = a[idx] + b[idx];
            }
