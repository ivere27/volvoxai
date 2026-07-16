@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;
        struct Params { size : u32 }
        @group(0) @binding(2) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let idx = global_id.x;
            if (idx >= params.size) { return; }
            let x = input[idx];
            var out_val = x;
            undefined
            output[idx] = out_val;
        }
