@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;
        struct Params { b : u32, d : u32 }
        @group(0) @binding(2) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let i = global_id.x;
            if (i >= params.b) { return; }
            let offset = i * params.d;
            var max_val = -100000.0;
            for (var j = 0u; j < params.d; j = j + 1u) {
                if (input[offset + j] > max_val) { max_val = input[offset + j]; }
            }
            var sum = 0.0;
            for (var j = 0u; j < params.d; j = j + 1u) {
                let e = exp(input[offset + j] - max_val);
                output[offset + j] = e;
                sum = sum + e;
            }
            for (var j = 0u; j < params.d; j = j + 1u) {
                output[offset + j] = output[offset + j] / sum;
            }
        }
