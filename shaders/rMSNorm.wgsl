@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read> weight : array<f32>;
        @group(0) @binding(2) var<storage, read_write> output : array<f32>;
        struct Params { seq_len : u32, d_model : u32, eps : f32 }
        @group(0) @binding(3) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let s = global_id.x;
            if (s >= params.seq_len) { return; }
            var var_val = 0.0;
            let offset = s * params.d_model;
            for (var d = 0u; d < params.d_model; d = d + 1u) {
                let v = input[offset + d];
                var_val = var_val + v * v;
            }
            var_val = var_val / f32(params.d_model);
            let inv_std = 1.0 / sqrt(var_val + params.eps);
            for (var d = 0u; d < params.d_model; d = d + 1u) {
                output[offset + d] = input[offset + d] * inv_std * weight[d];
            }
        }
