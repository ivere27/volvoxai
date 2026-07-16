@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read> weight : array<f32>;
        @group(0) @binding(2) var<storage, read> bias : array<f32>;
        @group(0) @binding(3) var<storage, read> running_mean : array<f32>;
        @group(0) @binding(4) var<storage, read> running_var : array<f32>;
        @group(0) @binding(5) var<storage, read_write> output : array<f32>;
        struct Params { b : u32, c : u32, h : u32, w : u32, eps : f32 }
        @group(0) @binding(6) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let idx = global_id.x;
            let total = params.b * params.c * params.h * params.w;
            if (idx >= total) { return; }
            let c = idx % params.c;
            let mean = running_mean[c];
            let var_val = running_var[c];
            let gamma = weight[c];
            let beta = bias[c];
            let inv_std = 1.0 / sqrt(var_val + params.eps);
            output[idx] = (input[idx] - mean) * inv_std * gamma + beta;
        }
