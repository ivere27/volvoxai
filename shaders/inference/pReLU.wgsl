@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read> weight : array<f32>;
        @group(0) @binding(2) var<storage, read_write> output : array<f32>;
        struct Params { size : u32, c : u32, weight_length : u32, _pad : u32 }
        @group(0) @binding(3) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let idx = global_id.x;
            if (idx >= params.size) { return; }
            let c_idx = idx % params.c;
            let alpha = weight[select(c_idx, 0u, params.weight_length == 1u)];
            let v = input[idx];
            if (v > 0.0) { output[idx] = v; } else { output[idx] = v * alpha; }
        }
