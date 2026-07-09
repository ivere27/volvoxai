@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read> scale : array<f32>;
        @group(0) @binding(2) var<storage, read> zero_point : array<f32>;
        @group(0) @binding(3) var<storage, read_write> output : array<f32>;
        struct Params { size : u32, has_zp : u32 }
        @group(0) @binding(4) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let idx = global_id.x;
            if (idx >= params.size) { return; }
            var zp = 0.0;
            if (params.has_zp == 1u) { zp = zero_point[0]; }
            output[idx] = (input[idx] - zp) * scale[0];
        }
