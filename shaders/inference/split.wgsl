@group(0) @binding(0) var<storage, read> input : array<u32>;
        @group(0) @binding(1) var<storage, read_write> output : array<u32>;
        struct Params { total : u32, inner : u32, split_size : u32, axis_in : u32, offset : u32 }
        @group(0) @binding(2) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
            let i = gid.x;
            if (i >= params.total) { return; }
            let inner_idx = i % params.inner;
            let s = (i / params.inner) % params.split_size;
            let outer_idx = i / (params.split_size * params.inner);
            let in_idx = outer_idx * (params.axis_in * params.inner)
                       + (params.offset + s) * params.inner + inner_idx;
            output[i] = input[in_idx];
        }
