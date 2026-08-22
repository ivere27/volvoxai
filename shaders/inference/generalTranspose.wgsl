@group(0) @binding(0) var<storage, read> input : array<u32>;
        @group(0) @binding(1) var<storage, read_write> output : array<u32>;
        @group(0) @binding(2) var<storage, read> md : array<u32>;
        @compute @workgroup_size(64)
        fn main(
            @builtin(global_invocation_id) gid : vec3<u32>,
            @builtin(num_workgroups) grid : vec3<u32>,
        ) {
            let total = md[0];
            let rank = md[1];
            let idx = gid.x + gid.y * (grid.x * 64u);
            if (idx >= total) { return; }
            var rem = idx;
            var in_idx = 0u;
            for (var d = 0u; d < rank; d = d + 1u) {
                let os = md[2u + d];
                let coord = rem / os;
                rem = rem - coord * os;
                in_idx = in_idx + coord * md[2u + rank + d];
            }
            output[idx] = input[in_idx];
        }
