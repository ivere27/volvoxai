@group(0) @binding(0) var<storage, read> a : array<f32>;
        @group(0) @binding(1) var<storage, read> b : array<f32>;
        @group(0) @binding(2) var<storage, read_write> output : array<f32>;
        @group(0) @binding(3) var<storage, read> md : array<u32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
            let idx = gid.x;
            let total = md[0];
            if (idx >= total) { return; }
            let rank = md[1];
            var rem = idx;
            var a_idx = 0u;
            var b_idx = 0u;
            for (var d = 0u; d < rank; d = d + 1u) {
                let os = md[2u + d];
                let coord = rem / os;
                rem = rem - coord * os;
                a_idx = a_idx + coord * md[2u + rank + d];
                b_idx = b_idx + coord * md[2u + 2u * rank + d];
            }
            let av = a[a_idx];
            let bv = b[b_idx];
            var out_val = 0.0;
            //__BINOP__
            output[idx] = out_val;
        }
