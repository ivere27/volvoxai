@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;
        struct Params { b: u32, in_h: u32, in_w: u32, c: u32, out_h: u32, out_w: u32, pt: u32, pl: u32, val: f32 }
        @group(0) @binding(2) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let idx = global_id.x;
            let total = params.b * params.out_h * params.out_w * params.c;
            if (idx >= total) { return; }
            let c = idx % params.c;
            let x = (idx / params.c) % params.out_w;
            let y = (idx / (params.c * params.out_w)) % params.out_h;
            let b = idx / (params.c * params.out_w * params.out_h);
            if (y >= params.pt && y < params.pt + params.in_h && x >= params.pl && x < params.pl + params.in_w) {
                output[idx] = input[((b * params.in_h + (y - params.pt)) * params.in_w + (x - params.pl)) * params.c + c];
            } else {
                output[idx] = params.val;
            }
        }
