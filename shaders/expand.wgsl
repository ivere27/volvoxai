@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;
        struct Params { in_b: u32, in_h: u32, in_w: u32, in_c: u32, out_b: u32, out_h: u32, out_w: u32, out_c: u32 }
        @group(0) @binding(2) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let idx = global_id.x;
            let total = params.out_b * params.out_h * params.out_w * params.out_c;
            if (idx >= total) { return; }
            let oc = idx % params.out_c;
            let ow = (idx / params.out_c) % params.out_w;
            let oh = (idx / (params.out_c * params.out_w)) % params.out_h;
            let ob = idx / (params.out_c * params.out_w * params.out_h);
            let ib = ob % params.in_b;
            let ih = oh % params.in_h; let iw = ow % params.in_w;
            let ic = oc % params.in_c;
            output[idx] = input[((ib * params.in_h + ih) * params.in_w + iw) * params.in_c + ic];
        }
