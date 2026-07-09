@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read> weight : array<f32>;
        @group(0) @binding(2) var<storage, read> bias : array<f32>;
        @group(0) @binding(3) var<storage, read_write> output : array<f32>;
        struct Params { b: u32, in_h: u32, in_w: u32, in_c: u32, out_h: u32, out_w: u32, out_c: u32, kh: u32, kw: u32, sh: u32, sw: u32, ph: u32, pw: u32, has_bias: u32 }
        @group(0) @binding(4) var<uniform> params : Params;
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let x = global_id.x; let y = global_id.y; let batch_c = global_id.z;
            if (x >= params.out_w || y >= params.out_h || batch_c >= (params.b * params.out_c)) { return; }
            let ob = batch_c / params.out_c;
            let oc = batch_c % params.out_c;
            var sum = 0.0;
            if (params.has_bias == 1u) { sum = bias[oc]; }
            for (var ic = 0u; ic < params.in_c; ic = ic + 1u) {
                for (var ky = 0u; ky < params.kh; ky = ky + 1u) {
                    for (var kx = 0u; kx < params.kw; kx = kx + 1u) {
                        let oy_shifted = i32(y) + i32(params.ph) - i32(ky);
                        let ox_shifted = i32(x) + i32(params.pw) - i32(kx);
                        if (oy_shifted % i32(params.sh) == 0 && ox_shifted % i32(params.sw) == 0) {
                            let iy = oy_shifted / i32(params.sh);
                            let ix = ox_shifted / i32(params.sw);
                            if (iy >= 0 && iy < i32(params.in_h) && ix >= 0 && ix < i32(params.in_w)) {
                                let in_val = input[((ob * params.in_h + u32(iy)) * params.in_w + u32(ix)) * params.in_c + ic];
                                let w_val = weight[ic * (params.out_c * params.kh * params.kw) + oc * (params.kh * params.kw) + ky * params.kw + kx];
                                sum = sum + in_val * w_val;
                            }
                        }
                    }
                }
            }
            output[((ob * params.out_h + y) * params.out_w + x) * params.out_c + oc] = sum;
        }
