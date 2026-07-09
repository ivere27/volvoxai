@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;
        struct Params { b: u32, in_h: u32, in_w: u32, c: u32, out_h: u32, out_w: u32, kh: u32, kw: u32, sh: u32, sw: u32, ph: u32, pw: u32 }
        @group(0) @binding(2) var<uniform> params : Params;
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let x = global_id.x; let y = global_id.y; let z = global_id.z;
            let b = z / params.c;
            let c = z - b * params.c;
            if (x >= params.out_w || y >= params.out_h || b >= params.b) { return; }
            var sum = 0.0; var count = 0u;
            for (var ky = 0u; ky < params.kh; ky = ky + 1u) {
                for (var kx = 0u; kx < params.kw; kx = kx + 1u) {
                    let in_y = i32(y * params.sh) - i32(params.ph) + i32(ky);
                    let in_x = i32(x * params.sw) - i32(params.pw) + i32(kx);
                    if (in_y >= 0 && in_y < i32(params.in_h) && in_x >= 0 && in_x < i32(params.in_w)) {
                        sum = sum + input[((b * params.in_h + u32(in_y)) * params.in_w + u32(in_x)) * params.c + c];
                        count = count + 1u;
                    }
                }
            }
            if (count == 0u) { count = 1u; }
            output[((b * params.out_h + y) * params.out_w + x) * params.c + c] = sum / f32(count);
        }
