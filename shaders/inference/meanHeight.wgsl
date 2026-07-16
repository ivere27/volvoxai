@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read_write> output : array<f32>;

            struct Params { in_h : u32, in_w : u32, in_c : u32, batch : u32 }
            @group(0) @binding(2) var<uniform> params : Params;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let x = global_id.x;
                let c = global_id.y;
                let b = global_id.z;
                if (x >= params.in_w || c >= params.in_c || b >= params.batch) { return; }

                let input_base = b * params.in_h * params.in_w * params.in_c;
                let output_base = b * params.in_c * params.in_w;

                var sum_val = 0.0;
                for (var y = 0u; y < params.in_h; y = y + 1u) {
                    sum_val = sum_val + input[input_base + (y * params.in_w + x) * params.in_c + c];
                }
                output[output_base + c * params.in_w + x] = sum_val / f32(params.in_h);
            }
