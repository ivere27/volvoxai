@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read_write> output : array<f32>;
            
            struct Params { in_h : u32, in_w : u32, in_c : u32, batch : u32 }
            @group(0) @binding(2) var<uniform> params : Params;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let y = global_id.x;
                let c = global_id.y;
                let b = global_id.z;
                
                if (y >= params.in_h || c >= params.in_c || b >= params.batch) { return; }

                let input_base = b * params.in_h * params.in_w * params.in_c;
                let output_base = b * 2u * params.in_c * params.in_h;
                
                // Seed from a real element rather than a finite sentinel so
                // every valid F32 activation range matches the CPU/full-WASM
                // reduction semantics.
                let first_val = input[input_base + y * params.in_w * params.in_c + c];
                var max_val = first_val;
                var sum_val = first_val;
                
                for (var x = 1u; x < params.in_w; x = x + 1u) {
                    let val = input[input_base + (y * params.in_w + x) * params.in_c + c];
                    if (val > max_val) { max_val = val; }
                    sum_val = sum_val + val;
                }
                
                let out_max_idx = c * params.in_h + y;
                let out_mean_idx = (c + params.in_c) * params.in_h + y;
                
                output[output_base + out_max_idx] = max_val;
                output[output_base + out_mean_idx] = sum_val / f32(params.in_w);
            }
