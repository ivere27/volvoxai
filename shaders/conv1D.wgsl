@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read> weight : array<f32>;
            @group(0) @binding(2) var<storage, read> bias : array<f32>;
            @group(0) @binding(3) var<storage, read_write> output : array<f32>;
            
            struct Params {
                in_c : u32, in_l : u32,
                out_c : u32, k : u32,
                stride : u32, pad : u32, relu : u32
            }
            @group(0) @binding(4) var<uniform> params : Params;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let x = global_id.x;
                let oc = global_id.y;
                let out_l = (params.in_l + 2u * params.pad - params.k) / params.stride + 1u;
                
                if (x >= out_l || oc >= params.out_c) { return; }
                
                var sum = bias[oc];
                let w_base = oc * params.in_c * params.k;
                
                for (var ic = 0u; ic < params.in_c; ic = ic + 1u) {
                    let in_base = ic * params.in_l;
                    let w_ic_base = w_base + ic * params.k;
                    for (var k = 0u; k < params.k; k = k + 1u) {
                        let in_x = i32(x * params.stride + k) - i32(params.pad);
                        if (in_x >= 0 && in_x < i32(params.in_l)) {
                            sum = sum + input[in_base + u32(in_x)] * weight[w_ic_base + k];
                        }
                    }
                }
                
                if (params.relu == 1u && sum < 0.0) { sum = 0.0; }
                output[oc * out_l + x] = sum;
            }
