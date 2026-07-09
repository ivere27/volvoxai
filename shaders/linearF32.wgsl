@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read> weight_f32 : array<f32>;
            @group(0) @binding(2) var<storage, read> dummyScale : array<f32>;
            @group(0) @binding(3) var<storage, read> bias : array<f32>;
            @group(0) @binding(4) var<storage, read_write> output : array<f32>;

            struct Params {
                seq_len : u32,
                d_in : u32,
                d_out : u32,
            }
            @group(0) @binding(5) var<uniform> params : Params;

            @compute @workgroup_size(64, 1, 1)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let row = global_id.y;
                let col = global_id.x;
                
                if (row >= params.seq_len || col >= params.d_out) { return; }
                
                var sum : f32 = 0.0;
                
                for (var k = 0u; k < params.d_in; k = k + 1u) {
                    let in_val = input[row * params.d_in + k];
                    let w_val = weight_f32[col * params.d_in + k];
                    sum = sum + in_val * w_val;
                }
                
                let b_val = bias[col];
                // dummyScale is passed just to keep bindings consistent but not used here.
                
                output[row * params.d_out + col] = sum + b_val;
            }
