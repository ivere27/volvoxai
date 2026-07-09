@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read> weight_int8_packed : array<u32>;
            @group(0) @binding(2) var<storage, read> weight_scales : array<f32>;
            @group(0) @binding(3) var<storage, read> bias : array<f32>;
            @group(0) @binding(4) var<storage, read_write> output : array<f32>;

            struct Params {
                seq_len : u32,
                d_in : u32,
                d_out : u32,
            }
            @group(0) @binding(5) var<uniform> params : Params;

            // Unpack one signed 8-bit integer from a 32-bit packed block
            fn unpack_i8(packed: u32, byte_idx: u32) -> f32 {
                let val_i32 = extractBits(i32(packed), byte_idx * 8u, 8u);
                return f32(val_i32);
            }

            @compute @workgroup_size(64, 1, 1)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let row = global_id.y;
                let col = global_id.x;
                
                if (row >= params.seq_len || col >= params.d_out) { return; }
                
                var sum : f32 = 0.0;
                let d_in_4 = params.d_in / 4u;
                
                for (var i = 0u; i < d_in_4; i = i + 1u) {
                    let w_packed = weight_int8_packed[col * d_in_4 + i];
                    let in_base = row * params.d_in + i * 4u;
                    
                    sum = sum + input[in_base + 0u] * f32(extractBits(i32(w_packed), 0u, 8u));
                    sum = sum + input[in_base + 1u] * f32(extractBits(i32(w_packed), 8u, 8u));
                    sum = sum + input[in_base + 2u] * f32(extractBits(i32(w_packed), 16u, 8u));
                    sum = sum + input[in_base + 3u] * f32(extractBits(i32(w_packed), 24u, 8u));
                }
                
                let scale = weight_scales[col];
                output[row * params.d_out + col] = (sum * scale) + bias[col];
            }
