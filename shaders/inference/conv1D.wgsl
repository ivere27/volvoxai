@group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read> weight : array<f32>;
            @group(0) @binding(2) var<storage, read> bias : array<f32>;
            @group(0) @binding(3) var<storage, read_write> output : array<f32>;
            
            struct Params {
                in_c : u32, in_l : u32,
                out_c : u32, k : u32,
                stride : u32, pad : u32, relu : u32,
                batch : u32, groups: u32, in_per_group: u32, out_l: u32, unused: u32,
            }
            @group(0) @binding(4) var<uniform> params : Params;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let x = global_id.x;
                let oc = global_id.y;
                let b = global_id.z;
                let out_l = params.out_l;
                
                if (x >= out_l || oc >= params.out_c || b >= params.batch) { return; }
                
                // NLC activations [batch, l, c]; WIO weights [k, in_per_group, out_c].
                // Adjacent oc lanes read adjacent weights and write adjacent
                // outputs, so both accesses coalesce.
                var sum = bias[oc];
                let group_out = params.out_c / params.groups;
                let group = oc / group_out;
                let input_base = b * params.in_l * params.in_c;
                let output_base = b * out_l * params.out_c;

                for (var k = 0u; k < params.k; k = k + 1u) {
                    let in_x = i32(x * params.stride + k) - i32(params.pad);
                    if (in_x < 0 || in_x >= i32(params.in_l)) { continue; }
                    let in_base = input_base + u32(in_x) * params.in_c +
                                  group * params.in_per_group;
                    let w_tap = k * params.in_per_group * params.out_c;
                    for (var local_ic = 0u; local_ic < params.in_per_group; local_ic = local_ic + 1u) {
                        sum = sum + input[in_base + local_ic] *
                              weight[w_tap + local_ic * params.out_c + oc];
                    }
                }

                if (params.relu == 1u && sum < 0.0) { sum = 0.0; }
                output[output_base + x * params.out_c + oc] = sum;
            }
