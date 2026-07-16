@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;
        struct Params {
            size : u32,
            axis_offset : u32,
            input_axis : u32,
            output_axis : u32,
            inner : u32,
            apply_sigmoid : u32,
            _p1 : u32,
            _p2 : u32,
        }
        @group(0) @binding(2) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
            let i = gid.x;
            if (i >= params.size) { return; }
            let inner_index = i % params.inner;
            let outer_axis = i / params.inner;
            let axis_index = outer_axis % params.input_axis;
            let outer_index = outer_axis / params.input_axis;
            let output_index = ((outer_index * params.output_axis + params.axis_offset + axis_index) * params.inner) + inner_index;
            var value = input[i];
            if (params.apply_sigmoid != 0u) {
                value = 1.0 / (1.0 + exp(-value));
            }
            output[output_index] = value;
        }
