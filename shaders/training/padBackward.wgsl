@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;
struct Params { input: vec4<u32>, output: vec4<u32>, pads: vec4<u32> }
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index=gid.x; let total=params.input.x*params.input.y*params.input.z*params.input.w; if(index>=total){return;}
  let c=index%params.input.w;let x=(index/params.input.w)%params.input.z;let y=(index/(params.input.w*params.input.z))%params.input.y;let b=index/(params.input.w*params.input.z*params.input.y);
  let output_index=((b*params.output.x+y+params.pads.x)*params.output.y+x+params.pads.y)*params.input.w+c;
  grad_input[index]=grad_input[index]+grad_output[output_index];
}
