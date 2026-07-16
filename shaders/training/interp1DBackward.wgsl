@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;
struct Params { batch:u32, channels:u32, input_length:u32, output_length:u32 }
@group(0) @binding(2) var<uniform> params:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let index=gid.x;let total=params.batch*params.channels*params.input_length;if(index>=total){return;}let ix=index%params.input_length;let row=index/params.input_length;let scale=f32(params.input_length)/f32(params.output_length);var sum=0.0;
 for(var ox=0u;ox<params.output_length;ox=ox+1u){var position=(f32(ox)+.5)*scale-.5;position=clamp(position,0.0,f32(params.input_length-1u));let x0=u32(position);let x1=min(x0+1u,params.input_length-1u);let fraction=position-f32(x0);let g=grad_output[row*params.output_length+ox];if(ix==x0){sum=sum+g*(1.0-fraction);}if(ix==x1){sum=sum+g*fraction;}}
 grad_input[index]=grad_input[index]+sum;
}
