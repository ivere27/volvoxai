@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_bias: array<f32>;
struct Params { input: vec4<u32>, output: vec4<u32>, kernel: vec4<u32>, padding: vec4<u32> }
@group(0) @binding(6) var<uniform> params: Params;
fn out_index(b:u32,y:u32,x:u32,c:u32)->u32{return ((b*params.output.x+y)*params.output.y+x)*params.output.z+c;}
// Weights are HWIO [kh, kw, in_c, out_c]; kernel.x=kh, kernel.y=kw, input.w=in_c, output.z=out_c.
fn weight_index(ky:u32,kx:u32,ic:u32,oc:u32)->u32{return ((ky*params.kernel.y+kx)*params.input.w+ic)*params.output.z+oc;}

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x; let total=params.input.x*params.input.y*params.input.z*params.input.w; if(i>=total){return;}
 let c=i%params.input.w; let x=(i/params.input.w)%params.input.z; let y=(i/(params.input.w*params.input.z))%params.input.y; let b=i/(params.input.w*params.input.z*params.input.y); var sum=0.0;
 for(var oc=0u;oc<params.output.z;oc=oc+1u){for(var ky=0u;ky<params.kernel.x;ky=ky+1u){for(var kx=0u;kx<params.kernel.y;kx=kx+1u){let oy=i32(y*params.kernel.z+ky)-i32(params.kernel.w);let ox=i32(x*params.output.w+kx)-i32(params.padding.x);if(oy>=0&&ox>=0&&u32(oy)<params.output.x&&u32(ox)<params.output.y){sum=sum+grad_output[out_index(b,u32(oy),u32(ox),oc)]*weight[weight_index(ky,kx,c,oc)];}}}}
 grad_input[i]=grad_input[i]+sum;
}

@compute @workgroup_size(64)
fn weight_main(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x; let total=params.input.w*params.output.z*params.kernel.x*params.kernel.y;if(i>=total){return;}
 let oc=i%params.output.z;let c=(i/params.output.z)%params.input.w;let kx=(i/(params.output.z*params.input.w))%params.kernel.y;let ky=i/(params.output.z*params.input.w*params.kernel.y);var sum=0.0;
 for(var b=0u;b<params.input.x;b=b+1u){for(var y=0u;y<params.input.y;y=y+1u){for(var x=0u;x<params.input.z;x=x+1u){let oy=i32(y*params.kernel.z+ky)-i32(params.kernel.w);let ox=i32(x*params.output.w+kx)-i32(params.padding.x);if(oy>=0&&ox>=0&&u32(oy)<params.output.x&&u32(ox)<params.output.y){sum=sum+input[((b*params.input.y+y)*params.input.z+x)*params.input.w+c]*grad_output[out_index(b,u32(oy),u32(ox),oc)];}}}}
 grad_weight[i]=grad_weight[i]+sum;
}

@compute @workgroup_size(64)
fn bias_main(@builtin(global_invocation_id) gid:vec3<u32>){let c=gid.x;if(c>=params.output.z){return;}var sum=0.0;for(var b=0u;b<params.input.x;b=b+1u){for(var y=0u;y<params.output.x;y=y+1u){for(var x=0u;x<params.output.y;x=x+1u){sum=sum+grad_output[out_index(b,y,x,c)];}}}grad_bias[c]=grad_bias[c]+sum;}
