// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> input:array<f32>;
@group(0) @binding(1) var<storage, read> weight:array<f32>;
@group(0) @binding(2) var<storage, read> bias:array<f32>;
@group(0) @binding(3) var<storage, read> scale:array<f32>;
@group(0) @binding(4) var<storage, read_write> output:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, pad0:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, pad2:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(5) var<uniform> p:Params;
fn sigmoid(x:f32)->f32 {
    if(x>=0.0) { return 1.0/(1.0+exp(-x)); }
    let e=exp(x); return e/(1.0+e);
}

fn raw(row:u32,index:u32,inverse:f32)->f32 {
    let width=p.streams*p.key_dim; var total=0.0;
    for(var d=0u;d<width;d++) {total+=input[row*width+d]*weight[index*width+d];}
    return total*inverse;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let row=gid.x; if(row>=p.work) {return;}
    let m=p.streams; let width=m*p.key_dim; var squared=0.0;
    for(var d=0u;d<width;d++) {squared+=input[row*width+d]*input[row*width+d];}
    let inverse=inverseSqrt(squared/f32(width)+p.eps);
    for(var c=0u;c<p.key_dim;c++) {output[row*p.key_dim+c]=0.0;}
    for(var i=0u;i<m;i++) {
        let pre=sigmoid(raw(row,i,inverse)*scale[0]+bias[i])+p.hc_eps;
        for(var c=0u;c<p.key_dim;c++) {output[row*p.key_dim+c]+=pre*input[(row*m+i)*p.key_dim+c];}
    }
}
