// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> hidden:array<f32>;
@group(0) @binding(1) var<storage, read> down_weight:array<f32>;
@group(0) @binding(2) var<storage, read> indices:array<f32>;
@group(0) @binding(3) var<storage, read> routes:array<f32>;
@group(0) @binding(4) var<storage, read_write> output:array<f32>;
struct Params {
    rows:u32, d_in:u32, hidden:u32, d_out:u32, experts:u32, top_k:u32, normalize:u32, scoring:u32,
    groups:u32, selected_groups:u32, group_top_k:u32, has_bias:u32, has_correction:u32, has_hash:u32, vocabulary:u32, input_scale:u32,
    slot:u32, work:u32, pad0:u32, pad1:u32, temperature:f32, scale:f32, epsilon:f32, limit:f32,
    pad2:u32, pad3:u32, pad4:u32, pad5:u32, pad6:u32, pad7:u32, pad8:u32, pad9:u32,
}
@group(0) @binding(5) var<uniform> p:Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let row=unit/p.d_out; let col=unit%p.d_out; let e=u32(indices[row*p.top_k+p.slot]); var value=0.0;
    for(var h=0u;h<p.hidden;h++) {value+=hidden[row*p.hidden+h]*down_weight[(e*p.hidden+h)*p.d_out+col];}
    if(p.input_scale==0u) {value*=routes[row*p.top_k+p.slot];}
    if(p.slot==0u) {output[unit]=value;} else {output[unit]+=value;}
}
