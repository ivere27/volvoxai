// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> q:array<f32>;
@group(0) @binding(1) var<storage, read> k:array<f32>;
@group(0) @binding(2) var<storage, read> v:array<f32>;
@group(0) @binding(3) var<storage, read> gates:array<f32>;
@group(0) @binding(4) var<storage, read> initial:array<f32>;
@group(0) @binding(5) var<storage, read_write> output:array<f32>;
@group(0) @binding(6) var<storage, read_write> final_state:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, pad0:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, pad2:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(7) var<uniform> p:Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let c=unit%p.value_dim; let head_unit=unit/p.value_dim; let h=head_unit%p.heads; let batch=head_unit/p.heads;
    let kh=h/(p.heads/p.kv_heads); var state:array<f32,512>;
    for(var d=0u;d<p.key_dim;d++) {state[d]=0.0; if((p.flags&1u)!=0u) {state[d]=initial[(head_unit*p.key_dim+d)*p.value_dim+c];}}
    for(var t=0u;t<p.sequence;t++) {
        let row=batch*p.sequence+t; let g=(row*p.heads+h)*4u; let qk=(row*p.kv_heads+kh)*p.key_dim;
        var prediction=0.0;
        for(var d=0u;d<p.key_dim;d++) {state[d]*=gates[g]; prediction+=state[d]*k[qk+d]*gates[g+3u];}
        let correction=(v[(row*p.heads+h)*p.value_dim+c]-prediction)*gates[g+1u]; var value=0.0;
        for(var d=0u;d<p.key_dim;d++) {state[d]+=k[qk+d]*gates[g+3u]*correction; value+=state[d]*q[qk+d]*gates[g+2u]*p.scale;}
        output[(row*p.heads+h)*p.value_dim+c]=value;
    }
    if((p.flags&2u)!=0u) {for(var d=0u;d<p.key_dim;d++) {final_state[(head_unit*p.key_dim+d)*p.value_dim+c]=state[d];}}
}
