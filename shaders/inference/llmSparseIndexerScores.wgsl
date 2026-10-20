// Whole-prefix compressed-bank scores. ABI: llm_shader_abi.md.
@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> k:array<f32>;
@group(0) @binding(2) var<storage,read> weights:array<f32>;
@group(0) @binding(3) var<storage,read> candidates:array<i32>;
@group(0) @binding(4) var<storage,read_write> scores:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first_unit:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, query_offset:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(5) var<uniform> p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let id=gid.x;if(id>=p.work){return;}
    let row=id/p.keys;let j=id%p.keys;let query=row%p.sequence;let batch=row/p.sequence;
    scores[id]=bitcast<f32>(0xff800000u);
    if(j>=(query+1u)/p.rate){return;}
    if((p.flags&1u)!=0u && candidates[id]==0){return;}
    var score=0.0;
    for(var h=0u;h<p.heads;h++) {
        var dot=0.0;
        for(var d=0u;d<p.key_dim;d++){dot+=q[(row*p.heads+h)*p.key_dim+d]*k[(batch*p.keys+j)*p.key_dim+d];}
        score+=max(dot,0.0)*weights[row*p.heads+h];
    }
    scores[id]=score/sqrt(f32(p.key_dim)*f32(p.heads));
}
