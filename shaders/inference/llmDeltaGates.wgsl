// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> q:array<f32>;
@group(0) @binding(1) var<storage, read> k:array<f32>;
@group(0) @binding(2) var<storage, read> a:array<f32>;
@group(0) @binding(3) var<storage, read> b:array<f32>;
@group(0) @binding(4) var<storage, read> a_log:array<f32>;
@group(0) @binding(5) var<storage, read> dt_bias:array<f32>;
@group(0) @binding(6) var<storage, read_write> gates:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, pad0:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, pad2:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(7) var<uniform> p:Params;
fn sigmoid(x:f32)->f32 {
    if(x>=0.0) { return 1.0/(1.0+exp(-x)); }
    let e=exp(x); return e/(1.0+e);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let h=unit%p.heads; let row=unit/p.heads; let kh=h/(p.heads/p.kv_heads);
    var qn=1.0; var kn=1.0;
    if(p.normalize!=0u) {
        var qq=0.0; var kk=0.0;
        for(var d=0u;d<p.key_dim;d++) {let offset=(row*p.kv_heads+kh)*p.key_dim+d; qq+=q[offset]*q[offset]; kk+=k[offset]*k[offset];}
        qn=inverseSqrt(qq+p.eps); kn=inverseSqrt(kk+p.eps);
    }
    let x=a[unit]+dt_bias[h];
    gates[unit*4u]=exp(-exp(a_log[h])*(max(x,0.0)+log(1.0+exp(-abs(x)))));
    gates[unit*4u+1u]=sigmoid(b[unit]); gates[unit*4u+2u]=qn; gates[unit*4u+3u]=kn;
}
