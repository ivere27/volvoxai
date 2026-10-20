// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> input:array<f32>;
@group(0) @binding(1) var<storage, read> gate_weight:array<f32>;
@group(0) @binding(2) var<storage, read> up_weight:array<f32>;
@group(0) @binding(3) var<storage, read> indices:array<f32>;
@group(0) @binding(4) var<storage, read> routes:array<f32>;
@group(0) @binding(5) var<storage, read_write> hidden:array<f32>;
struct Params {
    rows:u32, d_in:u32, hidden:u32, d_out:u32, experts:u32, top_k:u32, normalize:u32, scoring:u32,
    groups:u32, selected_groups:u32, group_top_k:u32, has_bias:u32, has_correction:u32, has_hash:u32, vocabulary:u32, input_scale:u32,
    slot:u32, work:u32, pad0:u32, pad1:u32, temperature:f32, scale:f32, epsilon:f32, limit:f32,
    pad2:u32, pad3:u32, pad4:u32, pad5:u32, pad6:u32, pad7:u32, pad8:u32, pad9:u32,
}
@group(0) @binding(6) var<uniform> p:Params;
fn sigmoid(x:f32)->f32 {
    if(x>=0.0) { return 1.0/(1.0+exp(-x)); }
    let e=exp(x); return e/(1.0+e);
}

// A hidden bank is reused for each selected expert; storage is rows*hidden,
// independent of top_k, and no giant concatenated expert bank is produced.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let row=unit/p.hidden; let h=unit%p.hidden; let e=u32(indices[row*p.top_k+p.slot]);
    var scale=1.0; if(p.input_scale!=0u) {scale=routes[row*p.top_k+p.slot];}
    var gate=0.0; var up=0.0;
    for(var d=0u;d<p.d_in;d++) {let x=input[row*p.d_in+d]*scale; let offset=(e*p.d_in+d)*p.hidden+h; gate+=x*gate_weight[offset]; up+=x*up_weight[offset];}
    if(p.limit>0.0) {gate=min(gate,p.limit); up=clamp(up,-p.limit,p.limit);}
    hidden[unit]=gate*sigmoid(gate)*up;
}
