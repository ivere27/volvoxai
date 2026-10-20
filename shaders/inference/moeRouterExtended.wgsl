// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> input:array<f32>;
@group(0) @binding(1) var<storage, read> weight:array<f32>;
@group(0) @binding(2) var<storage, read> bias:array<f32>;
@group(0) @binding(3) var<storage, read> correction:array<f32>;
@group(0) @binding(4) var<storage, read> tokens:array<i32>;
@group(0) @binding(5) var<storage, read> hash_indices:array<i32>;
@group(0) @binding(6) var<storage, read_write> indices:array<f32>;
@group(0) @binding(7) var<storage, read_write> routes:array<f32>;
struct Params {
    rows:u32, d_in:u32, hidden:u32, d_out:u32, experts:u32, top_k:u32, normalize:u32, scoring:u32,
    groups:u32, selected_groups:u32, group_top_k:u32, has_bias:u32, has_correction:u32, has_hash:u32, vocabulary:u32, input_scale:u32,
    slot:u32, work:u32, pad0:u32, pad1:u32, temperature:f32, scale:f32, epsilon:f32, limit:f32,
    pad2:u32, pad3:u32, pad4:u32, pad5:u32, pad6:u32, pad7:u32, pad8:u32, pad9:u32,
}
@group(0) @binding(8) var<uniform> p:Params;
fn sigmoid(x:f32)->f32 {
    if(x>=0.0) { return 1.0/(1.0+exp(-x)); }
    let e=exp(x); return e/(1.0+e);
}

// 8.5 KiB workgroup storage supports up to 1024 experts and 128 groups.
var<workgroup> scores:array<f32,1024>;
var<workgroup> selection:array<f32,1024>;
var<workgroup> group_scores:array<f32,128>;
fn corrected(e:u32)->f32 {var result=scores[e]; if(p.has_correction!=0u) {result+=correction[e];} return result;}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let row=wid.x; if(row>=p.rows) {return;}
    for(var e=lane;e<p.experts;e+=64u) {
        var x=0.0; if(p.has_bias!=0u) {x=bias[e];}
        for(var d=0u;d<p.d_in;d++) {x+=input[row*p.d_in+d]*weight[d*p.experts+e];}
        scores[e]=x/p.temperature;
    }
    workgroupBarrier(); if(lane!=0u) {return;}
    var maximum=scores[0]; for(var e=1u;e<p.experts;e++) {maximum=max(maximum,scores[e]);}
    var total=0.0;
    for(var e=0u;e<p.experts;e++) {
        let x=scores[e]; var value=exp(x-maximum);
        if(p.scoring==1u) {value=sigmoid(x);} else if(p.scoring==2u) {
            var softplus=log(1.0+exp(x)); if(x>20.0) {softplus=x;} else if(x< -20.0) {softplus=exp(x);} value=sqrt(softplus);
        }
        scores[e]=value; selection[e]=x; total+=value;
    }
    for(var e=0u;e<p.experts;e++) {if(p.scoring==0u) {scores[e]/=total;} if(p.has_correction!=0u) {selection[e]=corrected(e);}}
    let per_group=p.experts/p.groups;
    for(var g=0u;g<p.groups;g++) {
        group_scores[g]=0.0; if(p.selected_groups==p.groups) {continue;}
        for(var j=0u;j<per_group;j++) {
            let e=g*per_group+j; let candidate=corrected(e); var rank=0u;
            for(var l=0u;l<per_group;l++) {let other=g*per_group+l; let value=corrected(other); if(value>candidate || (value==candidate && other<e)) {rank++;}}
            if(rank<p.group_top_k) {group_scores[g]+=candidate;}
        }
    }
    for(var slot=0u;slot<p.top_k;slot++) {
        var best=p.experts;
        if(p.has_hash!=0u) {best=u32(hash_indices[u32(tokens[row])*p.top_k+slot]);}
        else {
            for(var e=0u;e<p.experts;e++) {
                let g=e/per_group; var rank=0u;
                for(var other=0u;other<p.groups;other++) {if(group_scores[other]>group_scores[g] || (group_scores[other]==group_scores[g] && other<g)) {rank++;}}
                if(rank>=p.selected_groups) {continue;}
                var used=false; for(var prior=0u;prior<slot;prior++) {if(u32(indices[row*p.top_k+prior])==e) {used=true;}}
                if(!used && (best==p.experts || selection[e]>selection[best])) {best=e;}
            }
        }
        indices[row*p.top_k+slot]=f32(best);
    }
    var denominator=p.epsilon; for(var slot=0u;slot<p.top_k;slot++) {denominator+=scores[u32(indices[row*p.top_k+slot])];}
    for(var slot=0u;slot<p.top_k;slot++) {var value=p.scale*scores[u32(indices[row*p.top_k+slot])]; if(p.normalize!=0u) {value/=denominator;} routes[row*p.top_k+slot]=value;}
}
