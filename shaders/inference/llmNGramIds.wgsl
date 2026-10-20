// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> tokens:array<i32>;
@group(0) @binding(1) var<storage, read> multipliers:array<u32>;
@group(0) @binding(2) var<storage, read> sizes:array<i32>;
@group(0) @binding(3) var<storage, read> offsets:array<i32>;
@group(0) @binding(4) var<storage, read_write> output:array<i32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first_unit:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, query_offset:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(5) var<uniform> p:Params;

// Exact two-word wrapping I64 products and signed Euclidean remainder need no
// optional shader-i64 feature; no hash or global embedding address passes F32.
fn product(a:u32,low:u32,high:u32)->vec2<u32> {
    let a0=a&65535u; let a1=a>>16u; let b0=low&65535u; let b1=low>>16u;
    let p0=a0*b0; let p1=a1*b0; let p2=a0*b1; let carry=(p0>>16u)+(p1&65535u)+(p2&65535u);
    return vec2<u32>((carry<<16u)|(p0&65535u),a1*b1+(p1>>16u)+(p2>>16u)+(carry>>16u)+a*high);
}
fn remainder(words:vec2<u32>,divisor:u32)->u32 {
    let negative=(words.y&0x80000000u)!=0u; var absolute=words;
    if(negative) {absolute.x=~words.x+1u; absolute.y=~words.y+select(0u,1u,absolute.x==0u);}
    var r=0u;
    for(var bit=0u;bit<64u;bit++) {var word=absolute.y; var shift=31u-bit; if(bit>=32u) {word=absolute.x; shift=63u-bit;}
        r=(r*2u+((word>>shift)&1u))%divisor;}
    if(negative && r!=0u) {return divisor-r;} return r;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    if(gid.x>=p.work) {return;} let unit=gid.x+p.first_unit;
    // heads=(ngram_size-1)*heads_per_ngram; groups=heads_per_ngram;
    // rotary_offset=eos (-1 disables reset), normalize=pad token.
    // sequence is the per-batch token row width.
    let head=unit%p.heads; let row=unit/p.heads; let position=row%p.sequence; let batch=row/p.sequence; let length=head/p.groups+2u;
    var mixed=product(u32(tokens[row]),multipliers[0],multipliers[1]); var boundary=false;
    for(var shift=1u;shift<length;shift++) {
        var token=p.normalize;
        if(shift<=position && !boundary) {token=u32(tokens[batch*p.sequence+position-shift]); if(i32(p.rotary_offset)>=0 && token==p.rotary_offset) {boundary=true;}}
        mixed=mixed^product(token,multipliers[shift*2u],multipliers[shift*2u+1u]);
    }
    output[unit]=i32(remainder(mixed,u32(sizes[head]))+u32(offsets[head]));
}
