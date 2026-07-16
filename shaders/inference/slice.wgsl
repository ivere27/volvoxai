@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
// General strided slice over 4 dims (leading dims are padded to 1). For each
// output element the source coord along axis k is start[k] + out_coord * step[k].
struct Params {
  out_b: u32, out_c: u32, out_h: u32, out_w: u32,
  in_c: u32, in_h: u32, in_w: u32,
  s0: u32, s1: u32, s2: u32, s3: u32,
  st0: u32, st1: u32, st2: u32, st3: u32,
  total: u32
}
@group(0) @binding(2) var<uniform> p : Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.total) { return; }
  let ow = idx % p.out_w;
  let oh = (idx / p.out_w) % p.out_h;
  let oc = (idx / (p.out_w * p.out_h)) % p.out_c;
  let ob = idx / (p.out_w * p.out_h * p.out_c);
  let ib = p.s0 + ob * p.st0;
  let ic = p.s1 + oc * p.st1;
  let ih = p.s2 + oh * p.st2;
  let iw = p.s3 + ow * p.st3;
  output[idx] = input[((ib * p.in_c + ic) * p.in_h + ih) * p.in_w + iw];
}
