// C supplies lane positions and the source mask layout. Values stay on device.
@group(0) @binding(0) var<storage, read> source : array<i32>;
@group(0) @binding(1) var<storage, read> positions : array<i32>;
@group(0) @binding(2) var<storage, read_write> output : array<i32>;
struct Params {
  lanes : u32,
  keys : u32,
  source_keys : u32,
  source_queries : u32,
  mask_mode : u32,
  causal : u32,
}
@group(0) @binding(3) var<uniform> params : Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.lanes * params.keys) { return; }
  let lane = gid.x / params.keys;
  let key = gid.x % params.keys;
  let position = positions[lane];
  if (position < 0 || (params.causal != 0u && key > u32(position))) {
    output[gid.x] = 0;
    return;
  }
  var value = 1;
  switch params.mask_mode {
    case 1u: { value = source[key]; }
    case 2u: { value = source[lane * params.source_keys + key]; }
    case 3u: { value = source[u32(position) * params.source_keys + key]; }
    case 4u: { value = source[(lane * params.source_queries + u32(position)) * params.source_keys + key]; }
    default: {}
  }
  output[gid.x] = select(0, 1, value != 0);
}
