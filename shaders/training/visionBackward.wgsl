// NHWC receipt-vision primitive backward kernels. Profile maxima use a strict
// comparison so ties select the first collapsed-axis value, matching CPU.
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> output : array<f32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input : array<f32>;

struct Params {
  height : u32,
  width : u32,
  channels : u32,
  batch : u32,
}

@group(0) @binding(4) var<uniform> params : Params;

fn input_index(batch : u32, y : u32, x : u32, channel : u32) -> u32 {
  return ((batch * params.height + y) * params.width + x) * params.channels + channel;
}

fn column_output_index(batch : u32, channel : u32, x : u32) -> u32 {
  return (batch * params.channels + channel) * params.width + x;
}

fn profile_x_output_index(batch : u32, channel : u32, x : u32) -> u32 {
  return (batch * 2u * params.channels + channel) * params.width + x;
}

fn profile_y_output_index(batch : u32, channel : u32, y : u32) -> u32 {
  return (batch * 2u * params.channels + channel) * params.height + y;
}

@compute @workgroup_size(64)
fn mean_height_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let channel = gid.y;
  let batch = gid.z;
  if (x >= params.width || channel >= params.channels || batch >= params.batch) { return; }

  let gradient = grad_output[column_output_index(batch, channel, x)] / f32(params.height);
  for (var y = 0u; y < params.height; y = y + 1u) {
    let index = input_index(batch, y, x, channel);
    grad_input[index] = grad_input[index] + gradient;
  }
}

@compute @workgroup_size(64)
fn profile_x_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let channel = gid.y;
  let batch = gid.z;
  if (x >= params.width || channel >= params.channels || batch >= params.batch) { return; }

  var max_y = 0u;
  var maximum = input[input_index(batch, 0u, x, channel)];
  for (var y = 1u; y < params.height; y = y + 1u) {
    let value = input[input_index(batch, y, x, channel)];
    if (value > maximum) {
      maximum = value;
      max_y = y;
    }
  }
  let max_gradient = grad_output[profile_x_output_index(batch, channel, x)];
  let mean_gradient = grad_output[profile_x_output_index(batch, channel + params.channels, x)] /
    f32(params.height);
  for (var y = 0u; y < params.height; y = y + 1u) {
    let index = input_index(batch, y, x, channel);
    grad_input[index] = grad_input[index] + mean_gradient;
  }
  let max_index = input_index(batch, max_y, x, channel);
  grad_input[max_index] = grad_input[max_index] + max_gradient;
}

@compute @workgroup_size(64)
fn profile_y_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let y = gid.x;
  let channel = gid.y;
  let batch = gid.z;
  if (y >= params.height || channel >= params.channels || batch >= params.batch) { return; }

  var max_x = 0u;
  var maximum = input[input_index(batch, y, 0u, channel)];
  for (var x = 1u; x < params.width; x = x + 1u) {
    let value = input[input_index(batch, y, x, channel)];
    if (value > maximum) {
      maximum = value;
      max_x = x;
    }
  }
  let max_gradient = grad_output[profile_y_output_index(batch, channel, y)];
  let mean_gradient = grad_output[profile_y_output_index(batch, channel + params.channels, y)] /
    f32(params.width);
  for (var x = 0u; x < params.width; x = x + 1u) {
    let index = input_index(batch, y, x, channel);
    grad_input[index] = grad_input[index] + mean_gradient;
  }
  let max_index = input_index(batch, y, max_x, channel);
  grad_input[max_index] = grad_input[max_index] + max_gradient;
}

@compute @workgroup_size(64)
fn spatial_softargmax_y_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let channel = gid.y;
  let batch = gid.z;
  if (x >= params.width || channel >= params.channels || batch >= params.batch) { return; }

  var maximum = input[input_index(batch, 0u, x, channel)];
  for (var y = 1u; y < params.height; y = y + 1u) {
    let value = input[input_index(batch, y, x, channel)];
    if (value > maximum) { maximum = value; }
  }
  var denominator = 0.0;
  for (var y = 0u; y < params.height; y = y + 1u) {
    denominator = denominator + exp(input[input_index(batch, y, x, channel)] - maximum);
  }
  if (!(denominator > 0.0)) { return; }

  let output_index = column_output_index(batch, channel, x);
  let upstream = grad_output[output_index];
  let expectation = output[output_index];
  for (var y = 0u; y < params.height; y = y + 1u) {
    let index = input_index(batch, y, x, channel);
    let probability = exp(input[index] - maximum) / denominator;
    let coordinate = (f32(y) + 0.5) / f32(params.height);
    grad_input[index] = grad_input[index] + upstream * probability * (coordinate - expectation);
  }
}
