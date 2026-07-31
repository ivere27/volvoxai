// Native graph control operations over conventional 32-bit storage.
//
// This shader deliberately uses raw u32 storage for every tensor. I32
// comparisons reinterpret those bits, while Cast performs an explicit numeric
// conversion. Copy preserves every bit, including F32 NaN payloads.
// F32 -> I32 matches the portable and CUDA contract: non-finite values become
// zero; finite values truncate toward zero and then wrap modulo 2^32.
//
// Metadata is a tightly packed storage buffer:
//   [elements, rank, operation, input_dtype, output_dtype, minimum_bits,
//    maximum_bits, pad, output_strides[8], a_strides[8], b_strides[8]]
//
// operation:
//   0 Equal(I32), 1 GreaterOrEqual(I32), 2 Not(I32), 3 Clip(I32),
//   4 raw copy, 5 Cast(F32/I32)
@group(0) @binding(0) var<storage, read> a_words : array<u32>;
@group(0) @binding(1) var<storage, read> b_words : array<u32>;
@group(0) @binding(2) var<storage, read_write> output_words : array<u32>;

struct Metadata {
  values : array<u32>,
}
@group(0) @binding(3) var<storage, read> metadata : Metadata;

fn operand_index(output_index : u32, stride_base : u32) -> u32 {
  var remainder = output_index;
  var index = 0u;
  for (var dimension = 0u;
       dimension < metadata.values[1];
       dimension = dimension + 1u) {
    let output_stride = metadata.values[8u + dimension];
    let coordinate = remainder / output_stride;
    remainder = remainder % output_stride;
    index = index + coordinate * metadata.values[stride_base + dimension];
  }
  return index;
}

fn cast_f32_to_u32_mod(value : f32) -> u32 {
  let bits = bitcast<u32>(value);
  let exponent = (bits >> 23u) & 0xffu;
  if (exponent == 0xffu || exponent < 127u) {
    return 0u;
  }

  let significand = (bits & 0x7fffffu) | 0x800000u;
  let shift = i32(exponent) - 127 - 23;
  var magnitude = 0u;
  if (shift < 32) {
    if (shift >= 0) {
      magnitude = significand << u32(shift);
    } else {
      magnitude = significand >> u32(-shift);
    }
  }
  return select(magnitude, 0u - magnitude, (bits & 0x80000000u) != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_index = gid.x;
  if (output_index >= metadata.values[0]) { return; }

  let operation = metadata.values[2];
  if (operation == 0u || operation == 1u) {
    let av = bitcast<i32>(a_words[operand_index(output_index, 16u)]);
    let bv = bitcast<i32>(b_words[operand_index(output_index, 24u)]);
    let predicate = select(av == bv, av >= bv, operation == 1u);
    output_words[output_index] = select(0u, 1u, predicate);
    return;
  }

  let input_word = a_words[output_index];
  if (operation == 2u) {
    output_words[output_index] =
      select(0u, 1u, bitcast<i32>(input_word) == 0);
  } else if (operation == 3u) {
    let minimum = bitcast<i32>(metadata.values[5]);
    let maximum = bitcast<i32>(metadata.values[6]);
    output_words[output_index] =
      bitcast<u32>(clamp(bitcast<i32>(input_word), minimum, maximum));
  } else if (operation == 4u ||
             metadata.values[3] == metadata.values[4]) {
    output_words[output_index] = input_word;
  } else if (metadata.values[3] == 16u &&
             metadata.values[4] == 18u) {
    output_words[output_index] =
      bitcast<u32>(f32(bitcast<i32>(input_word)));
  } else {
    output_words[output_index] = cast_f32_to_u32_mod(bitcast<f32>(input_word));
  }
}
