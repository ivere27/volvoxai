import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';
const trainingPrefix = 'volvoxai_training_';
const trainingExports = [
  'volvoxai_training_abi_version',
  'volvoxai_training_activation_backward_f32',
  'volvoxai_training_adamw_update_f32',
  'volvoxai_training_add_backward_f32',
  'volvoxai_training_add_f32',
  'volvoxai_training_all_finite_f32',
  'volvoxai_training_binary_broadcast_backward_f32',
  'volvoxai_training_binary_broadcast_f32',
  'volvoxai_training_expand_backward_f32',
  'volvoxai_training_expand_f32',
  'volvoxai_training_capabilities',
  'volvoxai_training_concat_backward_f32',
  'volvoxai_training_concat_f32',
  'volvoxai_training_conv2d_backward_f32',
  'volvoxai_training_conv1d_backward_f32',
  'volvoxai_training_conv1d_f32',
  'volvoxai_training_conv_transpose2d_f32',
  'volvoxai_training_conv_transpose2d_backward_f32',
  'volvoxai_training_pad2d_f32',
  'volvoxai_training_pad2d_backward_f32',
  'volvoxai_training_interp1d_f32',
  'volvoxai_training_interp1d_backward_f32',
  'volvoxai_training_prelu_backward_f32',
  'volvoxai_training_prelu_f32',
  'volvoxai_training_profile_x_backward_f32',
  'volvoxai_training_profile_x_f32',
  'volvoxai_training_profile_y_backward_f32',
  'volvoxai_training_profile_y_f32',
  'volvoxai_training_quantize_weight_f32_to_i8',
  'volvoxai_training_global_average_pool_backward_f32',
  'volvoxai_training_gather_backward_f32',
  'volvoxai_training_gather_elements_backward_f32',
  'volvoxai_training_gather_elements_f32',
  'volvoxai_training_gather_f32',
  'volvoxai_training_batchnorm2d_backward_f32',
  'volvoxai_training_averagepool2d_backward_f32',
  'volvoxai_training_cast_backward_f32',
  'volvoxai_training_cast_typed',
  'volvoxai_training_maxpool2d_backward_f32',
  'volvoxai_training_maxpool2d_f32',
  'volvoxai_training_mean_height_backward_f32',
  'volvoxai_training_mean_height_f32',
  'volvoxai_training_moe_linear_backward_f32',
  'volvoxai_training_moe_linear_f32',
  'volvoxai_training_moe_router_backward_f32',
  'volvoxai_training_moe_router_f32',
  'volvoxai_training_resize2d_backward_f32',
  'volvoxai_training_resize2d_f32',
  'volvoxai_training_split_backward_f32',
  'volvoxai_training_spatial_softargmax_y_backward_f32',
  'volvoxai_training_spatial_softargmax_y_f32',
  'volvoxai_training_clip_backward_f32',
  'volvoxai_training_copy_backward_f32',
  'volvoxai_training_cross_attention_backward_f32',
  'volvoxai_training_cross_attention_f32',
  'volvoxai_training_cross_sdpa_backward_f32',
  'volvoxai_training_cross_sdpa_f32',
  'volvoxai_training_cross_entropy_f32',
  'volvoxai_training_dequantize_weight_i8_to_f32',
  'volvoxai_training_dequantize_linear_backward_f32',
  'volvoxai_training_dequantize_linear_typed',
  'volvoxai_training_dropout_backward_f32',
  'volvoxai_training_dropout_f32',
  'volvoxai_training_embedding_backward_f32',
  'volvoxai_training_groupnorm_backward_f32',
  'volvoxai_training_groupnorm_f32',
  'volvoxai_training_layernorm_backward_f32',
  'volvoxai_training_linear_backward_f32',
  'volvoxai_training_linear_backward_packed_f32',
  'volvoxai_training_mul_backward_f32',
  'volvoxai_training_reduce_backward_f32',
  'volvoxai_training_rmsnorm_backward_f32',
  'volvoxai_training_scale_f32',
  'volvoxai_training_sdpa_backward_f32',
  'volvoxai_training_sdpa_f32',
  'volvoxai_training_sgd_update_f32',
  'volvoxai_training_slice_backward_f32',
  'volvoxai_training_slice_f32',
  'volvoxai_training_softmax_backward_f32',
  'volvoxai_training_softmax_f32',
  'volvoxai_training_sum_squares_f32',
  'volvoxai_training_transpose_backward_f32',
  'volvoxai_training_where_backward_f32',
  'volvoxai_training_where_f32',
  'volvoxai_training_zero_f32',
].sort();

async function compileWasm(output, sources) {
  await run(clang, [
    '--target=wasm32',
    '-O3',
    '-msimd128',
    '-nostdlib',
    '-Wl,--no-entry',
    '-Wl,--export-all',
    '-Wl,--allow-undefined',
    '-Inative/include',
    '-o', output,
    ...sources,
  ], {
    cwd: repositoryRoot,
    maxBuffer: 1024 * 1024,
  });
}

function importsFor(module) {
  const imports = {};
  for (const descriptor of WebAssembly.Module.imports(module)) {
    assert.equal(
      descriptor.kind,
      'function',
      `unexpected ${descriptor.kind} import ${descriptor.module}.${descriptor.name}`,
    );
    imports[descriptor.module] ||= {};
    imports[descriptor.module][descriptor.name] = () => 0;
  }
  return imports;
}

test('forward and full WASM artifacts enforce the training ABI boundary', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      t.skip(`${clang} is unavailable`);
      return;
    }
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-artifacts-'));
  try {
    const inferencePath = join(directory, 'volvoxai.wasm');
    const fullPath = join(directory, 'volvoxai.full.wasm');
    await compileWasm(inferencePath, ['native/src/kernels/kernels.c']);
    await compileWasm(fullPath, [
      'native/src/kernels/kernels.c',
      'native/src/kernels/training_kernels.c',
      'native/src/training/quantization.c',
    ]);

    const inferenceModule = await WebAssembly.compile(await readFile(inferencePath));
    const fullModule = await WebAssembly.compile(await readFile(fullPath));
    const inferenceEntries = WebAssembly.Module.exports(inferenceModule);
    const fullEntries = WebAssembly.Module.exports(fullModule);
    const fullByName = new Map(fullEntries.map((entry) => [entry.name, entry.kind]));

    assert.deepEqual(
      inferenceEntries.filter(({ name }) => name.startsWith(trainingPrefix)),
      [],
      'the inference module must not export any part of the training ABI',
    );
    assert.deepEqual(
      inferenceEntries.filter(({ name }) => name.startsWith('volvoxai_ptq_')),
      [],
      'the inference module must not export PTQ authoring helpers',
    );
    for (const { name, kind } of inferenceEntries) {
      assert.equal(fullByName.get(name), kind, `full WASM lost forward export '${name}'`);
    }
    const fullTrainingEntries = fullEntries
      .filter(({ name }) => name.startsWith(trainingPrefix));
    assert.deepEqual(
      fullTrainingEntries.map(({ name }) => name).sort(),
      trainingExports,
    );
    for (const { name, kind } of fullTrainingEntries) {
      assert.equal(kind, 'function', `training export '${name}' must be callable`);
    }

    const full = await WebAssembly.instantiate(fullModule, importsFor(fullModule));
    assert.equal(full.exports.volvoxai_training_abi_version(), 1);
    assert.equal(full.exports.volvoxai_training_capabilities(), 0x1ff);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
