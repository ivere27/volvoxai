/**
 * Dependency-free authority for the private JavaScript <-> release-WASM ABI.
 *
 * This is not an application API. Public operations continue to be owned only
 * by proto/volvoxai.proto. Keep this file data-only so code generation and
 * provenance checks can run before npm dependencies are installed.
 */

export const WASM_INTERNAL_ABI_FORMAT = 'volvoxai-wasm-internal-abi/v1';

const BOTH_PROFILES = Object.freeze(['inference', 'full']);
const FULL_PROFILE = Object.freeze(['full']);

function entries(names, {
  owner,
  consumer,
  group,
  kind = 'function',
  profiles = BOTH_PROFILES,
  tsSignature = 'numeric',
}) {
  return names.map((name) => ({
    name,
    owner,
    consumer,
    group,
    kind,
    profiles,
    tsSignature,
  }));
}

function deepFreeze(value) {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) deepFreeze(child);
  }
  return value;
}

// Absolute monotonic microseconds, shared with VxPlatformService samples.
// A negative f64 means unavailable; no wall-clock substitute is permitted.
const HOST_IMPORTS = [...entries(['vx_host_monotonic_micros_v1'], {
  owner: 'host-clock',
  consumer: 'release-parent',
  group: 'clock-import',
  tsSignature: 'numeric',
}), ...entries(['vx_host_random_u64_v1'], {
  owner: 'host-entropy', consumer: 'release-parent', group: 'entropy-import',
  tsSignature: 'i64Result',
})].map((entry) => ({ ...entry, module: 'host' }));

// Private device transport. Layouts and signatures are generated for both C
// and TypeScript; application lifecycle stays in volvoxai.proto.
export const GPU_BRIDGE_ABI = {
  version: 6,
  constants: { PENDING: -2, ERROR: -1, DEVICE_LOST: -3, OK: 0, NO_UNIFORM: 255 },
  structs: {
    VxGpuDispatch: ['variant_count', 'binding_count', 'params_bytes', 'node_index', 'params_slot'],
    VxGpuVariant: ['shader_id', 'workgroup[3]'],
    VxGpuBinding: ['slot', 'host_ptr', 'offset', 'bytes', 'writes'],
    VxGpuLimits: ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxUniformBufferBindingSize',
      'maxStorageBuffersPerShaderStage', 'maxUniformBuffersPerShaderStage', 'maxBindingsPerBindGroup',
      'maxBindGroups', 'minStorageBufferOffsetAlignment', 'maxComputeWorkgroupStorageSize',
      'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY',
      'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupsPerDimension'],
  },
  functions: {
    vx_gpu_available: ['int', 'uint32_t abi_hash', 'uint32_t catalog_hash'],
    vx_gpu_limits: ['int', 'uint32_t destination', 'uint32_t bytes'],
    vx_gpu_ensure: ['int', 'uint32_t host_ptr', 'uint32_t bytes', 'int is_weight'],
    vx_gpu_release: ['void', 'uint32_t host_ptr'],
    vx_gpu_invalidate: ['void', 'uint32_t host_ptr'],
    vx_gpu_begin: ['void'],
    vx_gpu_encode: ['int', 'uint32_t dispatch'],
    vx_gpu_end: ['int'],
    vx_gpu_snapshot: ['int', 'uint32_t host_ptr', 'uint32_t offset', 'uint32_t bytes'],
    vx_gpu_readback: ['int', 'uint32_t ticket', 'uint32_t destination', 'uint32_t bytes'],
    vx_gpu_readback_release: ['void', 'uint32_t ticket'],
  },
};

const GPU_BRIDGE_IMPORTS = entries(Object.keys(GPU_BRIDGE_ABI.functions), {
  owner: 'host-gpu-bridge',
  consumer: 'release-parent',
  group: 'gpu-bridge-import',
  kind: 'function',
  profiles: FULL_PROFILE,
  tsSignature: 'gpuBridgeImport',
}).map((entry) => ({ ...entry, module: 'gpu' }));

const ALLOCATOR_EXPORTS = [
  ...entries(['memory'], {
    owner: 'wasm-linker',
    consumer: 'provider-and-control',
    group: 'allocator',
    kind: 'memory',
    tsSignature: 'memory',
  }),
  ...entries(['__heap_base'], {
    owner: 'wasm-linker',
    consumer: 'provider-and-control',
    group: 'allocator',
    kind: 'global',
    tsSignature: 'global',
  }),
  ...entries([
    'wasm_runtime_abi_version',
    'alloc_bytes',
    'reset_heap',
    'heap_mark',
    'heap_rewind',
  ], {
    owner: 'portable-runtime',
    consumer: 'provider-and-control',
    group: 'allocator',
  }),
];

const SYNURANG_CALL_EXPORTS = entries([
  'synurang_module_abi_version', 'synurang_module_create',
  'synurang_module_destroy', 'synurang_module_open', 'synurang_module_send',
  'synurang_module_half_close', 'synurang_module_receive',
  'synurang_module_cancel', 'synurang_module_release', 'synurang_module_poll',
  'synurang_module_has_work', 'synurang_module_free', 'synurang_module_alloc',
], {
  owner: 'synurang-call', consumer: 'model-control', group: 'model-control',
  tsSignature: 'wasmFunction',
});

const SYNURANG_CALL_IMPORTS = entries(['wakeup'], {
  owner: 'synurang-call', consumer: 'release-parent', group: 'call-import',
}).map((entry) => ({ ...entry, module: 'synurang' }));

const MODEL_CONTROL_SUPPORT_EXPORTS = entries([
  'vx_wasm_inference_path_preflight_v1',
  'vx_wasm_mount_file_abort',
  'vx_wasm_mount_file_begin',
  'vx_wasm_mount_file_finish',
  'vx_wasm_mount_file_write',
  'vx_wasm_unmount_file',
], {
  owner: 'portable-runtime',
  consumer: 'model-control',
  group: 'model-control',
});

const GPU_CONTROL_EXPORTS = entries(['vx_wasm_prepare_gpu_v1'], {
  owner: 'portable-runtime',
  consumer: 'model-control',
  group: 'gpu-control',
  profiles: FULL_PROFILE,
});

const SCHEDULER_EXPORTS = [
  ...entries(['vx_scheduler_policy_abi_version'], {
    owner: 'portable-scheduler',
    consumer: 'runtime-scheduler',
    group: 'scheduler',
  }),
  ...entries(['vx_scheduler_policy_effective_priority_v1'], {
    owner: 'portable-scheduler',
    consumer: 'runtime-scheduler',
    group: 'scheduler',
    tsSignature: 'i64Result',
  }),
  ...entries([
    'vx_scheduler_policy_precedes_v1',
    'vx_scheduler_policy_deadline_expired_v1',
    'vx_scheduler_policy_latest_matches_v1',
    'vx_scheduler_policy_budget_admits_v1',
  ], {
    owner: 'portable-scheduler',
    consumer: 'runtime-scheduler',
    group: 'scheduler',
    tsSignature: 'mixedI64',
  }),
];

const FORWARD_EXPORTS = entries([
  'add_broadcast_f32',
  'argmax_axis_typed',
  'averagepool2d_f32',
  'batch_norm2d_f32',
  'binary_broadcast_f32',
  'cast_typed',
  'clip_f32',
  'clip_i32',
  'compare_broadcast_i32',
  'concat_slice_f32',
  'concat_slice_i8u8',
  'concat_slice_u32',
  'conv1d_f32',
  'conv2d_f32',
  'conv_transpose2d_f32',
  'copy_f32',
  'copy_i8u8',
  'cos_f32',
  'cross_attention_f32',
  'cross_sdpa_f32',
  'dequantize_linear_typed',
  'embedding_f32',
  'expand_nd_f32',
  'expand_nd_i8u8',
  'expand_nd_u32',
  'gather_elements_i32_f32',
  'gather_i32_f32',
  'gelu_f32',
  'gelu_tanh_f32',
  'gemm_f32_cache_budget_bytes',
  'gemm_f32_pack_b',
  'gemm_f32_packed',
  'gemm_f32_packed_elements',
  'gemm_f32_tile_kc',
  'gemm_f32_tile_mr',
  'gemm_f32_tile_nr',
  'gemm_f32_working_set_bytes',
  'global_average_pool_f32',
  'groupnorm_f32',
  'hardsigmoid_f32',
  'hardswish_f32',
  'interp1d_f32',
  'layernorm_f32',
  'leakyrelu_f32',
  'linear_f32',
  'logsoftmax_f32',
  'matmul_f32',
  'matmul_quantized_f32',
  'matmul_quantized_f32_packed',
  'maxpool2d_f32',
  'maxpool2d_i8u8',
  'mean_height_f32',
  'moe_linear_f32',
  'moe_router_f32',
  'mul_broadcast_f32',
  'mul_f32',
  'non_max_suppression_typed',
  'not_i32',
  'pack_q8_weight',
  'pack_q8_weight_canonical',
  'packed_q8_weight_canonical_size',
  'packed_q8_weight_size',
  'pad_2d_f32',
  'prelu_generic_f32',
  'profile_x_f32',
  'profile_y_f32',
  'qadd_i8u8',
  'qargmax_i8u8',
  'qbatch_matmul_i8u8',
  'qbatch_matmul_i8u8_simd128',
  'qconv2d_i8u8',
  'qconv2d_im2col_i8u8',
  'qembedding_i8u8',
  'qgelu_i8u8',
  'qgroupnorm_i8u8',
  'qlayernorm_i8u8',
  'qlinear_i8u8',
  'qlinear_i8u8_packed',
  'qmaskedmean_i8u8',
  'qsdpa_i8u8',
  'qsilu_i8u8',
  'quantize_linear_typed',
  'reduce_mean_f32',
  'reduce_sum_f32',
  'relu_f32',
  'requantize_linear_i8u8',
  'resize_bilinear_f32',
  'resize_nearest2d_f32',
  'resize_nearest2d_i8u8',
  'rmsnorm_f32',
  'rope_f32',
  'sdpa_f32',
  'sigmoid_f32',
  'silu_f32',
  'sin_f32',
  'slice_nd_f32',
  'slice_nd_u32',
  'softmax_f32',
  'spatial_softargmax_y_f32',
  'split_slice_f32',
  'split_slice_u32',
  'ssm_scan_f32',
  'tanh_f32',
  'transpose_nd_f32',
  'transpose_nd_i8u8',
  'transpose_nd_u32',
  'upsample_nearest2x_f32',
  'vx_moe_linear_banked_f32',
  'where_broadcast_32',
  'where_typed_32',
  'where_typed_f32',
], {
  owner: 'portable-kernels',
  consumer: 'wasm-provider',
  group: 'forward',
});

const TRAINING_EXPORTS = entries([
  'volvoxai_training_abi_version',
  'volvoxai_training_activation_backward_f32',
  'volvoxai_training_capabilities',
  'volvoxai_training_adamw_update_f32',
  'volvoxai_training_add_backward_f32',
  'volvoxai_training_add_f32',
  'volvoxai_training_all_finite_f32',
  'volvoxai_training_averagepool2d_backward_f32',
  'volvoxai_training_batchnorm2d_backward_f32',
  'volvoxai_training_binary_broadcast_backward_f32',
  'volvoxai_training_binary_broadcast_f32',
  'volvoxai_training_cast_backward_f32',
  'volvoxai_training_cast_typed',
  'volvoxai_training_clip_backward_f32',
  'volvoxai_training_concat_backward_f32',
  'volvoxai_training_concat_f32',
  'volvoxai_training_conv1d_backward_f32',
  'volvoxai_training_conv1d_f32',
  'volvoxai_training_conv2d_backward_f32',
  'volvoxai_training_conv_transpose2d_backward_f32',
  'volvoxai_training_conv_transpose2d_f32',
  'volvoxai_training_copy_backward_f32',
  'volvoxai_training_cross_attention_backward_f32',
  'volvoxai_training_cross_attention_f32',
  'volvoxai_training_cross_entropy_f32',
  'volvoxai_training_cross_sdpa_backward_f32',
  'volvoxai_training_cross_sdpa_f32',
  'volvoxai_training_dequantize_linear_backward_f32',
  'volvoxai_training_dequantize_linear_typed',
  'volvoxai_training_dequantize_weight_i8_to_f32',
  'volvoxai_training_dropout_backward_f32',
  'volvoxai_training_dropout_f32',
  'volvoxai_training_embedding_backward_f32',
  'volvoxai_training_expand_backward_f32',
  'volvoxai_training_expand_f32',
  'volvoxai_training_gather_backward_f32',
  'volvoxai_training_gather_elements_backward_f32',
  'volvoxai_training_gather_elements_f32',
  'volvoxai_training_gather_f32',
  'volvoxai_training_global_average_pool_backward_f32',
  'volvoxai_training_groupnorm_backward_f32',
  'volvoxai_training_groupnorm_f32',
  'volvoxai_training_interp1d_backward_f32',
  'volvoxai_training_interp1d_f32',
  'volvoxai_training_layernorm_backward_f32',
  'volvoxai_training_linear_backward_f32',
  'volvoxai_training_linear_backward_packed_f32',
  'volvoxai_training_maxpool2d_backward_f32',
  'volvoxai_training_maxpool2d_f32',
  'volvoxai_training_mean_height_backward_f32',
  'volvoxai_training_mean_height_f32',
  'volvoxai_training_moe_linear_backward_banked_f32',
  'volvoxai_training_moe_linear_backward_f32',
  'volvoxai_training_moe_linear_banked_f32',
  'volvoxai_training_moe_linear_f32',
  'volvoxai_training_moe_router_backward_f32',
  'volvoxai_training_moe_router_f32',
  'volvoxai_training_mul_backward_f32',
  'volvoxai_training_pad2d_backward_f32',
  'volvoxai_training_pad2d_f32',
  'volvoxai_training_prelu_backward_f32',
  'volvoxai_training_prelu_f32',
  'volvoxai_training_profile_x_backward_f32',
  'volvoxai_training_profile_x_f32',
  'volvoxai_training_profile_y_backward_f32',
  'volvoxai_training_profile_y_f32',
  'volvoxai_training_quantize_weight_f32_to_i8',
  'volvoxai_training_reduce_backward_f32',
  'volvoxai_training_resize2d_backward_f32',
  'volvoxai_training_resize2d_f32',
  'volvoxai_training_rmsnorm_backward_f32',
  'volvoxai_training_scale_f32',
  'volvoxai_training_sdpa_backward_f32',
  'volvoxai_training_sdpa_f32',
  'volvoxai_training_sgd_update_f32',
  'volvoxai_training_slice_backward_f32',
  'volvoxai_training_slice_f32',
  'volvoxai_training_softmax_backward_f32',
  'volvoxai_training_softmax_f32',
  'volvoxai_training_spatial_softargmax_y_backward_f32',
  'volvoxai_training_spatial_softargmax_y_f32',
  'volvoxai_training_split_backward_f32',
  'volvoxai_training_sum_squares_f32',
  'volvoxai_training_transpose_backward_f32',
  'volvoxai_training_where_backward_f32',
  'volvoxai_training_where_f32',
  'volvoxai_training_zero_f32',
], {
  owner: 'portable-training',
  consumer: 'wasm-training',
  group: 'training',
  profiles: FULL_PROFILE,
  tsSignature: 'wasmFunction',
});

const PTQ_EXPORTS = entries([
  'volvoxai_ptq_abi_version',
  'volvoxai_ptq_capabilities',
  'volvoxai_ptq_observer_reset',
  'volvoxai_ptq_observer_observe_f32',
  'volvoxai_ptq_calculate_params',
  'volvoxai_ptq_quantize_f32',
  'volvoxai_ptq_pack_weight_i8',
  'volvoxai_ptq_pack_bias_i32',
], {
  owner: 'portable-ptq',
  consumer: 'wasm-ptq',
  group: 'ptq',
  profiles: FULL_PROFILE,
  tsSignature: 'wasmFunction',
});

const ABI_VERSIONS = [
  ['runtime', 'wasm_runtime_abi_version', 1, BOTH_PROFILES],
  ['scheduler', 'vx_scheduler_policy_abi_version', 1, BOTH_PROFILES],
  ['training', 'volvoxai_training_abi_version', 1, FULL_PROFILE],
  ['ptq', 'volvoxai_ptq_abi_version', 1, FULL_PROFILE],
].map(([id, exportName, value, profiles]) => ({ id, export: exportName, value, profiles }));

const PORTABLE_CONTROL_FORBIDDEN = [
  '^volvoxai_model_control_',
  '^vx_graph_domain_',
  '^vx_independent_batch_',
  '^vx_runtime_lifecycle_',
  '^vx_safetensors_header_',
  '^vx_shape_domain_',
];

const RELAXED_SIMD_CHILD = {
  section: 'volvoxai.relaxed_simd.v1',
  owner: 'portable-kernels',
  consumer: 'wasm-provider',
  group: 'relaxed-simd-child',
  profiles: BOTH_PROFILES,
  optional: true,
  memoryMode: 'imports-parent-memory',
  imports: [{
    module: 'env',
    name: 'memory',
    owner: 'release-parent',
    consumer: 'relaxed-simd-child',
    group: 'relaxed-simd-child',
    kind: 'memory',
    profiles: BOTH_PROFILES,
    tsSignature: 'memory',
  }],
  exports: entries(['qlinear_i8u8_relaxed'], {
    owner: 'portable-kernels',
    consumer: 'wasm-provider',
    group: 'relaxed-simd-child',
  }),
  probe: { export: 'qlinear_i8u8_relaxed', arity: 17, zeroArgumentsResult: 0 },
};

const PTQ_AUTHORING_CHILD = {
  section: 'volvoxai.ptq_authoring.v1',
  owner: 'portable-ptq-authoring',
  consumer: 'full-engine-host',
  group: 'ptq-authoring-child',
  profiles: FULL_PROFILE,
  optional: false,
  memoryMode: 'private-defined-memory',
  imports: [
    'vx_ptq_host_format_double',
    'vx_ptq_host_parse_double',
  ].map((name) => ({
    module: 'env',
    name,
    owner: 'browser-host',
    consumer: 'ptq-authoring-child',
    group: 'ptq-authoring-child',
    kind: 'function',
    profiles: FULL_PROFILE,
    tsSignature: 'wasmFunction',
  })),
  exports: [
    ...entries(['memory'], {
      owner: 'wasm-linker',
      consumer: 'ptq-authoring-child',
      group: 'ptq-authoring-child',
      kind: 'memory',
      profiles: FULL_PROFILE,
      tsSignature: 'memory',
    }),
    ...entries([
      'vx_ptq_wasm_abi_version',
      'vx_ptq_wasm_alloc',
      'vx_ptq_wasm_free',
      'vx_ptq_wasm_release',
      'vx_ptq_wasm_author',
      'vx_ptq_wasm_template',
      'vx_ptq_wasm_template_length',
      'vx_ptq_wasm_message',
      'vx_ptq_wasm_status',
      'vx_ptq_wasm_quantized_nodes',
      'vx_ptq_wasm_retained_float_nodes',
      'vx_ptq_wasm_observer_count',
      'vx_ptq_wasm_observer_tensor',
      'vx_ptq_wasm_observer_quantized',
      'vx_ptq_wasm_observer_storage',
      'vx_ptq_wasm_observer_scheme',
      'vx_ptq_wasm_layer_count',
      'vx_ptq_wasm_layer_kind',
      'vx_ptq_wasm_layer_node_index',
      'vx_ptq_wasm_layer_weight_axis',
      'vx_ptq_wasm_layer_name',
    ], {
      owner: 'portable-ptq-authoring',
      consumer: 'full-engine-host',
      group: 'ptq-authoring-child',
      profiles: FULL_PROFILE,
      tsSignature: 'wasmFunction',
    }),
  ],
  abiVersion: { export: 'vx_ptq_wasm_abi_version', value: 1 },
};

export const WASM_INTERNAL_ABI_MANIFEST = deepFreeze({
  format: WASM_INTERNAL_ABI_FORMAT,
  profiles: ['inference', 'full'],
  gpuBridge: GPU_BRIDGE_ABI,
  parentImports: [...HOST_IMPORTS, ...SYNURANG_CALL_IMPORTS, ...GPU_BRIDGE_IMPORTS],
  parentExports: [
    ...ALLOCATOR_EXPORTS,
    ...SYNURANG_CALL_EXPORTS,
    ...MODEL_CONTROL_SUPPORT_EXPORTS,
    ...GPU_CONTROL_EXPORTS,
    ...SCHEDULER_EXPORTS,
    ...FORWARD_EXPORTS,
    ...TRAINING_EXPORTS,
    ...PTQ_EXPORTS,
  ],
  abiVersions: ABI_VERSIONS,
  forbiddenProfilePatterns: {
    inference: [
      ...PORTABLE_CONTROL_FORBIDDEN,
      '^volvoxai_training_',
      '^volvoxai_ptq_',
      '^vx_training_control_',
      '^vx_wasm_prepare_gpu_',
    ],
    full: PORTABLE_CONTROL_FORBIDDEN,
  },
  children: {
    relaxedSimd: RELAXED_SIMD_CHILD,
    ptqAuthoring: PTQ_AUTHORING_CHILD,
  },
});
