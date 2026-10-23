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
  // Version 13 gives snapshot(0, 0, 0, observe) completion-only semantics:
  // poll the ticket with readback(ticket, 0, 0), then release it normally.
  version: 13,
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
    vx_gpu_memory_start: ['int', 'uint32_t observer', 'uint32_t capacity'],
    vx_gpu_memory_stop: ['void', 'uint32_t observer'],
    vx_gpu_begin: ['void'],
    vx_gpu_begin_activity: ['void'],
    vx_gpu_begin_trace: ['int', 'uint32_t capacity', 'int nodes'],
    vx_gpu_trace_node_begin: ['int'],
    vx_gpu_trace_node_end: ['void'],
    vx_gpu_trace_program_begin: ['int'],
    vx_gpu_trace_program_end: ['void'],
    vx_gpu_debug_label: ['int', 'uint32_t text', 'uint32_t bytes'],
    vx_gpu_trace_read: ['int', 'uint32_t ticket', 'uint32_t result'],
    vx_gpu_trace_release: ['void', 'uint32_t ticket'],
    vx_gpu_await_read: ['int', 'uint32_t ticket', 'uint32_t result'],
    vx_gpu_await_release: ['void', 'uint32_t ticket'],
    vx_gpu_encode: ['int', 'uint32_t dispatch'],
    vx_gpu_end: ['int'],
    vx_gpu_snapshot: ['int', 'uint32_t host_ptr', 'uint32_t offset', 'uint32_t bytes', 'int observe'],
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

const GPU_CONTROL_EXPORTS = entries(['vx_wasm_prepare_gpu_v1', 'vx_wasm_gpu_memory_event', 'vx_wasm_gpu_activity'], {
  owner: 'portable-runtime',
  consumer: 'model-control',
  group: 'gpu-control',
  profiles: FULL_PROFILE,
});

const ABI_VERSIONS = [
  ['runtime', 'wasm_runtime_abi_version', 1, BOTH_PROFILES],
].map(([id, exportName, value, profiles]) => ({ id, export: exportName, value, profiles }));

// Execution stays behind the generated service dispatch in both profiles.
const INTERNAL_EXECUTION_FORBIDDEN = [
  '^vx_scheduler_policy_',
  '^volvoxai_training_',
  '^volvoxai_ptq_',
  '^(?:matmul_f32|pack_q8_weight(?:_canonical)?|packed_q8_weight(?:_canonical)?_size)$',
  '^(?:qlinear|qbatch_matmul|qconv2d|qsdpa)_',
];

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
  ],
  abiVersions: ABI_VERSIONS,
  forbiddenProfilePatterns: {
    inference: [
      ...PORTABLE_CONTROL_FORBIDDEN,
      ...INTERNAL_EXECUTION_FORBIDDEN,
      '^vx_training_control_',
      '^vx_wasm_prepare_gpu_',
    ],
    full: [...PORTABLE_CONTROL_FORBIDDEN, ...INTERNAL_EXECUTION_FORBIDDEN],
  },
  children: {
    relaxedSimd: RELAXED_SIMD_CHILD,
    ptqAuthoring: PTQ_AUTHORING_CHILD,
  },
});
