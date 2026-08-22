# VolvoxAI — top-level task runner.
#
# The native C engine is built and tested by CMake + CTest (see CMakeLists.txt,
# native/CMakeLists.txt). This Makefile is a thin wrapper: it drives Docker for
# a reproducible toolchain, forwards native targets to cmake/ctest, and keeps
# the web (npm/esbuild), proto codegen, model export, and Rust FFI plugin targets
# that have their own toolchains. Run `make help` for the common entry points.

.PHONY: all help build_docker compile_shaders \
        build_native test_native test_native_all test_all test_js test_exporter \
        verify_native_isa benchmark_native test_native_gpu clean_native \
        build_native_task_cli test_native_task_cli \
        build_tiny_receipt_split_native_example test_tiny_receipt_split_native_example \
        build_wasm test_wasm_relaxed_simd benchmark_wasm_w8a8_seed \
        benchmark_wasm_qbatch_matmul build_web \
        publish_npm webtest release_git \
        parity parity_native_gpu parity_webgpu parity_webgpu_matrix \
        parity_native_gpu_matrix parity_gpu_consensus parity_backward \
        parity_decode parity_kvcache parity_kvcache_webgpu parity_goldens \
        parity_ops parity_graphs parity_portable parity_portable_webgpu \
        parity_coverage parity_image parity_gpu_required \
        models models_efficientdet models_tinystories models_deps models_clean \
        validate_model_packages \
        proto_codegen_fetch proto_codegen proto_codegen_check \
        proto_enum_codegen proto_enum_codegen_check \
        kernel_registry_codegen kernel_registry_codegen_check \
        optimizer_registry_codegen optimizer_registry_codegen_check \
        proto_all proto_check proto_c proto_c_lite proto_c_native proto_typescript \
        proto_rust build_ffi clean_ffi

VERSION := $(shell grep '"version"' package.json | head -n 1 | cut -d '"' -f 4)
GIT_COMMIT := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo -dirty)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
DOCKER_IMAGE := volvoxai-build:latest
# GNU make computes CURDIR from the directory it actually entered.  The
# inherited PWD can still name an older symlink target and mount a different
# checkout into Docker.
DOCKER_RUN := docker run --rm -v $(CURDIR):/workspace -w /workspace $(DOCKER_IMAGE)
PY := python3

# Native engine (CMake). Native compilation uses clang inside the pinned Docker
# toolchain for reproducibility.
CMAKE_BUILD_DIR ?= build/cmake
CMAKE_CONFIG ?= -DCMAKE_C_COMPILER=clang
NATIVE_BUILD_JOBS ?= $(shell nproc)
NATIVE_INCLUDE_FLAGS := -Inative/include -Inative/src -Inative/src/runtime \
                        -Inative/src/kernels -Inative/src/backends -Inative/third_party \
                        -Inative/third_party/xz-embedded

# Web/WASM artifacts.
WEB_DIST_DIR := dist/$(VERSION)
WEB_JS_ARTIFACTS := $(WEB_DIST_DIR)/volvoxai.js $(WEB_DIST_DIR)/volvoxai.min.js \
                    $(WEB_DIST_DIR)/volvoxai.full.js $(WEB_DIST_DIR)/volvoxai.full.min.js \
                    $(WEB_DIST_DIR)/volvoxai.wasm.js $(WEB_DIST_DIR)/volvoxai.wasm.min.js
WEB_WASM_ARTIFACTS := $(WEB_DIST_DIR)/volvoxai.wasm $(WEB_DIST_DIR)/volvoxai.full.wasm
WEB_ARTIFACTS := $(WEB_JS_ARTIFACTS) $(WEB_WASM_ARTIFACTS)
WASM_RELAXED_SECTION_NAME := volvoxai.relaxed_simd.v1
WASM_RELAXED_CHILD := build/wasm/$(WASM_RELAXED_SECTION_NAME).wasm
WASM_CC ?= clang-17
WASM_RELAXED_CC ?= $(WASM_CC)

all: build_docker

help:
	@echo "VolvoxAI targets:"
	@echo "  Native engine (CMake + CTest, in Docker):"
	@echo "    build_native        compile all native binaries"
	@echo "    test_native         native regression suite (ctest -L native)"
	@echo "    test_native_all     regression + opt-in examples"
	@echo "    verify_native_isa   W8A8 ISA codegen checks (x86)"
	@echo "    benchmark_native    microbenchmarks (not a gate)"
	@echo "    test_native_gpu     configured GPU tests (needs a real device)"
	@echo "  Opt-in native example binaries (-> examples/target/bin/):"
	@echo "    build_native_task_cli  volvoxai-tasks (generate/detect/classify/...)"
	@echo "    build_tiny_receipt_split_native_example  tiny_receipt_split_w8a8"
	@echo "  Full suite:"
	@echo "    test_js             JS/TS unit suite (npm test)"
	@echo "    test_exporter       hermetic exporter contract tests"
	@echo "    test_all            test_js + test_native_all"
	@echo "    parity_gpu_required fail-closed physical WebGPU + native capability campaign"
	@echo "  Web / FFI / models:"
	@echo "    build_wasm build_web publish_npm  build_ffi  models"
	@echo "    benchmark_wasm_qbatch_matmul  compare scalar/SIMD128 QBatchMatMul"
	@echo "    validate_model_packages  enforce canonical graph package names/formats"
	@echo "    proto_all proto_check"

build_docker:
	docker build \
		--build-arg VOLVOXAI_VERSION=$(VERSION) \
		--build-arg VOLVOXAI_GIT_COMMIT=$(GIT_COMMIT)$(GIT_DIRTY) \
		--build-arg VOLVOXAI_BUILD_DATE=$(BUILD_DATE) \
		-t $(DOCKER_IMAGE) .

# Compile WGSL -> native shader formats (naga). Kept here because the Rust FFI
# plugin build (build_ffi) consumes native/shaders; the CMake build runs
# the same script internally for the engine.
compile_shaders: build_docker
	$(DOCKER_RUN) ./tools/compile_shaders.sh
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) native/shaders

# --------------------------------------------------------------------------
# Native engine: thin cmake/ctest wrappers. The shader pipeline runs inside
# the CMake build; the whole thing runs as root in Docker (naga/cargo need it)
# and chowns the generated tree back afterwards.
# --------------------------------------------------------------------------
build_native: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) -j"$(NATIVE_BUILD_JOBS)"; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			native/volvoxai native/volvoxai-full 2>/dev/null || true'

test_native: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -L native --output-on-failure

# Native regression and opt-in example policies.
test_native_all: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -L 'native|example' --output-on-failure

verify_native_isa: build_native
	$(DOCKER_RUN) env CLANG="$(WASM_CC)" \
		ctest --test-dir $(CMAKE_BUILD_DIR) -L verify --output-on-failure

benchmark_native: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -L benchmark --output-on-failure

# Build with the pinned toolchain, then execute on the host so CTest can reach
# the configured physical GPU and display devices.
test_native_gpu: build_native
	ctest --test-dir $(CMAKE_BUILD_DIR) -L gpu --output-on-failure

clean_native:
	rm -rf $(CMAKE_BUILD_DIR)

# --------------------------------------------------------------------------
# Opt-in native example binaries. These link the full engine, so they build via
# the same CMake-in-Docker path as build_native, then are copied into the
# example tree at the documented examples/target/bin/<tool> paths. The examples/
# Makefile forwards here (native_task_cli and the TinyReceipt applications).
# --------------------------------------------------------------------------
EXAMPLES_BIN_DIR := examples/target/bin

build_native_task_cli: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target volvoxai-tasks -j"$$(nproc)"; \
		mkdir -p $(EXAMPLES_BIN_DIR); \
		cp $(CMAKE_BUILD_DIR)/native/volvoxai-tasks $(EXAMPLES_BIN_DIR)/volvoxai-tasks; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			$(EXAMPLES_BIN_DIR) 2>/dev/null || true'
	@echo "Built $(EXAMPLES_BIN_DIR)/volvoxai-tasks"

test_native_task_cli: build_native_task_cli
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -R '^test_native_task_cli$$' --output-on-failure

build_tiny_receipt_split_native_example: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target tiny_receipt_split_w8a8 test_tiny_receipt_split_w8a8 -j"$$(nproc)"; \
		mkdir -p $(EXAMPLES_BIN_DIR); \
		cp $(CMAKE_BUILD_DIR)/native/tiny_receipt_split_w8a8 $(EXAMPLES_BIN_DIR)/tiny_receipt_split_w8a8; \
		cp $(CMAKE_BUILD_DIR)/native/test_tiny_receipt_split_w8a8 $(EXAMPLES_BIN_DIR)/test_tiny_receipt_split_w8a8; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			$(EXAMPLES_BIN_DIR) 2>/dev/null || true'
	@echo "Built $(EXAMPLES_BIN_DIR)/tiny_receipt_split_w8a8"

test_tiny_receipt_split_native_example: build_tiny_receipt_split_native_example
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -R '^test_tiny_receipt_split_native_example$$' --output-on-failure

# JS/TS unit suite (CI runs `npm test` directly; this is the Make alias).
test_js:
	npm test

# Network-free exporter contracts. Core tests use the Python standard library;
# installed ONNX/SafeTensors dependencies enable the frontend fixture suite.
test_exporter:
	PYTHONDONTWRITEBYTECODE=1 $(PY) -m unittest discover -s tools/exporter/tests -p 'test_*.py'
	PYTHONDONTWRITEBYTECODE=1 $(PY) -m unittest discover \
		-s examples/receipt_digit_reader/tests -p 'test_*.py'
	PYTHONDONTWRITEBYTECODE=1 $(PY) -m unittest discover \
		-s examples/tiny_receipt_vqa/tests -p 'test_*.py'

# Full portable regression: exporter contracts + JS suite + native suite.
test_all: test_exporter test_js test_native_all

# --------------------------------------------------------------------------
# Cross-tier whole-model parity + benchmark harness (tests/parity/).
# Runs the two shipped models through CPU-JS (the oracle), WASM, and native-CPU
# and gates on numeric tolerance + decision parity. Requires the model packages
# (see docs/models.md), a built web bundle (npm run build:all), and the native
# binary. Policy-required tiers fail closed when their producer is unavailable.
parity:
	node tests/parity/run.mjs fixtures
	node tests/parity/run.mjs produce cpu-js
	node tests/parity/run.mjs produce wasm
	bash tests/parity/produce_native.sh native-cpu
	node tests/parity/run.mjs native-sig native-cpu
	node tests/parity/run.mjs external-begin
	-python3 tests/parity/external/onnx_oracle.py
	-python3 tests/parity/external/tinystories_torch_oracle.py
	node tests/parity/run.mjs external-sig
	node tests/parity/run.mjs compare

# Audit the currently declared native Vulkan/OpenGL whole-model capability. A
# policy-authorized compile rejection is accepted only with sealed registry,
# model, campaign, physical-adapter, and exact-reason evidence. Any registry
# promotion invalidates the skip and requires deliberate promotion of this
# producer and policy to required numerical execution.
parity_native_gpu:
	bash tests/parity/produce_native.sh native-vulkan
	bash tests/parity/produce_native.sh native-opengl
	node tests/parity/run.mjs native-sig native-vulkan native-opengl
	node tests/parity/run.mjs compare

# Real-GPU WebGPU parity via Deno's surfaceless WebGPU (drives a physical GPU with
# no display and no root). Headless Chrome can't do this on a bare server -- its
# Vulkan init needs WSI surface extensions (a display), so navigator.gpu falls back
# to SwiftShader (CPU). Run on a box with a real GPU; gates webgpu + wasm vs the
# pure-JS oracle. Override DENO=/path/to/deno if it is not on PATH.
DENO ?= deno
parity_webgpu:
	$(DENO) run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi --allow-run=git tests/parity/webgpu_deno.js
	node tests/parity/run.mjs compare

# WebGPU L1/L2 matrix producer: runs every authored op/graph package on a real GPU
# (Deno surfaceless WebGPU) and emits <id>.webgpu.json beside the other tiers. Run on
# a GPU box AFTER `make parity_ops`/`parity_graphs` authored the packages, then re-run
# opmatrix-compare/graphmatrix-compare to fold the webgpu column into the matrices.
parity_webgpu_matrix:
	node tests/parity/run.mjs matrix-begin ops webgpu
	$(DENO) run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi tests/parity/webgpu_matrix_deno.js tests/parity/out/opcases tests/parity/out/ops
	node tests/parity/run.mjs matrix-import ops webgpu
	node tests/parity/run.mjs matrix-begin graphs webgpu
	$(DENO) run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi tests/parity/webgpu_matrix_deno.js tests/parity/out/graphcases tests/parity/out/graphsigs
	node tests/parity/run.mjs matrix-import graphs webgpu
	node tests/parity/run.mjs opmatrix-compare
	node tests/parity/run.mjs graphmatrix-compare

# Future/manual native-GPU L1/L2 execution qualification. This target requires
# exporter-qualified public Vulkan/OpenGL routes; it is intentionally not part
# of parity_gpu_required while those generated registry sets are empty.
parity_native_gpu_matrix:
	set -e; for be in "native-vulkan --vulkan" "native-opengl --opengl"; do set -- $$be; \
	  node tests/parity/run.mjs matrix-begin ops $$1; \
	  node tests/parity/native_matrix.mjs tests/parity/out/opcases   tests/parity/out/ops       $$1 $$2; \
	  node tests/parity/run.mjs matrix-import ops $$1; \
	  node tests/parity/run.mjs matrix-begin graphs $$1; \
	  node tests/parity/native_matrix.mjs tests/parity/out/graphcases tests/parity/out/graphsigs $$1 $$2; \
	  node tests/parity/run.mjs matrix-import graphs $$1; \
	done
	node tests/parity/run.mjs opmatrix-compare
	node tests/parity/run.mjs graphmatrix-compare

# Future/manual EfficientDet GPU consensus diagnostic: webgpu (Deno) versus
# qualified native OpenGL and Vulkan, bit-identical for int8 and ~1e-6 for
# fp32. It is not release evidence until both native public routes complete
# bounded-domain/exporter qualification and the script seals their lifecycle
# no-fallback reports. Override DENO=/NODE= if needed.
parity_gpu_consensus:
	DENO=$(DENO) bash tests/parity/gpu_consensus_check.sh

# Backward + optimizer-step parity: forward + cross-entropy + backward + one SGD step
# per case (Linear, MLP, LayerNorm) on cpu-js and wasm, gated vs each other and vs a
# PyTorch autograd oracle (both gradients and updated weights). WebGPU is produced on a
# GPU box (`deno run … tests/parity/backward/run_backward.mjs webgpu`) and folds in.
parity_backward:
	node tests/parity/backward/run_backward.mjs
	-python3 tests/parity/backward/backward_torch_oracle.py
	node tests/parity/backward/run_backward.mjs compare

# Autoregressive decode-path parity: greedy-decode TinyStories on cpu-js and wasm and check
# the token sequence is identical across tiers and matches PyTorch's greedy generate.
# WebGPU is produced on a GPU box (`deno run … tests/parity/decode/decode_parity.mjs webgpu`).
parity_decode:
	node tests/parity/decode/decode_parity.mjs
	-python3 tests/parity/decode/decode_torch_oracle.py
	node tests/parity/decode/decode_parity.mjs compare

# KV-cache decode parity: drive a small W8A8 decoder through DecodeSession's required
# row/KV-cache path and compare retained self-attention K/V with a CPU-JS full recompute.
parity_kvcache:
	node tests/parity/kvcache/kvcache_parity.mjs cpu-js wasm
	node tests/parity/kvcache/kvcache_parity.mjs compare cpu-js wasm

# The test-only internals bundle the paged-KV campaign imports: a page table is
# not on the public surface, and ShaderLibrary imports .wgsl, which only
# esbuild's text loader resolves. Needs the npm dev dependencies, so it is built
# on a dev box and shipped to the GPU host the same way the shader packs are.
parity_paged_kv_bundle:
	node tools/build_parity_internals.mjs

# Paged KV on a physical WebGPU adapter: a deliberately scattered page table
# bit-compared against a contiguous CPU decode, plus two requests decoding
# concurrently on one context from a shared prompt page. Run on a physical-GPU
# host after `make parity_paged_kv_bundle` has produced the bundle.
parity_paged_kv_webgpu:
	$(DENO) run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi tests/parity/kvcache/paged_kv_webgpu.mjs

# The same exact-int8 gate including WebGPU device-resident K/V readback. Run on a
# physical-GPU host through Deno's surfaceless WebGPU implementation.
parity_kvcache_webgpu:
	$(DENO) run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi tests/parity/kvcache/kvcache_parity.mjs cpu-js wasm webgpu
	node tests/parity/kvcache/kvcache_parity.mjs compare cpu-js wasm webgpu

# One fail-closed dynamic-v1 hardware campaign for protected self-hosted CI. It
# requires physical WebGPU whole-model, L1/L2, portable-closure, and KV-cache
# execution. Native Vulkan/OpenGL are capability probes until their public routes
# are qualified: only the exact policy/registry-backed compile rejection is
# accepted, and it is reported separately from executed parity. Native matrices
# and cross-GPU consensus remain future/manual qualification targets.
parity_gpu_required:
	node tests/parity/run.mjs gpu-begin
	$(MAKE) parity
	$(MAKE) parity_ops
	$(MAKE) parity_graphs
	$(MAKE) parity_coverage
	$(DENO) run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi --allow-run=git tests/parity/webgpu_deno.js
	bash tests/parity/produce_native.sh native-vulkan
	bash tests/parity/produce_native.sh native-opengl
	node tests/parity/run.mjs native-sig native-vulkan native-opengl
	node tests/parity/run.mjs compare
	$(MAKE) parity_webgpu_matrix
	$(MAKE) parity_portable_webgpu
	$(MAKE) parity_kvcache_webgpu
	node tests/parity/run.mjs gpu-verify

# Regenerate committed golden signatures from the CPU-JS reference. Review any diff.
parity_goldens:
	node tests/parity/run.mjs golden

# Level 1: per-op cross-tier matrix. Each op runs as a single-node graph on
# CPU-JS/WASM/native-CPU (gated vs CPU-JS) plus a best-effort PyTorch oracle
# (correctness). Skipped ops and known approximations are declared in ops/cases.mjs.
parity_ops:
	node tests/parity/run.mjs opmatrix
	node tests/parity/run.mjs matrix-begin ops torch
	-python3 tests/parity/graphs/torch_interp.py tests/parity/out/opcases tests/parity/out/ops
	node tests/parity/run.mjs matrix-import ops torch
	node tests/parity/run.mjs opmatrix-compare

# Level 2: mixed multi-op graphs (fusion / aliasing / layout / dtype seams).
# Same gating as L1: WASM + native-CPU vs CPU-JS, best-effort PyTorch oracle.
parity_graphs:
	node tests/parity/run.mjs graphmatrix
	node tests/parity/run.mjs matrix-begin graphs torch
	-python3 tests/parity/graphs/torch_interp.py tests/parity/out/graphcases tests/parity/out/graphsigs
	node tests/parity/run.mjs matrix-import graphs torch
	node tests/parity/run.mjs graphmatrix-compare

# Generated-inventory-bound numerical closure for the audited portable gaps.
# CPU-JS, strict WASM, and strict native CPU are hard gates; physical WebGPU uses
# these same cases through the protected hardware matrix campaign.
parity_portable:
	node tests/parity/run.mjs portablematrix
	node tests/parity/run.mjs portablematrix-compare

# Re-author the portable closure packages, execute every case on a required
# physical WebGPU adapter, seal the artifacts, and compare all four providers.
# Select a Linux adapter with DRI_PRIME/DENO_WEBGPU_BACKEND when necessary.
parity_portable_webgpu:
	node tests/parity/run.mjs portablematrix
	node tests/parity/run.mjs matrix-begin portable webgpu
	$(DENO) run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi tests/parity/webgpu_matrix_deno.js tests/parity/out/portablecases tests/parity/out/portable
	node tests/parity/run.mjs matrix-import portable webgpu
	node tests/parity/run.mjs portablematrix-compare

# Operator coverage vs generated kernel registry: how much of VolvoxAI is parity-tested.
parity_coverage:
	node tests/parity/run.mjs coverage

# Real-image EfficientDet parity: decode dog.jpg/cat.jpg once, feed identical input
# to VolvoxAI cpu-js/wasm/native + ONNX Runtime, compare all 19206 anchors + detections.
parity_image:
	node tests/parity/image_check.mjs examples/efficientdet_lite0/assets/dog.jpg int8
	node tests/parity/image_check.mjs examples/efficientdet_lite0/assets/cat.jpg int8
	node tests/parity/image_check.mjs examples/efficientdet_lite0/assets/dog.jpg fp32

release_git: build_native
	@echo "Creating git release..."
	# Example: gh release create v$(VERSION) ./native/volvoxai -t "Release v$(VERSION)"
	@echo "Git release automation should be added here."

# --------------------------------------------------------------------------
# Web build (JS/WASM/WGSL).
# --------------------------------------------------------------------------
build_wasm: build_docker
	$(DOCKER_RUN) mkdir -p $(WEB_DIST_DIR) $(dir $(WASM_RELAXED_CHILD))
	$(DOCKER_RUN) rm -f $(WEB_WASM_ARTIFACTS)
	$(DOCKER_RUN) $(WASM_RELAXED_CC) --target=wasm32 -std=c11 -O3 \
		-Wall -Wextra -Werror -msimd128 -mrelaxed-simd -nostdlib \
		-ffreestanding -Wl,--no-entry -Wl,--import-memory -Wl,--strip-all \
		-o $(WASM_RELAXED_CHILD) \
		native/src/kernels/qlinear_w8a8_wasm_relaxed.c
	$(DOCKER_RUN) $(WASM_CC) --target=wasm32 -O3 -msimd128 -nostdlib \
		-Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
		-o $(WEB_DIST_DIR)/volvoxai.wasm native/src/kernels/kernels.c
	$(DOCKER_RUN) $(WASM_CC) --target=wasm32 -O3 -msimd128 -nostdlib \
		-Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
		$(NATIVE_INCLUDE_FLAGS) \
		-o $(WEB_DIST_DIR)/volvoxai.full.wasm \
		native/src/kernels/kernels.c native/src/kernels/training_kernels.c \
		native/src/training/quantization.c
	$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py \
		--section-name $(WASM_RELAXED_SECTION_NAME) \
		--payload $(WASM_RELAXED_CHILD) \
		--input $(WEB_DIST_DIR)/volvoxai.wasm \
		--output $(WEB_DIST_DIR)/volvoxai.wasm
	$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py \
		--section-name $(WASM_RELAXED_SECTION_NAME) \
		--payload $(WASM_RELAXED_CHILD) \
		--input $(WEB_DIST_DIR)/volvoxai.full.wasm \
		--output $(WEB_DIST_DIR)/volvoxai.full.wasm
	$(DOCKER_RUN) chmod 0644 $(WEB_WASM_ARTIFACTS)
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) $(WEB_DIST_DIR)
	@echo "Built $(WEB_WASM_ARTIFACTS)"

test_wasm_relaxed_simd: build_wasm
	$(DOCKER_RUN) node --test tests/wasm_w8a32_packed_simd.test.mjs
	$(DOCKER_RUN) node tools/test_wasm_relaxed_simd.mjs --baseline-only \
		$(WEB_WASM_ARTIFACTS)
	$(DOCKER_RUN) node --experimental-wasm-relaxed-simd \
		tools/test_wasm_relaxed_simd.mjs $(WEB_WASM_ARTIFACTS)

benchmark_wasm_w8a8_seed: build_wasm
	$(DOCKER_RUN) node examples/tiny_receipt_vqa/tools/benchmark_wasm_w8a8_seed.mjs \
		$(WEB_DIST_DIR)/volvoxai.wasm

benchmark_wasm_qbatch_matmul: build_wasm
	$(DOCKER_RUN) node tools/benchmark_wasm_qbatch_matmul.mjs \
		$(WEB_DIST_DIR)/volvoxai.wasm

build_web: build_docker build_wasm
	$(DOCKER_RUN) npm install
	$(DOCKER_RUN) npm run build:all
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) $(WEB_DIST_DIR)
	@echo "Built $(WEB_ARTIFACTS)"

# Publish NPM from the already-built dist artifacts.
publish_npm:
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm not found in PATH."; exit 1; }
	@set -e; \
	for f in $(WEB_ARTIFACTS); do \
		if [ ! -f "$$f" ]; then \
			echo "Error: $$f not found."; \
			echo "Build web artifacts first: make build_web"; \
			exit 1; \
		fi; \
	done; \
	unexpected=$$(node -e "const f=require('node:fs'),d='$(WEB_DIST_DIR)';const keep=new Set(['volvoxai.js','volvoxai.min.js','volvoxai.full.js','volvoxai.full.min.js','volvoxai.wasm.js','volvoxai.wasm.min.js','volvoxai.wasm','volvoxai.full.wasm']);console.log(f.readdirSync(d).filter(x=>!keep.has(x)).map(x=>d+'/'+x).join('\\n'))"); \
	if [ -n "$$unexpected" ]; then \
		echo "Error: unexpected files remain in $(WEB_DIST_DIR)/:"; \
		echo "$$unexpected"; \
		echo "Run npm run build:all before publishing."; \
		exit 1; \
	fi; \
	if ! npm whoami >/dev/null 2>&1; then \
		echo "Error: npm is not authenticated. Use npm login or provide the existing npm auth environment."; \
		exit 1; \
	fi; \
	echo "Package contents:"; \
	npm pack --dry-run; \
	echo "Publishing volvoxai@$(VERSION) to npm..."; \
	npm publish --access public; \
	echo "npm publish complete."

webtest: build_web
	@echo "Starting local web server at http://localhost:8085"
	@echo "Open http://localhost:8085/examples/efficientdet_lite0.html in your browser"
	python3 -m http.server 8085

# --------------------------------------------------------------------------
# Models: the models/ directory is .gitignored (large weights), so regenerate
# it from public sources. These run on the HOST (need Python ML deps the build
# image does not carry). Install them once with `make models_deps`.
# --------------------------------------------------------------------------
models_deps:
	$(PY) -m pip install -r tools/requirements-export.txt
	$(PY) -m pip install -r examples/tinystories/requirements-export.txt

models: models_efficientdet models_tinystories

validate_model_packages:
	node tools/validate_model_packages.mjs \
		models/efficientdet_lite0_int8 models/efficientdet_lite0_fp16 \
		models/efficientdet_lite0_fp32 models/tinystories_1m

models_efficientdet:
	PY=$(PY) bash ./examples/efficientdet_lite0/tools/fetch_model.sh

models_tinystories:
	PY=$(PY) bash ./examples/tinystories/tools/fetch_model.sh

models_clean:
	rm -rf models/efficientdet_lite0_int8 models/efficientdet_lite0_fp16 \
	       models/efficientdet_lite0_fp32 models/tinystories_1m

# --------------------------------------------------------------------------
# Synurang in-process FFI plugin over the full C lifecycle (Rust runtime).
# Runs on the HOST. Needs Python 3 and protoc; the Rust plugin also needs a
# Rust toolchain and libvulkan-dev. See docs for PROTOC_GEN_SYNURANG_FFI.
# --------------------------------------------------------------------------
SYNURANG_CODEGEN_TOOL     := tools/generate_proto.py
PROTO_ENUM_CODEGEN_TOOL   := tools/generate_proto_enums.py
KERNEL_REGISTRY_CODEGEN_TOOL := tools/generate_kernel_registry.py
OPTIMIZER_REGISTRY_CODEGEN_TOOL := tools/generate_optimizer_registry.py
SYNURANG_CODEGEN_CACHE    ?= build/cache/synurang
SYNURANG_CODEGEN_TARGET   ?=
PROTOC                    ?= protoc
PROTOC_GEN_SYNURANG_FFI   ?=
PROTO_DIR                 := proto
PROTO_SOURCE              := $(PROTO_DIR)/volvoxai.proto
PROTO_C_GEN_DIR           := runtime/generated/c
PROTO_TYPESCRIPT_GEN_DIR  := runtime/generated/typescript
PROTO_RUST_GEN_DIR        := runtime/src/gen
RUNTIME_SHARED_LIBRARY    := $(if $(filter Darwin,$(shell uname -s)),libvolvoxai.dylib,libvolvoxai.so)

SYNURANG_GENERATOR_ARG = $(if $(strip $(PROTOC_GEN_SYNURANG_FFI)),--generator "$(PROTOC_GEN_SYNURANG_FFI)")
SYNURANG_TARGET_ARG = $(if $(strip $(SYNURANG_CODEGEN_TARGET)),--target "$(SYNURANG_CODEGEN_TARGET)")
SYNURANG_CODEGEN = $(PY) $(SYNURANG_CODEGEN_TOOL) \
	--protoc "$(PROTOC)" \
	--cache-dir "$(SYNURANG_CODEGEN_CACHE)" \
	--proto "$(PROTO_SOURCE)" --proto-root "$(PROTO_DIR)" \
	--c-out "$(PROTO_C_GEN_DIR)" \
	--typescript-out "$(PROTO_TYPESCRIPT_GEN_DIR)" \
	--rust-out "$(PROTO_RUST_GEN_DIR)" \
	$(SYNURANG_GENERATOR_ARG) $(SYNURANG_TARGET_ARG)

proto_codegen_fetch:
	$(SYNURANG_CODEGEN) --fetch-only

proto_codegen:
	$(SYNURANG_CODEGEN)
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)
	$(PY) $(KERNEL_REGISTRY_CODEGEN_TOOL)
	$(PY) $(OPTIMIZER_REGISTRY_CODEGEN_TOOL)

proto_codegen_check:
	$(SYNURANG_CODEGEN) --check
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL) --check
	$(PY) $(KERNEL_REGISTRY_CODEGEN_TOOL) --check
	$(PY) $(OPTIMIZER_REGISTRY_CODEGEN_TOOL) --check

proto_enum_codegen:
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)

proto_enum_codegen_check:
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL) --check

kernel_registry_codegen:
	$(PY) $(KERNEL_REGISTRY_CODEGEN_TOOL)

kernel_registry_codegen_check:
	$(PY) $(KERNEL_REGISTRY_CODEGEN_TOOL) --check

optimizer_registry_codegen:
	$(PY) $(OPTIMIZER_REGISTRY_CODEGEN_TOOL)

optimizer_registry_codegen_check:
	$(PY) $(OPTIMIZER_REGISTRY_CODEGEN_TOOL) --check

proto_all: proto_codegen

proto_check: proto_codegen_check

proto_c:
	$(SYNURANG_CODEGEN) --language c-lite,c-native
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)

proto_c_lite:
	$(SYNURANG_CODEGEN) --language c-lite
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)

proto_c_native:
	$(SYNURANG_CODEGEN) --language c-native
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)

proto_typescript:
	$(SYNURANG_CODEGEN) --language typescript
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)

proto_rust:
	$(SYNURANG_CODEGEN) --language rust
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)

build_ffi: compile_shaders proto_codegen
	cd runtime && cargo build --release
	@echo "Built runtime/target/release/$(RUNTIME_SHARED_LIBRARY)"

clean_ffi:
	rm -rf $(PROTO_RUST_GEN_DIR) runtime/target
