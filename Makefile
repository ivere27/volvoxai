# VolvoxAI — top-level task runner.
#
# The native C engine is built by CMake (see CMakeLists.txt and
# native/CMakeLists.txt). This Makefile is a thin wrapper: it drives Docker for
# a reproducible toolchain, forwards native build targets to CMake, and keeps
# the web (npm/esbuild), proto codegen, and model export targets
# that have their own toolchains. Run `make help` for the common entry points.

.PHONY: all help build_docker compile_shaders \
        build_native build_native_profiles check_native_cuda_ptx test_native test_native_cpu_selection \
        test_native_invariants test_native_gpu_static verify_native_isa \
        benchmark_native_w8a8_release clean_native \
        test_contracts test_onnx_oracle test_quantized_oracle test_model_corpus \
        build_native_libraries test_python test_ptq_golden build_native_ptq_test \
        test_wasm_ptq \
        build_native_task_cli \
        build_wasm wasm_abi_codegen wasm_abi_codegen_check \
        shader_catalog_codegen shader_catalog_codegen_check \
        size_report size_check benchmark_wasm_w8a8_prefill \
        benchmark_wasm_qbatch_matmul build_web verify_release \
        publish_npm release_git \
        models models_efficientdet models_tinystories models_deps models_clean \
        validate_model_packages \
        proto_codegen_fetch proto_codegen proto_codegen_check \
        proto_enum_codegen proto_enum_codegen_check \
        operator_vocabulary_codegen operator_vocabulary_codegen_check \
        operator_param_registry_codegen operator_param_registry_codegen_check \
        kernel_registry_codegen kernel_registry_codegen_check \
        optimizer_registry_codegen optimizer_registry_codegen_check \
        api_conformance \

VERSION := $(shell grep '"version"' package.json | head -n 1 | cut -d '"' -f 4)
GIT_COMMIT := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo -dirty)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
DOCKER_IMAGE := volvoxai-build:latest
SYNURANG_CODEGEN_CACHE := build/cache/synurang-codegen
SYNURANG_OFFLINE ?= 0
SYNURANG_OFFLINE_ARG = $(if $(filter 1 true TRUE yes YES on ON,$(SYNURANG_OFFLINE)),--offline)
# GNU make computes CURDIR from the directory it actually entered.  The
# inherited PWD can still name an older symlink target and mount a different
# checkout into Docker.
# The bind mount is intentionally the repository selected by CURDIR.  Trust
# that exact mount for evidence-producing builds that query the Git snapshot;
# the container runs as root while the host checkout normally does not.
DOCKER_RUN := docker run --rm \
	-e GIT_CONFIG_COUNT=1 \
	-e GIT_CONFIG_KEY_0=safe.directory \
	-e GIT_CONFIG_VALUE_0=/workspace \
	-v $(CURDIR):/workspace -w /workspace $(DOCKER_IMAGE)
PY := python3

# Native engine (CMake). Native compilation uses clang inside the pinned Docker
# toolchain for reproducibility.
CMAKE_BUILD_DIR ?= build/cmake
CMAKE_CONFIG ?= -DCMAKE_C_COMPILER=clang
override CMAKE_EVIDENCE_CONFIG := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
NATIVE_BUILD_JOBS ?= $(shell nproc)
ONNX_ORACLE_REPORT ?= build/test-reports/onnx-oracle.json
QUANTIZED_ORACLE_REPORT ?= build/test-reports/quantized-oracle.json
MODEL_CORPUS_REPORT ?= build/test-reports/model-corpus.json
NATIVE_W8A8_PERFORMANCE_BLOCKS ?= 4
NATIVE_W8A8_PERFORMANCE_MAX_BLOCKS ?= 40
NATIVE_W8A8_PERFORMANCE_MIN_TIMED_MS ?= 50
NATIVE_W8A8_PERFORMANCE_REPORT ?= build/performance/native-w8a8-release.json

# Web/WASM artifacts.
WEB_DIST_DIR := dist/$(VERSION)
WEB_JS_ARTIFACTS := $(WEB_DIST_DIR)/volvoxai.js $(WEB_DIST_DIR)/volvoxai.min.js \
                    $(WEB_DIST_DIR)/volvoxai.full.js $(WEB_DIST_DIR)/volvoxai.full.min.js
WEB_WASM_ARTIFACTS := $(WEB_DIST_DIR)/volvoxai.wasm $(WEB_DIST_DIR)/volvoxai.full.wasm
WEB_ARTIFACTS := $(WEB_JS_ARTIFACTS) $(WEB_WASM_ARTIFACTS)
NATIVE_RELEASE_ARTIFACTS := native/volvoxai native/volvoxai-full
NATIVE_DEBUG_ARTIFACTS := native/.debug/volvoxai.debug \
                          native/.debug/volvoxai-full.debug
NATIVE_DEBUG_EVIDENCE := native/.debug/volvoxai.debug.json \
                         native/.debug/volvoxai-full.debug.json
# Shared and static forms of both profiles. The .so file names carry a version
# suffix and a pair of symlinks, so match on the stem.
NATIVE_LIBRARY_ARTIFACTS := $(wildcard native/libvolvoxai*.so*) \
                            $(wildcard native/libvolvoxai*.a)
WASM_PROVENANCE_SECTION_NAME := volvoxai.release.provenance.v1
WASM_PROVENANCE_DIR := build/wasm/provenance
WASM_INFERENCE_PROVENANCE := $(WASM_PROVENANCE_DIR)/volvoxai.wasm.json
WASM_FULL_PROVENANCE := $(WASM_PROVENANCE_DIR)/volvoxai.full.wasm.json
WASM_INFERENCE_BUILD_EVIDENCE := $(WASM_PROVENANCE_DIR)/volvoxai.wasm.build.json
WASM_FULL_BUILD_EVIDENCE := $(WASM_PROVENANCE_DIR)/volvoxai.full.wasm.build.json
# The data-only object graph, per-object optimization, and pinned toolchain live
# only in release_profiles.mjs. Make invokes the one builder instead of
# duplicating source and flag lists.
override WASM_RELEASE_BUILDER := node tools/build_wasm_release.mjs
override WASM_RELEASE_OBJECT_ROOT := build/wasm/release

all: build_docker

help:
	@echo "VolvoxAI targets:"
	@echo "  Native engine (CMake, in Docker):"
	@echo "    build_native        compile all native binaries"
	@echo "    build_native_profiles  compile only inference/full release binaries"
	@echo "    check_native_cuda_ptx verify built PTX matches required host registries"
	@echo "    test_native         selection + inference invariants + ISA gates"
	@echo "    test_native_cpu_selection  build and test portable-C CPU selection"
	@echo "    test_native_invariants  W8A8/platform/packed runtime contracts"
	@echo "    test_native_gpu_static  device-free GPU dispatch contract"
	@echo "    verify_native_isa   per-clamp runtime parity + x86/Linux codegen checks"
	@echo "    benchmark_native_w8a8_release  <=2% selective-profile throughput gate"
	@echo "  Opt-in native example binaries (-> examples/target/bin/):"
	@echo "    build_native_task_cli  volvoxai-tasks (generate/detect/classify/...)"
	@echo "  Native libraries (source build -> native/):"
	@echo "    build_native_libraries  libvolvoxai[-full].so/.a — the public ABI"
	@echo "    test_python            drive the .so from Python over Synurang FFI"
	@echo "    test_ptq_golden        byte-level PTQ authoring reference"
	@echo "    test_wasm_ptq          same template from WebAssembly and native"
	@echo "  Web / FFI / models:"
	@echo "    build_wasm build_web verify_release publish_npm  models"
	@echo "    wasm_abi_codegen_check  verify generated private WASM ABI projections"
	@echo "    size_report size_check  inspect and gate the fixed release inventory"
	@echo "    test_contracts test_onnx_oracle  required semantic gates"
	@echo "    test_quantized_oracle  exact project QLinear boundary gate"
	@echo "    verify_release     rebuild and gate all eight fixed release artifacts"
	@echo "    benchmark_wasm_qbatch_matmul  compare scalar/SIMD128 QBatchMatMul"
	@echo "    validate_model_packages  enforce canonical graph package names/formats"
	@echo "    proto_codegen_fetch proto_codegen proto_codegen_check"

build_docker:
	docker build \
		--build-arg VOLVOXAI_VERSION=$(VERSION) \
		--build-arg VOLVOXAI_GIT_COMMIT=$(GIT_COMMIT)$(GIT_DIRTY) \
		--build-arg VOLVOXAI_BUILD_DATE=$(BUILD_DATE) \
		-t $(DOCKER_IMAGE) .

# Compile WGSL -> native shader formats (naga). The CMake build runs the same
# script internally for the engine; this target exists so shaders can be
# refreshed without a full native build.
compile_shaders: build_docker
	$(DOCKER_RUN) ./tools/compile_shaders.sh
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) native/shaders

# --------------------------------------------------------------------------
# Native engine: thin CMake wrappers. The shader pipeline runs inside
# the CMake build; the whole thing runs as root in Docker (naga/cargo need it)
# and chowns the generated tree back afterwards.
# --------------------------------------------------------------------------
build_native: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) -j"$(NATIVE_BUILD_JOBS)"; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			$(NATIVE_RELEASE_ARTIFACTS) native/.debug 2>/dev/null || true'

build_native_profiles: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target volvoxai volvoxai-full \
			-j"$(NATIVE_BUILD_JOBS)"; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			$(NATIVE_RELEASE_ARTIFACTS) native/.debug 2>/dev/null || true'

# The distributable libraries. proto/volvoxai.proto is the whole public API and
# Synurang_GetApi module dispatch is its entry point for every host.
build_native_libraries: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target \
			volvoxai_shared volvoxai_static \
			volvoxai-full_shared volvoxai-full_static \
			-j"$(NATIVE_BUILD_JOBS)"; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			$(NATIVE_LIBRARY_ARTIFACTS) 2>/dev/null || true'

# Drives source-built modules through the pinned Synurang Python call host.
# The host is vendored; python/requirements-test.txt supplies test fixtures.
test_python: build_native_libraries
	@PYTHONPATH="$(CURDIR)/python" $(PY) -c 'from volvoxai import ModuleHost' \
		|| { echo "Error: install python/requirements-test.txt"; exit 1; }
	$(PY) -m unittest discover -s python/tests -p 'test_*.py' -v

# The byte-level reference the C port of PTQ authoring has to meet. It needs no
# library, so it runs on its own and stays fast enough to run often.
test_ptq_golden:
	$(PY) -m unittest discover -s python/tests -p 'test_ptq_*.py' -v

build_native_ptq_test: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target test_ptq_authoring \
			-j"$(NATIVE_BUILD_JOBS)"; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) 2>/dev/null || true'

test_wasm_ptq: build_wasm build_native_ptq_test
	$(DOCKER_RUN) npm run test:wasm-ptq

check_native_cuda_ptx:
	$(PY) tools/check_native_cuda_ptx.py --build-dir "$(CMAKE_BUILD_DIR)"

test_native_cpu_selection: build_native
	$(DOCKER_RUN) npm run test:native-cpu-selection -- \
		--native-build-dir="$(CMAKE_BUILD_DIR)"

test_native_invariants: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) \
		-L native-invariant -LE native-isa-runtime --output-on-failure

test_native_gpu_static: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) \
		-L native-gpu-static --output-on-failure

verify_native_isa: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) \
		-L 'native-isa-runtime|native-isa-codegen' --output-on-failure

test_native: test_native_cpu_selection test_native_invariants verify_native_isa
	$(DOCKER_RUN) native/cmake/check_profile_boundaries.sh \
		$(CMAKE_BUILD_DIR)/native/release-link/volvoxai \
		$(CMAKE_BUILD_DIR)/native/release-link/volvoxai-full

# Compare the release selective hot/cold graph against a current-source all-O3
# graph. The private comparison executables link without relinking, finalizing,
# or publishing the release binary.
# The runner writes failure evidence before returning a non-zero gate status.
benchmark_native_w8a8_release: build_native_profiles
	$(DOCKER_RUN) npm run test:native-w8a8-release
	$(DOCKER_RUN) bash -c 'set +e; \
		CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" \
			npm run baseline:native-w8a8-release -- \
			--native-build-dir="$(CMAKE_BUILD_DIR)" \
			--blocks="$(NATIVE_W8A8_PERFORMANCE_BLOCKS)" \
			--max-blocks="$(NATIVE_W8A8_PERFORMANCE_MAX_BLOCKS)" \
			--min-timed-ms="$(NATIVE_W8A8_PERFORMANCE_MIN_TIMED_MS)" \
			--output="$(NATIVE_W8A8_PERFORMANCE_REPORT)"; \
		gate_status=$$?; \
		chown -R $(HOST_UID):$(HOST_GID) \
			$(CMAKE_BUILD_DIR) \
			$(NATIVE_W8A8_PERFORMANCE_REPORT) 2>/dev/null || true; \
		exit $$gate_status'

test_contracts:
	$(PY) -m unittest discover -s tools/tests -p 'test_*.py' -v

test_onnx_oracle:
	$(PY) tests/parity/external/onnx_oracle.py \
		--bundle $(WEB_DIST_DIR)/volvoxai.js \
		--wasm $(WEB_DIST_DIR)/volvoxai.wasm \
		--native native/volvoxai \
		--report $(ONNX_ORACLE_REPORT)

test_model_corpus:
	$(PY) tests/parity/external/model_corpus.py \
		--bundle $(WEB_DIST_DIR)/volvoxai.js \
		--wasm $(WEB_DIST_DIR)/volvoxai.wasm \
		--native native/volvoxai \
		--report $(MODEL_CORPUS_REPORT)

# The device bridge on real hardware.
#
# Deliberately not part of `make test`: it needs a GPU, and CI has none. The
# CPU-interpreted proof in tests/gpu_bridge_execution.test.mjs is what runs
# everywhere; this is what says the WGSL the engine named is right. DENO and
# WEBGPU_HOST let it run against a remote box.
DENO ?= deno
WEBGPU_BRIDGE_ENTRY ?= build/test-reports/webgpu_device_bridge_entry.mjs

$(WEBGPU_BRIDGE_ENTRY): tests/parity/external/webgpu_device_bridge_entry.ts
	@mkdir -p $(dir $@)
	npx esbuild $< --bundle --format=esm --platform=neutral --target=es2022 \
		--outfile=$@

# Generated codecs resolve submessages by their full protobuf name, including
# when a bundler renames JavaScript classes.
# Cases cut out of an exported model, for operators whose graph is hard to
# author by hand. A node the engine already runs is valid by construction, so
# these are extracted rather than written. Missing models leave the case out
# and the suite simply reports fewer cases.
WEBGPU_CASE_DIR := build/test-reports/cases

# A dense convolution and a depthwise one take different paths through the
# same shader -- the second derives its per-group input extent by division --
# so each gets its own case.
WEBGPU_CASES := $(WEBGPU_CASE_DIR)/QConv2D.graph.json \
	$(WEBGPU_CASE_DIR)/QConv2DDepthwise.graph.json

$(WEBGPU_CASE_DIR)/QConv2D.graph.json: tests/parity/external/extract_operator_case.py
	$(PY) $< --model models/efficientdet_lite0_int8 --op QConv2D \
		--out $(WEBGPU_CASE_DIR)

$(WEBGPU_CASE_DIR)/QConv2DDepthwise.graph.json: tests/parity/external/extract_operator_case.py
	$(PY) $< --model models/efficientdet_lite0_int8 --op QConv2D \
		--param groups=64 --name QConv2DDepthwise --out $(WEBGPU_CASE_DIR)

test_webgpu_device_bridge: $(WEBGPU_BRIDGE_ENTRY) $(WEBGPU_CASES)
	$(DENO) run --unstable-webgpu --allow-read --allow-env --allow-ffi \
		tests/parity/external/webgpu_device_bridge.mjs \
		--bundle $(WEBGPU_BRIDGE_ENTRY) \
		--wasm $(WEB_DIST_DIR)/volvoxai.full.wasm \
		--cases $(WEBGPU_CASE_DIR)

WEBGPU_COMPOSED_ENTRY ?= build/test-reports/webgpu_composed_entry.mjs

$(WEBGPU_COMPOSED_ENTRY): tests/parity/external/webgpu_composed_entry.ts
	@mkdir -p $(dir $@)
	npx esbuild $< --bundle --format=esm --platform=neutral --target=es2022 \
		--outfile=$@

# The browser `full` profile compiles through provider names. This asserts the
# `webgpu` name lands on the engine's own planner -- values against the host
# route, and the route evidence separately, because matching values alone
# would also come back from a provider that never reached the device.
test_webgpu_composed_runtime: $(WEBGPU_COMPOSED_ENTRY)
	$(DENO) run --unstable-webgpu --allow-read --allow-env --allow-ffi \
		tests/parity/external/webgpu_composed_runtime.mjs \
		--bundle $(WEBGPU_COMPOSED_ENTRY) \
		--wasm $(WEB_DIST_DIR)/volvoxai.full.wasm

test_quantized_oracle:
	$(PY) tests/parity/external/quantized_oracle.py \
		--bundle $(WEB_DIST_DIR)/volvoxai.js \
		--wasm $(WEB_DIST_DIR)/volvoxai.wasm \
		--native native/volvoxai \
		--report $(QUANTIZED_ORACLE_REPORT)

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
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target volvoxai-tasks -j"$$(nproc)"; \
		mkdir -p $(EXAMPLES_BIN_DIR); \
		cp $(CMAKE_BUILD_DIR)/native/volvoxai-tasks $(EXAMPLES_BIN_DIR)/volvoxai-tasks; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			$(EXAMPLES_BIN_DIR) 2>/dev/null || true'
	@echo "Built $(EXAMPLES_BIN_DIR)/volvoxai-tasks"

# --------------------------------------------------------------------------
release_git: build_native
	@echo "Creating git release..."
	# Example: gh release create v$(VERSION) ./native/volvoxai -t "Release v$(VERSION)"
	@echo "Git release automation should be added here."

# --------------------------------------------------------------------------
# Web build (JS/WASM/WGSL).
# --------------------------------------------------------------------------
shader_catalog_codegen:
	node tools/generate_shader_catalog.mjs
	node tools/generate_webgpu_dispatch.mjs

shader_catalog_codegen_check:
	node tools/generate_shader_catalog.mjs --check
	node tools/generate_webgpu_dispatch.mjs --check

wasm_abi_codegen:
	node tools/generate_wasm_internal_abi.mjs

wasm_abi_codegen_check:
	node tools/generate_wasm_internal_abi.mjs --check

build_wasm: build_docker wasm_abi_codegen_check shader_catalog_codegen_check
	$(DOCKER_RUN) mkdir -p $(WEB_DIST_DIR) $(WASM_PROVENANCE_DIR)
	$(DOCKER_RUN) rm -f $(WEB_WASM_ARTIFACTS) \
		$(WEB_DIST_DIR)/volvoxai.ptq.wasm \
		$(WASM_INFERENCE_PROVENANCE) $(WASM_FULL_PROVENANCE) \
		$(WASM_INFERENCE_BUILD_EVIDENCE) $(WASM_FULL_BUILD_EVIDENCE)
	$(DOCKER_RUN) $(WASM_RELEASE_BUILDER) all \
		--output-dir $(WEB_DIST_DIR) \
		--build-root $(WASM_RELEASE_OBJECT_ROOT) \
		--evidence-dir $(WASM_PROVENANCE_DIR)
	$(DOCKER_RUN) node tools/write_wasm_provenance.mjs write \
		--artifact $(WEB_DIST_DIR)/volvoxai.wasm \
		--output $(WASM_INFERENCE_PROVENANCE) \
		--build-evidence $(WASM_INFERENCE_BUILD_EVIDENCE)
	$(DOCKER_RUN) node tools/write_wasm_provenance.mjs write \
		--artifact $(WEB_DIST_DIR)/volvoxai.full.wasm \
		--output $(WASM_FULL_PROVENANCE) \
		--build-evidence $(WASM_FULL_BUILD_EVIDENCE)
	$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py \
		--section-name $(WASM_PROVENANCE_SECTION_NAME) \
		--payload $(WASM_INFERENCE_PROVENANCE) \
		--input $(WEB_DIST_DIR)/volvoxai.wasm \
		--output $(WEB_DIST_DIR)/volvoxai.wasm
	$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py \
		--section-name $(WASM_PROVENANCE_SECTION_NAME) \
		--payload $(WASM_FULL_PROVENANCE) \
		--input $(WEB_DIST_DIR)/volvoxai.full.wasm \
		--output $(WEB_DIST_DIR)/volvoxai.full.wasm
	$(DOCKER_RUN) node tools/write_wasm_provenance.mjs check \
		--artifact $(WEB_DIST_DIR)/volvoxai.wasm \
		--build-evidence $(WASM_INFERENCE_BUILD_EVIDENCE)
	$(DOCKER_RUN) node tools/write_wasm_provenance.mjs check \
		--artifact $(WEB_DIST_DIR)/volvoxai.full.wasm \
		--build-evidence $(WASM_FULL_BUILD_EVIDENCE)
	$(DOCKER_RUN) chmod 0644 $(WEB_WASM_ARTIFACTS)
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) $(WEB_DIST_DIR) build/wasm
	@echo "Built $(WEB_WASM_ARTIFACTS)"

benchmark_wasm_w8a8_prefill: build_wasm
	$(DOCKER_RUN) node examples/tiny_receipt_vqa/tools/benchmark_wasm_w8a8_prefill.mjs \
		$(WEB_DIST_DIR)/volvoxai.wasm

benchmark_wasm_qbatch_matmul: build_wasm
	$(DOCKER_RUN) node tools/benchmark_wasm_qbatch_matmul.mjs \
		$(WEB_DIST_DIR)/volvoxai.wasm

build_web: build_docker
	$(DOCKER_RUN) npm install
	$(DOCKER_RUN) npm run build:all
	$(MAKE) build_wasm
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) $(WEB_DIST_DIR)
	@echo "Built $(WEB_ARTIFACTS)"

# The one release build+verification gate. check:release remains read-only;
# this target deterministically rebuilds all browser, WASM, and native profiles
# before checking inventory, provenance, ABI exports, and native symbols.
verify_release: build_web test_native
	$(DOCKER_RUN) npm run test:proto-api
	$(DOCKER_RUN) npm run test:wasm-ptq
	$(DOCKER_RUN) npm run test:wasm-inference-plan
	$(DOCKER_RUN) npm run test:planning-call-sequence
	$(DOCKER_RUN) npm run test:size
	$(DOCKER_RUN) npm run check:release
	$(DOCKER_RUN) bash -c 'set +e; \
		CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run size:report; \
		report_status=$$?; \
		chown -R $(HOST_UID):$(HOST_GID) build/size-reports 2>/dev/null || true; \
		exit $$report_status'

size_report: build_docker
	$(DOCKER_RUN) bash -c 'set +e; \
		CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run size:report; \
		report_status=$$?; \
		chown -R $(HOST_UID):$(HOST_GID) build/size-reports 2>/dev/null || true; \
		exit $$report_status'

size_check: build_docker
	$(DOCKER_RUN) env CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run size:check

# Publish NPM from the already-built dist artifacts.
publish_npm:
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm not found in PATH."; exit 1; }
	@set -e; \
	node tools/check_release_artifacts.mjs --web-only; \
	if ! npm whoami >/dev/null 2>&1; then \
		echo "Error: npm is not authenticated. Use npm login or provide the existing npm auth environment."; \
		exit 1; \
	fi; \
	echo "Package contents:"; \
	npm pack --dry-run; \
	echo "Publishing volvoxai@$(VERSION) to npm..."; \
	npm publish --access public; \
	echo "npm publish complete."

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
# Synurang generated bindings and codegen. Runs on the host with Python 3,
# protoc, and the checksum-pinned Synurang GitHub release binary.
# Native builds consume committed projections and the vendored C runtime, so
# this cache is used only when regenerating or checking generated projections.
# --------------------------------------------------------------------------
SYNURANG_CODEGEN_TOOL     := tools/generate_proto.py
PROTO_ENUM_CODEGEN_TOOL   := tools/generate_proto_enums.py
OPERATOR_VOCABULARY_CODEGEN_TOOL := tools/generate_operator_vocabulary.py
OPERATOR_PARAM_REGISTRY_CODEGEN_TOOL := tools/generate_operator_param_registry.py
KERNEL_REGISTRY_CODEGEN_TOOL := tools/generate_kernel_registry.py
OPTIMIZER_REGISTRY_CODEGEN_TOOL := tools/generate_optimizer_registry.py
API_CONFORMANCE_TOOL      := tools/check_api_conformance.py
PROTOC                    ?= protoc
PROTOC_GEN_SYNURANG_FFI   ?=
PROTO_DIR                 := proto
PROTO_SOURCE              := $(PROTO_DIR)/volvoxai.proto
PROTO_C_GEN_DIR           := runtime/generated/c
PROTO_TYPESCRIPT_GEN_DIR  := runtime/generated/typescript

SYNURANG_GENERATOR_ARG = $(if $(strip $(PROTOC_GEN_SYNURANG_FFI)),--generator "$(PROTOC_GEN_SYNURANG_FFI)")
SYNURANG_CODEGEN = $(PY) $(SYNURANG_CODEGEN_TOOL) \
	--protoc "$(PROTOC)" \
	--cache-dir "$(SYNURANG_CODEGEN_CACHE)" \
	$(SYNURANG_OFFLINE_ARG) \
	--proto "$(PROTO_SOURCE)" --proto-root "$(PROTO_DIR)" \
	--c-out "$(PROTO_C_GEN_DIR)" \
	--typescript-out "$(PROTO_TYPESCRIPT_GEN_DIR)" \
	$(SYNURANG_GENERATOR_ARG)

proto_codegen_fetch:
	$(SYNURANG_CODEGEN) --fetch-only

proto_codegen:
	$(SYNURANG_CODEGEN)
	$(PY) tools/generate_api_contract.py
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)
	$(PY) $(OPERATOR_VOCABULARY_CODEGEN_TOOL)
	$(PY) $(OPERATOR_PARAM_REGISTRY_CODEGEN_TOOL)
	$(PY) $(KERNEL_REGISTRY_CODEGEN_TOOL)
	$(PY) $(OPTIMIZER_REGISTRY_CODEGEN_TOOL)

proto_codegen_check:
	$(PY) tools/generate_api_contract.py --check
	$(PY) tools/generate_tokenizer_unicode.py --check
	$(SYNURANG_CODEGEN) --check
	$(PY) $(API_CONFORMANCE_TOOL) --check
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL) --check
	$(PY) $(OPERATOR_VOCABULARY_CODEGEN_TOOL) --check
	$(PY) $(OPERATOR_PARAM_REGISTRY_CODEGEN_TOOL) --check --protoc-check --protoc "$(PROTOC)"
	$(PY) $(KERNEL_REGISTRY_CODEGEN_TOOL) --check
	$(PY) $(OPTIMIZER_REGISTRY_CODEGEN_TOOL) --check

api_conformance:
	$(PY) $(API_CONFORMANCE_TOOL)

proto_enum_codegen:
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL)

proto_enum_codegen_check:
	$(PY) $(PROTO_ENUM_CODEGEN_TOOL) --check

operator_vocabulary_codegen:
	$(PY) $(OPERATOR_VOCABULARY_CODEGEN_TOOL)

operator_vocabulary_codegen_check:
	$(PY) $(OPERATOR_VOCABULARY_CODEGEN_TOOL) --check

operator_param_registry_codegen:
	$(PY) $(OPERATOR_PARAM_REGISTRY_CODEGEN_TOOL)

operator_param_registry_codegen_check:
	$(PY) $(OPERATOR_PARAM_REGISTRY_CODEGEN_TOOL) --check --protoc-check --protoc "$(PROTOC)"

kernel_registry_codegen:
	$(PY) $(KERNEL_REGISTRY_CODEGEN_TOOL)

kernel_registry_codegen_check:
	$(PY) $(KERNEL_REGISTRY_CODEGEN_TOOL) --check

optimizer_registry_codegen:
	$(PY) $(OPTIMIZER_REGISTRY_CODEGEN_TOOL)

optimizer_registry_codegen_check:
	$(PY) $(OPTIMIZER_REGISTRY_CODEGEN_TOOL) --check
