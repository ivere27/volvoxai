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
        build_native_libraries build_dlpack_bridge test_python test_ptq_golden build_native_ptq_test \
        build_wheel_docker build_wheel \
        test_wasm_ptq \
        build_native_task_cli \
        build_wasm build_wasm_kernel_fixture test_wasm_baseline_simd wasm_abi_codegen wasm_abi_codegen_check \
        shader_catalog_codegen shader_catalog_codegen_check \
        check_release size_report size_check benchmark_wasm_w8a8_prefill \
        benchmark_wasm_qbatch_matmul build_web verify_release \
        publish_npm publish_pypi publish_npm_release publish_pypi_release publish_release release_git \
        models models_efficientdet models_tinystories models_deps models_clean \
        validate_model_packages \
        proto_codegen_fetch proto_codegen proto_codegen_check \
        proto_enum_codegen proto_enum_codegen_check \
        operator_vocabulary_codegen operator_vocabulary_codegen_check \
        operator_param_registry_codegen operator_param_registry_codegen_check \
        kernel_registry_codegen kernel_registry_codegen_check \
        optimizer_registry_codegen optimizer_registry_codegen_check \
        api_conformance \

VERSION := $(shell python3 -c 'import json; print(json.load(open("package.json", encoding="utf-8"))["version"])')
GIT_COMMIT := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo -dirty)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
DOCKER_IMAGE := volvoxai-build:latest
DOCKER_PREBUILT ?= 0
WHEEL_DOCKER_IMAGE := volvoxai-wheel-build:$(VERSION)
WHEEL_BUILD_JOBS ?= 2
DRY_RUN ?= 0
ifneq ($(DRY_RUN),0)
ifneq ($(DRY_RUN),1)
$(error DRY_RUN must be 0 or 1)
endif
endif
SYNURANG_CODEGEN_CACHE := build/cache/synurang-codegen
SYNURANG_OFFLINE ?= 0
SYNURANG_OFFLINE_ARG = $(if $(filter 1 true TRUE yes YES on ON,$(SYNURANG_OFFLINE)),--offline)
# GNU make computes CURDIR from the directory it actually entered.  The
# inherited PWD can still name an older symlink target and mount a different
# checkout into Docker.
# The bind mount is intentionally the repository selected by CURDIR.  Trust
# that exact mount for evidence-producing builds that query the Git snapshot;
# the container runs as root by default (while Web/WASM targets override
# DOCKER_USER_FLAGS to match the host UID/GID).
DOCKER_USER_FLAGS ?=
DOCKER_RUN = docker run --rm $(DOCKER_USER_FLAGS) \
	-e PYTHONDONTWRITEBYTECODE=1 \
	-e GIT_CONFIG_COUNT=1 \
	-e GIT_CONFIG_KEY_0=safe.directory \
	-e GIT_CONFIG_VALUE_0=/workspace \
	-v $(CURDIR):/workspace -w /workspace $(DOCKER_IMAGE)
# Restore host ownership of relative parent directories (such as build/) when a
# container running as root creates them before chowning a leaf artifact path.
CHOWN_HOST_PARENTS = for p in $(1); do \
	while case "$$p" in /*|../*|*/../*|*/..) false;; *) [ "$$p" != "." ] && [ "$$p" != ".." ] && [ -n "$$p" ];; esac; do \
		while case "$$p" in */) p=$${p%/};; *) false;; esac; do :; done; \
		case "$$p" in /*|../*|*/../*|*/..|""|.|..) break;; esac; \
		chown -h $(HOST_UID):$(HOST_GID) "$$p" 2>/dev/null || true; \
		p=$$(dirname "$$p"); \
	done; \
done
# Sweep any remaining root-owned files or directories created anywhere under
# /workspace when a build container runs as root (for example lockfiles or
# cache directories outside explicit target output paths).
CHOWN_HOST_WORKSPACE_ROOT = \
	if [ "$$(id -u)" -eq 0 ] && [ "$(HOST_UID)" != "0" ]; then \
		find /workspace -xdev -depth -user 0 -exec chown -h $(HOST_UID):$(HOST_GID) {} + 2>/dev/null || true; \
	fi
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
WEB_JS_ARTIFACTS := $(WEB_DIST_DIR)/volvoxai.lite.js $(WEB_DIST_DIR)/volvoxai.lite.min.js \
                    $(WEB_DIST_DIR)/volvoxai.js $(WEB_DIST_DIR)/volvoxai.min.js
WEB_WASM_ARTIFACTS := $(WEB_DIST_DIR)/volvoxai.lite.wasm $(WEB_DIST_DIR)/volvoxai.wasm
WEB_ARTIFACTS := $(WEB_JS_ARTIFACTS) $(WEB_WASM_ARTIFACTS)
NATIVE_RELEASE_ARTIFACTS := native/volvoxai-lite native/volvoxai
NATIVE_DEBUG_ARTIFACTS := native/.debug/volvoxai-lite.debug \
                          native/.debug/volvoxai.debug
NATIVE_DEBUG_EVIDENCE := native/.debug/volvoxai-lite.debug.json \
                         native/.debug/volvoxai.debug.json
# Shared and static forms of both profiles. The .so file names carry a version
# suffix and a pair of symlinks, so match on the stem.
NATIVE_LIBRARY_ARTIFACTS := $(wildcard native/libvolvoxai*.so*) \
                            $(wildcard native/libvolvoxai*.a)
WASM_PROVENANCE_SECTION_NAME := volvoxai.release.provenance.v1
WASM_PROVENANCE_DIR := build/wasm/provenance
WASM_INFERENCE_PROVENANCE := $(WASM_PROVENANCE_DIR)/volvoxai.lite.wasm.json
WASM_FULL_PROVENANCE := $(WASM_PROVENANCE_DIR)/volvoxai.wasm.json
WASM_INFERENCE_BUILD_EVIDENCE := $(WASM_PROVENANCE_DIR)/volvoxai.lite.wasm.build.json
WASM_FULL_BUILD_EVIDENCE := $(WASM_PROVENANCE_DIR)/volvoxai.wasm.build.json
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
	@echo "    build_native_libraries  libvolvoxai[-lite].so/.a — the public ABI"
	@echo "    build_dlpack_bridge     build volvoxai._dlpack for source-tree tests"
	@echo "    test_python            drive the .so from Python over Synurang FFI"
	@echo "    build_wheel            build/test Linux Python wheel -> dist/python/$(VERSION)/"
	@echo "    build_wheel_docker     build the separate manylinux wheel image"
	@echo "    publish_pypi           build/test Python wheel and upload to PyPI"
	@echo "    publish_npm/publish_pypi DRY_RUN=1  validate existing packages without uploading"
	@echo "    publish_npm_release TAG=vX.Y.Z   download verified npm tarballs from GitHub Release and publish locally"
	@echo "    publish_pypi_release TAG=vX.Y.Z  download verified Python wheel from GitHub Release and publish locally"
	@echo "    publish_release TAG=vX.Y.Z       download verified npm + wheel from GitHub Release and publish locally"
	@echo "    test_ptq_golden        byte-level PTQ authoring reference"
	@echo "    test_wasm_ptq          same template from WebAssembly and native"
	@echo "  Web / FFI / models:"
	@echo "    build_wasm build_web verify_release publish_npm  models"
	@echo "    wasm_abi_codegen_check  verify generated private WASM ABI projections"
	@echo "    test_wasm_baseline_simd  build and verify the developer kernel fixture"
	@echo "    size_report size_check  inspect and gate the fixed release inventory"
	@echo "    test_contracts test_onnx_oracle  required semantic gates"
	@echo "    test_quantized_oracle  exact project QLinear boundary gate"
	@echo "    verify_release     rebuild and gate all eight fixed release artifacts"
	@echo "    benchmark_wasm_qbatch_matmul  compare scalar/SIMD128 QBatchMatMul"
	@echo "    validate_model_packages  enforce canonical graph package names/formats"
	@echo "    proto_codegen_fetch proto_codegen proto_codegen_check"

build_docker:
	@if [ "$(DOCKER_PREBUILT)" = "1" ] && docker image inspect "$(DOCKER_IMAGE)" >/dev/null 2>&1; then \
		echo "Using prebuilt $(DOCKER_IMAGE)"; \
	else \
		docker build \
			--build-arg VOLVOXAI_VERSION=$(VERSION) \
			--build-arg VOLVOXAI_GIT_COMMIT=$(GIT_COMMIT)$(GIT_DIRTY) \
			--build-arg VOLVOXAI_BUILD_DATE=$(BUILD_DATE) \
			-t $(DOCKER_IMAGE) .; \
	fi

# Compile WGSL -> native shader formats (naga). The CMake build runs the same
# script internally for the engine; this target exists so shaders can be
# refreshed without a full native build.
compile_shaders: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,native/shaders); \
			chown -R $(HOST_UID):$(HOST_GID) native/shaders 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		./tools/compile_shaders.sh'

# --------------------------------------------------------------------------
# Native engine: thin CMake wrappers. The shader pipeline runs inside
# the CMake build; the whole thing runs as root in Docker (naga/cargo need it)
# and chowns the generated tree back afterwards.
# --------------------------------------------------------------------------
build_native: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,$(CMAKE_BUILD_DIR) $(NATIVE_RELEASE_ARTIFACTS)); \
			chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
				$(NATIVE_RELEASE_ARTIFACTS) native/libvolvoxai* \
				native/libsynurang_module_host* native/.debug 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) -j"$(NATIVE_BUILD_JOBS)"'

build_native_profiles: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,$(CMAKE_BUILD_DIR) $(NATIVE_RELEASE_ARTIFACTS)); \
			chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
				$(NATIVE_RELEASE_ARTIFACTS) native/.debug 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target volvoxai-lite volvoxai \
			-j"$(NATIVE_BUILD_JOBS)"'

# The distributable libraries. proto/volvoxai.proto is the whole public API and
# Synurang_GetApi module dispatch is its entry point for every host.
build_native_libraries: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,$(CMAKE_BUILD_DIR)); \
			chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
				native/libvolvoxai* native/libsynurang_module_host* \
				native/.debug 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target \
			volvoxai-lite_shared volvoxai-lite_static \
			volvoxai_shared volvoxai_static \
			-j"$(NATIVE_BUILD_JOBS)"'

build_dlpack_bridge:
	@$(PY) -c 'import setuptools' >/dev/null 2>&1 \
		|| { echo "Error: install python/requirements-test.txt"; exit 1; }
	$(PY) python/build_dlpack_bridge.py

# Drives source-built modules through the pinned Synurang Python call host.
# The host is vendored; python/requirements-test.txt supplies test fixtures.
test_python: build_native_libraries build_dlpack_bridge
	@PYTHONPATH="$(CURDIR)/python" $(PY) -c 'from volvoxai import ModuleHost' \
		|| { echo "Error: install python/requirements-test.txt"; exit 1; }
	$(PY) -m unittest discover -s python/tests -p 'test_*.py' -v

# Python packaging uses its own glibc baseline and never writes into the six
# fixed npm artifact paths or replaces the source-built native libraries.
build_wheel_docker:
	docker build --platform linux/amd64 -f python/Dockerfile.wheel \
		-t $(WHEEL_DOCKER_IMAGE) .

build_wheel: build_wheel_docker
	mkdir -p dist/python build/python-wheel
	docker run --rm --platform linux/amd64 --cpus=$(WHEEL_BUILD_JOBS) \
		--user $(HOST_UID):$(HOST_GID) \
		-e VOLVOXAI_WHEEL_IMAGE_ID=$(shell docker image inspect --format '{{.Id}}' $(WHEEL_DOCKER_IMAGE)) \
		-e VOLVOXAI_WHEEL_IMAGE_KEY=$(shell docker image inspect --format '{{json .RootFS.Layers}} {{json .Config.Env}} {{json .Config.Cmd}} {{json .Config.Entrypoint}}' $(WHEEL_DOCKER_IMAGE) | sha256sum | cut -d ' ' -f 1) \
		-v "$(CURDIR):/workspace:ro" -v "$(CURDIR)/dist/python:/out" \
		-v "$(CURDIR)/build/python-wheel:/wheel-work" \
		$(WHEEL_DOCKER_IMAGE) python python/build_wheel.py \
		--out /out --work-dir /wheel-work --jobs $(WHEEL_BUILD_JOBS)
	$(MAKE) publish_pypi DRY_RUN=1

# Reuse the wheel image's pinned Twine and preserve interactive token entry.
# Dry runs use the existing wheel, including when invoked at the end of its build.
publish_pypi: $(if $(filter 0,$(DRY_RUN)),build_wheel)
	@set -e; \
	docker run --rm --platform linux/amd64 \
		-v "$(CURDIR)/dist/python/$(VERSION):/dist:ro" -w /dist \
		$(WHEEL_DOCKER_IMAGE) sh -c \
		'sha256sum --check SHA256SUMS && python -m twine check --strict *.whl'; \
	if [ "$(DRY_RUN)" = 1 ]; then \
		echo "PyPI dry run passed; no upload performed."; \
		exit 0; \
	fi; \
	tty_flag=; if [ -t 0 ]; then tty_flag=-t; fi; \
	set --; \
	if [ -f "$$HOME/.pypirc" ]; then \
		set -- -v "$$HOME/.pypirc:/root/.pypirc:ro"; \
	fi; \
	docker run --rm --platform linux/amd64 -i $$tty_flag "$$@" \
		-e TWINE_USERNAME -e TWINE_PASSWORD -e TWINE_NON_INTERACTIVE \
		-e TWINE_REPOSITORY -e TWINE_REPOSITORY_URL \
		-v "$(CURDIR)/dist/python/$(VERSION):/dist:ro" -w /dist \
		$(WHEEL_DOCKER_IMAGE) sh -c \
		'python -m twine upload *.whl'

# The byte-level reference the C port of PTQ authoring has to meet. It needs no
# library, so it runs on its own and stays fast enough to run often.
test_ptq_golden:
	$(PY) -m unittest discover -s python/tests -p 'test_ptq_*.py' -v

build_native_ptq_test: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,$(CMAKE_BUILD_DIR)); \
			chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target test_ptq_authoring \
			-j"$(NATIVE_BUILD_JOBS)"'

test_wasm_ptq: build_wasm build_native_ptq_test
	$(DOCKER_RUN) npm run test:wasm-ptq

check_native_cuda_ptx:
	$(PY) tools/check_native_cuda_ptx.py --build-dir "$(CMAKE_BUILD_DIR)"

test_native_cpu_selection: build_native
	$(DOCKER_RUN) npm run test:native-cpu-selection -- \
		--native-build-dir="$(CMAKE_BUILD_DIR)"

test_native_invariants: build_native
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR)/Testing 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		ctest --test-dir $(CMAKE_BUILD_DIR) \
			-L native-invariant -LE native-isa-runtime --output-on-failure'

test_native_gpu_static: build_native
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR)/Testing 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		ctest --test-dir $(CMAKE_BUILD_DIR) \
			-L native-gpu-static --output-on-failure'

verify_native_isa: build_native
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR)/Testing 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		ctest --test-dir $(CMAKE_BUILD_DIR) \
			-L '\''native-isa-runtime|native-isa-codegen'\'' --output-on-failure'

test_native: test_native_cpu_selection test_native_invariants verify_native_isa
	$(DOCKER_RUN) native/cmake/check_profile_boundaries.sh \
		$(CMAKE_BUILD_DIR)/native/release-link/volvoxai-lite \
		$(CMAKE_BUILD_DIR)/native/release-link/volvoxai

# Compare the release selective hot/cold graph against a current-source all-O3
# graph. The private comparison executables link without relinking, finalizing,
# or publishing the release binary.
# The runner writes failure evidence before returning a non-zero gate status.
benchmark_native_w8a8_release: build_native_profiles
	$(DOCKER_RUN) npm run test:native-w8a8-release
	$(DOCKER_RUN) bash -c 'set +e; \
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,$(CMAKE_BUILD_DIR) $(NATIVE_W8A8_PERFORMANCE_REPORT)); \
			chown -R $(HOST_UID):$(HOST_GID) \
				$(CMAKE_BUILD_DIR) \
				$(NATIVE_W8A8_PERFORMANCE_REPORT) 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		physical_cores=$$(python3 tools/count_allowed_physical_cores.py); \
		if [ $$? -ne 0 ] || ! [[ "$$physical_cores" =~ ^[1-9][0-9]*$$ ]]; then \
			echo "::error::Failed to determine allowed physical core count" >&2; \
			exit 1; \
		elif [ "$$physical_cores" -lt 4 ]; then \
			echo "::notice::Skipping native W8A8 benchmark: host has $$physical_cores physical cores (< 4 required)"; \
		else \
			CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" \
				npm run baseline:native-w8a8-release -- \
				--native-build-dir="$(CMAKE_BUILD_DIR)" \
				--blocks="$(NATIVE_W8A8_PERFORMANCE_BLOCKS)" \
				--max-blocks="$(NATIVE_W8A8_PERFORMANCE_MAX_BLOCKS)" \
				--min-timed-ms="$(NATIVE_W8A8_PERFORMANCE_MIN_TIMED_MS)" \
				--output="$(NATIVE_W8A8_PERFORMANCE_REPORT)"; \
		fi'

test_contracts:
	$(PY) -m unittest discover -s tools/tests -p 'test_*.py' -v

test_onnx_oracle:
	$(PY) tests/parity/external/onnx_oracle.py \
		--bundle $(WEB_DIST_DIR)/volvoxai.lite.js \
		--wasm $(WEB_DIST_DIR)/volvoxai.lite.wasm \
		--native native/volvoxai-lite \
		--report $(ONNX_ORACLE_REPORT)

test_model_corpus:
	$(PY) tests/parity/external/model_corpus.py \
		--bundle $(WEB_DIST_DIR)/volvoxai.lite.js \
		--wasm $(WEB_DIST_DIR)/volvoxai.lite.wasm \
		--native native/volvoxai-lite \
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
		--wasm $(WEB_DIST_DIR)/volvoxai.wasm \
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
		--wasm $(WEB_DIST_DIR)/volvoxai.wasm

test_quantized_oracle:
	$(PY) tests/parity/external/quantized_oracle.py \
		--bundle $(WEB_DIST_DIR)/volvoxai.lite.js \
		--wasm $(WEB_DIST_DIR)/volvoxai.lite.wasm \
		--native native/volvoxai-lite \
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
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,$(CMAKE_BUILD_DIR) $(EXAMPLES_BIN_DIR)); \
			chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
				$(EXAMPLES_BIN_DIR) 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG) $(CMAKE_EVIDENCE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target volvoxai-tasks -j"$$(nproc)"; \
		mkdir -p $(EXAMPLES_BIN_DIR); \
		cp $(CMAKE_BUILD_DIR)/native/volvoxai-tasks $(EXAMPLES_BIN_DIR)/volvoxai-tasks'
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

# Run Web/WASM toolchain containers as the host user so every generated tree
# (dist/, build/wasm/, node_modules/, build/npm/) is owned by the host even on
# partial failures or interrupts.
build_wasm: DOCKER_USER_FLAGS := --user $(HOST_UID):$(HOST_GID) -e HOME=/tmp
build_wasm: build_docker wasm_abi_codegen_check shader_catalog_codegen_check
	$(DOCKER_RUN) mkdir -p $(WEB_DIST_DIR) $(WASM_PROVENANCE_DIR)
	$(DOCKER_RUN) rm -f $(WEB_WASM_ARTIFACTS) \
		$(WASM_INFERENCE_PROVENANCE) $(WASM_FULL_PROVENANCE) \
		$(WASM_INFERENCE_BUILD_EVIDENCE) $(WASM_FULL_BUILD_EVIDENCE)
	$(DOCKER_RUN) $(WASM_RELEASE_BUILDER) all \
		--output-dir $(WEB_DIST_DIR) \
		--build-root $(WASM_RELEASE_OBJECT_ROOT) \
		--evidence-dir $(WASM_PROVENANCE_DIR)
	$(DOCKER_RUN) node tools/write_wasm_provenance.mjs write \
		--artifact $(WEB_DIST_DIR)/volvoxai.lite.wasm \
		--output $(WASM_INFERENCE_PROVENANCE) \
		--build-evidence $(WASM_INFERENCE_BUILD_EVIDENCE)
	$(DOCKER_RUN) node tools/write_wasm_provenance.mjs write \
		--artifact $(WEB_DIST_DIR)/volvoxai.wasm \
		--output $(WASM_FULL_PROVENANCE) \
		--build-evidence $(WASM_FULL_BUILD_EVIDENCE)
	$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py \
		--section-name $(WASM_PROVENANCE_SECTION_NAME) \
		--payload $(WASM_INFERENCE_PROVENANCE) \
		--input $(WEB_DIST_DIR)/volvoxai.lite.wasm \
		--output $(WEB_DIST_DIR)/volvoxai.lite.wasm
	$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py \
		--section-name $(WASM_PROVENANCE_SECTION_NAME) \
		--payload $(WASM_FULL_PROVENANCE) \
		--input $(WEB_DIST_DIR)/volvoxai.wasm \
		--output $(WEB_DIST_DIR)/volvoxai.wasm
	$(DOCKER_RUN) node tools/write_wasm_provenance.mjs check \
		--artifact $(WEB_DIST_DIR)/volvoxai.lite.wasm \
		--build-evidence $(WASM_INFERENCE_BUILD_EVIDENCE)
	$(DOCKER_RUN) node tools/write_wasm_provenance.mjs check \
		--artifact $(WEB_DIST_DIR)/volvoxai.wasm \
		--build-evidence $(WASM_FULL_BUILD_EVIDENCE)
	$(DOCKER_RUN) chmod 0644 $(WEB_WASM_ARTIFACTS)
	@echo "Built $(WEB_WASM_ARTIFACTS)"

build_wasm_kernel_fixture: DOCKER_USER_FLAGS := --user $(HOST_UID):$(HOST_GID) -e HOME=/tmp
build_wasm_kernel_fixture: build_docker wasm_abi_codegen_check
	$(DOCKER_RUN) node tools/build_wasm_kernel_fixture.mjs

test_wasm_baseline_simd: build_wasm_kernel_fixture
	$(DOCKER_RUN) npm run test:wasm-baseline-simd

benchmark_wasm_w8a8_prefill: build_wasm_kernel_fixture
	$(DOCKER_RUN) node examples/tiny_receipt_vqa/tools/benchmark_wasm_w8a8_prefill.mjs \
		build/wasm/kernel-fixture/kernels.wasm

benchmark_wasm_qbatch_matmul: build_wasm_kernel_fixture
	$(DOCKER_RUN) node tools/benchmark_wasm_qbatch_matmul.mjs \
		build/wasm/kernel-fixture/kernels.wasm

build_web: DOCKER_USER_FLAGS := --user $(HOST_UID):$(HOST_GID) -e HOME=/tmp
build_web: build_docker
	$(DOCKER_RUN) sh -c 'set -e; npm install; npm run build:all'
	$(MAKE) build_wasm
	$(DOCKER_RUN) $(MAKE) publish_npm DRY_RUN=1
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
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,build/size-reports); \
			chown -R $(HOST_UID):$(HOST_GID) build/size-reports 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run size:report'

check_release: build_docker
	$(DOCKER_RUN) env CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run check:release

size_report: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cleanup() { \
			rc=$$?; set +e; \
			$(call CHOWN_HOST_PARENTS,build/size-reports); \
			chown -R $(HOST_UID):$(HOST_GID) build/size-reports 2>/dev/null || true; \
			$(CHOWN_HOST_WORKSPACE_ROOT); \
			exit $$rc; \
		}; \
		trap cleanup EXIT; \
		trap "exit 130" INT; trap "exit 143" TERM; \
		CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run size:report'

size_check: build_docker
	$(DOCKER_RUN) env CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run size:check

# Package and publish both npm distributions from the already-built artifacts.
publish_npm:
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm not found in PATH."; exit 1; }
	node tools/publish_npm.mjs $(if $(filter 1,$(DRY_RUN)),--dry-run)

_RELEASE_BOOL_FLAG = $(if $(filter 1 true TRUE yes YES on ON,$(strip $($(1)))),$(2))

# Download verified npm tarballs from a published GitHub Release and publish locally.
publish_npm_release:
	@test -n "$(strip $(TAG))" || { \
		echo "Error: TAG is required (for example: make publish_npm_release TAG=v$(VERSION))."; \
		exit 1; \
	}
	$(PY) tools/publish_release_local.py publish-npm-local \
		--tag "$(strip $(TAG))" \
		$(if $(strip $(REPO)),--repo "$(strip $(REPO))") \
		$(if $(strip $(NPM_TAG)),--npm-tag "$(strip $(NPM_TAG))") \
		$(if $(strip $(NPM_OTP)),--otp "$(strip $(NPM_OTP))") \
		$(call _RELEASE_BOOL_FLAG,ALLOW_PRERELEASE,--allow-prerelease) \
		$(call _RELEASE_BOOL_FLAG,ALLOW_DRAFT,--allow-draft) \
		$(call _RELEASE_BOOL_FLAG,DRY_RUN,--dry-run)

# Download verified Python wheel from a published GitHub Release and publish locally via twine.
publish_pypi_release:
	@test -n "$(strip $(TAG))" || { \
		echo "Error: TAG is required (for example: make publish_pypi_release TAG=v$(VERSION))."; \
		exit 1; \
	}
	$(PY) tools/publish_release_local.py publish-pypi-local \
		--tag "$(strip $(TAG))" \
		$(if $(strip $(REPO)),--repo "$(strip $(REPO))") \
		$(if $(strip $(TWINE_REPOSITORY)),--repository "$(strip $(TWINE_REPOSITORY))") \
		$(if $(strip $(TWINE_REPOSITORY_URL)),--repository-url "$(strip $(TWINE_REPOSITORY_URL))") \
		$(call _RELEASE_BOOL_FLAG,ALLOW_PRERELEASE,--allow-prerelease) \
		$(call _RELEASE_BOOL_FLAG,ALLOW_DRAFT,--allow-draft) \
		$(call _RELEASE_BOOL_FLAG,DRY_RUN,--dry-run)

# Download verified release artifacts from GitHub Release TAG=vX.Y.Z and publish locally (npm + PyPI).
publish_release:
	@test -n "$(strip $(TAG))" || { \
		echo "Error: TAG is required (for example: make publish_release TAG=v$(VERSION))."; \
		exit 1; \
	}
	@$(MAKE) publish_npm_release TAG="$(strip $(TAG))"
	@$(MAKE) publish_pypi_release TAG="$(strip $(TAG))"

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
