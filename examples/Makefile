.PHONY: all clean cpp_ffi_example run_cpp_ffi_example \
	cpp_receipt_train_example check_cpp_receipt_train_graph \
	run_cpp_receipt_train_smoke native_receipt_inference_example \
	test_native_receipt_inference run_native_receipt_inference \
	native_task_cli test_native_task_cli build_server

EXAMPLES_DIR           := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT                   := $(abspath $(EXAMPLES_DIR)/..)
RECEIPT_SOURCE_DIR     := $(EXAMPLES_DIR)/tiny_receipt_vqa
TARGET_DIR             := $(EXAMPLES_DIR)/target
BIN_DIR                := $(TARGET_DIR)/bin
OBJ_DIR                := $(TARGET_DIR)/obj
CPP_PROTO_DIR          := $(TARGET_DIR)/generated/cpp
CPP_PROTO_STAMP        := $(CPP_PROTO_DIR)/.volvoxai-proto.stamp
CPP_CJSON_OBJECT       := $(OBJ_DIR)/cJSON.o
CPP_IMAGE_IO_OBJECT    := $(OBJ_DIR)/image_io.o
CPP_FFI_LITE_DIR       := $(ROOT)/runtime/generated/c
CPP_FFI_LITE_HEADER    := $(CPP_FFI_LITE_DIR)/volvoxai_lite.h
CPP_FFI_LITE_SOURCE    := $(CPP_FFI_LITE_DIR)/volvoxai_lite.c
CPP_FFI_LITE_OBJECT    := $(OBJ_DIR)/volvoxai_lite.o
CPP_FFI_EXAMPLE        := $(BIN_DIR)/cpp_ffi_client
CPP_RECEIPT_EXAMPLE    := $(BIN_DIR)/tiny_receipt_vqa_train
NATIVE_RECEIPT_EXAMPLE := $(BIN_DIR)/tiny_receipt_w8a8
NATIVE_TASK_CLI        := $(BIN_DIR)/volvoxai-tasks

PROTOC                 ?= protoc
PROTO3_OPT             := --experimental_allow_proto3_optional
CXX                     ?= c++
CXXFLAGS                ?= -std=c++17 -O2 -Wall -Wextra
CPP_FFI_CFLAGS          ?= -std=c11 -O2 -Wall -Wextra -pedantic
CPP_RECEIPT_CFLAGS      := $(shell pkg-config --cflags protobuf 2>/dev/null)
CPP_RECEIPT_LIBS        := $(shell pkg-config --libs protobuf 2>/dev/null)
RUNTIME_SHARED_LIBRARY := $(if $(filter Darwin,$(shell uname -s)),libvolvoxai.dylib,libvolvoxai.so)
RUNTIME_LIBRARY        := $(ROOT)/runtime/target/release/$(RUNTIME_SHARED_LIBRARY)
DLOPEN_LIB             := $(if $(filter Darwin,$(shell uname -s)),,-ldl)
NAVERCAP_ROOT          ?=

all: cpp_ffi_example cpp_receipt_train_example

build_server:
	$(MAKE) -C $(ROOT) build_server

cpp_ffi_example: $(CPP_FFI_EXAMPLE)

$(CPP_FFI_LITE_OBJECT): $(CPP_FFI_LITE_SOURCE) $(CPP_FFI_LITE_HEADER)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CPP_FFI_CFLAGS) -I$(CPP_FFI_LITE_DIR) -c $(CPP_FFI_LITE_SOURCE) -o $@

$(CPP_FFI_EXAMPLE): $(EXAMPLES_DIR)/cpp_ffi_client.cpp $(CPP_FFI_LITE_OBJECT) \
		$(CPP_FFI_LITE_HEADER)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(CPP_FFI_LITE_DIR) $< $(CPP_FFI_LITE_OBJECT) \
		-o $@ $(DLOPEN_LIB)

run_cpp_ffi_example: build_server cpp_ffi_example
	$(CPP_FFI_EXAMPLE) $(RUNTIME_LIBRARY) $(ROOT)

$(CPP_PROTO_STAMP): $(ROOT)/proto/volvoxai.proto
	@command -v pkg-config >/dev/null 2>&1 || { echo "Error: pkg-config is required"; exit 1; }
	@pkg-config --exists protobuf || { \
		echo "Error: protobuf development files are required"; exit 1; }
	@mkdir -p $(CPP_PROTO_DIR)
	$(PROTOC) -I$(ROOT)/proto $(PROTO3_OPT) --cpp_out=$(CPP_PROTO_DIR) $<
	@touch $@

$(CPP_CJSON_OBJECT): $(ROOT)/native/third_party/cJSON.c $(ROOT)/native/third_party/cJSON.h
	@mkdir -p $(OBJ_DIR)
	$(CC) -O2 -I$(ROOT)/native/third_party -c $< -o $@

$(CPP_IMAGE_IO_OBJECT): $(EXAMPLES_DIR)/native_support/image_io.c \
		$(EXAMPLES_DIR)/native_support/image_io.h \
		$(ROOT)/native/third_party/stb_image.h
	@mkdir -p $(OBJ_DIR)
	$(CC) -O2 -I$(EXAMPLES_DIR)/native_support -I$(ROOT)/native/third_party -c $< -o $@

cpp_receipt_train_example: $(CPP_RECEIPT_EXAMPLE)

$(CPP_RECEIPT_EXAMPLE): $(CPP_PROTO_STAMP) $(CPP_CJSON_OBJECT) \
		$(CPP_IMAGE_IO_OBJECT) $(RECEIPT_SOURCE_DIR)/tiny_receipt_vqa_train.cc \
		$(RECEIPT_SOURCE_DIR)/volvoxai_grpc_ffi_client.h
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CPP_RECEIPT_CFLAGS) \
		-I$(RECEIPT_SOURCE_DIR) -I$(CPP_PROTO_DIR) \
		-I$(ROOT)/native/include -I$(EXAMPLES_DIR)/native_support -I$(ROOT)/native/third_party \
		$(RECEIPT_SOURCE_DIR)/tiny_receipt_vqa_train.cc \
		$(CPP_PROTO_DIR)/volvoxai.pb.cc $(CPP_CJSON_OBJECT) $(CPP_IMAGE_IO_OBJECT) \
		-o $@ $(CPP_RECEIPT_LIBS) $(DLOPEN_LIB) -lm -pthread

# This validates data and graph construction without loading libvolvoxai or
# initializing CPU/GPU execution backends.
check_cpp_receipt_train_graph: cpp_receipt_train_example
	@test -n "$(NAVERCAP_ROOT)" || { echo "Error: pass NAVERCAP_ROOT=/path/to/navercap"; exit 1; }
	$(CPP_RECEIPT_EXAMPLE) --smoke --dry-run --backend cpu \
		--navercap-root "$(NAVERCAP_ROOT)" \
		--out /tmp/volvoxai-receipt-graph-check

run_cpp_receipt_train_smoke: build_server cpp_receipt_train_example
	@test -n "$(NAVERCAP_ROOT)" || { echo "Error: pass NAVERCAP_ROOT=/path/to/navercap"; exit 1; }
	$(CPP_RECEIPT_EXAMPLE) --smoke --backend cpu \
		--navercap-root "$(NAVERCAP_ROOT)" \
		--lib $(RUNTIME_LIBRARY) \
		--out /tmp/volvoxai-receipt-smoke

native_receipt_inference_example:
	$(MAKE) -C $(ROOT) build_tiny_receipt_native_example

test_native_receipt_inference:
	$(MAKE) -C $(ROOT) test_tiny_receipt_native_example

run_native_receipt_inference: native_receipt_inference_example
	@test -n "$(PACKAGE)" || { echo "Error: pass PACKAGE=/path/to/package"; exit 1; }
	@test -n "$(IMAGE)" || { echo "Error: pass IMAGE=/path/to/receipt.png"; exit 1; }
	@test -n "$(PROMPT)" || { echo "Error: pass PROMPT='question'"; exit 1; }
	$(NATIVE_RECEIPT_EXAMPLE) "$(PACKAGE)" --image "$(IMAGE)" --prompt "$(PROMPT)"

native_task_cli:
	$(MAKE) -C $(ROOT) build_native_task_cli

test_native_task_cli:
	$(MAKE) -C $(ROOT) test_native_task_cli

clean:
	rm -rf $(TARGET_DIR)
