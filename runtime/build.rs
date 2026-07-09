// Build script for the VolvoxAI Synurang runtime cdylib.
//
//   1. prost-build turns proto/volvoxai.proto into the Rust message types
//      (volvoxai.v1.rs in OUT_DIR), included as `pb` in lib.rs.
//   2. cc compiles ALL of native/*.c (the engine, minus the CLI main.c) straight
//      into this cdylib, so the final libvolvoxai.so carries the whole engine.
//
// native/metal_engine.m is macOS-only and left out; native/nnapi_engine.c's
// Android include is guarded off unless USE_NNAPI is defined (as in build_native).

use std::path::Path;

fn main() {
    // 1) protobuf messages
    let mut cfg = prost_build::Config::new();
    cfg.protoc_arg("--experimental_allow_proto3_optional");
    cfg.compile_protos(&["../proto/volvoxai.proto"], &["../proto"])
        .expect("prost_build: failed to compile proto/volvoxai.proto");

    // 2) the C engine (same set as the Makefile build_native target, minus main.c)
    let native = Path::new("../native");
    let srcs = [
        "cJSON.c",
        "safetensors.c",
        "kernels.c",
        "quant_cpu_opt.c",
        "conv_f32_opt.c",
        "tensor_f32_opt.c",
        "engine_runtime.c",
        "engine.c",
        "image_io.c",
        "kie_runtime.c",
        "vulkan_engine.c",
        "opengl_engine.c",
        "tokenizer.c",
        "nnapi_engine.c",
    ];

    let mut build = cc::Build::new();
    build.include(native);
    build.flag_if_supported("-O3");
    build.flag_if_supported("-mavx2");
    build.flag_if_supported("-mfma");
    build.flag_if_supported("-pthread");
    build.warnings(false);
    for s in srcs {
        build.file(native.join(s));
        println!("cargo:rerun-if-changed=../native/{s}");
    }
    build.compile("volvoxengine"); // -> libvolvoxengine.a, linked into the cdylib

    // The engine dlopen's GPU/NPU backends and uses libm.
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rerun-if-changed=../proto/volvoxai.proto");
    println!("cargo:rerun-if-changed=src/gen/volvoxai_ffi_plugin.rs");
}
