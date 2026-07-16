// Build script for the VolvoxAI Synurang runtime cdylib.
//
//   1. prost-build turns proto/volvoxai.proto into the Rust message types
//      (volvoxai.v1.rs in OUT_DIR), included as `pb` in lib.rs.
//   2. pack_native_shaders.py generates the full embedded shader pack in
//      OUT_DIR.
//   3. cc compiles the native engine (minus the CLI main.c), shader store, XZ
//      decoder, and generated pack straight into this cdylib. macOS builds also
//      compile the Metal backend and link Apple's Foundation and Metal
//      frameworks.
//
// native/src/backends/nnapi_engine.c's Android include is guarded off unless
// USE_NNAPI is defined (as in the Android native build).

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn contains_file_with_extension(directory: &Path, extension: &str) -> bool {
    let Ok(entries) = fs::read_dir(directory) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if contains_file_with_extension(&path, extension) {
                return true;
            }
        } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            return true;
        }
    }
    false
}

fn require_compiled_shaders(compiled_dir: &Path) {
    let missing: Vec<&str> = [
        ("spv", "spv"),
        ("glsl", "comp"),
        ("gles", "comp"),
        ("metal", "metal"),
    ]
    .into_iter()
    .filter_map(|(backend, extension)| {
        (!contains_file_with_extension(&compiled_dir.join(backend), extension)).then_some(backend)
    })
    .collect();

    if !missing.is_empty() {
        panic!(
            "compiled native shader artifacts are missing for: {}. Run `make compile_shaders` from the repository root, then rebuild the runtime",
            missing.join(", ")
        );
    }
}

fn emit_shader_rerun_paths(directory: &Path, extension: &str) {
    println!("cargo:rerun-if-changed={}", directory.display());
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            emit_shader_rerun_paths(&path, extension);
        } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn generate_embedded_shaders(out_dir: &Path) -> PathBuf {
    let compiled_dir = Path::new("../native/shaders");
    let shader_source_dir = Path::new("../shaders");
    require_compiled_shaders(compiled_dir);

    let output_c = out_dir.join("embedded_shaders.c");
    let output_h = out_dir.join("embedded_shaders.h");
    let python = env::var_os("PYTHON")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| OsString::from("python3"));
    let output = Command::new(&python)
        .arg("../tools/pack_native_shaders.py")
        .arg("--compiled-dir")
        .arg(compiled_dir)
        .arg("--shader-source-dir")
        .arg(shader_source_dir)
        .arg("--profile")
        .arg("full")
        .arg("--output-c")
        .arg(&output_c)
        .arg("--output-h")
        .arg(&output_h)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to run {:?} ../tools/pack_native_shaders.py: {error}. Set PYTHON to a usable Python 3 executable",
                python
            )
        });
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "failed to generate the embedded full shader pack. Run `make compile_shaders` from the repository root if compiled artifacts are stale or missing.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    println!("cargo:rerun-if-env-changed=PYTHON");
    println!("cargo:rerun-if-changed=../tools/pack_native_shaders.py");
    emit_shader_rerun_paths(&shader_source_dir.join("inference"), "wgsl");
    emit_shader_rerun_paths(&shader_source_dir.join("training"), "wgsl");
    for (backend, extension) in [
        ("spv", "spv"),
        ("glsl", "comp"),
        ("gles", "comp"),
        ("metal", "metal"),
    ] {
        emit_shader_rerun_paths(&compiled_dir.join(backend), extension);
    }

    output_c
}

fn main() {
    // 1) protobuf messages
    let mut cfg = prost_build::Config::new();
    cfg.protoc_arg("--experimental_allow_proto3_optional");
    cfg.compile_protos(&["../proto/volvoxai.proto"], &["../proto"])
        .expect("prost_build: failed to compile proto/volvoxai.proto");

    // 2) the generic native engine. Cargo supplies CARGO_CFG_TARGET_OS for the
    // target, not the host; this keeps Linux builds C-only while cross/host
    // macOS builds include Metal.
    let target_os = env::var("CARGO_CFG_TARGET_OS")
        .expect("Cargo did not provide CARGO_CFG_TARGET_OS to build.rs");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")
        .expect("Cargo did not provide CARGO_CFG_TARGET_ARCH to build.rs");
    let target_is_macos = target_os == "macos";
    let target_has_android_dotprod = target_os == "android" && target_arch == "aarch64";
    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo did not provide OUT_DIR to build.rs"));
    let embedded_shaders_c = generate_embedded_shaders(&out_dir);
    let native = Path::new("../native");
    let srcs = [
        "third_party/cJSON.c",
        "third_party/xz-embedded/xz_crc32.c",
        "third_party/xz-embedded/xz_dec_lzma2.c",
        "third_party/xz-embedded/xz_dec_stream.c",
        "src/shader_store.c",
        "src/runtime/safetensors.c",
        "src/runtime/adapter_runtime.c",
        "src/runtime/attention_mask.c",
        "src/runtime/sequence_runtime.c",
        "src/runtime/backend.c",
        "src/runtime/arena.c",
        "src/runtime/backend_sdk.c",
        "src/runtime/incremental_runtime.c",
        "src/runtime/decode_session.c",
        "src/backends/w8a8_device_ops.c",
        "src/kernels/kernels.c",
        "src/kernels/cpu_features.c",
        "src/kernels/quant_cpu_opt.c",
        "src/kernels/qlinear_w8a8_x86.c",
        "src/kernels/qlinear_w8a8_arm.c",
        "src/kernels/qconv_w8a8_x86.c",
        "src/kernels/qconv_w8a8_arm.c",
        "src/kernels/qsdpa_w8a8_native.c",
        "src/kernels/conv_f32_opt.c",
        "src/kernels/tensor_f32_opt.c",
        "src/runtime/engine_runtime.c",
        "src/runtime/engine.c",
        "src/training/quantization.c",
        "src/training/quantization_runtime.c",
        "src/training/quantization_package.c",
        "src/backends/backend_manager.c",
        "src/backends/vulkan_engine.c",
        "src/backends/opengl_engine.c",
        "src/backends/nnapi_engine.c",
    ];

    let mut build = cc::Build::new();
    for include in [
        "include",
        "src",
        "src/runtime",
        "src/kernels",
        "src/backends",
        "third_party",
        "third_party/xz-embedded",
    ] {
        build.include(native.join(include));
    }
    build.include(&out_dir);
    if target_is_macos {
        if let Some(vulkan_sdk) = env::var_os("VULKAN_SDK") {
            build.include(Path::new(&vulkan_sdk).join("include"));
        }
        println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    }
    build.flag_if_supported("-O3");
    build.flag_if_supported("-mavx2");
    build.flag_if_supported("-mfma");
    build.flag_if_supported("-pthread");
    build.define("VOLVOXAI_ENABLE_TRAINING", Some("1"));
    if target_has_android_dotprod {
        build.define("VOLVOXAI_ARM_DOTPROD_OBJECT", Some("1"));
    }
    build.warnings(false);
    for s in srcs {
        build.file(native.join(s));
        println!("cargo:rerun-if-changed=../native/{s}");
    }
    // The Rust service exposes Generate, so it opts into the application-level
    // byte-BPE helper that fixed native CLI artifacts deliberately omit.
    for s in ["src/tokenization/tokenizer.c"] {
        build.file(native.join(s));
        println!("cargo:rerun-if-changed=../native/{s}");
    }
    build.file(&embedded_shaders_c);
    if target_is_macos {
        build.file(native.join("src/backends/metal_engine.m"));
        println!("cargo:rerun-if-changed=../native/src/backends/metal_engine.m");
    }
    // kernels.c is a portable amalgamation, so track its included units as well
    // as ordinary headers. Otherwise Cargo would not rebuild after editing one
    // of those implementation files.
    for dependency in [
        "include/volvoxai.h",
        "include/volvoxai_backend.h",
        "include/volvoxai_training.h",
        "src/shader_store.h",
        "src/runtime/adapter_runtime.h",
        "src/runtime/adapter_runtime_internal.h",
        "src/runtime/attention_mask.h",
        "src/runtime/backend.h",
        "src/runtime/backend_sdk.h",
        "src/runtime/backend_config.h",
        "src/runtime/engine_internal.h",
        "src/runtime/engine_runtime_dispatch.inc",
        "src/runtime/engine_runtime_f32_cpu.inc",
        "src/runtime/engine_runtime_f32_gpu.inc",
        "src/runtime/engine_runtime_model.inc",
        "src/runtime/engine_runtime_w8a8.inc",
        "src/runtime/graph_opt_fusion.inc",
        "src/runtime/incremental_runtime.h",
        "src/runtime/profiler.inc",
        "src/runtime/safetensors.h",
        "src/runtime/sequence_runtime.h",
        "include/volvoxai_tokenizer.h",
        "src/training/optimizer_runtime.inc",
        "src/training/training_forward_dropout.inc",
        "src/training/training_runtime.inc",
        "src/kernels/conv_f32_opt.h",
        "src/kernels/fusion_ops.h",
        "src/kernels/inference_kernels.h",
        "src/kernels/lora_linear.c",
        "src/kernels/lora_linear.h",
        "src/kernels/mathcompat.h",
        "src/kernels/quant_cpu_opt.h",
        "src/kernels/qlinear_w8a8_arm.h",
        "src/kernels/qlinear_w8a8_arm_internal.h",
        "src/kernels/qconv_w8a8_arm.h",
        "src/kernels/qconv_w8a8_arm_dotprod.c",
        "src/kernels/tensor_f32_opt.h",
        "src/kernels/thread_pool.c",
        "src/kernels/thread_pool.h",
        "src/kernels/wasm_simd128_polyfill.h",
        "src/kernels/activations.c",
        "src/kernels/broadcast_ops.c",
        "src/kernels/core.c",
        "src/kernels/cross_sdpa.c",
        "src/kernels/cross_attention.c",
        "src/kernels/cv_nlp.c",
        "src/kernels/edge_primitives.c",
        "src/kernels/embedding.c",
        "src/kernels/fast_math.c",
        "src/kernels/fusion_ops.c",
        "src/kernels/layernorm.c",
        "src/kernels/math_nlp.c",
        "src/kernels/matmul.c",
        "src/kernels/misc_ops.c",
        "src/kernels/portable_inference_kernels.c",
        "src/kernels/sdpa.c",
        "src/kernels/shape_math.c",
        "src/kernels/sequence_ops.c",
        "src/kernels/vision_ops.c",
        "src/backends/backend_manager.h",
        "src/backends/metal_engine.h",
        "src/backends/nnapi_engine.h",
        "src/backends/opengl_engine.h",
        "src/backends/vulkan_engine.h",
        "src/backends/w8a8_device_ops.h",
        "third_party/cJSON.h",
        "third_party/xz-embedded/xz.h",
        "third_party/xz-embedded/xz_config.h",
        "third_party/xz-embedded/xz_lzma2.h",
        "third_party/xz-embedded/xz_private.h",
        "third_party/xz-embedded/xz_stream.h",
    ] {
        println!("cargo:rerun-if-changed=../native/{dependency}");
    }
    build.compile("volvoxengine"); // -> libvolvoxengine.a, linked into the cdylib

    // Keep SDOT in a separate Armv8.2+dotprod archive. It is emitted after
    // the baseline engine archive so static linking resolves the optional
    // references from the baseline ARM kernels, while runtime HWCAP gating
    // prevents the instructions from executing on older ARM CPUs.
    if target_has_android_dotprod {
        let mut dotprod = cc::Build::new();
        for include in ["include", "src", "src/runtime", "src/kernels"] {
            dotprod.include(native.join(include));
        }
        dotprod.flag_if_supported("-O3");
        dotprod.flag("-march=armv8.2-a+dotprod");
        dotprod.warnings(false);
        dotprod.file(native.join("src/kernels/qlinear_w8a8_arm_dotprod.c"));
        dotprod.file(native.join("src/kernels/qconv_w8a8_arm_dotprod.c"));
        println!("cargo:rerun-if-changed=../native/src/kernels/qlinear_w8a8_arm_dotprod.c");
        println!("cargo:rerun-if-changed=../native/src/kernels/qconv_w8a8_arm_dotprod.c");
        dotprod.compile("volvoxarmdotprod");
    }

    // The engine dlopen's GPU/NPU backends and uses libm.
    println!("cargo:rustc-link-lib=dylib=m");
    // Darwin provides dlopen/dlsym through libSystem rather than a separate
    // libdl link dependency.
    if !target_is_macos {
        println!("cargo:rustc-link-lib=dylib=dl");
    }
    if target_is_macos {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
    }
    println!("cargo:rerun-if-changed=../proto/volvoxai.proto");
    println!("cargo:rerun-if-changed=src/gen/volvoxai_ffi_plugin.rs");
}
