// Build script for the VolvoxAI Synurang runtime cdylib.
//
//   1. prost-build turns proto/volvoxai.proto into the Rust message types
//      (volvoxai.runtime.rs in OUT_DIR), included as `pb` in lib.rs.
//   2. pack_native_shaders.py generates the full shader pack in OUT_DIR. With
//      the opt-in `cuda` Cargo feature, the authoritative forward and training
//      CUDA sources are separately compiled to PTX and embedded there as well.
//   3. cc compiles the native engine (minus the CLI main.c), shader store, XZ
//      decoder, and generated packs straight into this cdylib. macOS builds
//      also compile the Metal backend and link Apple's Foundation and Metal
//      frameworks. CUDA uses only dlopen/GetProcAddress for the Driver API; the
//      plugin never links CUDA SDK or runtime libraries.
//
// native/src/backends/nnapi_engine.c's Android include is guarded off unless
// USE_NNAPI is defined (as in the Android native build).

use std::collections::BTreeSet;
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

fn emit_transitive_fragment_rerun_paths(source: &Path, reject_host_fragments: bool) {
    let mut pending = vec![source.to_path_buf()];
    let mut visited = BTreeSet::<PathBuf>::new();

    while let Some(path) = pending.pop() {
        if !visited.insert(path.clone()) {
            continue;
        }
        let contents = fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!("failed to read source fragment {}: {error}", path.display())
        });
        println!("cargo:rerun-if-changed={}", path.display());

        let mut includes = Vec::<PathBuf>::new();
        for line in contents.lines() {
            let Some(include) = line.trim_start().strip_prefix("#include \"") else {
                continue;
            };
            let Some(end) = include.find('"') else {
                continue;
            };
            let include = &include[..end];
            if !include.ends_with(".inc") {
                continue;
            }
            if reject_host_fragments && include.ends_with("_host.inc") {
                panic!(
                    "CUDA PTX source {} includes host-only fragment {include}",
                    path.display()
                );
            }
            let include_path = path
                .parent()
                .expect("fragment source must have a parent directory")
                .join(include);
            if !include_path.is_file() {
                panic!(
                    "source {} includes missing fragment {}",
                    path.display(),
                    include_path.display()
                );
            }
            includes.push(include_path);
        }
        includes.sort();
        pending.extend(includes.into_iter().rev());
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

#[derive(Clone, Debug)]
enum CudaPtxCompiler {
    Nvcc(OsString),
    Clang(OsString),
}

impl CudaPtxCompiler {
    fn name(&self) -> String {
        match self {
            Self::Nvcc(program) => format!("{} (nvcc)", program.to_string_lossy()),
            Self::Clang(program) => format!("{} (clang NVPTX)", program.to_string_lossy()),
        }
    }

    fn compile(&self, source: &Path, output: &Path, architecture: &str, fast_fp32: bool) {
        let mut command = match self {
            Self::Nvcc(program) => {
                let mut command = Command::new(program);
                command
                    .arg("--ptx")
                    .arg("--std=c++14")
                    .arg("-O3")
                    .arg(if fast_fp32 {
                        "--fmad=true"
                    } else {
                        "--fmad=false"
                    })
                    .arg(format!("--gpu-architecture=compute_{architecture}"));
                command
            }
            Self::Clang(program) => {
                let mut command = Command::new(program);
                command
                    .arg("-x")
                    .arg("cuda")
                    .arg("--cuda-device-only")
                    .arg(format!("--cuda-gpu-arch=sm_{architecture}"))
                    .arg("-nocudainc")
                    .arg("-nocudalib")
                    .arg("-std=c++14")
                    .arg("-O3")
                    .arg(if fast_fp32 {
                        "-ffp-contract=fast"
                    } else {
                        "-ffp-contract=off"
                    })
                    .arg("-S");
                command
            }
        };
        command.arg(source).arg("-o").arg(output);
        run_checked_command(command, "CUDA PTX compilation");
    }
}

struct CudaArtifacts {
    include_dir: PathBuf,
    embedded_sources: [PathBuf; 2],
    fast_fp32: bool,
}

fn run_checked_command(mut command: Command, description: &str) {
    let rendered = format!("{command:?}");
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {description}: {rendered}: {error}"));
    if !output.status.success() {
        panic!(
            "{description} failed with {}\ncommand: {rendered}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

fn verify_generated_proto_contracts() {
    let python = env::var_os("PYTHON")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| OsString::from("python3"));
    let mut command = Command::new(&python);
    command
        .arg("../tools/generate_proto_enums.py")
        .arg("--check");
    run_checked_command(command, "protobuf enum contract check");
    let mut registry_command = Command::new(&python);
    registry_command
        .arg("../tools/generate_kernel_registry.py")
        .arg("--check");
    run_checked_command(registry_command, "kernel registry projection check");
    println!("cargo:rerun-if-env-changed=PYTHON");
    println!("cargo:rerun-if-changed=../tools/generate_proto_enums.py");
    println!("cargo:rerun-if-changed=../tools/generate_kernel_registry.py");
    println!("cargo:rerun-if-changed=../proto/kernel_registry.proto");
    println!("cargo:rerun-if-changed=../native/include/volvoxai_enums.h");
    println!("cargo:rerun-if-changed=../native/include/volvoxai_full_enums.h");
    println!("cargo:rerun-if-changed=../native/src/generated/kernel_registry.h");
    println!("cargo:rerun-if-changed=../native/src/generated/kernel_registry_full.h");
    println!("cargo:rerun-if-changed=../tools/exporter/generated/kernel_registry.py");
    println!("cargo:rerun-if-changed=../tools/exporter/generated/kernel_registry_full.py");
    println!("cargo:rerun-if-changed=../ts/generated/volvoxaiEnums.ts");
    println!("cargo:rerun-if-changed=../ts/generated/volvoxaiFullEnums.ts");
    println!("cargo:rerun-if-changed=../ts/generated/kernelRegistry.ts");
    println!("cargo:rerun-if-changed=../ts/generated/kernelRegistryFull.ts");
    println!("cargo:rerun-if-changed=../docs/generated/kernel-registry.md");
    println!("cargo:rerun-if-changed=generated/c/.synurang-c-lite.manifest.json");
    println!("cargo:rerun-if-changed=generated/c/.synurang-c-native.manifest.json");
    println!("cargo:rerun-if-changed=generated/typescript/.synurang-typescript.manifest.json");
    println!("cargo:rerun-if-changed=src/gen/.synurang-rust.manifest.json");
}

fn command_succeeds(program: &OsString, argument: &str) -> bool {
    Command::new(program)
        .arg(argument)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn disabled_tool_override(value: &OsString) -> bool {
    value.is_empty() || value.to_string_lossy().eq_ignore_ascii_case("off")
}

fn find_cuda_ptx_compiler() -> CudaPtxCompiler {
    let nvcc_override = env::var_os("VOLVOXAI_NVCC_EXECUTABLE");
    if let Some(program) = nvcc_override.as_ref() {
        if !disabled_tool_override(program) {
            if command_succeeds(program, "--version") {
                return CudaPtxCompiler::Nvcc(program.clone());
            }
            panic!(
                "VOLVOXAI_NVCC_EXECUTABLE={} is not an executable nvcc compiler",
                program.to_string_lossy()
            );
        }
    } else {
        let program = OsString::from("nvcc");
        if command_succeeds(&program, "--version") {
            return CudaPtxCompiler::Nvcc(program);
        }
    }

    let mut clang_candidates = Vec::<OsString>::new();
    if let Some(program) = env::var_os("VOLVOXAI_CUDA_CLANGXX_EXECUTABLE") {
        if !disabled_tool_override(&program) {
            clang_candidates.push(program);
        }
    }
    for name in ["clang++", "clang++-17", "clang++-16"] {
        let candidate = OsString::from(name);
        if !clang_candidates.contains(&candidate) {
            clang_candidates.push(candidate);
        }
    }
    for program in clang_candidates {
        let Ok(output) = Command::new(&program).arg("--print-targets").output() else {
            continue;
        };
        let targets = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if output.status.success() && targets.contains("nvptx64") {
            return CudaPtxCompiler::Clang(program);
        }
    }
    panic!(
        "Cargo feature `cuda` requires nvcc or a clang++ build with NVPTX64 support; set VOLVOXAI_NVCC_EXECUTABLE or VOLVOXAI_CUDA_CLANGXX_EXECUTABLE"
    );
}

fn cuda_architecture() -> String {
    let architecture = env::var("VOLVOXAI_CUDA_ARCH").unwrap_or_else(|_| "75".to_string());
    if architecture.is_empty() || !architecture.bytes().all(|byte| byte.is_ascii_digit()) {
        panic!("VOLVOXAI_CUDA_ARCH must be a numeric compute capability such as 75 or 86");
    }
    architecture
}

fn cuda_fast_fp32() -> bool {
    let Some(value) = env::var_os("VOLVOXAI_CUDA_FAST_FP32") else {
        return false;
    };
    match value.to_string_lossy().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "" | "0" | "false" | "no" | "off" => false,
        _ => panic!("VOLVOXAI_CUDA_FAST_FP32 must be one of 0/1, false/true, no/yes, or off/on"),
    }
}

fn embed_cuda_ptx(ptx: &Path, output_c: &Path, output_h: &Path, symbol: Option<&str>) {
    let python = env::var_os("PYTHON")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| OsString::from("python3"));
    let mut command = Command::new(&python);
    command
        .arg("../tools/embed_cuda_ptx.py")
        .arg("--input")
        .arg(ptx)
        .arg("--output-c")
        .arg(output_c)
        .arg("--output-h")
        .arg(output_h);
    if let Some(symbol) = symbol {
        command.arg("--symbol").arg(symbol);
    }
    run_checked_command(command, "deterministic CUDA PTX embedding");
}

fn generate_cuda_artifacts(out_dir: &Path) -> CudaArtifacts {
    let architecture = cuda_architecture();
    let fast_fp32 = cuda_fast_fp32();
    let contract = if fast_fp32 {
        "fast-fma"
    } else {
        "strict-no-fma"
    };
    let generated_dir = out_dir.join("cuda").join(contract);
    fs::create_dir_all(&generated_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create CUDA generation directory {}: {error}",
            generated_dir.display()
        )
    });

    let compiler = find_cuda_ptx_compiler();
    let forward_source = Path::new("../native/src/backends/cuda_kernels.cu");
    let training_source = Path::new("../native/src/backends/cuda_training_kernels.cu");
    let forward_ptx = generated_dir.join("cuda_kernels.ptx");
    let training_ptx = generated_dir.join("cuda_training_kernels.ptx");
    let forward_c = generated_dir.join("embedded_cuda_ptx.c");
    let forward_h = generated_dir.join("embedded_cuda_ptx.h");
    let training_c = generated_dir.join("embedded_cuda_training_ptx.c");
    let training_h = generated_dir.join("embedded_cuda_training_ptx.h");

    compiler.compile(forward_source, &forward_ptx, &architecture, fast_fp32);
    compiler.compile(training_source, &training_ptx, &architecture, fast_fp32);
    embed_cuda_ptx(&forward_ptx, &forward_c, &forward_h, None);
    embed_cuda_ptx(
        &training_ptx,
        &training_c,
        &training_h,
        Some("volvoxai_cuda_training_ptx"),
    );

    println!(
        "cargo:warning=VolvoxAI CUDA FFI plugin PTX: compiler={}, compute_{architecture}, {contract}",
        compiler.name()
    );
    println!("cargo:rerun-if-env-changed=PYTHON");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-changed=../tools/embed_cuda_ptx.py");
    emit_transitive_fragment_rerun_paths(forward_source, true);
    emit_transitive_fragment_rerun_paths(training_source, true);

    CudaArtifacts {
        include_dir: generated_dir,
        embedded_sources: [forward_c, training_c],
        fast_fp32,
    }
}

fn main() {
    verify_generated_proto_contracts();

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
    let target_is_windows = target_os == "windows";
    let target_has_android_dotprod = target_os == "android" && target_arch == "aarch64";
    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo did not provide OUT_DIR to build.rs"));
    let embedded_shaders_c = generate_embedded_shaders(&out_dir);
    let cuda = env::var_os("CARGO_FEATURE_CUDA")
        .is_some()
        .then(|| generate_cuda_artifacts(&out_dir));
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
        "src/runtime/runtime_state.c",
        "src/runtime/public_api.c",
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
        "src/training/trainer_api.c",
        "src/training/ptq_api.c",
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
    if let Some(cuda) = cuda.as_ref() {
        build.include(&cuda.include_dir);
    }
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
    build.flag_if_supported("-fvisibility=hidden");
    build.define("VOLVOXAI_ENABLE_TRAINING", Some("1"));
    build.define(
        "VOLVOXAI_ENABLE_CUDA",
        Some(if cuda.is_some() { "1" } else { "0" }),
    );
    build.define(
        "VOLVOXAI_CUDA_FAST_FP32",
        Some(if cuda.as_ref().is_some_and(|cuda| cuda.fast_fp32) {
            "1"
        } else {
            "0"
        }),
    );
    if target_has_android_dotprod {
        build.define("VOLVOXAI_ARM_DOTPROD_OBJECT", Some("1"));
    }
    build.warnings(false);
    for s in srcs {
        build.file(native.join(s));
        println!("cargo:rerun-if-changed=../native/{s}");
    }
    emit_transitive_fragment_rerun_paths(&native.join("src/runtime/engine.c"), false);
    emit_transitive_fragment_rerun_paths(&native.join("src/runtime/engine_runtime.c"), false);
    build.file(&embedded_shaders_c);
    if let Some(cuda) = cuda.as_ref() {
        build.file(native.join("src/backends/cuda_engine.c"));
        for source in &cuda.embedded_sources {
            build.file(source);
        }
        emit_transitive_fragment_rerun_paths(&native.join("src/backends/cuda_engine.c"), false);
    }
    if target_is_macos {
        build.file(native.join("src/backends/metal_engine.m"));
        println!("cargo:rerun-if-changed=../native/src/backends/metal_engine.m");
    }
    // kernels.c is a portable amalgamation, so track its included units as well
    // as ordinary headers. Otherwise Cargo would not rebuild after editing one
    // of those implementation files.
    for dependency in [
        "include/volvoxai.h",
        "include/volvoxai_enums.h",
        "include/volvoxai_full.h",
        "include/volvoxai_full_enums.h",
        "include/volvoxai_backend.h",
        "src/shader_store.h",
        "src/runtime/adapter_runtime.h",
        "src/runtime/adapter_runtime_internal.h",
        "src/runtime/attention_mask.h",
        "src/runtime/backend.h",
        "src/runtime/backend_sdk.h",
        "src/runtime/backend_config.h",
        "src/runtime/engine_internal.h",
        "src/runtime/incremental_runtime.h",
        "src/runtime/safetensors.h",
        "src/runtime/sequence_runtime.h",
        "src/training/training_core.h",
        "src/training/training_state.h",
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
        "src/backends/cuda_engine.h",
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
    // Re-export the hand-written `vx_*` C API (native/include/volvoxai.h and
    // volvoxai_full.h) from the cdylib alongside the generated `Synurang_*`
    // entry points, so a single libvolvoxai.{so,dylib} serves both tiers: the
    // portable Synurang protobuf FFI (message/copy based) and the direct native
    // C API (raw pointers, zero-copy capable). Without this, rustc localizes
    // every symbol pulled from the static engine archive, so `nm -D` shows only
    // `Synurang_*`. The engine is compiled with -fvisibility=hidden, so only the
    // VX_API-marked `vx_*` symbols have default visibility and land in the export
    // set. See docs/native-runtime.md ("Dual export surface").
    if !target_is_windows {
        if target_is_macos {
            // ld64 uses an exported-symbols list of Mach-O names (leading `_`).
            let list = out_dir.join("exported_symbols.txt");
            fs::write(&list, "_vx_*\n_Synurang_*\n")
                .expect("build.rs: failed to write exported symbols list");
            println!(
                "cargo:rustc-link-arg=-Wl,-exported_symbols_list,{}",
                list.display()
            );
        } else {
            // GNU ld / lld version script for ELF targets (Linux, Android).
            let script = out_dir.join("exports.ver");
            fs::write(&script, "{ global: Synurang_*; vx_*; local: *; };\n")
                .expect("build.rs: failed to write version script");
            println!(
                "cargo:rustc-link-arg=-Wl,--version-script={}",
                script.display()
            );
        }
    }

    println!("cargo:rerun-if-changed=../proto/volvoxai.proto");
    println!("cargo:rerun-if-changed=src/gen/volvoxai_ffi_plugin.rs");
    for variable in [
        "CARGO_FEATURE_CUDA",
        "VOLVOXAI_CUDA_ARCH",
        "VOLVOXAI_CUDA_FAST_FP32",
        "VOLVOXAI_NVCC_EXECUTABLE",
        "VOLVOXAI_CUDA_CLANGXX_EXECUTABLE",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
}
