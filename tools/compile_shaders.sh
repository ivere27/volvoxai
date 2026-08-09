#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
WGSL_DIR="$REPO_ROOT/shaders"
NATIVE_DIR="$REPO_ROOT/native/shaders"
SOURCE_ROOTS=("$WGSL_DIR/inference" "$WGSL_DIR/training")
CARGO_BUILD_DIR="${CARGO_TARGET_DIR:-$NATIVE_DIR/.native-shader-compiler-target}"
NATIVE_SHADER_COMPILER="${VOLVOXAI_NATIVE_SHADER_COMPILER:-}"

if [ -n "$NATIVE_SHADER_COMPILER" ] && [ ! -x "$NATIVE_SHADER_COMPILER" ]; then
    echo "VOLVOXAI_NATIVE_SHADER_COMPILER is not executable: $NATIVE_SHADER_COMPILER" >&2
    exit 1
fi

export LC_ALL=C

for source_root in "${SOURCE_ROOTS[@]}"; do
    if [ ! -d "$source_root" ]; then
        echo "Missing shader source directory: $source_root" >&2
        exit 1
    fi
done

# These are ignored build outputs. Recreate them so removed or renamed entry
# points cannot leave stale native shaders behind.
rm -rf "$NATIVE_DIR/spv" "$NATIVE_DIR/metal" "$NATIVE_DIR/glsl" "$NATIVE_DIR/gles"
mkdir -p "$NATIVE_DIR/spv" "$NATIVE_DIR/metal" "$NATIVE_DIR/glsl" "$NATIVE_DIR/gles"

echo "Compiling WGSL shaders to native formats using Naga..."

# Native artifact names are intentionally flat for the existing backends. Keep
# stems unique across every source directory so a nested source cannot silently
# overwrite another shader's output.
seen_stems=()
seen_paths=()
shader_count=0

compile_source_dir() {
    source_dir=$1
    native_shaders=()
    metal_shaders=()

    # The C-locale glob is deterministic and preserves spaces in filenames.
    for shader in "$source_dir"/*.wgsl; do
        [ -f "$shader" ] || continue

        filename=${shader##*/}
        basename=${filename%.wgsl}
        relative_path=${shader#"$WGSL_DIR"/}

        # Source-owned markers are shared with pack_native_shaders.py. Browser
        # WebGPU may use optional WGSL language features unavailable to every
        # native target. Some packed-dot modules are nevertheless valid SPIR-V
        # and are compiled for feature-gated Vulkan dispatch only.
        IFS= read -r first_line < "$shader" || true
        if [ "$first_line" = "// @volvoxai-browser-only" ]; then
            continue
        fi
        native_spv_only=0
        if [ "$first_line" = "// @volvoxai-native-spv-only" ]; then
            native_spv_only=1
        fi

        index=0
        while [ "$index" -lt "${#seen_stems[@]}" ]; do
            if [ "${seen_stems[$index]}" = "$basename" ]; then
                echo "Duplicate shader stem '$basename': ${seen_paths[$index]} and $relative_path" >&2
                echo "Native shader outputs are flat, so WGSL stems must be unique." >&2
                exit 1
            fi
            index=$((index + 1))
        done
        seen_stems+=("$basename")
        seen_paths+=("$relative_path")
        shader_count=$((shader_count + 1))

        echo "Processing $relative_path..."

        # These are JavaScript-side source templates whose `undefined` marker is
        # replaced with an operation before browser compilation. They are not
        # standalone WGSL modules and therefore have no native artifact.
        case "$basename" in
            binaryBroadcast|elementwise)
                continue
                ;;
        esac

        # Vulkan SPIR-V. Source ownership directories do not change the flat
        # development-override paths consumed by the native backends.
        naga "$shader" "$NATIVE_DIR/spv/${basename}.spv"

        if [ "$native_spv_only" -eq 1 ]; then
            continue
        fi

        native_shaders+=("$filename")

        # Metal is emitted below through Naga's Rust MSL backend with an explicit
        # per-entry-point resource map. These sources use operations unsupported
        # by the MSL backend; invalid source templates were skipped above.
        case "$basename" in
            crossAttention|linearInt8|linearInt8Tiled)
                ;;
            *)
                metal_shaders+=("$filename")
                ;;
        esac
    done

    if [ "${#native_shaders[@]}" -eq 0 ]; then
        return
    fi

    # Naga's CLI exposes neither GLSL `Options.binding_map` nor MSL
    # `Options.per_entry_point_map`. The library-based compiler supplies both
    # maps, so core430, es310, and Metal receive logical WGSL buffer indices.
    # Multi-entry WGSL modules become deterministic `<shader>_<entry>.comp`
    # files. Passing each containing source directory separately also supports
    # WGSL files below future nested inference/training subdirectories while
    # retaining flat native artifact names.
    if [ -n "$NATIVE_SHADER_COMPILER" ]; then
        "$NATIVE_SHADER_COMPILER" \
            glsl "$source_dir" "$NATIVE_DIR/glsl" "$NATIVE_DIR/gles" \
            "${native_shaders[@]}"
    else
        CARGO_TARGET_DIR="$CARGO_BUILD_DIR" cargo run --quiet --locked \
            --manifest-path "$SCRIPT_DIR/metal_shader_compiler/Cargo.toml" -- \
            glsl "$source_dir" "$NATIVE_DIR/glsl" "$NATIVE_DIR/gles" \
            "${native_shaders[@]}"
    fi

    if [ "${#metal_shaders[@]}" -gt 0 ]; then
        if [ -n "$NATIVE_SHADER_COMPILER" ]; then
            "$NATIVE_SHADER_COMPILER" \
                msl "$source_dir" "$NATIVE_DIR/metal" "${metal_shaders[@]}"
        else
            CARGO_TARGET_DIR="$CARGO_BUILD_DIR" cargo run --quiet --locked \
                --manifest-path "$SCRIPT_DIR/metal_shader_compiler/Cargo.toml" -- \
                msl "$source_dir" "$NATIVE_DIR/metal" "${metal_shaders[@]}"
        fi
    fi
}

# Walk source directories in stable order. The compiler accepts basenames, so
# each directory is compiled as one batch rather than flattening source files.
while IFS= read -r source_dir; do
    compile_source_dir "$source_dir"
done < <(find "${SOURCE_ROOTS[@]}" -type d -print | sort)

if [ "$shader_count" -eq 0 ]; then
    echo "No WGSL shaders found under $WGSL_DIR" >&2
    exit 1
fi

echo "Shader compilation complete!"
