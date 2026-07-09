#!/bin/bash
set -e

# Directories
WGSL_DIR="shaders"
NATIVE_DIR="native/shaders"

mkdir -p $NATIVE_DIR/spv
mkdir -p $NATIVE_DIR/metal
mkdir -p $NATIVE_DIR/glsl
mkdir -p $NATIVE_DIR/gles

echo "Compiling WGSL shaders to native formats using Naga..."

patch_glsl_bindings() {
    local glsl_file="$1"
    python3 - "$glsl_file" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

def add_binding(match):
    layout = match.group(1)
    prefix = match.group(2)
    binding = match.group(3)
    if "binding" not in layout:
        layout = f"binding = {binding}, {layout}"
    return f"layout({layout}) {prefix}"

src = re.sub(
    r"layout\(([^)]*)\) ((?:readonly )?buffer [^{]*_block_(\d+)Compute)",
    add_binding,
    src,
)
src = re.sub(
    r"layout\(([^)]*)\) (uniform [^{]*_block_(\d+)Compute)",
    add_binding,
    src,
)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PY
}

patch_msl_bindings() {
    local metal_file="$1"
    python3 - "$metal_file" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

idx = 0
def replace_binding(match):
    global idx
    out = f"[[buffer({idx})]]"
    idx += 1
    return out

src = re.sub(r"\[\[user\(fake\d+\)\]\]", replace_binding, src)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PY
}

for shader in $WGSL_DIR/*.wgsl; do
    filename=$(basename -- "$shader")
    basename="${filename%.*}"
    
    echo "Processing $basename..."
    
    # 1. Vulkan SPIR-V
    naga "$shader" "$NATIVE_DIR/spv/${basename}.spv" || true
    
    # 2. Apple Metal
    if naga "$shader" "$NATIVE_DIR/metal/${basename}.metal"; then
        patch_msl_bindings "$NATIVE_DIR/metal/${basename}.metal"
    fi

    # 3. OpenGL compute GLSL. Naga uses .comp for GLSL compute output.
    if naga --profile core430 --shader-stage compute "$shader" "$NATIVE_DIR/glsl/${basename}.comp"; then
        patch_glsl_bindings "$NATIVE_DIR/glsl/${basename}.comp"
    fi

    # 4. Android OpenGL ES compute GLSL.
    if naga --profile es310 --shader-stage compute "$shader" "$NATIVE_DIR/gles/${basename}.comp"; then
        patch_glsl_bindings "$NATIVE_DIR/gles/${basename}.comp"
    fi
done

echo "Shader compilation complete!"
