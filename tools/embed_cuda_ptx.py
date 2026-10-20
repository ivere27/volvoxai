#!/usr/bin/env python3
"""Deterministically embed CRC32 XZ data for a NUL-terminated CUDA PTX module."""

from __future__ import annotations

import argparse
import hashlib
import lzma
import re
from pathlib import Path


SYMBOL = "volvoxai_cuda_ptx"
SIZE_SYMBOL = "volvoxai_cuda_ptx_size"
BYTES_PER_LINE = 12
MAX_PTX_BYTES = 64 * 1024 * 1024 - 1
_C_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PTX_ENTRY = re.compile(r"\.entry\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", re.DOTALL)
_PTX_PARAM = re.compile(
    r"\.param\s+(?:\.align\s+\d+\s+)?\.([busf])(8|16|32|64)\s+"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\[\s*(\d+)\s*\])?"
)

FORWARD_REGISTRY = Path(__file__).resolve().parents[1] / (
    "native/src/backends/cuda/host/cuda_forward_function_registry_host.inc")


def argument_storage_kernels(registry: str) -> frozenset[str]:
    """Read the host registry's optional-record policy, not its argument ABI."""
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", registry, flags=re.DOTALL)
    pattern = re.compile(
        r"VOLVOXAI_CUDA_FORWARD_FUNCTION\(\s*(?:REQUIRED|DEFERRED)\s*,\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*,\s*\"([A-Za-z_][A-Za-z0-9_]*)\"\s*,\s*([01])\s*\)")
    entries = pattern.findall(source)
    if not entries or pattern.sub("", source).strip():
        raise ValueError("invalid CUDA forward registry or argument-storage policy")
    handles, symbols = set(), set()
    prepared = set()
    for handle, symbol, storage in entries:
        if handle in handles or symbol in symbols:
            raise ValueError("duplicate CUDA forward registry handle or symbol")
        handles.add(handle)
        symbols.add(symbol)
        if storage == "1":
            prepared.add(symbol)
    return frozenset(prepared)


# Unknown entries and by-value aggregates retain ordinary argument updates.
ARGUMENT_STORAGE_KERNELS = argument_storage_kernels(FORWARD_REGISTRY.read_text())


def input_hash(ptx: bytes) -> str:
    """Return the stable digest recorded in both generated files."""

    return hashlib.sha256(ptx).hexdigest()


def validate_ptx(ptx: bytes) -> None:
    if not ptx:
        raise ValueError("CUDA PTX input is empty")
    if len(ptx) > MAX_PTX_BYTES:
        raise ValueError("CUDA PTX exceeds the bounded embedded decoder span")
    if b"\x00" in ptx:
        raise ValueError("CUDA PTX input already contains a NUL byte")


def validate_symbol(symbol: str) -> None:
    if not _C_IDENTIFIER.fullmatch(symbol):
        raise ValueError(f"invalid C symbol: {symbol!r}")


def size_symbol(symbol: str) -> str:
    validate_symbol(symbol)
    return SIZE_SYMBOL if symbol == SYMBOL else f"{symbol}_size"


def header_guard(symbol: str) -> str:
    validate_symbol(symbol)
    if symbol == SYMBOL:
        return "VOLVOXAI_EMBEDDED_CUDA_PTX_H"
    return f"VOLVOXAI_EMBEDDED_{symbol.upper()}_H"


def kernel_argument_abis(ptx: bytes) -> list[tuple[str, tuple[int, ...] | None]]:
    """Read actual compiler parameter widths, refusing unknown PTX forms.

    Each argument image uses eight-byte aligned starts so it can also supply
    the Driver's kernelParams pointers without borrowing a caller's stack.
    By-value aggregates retain the compiler's exact byte width.
    """
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx.decode("utf-8"), flags=re.DOTALL)
    entries = []
    seen = set()
    for name, parameters in _PTX_ENTRY.findall(source):
        if name in seen:
            raise ValueError(f"duplicate PTX entry: {name}")
        seen.add(name)
        sizes = []
        supported = True
        for parameter in parameters.split(","):
            if not parameter.strip():
                continue
            parsed = _PTX_PARAM.fullmatch(parameter.strip())
            if not parsed:
                supported = False
                break
            size = int(parsed.group(2)) // 8 * int(parsed.group(3) or 1)
            if not 0 < size <= 65535:
                supported = False
                break
            sizes.append(size)
        packed = sum((size + 7) & ~7 for size in sizes)
        if len(sizes) > 64 or packed > 65535:
            supported = False
        entries.append((name, tuple(sizes) if supported else None))
    return entries


def render_argument_abis(ptx: bytes) -> str:
    entries = kernel_argument_abis(ptx)
    declarations = []
    rows = []
    for index, (name, sizes) in enumerate(entries):
        storage = int(name in ARGUMENT_STORAGE_KERNELS)
        if storage and (not sizes or len(sizes) < 2 or sizes[-1] != 8
                        or any(size > 8 for size in sizes[:-1])):
            raise ValueError(f"invalid prepared argument storage ABI: {name}")
        if sizes:
            declarations.append(
                f"static const uint16_t vx_cuda_argument_sizes_{index}[] = {{"
                + ", ".join(str(size) for size in sizes) + "};"
            )
        size_pointer = f"vx_cuda_argument_sizes_{index}" if sizes else "NULL"
        count = len(sizes) if sizes is not None else 65535
        packed = sum((size + 7) & ~7 for size in sizes or ())
        rows.append(f'    {{"{name}", {size_pointer}, {count}u, {packed}u, {storage}u}},')
    # An empty/unknown module still embeds normally; it cannot provide an
    # argument ABI that would authorize graph replay.
    if not rows:
        rows.append("    {NULL, NULL, 65535u, 0u, 0u},")
    return (
        "\n" + "\n".join(declarations) + "\n\n"
        "const VxCudaKernelArgumentABI volvoxai_cuda_argument_abi[] = {\n"
        + "\n".join(rows) + "\n};\n\n"
        f"const size_t volvoxai_cuda_argument_abi_count = {len(entries)}u;\n"
    )


def render_header(digest: str, symbol: str = SYMBOL) -> str:
    guard = header_guard(symbol)
    payload_size_symbol = size_symbol(symbol)
    argument_abi = "" if symbol != SYMBOL else """
/* Parameter widths are generated from the very PTX payload above. Unknown
 * forms use count=UINT16_MAX and never authorize graph replay. */
#include <stdint.h>
typedef struct {
    const char* symbol;
    const uint16_t* sizes;
    uint16_t count;
    uint16_t bytes;
    /* Full count includes a trailing eight-byte pointer when storage=1. */
    uint16_t storage;
} VxCudaKernelArgumentABI;
extern const VxCudaKernelArgumentABI volvoxai_cuda_argument_abi[];
extern const size_t volvoxai_cuda_argument_abi_count;
"""
    return f"""/* DO NOT EDIT: generated by tools/embed_cuda_ptx.py.
 * Input SHA-256: {digest}
 */
#ifndef {guard}
#define {guard}

#include <stddef.h>

/* CRC32 XZ payload expands to the exact PTX plus one trailing NUL. */
extern const unsigned char {symbol}_xz[];
extern const size_t {symbol}_xz_size;
/* Decompressed PTX bytes, excluding the trailing NUL. */
extern const size_t {payload_size_symbol};
{argument_abi}

#endif
"""


def render_source(
    ptx: bytes, digest: str, header_name: str, symbol: str = SYMBOL
) -> str:
    validate_ptx(ptx)
    validate_symbol(symbol)
    payload_size_symbol = size_symbol(symbol)
    encoded = lzma.compress(ptx + b"\x00", format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC32, filters=[{
            "id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME,
            "dict_size": 1 << 20,
            # PTX is text; these literal/position contexts reduce the stored
            # module without changing its bytes or the decoder memory bound.
            "lc": 2, "lp": 0, "pb": 0,
        }])
    lines = []
    for offset in range(0, len(encoded), BYTES_PER_LINE):
        chunk = encoded[offset : offset + BYTES_PER_LINE]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    body = "\n".join(lines)
    source = f"""/* DO NOT EDIT: generated by tools/embed_cuda_ptx.py.
 * Input SHA-256: {digest}
 */
#include \"{header_name}\"

const unsigned char {symbol}_xz[] = {{
{body}
}};

const size_t {symbol}_xz_size = sizeof({symbol}_xz);
const size_t {payload_size_symbol} = {len(ptx)}u;
"""
    return source + (render_argument_abis(ptx) if symbol == SYMBOL else "")


def write_embedded_ptx(
    ptx_path: Path, output_c: Path, output_h: Path, symbol: str = SYMBOL
) -> None:
    ptx = ptx_path.read_bytes()
    validate_ptx(ptx)
    validate_symbol(symbol)
    digest = input_hash(ptx)
    header = render_header(digest, symbol)
    source = render_source(ptx, digest, output_h.name, symbol)

    output_c.parent.mkdir(parents=True, exist_ok=True)
    output_h.parent.mkdir(parents=True, exist_ok=True)
    output_h.write_text(header, encoding="utf-8", newline="\n")
    output_c.write_text(source, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input CUDA PTX file")
    parser.add_argument("--output-c", required=True, type=Path,
                        help="generated C byte-array source")
    parser.add_argument("--output-h", required=True, type=Path,
                        help="generated C declaration header")
    parser.add_argument("--symbol", default=SYMBOL,
                        help="C byte-array symbol (default: %(default)s)")
    args = parser.parse_args()
    write_embedded_ptx(args.input, args.output_c, args.output_h, args.symbol)


if __name__ == "__main__":
    main()
