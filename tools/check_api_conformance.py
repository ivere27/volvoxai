#!/usr/bin/env python3
"""Verify that proto/volvoxai.proto is the only public API.

The drift this repository had to unwind was not caused by a bad schema. It was
caused by checks that only verified generated files matched the schema, while
nothing verified that the hand-written surfaces *covered* it. Operations
existed in C and TypeScript that the schema never declared, and the schema
declared failure codes for operations it did not expose.

This gate closes that hole. It fails when:

  1. an RPC declared in the schema has no dispatch entry in the generated C
     header, so a declared operation would be unreachable;
  2. an RPC has no handler assigned in native/src/api/, so a declared
     operation would answer "not implemented" forever;
  3. a handler is assigned for something the schema does not declare;
  4. a public header declares an engine function, which would reintroduce a
     second application API beside the schema;
  5. a public header declares an enum by hand instead of taking the generated
     projection;
  6. a file outside the engine reaches the internal lifecycle without being a
     declared, reasoned exception;
  7. an RPC one host implements is silently missing from the other.

Run with --check in CI; the two modes are identical because the gate never
writes anything.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_proto_enums import FULL_ONLY_SERVICES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "proto" / "volvoxai.proto"
GENERATED_C_HEADER = ROOT / "runtime" / "generated" / "c" / "volvoxai_ffi.h"
GENERATED_TS_CLIENT = (
    ROOT / "runtime" / "generated" / "typescript" / "volvoxai_ffi.ts"
)
GENERATED_PY_CLIENT = ROOT / "runtime" / "generated" / "python" / "volvoxai_client.py"
API_DIR = ROOT / "native" / "src" / "api"
PUBLIC_INCLUDE_DIR = ROOT / "native" / "include"

# The backend SPI has one process-composition function. Keep this an exact
# allowlist so an opaque engine handle or a second lifecycle cannot drift back
# into native/include/ under a broad header exemption.
PUBLIC_SPI_FUNCTIONS = {
    "volvoxai_backend.h": {"vx_backend_register_provider"},
}

# Generated enum projections are the intended way for public headers to carry
# enum values, so they are not hand-written declarations.
GENERATED_ENUM_HEADERS = {
    "volvoxai_enums.h",
    "volvoxai_full_enums.h",
}

# The web surface has no operation implementation: it forwards each service
# into the same generated C dispatcher as native. Keep production TypeScript
# confined to that transport and the device bridge.
TS_TRANSPORT_FILES = {
    'ts/backends/WebGPUHostBridge.ts',
    'ts/core/ModelControlWasm.ts',
    'ts/core/RuntimeErrors.ts',
    'ts/core/WasmReleaseModule.ts',
    'ts/full.ts',
    'ts/generated/gpuBridge.ts',
    'ts/generated/shaderCatalog.ts',
    'ts/generated/shaderCatalogInference.ts',
    'ts/generated/volvoxaiEnums.ts',
    'ts/generated/protoMethods.ts',
    'ts/generated/protoMethodsFull.ts',
    'ts/generated/wasmInternalAbi.ts',
    'ts/host/CallScope.ts',
    'ts/host/EngineHost.ts',
    'ts/host/InferenceWasmHost.ts',
    'ts/host/OperationReports.ts',
    'ts/index.ts',
}

INTERNAL_LIFECYCLE_HEADERS = (
    "vx_lifecycle.h",
    "vx_lifecycle_types.h",
    "vx_training_lifecycle.h",
)

# Intentional consumers below the generated API boundary. These inspect engine
# invariants or measurements that the schema does not declare; routing them
# through protobuf would test the wrong layer. Every application-facing file
# must use the generated dispatch, and any new direct consumer fails the gate.
INTERNAL_LIFECYCLE_CONSUMERS = {
    # Engine selection invariant.
    "native/tests/native_cpu_call_sequence_selection_test.c",
    # Engine-internal baseline measurements.
    "tools/native_runtime_baseline.c",
    # Arena and plan-cache measurements.
    "tools/native_dynamic_shape_performance.c",
    # Scheduler-internal performance measurements.
    "examples/native_dynamic_batch_benchmark/main.c",
}


class Failure(Exception):
    """A conformance violation, reported with the offending location."""


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def snake_case(name: str) -> str:
    """Mirror the generator's method-name conversion (CamelCase -> snake)."""
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper():
            if index:
                out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    return "".join(out)


def native_prefix(service: str) -> str:
    """Mirror the generator's C symbol prefix: strip Service, snake_case."""
    return snake_case(service.removesuffix("Service"))


def declared_rpcs() -> dict[str, list[str]]:
    """Returns {service: [rpc, ...]} exactly as the schema declares them."""
    text = strip_comments(PROTO.read_text())
    services: dict[str, list[str]] = {}
    for match in re.finditer(r"\bservice\s+(\w+)\s*\{(.*?)\n\}", text, flags=re.DOTALL):
        name = match.group(1)
        body = match.group(2)
        rpcs = re.findall(r"\brpc\s+(\w+)\s*\(", body)
        if not rpcs:
            raise Failure(f"service {name} declares no RPCs")
        services[name] = rpcs
    if not services:
        raise Failure(f"{PROTO} declares no services")
    return services


def dispatch_symbols() -> set[str]:
    """C entry points the generated header exposes, without the _pb suffix."""
    text = GENERATED_C_HEADER.read_text()
    symbols = set()
    for match in re.finditer(r"^\w[\w \*]*?\b(\w+)\(", text, flags=re.MULTILINE):
        symbols.add(match.group(1).removesuffix("_respond"))
    return symbols


def assigned_handlers() -> dict[str, set[str]]:
    """Returns {prefix: {handler_field, ...}} from the install functions.

    Handler tables are plain structs, so an assignment `handlers.execute = ...`
    is the authoritative statement that the RPC is implemented.
    """
    assigned: dict[str, set[str]] = {}
    for path in sorted(API_DIR.glob("*.c")):
        text = strip_comments(path.read_text())
        for block in re.finditer(
            r"\b(\w+)ServiceHandlers\s+handlers;(.*?)\breturn|\b(\w+)ServiceHandlers\s+handlers;(.*?)\}",
            text,
            flags=re.DOTALL,
        ):
            service = block.group(1) or block.group(3)
            body = block.group(2) or block.group(4) or ""
            prefix = native_prefix(f"{service}Service")
            fields = set(re.findall(r"handlers\.(\w+)\.\w+\s*=", body))
            assigned.setdefault(prefix, set()).update(fields)
    return assigned


def check_rpc_coverage(services: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    symbols = dispatch_symbols()
    handlers = assigned_handlers()

    for service, rpcs in services.items():
        prefix = native_prefix(service)
        assigned = handlers.get(prefix, set())
        for rpc in rpcs:
            method = snake_case(rpc)
            if f"{prefix}_{method}" not in symbols:
                problems.append(
                    f"{service}.{rpc}: no dispatch entry {prefix}_{method} in "
                    f"{GENERATED_C_HEADER.relative_to(ROOT)} "
                    "(regenerate with make proto_codegen)"
                )
            if method not in assigned:
                problems.append(
                    f"{service}.{rpc}: no handler assigned in "
                    f"{API_DIR.relative_to(ROOT)} "
                    f"(expected handlers.{method} = ...)"
                )
        extra = assigned - {snake_case(rpc) for rpc in rpcs}
        for name in sorted(extra):
            problems.append(
                f"{service}: handlers.{name} is assigned but the schema "
                "declares no such RPC"
            )

    for prefix in sorted(set(handlers) - {native_prefix(s) for s in services}):
        problems.append(
            f"handler table for unknown service prefix '{prefix}' in "
            f"{API_DIR.relative_to(ROOT)}"
        )
    return problems


def check_public_headers() -> list[str]:
    """Public headers must not declare the engine lifecycle or hand-written enums."""
    problems: list[str] = []
    for path in sorted(PUBLIC_INCLUDE_DIR.glob("*.h")):
        text = strip_comments(path.read_text())

        if path.name not in GENERATED_ENUM_HEADERS:
            for match in re.finditer(r"\btypedef\s+enum\s+(\w+)", text):
                name = match.group(1)
                problems.append(
                    f"{path.relative_to(ROOT)}: hand-written enum "
                    f"{name}; declare it in proto/volvoxai.proto and "
                    "take the generated projection"
                )

        if path.name in GENERATED_ENUM_HEADERS:
            continue
        allowed = PUBLIC_SPI_FUNCTIONS.get(path.name, set())
        found: set[str] = set()
        for match in re.finditer(r"^\s*VX_API\b[^;]*?\b(vx_\w+)\s*\(", text, flags=re.MULTILINE):
            name = match.group(1)
            found.add(name)
            if name not in allowed:
                problems.append(
                    f"{path.relative_to(ROOT)}: declares engine function "
                    f"{name}; the public API is proto/volvoxai.proto, so "
                    "lifecycle declarations belong under native/src/"
                )
        for name in sorted(allowed - found):
            problems.append(
                f"{path.relative_to(ROOT)}: allowed provider composition "
                f"function {name} is missing"
            )
    return problems


def check_host_symmetry(services: dict[str, list[str]]) -> list[str]:
    """Every service reaches C; production TS is only API/device transport."""
    problems: list[str] = []
    present = {p.relative_to(ROOT).as_posix() for p in (ROOT / "ts").rglob("*") if p.is_file()}
    for path in sorted(present - TS_TRANSPORT_FILES):
        problems.append(f"{path}: outside the production API/device transport boundary")
    for path in sorted(TS_TRANSPORT_FILES - present):
        problems.append(f"missing transport source {path}")
    transport = (ROOT / "ts/core/ModelControlWasm.ts").read_text()
    if "createWasmHost" not in transport or "synurang_module_create" not in transport:
        problems.append("ModelControlWasm: must use the generated Synurang call runtime")
    for path, declaration in (
        ("ts/full.ts", "class FullEngineHost extends InferenceWasmHost"),
        ("ts/host/EngineHost.ts", "class EngineHost extends InferenceWasmHost"),
    ):
        if declaration not in (ROOT / path).read_text():
            problems.append(f"{path}: must use the persistent C/WASM transport")
    return problems


def check_lifecycle_consumers() -> list[str]:
    """Only declared exceptions may reach the engine lifecycle directly."""
    problems: list[str] = []
    found: set[str] = set()
    roots = ("native", "examples", "tools", "tests", "ts")

    for root in roots:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in {".c", ".h", ".inc", ".cpp"}:
                continue
            relative = path.relative_to(ROOT).as_posix()
            # The engine implements the lifecycle; it is not a consumer of it.
            if relative.startswith("native/src/"):
                continue
            if "/build/" in f"/{relative}" or relative.startswith("build"):
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if not any(f'"{header}"' in text for header in INTERNAL_LIFECYCLE_HEADERS):
                continue
            found.add(relative)
            if relative not in INTERNAL_LIFECYCLE_CONSUMERS:
                problems.append(
                    f"{relative}: reaches the engine lifecycle without a declared "
                    "reason; use proto/volvoxai.proto, or add an entry to "
                    "INTERNAL_LIFECYCLE_CONSUMERS in "
                    "tools/check_api_conformance.py "
                    "explaining why this file belongs below the API"
                )

    for relative in sorted(set(INTERNAL_LIFECYCLE_CONSUMERS) - found):
        problems.append(
            f"{relative}: listed in INTERNAL_LIFECYCLE_CONSUMERS but no longer "
            "reaches the engine lifecycle; remove the entry"
        )
    return problems


def check_typescript_client(services: dict[str, list[str]]) -> list[str]:
    """Every service must reach TypeScript, or web and native surfaces diverge."""
    problems: list[str] = []
    if not GENERATED_TS_CLIENT.is_file():
        return [f"missing {GENERATED_TS_CLIENT.relative_to(ROOT)}"]
    for filename, expected in (
        ('volvoxai_ffi.ts', services),
        ('inference/volvoxai_ffi.ts', {s: rpcs for s, rpcs in services.items()
                            if s not in FULL_ONLY_SERVICES}),
    ):
        path = ROOT / 'runtime/generated/typescript' / filename
        if not path.is_file():
            problems.append(f'missing call client projection {path.relative_to(ROOT)}')
            continue
        source = strip_comments(path.read_text())
        declared = set(re.findall(r'export class (\w+)Client\b', source))
        if declared != set(expected):
            problems.append(f'{filename}: client services differ from the profile schema')
        routed = set(re.findall(r'[\"\']/volvoxai\.v1\.(\w+)/(\w+)[\"\']', source))
        if routed != {(s, rpc) for s, rpcs in expected.items() for rpc in rpcs}:
            problems.append(f'{filename}: client RPC paths differ from the profile schema')
    return problems


def check_python_client(services: dict[str, list[str]]) -> list[str]:
    """Every service must reach Python too.

    Python is not a convenience binding here — it is where PTQ authoring is
    driven from, so a service that never gets a client is a service the
    exporter cannot call. Python preserves acronym runs while C separates
    their capitals (ImportDLPack becomes import_dl_pack / import_d_l_pack).
    """
    problems: list[str] = []
    if not GENERATED_PY_CLIENT.is_file():
        return [f"missing {GENERATED_PY_CLIENT.relative_to(ROOT)}"]
    text = GENERATED_PY_CLIENT.read_text()
    relative = GENERATED_PY_CLIENT.relative_to(ROOT)
    for service, rpcs in services.items():
        if f"class {service}Client" not in text:
            problems.append(f"{service}: no {service}Client client in {relative}")
            continue
        for rpc in rpcs:
            python_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2",
                re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", rpc)).lower()
            if f"def {python_name}(" not in text:
                problems.append(f"{service}.{rpc}: no client method in {relative}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="accepted for symmetry with the other generators; this gate "
        "never writes, so both modes behave identically",
    )
    parser.parse_args()

    try:
        services = declared_rpcs()
    except Failure as error:
        print(f"API conformance failed: {error}", file=sys.stderr)
        return 1

    problems = (
        check_rpc_coverage(services)
        + check_public_headers()
        + check_lifecycle_consumers()
        + check_host_symmetry(services)
        + check_typescript_client(services)
        + check_python_client(services)
    )

    if problems:
        print("API conformance failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    total = sum(len(rpcs) for rpcs in services.values())
    print(
        f"Verified {total} RPCs across {len(services)} services: dispatch, "
        "handlers and TypeScript/Python clients all present; the full web host "
        "routes all RPCs to C; production TypeScript is API/device transport only; "
        "no engine function or hand-written enum in the public "
        f"headers; {len(INTERNAL_LIFECYCLE_CONSUMERS)} intentional internal "
        "lifecycle consumers"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
