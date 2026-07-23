#!/usr/bin/env python3
"""Validate native CUDA implementation-fragment composition and tracking."""

from __future__ import annotations

import glob
import re
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"
RUNTIME = ROOT / "runtime"

CUDA_FORWARD_ROOT = NATIVE / "src/backends/cuda_kernels.cu"
CUDA_TRAINING_ROOT = NATIVE / "src/backends/cuda_training_kernels.cu"
CUDA_HOST_ROOT = NATIVE / "src/backends/cuda_engine.c"
ENGINE_ROOT = NATIVE / "src/runtime/engine.c"
ENGINE_RUNTIME_ROOT = NATIVE / "src/runtime/engine_runtime.c"
CUDA_RUNTIME_ROOT = NATIVE / "src/runtime/engine_runtime_f32_gpu.inc"
CUDA_TRAIN_STEP_ROOT = NATIVE / "src/training/training_step.inc"

INCLUDE_ROOTS = (
    CUDA_FORWARD_ROOT,
    CUDA_TRAINING_ROOT,
    CUDA_HOST_ROOT,
    CUDA_RUNTIME_ROOT,
    CUDA_TRAIN_STEP_ROOT,
)
DEVICE_ROOTS = (CUDA_FORWARD_ROOT, CUDA_TRAINING_ROOT)
CARGO_DEVICE_ROOTS = (CUDA_FORWARD_ROOT,)
CARGO_COMPILED_ROOTS = (CUDA_HOST_ROOT, ENGINE_ROOT, ENGINE_RUNTIME_ROOT)
CARGO_RECURSIVE_ROOTS = (*CARGO_DEVICE_ROOTS, *CARGO_COMPILED_ROOTS)
CMAKE_DEVICE_GLOBS = (
    (CUDA_FORWARD_ROOT, "CUDA_FORWARD_KERNEL_INCLUDES"),
    (CUDA_TRAINING_ROOT, "CUDA_TRAINING_KERNEL_INCLUDES"),
)
CMAKE_CUSTOM_COMMAND_KEYWORDS = frozenset(
    {
        "APPEND",
        "BYPRODUCTS",
        "CODEGEN",
        "COMMAND",
        "COMMAND_EXPAND_LISTS",
        "COMMENT",
        "DEPENDS",
        "DEPFILE",
        "IMPLICIT_DEPENDS",
        "JOB_POOL",
        "JOB_SERVER_AWARE",
        "MAIN_DEPENDENCY",
        "OUTPUT",
        "USES_TERMINAL",
        "VERBATIM",
        "WORKING_DIRECTORY",
    }
)

LOCAL_INC_RE = re.compile(
    r'^[ \t]*\#[ \t]*include[ \t]*"(?P<path>[^"\r\n]+\.inc)"',
    re.MULTILINE,
)
CARGO_LOCAL_INC_RE = re.compile(
    r'^[ \t]*\#include "(?P<path>[^"\r\n]+\.inc)"',
    re.MULTILINE,
)
CMAKE_SET_RE = re.compile(
    r"\bset\s*\(\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?P<value>[^\s()]+)\s*\)",
    re.IGNORECASE,
)
CMAKE_TOKEN_RE = re.compile(r'"(?P<quoted>(?:\\.|[^"\\])*)"|(?P<bare>[^\s()]+)')
RUST_PATH_BINDING_RE = re.compile(
    r"\blet\s+(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*Path::new\s*\("
)
RUST_FUNCTION_RE = re.compile(
    r"\bfn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{"
)


class CompositionError(AssertionError):
    pass


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _strip_c_comments(source: str) -> str:
    """Replace C/C++ comments with spaces while preserving line positions."""
    output: list[str] = []
    index = 0
    state = "code"
    while index < len(source):
        current = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if current == "/" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "line_comment"
                continue
            if current == "/" and following == "*":
                output.extend((" ", " "))
                index += 2
                state = "block_comment"
                continue
            if current == '"':
                state = "string"
            elif current == "'":
                state = "character"
            output.append(current)
            index += 1
            continue
        if state == "line_comment":
            if current == "\n":
                output.append(current)
                state = "code"
            else:
                output.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if current == "*" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "code"
            else:
                output.append("\n" if current == "\n" else " ")
                index += 1
            continue
        output.append(current)
        if current == "\\" and index + 1 < len(source):
            output.append(source[index + 1])
            index += 2
            continue
        if (state == "string" and current == '"') or (
            state == "character" and current == "'"
        ):
            state = "code"
        index += 1
    return "".join(output)


def _mask_c_literals(source: str) -> str:
    """Mask quoted literals without changing offsets in comment-free source."""
    output = list(source)
    index = 0
    state = "code"
    while index < len(source):
        current = source[index]
        if state == "code":
            if current == '"':
                output[index] = " "
                state = "string"
            elif current == "'":
                output[index] = " "
                state = "character"
            index += 1
            continue
        output[index] = "\n" if current == "\n" else " "
        if current == "\\" and index + 1 < len(source):
            output[index + 1] = "\n" if source[index + 1] == "\n" else " "
            index += 2
            continue
        if (state == "string" and current == '"') or (
            state == "character" and current == "'"
        ):
            state = "code"
        index += 1
    return "".join(output)


def _inside_repo(path: Path) -> bool:
    try:
        path.relative_to(ROOT)
    except ValueError:
        return False
    return True


class LocalIncGraph:
    def __init__(self) -> None:
        self._includes: dict[Path, tuple[Path, ...]] = {}

    def includes(self, source: Path) -> tuple[Path, ...]:
        source = source.resolve()
        if source in self._includes:
            return self._includes[source]
        if not source.is_file():
            raise CompositionError(f"missing include-graph root: {_relative(source)}")
        cleaned = _strip_c_comments(source.read_text(encoding="utf-8"))
        dependencies: list[Path] = []
        for match in LOCAL_INC_RE.finditer(cleaned):
            spelling = match.group("path")
            line = cleaned.count("\n", 0, match.start()) + 1
            pure = PurePosixPath(spelling)
            if pure.is_absolute() or "\\" in spelling:
                raise CompositionError(
                    f"{_relative(source)}:{line}: .inc include must use a local "
                    f"POSIX-relative path, got {spelling!r}"
                )
            dependency = (source.parent / Path(*pure.parts)).resolve()
            if not _inside_repo(dependency):
                raise CompositionError(
                    f"{_relative(source)}:{line}: .inc include escapes the repository: "
                    f"{spelling!r}"
                )
            if not dependency.is_file():
                raise CompositionError(
                    f"{_relative(source)}:{line}: missing local .inc include "
                    f"{spelling!r} (resolved to {_relative(dependency)})"
                )
            dependencies.append(dependency)
        self._includes[source] = tuple(dependencies)
        return self._includes[source]

    def dependencies(self, root: Path) -> frozenset[Path]:
        root = root.resolve()
        reached: set[Path] = set()
        visited: set[Path] = set()
        active: list[Path] = []

        def visit(source: Path) -> None:
            if source in active:
                start = active.index(source)
                cycle = active[start:] + [source]
                raise CompositionError(
                    "cyclic local .inc include: "
                    + " -> ".join(_relative(path) for path in cycle)
                )
            if source in visited:
                return
            active.append(source)
            for dependency in self.includes(source):
                reached.add(dependency)
                visit(dependency)
            active.pop()
            visited.add(source)

        visit(root)
        return frozenset(reached)


def _strip_cmake_comments(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines(keepends=True):
        quoted = False
        escaped = False
        kept: list[str] = []
        for character in line:
            if escaped:
                kept.append(character)
                escaped = False
                continue
            if character == "\\" and quoted:
                kept.append(character)
                escaped = True
                continue
            if character == '"':
                quoted = not quoted
            if character == "#" and not quoted:
                kept.extend("\n" if line.endswith("\n") else "")
                break
            kept.append(character)
        lines.append("".join(kept))
    return "".join(lines)


def _expand_cmake_variables(spelling: str, variables: dict[str, str]) -> str:
    pattern = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
    for _ in range(len(variables) + 1):
        expanded = pattern.sub(
            lambda match: variables.get(match.group("name"), match.group(0)),
            spelling,
        )
        if expanded == spelling:
            break
        spelling = expanded
    return spelling


def _cmake_source() -> str:
    return _strip_cmake_comments(
        (NATIVE / "CMakeLists.txt").read_text(encoding="utf-8")
    )


def _cmake_tokens(body: str) -> tuple[str, ...]:
    return tuple(
        match.group("quoted")
        if match.group("quoted") is not None
        else match.group("bare")
        for match in CMAKE_TOKEN_RE.finditer(body)
    )


def _matching_cmake_parenthesis(source: str, start: int) -> int:
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(source)):
        current = source[index]
        if escaped:
            escaped = False
            continue
        if quoted and current == "\\":
            escaped = True
            continue
        if current == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if current == "(":
            depth += 1
        elif current == ")":
            depth -= 1
            if depth == 0:
                return index
    raise CompositionError("unclosed '(' delimiter in native/CMakeLists.txt")


def _cmake_command_bodies(source: str, command: str) -> tuple[str, ...]:
    pattern = re.compile(r"\b" + re.escape(command) + r"\s*\(", re.IGNORECASE)
    bodies: list[str] = []
    for match in pattern.finditer(source):
        start = match.end() - 1
        end = _matching_cmake_parenthesis(source, start)
        bodies.append(source[start + 1 : end])
    return tuple(bodies)


def _cmake_clause(tokens: tuple[str, ...], keyword: str) -> tuple[str, ...]:
    upper = tuple(token.upper() for token in tokens)
    try:
        start = upper.index(keyword.upper()) + 1
    except ValueError:
        return ()
    end = next(
        (
            index
            for index in range(start, len(tokens))
            if upper[index] in CMAKE_CUSTOM_COMMAND_KEYWORDS
        ),
        len(tokens),
    )
    return tokens[start:end]


def _cmake_variables(source: str) -> dict[str, str]:
    variables = {"CMAKE_CURRENT_SOURCE_DIR": NATIVE.as_posix()}
    for match in CMAKE_SET_RE.finditer(source):
        variables[match.group("name")] = _expand_cmake_variables(
            match.group("value"), variables
        )
    return variables


def _cmake_glob_specs(source: str) -> dict[str, tuple[bool, tuple[str, ...]]]:
    specs: dict[str, tuple[bool, tuple[str, ...]]] = {}
    for body in _cmake_command_bodies(source, "file"):
        tokens = _cmake_tokens(body)
        if len(tokens) < 2 or tokens[0].upper() != "GLOB":
            continue
        patterns = tuple(token for token in tokens[2:] if token.endswith(".inc"))
        specs[tokens[1]] = (
            "CONFIGURE_DEPENDS" in {token.upper() for token in tokens[2:]},
            patterns,
        )
    return specs


def _cmake_exclude_filters(source: str) -> dict[str, tuple[str, ...]]:
    filters: dict[str, list[str]] = {}
    for body in _cmake_command_bodies(source, "list"):
        tokens = _cmake_tokens(body)
        if len(tokens) < 5 or tokens[0].upper() != "FILTER":
            continue
        upper = tuple(token.upper() for token in tokens)
        try:
            exclude = upper.index("EXCLUDE", 2)
            regex = upper.index("REGEX", exclude + 1)
        except ValueError:
            continue
        if regex + 1 < len(tokens):
            filters.setdefault(tokens[1], []).append(tokens[regex + 1])
    return {name: tuple(patterns) for name, patterns in filters.items()}


def _cmake_regex(spelling: str) -> re.Pattern[str]:
    # CMake reduces a doubled backslash in a quoted argument to one regex escape.
    return re.compile(spelling.replace("\\\\", "\\"))


def _cmake_inc_paths_by_variable() -> dict[str, frozenset[str]]:
    source = _cmake_source()
    variables = _cmake_variables(source)
    filters = _cmake_exclude_filters(source)
    paths_by_variable: dict[str, frozenset[str]] = {}
    for variable, (_, patterns) in _cmake_glob_specs(source).items():
        declared_paths: set[Path] = set()
        for pattern in patterns:
            spelling = _expand_cmake_variables(pattern, variables)
            if "${" in spelling or "\\" in spelling:
                continue
            pure = PurePosixPath(spelling)
            candidate = (
                Path(*pure.parts) if pure.is_absolute() else NATIVE / Path(*pure.parts)
            )
            matches = (
                glob.glob(candidate.as_posix())
                if glob.has_magic(spelling)
                else [candidate]
            )
            declared_paths.update(Path(match).resolve() for match in matches)
        for pattern in filters.get(variable, ()):
            expression = _cmake_regex(pattern)
            declared_paths = {
                path
                for path in declared_paths
                if not expression.search(path.as_posix())
            }
        paths_by_variable[variable] = frozenset(
            _relative(path)
            for path in declared_paths
            if path.is_file() and _inside_repo(path)
        )
    return paths_by_variable


def _matching_delimiter(source: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    state = "code"
    index = start
    while index < len(source):
        current = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if current == "/" and following == "/":
                state = "line_comment"
                index += 2
                continue
            if current == "/" and following == "*":
                state = "block_comment"
                index += 2
                continue
            if current == '"':
                state = "string"
            elif current == "'":
                state = "character"
            elif current == opening:
                depth += 1
            elif current == closing:
                depth -= 1
                if depth == 0:
                    return index
            index += 1
            continue
        if state == "line_comment":
            if current == "\n":
                state = "code"
            index += 1
            continue
        if state == "block_comment":
            if current == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if current == "\\":
            index += 2
            continue
        if (state == "string" and current == '"') or (
            state == "character" and current == "'"
        ):
            state = "code"
        index += 1
    raise CompositionError(f"unclosed {opening!r} delimiter in runtime/build.rs")


def _next_code_character(source: str, start: int, wanted: str) -> int:
    index = start
    while index < len(source):
        if source[index].isspace():
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                raise CompositionError("unclosed block comment in runtime/build.rs")
            index = end + 2
            continue
        if source[index] == wanted:
            return index
        # Permit iterator adapters or formatting between `]` and the loop body.
        index += 1
    raise CompositionError(f"missing {wanted!r} after array loop in runtime/build.rs")


def _normalize_runtime_path(spelling: str) -> str | None:
    if not spelling or "\\" in spelling or "{" in spelling or "}" in spelling:
        return None
    pure = PurePosixPath(spelling)
    if pure.is_absolute():
        return None
    candidate = (RUNTIME / Path(*pure.parts)).resolve()
    if not _inside_repo(candidate):
        return None
    return _relative(candidate)


def _rust_recursive_tracker_names(source: str) -> frozenset[str]:
    """Find rerun emitters that actually traverse transitive local includes."""
    source = _strip_c_comments(source)
    code = _mask_c_literals(source)
    trackers: set[str] = set()
    for match in RUST_FUNCTION_RE.finditer(code):
        body_start = match.end() - 1
        body_end = _matching_delimiter(code, body_start, "{", "}")
        body = source[body_start + 1 : body_end]
        body_code = code[body_start + 1 : body_end]
        queue_loop = re.search(
            r"\bwhile\s+let\s+Some\s*\(\s*(?:mut\s+)?"
            r"(?P<current>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*=\s*"
            r"(?P<queue>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*pop\s*\(",
            body_code,
        )
        if not queue_loop:
            continue
        loop_start = _next_code_character(body_code, queue_loop.end(), "{")
        loop_end = _matching_delimiter(body_code, loop_start, "{", "}")
        loop = body[loop_start + 1 : loop_end]
        loop_code = body_code[loop_start + 1 : loop_end]
        current = re.escape(queue_loop.group("current"))
        queue = re.escape(queue_loop.group("queue"))
        resolved_child = re.search(
            r"\blet\s+(?P<child>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
            + current
            + r"\s*\.\s*parent\s*\(\s*\)\s*"
            r"(?:\.\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*?\)\s*)*"
            r"\.\s*join\s*\(\s*"
            r"(?P<include>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;",
            loop_code,
            re.DOTALL,
        )
        if not resolved_child:
            continue
        child = re.escape(resolved_child.group("child"))
        include = re.escape(resolved_child.group("include"))

        direct_enqueue = re.search(
            r"\b" + queue + r"\s*\.\s*push\s*\(\s*" + child
            + r"(?:\s*\.\s*clone\s*\(\s*\))?\s*\)",
            loop_code,
        )
        buffered_enqueue = False
        child_push = re.compile(
            r"\b(?P<buffer>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*push\s*"
            r"\(\s*" + child
            + r"(?:\s*\.\s*clone\s*\(\s*\))?\s*\)"
        )
        for push in child_push.finditer(loop_code):
            buffer = re.escape(push.group("buffer"))
            if re.search(
                r"\b"
                + queue
                + r"\s*\.\s*extend\s*\(\s*"
                + buffer
                + r"\s*\.\s*into_iter\s*\(\s*\)"
                r"(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*?\))*"
                r"\s*\)",
                loop_code[push.end() :],
                re.DOTALL,
            ):
                buffered_enqueue = True
                break

        rerun_pattern = re.compile(
            r'\bprintln\s*!\s*\(\s*"cargo:rerun-if-changed=\{\}"\s*,\s*'
            + current
            + r"\s*\.\s*display\s*\(\s*\)\s*\)",
            re.DOTALL,
        )
        reruns_current = any(
            re.match(r"\bprintln\s*!\s*\(", loop_code[rerun.start() :])
            for rerun in rerun_pattern.finditer(loop)
        )
        include_prefix = re.escape('"#include \\""')
        required_operations = (
            reruns_current,
            bool(
                re.search(
                    r"\bread_to_string\s*\(\s*&?\s*" + current + r"\s*\)",
                    loop_code,
                )
            ),
            bool(
                re.search(r"\b" + include + r"\s*\.\s*ends_with\s*\(", loop_code)
                and re.search(
                    r"\b"
                    + include
                    + r"\s*\.\s*ends_with\s*\(\s*\"\.inc\"\s*\)",
                    loop,
                )
            ),
            bool(
                re.search(
                    r"\b" + child + r"\s*\.\s*is_file\s*\(", loop_code
                )
            ),
            bool(
                re.search(
                    r"\bvisited\s*\.\s*insert\s*\(\s*"
                    + current
                    + r"(?:\s*\.\s*clone\s*\(\s*\))?\s*\)",
                    loop_code,
                )
            ),
            bool(
                re.search(
                    r"\.\s*trim_start\s*\(\s*\)\s*\.\s*strip_prefix\s*\(",
                    loop_code,
                )
                and re.search(
                    r"\.\s*trim_start\s*\(\s*\)\s*\.\s*strip_prefix\s*"
                    r"\(\s*" + include_prefix + r"\s*\)",
                    loop,
                )
            ),
            bool(direct_enqueue or buffered_enqueue),
        )
        if all(required_operations):
            trackers.add(match.group("name"))
    return frozenset(trackers)


def _rust_path_variables(source: str, code: str) -> dict[str, str]:
    variables: dict[str, str] = {}
    for match in RUST_PATH_BINDING_RE.finditer(code):
        start = match.end() - 1
        end = _matching_delimiter(code, start, "(", ")")
        argument = _first_rust_argument(source[start + 1 : end])
        literal = re.fullmatch(r'\s*"(?P<path>[^"\r\n]+)"\s*', argument)
        if literal:
            variables[match.group("variable")] = literal.group("path")
    return variables


def _first_rust_argument(arguments: str) -> str:
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    state = "code"
    index = 0
    while index < len(arguments):
        current = arguments[index]
        following = arguments[index + 1] if index + 1 < len(arguments) else ""
        if state == "code":
            if current == "/" and following == "/":
                state = "line_comment"
                index += 2
                continue
            if current == "/" and following == "*":
                state = "block_comment"
                index += 2
                continue
            if current == '"':
                state = "string"
            elif current == "'":
                state = "character"
            elif current in depths:
                depths[current] += 1
            elif current in closing:
                depths[closing[current]] -= 1
            elif current == "," and not any(depths.values()):
                return arguments[:index].strip()
            index += 1
            continue
        if state == "line_comment":
            if current == "\n":
                state = "code"
            index += 1
            continue
        if state == "block_comment":
            if current == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if current == "\\":
            index += 2
            continue
        if (state == "string" and current == '"') or (
            state == "character" and current == "'"
        ):
            state = "code"
        index += 1
    return arguments.strip()


def _resolve_rust_root_expression(
    expression: str, variables: dict[str, str]
) -> str | None:
    expression = expression.strip()
    while expression.startswith("&"):
        expression = expression[1:].lstrip()

    direct = re.fullmatch(r'Path::new\s*\(\s*"([^"\r\n]+)"\s*\)', expression)
    if direct:
        return _normalize_runtime_path(direct.group(1))

    variable = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression)
    if variable:
        spelling = variables.get(expression)
        return _normalize_runtime_path(spelling) if spelling else None

    joined = re.fullmatch(
        r'(?P<base>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*join\s*'
        r'\(\s*"(?P<child>[^"\r\n]+)"\s*\)',
        expression,
    )
    if not joined:
        return None
    base = variables.get(joined.group("base"))
    child = joined.group("child")
    if not base or "\\" in child or PurePosixPath(child).is_absolute():
        return None
    return _normalize_runtime_path((PurePosixPath(base) / child).as_posix())


def _rust_recursive_inc_roots() -> frozenset[str]:
    """Return exact source roots handled by a recursive .inc rerun emitter."""
    source = _strip_c_comments(
        (RUNTIME / "build.rs").read_text(encoding="utf-8")
    )
    code = _mask_c_literals(source)
    trackers = _rust_recursive_tracker_names(source)
    if not trackers:
        return frozenset()

    variables = _rust_path_variables(source, code)
    roots: set[str] = set()
    for tracker in trackers:
        call = re.compile(r"\b" + re.escape(tracker) + r"\s*\(")
        for match in call.finditer(code):
            start = match.end() - 1
            end = _matching_delimiter(code, start, "(", ")")
            expression = _first_rust_argument(source[start + 1 : end])
            normalized = _resolve_rust_root_expression(expression, variables)
            if normalized:
                roots.add(normalized)
    return frozenset(roots)


def _formatted(paths: set[str] | frozenset[str]) -> str:
    return "\n".join(f"  - {path}" for path in sorted(paths))


class CudaSourceCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = LocalIncGraph()

    def _dependencies(self, root: Path) -> frozenset[Path]:
        return self.graph.dependencies(root)

    def _device_dependencies(self) -> frozenset[Path]:
        return frozenset(
            dependency
            for root in DEVICE_ROOTS
            for dependency in self._dependencies(root)
        )

    def test_local_inc_graphs_are_complete_and_acyclic(self) -> None:
        for root in INCLUDE_ROOTS:
            with self.subTest(root=_relative(root)):
                self._dependencies(root)

    def test_device_roots_never_reach_host_fragments(self) -> None:
        reached = self._device_dependencies()
        host_fragments = {
            _relative(path) for path in reached if path.name.endswith("_host.inc")
        }
        self.assertEqual(
            host_fragments,
            set(),
            "CUDA device sources must not include host implementation fragments:\n"
            + _formatted(host_fragments),
        )

    def test_forward_device_root_never_reaches_training_fragments(self) -> None:
        forward = self._dependencies(CUDA_FORWARD_ROOT)
        training = self._dependencies(CUDA_TRAINING_ROOT)
        overlap = {_relative(path) for path in forward & training}
        training_named = {
            _relative(path)
            for path in forward
            if path.name.startswith("cuda_training_")
            or path.name == "cuda_quantize_w8_kernels.inc"
        }
        leaked = overlap | training_named
        self.assertEqual(
            leaked,
            set(),
            "forward CUDA PTX composition reaches training device fragments:\n"
            + _formatted(leaked),
        )

    def test_native_cmake_device_globs_are_dynamic_and_host_filtered(self) -> None:
        source = _cmake_source()
        specs = _cmake_glob_specs(source)
        filters = _cmake_exclude_filters(source)
        for _, variable in CMAKE_DEVICE_GLOBS:
            with self.subTest(variable=variable):
                self.assertIn(
                    variable,
                    specs,
                    f"native/CMakeLists.txt must declare file(GLOB {variable} ...)",
                )
                configure_depends, patterns = specs[variable]
                self.assertTrue(
                    configure_depends,
                    f"{variable} must use CONFIGURE_DEPENDS so new fragments "
                    "rebuild PTX",
                )
                self.assertTrue(
                    patterns,
                    f"{variable} must include at least one .inc glob",
                )
                host_filters = filters.get(variable, ())
                self.assertTrue(
                    any(
                        expression.search("cuda_example_host.inc")
                        and not expression.search("cuda_example_device.inc")
                        and not expression.search("cuda_example_host.inc.backup")
                        for expression in map(_cmake_regex, host_filters)
                    ),
                    f"{variable} must exclude *_host.inc from CUDA PTX dependencies",
                )

    def test_native_cmake_ptx_commands_depend_on_roots_and_globs(self) -> None:
        commands = tuple(
            _cmake_tokens(body)
            for body in _cmake_command_bodies(_cmake_source(), "add_custom_command")
        )
        expected = {
            "${CUDA_PTX}": {
                "${CUDA_KERNEL_SOURCE}",
                "${CUDA_FORWARD_KERNEL_INCLUDES}",
            },
            "${CUDA_TRAINING_PTX}": {
                "${CUDA_TRAINING_KERNEL_SOURCE}",
                "${CUDA_TRAINING_KERNEL_INCLUDES}",
            },
        }
        for output, dependencies in expected.items():
            with self.subTest(output=output):
                matching = [
                    tokens
                    for tokens in commands
                    if output in _cmake_clause(tokens, "OUTPUT")
                ]
                self.assertTrue(
                    matching,
                    f"native/CMakeLists.txt has no custom command producing {output}",
                )
                dependency_sets = [
                    set(_cmake_clause(tokens, "DEPENDS")) for tokens in matching
                ]
                self.assertTrue(
                    any(dependencies <= declared for declared in dependency_sets),
                    f"the {output} command must depend on:\n"
                    + _formatted(dependencies),
                )

    def test_native_cmake_tracks_every_device_include_dependency(self) -> None:
        tracked = _cmake_inc_paths_by_variable()
        for root, variable in CMAKE_DEVICE_GLOBS:
            with self.subTest(root=_relative(root), variable=variable):
                expected = {_relative(path) for path in self._dependencies(root)}
                missing = expected - set(tracked.get(variable, ()))
                self.assertEqual(
                    missing,
                    set(),
                    f"native/CMakeLists.txt {variable} is missing recursive "
                    f"dependencies of {_relative(root)}:\n"
                    + _formatted(missing),
                )

    def test_runtime_build_recursively_tracks_every_compiled_root(self) -> None:
        expected = {_relative(root) for root in CARGO_RECURSIVE_ROOTS}
        missing = expected - set(_rust_recursive_inc_roots())
        self.assertEqual(
            missing,
            set(),
            "runtime/build.rs must call a recursive .inc rerun tracker for every "
            "CUDA/Cargo composition root:\n"
            + _formatted(missing),
        )
        incompatible: set[str] = set()
        sources = {
            source.resolve()
            for root in CARGO_RECURSIVE_ROOTS
            for source in (root, *self._dependencies(root))
        }
        for source in sources:
            cleaned = _strip_c_comments(source.read_text(encoding="utf-8"))
            for include in LOCAL_INC_RE.finditer(cleaned):
                if CARGO_LOCAL_INC_RE.match(cleaned, include.start()):
                    continue
                line = cleaned.count("\n", 0, include.start()) + 1
                incompatible.add(
                    f"{_relative(source)}:{line}: {include.group('path')}"
                )
        self.assertEqual(
            incompatible,
            set(),
            "reachable .inc directives must match runtime/build.rs traversal "
            "syntax (`#include \"relative.inc\"`):\n"
            + _formatted(incompatible),
        )

    def test_runtime_build_tracks_every_device_include_dependency(self) -> None:
        expected = {
            _relative(path)
            for root in CARGO_DEVICE_ROOTS
            for path in self._dependencies(root)
        }
        tracked: set[str] = set()
        recursively_tracked = set(_rust_recursive_inc_roots())
        for root in CARGO_DEVICE_ROOTS:
            if _relative(root) in recursively_tracked:
                tracked.update(_relative(path) for path in self._dependencies(root))
        missing = expected - tracked
        self.assertEqual(
            missing,
            set(),
            "runtime/build.rs is missing recursive CUDA device rerun dependencies:\n"
            + _formatted(missing),
        )

    def test_runtime_build_tracks_reachable_fragments(self) -> None:
        fragments = {
            dependency
            for root in CARGO_COMPILED_ROOTS
            for dependency in self._dependencies(root)
        }
        expected = {_relative(path) for path in fragments}
        tracked: set[str] = set()
        recursively_tracked = set(_rust_recursive_inc_roots())
        for root in CARGO_COMPILED_ROOTS:
            if _relative(root) not in recursively_tracked:
                continue
            tracked.update(_relative(path) for path in self._dependencies(root))
        missing = expected - tracked
        self.assertEqual(
            missing,
            set(),
            "runtime/build.rs is missing Cargo rerun tracking for reachable "
            "implementation fragments:\n"
            + _formatted(missing),
        )


if __name__ == "__main__":
    unittest.main()
