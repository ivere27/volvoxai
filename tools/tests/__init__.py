"""Shared support for small repository contract checks."""
import re


_NON_CODE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)


def _code(source):
    return _NON_CODE.sub(
        lambda match: ''.join('\n' if c == '\n' else ' ' for c in match[0]), source)


def function_source(source, name, declaration=None):
    """Return one definition, excluding prototypes/calls and preserving bytes.

    ``declaration`` optionally selects an exact return type/storage prefix.
    Comments and quoted literals are masked only for locating C delimiters.
    """
    code = _code(source)
    prefix = (re.escape(_code(declaration)) if declaration is not None else
              r'(?:[A-Za-z_]\w*[ \t*]+)+')
    matches = list(re.finditer(r'(?m)^[ \t]*' + prefix + re.escape(name) +
                              r'\s*\([^;{}]*\)\s*\{', code))
    if len(matches) != 1:
        raise AssertionError(f'Expected one production definition for {name}, found {len(matches)}')
    start = matches[0].start()
    opening = code.index('{', start)
    depth = 0
    for end in range(opening, len(code)):
        depth += (code[end] == '{') - (code[end] == '}')
        if depth == 0:
            return source[start:end + 1] + '\n'
    raise AssertionError('Unterminated production definition: ' + name)
