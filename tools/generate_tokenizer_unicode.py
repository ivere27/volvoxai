#!/usr/bin/env python3
"""Project pinned Unicode categories and canonical NFC data for C tokenizers."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt'
EXCLUSIONS_URL = 'https://www.unicode.org/Public/17.0.0/ucd/CompositionExclusions.txt'
PROPERTIES_URL = 'https://www.unicode.org/Public/17.0.0/ucd/DerivedCoreProperties.txt'
LICENSE = ROOT / 'native/third_party/unicode/LICENSE.txt'
TARGET = ROOT / 'native/src/generated/tokenizer_unicode.h'
LICENSE_TARGET = ROOT / 'native/src/generated/tokenizer_unicode_license.inc'
SHA256 = '2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c'
EXCLUSIONS_SHA256 = '2f239196ef3b5b61db5cc476e9bd80f534d15aa1b74e1be1dea5d042a344c85f'
PROPERTIES_SHA256 = '24c7fed1195c482faaefd5c1e7eb821c5ee1fb6de07ecdbaa64b56a99da22c08'


def download_pinned(url, expected):
    try:
        with urlopen(url, timeout=30) as response:
            raw = response.read()
    except (OSError, URLError) as error:
        raise SystemExit(f'could not download pinned Unicode data from {url}: {error}') from error
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit(f'pinned Unicode source digest mismatch: {url}')
    return raw


def project(raw, excluded_raw, properties_raw):
    excluded = {int(line.split('#', 1)[0].strip(), 16)
                for line in excluded_raw.decode('utf8').splitlines()
                if line.split('#', 1)[0].strip()}
    ranges = []
    words = []
    combining = []
    decomposition = []
    first = None
    for line in raw.decode('utf8').splitlines():
        fields = line.split(';')
        code = int(fields[0], 16)
        kind = {'L': 1, 'N': 2}.get(fields[2][0], 0)
        ccc = int(fields[3])
        if ccc:
            combining.append((code, ccc))
        mapping = fields[5]
        if mapping and not mapping.startswith('<'):
            parts = [int(p, 16) for p in mapping.split()]
            if len(parts) not in (1, 2):
                raise ValueError('canonical decomposition requires one or two code points')
            decomposition.append((code, parts[0], parts[1] if len(parts) == 2 else 0))
        if fields[1].endswith(', First>'):
            first = code
            continue
        low = first if fields[1].endswith(', Last>') else code
        first = None
        if fields[2][0] == 'M' or fields[2] in ('Nd', 'Pc'):
            words.append((low, code))
        if not kind:
            continue
        if ranges and ranges[-1][1] + 1 == low and ranges[-1][2] == kind:
            ranges[-1] = (ranges[-1][0], code, kind)
        else:
            ranges.append((low, code, kind))
    # Rust regex Unicode \w is Alphabetic + Mark + Nd + Pc + Join_Control.
    # Alphabetic includes some symbols and excludes Number_Other; general
    # L/M/N categories do not reproduce added-token single_word boundaries.
    for line in properties_raw.decode('utf8').splitlines():
        definition = line.split('#', 1)[0].strip()
        if not definition:
            continue
        points, property_name = (part.strip() for part in definition.split(';', 1))
        if property_name != 'Alphabetic':
            continue
        bounds = points.split('..')
        words.append((int(bounds[0], 16), int(bounds[-1], 16)))
    words.append((0x200c, 0x200d))
    word_ranges = []
    for low, high in sorted(words):
        if word_ranges and low <= word_ranges[-1][1] + 1:
            word_ranges[-1] = (word_ranges[-1][0], max(high, word_ranges[-1][1]))
        else:
            word_ranges.append((low, high))
    classes = dict(combining)
    # Full composition exclusions: explicit exclusions, singletons, and
    # decompositions beginning with a nonstarter (UAX #15, Unicode 17.0.0).
    composition = sorted((first, second, point) for point, first, second in decomposition
                         if second and point not in excluded and not classes.get(first, 0))
    if len({(first, second) for first, second, _ in composition}) != len(composition):
        raise ValueError('canonical composition pairs must be unique')
    return ranges, word_ranges, combining, decomposition, composition


def render():
    raw = download_pinned(SOURCE_URL, SHA256)
    excluded_raw = download_pinned(EXCLUSIONS_URL, EXCLUSIONS_SHA256)
    properties_raw = download_pinned(PROPERTIES_URL, PROPERTIES_SHA256)
    ranges, words, combining, decomposition, composition = project(raw, excluded_raw, properties_raw)
    body = '\n'.join(f'    {{0x{low:x}u, 0x{high:x}u, {kind}u}},' for low, high, kind in ranges)
    def table(name, fields, rows):
        entries = '\n'.join('    {' + ', '.join(f'0x{value:x}u' for value in row) + '},' for row in rows)
        return f'static const struct {{ uint32_t {fields}; }} {name}[] = {{\n{entries}\n}};\n'
    extra = (table('vx_unicode_word', 'low, high', words)
             + table('vx_unicode_combining', 'point, kind', combining)
             + table('vx_unicode_decomposition', 'point, first, second', decomposition)
             + table('vx_unicode_composition', 'first, second, point', composition))
    return f'''/* Generated by tools/generate_tokenizer_unicode.py. Do not edit.
 * Unicode 17.0.0, https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt
 * Source SHA-256: {SHA256}
 * Composition exclusions: {EXCLUSIONS_URL}
 * Exclusions SHA-256: {EXCLUSIONS_SHA256}
 * Derived properties: {PROPERTIES_URL}
 * Properties SHA-256: {PROPERTIES_SHA256}
 * Unicode license: native/third_party/unicode/LICENSE.txt */
#ifndef VOLVOXAI_TOKENIZER_UNICODE_H
#define VOLVOXAI_TOKENIZER_UNICODE_H
#include <stdint.h>
static const struct {{ uint32_t low, high, kind; }} vx_unicode_categories[] = {{
{body}
}};
{extra}/* Hangul decomposition/composition is algorithmic and implemented in C. */
#endif
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    notice = LICENSE.read_text()
    assembly = '.section .custom_section.license.unicode,"",@\n'
    for line in notice.splitlines(keepends=True):
        assembly += '.ascii ' + json.dumps(line, ensure_ascii=False) + '\n'
    license_code = ('/* Generated Unicode data license. Do not edit. */\n'
                    '#if defined(__wasm__)\n__asm__(' + json.dumps(assembly) + ');\n'
                    '#else\nstatic const volatile char vx_unicode_license[] =\n' +
                    json.dumps(notice, ensure_ascii=False) + ';\n#endif\n')
    for target, result in [(TARGET, render()), (LICENSE_TARGET, license_code)]:
        if args.check:
            if not target.exists() or target.read_text() != result:
                raise SystemExit('stale tokenizer Unicode projection; run tools/generate_tokenizer_unicode.py')
        else:
            target.write_text(result)


if __name__ == '__main__':
    main()
