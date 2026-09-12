#!/usr/bin/env python3
"""Generate discoverable API contracts and reference docs from volvoxai.proto.

SourceCodeInfo comments may carry one @api JSON object on a single line.
Keys are checked against the corresponding public Api* message; field paths,
rule kinds and effects are checked during generation. Annotations document
semantic rules, while C remains the authority for operation validation.
Synurang itself and its emitted sources are not modified by this generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / 'proto/volvoxai.proto'
PREFIX = 'volvoxai.v1.'
MANIFEST = ROOT / 'native/src/api/generated/api_contract.manifest.json'


def load_dependencies():
    # Native release builds validate committed hashes without installing protoc
    # or Python protobuf. Regeneration and the stronger --check use both.
    global descriptor_pb2, descriptor_pool, message_factory, json_format, F
    global filter_generated_descriptor, INFERENCE_SERVICES, GENERATION_SPECS
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, json_format
    from filter_codegen_request import filter_generated_descriptor
    from generate_proto import INFERENCE_SERVICES, GENERATION_SPECS
    F = descriptor_pb2.FieldDescriptorProto


def descriptor():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'api.pb'
        subprocess.run(['protoc', '--experimental_allow_proto3_optional', '-I', str(PROTO.parent), '--include_source_info',
                        '--descriptor_set_out=' + str(target), str(PROTO)], check=True)
        return descriptor_pb2.FileDescriptorSet.FromString(target.read_bytes()).file[0]


def comments(fd):
    result = {}
    for loc in fd.source_code_info.location:
        prose, annotation = [], {}
        for line in (loc.leading_comments + loc.trailing_comments).splitlines():
            line = line.strip()
            if line.startswith('@api '):
                if annotation:
                    raise ValueError(f'duplicate @api at {list(loc.path)}')
                annotation = json.loads(line[5:])
                if not isinstance(annotation, dict):
                    raise ValueError('@api must be an object')
            else:
                prose.append(line)
        result[tuple(loc.path)] = ('\n'.join(prose).strip(), annotation)
    return result


def contracts(fd, pool):
    notes = comments(fd)
    messages = {m.name: m for m in fd.message_type}
    enums = {e.name: e for e in fd.enum_type}
    types = {}
    methods = []

    def make(kind, values):
        cls = message_factory.GetMessageClass(pool.FindMessageTypeByName(PREFIX + kind))
        return json_format.ParseDict(values, cls())

    def documented(path, allowed):
        description, annotation = notes.get(tuple(path), ('', {}))
        unknown = set(annotation) - set(allowed)
        if unknown:
            raise ValueError(f'unsupported @api keys {unknown} at {path}')
        return {'description': description, **annotation}

    def check_rules(values, owner):
        for rule in values.get('rules', []):
            rule['kind'] = 'API_RULE_KIND_' + rule['kind']
            for path in rule.get('fields', []):
                current = owner
                for part in path.split('.'):
                    field = next((f for f in current.field if f.name == part), None)
                    if field is None:
                        raise ValueError(f'unknown field path {owner.name}.{path}')
                    current = messages.get(field.type_name.removeprefix('.' + PREFIX))
                    if current is None and part != path.split('.')[-1]:
                        raise ValueError(f'non-message field path {owner.name}.{path}')

    for i, msg in enumerate(fd.message_type):
        value = {'name': PREFIX + msg.name, **documented([4, i], ['rules'])}
        check_rules(value, msg)
        fields, dependencies = [], set()
        for j, field in enumerate(msg.field):
            item = {'name': field.name, 'number': field.number,
                    **documented([4, i, 2, j], ['required', 'engine_default', 'handle_kind', 'rules'])}
            item['type'] = (field.type_name.lstrip('.') if field.type_name else
                            F.Type.Name(field.type).removeprefix('TYPE_').lower())
            repeated = field.label == F.LABEL_REPEATED
            item['repeated'] = repeated
            item['has_presence'] = (field.proto3_optional or field.HasField('oneof_index') or
                                    (field.type == F.TYPE_MESSAGE and not repeated))
            if field.HasField('oneof_index') and not field.proto3_optional:
                item['oneof'] = msg.oneof_decl[field.oneof_index].name
            if field.type_name:
                dependencies.add(field.type_name.lstrip('.'))
            default = ([] if repeated else None if field.type == F.TYPE_MESSAGE else
                       enums[field.type_name.removeprefix('.' + PREFIX)].value[0].name if field.type == F.TYPE_ENUM else
                       '' if field.type in (F.TYPE_STRING, F.TYPE_BYTES) else
                       False if field.type == F.TYPE_BOOL else
                       '0' if field.type in (F.TYPE_INT64, F.TYPE_UINT64, F.TYPE_SINT64, F.TYPE_FIXED64, F.TYPE_SFIXED64) else 0)
            item['proto_default'] = json.dumps(default, separators=(',', ':'))
            check_rules(item, msg)
            fields.append(item)
        value['fields'] = fields
        types[value['name']] = ('message', make('ApiMessage', value), sorted(dependencies))
    for i, enum in enumerate(fd.enum_type):
        value = {'name': PREFIX + enum.name, **documented([5, i], [])}
        value['values'] = [dict(name=v.name, number=v.number,
                               **documented([5, i, 2, j], [])) for j, v in enumerate(enum.value)]
        types[value['name']] = ('enum', make('ApiEnum', value), [])
    for i, service in enumerate(fd.service):
        for j, method in enumerate(service.method):
            value = dict(service=service.name, name=method.name,
                         request_type=method.input_type.lstrip('.'),
                         response_type=method.output_type.lstrip('.'),
                         **documented([6, i, 2, j], ['effect', 'rules']))
            if 'effect' in value:
                value['effect'] = 'API_EFFECT_' + value['effect']
            check_rules(value, messages[method.input_type.removeprefix('.' + PREFIX)])
            methods.append(make('ApiMethod', value))
    return methods, types


def c_source(methods, types, sha):
    lines = ['/* Generated by tools/generate_api_contract.py; do not edit. */',
             f'#define VX_API_SCHEMA_SHA256 "{sha}"']
    keys = list(types)
    indices = {key: i for i, key in enumerate(keys)}

    def blob(name, message):
        data = message.SerializeToString(deterministic=True)
        lines.append(f'static const unsigned char {name}[] = {{')
        for i in range(0, len(data), 24):
            lines.append('    ' + ','.join(str(b) for b in data[i:i + 24]) + ',')
        lines.append('};')
    for i, method in enumerate(methods):
        blob(f'vx_api_method_{i}', method)
    for i, (kind, value, deps) in enumerate(types.values()):
        blob(f'vx_api_type_{i}', value)
        if deps:
            lines.append(f'static const size_t vx_api_deps_{i}[] = {{' +
                         ','.join(str(indices[d]) for d in deps) + '};')
    lines.append('static const VxApiMethodEntry vx_api_method_entries[] = {')
    for i, method in enumerate(methods):
        lines.append(f'    {{"{method.service}", "{method.name}", vx_api_method_{i}, sizeof(vx_api_method_{i}), '
                     f'{indices[method.request_type]}, {indices[method.response_type]}}},')
    lines.append('};\nstatic const VxApiTypeEntry vx_api_type_entries[] = {')
    for i, (kind, value, deps) in enumerate(types.values()):
        lines.append(f'    {{{1 if kind == "enum" else 0}, vx_api_type_{i}, sizeof(vx_api_type_{i}), '
                     f'{"vx_api_deps_" + str(i) if deps else "NULL"}, {len(deps)}}},')
    lines.append('};\n')
    return '\n'.join(lines)


def markdown(methods, types, sha, profile):
    lines = [f'# Generated {profile} API contract', '',
             'Source: `proto/volvoxai.proto`. Regenerate with `make proto_codegen`.', '',
             f'Schema SHA-256: `{sha}`.', '',
             'Protobuf defaults and engine defaults are distinct. Required flags and',
             'structured rules are explicit annotations; remaining semantic constraints',
             'are described in the source comments and enforced by C. An absent rule',
             'does not imply that every value or state is accepted.', '',
             '`DescribeApi` returns the same contracts, filtered to the current build.',
             'Use `GetPlatformInfo` and `ListBackends` for build/transport and runtime',
             'backend availability; `CompileModel` decides model-specific admission.', '']
    def rules(values):
        for rule in values:
            lines.append(f'- `{rule.kind}`: ' + ', '.join(f'`{f}`' for f in rule.fields) +
                         (f' — {rule.description}' if rule.description else ''))
        if values:
            lines.append('')
    for method in methods:
        lines += [f'## {method.service}.{method.name}', '',
                  f'`{method.request_type}` → `{method.response_type}`', '',
                  f'Effect: `{method.DESCRIPTOR.fields_by_name["effect"].enum_type.values_by_number[method.effect].name}`.', '',
                  method.description, '']
        rules(method.rules)
    for name, (kind, value, _) in types.items():
        lines += [f'## {name}', '', value.description, '']
        if kind == 'message':
            rules(value.rules)
            for field in value.fields:
                lines += [f'### {field.name} ({field.number})', '',
                          f'`{field.type}`' + (' repeated' if field.repeated else '') +
                          (f'; oneof `{field.oneof}`' if field.oneof else '') +
                          ('; required' if field.required else '') + '.', '',
                          f'Protobuf default: `{field.proto_default}`.', '']
                if field.engine_default:
                    lines += [f'Engine default: {field.engine_default}', '']
                if field.handle_kind:
                    lines += [f'Handle kind: `{field.handle_kind}`.', '']
                if field.description:
                    lines += [field.description, '']
                rules(field.rules)
        else:
            for enum in value.values:
                lines += [f'- `{enum.name} = {enum.number}`' +
                          (f': {enum.description}' if enum.description else '')]
            lines.append('')
    return '\n'.join(lines)


def generate():
    load_dependencies()
    fd = descriptor()
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    sha = hashlib.sha256(PROTO.read_bytes()).hexdigest()
    outputs = {}
    for profile in ('full', 'inference'):
        selected = descriptor_pb2.FileDescriptorProto()
        selected.CopyFrom(fd)
        if profile == 'inference':
            filter_generated_descriptor(selected, set(INFERENCE_SERVICES),
                                        set(GENERATION_SPECS['c-inference'].omitted_enum_values))
        methods, types = contracts(selected, pool)
        outputs[ROOT / f'native/src/api/generated/api_contract_{profile}.inc'] = c_source(methods, types, sha)
        outputs[ROOT / f'docs/generated/api-contract.{profile}.md'] = markdown(methods, types, sha, profile)
        payload = {'schema_sha256': sha, 'profile': profile,
                   'methods': [json_format.MessageToDict(m, preserving_proto_field_name=True) for m in methods],
                   'messages': [json_format.MessageToDict(v, preserving_proto_field_name=True) for k,v,_ in types.values() if k == 'message'],
                   'enums': [json_format.MessageToDict(v, preserving_proto_field_name=True) for k,v,_ in types.values() if k == 'enum']}
        outputs[ROOT / f'docs/generated/api-contract.{profile}.json'] = json.dumps(payload, ensure_ascii=False, indent=2) + '\n'
    inputs = [PROTO, Path(__file__), ROOT / 'tools/filter_codegen_request.py',
              ROOT / 'tools/generate_proto.py']
    manifest = {'inputs': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                'outputs': {str(p.relative_to(ROOT)): hashlib.sha256(s.encode()).hexdigest() for p, s in outputs.items()}}
    outputs[MANIFEST] = json.dumps(manifest, indent=2) + '\n'
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-manifest', action='store_true')
    args = parser.parse_args()
    if args.check_manifest:
        manifest = json.loads(MANIFEST.read_text())
        for group in ('inputs', 'outputs'):
            for name, sha in manifest[group].items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                    raise SystemExit(f'API contract is stale: {name}; run make proto_codegen')
        print('API contract manifest check passed')
        return
    stale = []
    for path, content in generate().items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if stale:
        raise SystemExit('API contract output is stale; run make proto_codegen:\n' + '\n'.join(stale))
    print('API contract generation' + (' check' if args.check else '') + ' passed')


if __name__ == '__main__':
    main()
