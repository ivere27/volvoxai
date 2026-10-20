import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_proto as codegen


def tar_archive(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return data.getvalue()


class GeneratorReleaseTests(unittest.TestCase):
    target = "x86_64-unknown-linux-musl"

    def test_changed_typescript_writer_requires_review(self):
        with self.assertRaisesRegex(codegen.CodegenError, "bulk writer needs review"):
            codegen.typescript_bulk_writer(b"class ProtoWriter { /* unfamiliar upstream codec */ }")

    def test_changed_typescript_string_reader_requires_review(self):
        reader = b'''  readString(): string {
    return new TextDecoder().decode(this.readLengthDelimited());
  }
'''
        for changed in (b"", reader + reader,
                        reader.replace(b"readLengthDelimited()", b"readDifferentBytes()")):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(codegen.CodegenError, "string reader needs review"):
                    codegen.typescript_string_reader(changed)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        self.binary = b"verified generator fixture"
        self.archive = tar_archive({
            f"{codegen.GENERATOR_NAME}-{codegen.SYNURANG_VERSION}-{self.target}/"
            f"{codegen.GENERATOR_NAME}": self.binary,
        })
        digest = hashlib.sha256(self.archive).hexdigest()
        patches = [
            mock.patch.object(codegen, "generator_target", return_value=self.target),
            mock.patch.dict(codegen.GENERATOR_ARCHIVES, {self.target: ("tar.gz", digest)}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.archive_path = (self.cache / f"v{codegen.SYNURANG_VERSION}" / "archives"
                             / codegen.generator_archive_name(self.target))
        self.archive_path.parent.mkdir(parents=True)
        self.archive_path.write_bytes(self.archive)

    def test_offline_extracts_verified_release_without_network(self):
        with mock.patch.object(codegen.urllib.request, "urlopen") as network:
            executable = codegen.release_generator(self.cache, True)
            self.assertEqual(executable.read_bytes(), self.binary)
            self.assertEqual(codegen.release_generator(self.cache, True), executable)
        network.assert_not_called()

    def test_modified_binary_is_restored_from_verified_archive(self):
        executable = codegen.release_generator(self.cache, True)
        executable.write_bytes(b"x" * len(self.binary))
        self.assertEqual(codegen.release_generator(self.cache, True).read_bytes(), self.binary)

    def test_modified_archive_is_rejected_even_with_cached_binary(self):
        codegen.release_generator(self.cache, True)
        self.archive_path.write_bytes(b"modified release")
        with self.assertRaisesRegex(codegen.CodegenError, "SHA-256 mismatch"):
            codegen.release_generator(self.cache, True)

    def test_missing_offline_archive_does_not_download(self):
        self.archive_path.unlink()
        with mock.patch.object(codegen.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(codegen.CodegenError, "offline cache is missing"):
                codegen.release_generator(self.cache, True)
        network.assert_not_called()

    def test_symlink_cache_is_rejected(self):
        source = self.cache / "release.tar.gz"
        self.archive_path.rename(source)
        self.archive_path.symlink_to(source)
        with self.assertRaisesRegex(codegen.CodegenError, "not a regular file"):
            codegen.release_generator(self.cache, True)

    def test_download_verified_before_cache_write(self):
        self.archive_path.unlink()
        with mock.patch.object(codegen.urllib.request, "urlopen",
                               return_value=io.BytesIO(b"corrupt download")):
            with self.assertRaisesRegex(codegen.CodegenError, "SHA-256 mismatch"):
                codegen.release_generator(self.cache, False)
        self.assertFalse(self.archive_path.exists())

    def test_download_can_be_reused_offline(self):
        self.archive_path.unlink()
        with mock.patch.object(codegen.urllib.request, "urlopen",
                               return_value=io.BytesIO(self.archive)) as network:
            executable = codegen.release_generator(self.cache, False)
        self.assertEqual(network.call_count, 1)
        self.assertEqual(codegen.release_generator(self.cache, True), executable)

    def test_fetch_only_also_populates_runtime_source_cache(self):
        with mock.patch.object(codegen, "resolve_generator", return_value=Path("fixture")):
            with mock.patch.object(codegen, "source_archive_bytes") as source:
                self.assertEqual(codegen.main([
                    "--fetch-only", "--offline", "--cache-dir", str(self.cache),
                ]), 0)
        source.assert_called_once_with(self.cache.resolve(), True)


class GeneratorArchiveTests(unittest.TestCase):
    def test_selects_only_published_platform_assets(self):
        for system, machine, expected in [
            ("Linux", "AMD64", "x86_64-unknown-linux-musl"),
            ("Linux", "arm64", "aarch64-unknown-linux-musl"),
            ("Windows", "AMD64", "x86_64-pc-windows-gnu"),
        ]:
            with self.subTest(system=system, machine=machine):
                with mock.patch.object(codegen.platform, "system", return_value=system):
                    with mock.patch.object(codegen.platform, "machine", return_value=machine):
                        self.assertEqual(codegen.generator_target(), expected)
        with mock.patch.object(codegen.platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(codegen.CodegenError, "use --generator"):
                codegen.generator_target()

    def test_extracts_windows_release_executable(self):
        target = "x86_64-pc-windows-gnu"
        data = io.BytesIO()
        with zipfile.ZipFile(data, mode="w") as archive:
            archive.writestr(
                f"{codegen.GENERATOR_NAME}-{codegen.SYNURANG_VERSION}-{target}/"
                f"{codegen.GENERATOR_NAME}.exe", b"Windows generator fixture",
            )
        name, binary = codegen.extract_generator(data.getvalue(), target)
        self.assertEqual(name, codegen.GENERATOR_NAME + ".exe")
        self.assertEqual(binary, b"Windows generator fixture")

    def test_source_paths_cannot_escape_tag_root(self):
        prefix = f"synurang-{codegen.SYNURANG_VERSION}/"
        for path in (prefix + "../outside", "other/source.c"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(codegen.CodegenError, "invalid Synurang source path"):
                    codegen.source_files(tar_archive({path: b"fixture"}))

    def test_source_archive_keeps_upstream_bytes(self):
        prefix = f"synurang-{codegen.SYNURANG_VERSION}/"
        source = b"/* upstream runtime */\n"
        self.assertEqual(codegen.source_files(tar_archive({prefix + "src/call.c": source})),
                         {"src/call.c": source})

    def test_manifest_owned_legacy_runtime_files_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "synurang/plugin.py"
            legacy.parent.mkdir()
            legacy.write_text("# old upstream runtime, without a generated marker\n")
            codegen.manifest_path(root, "python").write_text(json.dumps({
                "files": ["synurang/plugin.py"],
            }))
            result = codegen.GenerationResult(
                files={}, mode_files={"python": ()}, mode_roots={"python": root},
            )
            self.assertEqual(codegen.obsolete_generated_paths(result, ("python",)), [legacy])

    def test_obsolete_symlink_does_not_delete_its_unowned_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "user.py"
            target.write_text("# user file\n")
            legacy = root / "plugin.py"
            legacy.symlink_to(target)
            codegen.manifest_path(root, "python").write_text(json.dumps({
                "files": ["plugin.py"],
            }))
            result = codegen.GenerationResult(
                files={}, mode_files={"python": ()}, mode_roots={"python": root},
            )
            codegen.install_or_check(result, ("python",), False)
            self.assertFalse(legacy.exists())
            self.assertTrue(target.is_file())


class PythonFreshMessageTests(unittest.TestCase):
    """Exercise the generated codec without importing a transport or native host."""

    def setUp(self):
        path = codegen.REPOSITORY_ROOT / "runtime/generated/python/synurang/protolite.py"
        source = path.read_bytes()
        self.codecs = []
        for variant in ("original", "generated"):
            name = f"_codegen_codec_{variant}_{id(self)}"
            codec = types.ModuleType(name)
            sys.modules[name] = codec
            self.addCleanup(sys.modules.pop, name, None)
            exec(compile(source, str(path), "exec"), codec.__dict__)
            if variant == "original":
                # The pinned runtime's original fresh-message API dispatch.
                def original_from_string(cls, data):
                    message = cls()
                    message.ParseFromString(data)
                    return message
                codec.ProtoMessage.FromString = classmethod(original_from_string)
            child = type("Child", (codec.ProtoMessage,), {
                "__proto_name__": "test.Child",
                "__fields__": (
                    codec.Field(1, "count", "int32"),
                    codec.Field(2, "values", "sint32", repeated=True),
                    codec.Field(3, "text", "string", oneof="choice"),
                    codec.Field(4, "token", "int32", oneof="choice"),
                ),
            })
            envelope = type("Envelope", (codec.ProtoMessage,), {
                "__proto_name__": "test.Envelope",
                "__fields__": (
                    codec.Field(1, "child", "message", type_name="test.Child"),
                    codec.Field(2, "children", "message", repeated=True,
                                type_name="test.Child"),
                    codec.Field(3, "entries", "message", is_map=True,
                                map_key_kind="string", map_value_kind="message",
                                map_value_type_name="test.Child"),
                ),
            })
            self.codecs.append((codec, child, envelope))

    def state(self, value):
        if any(isinstance(value, codec.ProtoMessage) for codec, _, _ in self.codecs):
            return {key: self.state(item) for key, item in value.__dict__.items()
                    if not callable(item)}
        if isinstance(value, dict):
            return {key: self.state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.state(item) for item in value]
        return value

    def test_changed_or_duplicated_upstream_parser_requires_review(self):
        original = b'''    @classmethod
    def FromString(cls: type[MessageT], data: bytes | bytearray | memoryview) -> MessageT:
        message = cls()
        message.ParseFromString(data)
        return message
'''
        for source in (b"unfamiliar codec", original + original):
            with self.subTest(source=source):
                with self.assertRaisesRegex(codegen.CodegenError, "parser needs review"):
                    codegen.python_fresh_message_parser(source)

    def test_nested_duplicates_merge_repeated_append_and_map_last_wins(self):
        _, child, envelope = self.codecs[0]
        payload = (envelope(child=child(count=9, values=[-2]),
                            children=[child(count=1)],
                            entries={"same": child(count=2)}).to_bytes()
                   + envelope(child=child(values=[1, 3]),
                              children=[child(count=4)],
                              entries={"same": child(count=8)}).to_bytes())
        for input_type in (bytes, bytearray, memoryview):
            decoded = [env.from_bytes(input_type(payload)) for _, _, env in self.codecs]
            self.assertEqual(self.state(decoded[0]), self.state(decoded[1]))
            self.assertEqual(decoded[0].to_bytes(), decoded[1].to_bytes())
            self.assertEqual(decoded[1].child.count, 9)
            self.assertEqual(decoded[1].child.values, [-2, 1, 3])
            self.assertEqual([item.count for item in decoded[1].children], [1, 4])
            self.assertEqual(decoded[1].entries["same"].count, 8)

    def test_oneof_last_wins_and_explicit_empty_presence(self):
        payloads = (b"\x1a\x05first\x20\x07", b"\x1a\x00", b"")
        for payload in payloads:
            decoded = [child.parseFrom(payload) for _, child, _ in self.codecs]
            self.assertEqual(self.state(decoded[0]), self.state(decoded[1]))
        parsed = self.codecs[1][1].FromString(payloads[0])
        self.assertEqual(parsed.WhichOneof("choice"), "token")
        self.assertEqual(parsed.token, 7)
        self.assertEqual(parsed.text, "")
        empty = self.codecs[1][1].from_bytes(payloads[1])
        self.assertEqual(empty.WhichOneof("choice"), "text")
        self.assertTrue(empty.HasField("text"))

    def test_packed_unpacked_unknown_and_malformed_wire(self):
        # sint32 values: unpacked -2, followed by packed 1 and 3.
        payload = b"\x10\x03\x12\x02\x02\x06\xf8\x1f\x01"
        for _, child, _ in self.codecs:
            self.assertEqual(child.from_bytes(payload).values, [-2, 1, 3])
        for payload in (b"\x00", b"\x80", b"\x1a\x03ab", b"\x1a\x01\xff", b"\x0f"):
            failures = []
            for _, child, _ in self.codecs:
                with self.assertRaises(ValueError) as caught:
                    child.from_bytes(payload)
                failures.append((type(caught.exception).__name__, str(caught.exception)))
            self.assertEqual(failures[0], failures[1])
        failures = []
        for _, _, envelope in self.codecs:
            with self.assertRaises(ValueError) as caught:
                envelope.from_bytes(b"\x0a\x01\x00")
            failures.append((type(caught.exception).__name__, str(caught.exception)))
        self.assertEqual(failures[0], failures[1])

    def test_existing_instance_parse_still_clears(self):
        for _, child, _ in self.codecs:
            message = child(count=9, values=[1, 2], text="old")
            self.assertEqual(message.ParseFromString(b""), 0)
            self.assertEqual(message.count, 0)
            self.assertEqual(message.values, [])
            self.assertIsNone(message.WhichOneof("choice"))

    def test_custom_dictionary_descriptor_does_not_change_parsing(self):
        states = []
        for codec, child, _ in self.codecs:
            class CustomDictionary(child):
                @property
                def __dict__(self):
                    raise RuntimeError("custom dictionary descriptor")

            payload = child(count=3, values=[-2, 7], token=9).to_bytes()
            parsed = CustomDictionary.from_bytes(payload)
            self.assertEqual(parsed.to_bytes(), payload)
            states.append((parsed.count, parsed.values, parsed.WhichOneof("choice"), parsed.token))
            self.assertEqual(CustomDictionary.from_bytes(b"").values, [])
            with self.assertRaises(codec.DecodeError):
                CustomDictionary.from_bytes(b"\x00")
        self.assertEqual(states[0], states[1])

    def test_static_base_method_aliases_preserve_instance_dispatch_errors(self):
        for method in ("ParseFromString", "Clear"):
            failures = []
            for codec, _, _ in self.codecs:
                base = codec.ProtoMessage
                alias = type("Static" + method, (base,), {
                    method: staticmethod(getattr(base, method)),
                })
                with self.assertRaises(TypeError) as caught:
                    alias.FromString(b"")
                failures.append((type(caught.exception).__name__, str(caught.exception)))
            self.assertEqual(failures[0], failures[1])

    def test_custom_construction_and_dispatch_preserved(self):
        results = []
        for codec, child, _ in self.codecs:
            base = codec.ProtoMessage

            class Seeded(child):
                def __init__(self):
                    super().__init__()
                    self.values = [999]

            class CustomParse(child):
                def ParseFromString(self, data):
                    self.parse_hook = True
                    return super().ParseFromString(data)

            class CustomClear(child):
                def Clear(self):
                    self.clear_hook = True
                    return super().Clear()

            class InstanceHook(child):
                def __new__(cls):
                    message = super().__new__(cls)
                    def parse(data):
                        message.instance_hook = True
                        return base.ParseFromString(message, data)
                    message.ParseFromString = parse
                    return message

            class AttributeHook(child):
                def __getattribute__(self, name):
                    if name == "ParseFromString":
                        def parse(data):
                            self.attribute_hook = True
                            return base.ParseFromString(self, data)
                        return parse
                    return super().__getattribute__(name)

            class CustomMeta(type):
                def __call__(cls):
                    message = super().__call__()
                    def clear():
                        message.metaclass_hook = True
                        return base.Clear(message)
                    message.Clear = clear
                    return message

            class MetaclassHook(child, metaclass=CustomMeta):
                pass

            classes = (Seeded, CustomParse, CustomClear, InstanceHook,
                       AttributeHook, MetaclassHook)
            parsed = [cls.FromString(b"") for cls in classes]
            self.assertEqual(parsed[0].values, [])
            for message, hook in zip(parsed[1:], ("parse_hook", "clear_hook",
                                                 "instance_hook", "attribute_hook",
                                                 "metaclass_hook")):
                self.assertTrue(getattr(message, hook))
            results.append([self.state(message) for message in parsed])
            for method in ("Clear", "ParseFromString"):
                shadow = type("Shadow", (base,), {
                    "__fields__": (codec.Field(1, method, "string"),),
                })
                with self.assertRaises(TypeError):
                    shadow.FromString(b"")
        self.assertEqual(results[0], results[1])


class PythonVarintReaderTests(unittest.TestCase):
    """Validate values, byte ownership and malformed wire at the generated reader."""

    def setUp(self):
        path = codegen.REPOSITORY_ROOT / "runtime/generated/python/synurang/protolite.py"
        name = f"_codegen_varint_{id(self)}"
        self.codec = types.ModuleType(name)
        sys.modules[name] = self.codec
        self.addCleanup(sys.modules.pop, name, None)
        exec(compile(path.read_bytes(), str(path), "exec"), self.codec.__dict__)

    def test_known_uint64_values_and_exact_next_byte(self):
        cases = [(0, b"\0"), (127, b"\x7f"), (128, b"\x80\x01"),
                 (16383, b"\xff\x7f"), (16384, b"\x80\x80\x01"),
                 ((1 << 32) - 1, b"\xff" * 4 + b"\x0f"),
                 (1 << 63, b"\x80" * 9 + b"\x01"),
                 ((1 << 64) - 1, b"\xff" * 9 + b"\x01")]
        cases.extend((value, bytes([value])) for value in range(128))
        for value, wire in cases:
            for owner in (bytes, bytearray, memoryview):
                with self.subTest(value=value, owner=owner):
                    reader = self.codec._Reader(owner(b"prefix" + wire + b"next"))
                    reader.offset = len(b"prefix")
                    self.assertEqual(reader.varint(), value)
                    self.assertEqual(reader.offset, len(b"prefix") + len(wire))
                    self.assertEqual(reader.raw(4), b"next")

    def test_truncation_and_overflow_keep_consumed_offset(self):
        for wire, message, consumed in [
            (b"", "truncated protobuf varint", 0),
            (b"\x80", "truncated protobuf varint", 1),
            (b"\xff" * 9, "truncated protobuf varint", 9),
            (b"\xff" * 10 + b"next", "protobuf varint exceeds 64 bits", 10),
            (b"\x80" * 9 + b"\x02next", "protobuf varint exceeds 64 bits", 10),
        ]:
            with self.subTest(wire=wire):
                reader = self.codec._Reader(wire)
                with self.assertRaisesRegex(self.codec.DecodeError, message):
                    reader.varint()
                self.assertEqual(reader.offset, consumed)
        reader = self.codec._Reader(b"\x80" * 9 + b"\0next")
        self.assertEqual(reader.varint(), 0)
        self.assertEqual(reader.offset, 10)
        self.assertEqual(reader.raw(4), b"next")

    def test_uint64_message_packed_unknown_and_malformed_wire(self):
        message = type("UInt64ReaderMessage", (self.codec.ProtoMessage,), {
            "__fields__": (self.codec.Field(1, "count", "uint64"),
                           self.codec.Field(2, "values", "uint64", repeated=True),
                           self.codec.Field(3, "text", "string")),
        })
        maximum = b"\xff" * 9 + b"\x01"
        # Unknown uint64 precedes known scalar and packed uint64 fields.
        wire = b"\x48" + maximum + b"\x08" + maximum + b"\x12\x0b\0" + maximum
        parsed = message.FromString(wire)
        self.assertEqual(parsed.count, (1 << 64) - 1)
        self.assertEqual(parsed.values, [0, (1 << 64) - 1])
        for wire, error in [(b"\x08" + b"\xff" * 10, "exceeds 64 bits"),
                            (b"\x12\x01\x80", "truncated protobuf varint"),
                            (b"\x1a\x01\xff", "invalid UTF-8")]:
            with self.subTest(wire=wire):
                with self.assertRaisesRegex(self.codec.DecodeError, error):
                    message.FromString(wire)


if __name__ == "__main__":
    unittest.main()
