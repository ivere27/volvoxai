"""Official messages with authentically generated Synurang clients."""
import asyncio
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from google.protobuf.message import DecodeError, Message

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_proto as codegen


class ByteCall:
    def __init__(self, payload):
        self.payloads = [payload, None]
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        return self.payloads.pop(0)

    def half_close(self):
        pass

    def close(self):
        self.closed = True

    def cancel(self, code=1):
        self.close()


class ByteHost:
    def __init__(self, payload):
        self.payload = payload
        self.records = []
        self.call = None

    def unary(self, method, payload, *, timeout=None):
        self.records.append((method, payload, timeout))
        return self.payload

    def server_stream(self, method, payload, *, timeout=None):
        self.records.append((method, payload, timeout))
        def responses():
            yield self.payload
        return responses()

    def client_stream(self, method, payloads, *, timeout=None):
        self.records.append((method, list(payloads), timeout))
        return self.payload

    def open(self, method, **options):
        self.records.append((method, options))
        self.call = ByteCall(self.payload)
        return self.call


class AsyncByteCall(ByteCall):
    async def send(self, payload):
        super().send(payload)

    async def recv(self):
        return super().recv()

    async def half_close(self):
        pass


class AsyncByteHost(ByteHost):
    async def unary(self, method, payload, *, timeout=None):
        return super().unary(method, payload, timeout=timeout)

    def server_stream(self, method, payload, *, timeout=None):
        self.records.append((method, payload, timeout))
        async def responses():
            yield self.payload
        return responses()

    async def client_stream(self, method, payloads, *, timeout=None):
        self.records.append((method, [payload async for payload in payloads], timeout))
        return self.payload

    def open(self, method, **options):
        self.records.append((method, options))
        self.call = AsyncByteCall(self.payload)
        return self.call


class OfficialProtobufTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator_env = os.environ.get("VOLVOXAI_TEST_SYNURANG_GENERATOR") or os.environ.get("PROTOC_GEN_SYNURANG_FFI")
        if generator_env:
            generator = Path(generator_env)
        else:
            try:
                target = codegen.generator_target()
                candidate = (
                    codegen.REPOSITORY_ROOT
                    / "build"
                    / "cache"
                    / "synurang-codegen"
                    / f"v{codegen.SYNURANG_VERSION}"
                    / target
                    / codegen.GENERATOR_NAME
                )
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    generator = candidate
                else:
                    generator = codegen.release_generator(
                        codegen.REPOSITORY_ROOT / "build" / "cache" / "synurang-codegen",
                        True,
                    )
            except Exception as e:
                raise unittest.SkipTest(
                    f"synurang generator not available: {e}"
                ) from e
        if not generator.is_file() or not os.access(generator, os.X_OK):
            raise unittest.SkipTest(f"synurang generator {generator} not found or not executable")
        cls.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-official-proto-")
        cls.directory = Path(cls.temporary.name)
        proto = cls.directory / "official_fixture.proto"
        proto.write_text('''syntax = "proto3";
package official.fixture;
option go_package = "official/fixture";
message Item {
  int32 value = 1;
  repeated int32 values = 2;
  map<string, int32> counts = 3;
  optional int32 optional_value = 4;
  oneof selection { string word = 5; int32 number = 6; }
  Item child = 7;
}
service Codec {
  rpc Unary(Item) returns (Item);
  rpc Server(Item) returns (stream Item);
  rpc Client(stream Item) returns (Item);
  rpc Bidi(stream Item) returns (stream Item);
}
''')
        descriptors = cls.directory / "descriptor.pb"
        command = ["protoc", "--experimental_allow_proto3_optional",
                   f"--proto_path={cls.directory}", f"--descriptor_set_out={descriptors}",
                   f"--plugin=protoc-gen-synurang-ffi={generator}",
                   f"--synurang-ffi_out={cls.directory}",
                   "--synurang-ffi_opt=lang=python,mode=client", proto.name]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        cls.upstream_client = (cls.directory / "official_fixture_client.py").read_text()
        codegen.python_protobuf_projection(cls.directory, descriptors)
        sys.path.insert(0, str(cls.directory))
        sys.path.insert(0, str(codegen.REPOSITORY_ROOT / "runtime/generated/python"))
        cls.messages = importlib.import_module("official_fixture_pb2")
        cls.clients = importlib.import_module("official_fixture_client")

    @classmethod
    def tearDownClass(cls):
        for name in ("official_fixture_pb2", "official_fixture_client"):
            sys.modules.pop(name, None)
        if hasattr(cls, "directory") and str(cls.directory) in sys.path:
            sys.path.remove(str(cls.directory))
        runtime_path = str(codegen.REPOSITORY_ROOT / "runtime/generated/python")
        if runtime_path in sys.path:
            sys.path.remove(runtime_path)
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def test_unexpected_client_projection_refuses_old_codec(self):
        mutations = [
            self.upstream_client.replace(".to_bytes()", ".to_bytes(False)", 1),
            self.upstream_client + "\nfrom unexpected_lite import Item\n",
        ]
        for source in mutations:
            with self.subTest(source=source[-80:]), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "official_fixture_client.py").write_text(source)
                with self.assertRaisesRegex(codegen.CodegenError, "projection needs review"):
                    codegen.python_protobuf_projection(output, self.directory / "descriptor.pb")

    def test_official_classes_and_native_field_semantics(self):
        item = self.messages.Item(child={"value": 7}, values=[1, 2], counts={"x": 3})
        self.assertIsInstance(item, Message)
        self.assertEqual(item.DESCRIPTOR.full_name, "official.fixture.Item")
        self.assertTrue(item.HasField("child"))
        self.assertFalse(item.HasField("optional_value"))
        item.optional_value = 0
        self.assertTrue(item.HasField("optional_value"))
        item.child.CopyFrom(self.messages.Item(value=8))
        item.values.extend([3])
        self.assertEqual((item.child.value, list(item.values)), (8, [1, 2, 3]))
        other = self.messages.Item()
        self.assertEqual(list(other.values), [])
        self.assertFalse(other.HasField("child"))

    def test_oneof_merge_unknown_fields_and_malformed_wire(self):
        item = self.messages.Item(word="first")
        item.number = 0
        self.assertEqual(item.WhichOneof("selection"), "number")
        payload = self.messages.Item(value=7, values=[1, 2]).SerializeToString()
        payload += b"\x98\x06\x01"  # Unknown field 99 is preserved.
        result = self.messages.Item.FromString(payload)
        self.assertEqual(result.SerializeToString(), payload)
        result.MergeFrom(self.messages.Item(values=[3], child={"value": 2}))
        self.assertEqual(list(result.values), [1, 2, 3])
        with self.assertRaises(DecodeError):
            self.messages.Item.FromString(b"\x80")

    def test_sync_unary_and_both_stream_directions(self):
        request = self.messages.Item(value=4)
        response = self.messages.Item(value=9)
        host = ByteHost(response.SerializeToString())
        client = self.clients.CodecClient(host)
        self.assertEqual(client.unary(request, timeout=1.25), response)
        self.assertEqual(list(client.server(request, timeout=1.25)), [response])
        self.assertEqual(client.client([request, request], timeout=1.25), response)
        self.assertEqual(host.records[0][1:], (request.SerializeToString(), 1.25))
        self.assertEqual(host.records[2][1], [request.SerializeToString()] * 2)

    def test_sync_bidi_closes_on_decode_failure(self):
        host = ByteHost(b"\x80")
        call = self.clients.CodecClient(host).bidi(timeout=1.25)
        request = self.messages.Item(value=4)
        call.send(request)
        self.assertEqual(host.call.sent, [request.SerializeToString()])
        with self.assertRaises(DecodeError):
            call.recv()
        self.assertTrue(host.call.closed)

    def test_async_unary_server_client_and_bidi(self):
        async def run():
            request = self.messages.Item(value=4)
            response = self.messages.Item(value=9)
            host = AsyncByteHost(response.SerializeToString())
            client = self.clients.CodecAsyncClient(host)
            self.assertEqual(await client.unary(request, timeout=2), response)
            self.assertEqual([x async for x in client.server(request, timeout=2)], [response])
            self.assertEqual(await client.client([request, request], timeout=2), response)
            call = client.bidi(timeout=2)
            await call.send(request)
            self.assertEqual(await call.recv(), response)
            self.assertIsNone(await call.recv())
            call.close()
            self.assertTrue(host.call.closed)
            self.assertEqual(host.records[2][1], [request.SerializeToString()] * 2)
        asyncio.run(run())

    def test_async_cancellation_is_unmodified(self):
        class CanceledHost(AsyncByteHost):
            async def unary(self, *args, **kwargs):
                raise asyncio.CancelledError()
        async def run():
            with self.assertRaises(asyncio.CancelledError):
                await self.clients.CodecAsyncClient(CanceledHost(b"")).unary(self.messages.Item())
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
