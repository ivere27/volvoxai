"""PTX preparation must preserve quoted bytes, token boundaries and provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.embed_cuda_ptx import kernel_argument_abis
from tools.prepare_cuda_ptx import prepare_ptx, strip_line_comments


class PrepareCudaPtxTests(unittest.TestCase):
    def test_quotes_escapes_block_comments_and_newlines(self):
        source = (b'// compiler banner\r\n.version 7.1 // version\n'
                  b'.file 1 "path//part/*literal*/\\\"quoted\\\\tail"\n'
                  b".b8 'x', '/', '\\'', '\\\\'; // character constants\r\n"
                  b'/* keep // inside block\n and "unterminated quote */\n'
                  b'first:// label comment\nmov.u32 %r0, 1;/* separator */ret;// EOF')
        expected = (b'\r\n.version 7.1 \n'
                    b'.file 1 "path//part/*literal*/\\\"quoted\\\\tail"\n'
                    b".b8 'x', '/', '\\'', '\\\\'; \r\n"
                    b'/* keep // inside block\n and "unterminated quote */\n'
                    b'first:\nmov.u32 %r0, 1;/* separator */ret;')
        actual, count = strip_line_comments(source)
        self.assertEqual(actual, expected)
        self.assertEqual(count, 5)
        self.assertEqual(actual.count(b'\n'), source.count(b'\n'))
        self.assertEqual(strip_line_comments(actual), (actual, 0))
        self.assertEqual(strip_line_comments(b'// hidden\rstill hidden\nret;')[0], b'\nret;')

    def test_generated_artifacts_preserve_original_and_argument_abi(self):
        source = (b'// generated\n.version 7.1\n.target sm_86\n.address_size 64\n'
                  b'.visible .entry test(\n.param .u64 a,// pointer\n'
                  b'.param .align 4 .b8 arg[12]) {\n'
                  b'.reg .u64 %rd;\nld.param.u64 %rd, [a]; // load\nret;\n}\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, prepared, manifest = root/'module.ptx.raw', root/'module.ptx', root/'module.json'
            original.write_bytes(source)
            prepare_ptx(original, prepared, manifest)
            result = prepared.read_bytes(); report = json.loads(manifest.read_text())
            self.assertEqual(original.read_bytes(), source)
            self.assertEqual(kernel_argument_abis(source), [('test', (8, 12))])
            self.assertEqual(kernel_argument_abis(result), kernel_argument_abis(source))
            self.assertEqual(report['compiler_ptx']['sha256'], hashlib.sha256(source).hexdigest())
            self.assertEqual(report['prepared_ptx']['sha256'], hashlib.sha256(result).hexdigest())
            self.assertEqual(report['line_comments_removed'], 3)
            first = manifest.read_bytes()
            prepare_ptx(original, prepared, manifest)
            self.assertEqual(manifest.read_bytes(), first)
            for paths in ((original, original, manifest), (original, prepared, original), (original, prepared, prepared)):
                with self.assertRaises(ValueError): prepare_ptx(*paths)
            self.assertEqual(original.read_bytes(), source)
            prepared.unlink()
            prepared.hardlink_to(original)
            with self.assertRaises(ValueError): prepare_ptx(original, prepared, manifest)
            self.assertEqual(original.read_bytes(), source)

    def test_malformed_input_rejected_before_output_write(self):
        for source in (b'', b'\0', b'mov; /* unterminated', b'.file "unterminated', b".b8 'x", b'.file "escape\\'):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); original, prepared, manifest = root/'raw', root/'ptx', root/'json'
                original.write_bytes(source)
                with self.assertRaises(ValueError): prepare_ptx(original, prepared, manifest)
                self.assertEqual(original.read_bytes(), source)
                self.assertFalse(prepared.exists()); self.assertFalse(manifest.exists())


if __name__ == '__main__': unittest.main()
