"""Exercise the production executable link policy with observable addresses."""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    sys.platform.startswith("linux")
    and all(shutil.which(tool) for tool in ("clang", "ld.lld", "cmake", "nm", "readelf")),
    "Linux Clang/lld tools required",
)
class NativeReleaseIcfTests(unittest.TestCase):
    def test_safe_folding_preserves_observable_function_identity_and_unwind(self):
        cmake = (ROOT / "native/CMakeLists.txt").read_text()
        start = cmake.index("# Fold identical, address-insignificant functions")
        end = cmake.index("endfunction()", start) + len("endfunction()")
        policy = cmake[start:end]
        with tempfile.TemporaryDirectory(prefix="volvox-release-icf-") as directory:
            root = Path(directory)
            (root / "a.c").write_text(
                "__attribute__((noinline)) int direct_one(int x) { return x*3+7; }\n"
                "__attribute__((noinline)) int observed_one(int x) { return x*5+9; }\n"
            )
            (root / "b.c").write_text(
                "__attribute__((noinline)) int direct_two(int x) { return x*3+7; }\n"
                "__attribute__((noinline)) int observed_two(int x) { return x*5+9; }\n"
            )
            (root / "main.c").write_text(
                "extern int direct_one(int), direct_two(int);\n"
                "extern int observed_one(int), observed_two(int);\n"
                "int (*volatile addresses[2])(int) = {observed_one, observed_two};\n"
                "int main(int argc, char **argv) { (void)argv;\n"
                "  if (addresses[0] == addresses[1]) return 1;\n"
                "  if (direct_one(argc) != direct_two(argc)) return 2;\n"
                "  return addresses[0](argc) != addresses[1](argc); }\n"
            )
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\nproject(icf_contract C)\n"
                + policy
                + "\nadd_executable(probe main.c a.c b.c)\n"
                "target_compile_options(probe PRIVATE -O3 -g -ffunction-sections -fdata-sections)\n"
                "volvox_release_section_gc(probe)\n"
            )

            def run(*args):
                return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

            build = root / "build"
            run("cmake", "-S", str(root), "-B", str(build), "-DCMAKE_C_COMPILER=clang")
            run("cmake", "--build", str(build))
            executable = build / "probe"
            run(str(executable))
            symbols = dict((name, int(address, 16)) for address, name in re.findall(
                r"^([0-9a-f]+) [tT] (\S+)$", run("nm", "-n", str(executable)), re.M
            ))
            self.assertEqual(symbols["direct_one"], symbols["direct_two"])
            self.assertNotEqual(symbols["observed_one"], symbols["observed_two"])
            sections = run("readelf", "-SW", str(executable))
            self.assertIn(".eh_frame", sections)
            self.assertIn(".debug_info", sections)
            link = (build / "CMakeFiles/probe.dir/link.txt").read_text()
            self.assertIn("--icf=safe", link)
            self.assertRegex(link, r"-fuse-ld=/\S*ld\.lld")
            if shutil.which("llvm-addr2line"):
                location = run("llvm-addr2line", "-e", str(executable), hex(symbols["main"]))
                self.assertIn("main.c:", location)


if __name__ == "__main__":
    unittest.main()
