import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TRAINING_DROPOUT_SYMBOLS = {
    "g_native_training_mode",
    "g_native_training_counter",
    "dropout_random_bits",
    "native_attention_dropout_params",
    "attention_probability_index",
    "attention_dropout_multiplier",
    "native_dropout_params",
    "dropout_forward_f32",
    "sdpa_attention_dropout_forward",
    "cross_sdpa_attention_dropout_forward",
}

TRAINING_DROPOUT_STRINGS = (
    b"attention_dropout",
    b"dropout_seed",
    b"training_seed",
)

TRAINING_OPTIMIZER_SYMBOLS = {
    "g_opt_states",
    "optimizer_metadata_long",
    "optimizer_state_for",
    "volvoxai_engine_apply_tensor_update_f32_impl",
    "volvoxai_engine_apply_tensor_update_f32_locked",
    "volvoxai_engine_prepare_tensor_update_f32",
    "volvoxai_engine_apply_tensor_update_f32",
    "volvoxai_engine_save_optimizer_state",
    "volvoxai_engine_load_optimizer_state",
}

TRAINING_OPTIMIZER_STRINGS = (
    b"volvox.optimizer_format",
    b"volvox.optimizer.v1",
    b"volvox.training_step",
    b"volvox.optimizer_state_count",
)

TRAINING_DEFAULT_SOURCES = (
    "native/cli/main.c",
    "native/src/runtime/engine_internal.h",
    "native/src/backends/vulkan_engine.h",
    "native/src/backends/opengl_engine.h",
    "native/src/backends/metal_engine.h",
)


class NativeTrainingBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = shlex.split(os.environ.get("CC", "clang"))
        cls.nm = shlex.split(os.environ.get("NM", "nm"))
        if not cls.compiler or not shutil.which(cls.compiler[0]):
            raise unittest.SkipTest(f"C compiler not found: {cls.compiler!r}")
        if not cls.nm or not shutil.which(cls.nm[0]):
            raise unittest.SkipTest(f"nm not found: {cls.nm!r}")

        cls.temporary = tempfile.TemporaryDirectory()
        cls.objects = {"runtime": {}, "optimizer": {}}
        cls.symbols = {"runtime": {}, "optimizer": {}}
        sources = {
            "runtime": "native/src/runtime/engine_runtime.c",
            "optimizer": "native/src/runtime/engine.c",
        }
        for unit, source in sources.items():
            for profile, training_enabled in (("inference", 0), ("full", 1), ("default", None)):
                object_path = Path(cls.temporary.name) / f"{unit}-{profile}.o"
                command = [
                    *cls.compiler,
                    "-std=c11",
                    "-O0",
                    "-pthread",
                    "-DVOLVOXAI_ENABLE_VULKAN=1",
                    "-DVOLVOXAI_ENABLE_OPENGL=1",
                    "-DVOLVOXAI_ENABLE_METAL=0",
                    "-DVOLVOXAI_ENABLE_NNAPI=0",
                ]
                if training_enabled is not None:
                    command.append(f"-DVOLVOXAI_ENABLE_TRAINING={training_enabled}")
                for include in (
                    "native/include",
                    "native/src",
                    "native/src/runtime",
                    "native/src/kernels",
                    "native/src/backends",
                    "native/cli",
                    "native/third_party",
                    "native/third_party/xz-embedded",
                ):
                    command.extend(("-I", str(ROOT / include)))
                command.extend(("-c", str(ROOT / source), "-o", str(object_path)))
                subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
                nm_result = subprocess.run(
                    [*cls.nm, str(object_path)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                cls.objects[unit][profile] = object_path.read_bytes()
                cls.symbols[unit][profile] = {
                    line.split()[-1].lstrip("_")
                    for line in nm_result.stdout.splitlines()
                    if len(line.split()) >= 2 and line.split()[-2].upper() != "U"
                }

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def test_inference_object_has_no_training_dropout_implementation(self) -> None:
        leaked = TRAINING_DROPOUT_SYMBOLS & self.symbols["runtime"]["inference"]
        self.assertEqual(leaked, set())
        for token in TRAINING_DROPOUT_STRINGS:
            self.assertNotIn(token, self.objects["runtime"]["inference"])

    def test_full_object_retains_training_dropout_implementation(self) -> None:
        missing = TRAINING_DROPOUT_SYMBOLS - self.symbols["runtime"]["full"]
        self.assertEqual(missing, set())
        for token in TRAINING_DROPOUT_STRINGS:
            self.assertIn(token, self.objects["runtime"]["full"])

    def test_inference_object_has_no_optimizer_implementation(self) -> None:
        leaked = TRAINING_OPTIMIZER_SYMBOLS & self.symbols["optimizer"]["inference"]
        self.assertEqual(leaked, set())
        for token in TRAINING_OPTIMIZER_STRINGS:
            self.assertNotIn(token, self.objects["optimizer"]["inference"])

    def test_full_object_retains_optimizer_implementation(self) -> None:
        missing = TRAINING_OPTIMIZER_SYMBOLS - self.symbols["optimizer"]["full"]
        self.assertEqual(missing, set())
        for token in TRAINING_OPTIMIZER_STRINGS:
            self.assertIn(token, self.objects["optimizer"]["full"])

    def test_unspecified_profile_compiles_as_inference(self) -> None:
        for unit, symbols, strings in (
            ("runtime", TRAINING_DROPOUT_SYMBOLS, TRAINING_DROPOUT_STRINGS),
            ("optimizer", TRAINING_OPTIMIZER_SYMBOLS, TRAINING_OPTIMIZER_STRINGS),
        ):
            with self.subTest(unit=unit):
                self.assertEqual(symbols & self.symbols[unit]["default"], set())
                for token in strings:
                    self.assertNotIn(token, self.objects[unit]["default"])

    def test_native_training_fallbacks_fail_closed(self) -> None:
        fallback = "#ifndef VOLVOXAI_ENABLE_TRAINING\n#define VOLVOXAI_ENABLE_TRAINING 0\n#endif"
        for relative_path in TRAINING_DEFAULT_SOURCES:
            with self.subTest(path=relative_path):
                self.assertIn(fallback, (ROOT / relative_path).read_text())


if __name__ == "__main__":
    unittest.main()
