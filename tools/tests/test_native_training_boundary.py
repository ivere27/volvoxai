import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TRAINING_DROPOUT_SYMBOLS = {
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

TRAINING_AUTOGRAD_SYMBOLS = {
    "volvoxai_autograd_abi_version",
    "volvoxai_autograd_context_create",
    "volvoxai_autograd_context_destroy",
    "volvoxai_autograd_context_backend",
    "volvoxai_autograd_tensor_create",
    "volvoxai_autograd_tensor_detach",
    "volvoxai_autograd_tensor_set_requires_grad",
    "volvoxai_autograd_tensor_set_data",
    "volvoxai_autograd_tensor_copy_data",
    "volvoxai_autograd_tensor_copy_grad",
    "volvoxai_autograd_tensor_info",
    "volvoxai_autograd_tensor_release",
    "volvoxai_autograd_apply",
    "volvoxai_autograd_no_grad_begin",
    "volvoxai_autograd_no_grad_end",
    "volvoxai_autograd_backward",
    "volvoxai_autograd_zero_grad",
    "volvoxai_autograd_clear_graph",
}

TRAINING_AUTOGRAD_STRINGS = (
    b"__vx_autograd_",
    b"volvoxai_autograd_abi_version",
)

PROCESS_GLOBAL_TRAINING_STATE_SYMBOLS = {
    "g_opt_states",
    "g_native_training_mode",
    "g_native_training_counter",
}

ENGINE_TRAINING_STATE_MEMBERS = (
    "EngineOptimizerState optimizer_states[MAXT];",
    "TrainingAccumulationState training_accumulation;",
    "int native_training_mode;",
    "uint32_t native_training_counter;",
)

ENGINE_TRAINING_STATE_ACCESSORS = {
    "g_opt_states": "optimizer_states",
    "g_native_training_mode": "native_training_mode",
    "g_native_training_counter": "native_training_counter",
}


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
        cls.objects = {"runtime": {}, "optimizer": {}, "cli": {}}
        cls.symbols = {"runtime": {}, "optimizer": {}, "cli": {}}
        sources = {
            "runtime": "native/src/runtime/engine_runtime.c",
            "optimizer": "native/src/runtime/engine.c",
            "cli": "native/cli/main.c",
        }
        for unit, source in sources.items():
            for profile, training_enabled in (("inference", 0), ("full", 1)):
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

    def test_inference_object_has_no_dynamic_autograd_implementation(self) -> None:
        leaked = TRAINING_AUTOGRAD_SYMBOLS & self.symbols["runtime"]["inference"]
        self.assertEqual(leaked, set())
        for token in TRAINING_AUTOGRAD_STRINGS:
            self.assertNotIn(token, self.objects["runtime"]["inference"])

    def test_full_object_retains_dynamic_autograd_implementation(self) -> None:
        missing = TRAINING_AUTOGRAD_SYMBOLS - self.symbols["runtime"]["full"]
        self.assertEqual(missing, set())
        for token in TRAINING_AUTOGRAD_STRINGS:
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

    def test_training_state_is_owned_by_each_engine(self) -> None:
        for unit in ("runtime", "optimizer", "cli"):
            for profile in ("inference", "full"):
                with self.subTest(unit=unit, profile=profile):
                    self.assertEqual(
                        PROCESS_GLOBAL_TRAINING_STATE_SYMBOLS
                        & self.symbols[unit][profile],
                        set(),
                    )

        state_header = (ROOT / "native/src/runtime/runtime_state.h").read_text()
        for member in ENGINE_TRAINING_STATE_MEMBERS:
            with self.subTest(member=member):
                self.assertIn(member, state_header)

        internal_header = (
            ROOT / "native/src/runtime/engine_internal.h"
        ).read_text()
        logical_header = " ".join(
            internal_header.replace("\\\n", "").split()
        )
        for name, member in ENGINE_TRAINING_STATE_ACCESSORS.items():
            with self.subTest(accessor=name):
                expected = (
                    f"#define {name} "
                    f"(vx_engine_state_current()->{member})"
                )
                self.assertIn(expected, logical_header)

    def test_cli_profile_is_selected_by_build_composition(self) -> None:
        self.assertNotIn("command_train", self.symbols["cli"]["inference"])
        self.assertIn("command_train", self.symbols["cli"]["full"])
        self.assertNotIn(b"Training backend=", self.objects["cli"]["inference"])
        self.assertIn(b"Training backend=", self.objects["cli"]["full"])

        cli_source = (ROOT / "native/cli/main.c").read_text()
        self.assertNotIn("#define VOLVOXAI_ENABLE_TRAINING", cli_source)
        self.assertIn("#if VOLVOXAI_ENABLE_TRAINING", cli_source)

        composition = (ROOT / "native/CMakeLists.txt").read_text()
        self.assertIn(
            "target_compile_definitions(volvoxai PRIVATE "
            "VOLVOXAI_ENABLE_TRAINING=0",
            composition,
        )
        self.assertIn(
            "target_compile_definitions(volvoxai-full PRIVATE "
            "VOLVOXAI_ENABLE_TRAINING=1",
            composition,
        )


if __name__ == "__main__":
    unittest.main()
