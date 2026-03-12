"""Unit tests for stablegen_cli.py — tests pure-Python logic only (no bpy required)."""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Stub out bpy before importing the module
bpy_stub = MagicMock()
sys.modules["bpy"] = bpy_stub
sys.modules["mathutils"] = MagicMock()

# Patch sys.argv so _parse_args() returns defaults at import time
with patch.object(sys, "argv", ["blender", "--", "--mesh", "test.obj", "--prompt", "test"]):
    import importlib, types
    # We need to import without executing main() at module level
    # Patch main() to a no-op during import
    import builtins
    _real_import = builtins.__import__
    spec = importlib.util.spec_from_file_location(
        "stablegen_cli",
        os.path.join(os.path.dirname(__file__), "stablegen_cli.py"),
    )
    mod = types.ModuleType("stablegen_cli")
    mod.__spec__ = spec
    # Exec with main() stubbed
    src = open(spec.origin, encoding="utf-8").read()
    src_no_main = src.replace("\nmain()\n", "\n# main() disabled for testing\n")
    exec(compile(src_no_main, spec.origin, "exec"), mod.__dict__)
    sys.modules["stablegen_cli"] = mod


class TestParseArgs(unittest.TestCase):
    def _parse(self, argv):
        with patch.object(sys, "argv", ["blender", "--"] + argv):
            mod2 = types.ModuleType("stablegen_cli_tmp")
            src2 = open(mod.__spec__.origin, encoding="utf-8").read()
            src2 = src2.replace("\nmain()\n", "\n")
            exec(compile(src2, mod.__spec__.origin, "exec"), mod2.__dict__)
            return mod2._parse_args()

    def test_defaults(self):
        args = self._parse(["--mesh", "foo.obj"])
        self.assertEqual(args.cameras, 6)
        self.assertEqual(args.camera_mode, 5)
        self.assertEqual(args.seed, -1)
        self.assertEqual(args.server, "127.0.0.1:8188")
        self.assertEqual(args.output, "./stablegen_out")
        self.assertEqual(args.export, "none")
        self.assertEqual(args.preset, "Default")
        self.assertFalse(args.bake)
        self.assertFalse(args.no_controlnet)
        self.assertEqual(args.architecture, "sdxl")

    def test_custom_values(self):
        args = self._parse([
            "--mesh", "robot.glb",
            "--prompt", "cyberpunk robot",
            "--cameras", "8",
            "--camera-mode", "3",
            "--seed", "42",
            "--steps", "30",
            "--cfg", "7.5",
            "--export", "glb",
            "--architecture", "flux",
            "--bake",
        ])
        self.assertEqual(args.cameras, 8)
        self.assertEqual(args.camera_mode, 3)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.steps, 30)
        self.assertAlmostEqual(args.cfg, 7.5)
        self.assertEqual(args.export, "glb")
        self.assertEqual(args.architecture, "flux")
        self.assertTrue(args.bake)

    def test_no_separator_gives_empty(self):
        with patch.object(sys, "argv", ["blender", "mesh.obj"]):
            args = mod._parse_args()
        self.assertIsNone(args.mesh)

    def test_invalid_camera_mode_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(["--mesh", "a.obj", "--camera-mode", "9"])

    def test_invalid_export_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(["--mesh", "a.obj", "--export", "mp4"])

    def test_invalid_architecture_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(["--mesh", "a.obj", "--architecture", "dalle"])


class TestCameraModeMap(unittest.TestCase):
    def test_all_modes_present(self):
        m = mod._CAMERA_MODE_MAP
        self.assertEqual(m[1], "orbit")
        self.assertEqual(m[2], "fan")
        self.assertEqual(m[3], "hemisphere")
        self.assertEqual(m[4], "pca")
        self.assertEqual(m[5], "kmeans")
        self.assertEqual(m[6], "greedy")
        self.assertEqual(m[7], "visibility")
        self.assertEqual(len(m), 7)


class TestArchMap(unittest.TestCase):
    def test_all_archs(self):
        m = mod._ARCH_MAP
        self.assertEqual(m["sdxl"], "sdxl")
        self.assertEqual(m["flux"], "flux1")
        self.assertEqual(m["qwen"], "qwen")


class TestCheckServer(unittest.TestCase):
    def test_reachable(self):
        with patch("urllib.request.urlopen"):
            result = mod._check_server("127.0.0.1:8188")
        self.assertTrue(result)

    def test_unreachable(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = mod._check_server("127.0.0.1:9999")
        self.assertFalse(result)


class TestImportMeshExtension(unittest.TestCase):
    def test_unsupported_extension_raises(self):
        with self.assertRaises(ValueError):
            mod._import_mesh("model.abc")

    def test_supported_extensions_dispatch(self):
        for ext in [".obj", ".glb", ".gltf", ".fbx", ".stl"]:
            bpy_stub.data.objects.__iter__ = MagicMock(return_value=iter([]))
            # Should not raise ValueError (will raise RuntimeError for no objects)
            try:
                mod._import_mesh(f"model{ext}")
            except RuntimeError:
                pass  # expected — no real objects added
            except ValueError as e:
                self.fail(f"ValueError for supported ext {ext}: {e}")


class TestExport(unittest.TestCase):
    def test_none_format_is_noop(self):
        # Should return without calling any bpy ops
        bpy_stub.ops.export_scene.gltf.reset_mock()
        mod._export([], "/tmp/out", "none")
        bpy_stub.ops.export_scene.gltf.assert_not_called()

    def test_glb_calls_gltf_op(self):
        mod._export([MagicMock()], "/tmp/out", "glb")
        bpy_stub.ops.export_scene.gltf.assert_called_once()

    def test_fbx_calls_fbx_op(self):
        mod._export([MagicMock()], "/tmp/out", "fbx")
        bpy_stub.ops.export_scene.fbx.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
