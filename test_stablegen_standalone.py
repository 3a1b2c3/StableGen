"""Unit tests for stablegen_standalone.py — no Blender, no ComfyUI, no network."""

import math
import os
import sys
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

STABLEGEN_DIR = os.path.dirname(os.path.abspath(__file__))
if STABLEGEN_DIR not in sys.path:
    sys.path.insert(0, STABLEGEN_DIR)

import stablegen_standalone as sg
import trimesh


# ── helpers ───────────────────────────────────────────────────────────────────

def _unit_cube():
    return trimesh.creation.box()


def _sphere_mesh():
    return trimesh.creation.icosphere(subdivisions=2)


def _mesh_with_uv():
    """Minimal quad (2 triangles) with explicit UV coords."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv    = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vis   = trimesh.visual.texture.TextureVisuals(uv=uv)
    mesh.visual = vis
    return mesh, uv


def _solid_image(color=(128, 64, 32), size=(32, 32)):
    return Image.new("RGB", size, color)


# ── _look_at ──────────────────────────────────────────────────────────────────

class TestLookAt(unittest.TestCase):

    def test_returns_two_4x4_matrices(self):
        view, pose = sg._look_at(np.array([0., 0., 3.]), np.array([0., 0., 0.]))
        self.assertEqual(view.shape, (4, 4))
        self.assertEqual(pose.shape, (4, 4))

    def test_view_is_approximately_inverse_of_pose(self):
        view, pose = sg._look_at(np.array([2., 1., 3.]), np.array([0., 0., 0.]))
        np.testing.assert_allclose(view @ pose, np.eye(4), atol=1e-10)

    def test_camera_position_in_pose(self):
        eye = np.array([5., 0., 0.])
        _, pose = sg._look_at(eye, np.array([0., 0., 0.]))
        np.testing.assert_allclose(pose[:3, 3], eye, atol=1e-10)

    def test_pose_rotation_is_orthonormal(self):
        _, pose = sg._look_at(np.array([1., 2., 3.]), np.array([0., 0., 0.]))
        R = pose[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_degenerate_direction_does_not_crash(self):
        eye = target = np.array([1., 0., 0.])
        try:
            sg._look_at(eye, target)
        except Exception as e:
            self.fail(f"_look_at raised {e} for eye == target")

    def test_up_parallel_to_forward_uses_fallback(self):
        """Looking straight up with default up=[0,0,1] triggers the Y fallback."""
        view, pose = sg._look_at(np.array([0., 0., 3.]), np.array([0., 0., 0.]),
                                  up=np.array([0., 0., 1.]))
        self.assertEqual(view.shape, (4, 4))


# ── _perspective_matrix ───────────────────────────────────────────────────────

class TestPerspectiveMatrix(unittest.TestCase):

    def test_shape(self):
        self.assertEqual(sg._perspective_matrix(math.radians(60), 1.0).shape, (4, 4))

    def test_bottom_row(self):
        P = sg._perspective_matrix(math.radians(60), 1.0)
        np.testing.assert_array_equal(P[3], [0, 0, -1, 0])

    def test_focal_length_matches_yfov(self):
        yfov = math.radians(90)
        P = sg._perspective_matrix(yfov, 1.0)
        self.assertAlmostEqual(P[1, 1], 1.0 / math.tan(yfov * 0.5), places=10)

    def test_aspect_ratio_applied_to_x(self):
        yfov, aspect = math.radians(60), 2.0
        P = sg._perspective_matrix(yfov, aspect)
        f = 1.0 / math.tan(yfov * 0.5)
        self.assertAlmostEqual(P[0, 0], f / aspect, places=10)
        self.assertAlmostEqual(P[1, 1], f,           places=10)

    def test_point_behind_camera_has_negative_w(self):
        # P[3] = [0, 0, -1, 0] so w = -z.  z=+1 → w=-1 < 0 (behind camera).
        P = sg._perspective_matrix(math.radians(60), 1.0)
        p_clip = P @ np.array([0., 0., 1., 1.])
        self.assertLess(p_clip[3], 0)

    def test_point_in_front_of_camera_has_positive_w(self):
        P = sg._perspective_matrix(math.radians(60), 1.0)
        p_clip = P @ np.array([0., 0., -5., 1.])
        self.assertGreater(p_clip[3], 0)


# ── _barycentric_2d ───────────────────────────────────────────────────────────

class TestBarycentric2D(unittest.TestCase):

    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.0, 1.0])

    def _bary(self, p):
        # Pass as a single-row (1, 2) array; squeeze results to scalars.
        aa, bb, gg = sg._barycentric_2d(np.atleast_2d(p), self.A, self.B, self.C)
        return float(np.squeeze(aa)), float(np.squeeze(bb)), float(np.squeeze(gg))

    def test_vertex_a(self):
        aa, bb, gg = self._bary(self.A)
        np.testing.assert_allclose([aa, bb, gg], [1., 0., 0.], atol=1e-9)

    def test_vertex_b(self):
        aa, bb, gg = self._bary(self.B)
        np.testing.assert_allclose([aa, bb, gg], [0., 1., 0.], atol=1e-9)

    def test_vertex_c(self):
        aa, bb, gg = self._bary(self.C)
        np.testing.assert_allclose([aa, bb, gg], [0., 0., 1.], atol=1e-9)

    def test_centroid(self):
        centroid = (self.A + self.B + self.C) / 3
        aa, bb, gg = self._bary(centroid)
        np.testing.assert_allclose([aa, bb, gg], [1/3, 1/3, 1/3], atol=1e-9)

    def test_outside_point_has_negative_coord(self):
        aa, bb, gg = sg._barycentric_2d(np.array([[-1., -1.]]), self.A, self.B, self.C)
        self.assertTrue(aa[0] < 0 or bb[0] < 0 or gg[0] < 0)

    def test_coords_sum_to_one_inside(self):
        pts = np.array([[0.1, 0.1], [0.3, 0.3]])
        aa, bb, gg = sg._barycentric_2d(pts, self.A, self.B, self.C)
        np.testing.assert_allclose(aa + bb + gg, [1., 1.], atol=1e-10)

    def test_batch_shape_preserved(self):
        pts = np.zeros((5, 3, 2))
        aa, bb, gg = sg._barycentric_2d(pts, self.A, self.B, self.C)
        self.assertEqual(aa.shape, (5, 3))


# ── _spherical_uv ─────────────────────────────────────────────────────────────

class TestSphericalUV(unittest.TestCase):

    def test_shape(self):
        mesh = _sphere_mesh()
        uv = sg._spherical_uv(mesh)
        self.assertEqual(uv.shape, (len(mesh.vertices), 2))

    def test_range_0_to_1(self):
        uv = sg._spherical_uv(_sphere_mesh())
        self.assertGreaterEqual(uv.min(), -1e-6)
        self.assertLessEqual(uv.max(), 1.0 + 1e-6)

    def test_no_nan(self):
        self.assertFalse(np.isnan(sg._spherical_uv(_sphere_mesh())).any())

    def test_cube_uv_2_columns(self):
        self.assertEqual(sg._spherical_uv(_unit_cube()).shape[1], 2)


# ── _get_uv ───────────────────────────────────────────────────────────────────

class TestGetUV(unittest.TestCase):

    def test_returns_uv_when_present(self):
        mesh, expected = _mesh_with_uv()
        uv = sg._get_uv(mesh)
        self.assertIsNotNone(uv)
        np.testing.assert_allclose(uv, expected, atol=1e-10)

    def test_returns_none_for_cube(self):
        self.assertIsNone(sg._get_uv(_unit_cube()))

    def test_returns_none_for_sphere(self):
        self.assertIsNone(sg._get_uv(_sphere_mesh()))


# ── ensure_uv ─────────────────────────────────────────────────────────────────

class TestEnsureUV(unittest.TestCase):

    def test_uses_existing_uv(self):
        mesh, expected = _mesh_with_uv()
        out_mesh, uv = sg.ensure_uv(mesh)
        self.assertIs(out_mesh, mesh)
        np.testing.assert_allclose(uv, expected, atol=1e-10)

    def test_fallback_spherical_when_no_uv(self):
        mesh = _unit_cube()
        _, uv = sg.ensure_uv(mesh, force_spherical=True)
        self.assertEqual(uv.shape, (len(mesh.vertices), 2))
        self.assertTrue((uv >= 0).all() and (uv <= 1 + 1e-6).all())

    def test_force_spherical_ignores_existing_uv(self):
        mesh, _ = _mesh_with_uv()
        _, uv = sg.ensure_uv(mesh, force_spherical=True)
        self.assertEqual(uv.shape[1], 2)


# ── depth_to_image ────────────────────────────────────────────────────────────

class TestDepthToImage(unittest.TestCase):

    def test_returns_pil_image(self):
        self.assertIsInstance(
            sg.depth_to_image(np.ones((4, 4), np.float32)), Image.Image)

    def test_output_size_matches_input(self):
        img = sg.depth_to_image(np.ones((64, 128), np.float32))
        self.assertEqual(img.size, (128, 64))

    def test_all_nan_gives_black(self):
        arr = np.array(sg.depth_to_image(np.full((4, 4), np.nan, np.float32)))
        self.assertEqual(arr.max(), 0)

    def test_uniform_depth_all_same_pixel(self):
        arr = np.array(sg.depth_to_image(np.full((8, 8), 3.0, np.float32)))
        self.assertEqual(arr.min(), arr.max())

    def test_pixel_range_0_255(self):
        depth = np.linspace(1, 10, 100).reshape(10, 10).astype(np.float32)
        arr = np.array(sg.depth_to_image(depth))
        self.assertGreaterEqual(arr.min(), 0)
        self.assertLessEqual(arr.max(), 255)

    def test_rgb_mode(self):
        self.assertEqual(sg.depth_to_image(np.ones((4, 4), np.float32)).mode, "RGB")


# ── ComfyUI workflow builders ─────────────────────────────────────────────────

class TestBuildSdxlTxt2Img(unittest.TestCase):

    def _build(self, seed=42):
        return sg._build_sdxl_txt2img(
            "test prompt", "bad quality",
            "sdxl.safetensors", 20, 7.0, seed, 1024, 1024)

    def test_returns_dict(self):
        self.assertIsInstance(self._build(), dict)

    def test_has_checkpoint_loader(self):
        wf = self._build()
        nodes = [v for v in wf.values() if v["class_type"] == "CheckpointLoaderSimple"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["inputs"]["ckpt_name"], "sdxl.safetensors")

    def test_has_ksampler(self):
        wf = self._build()
        self.assertEqual(
            len([v for v in wf.values() if v["class_type"] == "KSampler"]), 1)

    def test_ksampler_params(self):
        wf = self._build(seed=99)
        inp = next(v for v in wf.values() if v["class_type"] == "KSampler")["inputs"]
        self.assertEqual(inp["steps"], 20)
        self.assertAlmostEqual(inp["cfg"], 7.0)
        self.assertEqual(inp["seed"], 99)

    def test_has_save_image_websocket(self):
        wf = self._build()
        self.assertEqual(
            len([v for v in wf.values() if v["class_type"] == "SaveImageWebsocket"]), 1)

    def test_resolution_in_empty_latent(self):
        wf = sg._build_sdxl_txt2img("p", "", "c.safetensors", 20, 7.0, 1, 512, 768)
        latent = next(v for v in wf.values() if v["class_type"] == "EmptyLatentImage")
        self.assertEqual(latent["inputs"]["width"],  512)
        self.assertEqual(latent["inputs"]["height"], 768)

    def test_random_seed_for_negative_input(self):
        wf = sg._build_sdxl_txt2img("p", "", "c.safetensors", 20, 7.0, -1, 1024, 1024)
        inp = next(v for v in wf.values() if v["class_type"] == "KSampler")["inputs"]
        self.assertGreaterEqual(inp["seed"], 0)

    def test_prompt_in_positive_node(self):
        wf = self._build()
        pos = [v for v in wf.values()
               if v["class_type"] == "CLIPTextEncode"
               and v["inputs"]["text"] == "test prompt"]
        self.assertEqual(len(pos), 1)

    def test_negative_prompt_in_negative_node(self):
        wf = self._build()
        neg = [v for v in wf.values()
               if v["class_type"] == "CLIPTextEncode"
               and v["inputs"]["text"] == "bad quality"]
        self.assertEqual(len(neg), 1)

    def test_vae_decode_present(self):
        self.assertEqual(
            len([v for v in self._build().values() if v["class_type"] == "VAEDecode"]), 1)


class TestBuildSdxlControlnetDepth(unittest.TestCase):

    def _build(self, seed=1):
        return sg._build_sdxl_controlnet_depth(
            "stone wall", "", "sdxl.safetensors",
            "depth.safetensors", 0.6, "depth_00.png",
            20, 7.0, seed, 1024, 1024)

    def test_returns_dict(self):
        self.assertIsInstance(self._build(), dict)

    def test_has_controlnet_loader(self):
        wf = self._build()
        nodes = [v for v in wf.values() if v["class_type"] == "ControlNetLoader"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["inputs"]["control_net_name"], "depth.safetensors")

    def test_has_load_image_with_correct_name(self):
        wf = self._build()
        li = [v for v in wf.values() if v["class_type"] == "LoadImage"]
        self.assertEqual(len(li), 1)
        self.assertEqual(li[0]["inputs"]["image"], "depth_00.png")

    def test_has_controlnet_apply_advanced(self):
        wf = self._build()
        self.assertEqual(
            len([v for v in wf.values() if v["class_type"] == "ControlNetApplyAdvanced"]), 1)

    def test_controlnet_strength(self):
        wf = self._build()
        cn = next(v for v in wf.values() if v["class_type"] == "ControlNetApplyAdvanced")
        self.assertAlmostEqual(cn["inputs"]["strength"], 0.6)

    def test_has_save_image_websocket(self):
        self.assertEqual(
            len([v for v in self._build().values()
                 if v["class_type"] == "SaveImageWebsocket"]), 1)

    def test_random_seed_for_negative(self):
        wf = sg._build_sdxl_controlnet_depth(
            "p", "", "c.safetensors", "d.safetensors", 0.5, "img.png",
            20, 7.0, -1, 1024, 1024)
        inp = next(v for v in wf.values() if v["class_type"] == "KSampler")["inputs"]
        self.assertGreaterEqual(inp["seed"], 0)


# ── build_uv_3d_map ───────────────────────────────────────────────────────────

class TestBuildUv3dMap(unittest.TestCase):

    def setUp(self):
        self.mesh, self.uv = _mesh_with_uv()

    def test_output_shapes(self):
        pos, norm, valid = sg.build_uv_3d_map(self.mesh, self.uv, tex_size=32)
        self.assertEqual(pos.shape,   (32, 32, 3))
        self.assertEqual(norm.shape,  (32, 32, 3))
        self.assertEqual(valid.shape, (32, 32))

    def test_valid_texels_have_non_nan_position(self):
        pos, _, valid = sg.build_uv_3d_map(self.mesh, self.uv, tex_size=32)
        self.assertFalse(np.isnan(pos[valid]).any())

    def test_invalid_texels_are_nan(self):
        pos, _, valid = sg.build_uv_3d_map(self.mesh, self.uv, tex_size=32)
        self.assertTrue(np.isnan(pos[~valid]).all())

    def test_some_texels_covered(self):
        _, _, valid = sg.build_uv_3d_map(self.mesh, self.uv, tex_size=32)
        self.assertGreater(valid.sum(), 0)

    def test_positions_on_z0_plane(self):
        """All 3-D positions from our z=0 quad should have z≈0."""
        pos, _, valid = sg.build_uv_3d_map(self.mesh, self.uv, tex_size=32)
        np.testing.assert_allclose(pos[valid, 2], 0.0, atol=1e-5)

    def test_unit_quad_near_full_coverage(self):
        _, _, valid = sg.build_uv_3d_map(self.mesh, self.uv, tex_size=32)
        self.assertGreater(valid.sum() / (32 * 32), 0.8)


# ── bake_texture ──────────────────────────────────────────────────────────────

class TestBakeTexture(unittest.TestCase):

    def _setup(self, tex_size=16):
        mesh, uv = _mesh_with_uv()
        pos_map, normal_map, valid_mask = sg.build_uv_3d_map(mesh, uv, tex_size=tex_size)
        cam_pos = np.array([0.5, 0.5, 5.0])
        view, pose = sg._look_at(cam_pos, np.array([0.5, 0.5, 0.0]))
        proj = sg._perspective_matrix(math.radians(60), 1.0)
        cameras = [{"pos": cam_pos, "view": view, "proj": proj,
                    "yfov": math.radians(60), "pose": pose}]
        gen_images = [_solid_image(color=(200, 100, 50), size=(64, 64))]
        return pos_map, normal_map, valid_mask, cameras, gen_images, tex_size

    def test_returns_pil_image(self):
        self.assertIsInstance(sg.bake_texture(*self._setup()), Image.Image)

    def test_output_size_matches_tex_size(self):
        result = sg.bake_texture(*self._setup(tex_size=16))
        self.assertEqual(result.size, (16, 16))

    def test_visible_texels_are_painted(self):
        pos_map, normal_map, valid_mask, cameras, gen_images, tex_size = self._setup(16)
        result = sg.bake_texture(pos_map, normal_map, valid_mask,
                                  cameras, gen_images, tex_size)
        arr = np.array(result)
        painted = (arr > 0).any(axis=-1) & valid_mask
        self.assertGreater(painted.sum(), 0)

    def test_all_none_images_gives_black(self):
        pos_map, normal_map, valid_mask, cameras, _, tex_size = self._setup(16)
        result = sg.bake_texture(pos_map, normal_map, valid_mask,
                                  cameras, [None], tex_size)
        self.assertEqual(np.array(result).max(), 0)

    def test_pixel_range_0_255(self):
        arr = np.array(sg.bake_texture(*self._setup()))
        self.assertGreaterEqual(arr.min(), 0)
        self.assertLessEqual(arr.max(), 255)

    def test_back_face_not_painted(self):
        """Camera below the quad (negative Z) should paint nothing."""
        mesh, uv = _mesh_with_uv()
        pos_map, normal_map, valid_mask = sg.build_uv_3d_map(mesh, uv, tex_size=16)
        cam_pos = np.array([0.5, 0.5, -5.0])
        view, pose = sg._look_at(cam_pos, np.array([0.5, 0.5, 0.0]))
        proj = sg._perspective_matrix(math.radians(60), 1.0)
        cameras = [{"pos": cam_pos, "view": view, "proj": proj,
                    "yfov": math.radians(60), "pose": pose}]
        result = sg.bake_texture(pos_map, normal_map, valid_mask,
                                  cameras, [_solid_image((255, 0, 0))], 16)
        self.assertEqual(np.array(result).max(), 0,
                         "Back-facing camera must not paint anything")

    def test_multiple_cameras_contribute(self):
        mesh_s = _sphere_mesh()
        uv_s   = sg._spherical_uv(mesh_s)
        pos_map, normal_map, valid_mask = sg.build_uv_3d_map(mesh_s, uv_s, tex_size=16)
        c = mesh_s.centroid
        r = np.linalg.norm(mesh_s.extents) * 0.8 + 1.0
        cameras = []
        for d in [np.array([1., 0., 0.]), np.array([-1., 0., 0.])]:
            pos = c + r * d
            view, pose = sg._look_at(pos, c)
            proj = sg._perspective_matrix(math.radians(60), 1.0)
            cameras.append({"pos": pos, "view": view, "proj": proj,
                             "yfov": math.radians(60), "pose": pose})
        result = sg.bake_texture(pos_map, normal_map, valid_mask,
                                  cameras,
                                  [_solid_image((200, 100, 50)),
                                   _solid_image((50, 100, 200))], 16)
        self.assertGreater(np.array(result).max(), 0)


# ── build_cameras ─────────────────────────────────────────────────────────────

class TestBuildCameras(unittest.TestCase):

    def test_returns_correct_count(self):
        self.assertEqual(len(sg.build_cameras(_sphere_mesh(), 4, 1, 512, 512)), 4)

    def test_camera_dict_keys(self):
        cam = sg.build_cameras(_sphere_mesh(), 1, 1, 512, 512)[0]
        for key in ("pos", "view", "proj", "yfov", "pose"):
            self.assertIn(key, cam)

    def test_view_and_pose_are_inverses(self):
        for cam in sg.build_cameras(_sphere_mesh(), 3, 1, 256, 256):
            np.testing.assert_allclose(cam["view"] @ cam["pose"], np.eye(4), atol=1e-9)

    def test_all_camera_modes(self):
        mesh = _sphere_mesh()
        for mode in range(1, 8):
            cams = sg.build_cameras(mesh, 4, mode, 256, 256)
            self.assertGreater(len(cams), 0, f"mode {mode} returned no cameras")

    def test_camera_pos_outside_mesh(self):
        mesh = _sphere_mesh()
        r = np.linalg.norm(mesh.extents) / 2
        for cam in sg.build_cameras(mesh, 6, 5, 256, 256):
            dist = np.linalg.norm(cam["pos"] - mesh.centroid)
            self.assertGreater(dist, r * 0.5)

    def test_projection_matrix_shape(self):
        cam = sg.build_cameras(_sphere_mesh(), 1, 1, 256, 256)[0]
        self.assertEqual(cam["proj"].shape, (4, 4))


# ── export_textured_mesh ──────────────────────────────────────────────────────

class TestExportTexturedMesh(unittest.TestCase):

    def _run(self, fmt):
        mesh, uv = _mesh_with_uv()
        tex = _solid_image(size=(64, 64))
        with tempfile.TemporaryDirectory() as tmp:
            sg.export_textured_mesh(mesh, uv, tex, tmp, fmt)
            return os.listdir(tmp)

    def test_none_writes_nothing(self):
        mesh, uv = _mesh_with_uv()
        with tempfile.TemporaryDirectory() as tmp:
            sg.export_textured_mesh(mesh, uv, _solid_image(), tmp, "none")
            self.assertEqual(os.listdir(tmp), [])

    def test_glb_creates_file(self):
        self.assertTrue(any(f.endswith(".glb") for f in self._run("glb")))

    def test_obj_creates_file(self):
        self.assertTrue(any(f.endswith(".obj") for f in self._run("obj")))

    def test_texture_png_saved_for_glb(self):
        self.assertIn("texture.png", self._run("glb"))

    def test_texture_png_saved_for_obj(self):
        self.assertIn("texture.png", self._run("obj"))

    def test_creates_output_dir_if_missing(self):
        mesh, uv = _mesh_with_uv()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "new_subdir")
            sg.export_textured_mesh(mesh, uv, _solid_image(), out, "glb")
            self.assertTrue(os.path.isdir(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
