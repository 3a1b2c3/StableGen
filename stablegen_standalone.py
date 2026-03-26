"""
StableGen standalone CLI — AI texturing pipeline without Blender.

Pipeline:
  1. Load mesh (OBJ / GLB / STL) with trimesh
  2. Place cameras using camera_placement.py strategies
  3. Render depth views with pyrender (optional ControlNet signal)
  4. Generate one image per camera view via ComfyUI SDXL
  5. Bake all views onto the mesh UV atlas (vectorised NumPy)
  6. Export textured mesh (GLB / OBJ+MTL)

Usage (from PS C:\\workspace\\MODEL\\StableGen):
    .\\.venv\\Scripts\\python.exe stablegen_standalone.py `
        --mesh sphere.obj `
        --prompt "ancient stone wall with moss" `
        --output ./out `
        --server 127.0.0.1:8188

    # Check ComfyUI is reachable first:
    .\\.venv\\Scripts\\python.exe stablegen_standalone.py --check

Dependencies:
    .\\.venv\\Scripts\\python.exe -m pip install trimesh pillow numpy requests websocket-client
    .\\.venv\\Scripts\\python.exe -m pip install pyrender      # optional: depth rendering for ControlNet
    .\\.venv\\Scripts\\python.exe -m pip install opencv-python # optional: Canny edges
    .\\.venv\\Scripts\\python.exe -m pip install xatlas        # optional: proper UV unwrap if mesh has no UVs

Arguments:
  --mesh FILE          Input mesh (.obj, .glb, .stl, .fbx)
  --prompt TEXT        Generation prompt
  --negative TEXT      Negative prompt  (default: "")
  --checkpoint FILE    Checkpoint filename in ComfyUI models/checkpoints/
                       (default: sd_xl_base_1.0.safetensors)
  --cameras N          Number of cameras  (default: 6)
  --camera-mode 1-7    Placement strategy (default: 5 = K-means normals)
  --steps N            Diffusion steps    (default: 20)
  --cfg F              CFG scale          (default: 7.0)
  --seed N             Seed, -1 = random  (default: -1)
  --width N            Image width        (default: 1024)
  --height N           Image height       (default: 1024)
  --tex-size N         UV atlas resolution (default: 1024)
  --server ADDR        ComfyUI address    (default: 127.0.0.1:8188)
  --output DIR         Output directory   (default: ./sg_out)
  --export FORMAT      glb | obj | none   (default: glb)
  --no-controlnet      Skip depth ControlNet (txt2img only)
  --controlnet MODEL   ControlNet depth model name in ComfyUI
                       (default: control-lora-depth-rank128.safetensors)
  --controlnet-strength F  ControlNet strength (default: 0.6)
  --save-views         Save each camera's generated image to output dir
"""

import argparse
import json
import math
import os
import random
import sys
import traceback
import urllib.request
import uuid
from io import BytesIO

import numpy as np
import requests
from PIL import Image

import trimesh
import trimesh.visual
import trimesh.visual.texture

STABLEGEN_DIR = os.path.dirname(os.path.abspath(__file__))
if STABLEGEN_DIR not in sys.path:
    sys.path.insert(0, STABLEGEN_DIR)

from camera_placement import (
    kmeans_on_sphere,
    orbit_ring_directions,
    fibonacci_sphere_points,
    compute_pca_axes,
    greedy_coverage_directions,
    visibility_weighted_directions,
    fan_arc_directions,
)

try:
    import pyrender
    _PYRENDER = True
except ImportError:
    _PYRENDER = False

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import xatlas
    _XATLAS = True
except ImportError:
    _XATLAS = False


# ── Camera placement ──────────────────────────────────────────────────────────

def _place_cameras_for_mode(mode, n, normals, areas, verts):
    if mode == 1:
        return orbit_ring_directions(n)
    elif mode == 2:
        return fan_arc_directions(n)
    elif mode == 3:
        return np.array(fibonacci_sphere_points(n))
    elif mode == 4:
        axes = compute_pca_axes(verts)
        dirs = np.vstack([axes, -axes])
        return dirs[:n]
    elif mode == 5:
        return kmeans_on_sphere(normals, areas, k=n)
    elif mode == 6:
        selected, _ = greedy_coverage_directions(normals, areas, max_cameras=n)
        return np.array(selected) if selected else orbit_ring_directions(n)
    elif mode == 7:
        return visibility_weighted_directions(normals, areas, k=n)
    else:
        return orbit_ring_directions(n)


def _look_at(eye, target, up=None):
    """Return (view_matrix 4x4, pyrender_pose 4x4) for a camera."""
    if up is None:
        up = np.array([0., 0., 1.])
    forward = target - eye
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-10:
        forward = np.array([0., 1., 0.])
    else:
        forward /= forward_len

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0., 1., 0.])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-10

    up_actual = np.cross(right, forward)

    # View matrix: world → camera
    R = np.eye(4)
    R[0, :3] = right
    R[1, :3] = up_actual
    R[2, :3] = -forward
    T = np.eye(4)
    T[:3, 3] = -eye
    view = R @ T

    # pyrender pose: camera → world (only used for pyrender rendering)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up_actual
    pose[:3, 2] = -forward
    pose[:3, 3] = eye

    return view, pose


def _perspective_matrix(yfov, aspect, znear=0.01, zfar=1000.0):
    """OpenGL-convention perspective projection matrix."""
    f = 1.0 / math.tan(yfov * 0.5)
    return np.array([
        [f / aspect, 0,  0,                                              0],
        [0,          f,  0,                                              0],
        [0,          0,  (zfar + znear) / (znear - zfar),  2 * zfar * znear / (znear - zfar)],
        [0,          0, -1,                                              0],
    ])


def estimate_settings(mesh):
    """
    Analyse mesh geometry and suggest pipeline settings for an unattended run.

    Camera-mode selection signals
    ─────────────────────────────
    Mode 1  orbit ring     — rotationally symmetric around Z (columns, vases,
                             characters): high Z-elongation + Z-aligned PCA axis
    Mode 2  fan arc        — strong dominant front face (reliefs, logos, busts):
                             area-weighted mean normal has high magnitude
    Mode 3  hemisphere     — nearly uniform normal distribution, no strong structure
    Mode 4  PCA axes       — clearly box-like / elongated along one axis:
                             high PCA eigenvalue ratio, non-Z-aligned primary axis
    Mode 5  K-means        — organic general shape (default)
    Mode 6  greedy         — concave / undercut mesh: low mean face visibility
    Mode 7  vis-weighted   — complex with many hard-to-see faces: high visibility
                             variance (some faces rarely visible)

    Returns a dict with keys matching argparse dest names plus a 'reasons' dict.
    """
    fn    = np.array(mesh.face_normals, dtype=np.float32)   # (F, 3)
    verts = np.array(mesh.vertices,     dtype=np.float32)   # (V, 3)
    areas = np.array([f.area for f in mesh.faces_unique_edges], dtype=np.float32) \
            if hasattr(mesh, 'faces_unique_edges') else \
            np.ones(len(fn), dtype=np.float32)
    # Reliable face areas via cross product
    v0 = verts[mesh.faces[:, 0]]
    v1 = verts[mesh.faces[:, 1]]
    v2 = verts[mesh.faces[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    areas = np.maximum(areas, 1e-12)
    n_faces = len(fn)

    # ── Signal 1: PCA eigenvalues → shape elongation ──────────────────────────
    centered  = verts - verts.mean(axis=0)
    cov       = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)               # ascending order
    eigvals   = np.abs(eigvals[::-1])                     # descending
    eigvecs   = eigvecs[:, ::-1]                          # matching eigenvectors
    primary_axis = eigvecs[:, 0]                          # longest axis
    elongation   = float(eigvals[0] / (eigvals[2] + 1e-12))
    z_align      = float(abs(primary_axis[2]))            # 1.0 = axis is Z

    # ── Signal 2: dominant front-face (area-weighted mean normal magnitude) ───
    aw_mean = (fn * areas[:, np.newaxis]).sum(axis=0)
    aw_mean_mag = float(np.linalg.norm(aw_mean)) / float(areas.sum())
    # 0 = symmetric (all sides equal), 1 = all faces point the same way

    # ── Signal 3: mean face visibility (back-face only) ───────────────────────
    # Sample 100 sphere directions; fraction visible to each face → mean
    cands = np.array(fibonacci_sphere_points(100), dtype=np.float32)
    vis   = fn @ cands.T > 0.26                           # (F, 100) bool
    mean_vis  = float(vis.mean())                         # 0..1
    vis_std   = float(vis.astype(np.float32).std())       # high = uneven visibility

    # ── Signal 4: normal spread ───────────────────────────────────────────────
    normal_spread = float(fn.std(axis=0).mean())          # ~0 convex, ~0.58 uniform

    # ── Camera mode decision ──────────────────────────────────────────────────
    # Priority: most specific signal wins.

    if mean_vis < 0.18:
        # Many faces are rarely visible → greedy maximises coverage
        camera_mode = 6
        mode_reason = f"low mean visibility {mean_vis:.2f} — concave/undercut mesh"

    elif vis_std > 0.38 and mean_vis < 0.30:
        # High spread in visibility → many hard-to-see faces
        camera_mode = 7
        mode_reason = (f"high visibility variance {vis_std:.2f} — "
                       "visibility-weighted K-means")

    elif aw_mean_mag > 0.55:
        # Strong dominant direction → fan arc covers the important face well
        camera_mode = 2
        mode_reason = f"dominant front face (area-weighted normal mag {aw_mean_mag:.2f})"

    elif elongation > 3.5 and z_align > 0.65:
        # Tall object symmetric around Z → orbit ring
        camera_mode = 1
        mode_reason = (f"Z-elongated symmetric shape "
                       f"(elongation {elongation:.1f}, Z-align {z_align:.2f})")

    elif elongation > 3.0 and z_align < 0.35:
        # Elongated but not upright → PCA axes capture the key views
        camera_mode = 4
        mode_reason = (f"elongated non-upright shape "
                       f"(elongation {elongation:.1f}, Z-align {z_align:.2f})")

    elif normal_spread > 0.50:
        # High normal entropy, no special structure → even hemisphere coverage
        camera_mode = 3
        mode_reason = f"high normal spread {normal_spread:.2f} — hemisphere"

    else:
        camera_mode = 5
        mode_reason = f"general organic shape — K-means normals (default)"

    # ── Camera count ──────────────────────────────────────────────────────────
    # More cameras needed when visibility is low or normal entropy is high.
    if mean_vis < 0.20 or normal_spread > 0.50:
        cameras, cam_reason = 10, f"complex surface (vis={mean_vis:.2f}, spread={normal_spread:.2f})"
    elif mean_vis < 0.28 or normal_spread > 0.38:
        cameras, cam_reason = 8,  f"moderate complexity (vis={mean_vis:.2f})"
    elif aw_mean_mag > 0.55:
        cameras, cam_reason = 4,  f"dominant front face — fewer views needed"
    else:
        cameras, cam_reason = 6,  f"standard (vis={mean_vis:.2f}, spread={normal_spread:.2f})"

    # ── Texture size ──────────────────────────────────────────────────────────
    raw = math.sqrt(n_faces) * 4
    tex_size = 512
    for s in (512, 1024, 2048):
        if raw >= s:
            tex_size = s
    tex_reason = f"face count {n_faces:,} → target ~{int(raw)}px"

    # ── ControlNet strength ───────────────────────────────────────────────────
    # Use depth variance relative to bounding-box depth as a structure signal.
    bbox_depth = float(mesh.extents[2]) if mesh.extents[2] > 0 else 1.0
    rel_var    = float(np.var(verts[:, 2])) / (bbox_depth ** 2)
    if rel_var > 0.05:
        cs, cs_reason = 0.75, f"high depth variance {rel_var:.3f} — structured mesh"
    elif rel_var < 0.01:
        cs, cs_reason = 0.50, f"low depth variance {rel_var:.3f} — flat/organic"
    else:
        cs, cs_reason = 0.60, "moderate depth variance (default)"

    return {
        "cameras":              cameras,
        "camera_mode":          camera_mode,
        "tex_size":             tex_size,
        "steps":                20,
        "cfg":                  7.0,
        "controlnet_strength":  cs,
        "reasons": {
            "cameras":             cam_reason,
            "camera_mode":         mode_reason,
            "tex_size":            tex_reason,
            "controlnet_strength": cs_reason,
        },
    }


def build_cameras(mesh, n_cameras, mode, render_w, render_h, yfov_deg=60.0):
    """
    Place cameras around the mesh and return a list of camera dicts.

    Each dict:  {pos, view, proj, yfov, pose}
    """
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area_vec = np.linalg.norm(cross, axis=1)
    areas = area_vec * 0.5
    valid = areas > 1e-12
    normals = cross[valid] / (area_vec[valid, np.newaxis] + 1e-10)
    areas_v = areas[valid]

    dirs = _place_cameras_for_mode(mode, n_cameras, normals, areas_v, verts)
    dirs = np.array(dirs, dtype=float)

    # Scale directions by bounding-box radius
    centroid = mesh.centroid
    extents = mesh.extents
    radius = np.linalg.norm(extents) * 0.8 + 1e-6

    yfov = math.radians(yfov_deg)
    aspect = render_w / render_h

    cameras = []
    for d in dirs:
        d = d / (np.linalg.norm(d) + 1e-10)
        pos = centroid + radius * d
        view, pose = _look_at(pos, centroid)
        proj = _perspective_matrix(yfov, aspect)
        cameras.append({"pos": pos, "view": view, "proj": proj,
                         "yfov": yfov, "pose": pose})
    return cameras


# ── Depth rendering with pyrender ─────────────────────────────────────────────

def render_depth_view(mesh, cam, width, height):
    """
    Render a depth image from the given camera using pyrender.
    Returns a (H, W) float32 array of metric depths, or None if pyrender unavailable.
    """
    if not _PYRENDER:
        return None

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    mesh_py = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_py)

    yfov = cam["yfov"]
    znear = 0.01
    zfar = max(np.linalg.norm(mesh.extents) * 10, 100.0)
    camera = pyrender.PerspectiveCamera(yfov=yfov, znear=znear, zfar=zfar)
    scene.add(camera, pose=cam["pose"])

    try:
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        renderer = pyrender.OffscreenRenderer(width, height)
        _, depth = renderer.render(scene)
        renderer.delete()
    except Exception as e:
        print(f"[standalone] pyrender error: {e}", file=sys.stderr)
        return None

    depth[depth == 0] = np.nan
    return depth.astype(np.float32)


def depth_to_image(depth):
    """Normalise a (H, W) depth array to a (H, W, 3) uint8 image for ControlNet."""
    valid = ~np.isnan(depth)
    if not valid.any():
        h, w = depth.shape
        return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    dmin, dmax = depth[valid].min(), depth[valid].max()
    norm = np.zeros_like(depth)
    norm[valid] = (depth[valid] - dmin) / (dmax - dmin + 1e-10)
    norm = np.nan_to_num(norm, nan=0.0)
    grey = (norm * 255).astype(np.uint8)
    rgb = np.stack([grey, grey, grey], axis=-1)
    return Image.fromarray(rgb)


# ── ComfyUI communication ─────────────────────────────────────────────────────

def _upload_image(server, image_path):
    """Upload image to ComfyUI /upload/image.  Returns {'name': ..., ...} or None."""
    url = f"http://{server}/upload/image"
    with open(image_path, "rb") as f:
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        resp = requests.post(url,
                             files={"image": (os.path.basename(image_path), f, mime)},
                             data={"overwrite": "true", "type": "input"},
                             timeout=30)
    resp.raise_for_status()
    return resp.json()


def check_server(server, timeout=5):
    """
    Verify ComfyUI is reachable and return a dict with server info, or raise.

    Returns dict with keys: online, version, vram_free_mb, vram_total_mb, devices.
    Raises SystemExit(1) with a clear message on failure.
    """
    url = f"http://{server}/system_stats"
    try:
        raw = urllib.request.urlopen(url, timeout=timeout).read()
        data = json.loads(raw)
    except urllib.error.URLError as e:
        print(f"[standalone] ERROR: ComfyUI not reachable at {server}")
        print(f"[standalone]   → {e}")
        print(f"[standalone]   Make sure ComfyUI is running: python main.py --listen")
        sys.exit(1)
    except Exception as e:
        print(f"[standalone] ERROR: unexpected error checking {server}: {e}")
        sys.exit(1)

    devices = data.get("devices", [{}])
    dev = devices[0] if devices else {}
    vram_free  = dev.get("vram_free",  0) / (1024 ** 2)
    vram_total = dev.get("vram_total", 0) / (1024 ** 2)

    info = {
        "online":        True,
        "version":       data.get("system", {}).get("comfyui_version", "unknown"),
        "vram_free_mb":  vram_free,
        "vram_total_mb": vram_total,
        "devices":       devices,
    }
    print(f"[standalone] ComfyUI OK  version={info['version']}  "
          f"VRAM {vram_free:.0f}/{vram_total:.0f} MB free  ({server})")
    return info


def validate_checkpoint(server, checkpoint, timeout=10):
    """Query ComfyUI for available checkpoints and verify *checkpoint* is listed.
    Exits with code 1 and a clear message if not found.
    """
    url = f"http://{server}/object_info/CheckpointLoaderSimple"
    try:
        raw = urllib.request.urlopen(url, timeout=timeout).read()
        data = json.loads(raw)
    except Exception as e:
        print(f"[standalone] WARNING: could not fetch checkpoint list: {e}", file=sys.stderr)
        return  # non-fatal — let generation fail with a more specific error if needed

    available = (data.get("CheckpointLoaderSimple", {})
                     .get("input", {})
                     .get("required", {})
                     .get("ckpt_name", [None])[0]) or []
    if isinstance(available, list) and checkpoint not in available:
        print(f"[standalone] ERROR: checkpoint not found on server: {checkpoint!r}", file=sys.stderr)
        print(f"[standalone]   Available checkpoints:", file=sys.stderr)
        for name in sorted(available):
            print(f"[standalone]     {name}", file=sys.stderr)
        print(f"[standalone]   Use --checkpoint to pick one of the above.", file=sys.stderr)
        sys.exit(1)


def _queue_prompt(server, client_id, prompt):
    data = json.dumps({"prompt": prompt, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"http://{server}/prompt", data=data,
        headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            detail = json.loads(body)
            # ComfyUI puts validation errors under 'error' and 'node_errors'
            msg = detail.get("error", {})
            node_errors = detail.get("node_errors", {})
            print(f"[standalone] ComfyUI 400 error: {msg}", file=sys.stderr)
            for nid, nerr in node_errors.items():
                print(f"[standalone]   node {nid}: {nerr}", file=sys.stderr)
        except Exception:
            print(f"[standalone] ComfyUI 400 response: {body[:500]}", file=sys.stderr)
        raise
    return resp["prompt_id"]


def _run_workflow(server, prompt_dict):
    """Queue a prompt and collect the first image from a SaveImageWebsocket node."""
    try:
        import websocket as _websocket
        if not hasattr(_websocket, 'create_connection'):
            raise ImportError(
                "Wrong 'websocket' package installed (WSGI server package). "
                "Run: .venv\\Scripts\\python.exe -m pip uninstall websocket -y && "
                ".venv\\Scripts\\python.exe -m pip install websocket-client"
            )
    except ImportError as _e:
        print(f"[standalone] ERROR: {_e}", file=sys.stderr)
        return None
    client_id = str(uuid.uuid4())
    ws = _websocket.create_connection(
        f"ws://{server}/ws?clientId={client_id}", timeout=10)
    ws.settimeout(300)

    prompt_id = _queue_prompt(server, client_id, prompt_dict)
    images = {}
    current_node = ""

    while True:
        try:
            out = ws.recv()
        except _websocket.WebSocketTimeoutException:
            print("[standalone] WebSocket timeout", file=sys.stderr)
            break
        except Exception as e:
            print(f"[standalone] WebSocket error: {e}", file=sys.stderr)
            break

        if isinstance(out, str):
            msg = json.loads(out)
            if msg["type"] == "executing":
                d = msg["data"]
                if d.get("prompt_id") == prompt_id:
                    if d["node"] is None:
                        print()
                        break
                    if d["node"] != current_node:
                        if current_node:
                            print()   # finish previous node line
                        current_node = d["node"]
                        print(f"[standalone]   node {current_node} ...", flush=True)
            elif msg["type"] == "progress":
                v, m = msg["data"]["value"], msg["data"]["max"]
                bar_w = 30
                filled = int(bar_w * v / m) if m else 0
                bar = "#" * filled + "-" * (bar_w - filled)
                print(f"\r  [{bar}] {v:>4}/{m}", end="", flush=True)
            elif msg["type"] == "execution_error":
                print(f"\n[standalone] Execution error: {msg['data']}", file=sys.stderr)
                break
        elif isinstance(out, bytes):
            # Binary: [8-byte header][PNG bytes]
            if len(out) > 8:
                images.setdefault(current_node, []).append(out[8:])

    ws.close()
    return images


def _build_sdxl_txt2img(prompt, negative, checkpoint, steps, cfg, seed, width, height):
    if seed < 0:
        seed = random.randint(0, 2**31)
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0],
                         "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "karras",
                         "denoise": 1.0}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImageWebsocket",
              "inputs": {"images": ["6", 0]}},
    }


def _build_sdxl_controlnet_depth(prompt, negative, checkpoint,
                                  controlnet_model, controlnet_strength,
                                  depth_image_name,
                                  steps, cfg, seed, width, height):
    if seed < 0:
        seed = random.randint(0, 2**31)
    return {
        "1":  {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": checkpoint}},
        "2":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": negative, "clip": ["1", 1]}},
        "4":  {"class_type": "EmptyLatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        # ControlNet
        "10": {"class_type": "ControlNetLoader",
               "inputs": {"control_net_name": controlnet_model}},
        "11": {"class_type": "LoadImage",
               "inputs": {"image": depth_image_name}},
        "12": {"class_type": "ControlNetApplyAdvanced",
               "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                          "control_net": ["10", 0], "image": ["11", 0],
                          "strength": controlnet_strength,
                          "start_percent": 0.0, "end_percent": 1.0}},
        # Sampler + decode
        "5":  {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "positive": ["12", 0],
                          "negative": ["12", 1], "latent_image": ["4", 0],
                          "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "euler", "scheduler": "karras",
                          "denoise": 1.0}},
        "6":  {"class_type": "VAEDecode",
               "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7":  {"class_type": "SaveImageWebsocket",
               "inputs": {"images": ["6", 0]}},
    }


# ── Known upscale models ───────────────────────────────────────────────────────

_UPSCALE_MODELS = {
    "4x-UltraSharp.pth": {
        "url":  "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        "desc": "4x — very sharp edges, great for textures (recommended)",
        "size": "~67 MB",
    },
    "RealESRGAN_x4plus.pth": {
        "url":  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "desc": "4x — natural/organic surfaces, less aggressive sharpening",
        "size": "~64 MB",
    },
    "4x_foolhardy_Remacri.pth": {
        "url":  "https://huggingface.co/FacehuggerR/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
        "desc": "4x — painterly / stylised output",
        "size": "~67 MB",
    },
    "8x_NMKD-Superscale_150000_G.pth": {
        "url":  "https://huggingface.co/Akumetsu971/SD_Anime_Futuristic_Armor/resolve/main/8x_NMKD-Superscale_150000_G.pth",
        "desc": "8x — maximum resolution, longer processing",
        "size": "~67 MB",
    },
}


def download_upscaler(model_name, dest_dir):
    """
    Download a known upscale model into dest_dir with a progress bar.
    dest_dir should be ComfyUI's models/upscale_models/ folder.
    Returns the saved file path, or None on failure.
    """
    if model_name not in _UPSCALE_MODELS:
        print(f"[download] Unknown model '{model_name}'.", file=sys.stderr)
        print(f"[download] Known models:", file=sys.stderr)
        for name, info in _UPSCALE_MODELS.items():
            print(f"             {name}  — {info['desc']}", file=sys.stderr)
        return None

    entry    = _UPSCALE_MODELS[model_name]
    url      = entry["url"]
    out_path = os.path.join(dest_dir, model_name)

    if os.path.exists(out_path):
        print(f"[download] Already exists: {out_path}")
        return out_path

    os.makedirs(dest_dir, exist_ok=True)
    print(f"[download] {model_name}  ({entry['size']})  {entry['desc']}")
    print(f"[download] → {out_path}")

    try:
        resp = requests.get(url, stream=True, timeout=60,
                            headers={"User-Agent": "stablegen-standalone/1.0"})
        resp.raise_for_status()
        total      = int(resp.headers.get("content-length", 0))
        downloaded = 0
        bar_width  = 40

        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    frac   = downloaded / total
                    filled = int(bar_width * frac)
                    bar    = "#" * filled + "-" * (bar_width - filled)
                    mb     = downloaded / (1024 ** 2)
                    total_mb = total   / (1024 ** 2)
                    print(f"\r  [{bar}] {mb:.1f}/{total_mb:.1f} MB",
                          end="", flush=True)
        print()
        print(f"[download] Saved: {out_path}")
        return out_path

    except Exception as e:
        print(f"\n[download] Failed: {e}", file=sys.stderr)
        if os.path.exists(out_path):
            os.remove(out_path)
        return None


def _build_upscale_workflow(image_name, upscale_model):
    """
    ComfyUI workflow: load image → ImageUpscaleWithModel → SaveImageWebsocket.
    Requires an upscale model in ComfyUI models/upscale_models/
    (e.g. 4x-UltraSharp.pth, RealESRGAN_x4plus.pth).
    """
    return {
        "1": {"class_type": "UpscaleModelLoader",
              "inputs": {"model_name": upscale_model}},
        "2": {"class_type": "LoadImage",
              "inputs": {"image": image_name}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["1", 0], "image": ["2", 0]}},
        "4": {"class_type": "SaveImageWebsocket",
              "inputs": {"images": ["3", 0]}},
    }


def upscale_texture(server, texture_img, upscale_model, save_dir=None):
    """
    Upload texture_img to ComfyUI, run an upscale model, return upscaled PIL.Image.
    Returns the original image unchanged if anything fails.
    """
    tmp = os.path.join(save_dir or ".", "_texture_for_upscale.png")
    texture_img.save(tmp)
    try:
        info = _upload_image(server, tmp)
        if not info:
            print("[upscale] Upload failed — skipping upscale", file=sys.stderr)
            return texture_img
        image_name = info.get("name", os.path.basename(tmp))
        wf = _build_upscale_workflow(image_name, upscale_model)
        result = _run_workflow(server, wf)
        if result is None:
            print("[upscale] Workflow returned no image — skipping upscale",
                  file=sys.stderr)
            return texture_img
        print(f"[upscale] {texture_img.size[0]}x{texture_img.size[1]}"
              f" → {result.size[0]}x{result.size[1]}  (model: {upscale_model})")
        return result
    except Exception as e:
        print(f"[upscale] Failed: {e} — keeping original texture", file=sys.stderr)
        return texture_img
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def generate_view(server, args, cam_idx, depth_img=None, save_dir=None):
    """
    Generate one camera-view image via ComfyUI.
    Returns PIL.Image (RGB) or None on failure.
    """
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31)
    # Vary seed per view so each camera gets a different image
    seed = (seed + cam_idx * 13337) & 0x7FFFFFFF

    depth_name = None
    if depth_img is not None and not args.no_controlnet:
        tmp_depth = os.path.join(save_dir or ".", f"_depth_{cam_idx:02d}.png")
        depth_img.save(tmp_depth)
        try:
            info = _upload_image(server, tmp_depth)
            depth_name = info.get("name") if info else None
        except Exception as e:
            print(f"[standalone] depth upload failed: {e}", file=sys.stderr)
            depth_name = None

    if depth_name:
        wf = _build_sdxl_controlnet_depth(
            args.prompt, args.negative, args.checkpoint,
            args.controlnet, args.controlnet_strength,
            depth_name,
            args.steps, args.cfg, seed, args.width, args.height,
        )
    else:
        wf = _build_sdxl_txt2img(
            args.prompt, args.negative, args.checkpoint,
            args.steps, args.cfg, seed, args.width, args.height,
        )

    try:
        images = _run_workflow(server, wf)
    except Exception as e:
        print(f"[standalone] generation failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

    if not images:
        return None

    # Find first binary result from node "7" (SaveImageWebsocket)
    raw = None
    for node_id in ("7", "6", "5"):
        if node_id in images and images[node_id]:
            raw = images[node_id][0]
            break
    if raw is None and images:
        raw = next(iter(images.values()))[0]
    if raw is None:
        print(f"[standalone] no image received for view {cam_idx}", file=sys.stderr)
        return None

    img = Image.open(BytesIO(raw)).convert("RGB")

    if save_dir and args.save_views:
        p = os.path.abspath(os.path.join(save_dir, f"view_{cam_idx:02d}.png"))
        img.save(p)
        print(f"[standalone] Saved: {p}")

    return img


# ── UV handling ───────────────────────────────────────────────────────────────

def _get_uv(mesh):
    """Return per-vertex UV array (n_verts, 2) or None."""
    vis = mesh.visual
    if isinstance(vis, trimesh.visual.texture.TextureVisuals):
        uv = vis.uv
        if uv is not None and len(uv) == len(mesh.vertices):
            return np.array(uv, dtype=float)
    return None


def _spherical_uv(mesh):
    """Fallback: cylindrical projection UV mapping."""
    verts = mesh.vertices - mesh.centroid
    norm = verts / (np.linalg.norm(verts, axis=1, keepdims=True) + 1e-10)
    u = 0.5 + np.arctan2(norm[:, 1], norm[:, 0]) / (2 * math.pi)
    v = 0.5 + np.arcsin(np.clip(norm[:, 2], -1.0, 1.0)) / math.pi
    return np.stack([u, v], axis=1)


def _xatlas_uv(mesh):
    """Use xatlas to generate a proper UV parameterization."""
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    atlas.generate()
    vmapping, indices, uvs = atlas[0]
    new_verts = mesh.vertices[vmapping]
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=indices, process=False)
    return new_mesh, uvs


def ensure_uv(mesh, force_spherical=False, force_xatlas=False):
    """
    Return (mesh, uv) where uv is (n_verts, 2) in [0,1].
    Priority: existing UVs → xatlas → spherical.
    """
    if not force_spherical and not force_xatlas:
        uv = _get_uv(mesh)
        if uv is not None:
            print(f"[standalone] Using existing UV coords ({len(uv)} verts)")
            return mesh, uv

    if _XATLAS and not force_spherical:
        try:
            print("[standalone] Generating UV with xatlas ...")
            mesh_new, uv = _xatlas_uv(mesh)
            print(f"[standalone] xatlas UV: {len(uv)} verts, {len(mesh_new.faces)} faces")
            return mesh_new, uv
        except Exception as e:
            print(f"[standalone] xatlas failed: {e}, falling back to spherical", file=sys.stderr)

    print("[standalone] Using spherical UV mapping (install xatlas for better results)")
    uv = _spherical_uv(mesh)
    return mesh, uv


# ── UV-to-3D precomputation ───────────────────────────────────────────────────

def _barycentric_2d(p, a, b, c):
    """
    Vectorised 2-D barycentric coordinates.
    p: (..., 2), a/b/c: (2,)  →  u, v, w  shape (...)
    """
    v0, v1 = b - a, c - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    v2 = p - a
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01 + 1e-12
    bb = (d11 * d20 - d01 * d21) / denom
    gg = (d00 * d21 - d01 * d20) / denom
    aa = 1.0 - bb - gg
    return aa, bb, gg


def build_uv_3d_map(mesh, uv, tex_size):
    """
    Rasterise every triangle in UV space and build per-texel lookups.

    Returns:
        pos_map    (tex_size, tex_size, 3) float32  — world-space position
        normal_map (tex_size, tex_size, 3) float32  — face normal
        valid_mask (tex_size, tex_size)    bool
    """
    print(f"[standalone] Building UV→3D map ({tex_size}×{tex_size}) ...")
    uv_arr = np.array(uv, dtype=float)
    print(f"[standalone] UV range: U [{uv_arr[:,0].min():.4f}, {uv_arr[:,0].max():.4f}]  "
          f"V [{uv_arr[:,1].min():.4f}, {uv_arr[:,1].max():.4f}]")
    verts = np.array(mesh.vertices, dtype=float)
    faces = np.array(mesh.faces)
    face_normals = np.array(mesh.face_normals, dtype=float)

    pos_map    = np.full((tex_size, tex_size, 3), np.nan, dtype=np.float32)
    normal_map = np.full((tex_size, tex_size, 3), np.nan, dtype=np.float32)

    for fi, face in enumerate(faces):
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        u0, u1, u2 = uv[face[0]], uv[face[1]], uv[face[2]]

        # UV → pixel coords (origin top-left, V flipped)
        px = np.array([u0[0], u1[0], u2[0]]) * tex_size
        py = np.array([(1 - u0[1]), (1 - u1[1]), (1 - u2[1])]) * tex_size

        mn_x = max(0, int(math.floor(px.min())))
        mx_x = min(tex_size - 1, int(math.ceil(px.max())))
        mn_y = max(0, int(math.floor(py.min())))
        mx_y = min(tex_size - 1, int(math.ceil(py.max())))

        if mx_x < mn_x or mx_y < mn_y:
            continue

        # Grid of candidate pixels (centre of each texel)
        xs = np.arange(mn_x, mx_x + 1) + 0.5
        ys = np.arange(mn_y, mx_y + 1) + 0.5
        GX, GY = np.meshgrid(xs, ys)  # (H_sub, W_sub)

        pu = GX / tex_size
        pv = 1.0 - GY / tex_size
        pts = np.stack([pu, pv], axis=-1)  # (H_sub, W_sub, 2)

        a2d = np.array([u0[0], u0[1]])
        b2d = np.array([u1[0], u1[1]])
        c2d = np.array([u2[0], u2[1]])

        aa, bb, gg = _barycentric_2d(pts, a2d, b2d, c2d)
        inside = (aa >= -1e-6) & (bb >= -1e-6) & (gg >= -1e-6)

        if not inside.any():
            continue

        P3 = (aa[..., None] * v0 + bb[..., None] * v1 + gg[..., None] * v2)

        ty_all = (GY - 0.5).astype(int)
        tx_all = (GX - 0.5).astype(int)
        ty_valid = ty_all[inside]
        tx_valid = tx_all[inside]

        pos_map[ty_valid, tx_valid]    = P3[inside].astype(np.float32)
        normal_map[ty_valid, tx_valid] = face_normals[fi].astype(np.float32)

    valid_mask = ~np.isnan(pos_map[:, :, 0])
    n_valid = valid_mask.sum()
    print(f"[standalone] UV map: {n_valid:,} texels covered "
          f"({100 * n_valid / tex_size**2:.1f}%)")
    return pos_map, normal_map, valid_mask


# ── Texture baking ────────────────────────────────────────────────────────────

def bake_texture(pos_map, normal_map, valid_mask, cameras, gen_images, tex_size):
    """
    Vectorised multi-view texture baking.

    For every valid texel (u, v):
      • Find all cameras that see the underlying 3-D point.
      • Sample the corresponding pixel from each camera's generated image.
      • Blend with angle-based weights (cos(θ) = dot(normal, view_dir)).

    Returns PIL.Image (RGB).
    """
    print("[standalone] Baking texture ...")
    tex   = np.zeros((tex_size, tex_size, 3), np.float32)
    wts   = np.zeros((tex_size, tex_size),    np.float32)

    ty_all, tx_all = np.where(valid_mask)
    P_flat = pos_map[ty_all, tx_all]      # (N, 3)
    N_flat = normal_map[ty_all, tx_all]   # (N, 3)
    N_flat = N_flat / (np.linalg.norm(N_flat, axis=1, keepdims=True) + 1e-10)

    for ci, (cam, gen_img) in enumerate(zip(cameras, gen_images)):
        if gen_img is None:
            continue

        img_arr = np.array(gen_img.convert("RGB"), dtype=np.float32)
        IH, IW = img_arr.shape[:2]

        cam_pos = np.array(cam["pos"], dtype=float)
        view    = cam["view"]   # (4, 4)
        proj    = cam["proj"]   # (4, 4)

        # --- Compute view directions (N, 3) ---
        vd   = cam_pos - P_flat
        dist = np.linalg.norm(vd, axis=1, keepdims=True)
        vd_n = vd / (dist + 1e-10)

        # Front-face: dot > 0
        dots = (N_flat * vd_n).sum(axis=1)
        front = dots > 0.0
        if not front.any():
            continue

        # --- World → clip space (batch mat-mul) ---
        P_h    = np.concatenate([P_flat, np.ones((len(P_flat), 1))], axis=1)  # (N, 4)
        P_cam  = (view  @ P_h.T).T   # (N, 4)
        P_clip = (proj  @ P_cam.T).T # (N, 4)

        w      = P_clip[:, 3]
        valid_w = w > 1e-6
        ndc    = P_clip[:, :3] / (w[:, None] + 1e-10)

        in_frustum = ((ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) &
                      (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1))

        valid = front & valid_w & in_frustum
        if not valid.any():
            continue

        # --- Screen coordinates ---
        sx = np.clip(( ndc[:, 0] * 0.5 + 0.5) * IW, 0, IW - 1).astype(int)
        sy = np.clip((1 - (ndc[:, 1] * 0.5 + 0.5)) * IH, 0, IH - 1).astype(int)

        # --- Sample colours ---
        colors = img_arr[sy, sx]           # (N, 3)
        w_ang  = np.where(valid, dots, 0.0)

        # Accumulate into texture
        idx_valid = np.where(valid)[0]
        np.add.at(tex, (ty_all[idx_valid], tx_all[idx_valid]),
                  (colors * w_ang[:, None])[idx_valid])
        np.add.at(wts, (ty_all[idx_valid], tx_all[idx_valid]),
                  w_ang[idx_valid])

        n_painted = int(valid.sum())
        print(f"[standalone] Camera {ci+1}/{len(cameras)}: painted {n_painted:,} texels")

    # Normalise
    mask = wts > 0
    tex[mask] /= wts[mask, np.newaxis]
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    print(f"[standalone] Bake complete — {mask.sum():,} texels filled")
    return Image.fromarray(tex, "RGB")


# ── Mesh export ───────────────────────────────────────────────────────────────

def export_textured_mesh(mesh, uv, texture_img, output_dir, fmt):
    if fmt == "none":
        return

    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, "texture.png")
    texture_img.save(tex_path)
    print(f"[standalone] Texture saved: {os.path.abspath(tex_path)}")

    # Attach UV + texture to mesh copy
    material = trimesh.visual.texture.SimpleMaterial(image=texture_img)
    vis = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    mesh_out = mesh.copy()
    mesh_out.visual = vis

    if fmt == "glb":
        out_path = os.path.join(output_dir, "textured_mesh.glb")
        mesh_out.export(out_path)
    elif fmt == "obj":
        out_path = os.path.join(output_dir, "textured_mesh.obj")
        mesh_out.export(out_path)
    else:
        print(f"[standalone] Unknown export format: {fmt}", file=sys.stderr)
        return

    print(f"[standalone] Exported: {os.path.abspath(out_path)}")


from stablegen_viewer import view_result, view_cameras


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="StableGen standalone — AI texturing without Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mesh",                metavar="FILE")
    p.add_argument("--prompt",              metavar="TEXT",    default="")
    p.add_argument("--negative",            metavar="TEXT",    default="")
    p.add_argument("--checkpoint",          metavar="FILE",    default="RealVisXL_V5.0_fp16.safetensors")
    p.add_argument("--cameras",             type=int,          default=6)
    p.add_argument("--camera-mode",         type=int,          default=5,
                   choices=range(1, 8), metavar="1-7")
    p.add_argument("--steps",               type=int,          default=20)
    p.add_argument("--cfg",                 type=float,        default=7.0)
    p.add_argument("--seed",                type=int,          default=-1)
    p.add_argument("--width",               type=int,          default=1024)
    p.add_argument("--height",              type=int,          default=1024)
    p.add_argument("--tex-size",            type=int,          default=1024)
    p.add_argument("--server",              metavar="ADDR",    default="127.0.0.1:8188")
    p.add_argument("--output",              metavar="DIR",     default="./sg_out")
    p.add_argument("--export",              metavar="FORMAT",  default="glb",
                   choices=["glb", "obj", "none"])
    p.add_argument("--no-controlnet",       action="store_true")
    p.add_argument("--controlnet",          metavar="MODEL",
                   default="control-lora-depth-rank128.safetensors")
    p.add_argument("--controlnet-strength", type=float,        default=0.6)
    p.add_argument("--save-views",          action="store_true")
    p.add_argument("--spherical-uv",        action="store_true",
                   help="Force spherical UV instead of mesh UVs or xatlas")
    p.add_argument("--upscale-result",      action="store_true",
                   help="Upscale the baked texture with an upscale model after generation")
    p.add_argument("--upscale-model",       metavar="MODEL",
                   default="4x-UltraSharp.pth",
                   help="Upscale model in ComfyUI models/upscale_models/ "
                        "(default: 4x-UltraSharp.pth)")
    p.add_argument("--list-upscalers",      action="store_true",
                   help="List known downloadable upscale models and exit")
    p.add_argument("--download-upscaler",   metavar="MODEL",
                   help="Download a known upscale model (use --list-upscalers to see names)")
    p.add_argument("--comfyui-dir",         metavar="DIR",
                   default="",
                   help="Path to ComfyUI installation (used by --download-upscaler to "
                        "place model in models/upscale_models/)")
    p.add_argument("--check",               action="store_true",
                   help="Check ComfyUI connection and print server info, then exit")
    p.add_argument("--suggest",             action="store_true",
                   help="Load mesh, print recommended settings as a command line, then exit")
    p.add_argument("--auto",                action="store_true",
                   help="Apply estimated settings (cameras/camera-mode/tex-size/controlnet-strength)"
                        " unless those flags were explicitly passed")
    p.add_argument("--view",                action="store_true",
                   help="Open an interactive pygame viewer after texturing completes")
    p.add_argument("--camera-gui",          action="store_true",
                   help="Show camera placement GUI before generation (Enter=proceed, Esc=abort)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # --list-upscalers / --download-upscaler: no server or mesh needed
    if args.list_upscalers:
        print("\nKnown upscale models (--download-upscaler <name>):\n")
        for name, info in _UPSCALE_MODELS.items():
            print(f"  {name}")
            print(f"    {info['desc']}  {info['size']}")
            print(f"    {info['url']}")
            print()
        sys.exit(0)

    if args.download_upscaler:
        if args.comfyui_dir:
            dest = os.path.join(args.comfyui_dir, "models", "upscale_models")
        else:
            dest = "."
            print("[download] --comfyui-dir not set; saving to current directory")
            print("[download] Use --comfyui-dir <path> to save directly into ComfyUI")
        result = download_upscaler(args.download_upscaler, dest)
        sys.exit(0 if result else 1)

    # Track which auto-able args were explicitly provided vs left at default.
    _AUTO_KEYS = ("cameras", "camera_mode", "tex_size", "controlnet_strength")
    _explicit = {k: getattr(args, k) for k in _AUTO_KEYS
                 if getattr(args, k) != {
                     "cameras": 6, "camera_mode": 5,
                     "tex_size": 1024, "controlnet_strength": 0.6,
                 }[k]}

    # 1. Check ComfyUI — exits with code 1 and a clear message on failure.
    #    --suggest skips this (no server needed to analyse a mesh).
    if not args.suggest:
        check_server(args.server)
        validate_checkpoint(args.server, args.checkpoint)
    if args.check:
        sys.exit(0)

    if not args.mesh:
        print("[standalone] ERROR: --mesh required", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.mesh):
        print(f"[standalone] ERROR: mesh not found: {args.mesh}", file=sys.stderr)
        sys.exit(1)

    # 2. Load mesh
    print(f"[standalone] Loading mesh ...")
    raw = trimesh.load(args.mesh, process=False)
    if isinstance(raw, trimesh.Scene):
        parts = list(raw.geometry.values())
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    else:
        mesh = raw
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump()) if hasattr(mesh, "dump") else trimesh.Trimesh()

    if not isinstance(mesh, trimesh.Trimesh):
        print("[standalone] ERROR: could not load a Trimesh", file=sys.stderr)
        sys.exit(1)

    print(f"[standalone] Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    # 2b. --suggest: print recommended command line and exit
    if args.suggest:
        s = estimate_settings(mesh)
        r = s["reasons"]
        print("\n[suggest] ── Recommended settings ──────────────────────────")
        print(f"[suggest]  --cameras             {s['cameras']:>4}   # {r['cameras']}")
        print(f"[suggest]  --camera-mode         {s['camera_mode']:>4}   # {r['camera_mode']}")
        print(f"[suggest]  --tex-size            {s['tex_size']:>4}   # {r['tex_size']}")
        print(f"[suggest]  --steps               {s['steps']:>4}")
        print(f"[suggest]  --cfg                {s['cfg']:>5.1f}")
        print(f"[suggest]  --controlnet-strength {s['controlnet_strength']:.2f}   # {r['controlnet_strength']}")
        print()
        prompt_q = f'--prompt "YOUR PROMPT"' if not args.prompt else f'--prompt {args.prompt!r}'
        print(f"[suggest] Full command:")
        print(f"  .\\venv\\Scripts\\python.exe stablegen_standalone.py \\")
        print(f"    --mesh {args.mesh} {prompt_q} \\")
        print(f"    --cameras {s['cameras']} --camera-mode {s['camera_mode']} \\")
        print(f"    --tex-size {s['tex_size']} --steps {s['steps']} --cfg {s['cfg']} \\")
        print(f"    --controlnet-strength {s['controlnet_strength']} \\")
        print(f"    --checkpoint {args.checkpoint} --server {args.server} --output {args.output}")
        sys.exit(0)

    # 2c. --auto: apply estimated settings for any arg not explicitly passed
    if args.auto:
        s = estimate_settings(mesh)
        for key in _AUTO_KEYS:
            if key not in _explicit:
                setattr(args, key, s[key])
                print(f"[auto] {key}={s[key]}  ({s['reasons'].get(key, '')})")

    print(f"[standalone] ── StableGen standalone ──")
    print(f"[standalone] Mesh      : {args.mesh}")
    print(f"[standalone] Prompt    : {args.prompt!r}")
    print(f"[standalone] Checkpoint: {args.checkpoint}")
    print(f"[standalone] Server    : {args.server}")
    print(f"[standalone] Cameras   : {args.cameras} (mode {args.camera_mode})")
    print(f"[standalone] pyrender  : {'yes' if _PYRENDER else 'no (pip install pyrender)'}")
    print(f"[standalone] xatlas    : {'yes' if _XATLAS else 'no (pip install xatlas)'}")

    os.makedirs(args.output, exist_ok=True)

    # 3. UV
    mesh, uv = ensure_uv(mesh, force_spherical=args.spherical_uv)

    # 4. Place cameras
    cameras = build_cameras(mesh, args.cameras, args.camera_mode,
                            args.width, args.height)
    print(f"[standalone] Placed {len(cameras)} cameras")

    # 4b. Optional camera placement GUI
    if args.camera_gui:
        if not view_cameras(mesh, cameras):
            print("[standalone] Aborted by user in camera GUI.")
            sys.exit(0)

    # 5. Generate one image per camera
    gen_images = []
    view_warnings = []

    if not _PYRENDER:
        view_warnings.append("pyrender not installed — depth ControlNet skipped")
    if not _XATLAS and args.spherical_uv:
        view_warnings.append("xatlas not installed — spherical UV used")

    for ci, cam in enumerate(cameras):
        print(f"\n[standalone] ── Camera {ci+1}/{len(cameras)} ──")

        # Render depth (optional)
        depth_img = None
        if _PYRENDER and not args.no_controlnet:
            depth = render_depth_view(mesh, cam, args.width, args.height)
            if depth is not None:
                depth_img = depth_to_image(depth)
                print(f"[standalone] Depth rendered")
            else:
                print("[standalone] Depth rendering failed, skipping ControlNet")

        img = generate_view(args.server, args, ci, depth_img=depth_img,
                            save_dir=args.output)
        if img is None:
            print(f"[standalone] ERROR: camera {ci+1} generation failed", file=sys.stderr)
            sys.exit(1)
        print(f"[standalone] Image: {img.size}")
        gen_images.append(img)

    # 6. Build UV map and bake
    pos_map, normal_map, valid_mask = build_uv_3d_map(mesh, uv, args.tex_size)
    texture = bake_texture(pos_map, normal_map, valid_mask,
                           cameras, gen_images, args.tex_size)

    # 7. Optional texture upscale
    if args.upscale_result:
        print(f"[standalone] Upscaling texture with {args.upscale_model} ...")
        texture = upscale_texture(args.server, texture, args.upscale_model,
                                  save_dir=args.output)

    # 8. Export
    export_textured_mesh(mesh, uv, texture, args.output, args.export)
    print(f"\n[standalone] Done.  Output: {os.path.abspath(args.output)}")

    # 9. Optional interactive viewer
    if args.view:
        view_result(mesh, uv, texture,
                    warnings=view_warnings or None,
                    gen_images=gen_images)


if __name__ == "__main__":
    main()
