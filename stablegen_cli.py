"""
StableGen headless CLI — runs the full texturing pipeline without opening Blender's GUI.

Usage (run from the StableGen directory):
    blender --background --python stablegen_cli.py -- [options]

Examples:
    blender --background --python stablegen_cli.py -- \
        --mesh sphere.obj \
        --prompt "ancient stone wall with moss" \
        --output ./out \
        --server 127.0.0.1:8188

    blender --background --python stablegen_cli.py -- \
        --mesh mymodel.glb \
        --prompt "cyberpunk neon robot" \
        --preset "Default" \
        --cameras 6 \
        --camera-mode 5 \
        --checkpoint sdxl_base_1.0.safetensors \
        --output ./out \
        --export glb

Arguments:
    --mesh FILE             Input mesh file (.obj, .glb, .fbx, .stl)
    --prompt TEXT           Generation prompt
    --negative TEXT         Negative prompt  (default: "")
    --preset NAME           StableGen preset name  (default: "Default")
    --checkpoint FILE       Checkpoint filename in ComfyUI models/checkpoints/
    --cameras N             Number of cameras  (default: 6)
    --camera-mode 1-7       Camera placement strategy  (default: 5 = K-means)
    --steps N               Diffusion steps  (default: from preset)
    --cfg F                 CFG scale  (default: from preset)
    --seed N                Seed (-1 = random)  (default: -1)
    --server ADDR           ComfyUI server address  (default: 127.0.0.1:8188)
    --output DIR            Output directory  (default: ./stablegen_out)
    --export FORMAT         Export format after baking: glb | fbx | obj | none
                            (default: none)
    --bake                  Bake textures to UV map after generation
    --no-controlnet         Disable ControlNet (faster, less geometry-aware)
    --architecture ARCH     Model architecture: sdxl | flux | qwen  (default: sdxl)
    --blend FILE            Use existing .blend file instead of importing mesh
"""

import sys
import os
import argparse

# ── Parse args before bpy loads (args after '--' separator) ───────────────────
def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(
        description="StableGen headless CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mesh",         metavar="FILE",   default=None)
    p.add_argument("--blend",        metavar="FILE",   default=None)
    p.add_argument("--prompt",       metavar="TEXT",   default="")
    p.add_argument("--negative",     metavar="TEXT",   default="")
    p.add_argument("--preset",       metavar="NAME",   default="Default")
    p.add_argument("--checkpoint",   metavar="FILE",   default="")
    p.add_argument("--cameras",      type=int,         default=6)
    p.add_argument("--camera-mode",  type=int,         default=5,
                   choices=range(1, 8), metavar="1-7")
    p.add_argument("--steps",        type=int,         default=None)
    p.add_argument("--cfg",          type=float,       default=None)
    p.add_argument("--seed",         type=int,         default=-1)
    p.add_argument("--server",       metavar="ADDR",   default="127.0.0.1:8188")
    p.add_argument("--output",       metavar="DIR",    default="./stablegen_out")
    p.add_argument("--export",       metavar="FORMAT", default="none",
                   choices=["glb", "fbx", "obj", "none"])
    p.add_argument("--bake",         action="store_true")
    p.add_argument("--no-controlnet",action="store_true")
    p.add_argument("--architecture", metavar="ARCH",   default="sdxl",
                   choices=["sdxl", "flux", "qwen"])
    return p.parse_args(argv)


args = _parse_args()

# ── Now import bpy (Blender Python API) ───────────────────────────────────────
import bpy

STABLEGEN_DIR = os.path.dirname(os.path.abspath(__file__))
if STABLEGEN_DIR not in sys.path:
    sys.path.insert(0, STABLEGEN_DIR)

# ── Camera placement mode → StableGen enum string ────────────────────────────
_CAMERA_MODE_MAP = {
    1: "orbit",
    2: "fan",
    3: "hemisphere",
    4: "pca",
    5: "kmeans",
    6: "greedy",
    7: "visibility",
}

# ── Architecture → StableGen enum string ─────────────────────────────────────
_ARCH_MAP = {
    "sdxl":  "sdxl",
    "flux":  "flux1",
    "qwen":  "qwen",
}


def _check_server(server_address):
    """Verify ComfyUI is reachable before starting."""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://{server_address}/system_stats")
        urllib.request.urlopen(req, timeout=5)
        print(f"[stablegen_cli] ComfyUI reachable at {server_address}")
        return True
    except Exception as e:
        print(f"[stablegen_cli] ERROR: Cannot reach ComfyUI at {server_address}: {e}",
              file=sys.stderr)
        return False


def _setup_scene():
    """Reset to a clean empty scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Ensure Cycles is available (needed for baking)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"


def _import_mesh(mesh_path):
    """Import mesh from file, return list of imported mesh objects."""
    ext = os.path.splitext(mesh_path)[1].lower()
    before = set(bpy.data.objects)

    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not imported:
        raise RuntimeError(f"No mesh objects found after importing {mesh_path}")

    print(f"[stablegen_cli] Imported {len(imported)} mesh object(s): "
          f"{[o.name for o in imported]}")
    return imported


def _add_cameras(n_cameras, camera_mode, objects):
    """Place cameras around the mesh using StableGen's placement operator."""
    # Select all mesh objects
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    # Set placement parameters on scene
    scene = bpy.context.scene
    scene.sg_camera_count = n_cameras
    scene.sg_camera_placement = _CAMERA_MODE_MAP.get(camera_mode, "kmeans")

    # Run StableGen's Add Cameras operator
    try:
        bpy.ops.object.stablegen_add_cameras(
            center_type="Object",
            n_cameras=n_cameras,
            placement=_CAMERA_MODE_MAP.get(camera_mode, "kmeans"),
            confirm=True,
        )
        cameras = [o for o in bpy.data.objects if o.type == "CAMERA"]
        print(f"[stablegen_cli] Added {len(cameras)} camera(s)")
        return cameras
    except Exception as e:
        print(f"[stablegen_cli] Camera placement operator failed: {e}", file=sys.stderr)
        print("[stablegen_cli] Falling back to orbit ring placement", file=sys.stderr)
        return _fallback_cameras(n_cameras, objects)


def _fallback_cameras(n_cameras, objects):
    """Simple orbit ring fallback if the operator isn't available."""
    import math
    from mathutils import Vector

    # Compute bounding box centre and radius
    all_verts = []
    for obj in objects:
        for v in obj.data.vertices:
            all_verts.append(obj.matrix_world @ v.co)
    if not all_verts:
        centre = Vector((0, 0, 0))
        radius = 3.0
    else:
        xs = [v.x for v in all_verts]
        ys = [v.y for v in all_verts]
        zs = [v.z for v in all_verts]
        centre = Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2))
        radius = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) * 1.5

    cameras = []
    elevation = math.radians(30)
    for i in range(n_cameras):
        az = 2 * math.pi * i / n_cameras
        x = centre.x + radius * math.cos(az) * math.cos(elevation)
        y = centre.y + radius * math.sin(az) * math.cos(elevation)
        z = centre.z + radius * math.sin(elevation)

        cam_data = bpy.data.cameras.new(name=f"SGCam_{i:02d}")
        cam_obj = bpy.data.objects.new(f"SGCam_{i:02d}", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        cam_obj.location = (x, y, z)

        # Point toward centre
        direction = centre - cam_obj.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam_obj.rotation_euler = rot_quat.to_euler()
        cameras.append(cam_obj)

    print(f"[stablegen_cli] Fallback: placed {len(cameras)} orbit cameras")
    return cameras


def _configure_addon_prefs(server, output_dir):
    """Set StableGen addon preferences."""
    try:
        prefs = bpy.context.preferences.addons["stablegen"].preferences
        prefs.server_address = server
        prefs.output_dir = os.path.abspath(output_dir)
        os.makedirs(prefs.output_dir, exist_ok=True)
        print(f"[stablegen_cli] Output dir: {prefs.output_dir}")
        print(f"[stablegen_cli] Server: {prefs.server_address}")
    except KeyError:
        print("[stablegen_cli] WARNING: stablegen addon not found in preferences. "
              "Make sure it is installed and enabled.", file=sys.stderr)


def _apply_preset(preset_name):
    """Apply a StableGen preset by name."""
    try:
        bpy.context.scene.sg_preset = preset_name
        bpy.ops.object.stablegen_apply_preset()
        print(f"[stablegen_cli] Applied preset: {preset_name}")
    except Exception as e:
        print(f"[stablegen_cli] Could not apply preset '{preset_name}': {e}",
              file=sys.stderr)


def _configure_generation(args):
    """Set scene-level generation parameters."""
    scene = bpy.context.scene

    if args.prompt:
        scene.comfyui_prompt = args.prompt
    if args.negative:
        scene.comfyui_negative_prompt = args.negative
    if args.checkpoint:
        scene.sg_checkpoint = args.checkpoint
    if args.seed >= 0:
        scene.sg_seed = args.seed
        scene.sg_control_after_generate = "fixed"
    if args.steps is not None:
        scene.sg_steps = args.steps
    if args.cfg is not None:
        scene.sg_cfg = args.cfg

    # Architecture
    arch = _ARCH_MAP.get(args.architecture, "sdxl")
    try:
        scene.sg_model_architecture = arch
    except Exception:
        pass

    # ControlNet
    if args.no_controlnet:
        try:
            for unit in scene.sg_controlnet_units:
                unit.enabled = False
        except Exception:
            pass

    print(f"[stablegen_cli] Prompt     : {scene.comfyui_prompt!r}")
    print(f"[stablegen_cli] Architecture: {arch}")
    print(f"[stablegen_cli] Checkpoint : {args.checkpoint or '(from preset)'}")


def _run_generation():
    """Invoke the StableGen generation operator synchronously."""
    print("[stablegen_cli] Starting generation ...")
    try:
        # Override context so the operator has a valid window/area
        ctx = bpy.context.copy()
        result = bpy.ops.object.test_stable(ctx)
        print(f"[stablegen_cli] Generation result: {result}")
        return "FINISHED" in result
    except Exception as e:
        print(f"[stablegen_cli] Generation operator error: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return False


def _bake(objects):
    """Bake StableGen materials to UV textures."""
    print("[stablegen_cli] Baking textures ...")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    try:
        bpy.ops.object.stablegen_bake_textures()
        print("[stablegen_cli] Bake complete")
    except Exception as e:
        print(f"[stablegen_cli] Bake failed: {e}", file=sys.stderr)


def _export(objects, output_dir, fmt):
    """Export the textured mesh."""
    if fmt == "none":
        return
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"textured_mesh.{fmt}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)

    try:
        if fmt == "glb":
            bpy.ops.export_scene.gltf(filepath=out_path, export_selected=True,
                                      export_format="GLB")
        elif fmt == "fbx":
            bpy.ops.export_scene.fbx(filepath=out_path, use_selection=True)
        elif fmt == "obj":
            bpy.ops.wm.obj_export(filepath=out_path, export_selected_objects=True)
        print(f"[stablegen_cli] Exported: {out_path}")
    except Exception as e:
        print(f"[stablegen_cli] Export failed: {e}", file=sys.stderr)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("[stablegen_cli] ── StableGen headless CLI ──")
    print(f"[stablegen_cli] Args: {vars(args)}")

    # 1. Verify ComfyUI is up
    if not _check_server(args.server):
        sys.exit(1)

    # 2. Scene setup
    if args.blend:
        print(f"[stablegen_cli] Loading blend file: {args.blend}")
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
        mesh_objects = [o for o in bpy.data.objects if o.type == "MESH"]
    else:
        _setup_scene()
        if not args.mesh:
            print("[stablegen_cli] ERROR: provide --mesh or --blend", file=sys.stderr)
            sys.exit(1)
        mesh_objects = _import_mesh(os.path.abspath(args.mesh))

    # 3. Addon prefs
    _configure_addon_prefs(args.server, args.output)

    # 4. Enable the addon if not already active
    if "stablegen" not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module="stablegen")

    # 5. Preset + parameters
    _apply_preset(args.preset)
    _configure_generation(args)

    # 6. Place cameras
    _add_cameras(args.cameras, args.camera_mode, mesh_objects)

    # 7. Generate
    ok = _run_generation()
    if not ok:
        print("[stablegen_cli] Generation did not finish cleanly", file=sys.stderr)

    # 8. Bake (optional)
    if args.bake:
        _bake(mesh_objects)

    # 9. Export (optional)
    if args.export != "none":
        baked_dir = os.path.join(args.output, "baked")
        _export(mesh_objects, baked_dir, args.export)

    print("[stablegen_cli] Done.")


main()
