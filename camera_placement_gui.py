"""
Interactive matplotlib GUI for camera placement algorithms.
Run: python camera_placement_gui.py
"""

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons, Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from camera_placement import (
    fibonacci_sphere_points,
    kmeans_on_sphere,
    compute_pca_axes,
    greedy_coverage_directions,
)

# ── Mesh presets ───────────────────────────────────────────────────────────────

def _make_sphere_mesh(n=300, seed=0):
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = rng.uniform(0.1, 1.0, n)
    verts = normals * rng.uniform(0.8, 1.2, (n, 1))
    return normals, areas, verts

def _make_elongated_mesh(n=300, seed=0):
    """Stretched along X — good for showing PCA preference."""
    rng = np.random.default_rng(seed)
    verts = rng.standard_normal((n, 3)) * np.array([3.0, 1.0, 0.5])
    normals = rng.standard_normal((n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # Bias normals toward the long axis
    normals = normals * np.array([2.0, 1.0, 0.5])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = rng.uniform(0.1, 1.0, n)
    return normals, areas, verts

def _make_top_heavy_mesh(n=300, seed=0):
    """Most faces point upward — greedy/kmeans should prefer top views."""
    rng = np.random.default_rng(seed)
    normals_top = rng.standard_normal((int(n * 0.7), 3))
    normals_top[:, 2] = np.abs(normals_top[:, 2]) + 1.0  # bias upward
    normals_rest = rng.standard_normal((n - int(n * 0.7), 3))
    normals = np.vstack([normals_top, normals_rest])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = rng.uniform(0.1, 1.0, n)
    verts = normals * rng.uniform(0.5, 1.5, (n, 1))
    return normals, areas, verts

MESH_PRESETS = {
    "Sphere (uniform)": _make_sphere_mesh,
    "Elongated (X axis)": _make_elongated_mesh,
    "Top-heavy": _make_top_heavy_mesh,
}

# ── OBJ loader ─────────────────────────────────────────────────────────────────

def load_obj(path):
    """Parse an OBJ file and return (normals, areas, verts).
    Supports triangles and quads (quads are split into two triangles).
    No external dependencies — pure Python + numpy."""
    verts = []
    faces = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # OBJ indices are 1-based; strip texture/normal suffixes (v/vt/vn)
                indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                # Fan-triangulate polygons
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])

    if not verts or not faces:
        raise ValueError(f"No geometry found in {path}")

    V = np.array(verts, dtype=float)
    F = np.array(faces, dtype=int)

    # Clamp any out-of-range indices (malformed OBJ safety)
    F = np.clip(F, 0, len(V) - 1)

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    area_vec = np.linalg.norm(cross, axis=1)
    areas = area_vec * 0.5

    # Avoid zero-area degenerate faces
    valid = areas > 1e-12
    cross = cross[valid]
    areas = areas[valid]
    area_vec = area_vec[valid]

    normals = cross / area_vec[:, np.newaxis]

    # Normalize the mesh to unit bounding box for display
    V -= V.min(axis=0)
    scale = V.max()
    if scale > 0:
        V /= scale
    V -= V.mean(axis=0)

    return normals, areas, V

# ── Strategy runners ───────────────────────────────────────────────────────────

def run_hemisphere(normals, areas, verts, n_cams, coverage):
    dirs = np.array(fibonacci_sphere_points(n_cams))
    return dirs, f"Hemisphere — {len(dirs)} directions"

def run_kmeans(normals, areas, verts, n_cams, coverage):
    dirs = kmeans_on_sphere(normals, areas, k=n_cams)
    return dirs, f"K-means — {len(dirs)} cameras"

def run_pca(normals, areas, verts, n_cams, coverage):
    axes = compute_pca_axes(verts)          # 3 axes
    # Use ±axes to get 6 views, trim to n_cams
    dirs = np.vstack([axes, -axes])[:n_cams]
    return dirs, f"PCA — {len(dirs)} axes (of max 6)"

def run_greedy(normals, areas, verts, n_cams, coverage):
    selected, frac = greedy_coverage_directions(
        normals, areas, max_cameras=n_cams, coverage_target=coverage
    )
    if not selected:
        return np.zeros((1, 3)), "Greedy — no cameras placed"
    dirs = np.array(selected)
    return dirs, f"Greedy — {len(dirs)} cameras, {frac:.1%} covered"

STRATEGIES = {
    "Hemisphere": run_hemisphere,
    "K-means": run_kmeans,
    "PCA Axes": run_pca,
    "Greedy Coverage": run_greedy,
}

# ── GUI state ──────────────────────────────────────────────────────────────────

state = {
    "strategy": "K-means",
    "mesh": "Sphere (uniform)",
    "n_cams": 6,
    "coverage": 0.90,
}

# ── Build figure ───────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 8), facecolor="#1e1e1e")
fig.canvas.manager.set_window_title("Camera Placement Visualizer")

gs = gridspec.GridSpec(
    1, 2,
    width_ratios=[3, 1],
    left=0.03, right=0.97,
    top=0.93, bottom=0.05,
    wspace=0.05,
)

ax3d = fig.add_subplot(gs[0], projection="3d", facecolor="#1e1e1e")
ax3d.set_title("", color="white", pad=6)

panel = fig.add_subplot(gs[1], facecolor="#2a2a2a")
panel.set_axis_off()

# ── Control layout inside right panel ─────────────────────────────────────────
# We'll place widgets using figure-fraction coordinates.
# Right panel spans roughly x: 0.74–0.97, y: 0.05–0.93

PX0, PX1 = 0.745, 0.965
PY_TOP = 0.88

def _rect(y, h=0.04, x0=PX0, x1=PX1):
    return [x0, y - h, x1 - x0, h]

# Strategy selector
fig.text(PX0, PY_TOP + 0.01, "Strategy", color="#aaaaaa", fontsize=9, va="bottom")
ax_strat = fig.add_axes([PX0, PY_TOP - 0.18, PX1 - PX0, 0.18], facecolor="#333333")
radio_strat = RadioButtons(
    ax_strat, list(STRATEGIES.keys()),
    active=list(STRATEGIES.keys()).index(state["strategy"]),
    activecolor="#4fc3f7",
)
for lbl in radio_strat.labels:
    lbl.set_color("white")
    lbl.set_fontsize(9)

# Mesh preset selector
MY = PY_TOP - 0.28
fig.text(PX0, MY + 0.01, "Mesh preset", color="#aaaaaa", fontsize=9, va="bottom")
ax_mesh = fig.add_axes([PX0, MY - 0.12, PX1 - PX0, 0.12], facecolor="#333333")
radio_mesh = RadioButtons(
    ax_mesh, list(MESH_PRESETS.keys()),
    active=list(MESH_PRESETS.keys()).index(state["mesh"]),
    activecolor="#4fc3f7",
)
for lbl in radio_mesh.labels:
    lbl.set_color("white")
    lbl.set_fontsize(9)

# Num cameras slider
SY = MY - 0.20
fig.text(PX0, SY + 0.005, "Cameras / K", color="#aaaaaa", fontsize=9)
ax_ncams = fig.add_axes([PX0, SY - 0.045, PX1 - PX0, 0.025], facecolor="#333333")
sl_ncams = Slider(ax_ncams, "", 1, 12, valinit=state["n_cams"], valstep=1,
                  color="#4fc3f7", track_color="#444444")
sl_ncams.valtext.set_color("white")

# Coverage slider
CY = SY - 0.12
fig.text(PX0, CY + 0.005, "Coverage target (Greedy)", color="#aaaaaa", fontsize=9)
ax_cov = fig.add_axes([PX0, CY - 0.045, PX1 - PX0, 0.025], facecolor="#333333")
sl_cov = Slider(ax_cov, "", 0.5, 1.0, valinit=state["coverage"], valstep=0.05,
                color="#4fc3f7", track_color="#444444")
sl_cov.valtext.set_color("white")

# Regenerate button
BY = CY - 0.12
ax_btn = fig.add_axes([PX0, BY - 0.05, PX1 - PX0, 0.05], facecolor="#333333")
btn_regen = Button(ax_btn, "Regenerate mesh", color="#333333", hovercolor="#555555")
btn_regen.label.set_color("white")
btn_regen.label.set_fontsize(9)

# Load OBJ button
OY = BY - 0.08
ax_obj = fig.add_axes([PX0, OY - 0.05, PX1 - PX0, 0.05], facecolor="#333333")
btn_obj = Button(ax_obj, "Load OBJ...", color="#2a3a2a", hovercolor="#3a5a3a")
btn_obj.label.set_color("#88dd88")
btn_obj.label.set_fontsize(9)

# Info text
ax_info = fig.add_axes([PX0, 0.05, PX1 - PX0, OY - 0.10], facecolor="#1e1e1e")
ax_info.set_axis_off()
info_text = ax_info.text(0.05, 0.95, "", color="#cccccc", fontsize=8,
                         va="top", ha="left", transform=ax_info.transAxes,
                         wrap=True, family="monospace")

# ── Draw function ──────────────────────────────────────────────────────────────

_mesh_cache = {}

def get_mesh():
    key = state["mesh"]
    if key not in _mesh_cache:
        if key in MESH_PRESETS:
            _mesh_cache[key] = MESH_PRESETS[key]()
        else:
            raise KeyError(f"Mesh '{key}' not in cache")
    return _mesh_cache[key]

def redraw():
    normals, areas, verts = get_mesh()
    fn = STRATEGIES[state["strategy"]]
    dirs, label = fn(normals, areas, verts, state["n_cams"], state["coverage"])

    ax3d.cla()
    ax3d.set_facecolor("#1e1e1e")
    ax3d.tick_params(colors="#555555", labelsize=7)
    for spine in ax3d.spines.values():
        spine.set_color("#444444")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#333333")
    ax3d.yaxis.pane.set_edgecolor("#333333")
    ax3d.zaxis.pane.set_edgecolor("#333333")
    ax3d.set_xlabel("X", color="#666666", fontsize=8)
    ax3d.set_ylabel("Y", color="#666666", fontsize=8)
    ax3d.set_zlabel("Z", color="#666666", fontsize=8)

    # Wireframe reference sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_wireframe(sx, sy, sz, color="#333333", linewidth=0.4, alpha=0.5)

    # Face normals (sample 120 for clarity)
    idx = np.linspace(0, len(normals) - 1, min(120, len(normals)), dtype=int)
    nx, ny, nz = normals[idx, 0], normals[idx, 1], normals[idx, 2]
    c = (areas[idx] - areas.min()) / (areas.max() - areas.min() + 1e-9)
    ax3d.scatter(nx, ny, nz, c=c, cmap="Blues", s=8, alpha=0.4, zorder=1)

    # Camera directions as arrows + markers
    RADIUS = 1.45
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(dirs)))
    for i, (d, col) in enumerate(zip(dirs, colors)):
        tip = np.asarray(d)
        tip /= np.linalg.norm(tip) + 1e-12
        tip *= RADIUS
        ax3d.quiver(0, 0, 0, tip[0], tip[1], tip[2],
                    color=col, linewidth=2, arrow_length_ratio=0.15)
        ax3d.scatter(*tip, color=col, s=60, zorder=5,
                     edgecolors="white", linewidths=0.5)
        ax3d.text(tip[0] * 1.08, tip[1] * 1.08, tip[2] * 1.08,
                  str(i + 1), color=col, fontsize=7, ha="center", va="center")

    ax3d.set_xlim(-1.8, 1.8)
    ax3d.set_ylim(-1.8, 1.8)
    ax3d.set_zlim(-1.8, 1.8)
    ax3d.set_title(
        f"{state['strategy']}  ·  {state['mesh']}",
        color="white", fontsize=10, pad=8,
    )

    # Info panel
    lines = [label, ""]
    lines.append(f"Cameras placed : {len(dirs)}")
    lines.append(f"Normals sampled: {len(normals)}")
    lines.append("")
    lines.append("Directions (xyz):")
    for i, d in enumerate(dirs):
        dn = np.asarray(d) / (np.linalg.norm(d) + 1e-12)
        lines.append(f"  {i+1:2d}  {dn[0]:+.3f}  {dn[1]:+.3f}  {dn[2]:+.3f}")
    info_text.set_text("\n".join(lines))

    fig.canvas.draw_idle()

# ── Callbacks ──────────────────────────────────────────────────────────────────

def on_strategy(label):
    state["strategy"] = label
    redraw()

def on_mesh(label):
    state["mesh"] = label
    redraw()

def on_ncams(val):
    state["n_cams"] = int(val)
    redraw()

def on_cov(val):
    state["coverage"] = float(val)
    redraw()

def on_regen(event):
    key = state["mesh"]
    seed = np.random.randint(0, 9999)
    _mesh_cache[key] = MESH_PRESETS[key](seed=seed)
    redraw()

def on_load_obj(event):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select OBJ file",
        filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        return
    try:
        normals, areas, verts = load_obj(path)
    except Exception as exc:
        info_text.set_text(f"OBJ load error:\n{exc}")
        fig.canvas.draw_idle()
        return
    name = os.path.basename(path)
    _mesh_cache[name] = (normals, areas, verts)
    state["mesh"] = name
    # Rebuild mesh radio to include the new entry
    labels = list(MESH_PRESETS.keys()) + [
        k for k in _mesh_cache if k not in MESH_PRESETS
    ]
    # Update radio button labels in-place
    for i, lbl in enumerate(radio_mesh.labels):
        lbl.set_text(labels[i] if i < len(labels) else "")
    # Add new label if needed
    redraw()

radio_strat.on_clicked(on_strategy)
radio_mesh.on_clicked(on_mesh)
sl_ncams.on_changed(on_ncams)
sl_cov.on_changed(on_cov)
btn_regen.on_clicked(on_regen)
btn_obj.on_clicked(on_load_obj)

# ── Initial draw ───────────────────────────────────────────────────────────────

# Load OBJ from command line if provided: python camera_placement_gui.py model.obj
if len(sys.argv) > 1:
    _path = sys.argv[1]
    if os.path.isfile(_path):
        try:
            _n, _a, _v = load_obj(_path)
            _name = os.path.basename(_path)
            _mesh_cache[_name] = (_n, _a, _v)
            state["mesh"] = _name
        except Exception as _e:
            print(f"Could not load {_path}: {_e}", file=sys.stderr)
    else:
        print(f"File not found: {_path}", file=sys.stderr)

redraw()
plt.show()
