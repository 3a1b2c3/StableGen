"""
Pygame OBJ viewer — flat shading + texture + camera mode menu.
Usage: python camera_placement_pygame.py [model.obj]
Controls:
  Left-drag  — rotate
  Scroll     — zoom
  Tab        — cycle camera mode
"""

import math
import os
import sys
import numpy as np
import pygame

from camera_placement import (
    fibonacci_sphere_points, orbit_ring_directions,
    fan_arc_directions, kmeans_on_sphere, greedy_coverage_directions,
    visibility_weighted_directions,
)

W, H     = 900, 700
VIEW_CX  = W // 2
VIEW_CY  = H // 2
MODEL_R  = 260

C_BG      = (22,  22,  28)
C_MENU_BG = (32,  32,  42)
C_ACTIVE  = (79, 195, 247)
C_DIM     = (80,  80,  95)
C_TEXT    = (200, 200, 210)

LIGHT = np.array([0.4, 0.7, 0.6], dtype=float)
LIGHT /= np.linalg.norm(LIGHT)

CAM_COLORS = [
    (255,100,100),(100,255,150),(100,150,255),(255,220, 80),
    (255,130,200),( 80,220,220),(200,160,255),(255,180, 80),
]
CAM_DIST = 1.45
CAM_SIZE = 18

N_CAMS = 20


# ── Camera modes ───────────────────────────────────────────────────────────────

def _dirs_fibonacci(normals, areas):
    return np.array(fibonacci_sphere_points(N_CAMS))

def _dirs_orbit(normals, areas):
    return orbit_ring_directions(N_CAMS, elevation_deg=30.0)

def _dirs_fan(normals, areas):
    return fan_arc_directions(N_CAMS, fan_angle_deg=120.0, elevation_deg=15.0)

def _dirs_kmeans(normals, areas):
    return kmeans_on_sphere(normals, areas, k=N_CAMS)

def _dirs_greedy(normals, areas):
    sel, _ = greedy_coverage_directions(normals, areas, max_cameras=N_CAMS)
    return np.array(sel) if sel else np.zeros((1, 3))

def _dirs_visibility(normals, areas):
    return visibility_weighted_directions(normals, areas, k=N_CAMS)

MODES = [
    ("Fibonacci",   _dirs_fibonacci),
    ("Orbit Ring",  _dirs_orbit),
    ("Fan Arc",     _dirs_fan),
    ("K-means",     _dirs_kmeans),
    ("Greedy",      _dirs_greedy),
    ("Vis-Weighted",_dirs_visibility),
]


def fibonacci_sphere(n):
    pts = []
    phi = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        y  = 1 - (i / (n - 1)) * 2
        r  = math.sqrt(max(0, 1 - y * y))
        th = phi * i
        pts.append([r * math.cos(th), y, r * math.sin(th)])
    return np.array(pts, dtype=float)


# ── OBJ loader ─────────────────────────────────────────────────────────────────

def load_obj(path):
    verts, uvs_raw, faces = [], [], []

    with open(path, encoding="latin-1") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif p[0] == "vt":
                uvs_raw.append([float(p[1]), float(p[2])])
            elif p[0] == "f":
                corners = []
                for tok in p[1:]:
                    parts = tok.split("/")
                    vi = int(parts[0]) - 1
                    ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                    corners.append((vi, ti))
                for i in range(1, len(corners) - 1):
                    faces.append([corners[0], corners[i], corners[i + 1]])

    V   = np.array(verts,   dtype=float)
    UVs = np.array(uvs_raw, dtype=float) if uvs_raw else None

    vi = np.array([[f[0][0], f[1][0], f[2][0]] for f in faces], dtype=int)
    ti = np.array([[f[0][1], f[1][1], f[2][1]] for f in faces], dtype=int)

    V -= (V.max(axis=0) + V.min(axis=0)) * 0.5
    s  = (V.max(axis=0) - V.min(axis=0)).max()
    if s > 0:
        V = V / s * MODEL_R * 1.8

    v0, v1, v2 = V[vi[:, 0]], V[vi[:, 1]], V[vi[:, 2]]
    cross   = np.cross(v1 - v0, v2 - v0)
    norms   = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
    centres = (v0 + v1 + v2) / 3.0

    uv_face = None
    if UVs is not None and ti.min() >= 0:
        uv_face = (UVs[ti[:, 0]] + UVs[ti[:, 1]] + UVs[ti[:, 2]]) / 3.0

    return V, vi, norms, centres, uv_face


# ── Texture loader ─────────────────────────────────────────────────────────────

def load_texture(obj_path):
    obj_dir = os.path.dirname(os.path.abspath(obj_path))
    mtl_name = None
    with open(obj_path, encoding="latin-1") as fh:
        for line in fh:
            p = line.split()
            if p and p[0] == "mtllib":
                mtl_name = p[1]; break

    if mtl_name:
        mtl_path = os.path.join(obj_dir, mtl_name)
        if os.path.isfile(mtl_path):
            with open(mtl_path, encoding="latin-1") as fh:
                for line in fh:
                    p = line.split()
                    if p and p[0].lower() == "map_kd":
                        tp = os.path.join(obj_dir, p[1])
                        if os.path.isfile(tp):
                            return _load_image(tp)

    for fname in os.listdir(obj_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".bmp")):
            return _load_image(os.path.join(obj_dir, fname))
    return None


def _load_image(path):
    surf = pygame.image.load(path).convert()
    arr  = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
    return arr


# ── 3D helpers ─────────────────────────────────────────────────────────────────

def rot_matrix(az, el):
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=float)
    return Ry @ Rx


def project_pts(pts, R, zoom, fov=900):
    p  = pts @ R.T
    z  = np.maximum(p[:, 2] + fov, 1.0)
    sx = VIEW_CX + p[:, 0] * fov / z * zoom
    sy = VIEW_CY - p[:, 1] * fov / z * zoom
    return np.stack([sx, sy], axis=1), p[:, 2]


# ── Draw ───────────────────────────────────────────────────────────────────────

def draw_mesh(surf, V, F, norms, centres, uv_face, tex, R, zoom):
    tN      = norms @ R.T
    diffuse = np.clip(tN @ LIGHT, 0.0, 1.0)
    visible = tN[:, 2] < 0.0
    tC      = centres @ R.T
    order   = np.argsort(-tC[visible, 2])
    vis     = np.where(visible)[0][order]
    sc, _   = project_pts(V, R, zoom)
    sc_i    = sc.astype(int)
    tex_H, tex_W = (tex.shape[0], tex.shape[1]) if tex is not None else (0, 0)
    ambient = 0.25

    for fi in vis:
        pts2d = [sc_i[F[fi, k]] for k in range(3)]
        d     = diffuse[fi]
        if tex is not None and uv_face is not None:
            u, v = uv_face[fi]
            px = int((u % 1.0) * (tex_W - 1))
            py = int((1.0 - v % 1.0) * (tex_H - 1))
            base = tex[py, px].astype(float)
            lit  = np.clip(base * (ambient + (1.0 - ambient) * d), 0, 255).astype(int)
            col  = (int(lit[0]), int(lit[1]), int(lit[2]))
        else:
            shade = int(40 + 160 * (ambient + (1.0 - ambient) * d))
            col   = (int(shade * 0.70), int(shade * 0.80), shade)
        pygame.draw.polygon(surf, col, pts2d)


def draw_cameras(surf, dirs, R, zoom, font):
    radius = MODEL_R * CAM_DIST
    for i, d in enumerate(dirs):
        d   = d / (np.linalg.norm(d) + 1e-12)
        pos = d * radius
        up  = np.array([0, 1, 0], dtype=float)
        if abs(np.dot(d, up)) > 0.9:
            up = np.array([1, 0, 0], dtype=float)
        right = np.cross(d, up); right /= np.linalg.norm(right)
        up2   = np.cross(right, d)
        s     = CAM_SIZE
        corners = np.array([
            pos + s * (-right + up2),
            pos + s * ( right + up2),
            pos + s * ( right - up2),
            pos + s * (-right - up2),
        ])
        sc, _  = project_pts(corners, R, zoom)
        sc_i   = sc.astype(int)
        col    = CAM_COLORS[i % len(CAM_COLORS)]
        pygame.draw.polygon(surf, tuple(c // 3 for c in col), sc_i)
        pygame.draw.polygon(surf, col, sc_i, 2)
        cx = int(sc_i[:, 0].mean())
        cy = int(sc_i[:, 1].mean())
        lbl = font.render(str(i + 1), True, col)
        surf.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))


def draw_mode_menu(surf, mode_idx, rects, font):
    """Horizontal tab bar at the bottom of the screen."""
    for i, (name, _) in enumerate(MODES):
        r    = rects[i]
        active = (i == mode_idx)
        bg   = C_ACTIVE if active else C_MENU_BG
        tc   = (20, 20, 25) if active else C_DIM
        pygame.draw.rect(surf, bg, r, border_radius=3)
        lbl  = font.render(name, True, tc)
        surf.blit(lbl, lbl.get_rect(center=r.center))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sphere.obj"

    pygame.init()

    try:
        V, F, norms, centres, uv_face = load_obj(path)
    except Exception as e:
        print(f"Could not load {path}: {e}", file=sys.stderr)
        pygame.quit(); return

    # face normals for mesh-aware modes
    mesh_normals = norms
    mesh_areas   = np.ones(len(norms))   # uniform if no area data

    tex = None
    if uv_face is not None:
        try:
            tex = load_texture(path)
        except Exception as e:
            print(f"Texture load failed: {e}", file=sys.stderr)

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"OBJ viewer — {os.path.basename(path)}")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("segoeui", 12)
    font_b = pygame.font.SysFont("segoeui", 12, bold=True)

    # ── Mode menu layout ───────────────────────────────────────────────────────
    TAB_H  = 28
    TAB_W  = (W - 20) // len(MODES)
    menu_y = H - TAB_H - 6
    menu_rects = [
        pygame.Rect(10 + i * TAB_W, menu_y, TAB_W - 4, TAB_H)
        for i in range(len(MODES))
    ]

    mode_idx = 0
    cam_dirs = MODES[mode_idx][1](mesh_normals, mesh_areas)

    az, el, zoom = 0.5, 0.3, 1.0
    drag = False; drag_start = (0, 0); az0 = az; el0 = el

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_TAB:
                    mode_idx = (mode_idx + 1) % len(MODES)
                    cam_dirs = MODES[mode_idx][1](mesh_normals, mesh_areas)

            # menu clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(menu_rects):
                    if r.collidepoint(event.pos):
                        mode_idx = i
                        cam_dirs = MODES[mode_idx][1](mesh_normals, mesh_areas)
                        break
                else:
                    drag = True; drag_start = event.pos; az0, el0 = az, el
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drag = False
            if event.type == pygame.MOUSEMOTION and drag:
                dx = event.pos[0] - drag_start[0]
                dy = event.pos[1] - drag_start[1]
                az = az0 - dx * 0.005
                el = max(-math.pi / 2, min(math.pi / 2, el0 + dy * 0.005))
            if event.type == pygame.MOUSEWHEEL:
                zoom = max(0.3, min(3.5, zoom + event.y * 0.07))

        screen.fill(C_BG)
        R = rot_matrix(az, el)
        draw_mesh(screen, V, F, norms, centres, uv_face, tex, R, zoom)
        draw_cameras(screen, cam_dirs, R, zoom, font)
        draw_mode_menu(screen, mode_idx, menu_rects, font_b)

        label = f"{os.path.basename(path)}  |  {len(F)} tris  |  {len(cam_dirs)} cams"
        screen.blit(font.render(label, True, (90, 90, 100)), (10, 10))
        screen.blit(font.render("Tab to cycle  •  click tab to select", True, C_DIM),
                    (10, H - TAB_H - 22))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
