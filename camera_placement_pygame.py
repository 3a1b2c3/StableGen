"""
Pygame OBJ viewer — flat shading + texture.
Usage: python camera_placement_pygame.py [model.obj]
Controls:
  Left-drag  — rotate
  Scroll     — zoom
"""

import math
import os
import sys
import numpy as np
import pygame

W, H     = 900, 700
VIEW_CX  = W // 2
VIEW_CY  = H // 2
MODEL_R  = 260

C_BG    = (22, 22, 28)
LIGHT   = np.array([0.4, 0.7, 0.6], dtype=float)
LIGHT  /= np.linalg.norm(LIGHT)

CAM_COLORS = [
    (255,100,100),(100,255,150),(100,150,255),(255,220, 80),
    (255,130,200),( 80,220,220),(200,160,255),(255,180, 80),
]
CAM_DIST   = 1.45   # multiple of MODEL_R
CAM_SIZE   = 18     # half-width of square in world units


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
                    vi  = int(parts[0]) - 1
                    ti  = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                    corners.append((vi, ti))
                for i in range(1, len(corners) - 1):
                    faces.append([corners[0], corners[i], corners[i + 1]])

    V   = np.array(verts,   dtype=float)
    UVs = np.array(uvs_raw, dtype=float) if uvs_raw else None

    # Build per-face arrays
    vi  = np.array([[f[0][0], f[1][0], f[2][0]] for f in faces], dtype=int)
    ti  = np.array([[f[0][1], f[1][1], f[2][1]] for f in faces], dtype=int)

    # normalise verts to MODEL_R
    V -= (V.max(axis=0) + V.min(axis=0)) * 0.5
    s  = (V.max(axis=0) - V.min(axis=0)).max()
    if s > 0:
        V = V / s * MODEL_R * 1.8

    v0, v1, v2 = V[vi[:, 0]], V[vi[:, 1]], V[vi[:, 2]]
    cross    = np.cross(v1 - v0, v2 - v0)
    norms    = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
    centres  = (v0 + v1 + v2) / 3.0

    # per-face centroid UV (or None)
    uv_face = None
    if UVs is not None and ti.min() >= 0:
        uv_face = (UVs[ti[:, 0]] + UVs[ti[:, 1]] + UVs[ti[:, 2]]) / 3.0

    return V, vi, norms, centres, uv_face


# ── Texture loader ─────────────────────────────────────────────────────────────

def load_texture(obj_path):
    obj_dir = os.path.dirname(os.path.abspath(obj_path))

    # 1. Try MTL map_Kd
    mtl_name = None
    with open(obj_path, encoding="latin-1") as fh:
        for line in fh:
            p = line.split()
            if p and p[0] == "mtllib":
                mtl_name = p[1]
                break

    if mtl_name:
        mtl_path = os.path.join(obj_dir, mtl_name)
        if os.path.isfile(mtl_path):
            with open(mtl_path, encoding="latin-1") as fh:
                for line in fh:
                    p = line.split()
                    if p and p[0].lower() == "map_kd":
                        tex_path = os.path.join(obj_dir, p[1])
                        if os.path.isfile(tex_path):
                            return _load_image(tex_path)

    # 2. Fall back: first image in OBJ directory
    for fname in os.listdir(obj_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".bmp")):
            return _load_image(os.path.join(obj_dir, fname))

    return None


def _load_image(path):
    surf = pygame.image.load(path).convert()
    arr  = pygame.surfarray.array3d(surf)   # shape (W, H, 3) — pygame is col-major
    arr  = arr.transpose(1, 0, 2)           # → (H, W, 3)
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

    tC    = centres @ R.T
    order = np.argsort(-tC[visible, 2])
    vis   = np.where(visible)[0][order]

    sc, _ = project_pts(V, R, zoom)
    sc_i  = sc.astype(int)

    tex_H, tex_W = (tex.shape[0], tex.shape[1]) if tex is not None else (0, 0)

    for fi in vis:
        pts2d = [sc_i[F[fi, k]] for k in range(3)]
        d     = diffuse[fi]
        ambient = 0.25

        if tex is not None and uv_face is not None:
            u, v = uv_face[fi]
            u = u % 1.0
            v = 1.0 - (v % 1.0)   # flip V (OBJ origin bottom-left)
            px = int(u * (tex_W - 1))
            py = int(v * (tex_H - 1))
            base = tex[py, px].astype(float)
            lit  = np.clip(base * (ambient + (1.0 - ambient) * d), 0, 255).astype(int)
            col  = (int(lit[0]), int(lit[1]), int(lit[2]))
        else:
            shade = int(40 + 160 * (ambient + (1.0 - ambient) * d))
            col   = (int(shade * 0.70), int(shade * 0.80), shade)

        pygame.draw.polygon(surf, col, pts2d)


def draw_cameras(surf, dirs, R, zoom, font):
    """Draw each camera as a small 3D square (film-plane facing origin)."""
    radius = MODEL_R * CAM_DIST

    for i, d in enumerate(dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        pos = d * radius

        # Two axes perpendicular to d for the square
        up  = np.array([0, 1, 0], dtype=float)
        if abs(np.dot(d, up)) > 0.9:
            up = np.array([1, 0, 0], dtype=float)
        right = np.cross(d, up); right /= np.linalg.norm(right)
        up2   = np.cross(right, d)

        s = CAM_SIZE
        corners = np.array([
            pos + s * (-right + up2),
            pos + s * ( right + up2),
            pos + s * ( right - up2),
            pos + s * (-right - up2),
        ])

        sc, depths = project_pts(corners, R, zoom)
        sc_i = sc.astype(int)

        # depth of camera centre for back-face / brightness
        cen_t  = (R @ pos)[2]
        col    = CAM_COLORS[i % len(CAM_COLORS)]

        # filled square
        pygame.draw.polygon(surf, tuple(c // 3 for c in col), sc_i)
        # border
        pygame.draw.polygon(surf, col, sc_i, 2)

        # label
        cx = int(sc_i[:, 0].mean())
        cy = int(sc_i[:, 1].mean())
        lbl = font.render(str(i + 1), True, col)
        surf.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sphere.obj"

    pygame.init()

    try:
        V, F, norms, centres, uv_face = load_obj(path)
    except Exception as e:
        print(f"Could not load {path}: {e}", file=sys.stderr)
        pygame.quit(); return

    n_cams = 6
    cam_dirs = fibonacci_sphere(n_cams)

    tex = None
    if uv_face is not None:
        try:
            tex = load_texture(path)
            if tex is not None:
                print(f"Texture loaded: {tex.shape[1]}×{tex.shape[0]}")
            else:
                print("No texture found — flat shading")
        except Exception as e:
            print(f"Texture load failed: {e}", file=sys.stderr)

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"OBJ viewer — {os.path.basename(path)}")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("segoeui", 12)

    az, el, zoom = 0.5, 0.3, 1.0
    drag = False; drag_start = (0, 0); az0 = az; el0 = el

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
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
        label = f"{os.path.basename(path)}  |  {len(F)} tris"
        if tex is not None:
            label += f"  |  tex {tex.shape[1]}×{tex.shape[0]}"
        screen.blit(font.render(label, True, (90, 90, 100)), (10, 10))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
