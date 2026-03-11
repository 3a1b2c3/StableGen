"""
Pygame OBJ viewer — flat shading.
Usage: python camera_placement_pygame.py [model.obj]
Controls:
  Left-drag  — rotate
  Scroll     — zoom
"""

import math
import sys
import numpy as np
import pygame

W, H     = 900, 700
VIEW_CX  = W // 2
VIEW_CY  = H // 2
MODEL_R  = 260          # display radius

C_BG     = (22, 22, 28)
LIGHT    = np.array([0.4, 0.7, 0.6], dtype=float)
LIGHT   /= np.linalg.norm(LIGHT)


# ── OBJ loader ─────────────────────────────────────────────────────────────────

def load_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif p[0] == "f":
                idx = [int(x.split("/")[0]) - 1 for x in p[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    V = np.array(verts, dtype=float)
    F = np.array(faces,  dtype=int)

    # normalise to MODEL_R
    V -= (V.max(axis=0) + V.min(axis=0)) * 0.5
    s = (V.max(axis=0) - V.min(axis=0)).max()
    if s > 0:
        V = V / s * MODEL_R * 1.8

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross  = np.cross(v1 - v0, v2 - v0)
    norms  = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
    centres = (v0 + v1 + v2) / 3.0
    return V, F, norms, centres


# ── 3D helpers ─────────────────────────────────────────────────────────────────

def rot_matrix(az, el):
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=float)
    return Ry @ Rx


def project_pts(pts, R, zoom, fov=900):
    p = pts @ R.T
    z = np.maximum(p[:, 2] + fov, 1.0)
    sx = VIEW_CX + p[:, 0] * fov / z * zoom
    sy = VIEW_CY - p[:, 1] * fov / z * zoom
    return np.stack([sx, sy], axis=1), p[:, 2]


# ── Draw ───────────────────────────────────────────────────────────────────────

def draw_mesh(surf, V, F, norms, centres, R, zoom):
    # face normals → diffuse
    tN      = norms @ R.T
    diffuse = np.clip(tN @ LIGHT, 0.0, 1.0)

    # back-face cull: drop faces whose transformed normal faces away
    visible = tN[:, 2] < 0.0

    # depth-sort visible faces (back to front)
    tC      = centres @ R.T
    order   = np.argsort(-tC[visible, 2])   # descending depth

    vis_idx = np.where(visible)[0][order]

    # project all vertices once
    sc, _   = project_pts(V, R, zoom)
    sc_int  = sc.astype(int)

    for fi in vis_idx:
        pts2d = [sc_int[F[fi, k]] for k in range(3)]
        d     = diffuse[fi]
        col   = (int(40 + 160 * d * 0.70),
                 int(40 + 160 * d * 0.80),
                 int(40 + 160 * d))
        pygame.draw.polygon(surf, col, pts2d)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sphere.obj"

    try:
        V, F, norms, centres = load_obj(path)
        title = path
    except Exception as e:
        print(f"Could not load {path}: {e}", file=sys.stderr)
        return

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"OBJ viewer — {title}")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("segoeui", 12)

    az, el, zoom = 0.5, 0.3, 1.0
    drag = False
    drag_start = (0, 0)
    az0 = az; el0 = el

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
        draw_mesh(screen, V, F, norms, centres, R, zoom)
        screen.blit(font.render(f"{title}  |  {len(F)} tris", True, (90, 90, 100)), (10, 10))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
