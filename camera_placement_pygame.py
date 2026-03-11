"""
Pygame interactive GUI for camera placement algorithms.
Run: python camera_placement_pygame.py [model.obj]

Controls:
  Left-drag  — rotate 3D view
  Scroll     — zoom
"""

import math
import os
import sys
import numpy as np
import pygame

from camera_placement import (
    fibonacci_sphere_points, kmeans_on_sphere, compute_pca_axes,
    greedy_coverage_directions, orbit_ring_directions,
    fan_arc_directions, visibility_weighted_directions,
)

# ── Layout ─────────────────────────────────────────────────────────────────────
W, H      = 1280, 760
PANEL_X   = 880
VIEW_CX   = PANEL_X // 2
VIEW_CY   = H // 2
SPHERE_R  = 240

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG       = (22,  22,  28)
C_PANEL    = (32,  32,  40)
C_SEP      = (55,  55,  65)
C_TEXT     = (200, 200, 210)
C_DIM      = ( 95,  95, 110)
C_ACTIVE   = ( 79, 195, 247)
C_BTN      = ( 48,  48,  60)
C_BTN_HOV  = ( 65,  65,  80)
C_GREEN    = (100, 200, 120)
C_SPHERE   = ( 48,  52,  62)

CAM_COLORS = [
    (255,100,100),(100,255,150),(100,150,255),(255,220, 80),
    (255,130,200),( 80,220,220),(200,160,255),(255,180, 80),
    (150,255,100),( 80,180,255),(255, 80,150),(180,255,180),
]

# ── OBJ loader ─────────────────────────────────────────────────────────────────

def load_obj(path):
    verts, faces = [], []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i+1]])
    if not verts or not faces:
        raise ValueError(f"No geometry in {path}")
    V = np.array(verts, dtype=float)
    F = np.clip(np.array(faces, dtype=int), 0, len(V)-1)
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    cross = np.cross(v1-v0, v2-v0)
    av = np.linalg.norm(cross, axis=1)
    areas = av * 0.5
    valid = areas > 1e-12
    cross, areas, av = cross[valid], areas[valid], av[valid]
    normals = cross / av[:, np.newaxis]
    return normals, areas, V

# ── Synthetic mesh presets ─────────────────────────────────────────────────────

def _sphere_mesh(seed=0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((300, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n, rng.uniform(0.1, 1.0, 300), n * rng.uniform(0.8, 1.2, (300,1))

def _elongated_mesh(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((300, 3)) * [3., 1., .5]
    n = rng.standard_normal((300, 3)) * [2., 1., .5]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n, rng.uniform(0.1, 1., 300), v

def _top_heavy_mesh(seed=0):
    rng = np.random.default_rng(seed)
    nt = rng.standard_normal((210, 3)); nt[:,2] = np.abs(nt[:,2]) + 1.
    nr = rng.standard_normal((90, 3))
    n = np.vstack([nt, nr]); n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n, rng.uniform(0.1, 1., 300), n * rng.uniform(.5, 1.5, (300,1))

MESH_PRESETS = {"Sphere": _sphere_mesh, "Elongated": _elongated_mesh, "Top-heavy": _top_heavy_mesh}

# ── Strategy runners ───────────────────────────────────────────────────────────

def run_orbit(normals, areas, verts, st):
    return orbit_ring_directions(st["n_cams"], elevation_deg=st["elevation"]), \
           f"Orbit Ring  el={st['elevation']:.0f}°"

def run_fan(normals, areas, verts, st):
    return fan_arc_directions(st["n_cams"], fan_angle_deg=st["fan_angle"],
                              elevation_deg=st["elevation"]), \
           f"Fan Arc  fan={st['fan_angle']:.0f}°"

def run_hemisphere(normals, areas, verts, st):
    return np.array(fibonacci_sphere_points(st["n_cams"])), "Hemisphere"

def run_pca(normals, areas, verts, st):
    axes = compute_pca_axes(verts)
    dirs = np.vstack([axes, -axes])[:st["n_cams"]]
    return dirs, f"PCA Axes ({len(dirs)})"

def run_kmeans(normals, areas, verts, st):
    return kmeans_on_sphere(normals, areas, k=st["n_cams"]), "K-means"

def run_greedy(normals, areas, verts, st):
    sel, frac = greedy_coverage_directions(normals, areas,
                    max_cameras=st["n_cams"], coverage_target=st["coverage"])
    dirs = np.array(sel) if sel else np.zeros((1,3))
    return dirs, f"Greedy  {frac:.0%} covered"

def run_visibility(normals, areas, verts, st):
    dirs = visibility_weighted_directions(normals, areas, k=st["n_cams"], balance=st["balance"])
    return dirs, f"Vis-Weighted  bal={st['balance']:.1f}"

STRATEGIES = [
    ("1  Orbit Ring",    run_orbit),
    ("2  Fan Arc",       run_fan),
    ("3  Hemisphere",    run_hemisphere),
    ("4  PCA Axes",      run_pca),
    ("5  K-means",       run_kmeans),
    ("6  Greedy",        run_greedy),
    ("7  Vis-Weighted",  run_visibility),
]

# ── 3D projection ──────────────────────────────────────────────────────────────

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

def project1(pt, R, zoom, fov=900):
    p = R @ np.asarray(pt, dtype=float)
    z = max(p[2] + fov, 1.0)
    sx = VIEW_CX + p[0] * fov / z * zoom
    sy = VIEW_CY - p[1] * fov / z * zoom
    return (int(sx), int(sy)), p[2]

# ── 3D draw routines ───────────────────────────────────────────────────────────

def draw_wireframe_sphere(surf, R, zoom, radius, color):
    N_LAT, N_LON = 9, 12
    for i in range(1, N_LAT):
        el = math.pi * i / N_LAT - math.pi / 2
        pts = np.array([
            [math.cos(el)*math.cos(2*math.pi*j/N_LON),
             math.sin(el),
             math.cos(el)*math.sin(2*math.pi*j/N_LON)]
            for j in range(N_LON + 1)
        ]) * radius
        sc, _ = project_pts(pts, R, zoom)
        for j in range(N_LON):
            pygame.draw.line(surf, color, sc[j].astype(int), sc[j+1].astype(int), 1)
    for j in range(N_LON):
        az = 2 * math.pi * j / N_LON
        pts = np.array([
            [math.cos(math.pi*i/N_LAT - math.pi/2)*math.cos(az),
             math.sin(math.pi*i/N_LAT - math.pi/2),
             math.cos(math.pi*i/N_LAT - math.pi/2)*math.sin(az)]
            for i in range(N_LAT + 1)
        ]) * radius
        sc, _ = project_pts(pts, R, zoom)
        for i in range(N_LAT):
            pygame.draw.line(surf, color, sc[i].astype(int), sc[i+1].astype(int), 1)


def draw_normals_dots(surf, normals, areas, R, zoom, radius, n_sample=180):
    idx = np.linspace(0, len(normals)-1, min(n_sample, len(normals)), dtype=int)
    pts = normals[idx] * radius * 0.90
    a_n = (areas[idx] - areas.min()) / (areas.max() - areas.min() + 1e-9)
    sc, depths = project_pts(pts, R, zoom)
    for i in np.argsort(depths):
        b = int(35 + 75 * a_n[i])
        pygame.draw.circle(surf, (b, b+18, b+48), sc[i].astype(int), 2)


def draw_camera_arrows(surf, dirs, R, zoom, radius, font_sm):
    origin_s, _ = project1([0, 0, 0], R, zoom)
    norms = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    tips  = norms * radius * 1.38
    sc, depths = project_pts(tips, R, zoom)
    for i in np.argsort(depths):           # back-to-front
        col = CAM_COLORS[i % len(CAM_COLORS)]
        tip_s = sc[i].astype(int)
        pygame.draw.line(surf, col, origin_s, tip_s, 2)
        pygame.draw.circle(surf, col, tip_s, 6)
        pygame.draw.circle(surf, (230, 240, 255), tip_s, 6, 1)
        # Label offset outward from origin
        off = np.array(tip_s) - np.array(origin_s)
        n = np.linalg.norm(off) + 1e-6
        lx = int(tip_s[0] + off[0]/n * 15)
        ly = int(tip_s[1] + off[1]/n * 15)
        lbl = font_sm.render(str(i+1), True, col)
        surf.blit(lbl, (lx - lbl.get_width()//2, ly - lbl.get_height()//2))

# ── UI widgets ─────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, rect, text, active=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.active = active
        self._hov = False

    def draw(self, surf, font):
        if self.active:
            bg, tc = C_ACTIVE, (20, 20, 25)
        elif self._hov:
            bg, tc = C_BTN_HOV, C_TEXT
        else:
            bg, tc = C_BTN, C_DIM
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        lbl = font.render(self.text, True, tc)
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hov = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Slider:
    def __init__(self, rect, label, vmin, vmax, val, step=None):
        self.rect  = pygame.Rect(rect)
        self.label = label
        self.vmin, self.vmax = vmin, vmax
        self.val   = val
        self.step  = step
        self._drag = False

    @property
    def _frac(self):
        return (self.val - self.vmin) / (self.vmax - self.vmin)

    def draw(self, surf, font):
        lbl = font.render(f"{self.label}:  {self.val:.3g}", True, C_TEXT)
        surf.blit(lbl, (self.rect.x, self.rect.y - 15))
        track = pygame.Rect(self.rect.x, self.rect.centery - 2, self.rect.width, 4)
        pygame.draw.rect(surf, C_SEP, track, border_radius=2)
        fw = int(self.rect.width * self._frac)
        if fw > 0:
            pygame.draw.rect(surf, C_ACTIVE,
                             pygame.Rect(self.rect.x, self.rect.centery-2, fw, 4),
                             border_radius=2)
        hx = self.rect.x + fw
        pygame.draw.circle(surf, C_ACTIVE, (hx, self.rect.centery), 7)
        pygame.draw.circle(surf, (200, 230, 255), (hx, self.rect.centery), 7, 1)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._drag = True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag = False
        if event.type == pygame.MOUSEMOTION and self._drag:
            frac = max(0., min(1., (event.pos[0] - self.rect.x) / self.rect.width))
            v = self.vmin + frac * (self.vmax - self.vmin)
            if self.step:
                v = round(v / self.step) * self.step
            if v != self.val:
                self.val = v
                return True
        return False

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Camera Placement Visualizer")
    clock = pygame.time.Clock()

    font      = pygame.font.SysFont("segoeui", 13)
    font_sm   = pygame.font.SysFont("segoeui", 11)
    font_bold = pygame.font.SysFont("segoeui", 14, bold=True)

    # ── App state ──────────────────────────────────────────────────────────────
    state = {"strategy": 4, "mesh": "Sphere", "n_cams": 6,
             "coverage": 0.90, "elevation": 30.0, "fan_angle": 90.0, "balance": 0.5}
    mesh_cache = {}

    def get_mesh():
        key = state["mesh"]
        if key not in mesh_cache and key in MESH_PRESETS:
            mesh_cache[key] = MESH_PRESETS[key]()
        return mesh_cache[key]

    def recompute():
        n, a, v = get_mesh()
        _, fn = STRATEGIES[state["strategy"]]
        return fn(n, a, v, state)

    # Parse simple args: [obj_path] [--strategy N]
    _args = sys.argv[1:]
    _obj_path = None
    _start_strategy = None
    i = 0
    while i < len(_args):
        if _args[i] == "--strategy" and i+1 < len(_args):
            try:
                _start_strategy = int(_args[i+1]) - 1  # 1-based → 0-based
            except ValueError:
                pass
            i += 2
        else:
            if os.path.isfile(_args[i]):
                _obj_path = _args[i]
            i += 1

    if _obj_path:
        try:
            n, a, v = load_obj(_obj_path)
            nm = os.path.basename(_obj_path)
            mesh_cache[nm] = (n, a, v)
            state["mesh"] = nm
        except Exception as e:
            print(f"Could not load {_obj_path}: {e}", file=sys.stderr)

    if _start_strategy is not None and 0 <= _start_strategy < len(STRATEGIES):
        state["strategy"] = _start_strategy

    dirs, label = recompute()

    # ── 3D view state ──────────────────────────────────────────────────────────
    az, el, zoom = 0.5, 0.3, 1.0
    drag = False; drag_start = (0, 0); az0 = az; el0 = el

    # ── Panel layout ──────────────────────────────────────────────────────────
    PX = PANEL_X + 10
    PW = W - PX - 10

    # Strategy buttons  (y 42 – 294)
    strat_btns = [Button((PX, 42 + i*36, PW, 28), name, active=(i==state["strategy"]))
                  for i, (name, _) in enumerate(STRATEGIES)]

    # Mesh buttons (y 310 – 400)
    def make_mesh_btns():
        keys = list(MESH_PRESETS.keys()) + [k for k in mesh_cache if k not in MESH_PRESETS]
        return [Button((PX, 310 + i*30, PW, 24), k, active=(k==state["mesh"]))
                for i, k in enumerate(keys)]
    mesh_btns = make_mesh_btns()

    # Sliders (y 430 – 660)
    SY = 430
    sl_cams = Slider((PX, SY+15,       PW, 18), "Cameras",   1,   20,  state["n_cams"],   step=1)
    sl_cov  = Slider((PX, SY+15+50,    PW, 18), "Coverage",  0.5, 1.0, state["coverage"], step=0.05)
    sl_elev = Slider((PX, SY+15+100,   PW, 18), "Elevation°",-60, 60,  state["elevation"],step=5)
    sl_fan  = Slider((PX, SY+15+150,   PW, 18), "Fan angle°", 10, 180, state["fan_angle"],step=5)
    sl_bal  = Slider((PX, SY+15+200,   PW, 18), "Balance",   0.0, 1.0, state["balance"],  step=0.05)
    sliders = [sl_cams, sl_cov, sl_elev, sl_fan, sl_bal]

    # Action buttons
    btn_regen = Button((PX,            H-46, PW//2-4, 32), "Regenerate")
    btn_load  = Button((PX+PW//2+4,    H-46, PW//2-4, 32), "Load OBJ...", )

    def refresh_strat():
        for i, b in enumerate(strat_btns): b.active = (i == state["strategy"])

    def refresh_mesh():
        keys = list(MESH_PRESETS.keys()) + [k for k in mesh_cache if k not in MESH_PRESETS]
        for i, b in enumerate(mesh_btns):
            if i < len(keys): b.active = (keys[i] == state["mesh"])

    # ── Event loop ─────────────────────────────────────────────────────────────
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # 3D drag
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] < PANEL_X:
                    drag = True; drag_start = event.pos; az0, el0 = az, el
            if event.type == pygame.MOUSEBUTTONUP   and event.button == 1:
                drag = False
            if event.type == pygame.MOUSEMOTION and drag:
                dx = event.pos[0] - drag_start[0]
                dy = event.pos[1] - drag_start[1]
                az = az0 - dx * 0.005
                el = max(-math.pi/2, min(math.pi/2, el0 + dy * 0.005))
            if event.type == pygame.MOUSEWHEEL:
                zoom = max(0.3, min(3.5, zoom + event.y * 0.07))

            # Strategy buttons
            for i, btn in enumerate(strat_btns):
                if btn.handle(event):
                    state["strategy"] = i; refresh_strat()
                    dirs, label = recompute()

            # Mesh buttons
            mesh_keys = list(MESH_PRESETS.keys()) + [k for k in mesh_cache if k not in MESH_PRESETS]
            for i, btn in enumerate(mesh_btns):
                if i < len(mesh_keys) and btn.handle(event):
                    state["mesh"] = mesh_keys[i]; refresh_mesh()
                    dirs, label = recompute()

            # Sliders
            changed = any(sl.handle(event) for sl in sliders)
            if changed:
                state["n_cams"]    = int(sl_cams.val)
                state["coverage"]  = sl_cov.val
                state["elevation"] = sl_elev.val
                state["fan_angle"] = sl_fan.val
                state["balance"]   = sl_bal.val
                dirs, label = recompute()

            # Regenerate
            if btn_regen.handle(event):
                key = state["mesh"]
                if key in MESH_PRESETS:
                    mesh_cache[key] = MESH_PRESETS[key](seed=int(np.random.randint(9999)))
                    dirs, label = recompute()

            # Load OBJ
            if btn_load.handle(event):
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk(); root.withdraw()
                root.attributes("-topmost", True)
                path = filedialog.askopenfilename(
                    title="Select OBJ file",
                    filetypes=[("OBJ files","*.obj"),("All files","*.*")])
                root.destroy()
                if path and os.path.isfile(path):
                    try:
                        n, a, v = load_obj(path)
                        nm = os.path.basename(path)
                        mesh_cache[nm] = (n, a, v)
                        state["mesh"] = nm
                        mesh_btns = make_mesh_btns()
                        refresh_mesh()
                        dirs, label = recompute()
                    except Exception as exc:
                        print(f"OBJ load error: {exc}", file=sys.stderr)

            btn_regen.handle(event)
            btn_load.handle(event)

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(C_BG)
        pygame.draw.rect(screen, C_PANEL, (PANEL_X, 0, W-PANEL_X, H))
        pygame.draw.line(screen, C_SEP, (PANEL_X, 0), (PANEL_X, H), 1)

        # 3D scene
        R = rot_matrix(az, el)
        try:
            normals, areas, verts = get_mesh()
            draw_wireframe_sphere(screen, R, zoom, SPHERE_R * 0.93, C_SPHERE)
            draw_normals_dots(screen, normals, areas, R, zoom, SPHERE_R * 0.88)
        except Exception:
            pass
        draw_camera_arrows(screen, dirs, R, zoom, SPHERE_R, font_sm)

        # 3D labels
        screen.blit(font_bold.render(label, True, C_TEXT), (12, 10))
        screen.blit(font_sm.render("Drag to rotate  •  Scroll to zoom", True, C_DIM), (12, H-18))

        # ── Panel ────────────────────────────────────────────────────────────
        screen.blit(font_bold.render("Strategy", True, C_ACTIVE), (PX, 26))
        for b in strat_btns:
            b.draw(screen, font)

        mesh_keys = list(MESH_PRESETS.keys()) + [k for k in mesh_cache if k not in MESH_PRESETS]
        screen.blit(font_bold.render("Mesh", True, C_ACTIVE), (PX, 294))
        for b in mesh_btns:
            b.draw(screen, font)

        screen.blit(font_bold.render("Parameters", True, C_ACTIVE), (PX, SY - 14))
        for sl in sliders:
            sl.draw(screen, font_sm)

        # Camera direction readout
        ry = SY + 15 + 250
        screen.blit(font_bold.render(f"Cameras: {len(dirs)}", True, C_TEXT), (PX, ry))
        for i, d in enumerate(dirs[:9]):
            dn = d / (np.linalg.norm(d) + 1e-12)
            az_d = math.degrees(math.atan2(dn[1], dn[0]))
            el_d = math.degrees(math.asin(float(np.clip(dn[2], -1, 1))))
            col  = CAM_COLORS[i % len(CAM_COLORS)]
            txt  = font_sm.render(f"  {i+1:2d}  {az_d:+6.1f}°  {el_d:+5.1f}°", True, col)
            screen.blit(txt, (PX, ry + 16 + i*13))
        if len(dirs) > 9:
            screen.blit(font_sm.render(f"  ... +{len(dirs)-9} more", True, C_DIM),
                        (PX, ry + 16 + 9*13))

        btn_regen.draw(screen, font)
        btn_load.draw(screen, font)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
