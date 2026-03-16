"""
Interactive pygame/OpenGL viewers for StableGen standalone.

Functions:
    view_result(mesh, uv, texture_img, warnings=None, gen_images=None)
        Open a textured-mesh viewer window.

    view_images_flat(gen_images)
        Show all generated camera images in a flat 2-D grid window.

    view_cameras(mesh, cameras)
        Open a camera-placement preview window.
        Returns True to proceed, False to abort.
"""

import sys
import math
import numpy as np


# Palette: one distinct colour per camera (up to 12, then cycles)
_CAM_COLORS = [
    (1.00, 0.33, 0.33), (0.33, 0.75, 1.00), (0.40, 0.90, 0.40),
    (1.00, 0.85, 0.20), (0.90, 0.40, 0.90), (0.30, 0.95, 0.85),
    (1.00, 0.60, 0.20), (0.60, 0.40, 1.00), (0.80, 1.00, 0.40),
    (1.00, 0.45, 0.70), (0.40, 0.70, 1.00), (0.95, 0.70, 0.40),
]


# ── Interactive viewer ────────────────────────────────────────────────────────

def view_result(mesh, uv, texture_img, warnings=None, gen_images=None, coverage_texture=None):
    """
    Open a pygame/OpenGL window showing the textured mesh with a HUD overlay.
    Controls: left-drag = orbit, scroll = zoom, Esc/Q = close.
    warnings:   optional list of strings shown in the HUD as caution lines.
    gen_images: optional list of PIL Images (one per camera) shown as a
                thumbnail strip along the bottom of the window.
    """
    try:
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        from OpenGL.GL import (
            GL_AMBIENT, GL_AMBIENT_AND_DIFFUSE, GL_BLEND,
            GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
            GL_COLOR_MATERIAL, GL_DIFFUSE, GL_FRONT_AND_BACK, GL_LIGHT0,
            GL_LIGHTING, GL_LINE_LOOP, GL_LINES, GL_LINEAR, GL_MODELVIEW, GL_NORMALIZE,
            GL_ONE_MINUS_SRC_ALPHA, GL_POLYGON_OFFSET_FILL,
            GL_POSITION, GL_PROJECTION, GL_QUADS, GL_RGBA, GL_SRC_ALPHA,
            GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
            GL_TRIANGLES, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE,
            glBegin, glBindTexture, glBlendFunc, glClear, glClearColor,
            glColor3f, glColor4f, glDeleteTextures, glDisable, glEnable, glEnd,
            glGenTextures, glLightfv, glLineWidth, glLoadIdentity, glMaterialfv,
            glMatrixMode, glNormal3fv, glOrtho, glPixelStorei,
            glPolygonOffset, glPopMatrix, glPushMatrix, glRotatef, glTexCoord2f,
            glTexCoord2fv, glTexImage2D, glTexParameteri, glTranslatef,
            glVertex2f, glVertex3f, glVertex3fv,
        )
        from OpenGL.GLU import gluPerspective
    except ImportError as e:
        print(f"[view] pygame/PyOpenGL not available: {e}", file=sys.stderr)
        print("[view] Install: .\\venv\\Scripts\\python.exe -m pip install pygame PyOpenGL",
              file=sys.stderr)
        return

    # PyOpenGL bug: glGenTextures(1) uses a ctypes scalar that ctypes converts to
    # an unregistered <cparam 'P'> pointer type.  count=2 takes the numpy array
    # path instead and always works; we just use the first ID.
    def _gen_tex():
        return int(glGenTextures(2)[0])

    # ── Normalise mesh to unit cube centred at origin ──────────────────────────
    verts   = np.array(mesh.vertices,       dtype=np.float32)
    faces   = np.array(mesh.faces,          dtype=np.int32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)
    uv_arr  = np.array(uv,                  dtype=np.float32)

    verts -= verts.mean(axis=0)
    scale   = np.abs(verts).max()
    if scale > 0:
        verts /= scale

    # Unique mesh edges for wireframe overlay
    _edge_count = {}
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1) % 3])
            key = (min(a, b), max(a, b))
            _edge_count[key] = _edge_count.get(key, 0) + 1
    mesh_edges = np.array(list(_edge_count.keys()), dtype=np.int32)   # (E, 2)

    # UV island boundary edges: 2D inset pairs + 3D on-mesh pairs
    _uv_seam_pairs = []   # (uv_a, uv_b) in [0,1] for 2D inset
    _uv_seam_3d    = []   # (vert_a, vert_b) in normalised 3D space
    for (a, b), cnt in _edge_count.items():
        if cnt == 1:
            _uv_seam_pairs.append((uv_arr[a], uv_arr[b]))
            _uv_seam_3d.append((verts[a], verts[b]))

    # ── Upload mesh texture ────────────────────────────────────────────────────
    # PIL row-0 = top; OpenGL row-0 = bottom → flip so UV v=0 is bottom.
    tex_rgba = np.ascontiguousarray(
        np.flipud(np.array(texture_img.convert("RGBA"), dtype=np.uint8))
    )
    tex_h, tex_w = tex_rgba.shape[:2]

    # ── Thumbnail strip layout ─────────────────────────────────────────────────
    THUMB   = 96          # thumbnail height (and width, images are square)
    T_PAD   = 6           # gap between thumbnails / border
    T_LABEL = 16          # label height below each thumbnail
    STRIP_H = THUMB + T_PAD * 2 + T_LABEL if gen_images else 0
    BTN_BAR_H = 36        # toggle button bar height

    # ── pygame / OpenGL init ───────────────────────────────────────────────────
    W, H = 900, 700 + BTN_BAR_H + STRIP_H
    VIEW_H = 700          # 3D viewport height
    pygame.init()
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("StableGen -- textured mesh")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, W / VIEW_H, 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)

    def _upload_tex(rgba_arr):
        h, w = rgba_arr.shape[:2]
        tid = _gen_tex()
        glBindTexture(GL_TEXTURE_2D, tid)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_arr.tobytes())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tid

    mesh_tex_id = _upload_tex(tex_rgba)

    coverage_tex_id = None
    if coverage_texture is not None:
        cov_rgba = np.ascontiguousarray(
            np.flipud(np.array(coverage_texture.convert("RGBA"), dtype=np.uint8))
        )
        coverage_tex_id = _upload_tex(cov_rgba)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_TEXTURE_2D)
    glLightfv(GL_LIGHT0, GL_POSITION, np.array([2.0, 3.0, 2.0, 0.0], dtype=np.float32))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  np.array([0.85, 0.85, 0.85, 1.0], dtype=np.float32))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float32))
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    glClearColor(0.13, 0.13, 0.13, 1.0)

    # ── Upload generated camera thumbnails ─────────────────────────────────────
    thumb_ids = []   # list of (gl_id, label_gl_id, lw, lh) or None
    if gen_images:
        font_th = pygame.font.SysFont("consolas", 12)
        for ci, img in enumerate(gen_images):
            if img is None:
                thumb_ids.append(None)
                continue
            # Image thumbnail
            thumb = img.resize((THUMB, THUMB)).convert("RGBA")
            td = np.ascontiguousarray(np.flipud(np.array(thumb, dtype=np.uint8)))
            tid = _gen_tex()
            glBindTexture(GL_TEXTURE_2D, tid)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, THUMB, THUMB,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, td.tobytes())
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            # Label surface
            lsurf = font_th.render(f"Cam {ci+1}", True, (200, 200, 200))
            ldata = pygame.image.tostring(lsurf, "RGBA", True)
            lw, lh = lsurf.get_size()
            lid = _gen_tex()
            glBindTexture(GL_TEXTURE_2D, lid)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lw, lh,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, ldata)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            thumb_ids.append((tid, lid, lw, lh))

    # ── HUD helpers ────────────────────────────────────────────────────────────
    font_sm = pygame.font.SysFont("consolas", 14)
    font_warn = pygame.font.SysFont("consolas", 14, bold=True)

    def _surface_to_gl_texture(surf):
        """Upload a pygame Surface as an RGBA GL texture; return (id, w, h)."""
        data = pygame.image.tostring(surf, "RGBA", True)
        sw, sh = surf.get_size()
        tid = _gen_tex()
        glBindTexture(GL_TEXTURE_2D, tid)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sw, sh,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tid, sw, sh

    def _make_hud_texture(info_lines, warn_lines, fps):
        """Render HUD lines to a pygame Surface and upload as GL texture.
        info_lines: list of str  OR  list of (str, rgb_tuple).
        """
        COL_INFO  = (210, 210, 210)
        COL_WARN  = (255, 200,  60)
        COL_FPS   = (120, 220, 120)
        PAD = 6

        rendered = []
        for line in info_lines:
            if isinstance(line, tuple):
                text, col = line
            else:
                text, col = line, COL_INFO
            rendered.append(font_sm.render(text, True, col))
        for line in warn_lines:
            rendered.append(font_warn.render(f"! {line}", True, COL_WARN))
        rendered.append(font_sm.render(f"FPS {fps:3d}", True, COL_FPS))

        line_h = rendered[0].get_height() + 3
        sw = max(s.get_width() for s in rendered) + PAD * 2
        sh = line_h * len(rendered) + PAD * 2

        surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 130))
        for i, s in enumerate(rendered):
            surf.blit(s, (PAD, PAD + i * line_h))
        return _surface_to_gl_texture(surf)

    def _draw_hud_quad(tid, qw, qh, x, y):
        """Blit a GL texture as a 2-D quad at screen pixel (x, y) from top-left."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, W, H, 0, -1, 1)        # top-left origin
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, tid)
        glColor4f(1, 1, 1, 1)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x,      y)
        glTexCoord2f(1, 1); glVertex2f(x + qw, y)
        glTexCoord2f(1, 0); glVertex2f(x + qw, y + qh)
        glTexCoord2f(0, 0); glVertex2f(x,      y + qh)
        glEnd()

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    _btn_font = pygame.font.SysFont("consolas", 13, bold=True)

    def _draw_btn_bar():
        """Draw the toggle button bar between the 3D view and thumbnail strip."""
        bar_y = VIEW_H   # top of button bar in window coords
        mx, my = pygame.mouse.get_pos()

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, W, H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_TEXTURE_2D)

        # Background
        glColor4f(0.10, 0.10, 0.14, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(0, bar_y); glVertex2f(W, bar_y)
        glVertex2f(W, bar_y + BTN_BAR_H); glVertex2f(0, bar_y + BTN_BAR_H)
        glEnd()

        # Button definitions: (label, active_flag)
        _cov_labels = ["Texture", "Overlay", "Coverage"]
        btns = [
            (_cov_labels[cov_mode], cov_mode > 0, "cov"),
            ("Wireframe",           show_wireframe, "wf"),
            ("UV Seams",            show_uvs,       "uv"),
        ]
        btn_w = W // len(btns)
        for i, (label, active, tag) in enumerate(btns):
            bx = i * btn_w
            hover = (bx <= mx < bx + btn_w) and (bar_y <= my < bar_y + BTN_BAR_H)
            if active:
                r, g, b = (0.20, 0.55, 0.85)
            elif hover:
                r, g, b = (0.28, 0.28, 0.36)
            else:
                r, g, b = (0.16, 0.16, 0.22)
            glDisable(GL_TEXTURE_2D)
            glColor4f(r, g, b, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(bx + 1,         bar_y + 2)
            glVertex2f(bx + btn_w - 1, bar_y + 2)
            glVertex2f(bx + btn_w - 1, bar_y + BTN_BAR_H - 2)
            glVertex2f(bx + 1,         bar_y + BTN_BAR_H - 2)
            glEnd()
            col = (240, 240, 255) if active else (180, 180, 200)
            surf = _btn_font.render(label, True, col)
            tid, tw, th = _surface_to_gl_texture(surf)
            tx = bx + (btn_w - tw) // 2
            ty = bar_y + (BTN_BAR_H - th) // 2
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, tid)
            glColor4f(1, 1, 1, 1)
            glBegin(GL_QUADS)
            glTexCoord2f(0,1); glVertex2f(tx,      ty)
            glTexCoord2f(1,1); glVertex2f(tx + tw, ty)
            glTexCoord2f(1,0); glVertex2f(tx + tw, ty + th)
            glTexCoord2f(0,0); glVertex2f(tx,      ty + th)
            glEnd()
            glDeleteTextures([tid])

        glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()

    def _btn_bar_click(mx, my):
        """Handle a click at (mx, my); return True if it hit a button."""
        nonlocal cov_mode, show_wireframe, show_uvs
        bar_y = VIEW_H
        if not (bar_y <= my < bar_y + BTN_BAR_H):
            return False
        btn_w = W // 3
        col = mx // btn_w
        if col == 0:
            cov_mode = (cov_mode + 1) % 3
        elif col == 1:
            show_wireframe = not show_wireframe
        elif col == 2:
            show_uvs = not show_uvs
        return True

    # ── Interaction state ──────────────────────────────────────────────────────
    yaw, pitch   = 30.0, -20.0
    zoom         = -3.0
    dragging     = False
    last_mouse   = None
    clock        = pygame.time.Clock()
    fps          = 0
    hud_tid      = None
    hud_w = hud_h = 0
    # 0 = texture only, 1 = texture + coverage overlay, 2 = coverage only
    cov_mode      = 0
    show_wireframe = False   # W toggles mesh edge overlay
    show_uvs       = False   # U toggles UV edge overlay

    def _make_info_lines():
        DIM   = (110, 110, 130)
        ON    = (255, 255, 255)
        OFF   = (130, 130, 150)
        KEY   = ( 80, 160, 220)
        cov_opts = ["Texture", "Overlay", "Coverage"]
        cov_str = "  ".join(
            f"[{o}]" if i == cov_mode else f" {o} "
            for i, o in enumerate(cov_opts)
        )
        wf_str = f"Wireframe: {'ON ' if show_wireframe else 'off'}"
        uv_str = f"UV Seams:  {'ON ' if show_uvs       else 'off'}"
        return [
            (f"Verts: {len(mesh.vertices):,}   Faces: {len(mesh.faces):,}", DIM),
            (f"T  {cov_str}",         ON  if cov_mode > 0 else OFF),
            (f"W  {wf_str}",          ON  if show_wireframe else OFF),
            (f"U  {uv_str}",          ON  if show_uvs       else OFF),
            ("Drag: orbit   Scroll: zoom   Q: quit", DIM),
        ]

    warn_lines = list(warnings) if warnings else []

    print("[view] Window open — drag to orbit, scroll to zoom, T: coverage, W: wireframe, U: UVs, Q/Esc to close")
    if warn_lines:
        for w in warn_lines:
            print(f"[view] WARNING: {w}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_t and coverage_tex_id is not None:
                    cov_mode = (cov_mode + 1) % 3
                    hud_tid = None
                elif event.key == pygame.K_w:
                    show_wireframe = not show_wireframe
                    hud_tid = None
                elif event.key == pygame.K_u:
                    show_uvs = not show_uvs
                    hud_tid = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if _btn_bar_click(mx, my):
                        hud_tid = None
                    else:
                        dragging, last_mouse = True, event.pos
                elif event.button == 4:
                    zoom = min(zoom + 0.15, -0.3)
                elif event.button == 5:
                    zoom = max(zoom - 0.15, -20.0)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging and last_mouse:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    yaw   += dx * 0.5
                    pitch += dy * 0.5
                    last_mouse = event.pos

        # ── 3-D mesh ───────────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(pitch, 1, 0, 0)
        glRotatef(yaw,   0, 1, 0)

        def _draw_mesh_with_tex(tex_id, alpha=1.0):
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glColor4f(1, 1, 1, alpha)
            glBegin(GL_TRIANGLES)
            for face in faces:
                for vi in face:
                    glTexCoord2fv(uv_arr[vi])
                    glNormal3fv(normals[vi])
                    glVertex3fv(verts[vi])
            glEnd()

        if cov_mode == 2 and coverage_tex_id:
            # Coverage only
            _draw_mesh_with_tex(coverage_tex_id)
        elif cov_mode == 1 and coverage_tex_id:
            # Texture base + transparent coverage overlay
            _draw_mesh_with_tex(mesh_tex_id)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, -1.0)
            _draw_mesh_with_tex(coverage_tex_id, alpha=0.5)
            glDisable(GL_POLYGON_OFFSET_FILL)
            glDisable(GL_BLEND)
        else:
            # Texture only
            _draw_mesh_with_tex(mesh_tex_id)

        # ── Wireframe overlay (W) ──────────────────────────────────────────────
        if show_wireframe:
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, -1.0)
            glLineWidth(1.0)
            glColor4f(0.0, 0.0, 0.0, 0.4)
            glBegin(GL_LINES)
            for a, b in mesh_edges:
                glVertex3fv(verts[a])
                glVertex3fv(verts[b])
            glEnd()
            glDisable(GL_POLYGON_OFFSET_FILL)
            glDisable(GL_BLEND)
            glEnable(GL_LIGHTING)
            glEnable(GL_TEXTURE_2D)

        # ── UV seam edges on mesh (U) ─────────────────────────────────────────
        if show_uvs and _uv_seam_3d:
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)
            glColor3f(1.0, 0.85, 0.1)
            glLineWidth(1.5)
            glBegin(GL_LINES)
            for va, vb in _uv_seam_3d:
                glVertex3fv(va); glVertex3fv(vb)
            glEnd()
            glLineWidth(1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_TEXTURE_2D)


        # ── HUD overlay (rebuilt on mode change or each second for live FPS) ────
        new_fps = int(clock.get_fps())
        if hud_tid is None or new_fps != fps:
            if hud_tid is not None:
                glDeleteTextures([hud_tid])
            fps = new_fps
            hud_tid, hud_w, hud_h = _make_hud_texture(_make_info_lines(), warn_lines, fps)

        _draw_hud_quad(hud_tid, hud_w, hud_h, x=10, y=10)
        _draw_btn_bar()

        # ── Camera thumbnail strip ─────────────────────────────────────────────
        if thumb_ids:
            n     = len(thumb_ids)
            total = n * THUMB + (n - 1) * T_PAD
            x0    = max(T_PAD, (W - total) // 2)
            ty    = H - STRIP_H + T_PAD           # top-left Y of thumbnails

            glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
            glOrtho(0, W, H, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
            glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_TEXTURE_2D)

            for ci, entry in enumerate(thumb_ids):
                tx = x0 + ci * (THUMB + T_PAD)
                if entry is None:
                    continue
                tid, lid, lw, lh = entry
                col = _CAM_COLORS[ci % len(_CAM_COLORS)]
                glColor4f(*col, 1.0)
                glBindTexture(GL_TEXTURE_2D, tid)
                glBegin(GL_QUADS)
                glTexCoord2f(0,1); glVertex2f(tx,          ty)
                glTexCoord2f(1,1); glVertex2f(tx + THUMB,  ty)
                glTexCoord2f(1,0); glVertex2f(tx + THUMB,  ty + THUMB)
                glTexCoord2f(0,0); glVertex2f(tx,          ty + THUMB)
                glEnd()
                # label centred below thumbnail
                lx = tx + (THUMB - lw) // 2
                ly = ty + THUMB + 2
                glColor4f(1, 1, 1, 1)
                glBindTexture(GL_TEXTURE_2D, lid)
                glBegin(GL_QUADS)
                glTexCoord2f(0,1); glVertex2f(lx,      ly)
                glTexCoord2f(1,1); glVertex2f(lx + lw, ly)
                glTexCoord2f(1,0); glVertex2f(lx + lw, ly + lh)
                glTexCoord2f(0,0); glVertex2f(lx,      ly + lh)
                glEnd()

            glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D)
            glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
            glMatrixMode(GL_PROJECTION); glPopMatrix()
            glMatrixMode(GL_MODELVIEW);  glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

    if hud_tid is not None:
        glDeleteTextures([hud_tid])
    if thumb_ids:
        for entry in thumb_ids:
            if entry:
                glDeleteTextures([entry[0], entry[1]])
    pygame.quit()


# ── Camera placement viewer ───────────────────────────────────────────────────

_CAM_MODES = [
    (1, "Orbit"),
    (2, "Fan"),
    (3, "Hemisphere"),
    (4, "PCA"),
    (5, "K-Means"),
    (6, "Greedy"),
    (7, "Visibility"),
]


def view_cameras(mesh, cameras, build_fn=None, init_mode=5, init_n=None, texture_img=None, uv=None):
    """
    Show the mesh + placed cameras in a pygame/OpenGL window.
    Each camera is drawn with its position, a line to the mesh centroid,
    and a frustum outline.

    Optional args:
        build_fn(n, mode_int) -> list[cam_dict]  – called when the user
            changes mode or count.  If None, controls are hidden.
        init_mode   – initial camera mode integer (1-7, default 5 = K-Means).
        init_n      – initial camera count (defaults to len(cameras)).

    Returns (proceed: bool, cameras: list).
        proceed is True  when the user presses Enter/Space,
                   False when they press Esc.
    """
    try:
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        from OpenGL.GL import (
            GL_AMBIENT, GL_AMBIENT_AND_DIFFUSE, GL_BLEND,
            GL_COLOR_BUFFER_BIT, GL_COLOR_MATERIAL, GL_DEPTH_BUFFER_BIT,
            GL_DEPTH_TEST, GL_DIFFUSE, GL_FILL, GL_FRONT_AND_BACK, GL_LIGHT0,
            GL_LIGHTING, GL_LINE_LOOP, GL_LINES, GL_LINEAR, GL_MODELVIEW, GL_NORMALIZE,
            GL_ONE_MINUS_SRC_ALPHA, GL_POINTS, GL_POLYGON_OFFSET_FILL,
            GL_POSITION, GL_PROJECTION, GL_QUADS, GL_RGBA, GL_SRC_ALPHA,
            GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
            GL_TRIANGLES, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE,
            glBegin, glBindTexture, glBlendFunc, glClear, glClearColor,
            glColor3f, glColor4f, glColorMaterial, glDeleteTextures,
            glDisable, glEnable, glEnd, glGenTextures, glLightfv, glLineWidth,
            glLoadIdentity, glMaterialfv, glMatrixMode, glNormal3fv, glOrtho,
            glPixelStorei, glPointSize, glPolygonMode, glPolygonOffset,
            glPopMatrix, glPushMatrix, glRotatef, glTexCoord2f, glTexImage2D,
            glTexParameteri, glTranslatef, glVertex3f, glVertex3fv, glVertex2f,
        )
        from OpenGL.GLU import gluPerspective
    except ImportError as e:
        print(f"[camera-gui] pygame/PyOpenGL not available: {e}", file=sys.stderr)
        return True, cameras   # proceed anyway

    # Same glGenTextures(1) bug workaround as view_result.
    def _gen_tex():
        return int(glGenTextures(2)[0])

    # ── Mesh geometry ──────────────────────────────────────────────────────────
    verts   = np.array(mesh.vertices,       dtype=np.float32)
    faces   = np.array(mesh.faces,          dtype=np.int32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)

    centroid = verts.mean(axis=0)
    verts   -= centroid
    scale    = np.abs(verts).max()
    if scale > 0:
        verts /= scale

    face_normals_raw = np.array(mesh.face_normals, dtype=np.float32)

    mesh_uv = None
    _cam_uv_seam_pairs = []   # 2D UV-space seam pairs for inset
    _cam_seam_edges_3d = []   # 3D vertex pairs for on-mesh overlay
    try:
        uv_raw = uv if uv is not None else mesh.visual.uv
        if uv_raw is not None and len(uv_raw) == len(mesh.vertices):
            mesh_uv = np.array(uv_raw, dtype=np.float32)
            _edge_cnt = {}
            for tri in faces:
                for i in range(3):
                    a, b = int(tri[i]), int(tri[(i+1) % 3])
                    key = (min(a, b), max(a, b))
                    _edge_cnt[key] = _edge_cnt.get(key, 0) + 1
            for (a, b), cnt in _edge_cnt.items():
                if cnt == 1:
                    _cam_uv_seam_pairs.append((mesh_uv[a], mesh_uv[b]))
                    _cam_seam_edges_3d.append((verts[a], verts[b]))
    except Exception:
        pass

    cam_positions = []
    cam_frustums  = []
    for cam in cameras:
        pos = (np.array(cam["pos"], dtype=np.float32) - centroid) / (scale if scale else 1)
        cam_positions.append(pos)

        # Frustum corners from pose matrix (camera→world axes)
        pose   = np.array(cam["pose"], dtype=np.float32)
        right  = pose[:3, 0]
        up     = pose[:3, 1]
        fwd    = -pose[:3, 2]       # camera looks down -Z
        near   = np.linalg.norm(pos) * 0.15
        hh     = near * math.tan(cam["yfov"] / 2)
        hw     = hh * 1.0           # assume square frustum for display

        tip     = pos
        base_c  = pos + fwd * near
        corners = [
            base_c + right * hw + up * hh,
            base_c - right * hw + up * hh,
            base_c - right * hw - up * hh,
            base_c + right * hw - up * hh,
        ]
        cam_frustums.append((tip, corners))

    def _recompute_coverage(positions):
        """Return (face_cam, face_max) for a list of normalised cam positions."""
        if positions:
            dirs = np.array(
                [-p / (np.linalg.norm(p) + 1e-8) for p in positions],
                dtype=np.float32,
            )
            sc  = face_normals_raw @ dirs.T
            fc  = np.argmax(sc, axis=1)
            fm  = sc[np.arange(len(sc)), fc]
        else:
            fc = np.zeros(len(faces), dtype=np.int32)
            fm = np.ones(len(faces), dtype=np.float32)
        return fc, fm

    face_cam, face_max = _recompute_coverage(cam_positions)

    # ── Interactive mode state ─────────────────────────────────────────────────
    TAB_H  = 32
    TAB_W  = 900 // max(len(_CAM_MODES), 1)   # pixels per mode tab
    CTRL_W = 140                                # width of cam-count control area

    mode_int = init_mode
    n_cams   = init_n if init_n is not None else len(cameras)
    # 0-based index of current mode in _CAM_MODES
    mode_idx = next((i for i, (m, _) in enumerate(_CAM_MODES) if m == mode_int), 4)

    def _rebuild(new_mode_idx, new_n):
        """Rebuild cameras and update coverage when mode/count changes."""
        nonlocal cameras, cam_positions, cam_frustums, face_cam, face_max
        nonlocal mode_idx, n_cams, mode_int
        if build_fn is None:
            return
        mode_idx = new_mode_idx % len(_CAM_MODES)
        mode_int, _ = _CAM_MODES[mode_idx]
        n_cams = max(1, min(new_n, 32))
        try:
            cameras = build_fn(n_cams, mode_int)
        except Exception as e:
            print(f"[debug_camera-gui] build_fn error: {e}", file=sys.stderr)
            return
        # Recompute positions and frustums
        cam_positions.clear()
        cam_frustums.clear()
        for cam in cameras:
            pos = (np.array(cam["pos"], dtype=np.float32) - centroid) / (scale if scale else 1)
            cam_positions.append(pos)
            pose   = np.array(cam["pose"], dtype=np.float32)
            right  = pose[:3, 0]
            up     = pose[:3, 1]
            fwd    = -pose[:3, 2]
            near   = np.linalg.norm(pos) * 0.15
            hh     = near * math.tan(cam["yfov"] / 2)
            hw     = hh * 1.0
            tip    = pos
            base_c = pos + fwd * near
            corners = [
                base_c + right * hw + up * hh,
                base_c - right * hw + up * hh,
                base_c - right * hw - up * hh,
                base_c + right * hw - up * hh,
            ]
            cam_frustums.append((tip, corners))
        face_cam, face_max = _recompute_coverage(cam_positions)

    # ── pygame / OpenGL init ───────────────────────────────────────────────────
    W, H = 900, 700 + (TAB_H if build_fn else 0)
    pygame.init()
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(
        "Camera placement  — Enter/Space: proceed   Esc: abort")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    VIEW_H = H - (TAB_H if build_fn else 0)
    gluPerspective(45.0, W / VIEW_H, 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glLightfv(GL_LIGHT0, GL_POSITION, np.array([2.0, 3.0, 2.0, 0.0], dtype=np.float32))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float32))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  np.array([0.30, 0.30, 0.30, 1.0], dtype=np.float32))
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, np.array([0.55, 0.55, 0.60, 1.0], dtype=np.float32))
    glClearColor(0.10, 0.10, 0.10, 1.0)

    mesh_tex_id = None
    if texture_img is not None and mesh_uv is not None:
        try:
            mesh_tex_id = _gen_tex()
            tex_data = np.flipud(np.array(texture_img.convert("RGBA"), dtype=np.uint8))
            glBindTexture(GL_TEXTURE_2D, mesh_tex_id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                         tex_data.shape[1], tex_data.shape[0],
                         0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data.tobytes())
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        except Exception as e:
            print(f"[debug_camera-gui] texture upload failed: {e}", file=sys.stderr)
            mesh_tex_id = None

    # ── HUD helpers ───────────────────────────────────────────────────────────
    font_sm   = pygame.font.SysFont("consolas", 14)
    font_bold = pygame.font.SysFont("consolas", 14, bold=True)

    def _surface_to_gl(surf):
        data = pygame.image.tostring(surf, "RGBA", True)
        sw, sh = surf.get_size()
        tid = _gen_tex()
        glBindTexture(GL_TEXTURE_2D, tid)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sw, sh,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tid, sw, sh

    tab_tex_ids = {}   # mode_idx -> (tid, w, h) for tab labels; rebuilt on change
    tab_dirty   = [True]

    def _build_tab_textures():
        tab_tex_ids.clear()
        for i, (_, name) in enumerate(_CAM_MODES):
            active = (i == mode_idx)
            col    = (20, 20, 25) if active else (200, 200, 210)
            fnt    = font_bold if active else font_sm
            surf   = fnt.render(name, True, col)
            tab_tex_ids[i] = _surface_to_gl(surf)
        tab_dirty[0] = False

    def _draw_tab_bar():
        """Draw the mode tab bar and camera-count control at the bottom."""
        if not build_fn:
            return
        if tab_dirty[0]:
            _build_tab_textures()
        BAR_Y = VIEW_H
        tab_w = (W - CTRL_W) // len(_CAM_MODES)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, W, H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_TEXTURE_2D)

        # Background bar
        glColor4f(0.12, 0.12, 0.16, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(0, BAR_Y); glVertex2f(W, BAR_Y)
        glVertex2f(W, H);     glVertex2f(0, H)
        glEnd()

        # Mode tabs
        for i in range(len(_CAM_MODES)):
            tx = i * tab_w
            active = (i == mode_idx)
            r, g, b = (0.25, 0.55, 0.85) if active else (0.18, 0.18, 0.22)
            glColor4f(r, g, b, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(tx + 1,        BAR_Y + 2)
            glVertex2f(tx + tab_w - 1, BAR_Y + 2)
            glVertex2f(tx + tab_w - 1, H - 2)
            glVertex2f(tx + 1,        H - 2)
            glEnd()
            tid, tw, th = tab_tex_ids[i]
            cx = tx + (tab_w - tw) // 2
            cy = BAR_Y + (TAB_H - th) // 2
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, tid)
            glColor4f(1, 1, 1, 1)
            glBegin(GL_QUADS)
            glTexCoord2f(0,1); glVertex2f(cx,      cy)
            glTexCoord2f(1,1); glVertex2f(cx + tw, cy)
            glTexCoord2f(1,0); glVertex2f(cx + tw, cy + th)
            glTexCoord2f(0,0); glVertex2f(cx,      cy + th)
            glEnd()
            glDisable(GL_TEXTURE_2D)

        # Camera count control  [ - ]  Cams: N  [ + ]
        BTN_W = 28
        cx0   = W - CTRL_W
        mid_w = CTRL_W - BTN_W * 2

        def _btn(x, label, hover=False):
            r, g, b = (0.35, 0.60, 0.90) if hover else (0.22, 0.22, 0.28)
            glDisable(GL_TEXTURE_2D)
            glColor4f(r, g, b, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(x + 1,         BAR_Y + 2)
            glVertex2f(x + BTN_W - 1, BAR_Y + 2)
            glVertex2f(x + BTN_W - 1, H - 2)
            glVertex2f(x + 1,         H - 2)
            glEnd()
            s = font_bold.render(label, True, (230, 230, 255))
            t, tw, th = _surface_to_gl(s)
            bx = x + (BTN_W - tw) // 2
            by = BAR_Y + (TAB_H - th) // 2
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, t)
            glColor4f(1, 1, 1, 1)
            glBegin(GL_QUADS)
            glTexCoord2f(0,1); glVertex2f(bx,      by)
            glTexCoord2f(1,1); glVertex2f(bx + tw, by)
            glTexCoord2f(1,0); glVertex2f(bx + tw, by + th)
            glTexCoord2f(0,0); glVertex2f(bx,      by + th)
            glEnd()
            glDeleteTextures([t])

        mx_now, my_now = pygame.mouse.get_pos()
        minus_hover = (cx0 <= mx_now < cx0 + BTN_W) and my_now >= VIEW_H
        plus_hover  = (W - BTN_W <= mx_now < W)     and my_now >= VIEW_H

        _btn(cx0,          "−", hover=minus_hover)
        _btn(W - BTN_W,    "+", hover=plus_hover)

        # Count label in the middle
        glDisable(GL_TEXTURE_2D)
        glColor4f(0.15, 0.15, 0.20, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(cx0 + BTN_W,         BAR_Y + 2)
        glVertex2f(cx0 + BTN_W + mid_w, BAR_Y + 2)
        glVertex2f(cx0 + BTN_W + mid_w, H - 2)
        glVertex2f(cx0 + BTN_W,         H - 2)
        glEnd()
        s = font_bold.render(f"{n_cams}", True, (200, 220, 255))
        t, tw, th = _surface_to_gl(s)
        lx = cx0 + BTN_W + (mid_w - tw) // 2
        ly = BAR_Y + (TAB_H - th) // 2
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, t)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,1); glVertex2f(lx,      ly)
        glTexCoord2f(1,1); glVertex2f(lx + tw, ly)
        glTexCoord2f(1,0); glVertex2f(lx + tw, ly + th)
        glTexCoord2f(0,0); glVertex2f(lx,      ly + th)
        glEnd()
        glDeleteTextures([t])
        glDisable(GL_TEXTURE_2D)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()

    def _make_hud(fps):
        PAD, LH = 6, 18
        DIM = (110, 110, 130)
        ON  = (255, 255, 255)
        OFF = (130, 130, 150)
        r255 = lambda rgb: tuple(int(c * 255) for c in rgb)

        mode_name = _CAM_MODES[mode_idx][1] if build_fn else ""
        cov_opts = ["Solid", "Overlay", "Plain"] + (["Textured"] if mesh_tex_id is not None else [])
        cov_str = "  ".join(
            f"[{o}]" if i == cov_draw_mode else f" {o} "
            for i, o in enumerate(cov_opts)
        )

        uv_str = f"UV Seams: {'ON ' if show_uvs else 'off'}"

        lines = [
            (font_sm, f"Verts: {len(mesh.vertices):,}   Faces: {len(mesh.faces):,}"
                      f"   Cams: {len(cameras)}   FPS {fps:3d}"
                      + (f"   Mode: {mode_name}" if build_fn else ""),
             DIM),
            (font_sm, f"C  {cov_str}",
             ON if cov_draw_mode != 0 else OFF),
            (font_sm, f"U  {uv_str}",
             ON if show_uvs else OFF),
            (font_sm, "Enter/Space: proceed   Esc: abort   drag: orbit   scroll: zoom"
                      + ("   </>/Tab: mode   +/-: count" if build_fn else ""),
             DIM),
            (font_sm, "", (0, 0, 0)),
        ]
        for ci, pos in enumerate(cam_positions):
            col = _CAM_COLORS[ci % len(_CAM_COLORS)]
            lines.append((font_bold,
                          f"  Cam {ci+1:2d}  ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})",
                          r255(col)))

        sw = max(fnt.size(txt)[0] for fnt, txt, _ in lines) + PAD * 2
        sh = LH * len(lines) + PAD * 2
        surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 140))
        for i, (fnt, txt, col) in enumerate(lines):
            surf.blit(fnt.render(txt, True, col), (PAD, PAD + i * LH))
        return _surface_to_gl(surf)

    def _draw_quad(tid, qw, qh, x, y):
        glDisable(GL_TEXTURE_2D)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, W, H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tid)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,1); glVertex2f(x,      y)
        glTexCoord2f(1,1); glVertex2f(x+qw,   y)
        glTexCoord2f(1,0); glVertex2f(x+qw,   y+qh)
        glTexCoord2f(0,0); glVertex2f(x,       y+qh)
        glEnd()
        glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()

    # ── Interaction state ──────────────────────────────────────────────────────
    yaw, pitch = 20.0, -15.0
    zoom       = -4.0
    dragging   = False
    last_mouse = None
    clock      = pygame.time.Clock()
    fps        = 0
    hud_tid    = None
    hud_w = hud_h = 0
    proceed    = True
    # 0 = solid coverage colours, 1 = transparent overlay over grey, 2 = plain grey
    cov_draw_mode = 0
    show_uvs = False

    print(f"[debug_camera-gui] {len(cameras)} cameras placed — Enter to proceed, Esc to abort, C: cycle coverage, U: UV seams")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    running = False
                elif event.key == pygame.K_ESCAPE:
                    proceed, running = False, False
                elif event.key in (pygame.K_RIGHT, pygame.K_TAB) and build_fn:
                    _rebuild(mode_idx + 1, n_cams)
                    tab_dirty[0] = True; hud_tid = None
                elif event.key == pygame.K_LEFT and build_fn:
                    _rebuild(mode_idx - 1, n_cams)
                    tab_dirty[0] = True; hud_tid = None
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS) and build_fn:
                    _rebuild(mode_idx, n_cams + 1)
                    hud_tid = None
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) and build_fn:
                    _rebuild(mode_idx, n_cams - 1)
                    hud_tid = None
                elif event.key == pygame.K_c:
                    _n_cov = 4 if mesh_tex_id is not None else 3
                    cov_draw_mode = (cov_draw_mode + 1) % _n_cov
                    hud_tid = None
                elif event.key == pygame.K_u:
                    show_uvs = not show_uvs
                    hud_tid = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    # Click on mode tab bar / count buttons?
                    if build_fn and my >= VIEW_H:
                        BTN_W = 28
                        if mx >= W - BTN_W:               # [+]
                            _rebuild(mode_idx, n_cams + 1); hud_tid = None
                        elif mx >= W - CTRL_W and mx < W - CTRL_W + BTN_W:  # [-]
                            _rebuild(mode_idx, n_cams - 1); hud_tid = None
                        else:
                            tab_w = (W - CTRL_W) // len(_CAM_MODES)
                            ti = mx // tab_w
                            if ti < len(_CAM_MODES):
                                _rebuild(ti, n_cams)
                                tab_dirty[0] = True; hud_tid = None
                    else:
                        dragging, last_mouse = True, event.pos
                elif event.button == 4:
                    zoom = min(zoom + 0.2, -0.5)
                elif event.button == 5:
                    zoom = max(zoom - 0.2, -20.0)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging and last_mouse:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    yaw   += dx * 0.5
                    pitch += dy * 0.5
                    last_mouse = event.pos

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(pitch, 1, 0, 0)
        glRotatef(yaw,   0, 1, 0)

        # ── Mesh ───────────────────────────────────────────────────────────────
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        def _draw_coverage_faces(alpha=1.0):
            glBegin(GL_TRIANGLES)
            for fi, face in enumerate(faces):
                ci  = int(face_cam[fi])
                col = _CAM_COLORS[ci % len(_CAM_COLORS)]
                if face_max[fi] > 0:
                    glColor4f(*col, alpha)
                else:
                    glColor4f(0.25, 0.25, 0.28, alpha)
                for vi in face:
                    glNormal3fv(normals[vi])
                    glVertex3fv(verts[vi])
            glEnd()

        if cov_draw_mode == 2:
            # Plain grey — no coverage colours
            glColor4f(0.55, 0.55, 0.60, 1.0)
            glBegin(GL_TRIANGLES)
            for fi, face in enumerate(faces):
                for vi in face:
                    glNormal3fv(normals[vi])
                    glVertex3fv(verts[vi])
            glEnd()
        elif cov_draw_mode == 1:
            # Grey base + transparent coverage overlay
            glColor4f(0.55, 0.55, 0.60, 1.0)
            glBegin(GL_TRIANGLES)
            for fi, face in enumerate(faces):
                for vi in face:
                    glNormal3fv(normals[vi])
                    glVertex3fv(verts[vi])
            glEnd()
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glPolygonOffset(-1.0, -1.0)
            _draw_coverage_faces(alpha=0.5)
            glPolygonOffset(1.0, 1.0)
            glDisable(GL_BLEND)
        elif cov_draw_mode == 3 and mesh_tex_id is not None and mesh_uv is not None:
            # Textured
            glDisable(GL_COLOR_MATERIAL)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, mesh_tex_id)
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glBegin(GL_TRIANGLES)
            for fi, face in enumerate(faces):
                for vi in face:
                    glTexCoord2f(mesh_uv[vi, 0], 1.0 - mesh_uv[vi, 1])
                    glNormal3fv(normals[vi])
                    glVertex3fv(verts[vi])
            glEnd()
            glDisable(GL_TEXTURE_2D)
        else:
            # Solid coverage colours (default)
            _draw_coverage_faces(alpha=1.0)

        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_POLYGON_OFFSET_FILL)

        # ── Camera markers ─────────────────────────────────────────────────────
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glLineWidth(1.5)

        for ci, (pos, (tip, corners)) in enumerate(zip(cam_positions, cam_frustums)):
            col = _CAM_COLORS[ci % len(_CAM_COLORS)]
            origin = np.zeros(3, dtype=np.float32)   # mesh centroid (already shifted)

            # Line: camera → centroid
            glColor3f(*col)
            glBegin(GL_LINES)
            glVertex3f(*pos); glVertex3f(*origin)
            glEnd()

            # Frustum edges: tip → each corner
            glBegin(GL_LINES)
            for c in corners:
                glVertex3f(*tip); glVertex3f(*c)
            glEnd()
            # Frustum base rectangle
            glBegin(GL_LINES)
            for j in range(4):
                glVertex3f(*corners[j]); glVertex3f(*corners[(j+1) % 4])
            glEnd()

        # Camera dots on top of everything
        glPointSize(10.0)
        glBegin(GL_POINTS)
        for ci, pos in enumerate(cam_positions):
            glColor3f(*_CAM_COLORS[ci % len(_CAM_COLORS)])
            glVertex3f(*pos)
        glEnd()
        glPointSize(1.0)
        glLineWidth(1.0)

        # ── UV seam edges on mesh ──────────────────────────────────────────────
        if show_uvs and _cam_seam_edges_3d:
            glDisable(GL_DEPTH_TEST)
            glColor3f(1.0, 0.85, 0.1)
            glLineWidth(1.5)
            glBegin(GL_LINES)
            for va, vb in _cam_seam_edges_3d:
                glVertex3f(*va); glVertex3f(*vb)
            glEnd()
            glLineWidth(1.0)
            glEnable(GL_DEPTH_TEST)

        # ── HUD ───────────────────────────────────────────────────────────────
        new_fps = int(clock.get_fps())
        if hud_tid is None or new_fps != fps:
            if hud_tid is not None:
                glDeleteTextures([hud_tid])
            fps = new_fps
            hud_tid, hud_w, hud_h = _make_hud(fps)

        _draw_quad(hud_tid, hud_w, hud_h, x=10, y=10)
        _draw_tab_bar()

        pygame.display.flip()
        clock.tick(60)

    if hud_tid is not None:
        glDeleteTextures([hud_tid])
    for tid, _, _ in tab_tex_ids.values():
        glDeleteTextures([tid])
    if mesh_tex_id is not None:
        glDeleteTextures([mesh_tex_id])
    pygame.quit()
    return proceed, cameras
