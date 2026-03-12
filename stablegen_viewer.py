"""
Interactive pygame/OpenGL viewers for StableGen standalone.

Functions:
    view_result(mesh, uv, texture_img, warnings=None)
        Open a textured-mesh viewer window.

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

def view_result(mesh, uv, texture_img, warnings=None, gen_images=None):
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
            GL_DIFFUSE, GL_FRONT_AND_BACK, GL_LIGHT0, GL_LIGHTING,
            GL_LINEAR, GL_MODELVIEW, GL_NORMALIZE, GL_ONE_MINUS_SRC_ALPHA,
            GL_POSITION, GL_PROJECTION, GL_QUADS, GL_RGBA, GL_SRC_ALPHA,
            GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
            GL_TRIANGLES, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE,
            glBegin, glBindTexture, glBlendFunc, glClear, glClearColor,
            glColor4f, glDeleteTextures, glDisable, glEnable, glEnd,
            glGenTextures, glLightfv, glLoadIdentity, glMaterialfv,
            glMatrixMode, glNormal3fv, glOrtho, glPixelStorei,
            glPopMatrix, glPushMatrix, glRotatef, glTexCoord2f,
            glTexCoord2fv, glTexImage2D, glTexParameteri, glTranslatef,
            glVertex2f, glVertex3fv,
        )
        from OpenGL.GLU import gluPerspective
    except ImportError as e:
        print(f"[view] pygame/PyOpenGL not available: {e}", file=sys.stderr)
        print("[view] Install: .\\venv\\Scripts\\python.exe -m pip install pygame PyOpenGL",
              file=sys.stderr)
        return

    # ── Normalise mesh to unit cube centred at origin ──────────────────────────
    verts   = np.array(mesh.vertices,       dtype=np.float32)
    faces   = np.array(mesh.faces,          dtype=np.int32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)
    uv_arr  = np.array(uv,                  dtype=np.float32)

    verts -= verts.mean(axis=0)
    scale   = np.abs(verts).max()
    if scale > 0:
        verts /= scale

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

    # ── pygame / OpenGL init ───────────────────────────────────────────────────
    W, H = 900, 700 + STRIP_H
    pygame.init()
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("StableGen — textured mesh")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, W / (H - STRIP_H), 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)

    mesh_tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, mesh_tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_w, tex_h,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_rgba.tobytes())
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_TEXTURE_2D)
    glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.0, 2.0, 0.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.85, 0.85, 0.85, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.35, 0.35, 0.35, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
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
            tid = glGenTextures(1)
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
            lid = glGenTextures(1)
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
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sw, sh,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tid, sw, sh

    def _make_hud_texture(info_lines, warn_lines, fps):
        """Render HUD lines to a pygame Surface and upload as GL texture."""
        COL_INFO  = (210, 210, 210)
        COL_WARN  = (255, 200,  60)
        COL_FPS   = (120, 220, 120)
        PAD = 6

        rendered = []
        for line in info_lines:
            rendered.append(font_sm.render(line, True, COL_INFO))
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

    # ── Build static HUD content ───────────────────────────────────────────────
    info_lines = [
        f"Verts: {len(mesh.vertices):,}   Faces: {len(mesh.faces):,}",
        f"Texture: {texture_img.width} x {texture_img.height}",
        "Drag: orbit   Scroll: zoom   Q: quit",
    ]
    warn_lines = list(warnings) if warnings else []

    # ── Interaction state ──────────────────────────────────────────────────────
    yaw, pitch = 30.0, -20.0
    zoom       = -3.0
    dragging   = False
    last_mouse = None
    clock      = pygame.time.Clock()
    fps        = 0
    hud_tid    = None
    hud_w = hud_h = 0

    print("[view] Window open — drag to orbit, scroll to zoom, Q/Esc to close")
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
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

        glBindTexture(GL_TEXTURE_2D, mesh_tex_id)
        glBegin(GL_TRIANGLES)
        for face in faces:
            for vi in face:
                glTexCoord2fv(uv_arr[vi])
                glNormal3fv(normals[vi])
                glVertex3fv(verts[vi])
        glEnd()

        # ── HUD overlay (rebuilt each second for live FPS) ─────────────────────
        new_fps = int(clock.get_fps())
        if hud_tid is None or new_fps != fps:
            if hud_tid is not None:
                glDeleteTextures([hud_tid])
            fps = new_fps
            hud_tid, hud_w, hud_h = _make_hud_texture(info_lines, warn_lines, fps)

        _draw_hud_quad(hud_tid, hud_w, hud_h, x=10, y=10)

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

def view_cameras(mesh, cameras):
    """
    Show the mesh + placed cameras in a pygame/OpenGL window.
    Each camera is drawn with its position, a line to the mesh centroid,
    and a frustum outline.

    Returns True if the user pressed Enter/Space to proceed with texturing,
    False if they pressed Esc to abort.
    """
    try:
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        from OpenGL.GL import (
            GL_AMBIENT, GL_AMBIENT_AND_DIFFUSE, GL_BLEND,
            GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
            GL_DIFFUSE, GL_FILL, GL_FRONT_AND_BACK, GL_LIGHT0, GL_LIGHTING,
            GL_LINES, GL_LINEAR, GL_MODELVIEW, GL_NORMALIZE,
            GL_ONE_MINUS_SRC_ALPHA, GL_POINTS, GL_POLYGON_OFFSET_FILL,
            GL_POSITION, GL_PROJECTION, GL_QUADS, GL_RGBA, GL_SRC_ALPHA,
            GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
            GL_TRIANGLES, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE,
            glBegin, glBindTexture, glBlendFunc, glClear, glClearColor,
            glColor3f, glColor4f, glDeleteTextures, glDisable, glEnable,
            glEnd, glGenTextures, glLightfv, glLineWidth, glLoadIdentity,
            glMaterialfv, glMatrixMode, glNormal3fv, glOrtho, glPixelStorei,
            glPointSize, glPolygonMode, glPolygonOffset, glPopMatrix,
            glPushMatrix, glRotatef, glTexCoord2f, glTexImage2D,
            glTexParameteri, glTranslatef, glVertex3f, glVertex3fv,
            glVertex2f,
        )
        from OpenGL.GLU import gluPerspective
    except ImportError as e:
        print(f"[camera-gui] pygame/PyOpenGL not available: {e}", file=sys.stderr)
        return True   # proceed anyway

    # ── Mesh geometry ──────────────────────────────────────────────────────────
    verts   = np.array(mesh.vertices,       dtype=np.float32)
    faces   = np.array(mesh.faces,          dtype=np.int32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)

    centroid = verts.mean(axis=0)
    verts   -= centroid
    scale    = np.abs(verts).max()
    if scale > 0:
        verts /= scale

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

    # ── pygame / OpenGL init ───────────────────────────────────────────────────
    W, H = 900, 700
    pygame.init()
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(
        "Camera placement  — Enter/Space: proceed   Esc: abort")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, W / H, 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.0, 2.0, 0.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.75, 0.75, 0.75, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.30, 0.30, 0.30, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.55, 0.55, 0.60, 1.0])
    glClearColor(0.10, 0.10, 0.10, 1.0)

    # ── HUD helpers ───────────────────────────────────────────────────────────
    font_sm   = pygame.font.SysFont("consolas", 14)
    font_bold = pygame.font.SysFont("consolas", 14, bold=True)

    def _surface_to_gl(surf):
        data = pygame.image.tostring(surf, "RGBA", True)
        sw, sh = surf.get_size()
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sw, sh,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tid, sw, sh

    def _make_hud(fps):
        PAD, LH = 6, 18
        lines = []
        r255  = lambda rgb: tuple(int(c * 255) for c in rgb)
        lines.append((font_sm,   f"{len(cameras)} cameras  |  "
                                  f"{len(mesh.vertices):,} verts  |  FPS {fps:3d}",
                      (200, 200, 200)))
        lines.append((font_sm,   "Enter/Space = proceed   Esc = abort   drag = orbit   scroll = zoom",
                      (160, 160, 160)))
        lines.append((font_sm,   "", (0, 0, 0)))
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

    print(f"[camera-gui] {len(cameras)} cameras placed — Enter to proceed, Esc to abort")

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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
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

        # ── Mesh (solid, slightly recessed so wireframe sits on top) ──────────
        glEnable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_TRIANGLES)
        for face in faces:
            for vi in face:
                glNormal3fv(normals[vi])
                glVertex3fv(verts[vi])
        glEnd()
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

        # ── HUD ───────────────────────────────────────────────────────────────
        new_fps = int(clock.get_fps())
        if hud_tid is None or new_fps != fps:
            if hud_tid is not None:
                glDeleteTextures([hud_tid])
            fps = new_fps
            hud_tid, hud_w, hud_h = _make_hud(fps)

        _draw_quad(hud_tid, hud_w, hud_h, x=10, y=10)

        pygame.display.flip()
        clock.tick(60)

    if hud_tid is not None:
        glDeleteTextures([hud_tid])
    pygame.quit()
    return proceed
