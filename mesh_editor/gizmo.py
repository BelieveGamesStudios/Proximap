"""
Unity-style 3D transform gizmo — pure OpenGL, no ImGuizmo.

Matrix convention (mirrors Object.get_model_matrix):
  trimesh column-major TRS composition  →  .T  →  glUniformMatrix4fv(..., GL_FALSE, ...)
This exactly matches how viewport.py passes model/view/proj to the object shader.
"""
import os
import numpy as np
import pyrr
import trimesh
import OpenGL.GL as gl


# ---------------------------------------------------------------------------
# Shader helpers
# ---------------------------------------------------------------------------

def _compile_shader(shader_type: int, source: str) -> int:
    s = gl.glCreateShader(shader_type)
    gl.glShaderSource(s, source)
    gl.glCompileShader(s)
    if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
        log = gl.glGetShaderInfoLog(s).decode()
        raise RuntimeError(f"Gizmo shader compile error:\n{log}")
    return s


def _link_program(vert_src: str, frag_src: str) -> int:
    v = _compile_shader(gl.GL_VERTEX_SHADER, vert_src)
    f = _compile_shader(gl.GL_FRAGMENT_SHADER, frag_src)
    p = gl.glCreateProgram()
    gl.glAttachShader(p, v)
    gl.glAttachShader(p, f)
    gl.glLinkProgram(p)
    if not gl.glGetProgramiv(p, gl.GL_LINK_STATUS):
        log = gl.glGetProgramInfoLog(p).decode()
        raise RuntimeError(f"Gizmo shader link error:\n{log}")
    gl.glDeleteShader(v)
    gl.glDeleteShader(f)
    return p


# ---------------------------------------------------------------------------
# VAO helpers
# ---------------------------------------------------------------------------

def _vao_vbo(vertices: np.ndarray):
    """Create VAO+VBO for float32 position-only data (location 0, stride 0)."""
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glBindVertexArray(0)
    return vao, vbo


def _vao_vbo_ebo(vertices: np.ndarray, indices: np.ndarray):
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
    gl.glBindVertexArray(0)
    return vao, vbo, ebo


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def _gen_cone(sectors: int = 16, height: float = 0.25, radius: float = 0.05):
    """Cone pointing in +Z.  Base centre at origin, tip at (0,0,height)."""
    verts = [[0.0, 0.0, 0.0]]  # idx 0: base centre
    for i in range(sectors):
        a = 2.0 * np.pi * i / sectors
        verts.append([radius * np.cos(a), radius * np.sin(a), 0.0])
    verts.append([0.0, 0.0, height])  # idx sectors+1: tip
    vertices = np.array(verts, dtype=np.float32)

    idxs: list = []
    tip = sectors + 1
    for i in range(sectors):
        n = (i % sectors) + 1
        nn = ((i + 1) % sectors) + 1
        idxs.extend([0, nn, n])     # base fan
        idxs.extend([n, nn, tip])   # side fan
    return vertices, np.array(idxs, dtype=np.uint32)


def _gen_cube(size: float = 0.08):
    h = size * 0.5
    v = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h,  h], [h, -h,  h], [h, h,  h], [-h, h,  h],
    ], dtype=np.float32)
    i = np.array([
        0, 2, 1,  0, 3, 2,
        4, 5, 6,  4, 6, 7,
        0, 1, 5,  0, 5, 4,
        1, 2, 6,  1, 6, 5,
        2, 3, 7,  2, 7, 6,
        3, 0, 4,  3, 4, 7,
    ], dtype=np.uint32)
    return v, i


def _gen_circle(segs: int = 64, radius: float = 1.0):
    pts = []
    for i in range(segs):
        a = 2.0 * np.pi * i / segs
        pts.extend([radius * np.cos(a), radius * np.sin(a), 0.0])
    return np.array(pts, dtype=np.float32)


def _gen_plane_quad(start: float = 0.15, size: float = 0.20):
    e = start + size
    v = np.array([
        [start, start, 0], [e, start, 0], [e, e, 0], [start, e, 0],
    ], dtype=np.float32)
    i = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    return v, i


# ---------------------------------------------------------------------------
# Gizmo class
# ---------------------------------------------------------------------------

class Gizmo:
    """Unity-style 3D transform gizmo drawn with raw OpenGL calls."""

    # ---- axis colours ----
    COL_X     = np.array([0.85, 0.15, 0.15, 1.0], dtype=np.float32)
    COL_Y     = np.array([0.15, 0.85, 0.15, 1.0], dtype=np.float32)
    COL_Z     = np.array([0.20, 0.50, 0.95, 1.0], dtype=np.float32)
    COL_WHITE = np.array([0.90, 0.90, 0.90, 1.0], dtype=np.float32)
    COL_HOVER = np.array([1.00, 0.90, 0.00, 1.0], dtype=np.float32)

    # Number of triangle indices for each solid
    _CONE_TRIS  = 16 * 6   # sectors × (3 base + 3 side)
    _CUBE_TRIS  = 36
    _PLANE_TRIS = 6

    def __init__(self):
        self.operation = "translate"   # "translate" | "rotate" | "scale"
        self.space     = "global"      # "local"     | "global"

        self.use_snapping      = False
        self.snap_translation  = 0.5
        self.snap_rotation     = 15.0
        self.snap_scale        = 0.1

        self.hovered_handle = None
        self.active_handle  = None

        # drag bookmarks
        self._drag_start_pos   = None
        self._drag_start_rot   = None
        self._drag_start_scale = None
        self._drag_start_mouse = None
        self._drag_t_start     = 0.0
        self._drag_plane_start = None
        self._drag_plane_normal = None
        self._drag_angle_start = 0.0

        # GL objects
        self.program = None
        self._loc: dict = {}

        self._line_vao = self._line_vbo = None
        self._cone_vao = self._cone_vbo = self._cone_ebo = None
        self._cube_vao = self._cube_vbo = self._cube_ebo = None
        self._circle_vao = self._circle_vbo = None
        self._plane_vao  = self._plane_vbo = self._plane_ebo = None

    # ------------------------------------------------------------------ init

    def initialize(self, shaders_dir: str):
        with open(os.path.join(shaders_dir, "gizmo.vert")) as fh:
            vert = fh.read()
        with open(os.path.join(shaders_dir, "gizmo.frag")) as fh:
            frag = fh.read()
        self.program = _link_program(vert, frag)
        self._loc = {
            "model": gl.glGetUniformLocation(self.program, "model"),
            "view":  gl.glGetUniformLocation(self.program, "view"),
            "proj":  gl.glGetUniformLocation(self.program, "proj"),
            "color": gl.glGetUniformLocation(self.program, "uColor"),
        }

        line_v = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
        self._line_vao, self._line_vbo = _vao_vbo(line_v)

        cv, ci = _gen_cone()
        self._cone_vao, self._cone_vbo, self._cone_ebo = _vao_vbo_ebo(cv, ci)

        cuv, cui = _gen_cube()
        self._cube_vao, self._cube_vbo, self._cube_ebo = _vao_vbo_ebo(cuv, cui)

        circ = _gen_circle()
        self._circle_vao, self._circle_vbo = _vao_vbo(circ)

        pv, pi = _gen_plane_quad()
        self._plane_vao, self._plane_vbo, self._plane_ebo = _vao_vbo_ebo(pv, pi)

    def shutdown(self):
        if self.program:
            gl.glDeleteProgram(self.program)
            self.program = None
        for items in [
            (self._line_vao,   self._line_vbo,   None),
            (self._cone_vao,   self._cone_vbo,   self._cone_ebo),
            (self._cube_vao,   self._cube_vbo,   self._cube_ebo),
            (self._circle_vao, self._circle_vbo, None),
            (self._plane_vao,  self._plane_vbo,  self._plane_ebo),
        ]:
            try:
                if items[0]: gl.glDeleteVertexArrays(1, [items[0]])
                if items[1]: gl.glDeleteBuffers(1, [items[1]])
                if items[2]: gl.glDeleteBuffers(1, [items[2]])
            except Exception:
                pass

    # ---------------------------------------------------------------- helpers

    def get_scale_factor(self, camera) -> float:
        if camera.is_perspective:
            return camera.distance * 0.16
        return camera.ortho_scale * 0.16

    def _obj_r_col(self, obj) -> np.ndarray:
        """Return trimesh column-major rotation matrix from obj.rotation (Euler °)."""
        rx = np.radians(-obj.rotation[0])
        ry = np.radians(-obj.rotation[1])
        rz = np.radians(-obj.rotation[2])
        return trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')

    def get_gizmo_transforms(self, obj):
        """Legacy helper kept for hit-test code: returns (t_mat, r_mat, gizmo_mat)."""
        t_mat = trimesh.transformations.translation_matrix(obj.position)
        if self.space == "local":
            r_mat = self._obj_r_col(obj)
        else:
            r_mat = np.eye(4, dtype=np.float64)
        gizmo_matrix = (t_mat @ r_mat).astype(np.float32)
        return t_mat, r_mat, gizmo_matrix

    def get_axes_vectors(self, r_mat) -> list:
        if self.space == "local":
            return [r_mat[:3, 0], r_mat[:3, 1], r_mat[:3, 2]]
        return [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
        ]

    def is_handle_active_or_hovered(self, name: str) -> bool:
        if self.active_handle is not None:
            return self.active_handle == name
        return self.hovered_handle == name

    # --------------------------------------------------------- matrix builder

    def _build_model(self, obj, scale: float,
                     orient_col=None, tip_col=None,
                     force_global: bool = False) -> np.ndarray:
        """
        Build a 4×4 model matrix for a gizmo handle.

        Convention mirrors Object.get_model_matrix():
          1. Compose full trimesh column-major matrix (T @ R @ S @ orient @ tip)
          2. Transpose once
          3. Result is passed to GL with GL_FALSE (no additional transpose)

        orient_col : extra rotation in trimesh convention (applied after scale)
        tip_col    : extra translation in trimesh convention (applied last, e.g. cone offset)
        force_global: if True skip object rotation even in local mode (used for outer ring)
        """
        T = trimesh.transformations.translation_matrix(obj.position)
        S = np.diag([scale, scale, scale, 1.0])           # uniform scale (float64)

        if self.space == "local" and not force_global:
            R = self._obj_r_col(obj)
            m = T @ R @ S
        else:
            m = T @ S

        if orient_col is not None:
            m = m @ orient_col
        if tip_col is not None:
            m = m @ tip_col

        # Transpose — same as Object.get_model_matrix() — then cast to float32
        return m.T.astype(np.float32)

    # ------------------------------------------------------ uniform shortcuts

    def _set_model(self, mat: np.ndarray):
        gl.glUniformMatrix4fv(self._loc["model"], 1, gl.GL_FALSE, mat)

    def _set_color(self, col: np.ndarray):
        gl.glUniform4fv(self._loc["color"], 1, col)

    def _active_col(self, handle: str, default: np.ndarray) -> np.ndarray:
        return self.COL_HOVER if self.is_handle_active_or_hovered(handle) else default

    # ------------------------------------------------------------------- draw

    def draw(self, obj, camera, width: float, height: float):
        if obj is None or not self.program:
            return

        gl.glUseProgram(self.program)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(1.0)

        aspect    = width / max(height, 1.0)
        view_mat  = camera.get_view_matrix()
        proj_mat  = camera.get_projection_matrix(aspect)
        sf        = self.get_scale_factor(camera)

        # Pass view/proj exactly as the object shader does (pyrr → GL_FALSE, no extra transform)
        gl.glUniformMatrix4fv(self._loc["view"], 1, gl.GL_FALSE, view_mat)
        gl.glUniformMatrix4fv(self._loc["proj"], 1, gl.GL_FALSE, proj_mat)

        # Trimesh column-major orientation matrices: align +Z geometry to each world axis
        # rotation_matrix(angle, axis) follows trimesh convention (column-major)
        R_to_x = trimesh.transformations.rotation_matrix( np.pi / 2, [0, 1, 0])  # +Z → +X
        R_to_y = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])  # +Z → +Y
        R_to_z = np.eye(4, dtype=np.float64)                                      # +Z → +Z

        # Plane quad is native in the XY plane; rotate to sit on each face
        R_xy_to_xz = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        R_xy_to_yz = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])

        # Tip offset: translate +1 along Z in local space (moves arrowhead to shaft end)
        TIP = trimesh.transformations.translation_matrix([0.0, 0.0, 1.0])

        if self.operation == "translate":
            self._draw_translate(obj, sf, R_to_x, R_to_y, R_to_z,
                                 R_xy_to_xz, R_xy_to_yz, TIP)
        elif self.operation == "scale":
            self._draw_scale(obj, sf, R_to_x, R_to_y, R_to_z, TIP)
        elif self.operation == "rotate":
            self._draw_rotate(obj, camera, sf, R_to_x, R_to_y, R_to_z)

        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(1.0)

    # ------------------------------------------------ per-operation draw

    def _draw_translate(self, obj, sf, Rx, Ry, Rz, Rxz, Ryz, TIP):
        ax_names   = ['axis_x', 'axis_y', 'axis_z']
        ax_orients = [Rx, Ry, Rz]
        ax_cols    = [self.COL_X, self.COL_Y, self.COL_Z]
        active     = self.active_handle

        # ---- shafts (lines) & arrowheads (cones) ----
        for name, ori, col in zip(ax_names, ax_orients, ax_cols):
            if active is not None and active != name:
                continue
            # shaft
            gl.glBindVertexArray(self._line_vao)
            self._set_color(self._active_col(name, col))
            self._set_model(self._build_model(obj, sf, ori))
            gl.glDrawArrays(gl.GL_LINES, 0, 2)
            # cone arrowhead
            gl.glBindVertexArray(self._cone_vao)
            self._set_color(self._active_col(name, col))
            self._set_model(self._build_model(obj, sf, ori, TIP))
            gl.glDrawElements(gl.GL_TRIANGLES, self._CONE_TRIS, gl.GL_UNSIGNED_INT, None)

        # ---- plane handles (translucent quads) ----
        pl_names   = ['plane_xy', 'plane_xz', 'plane_yz']
        pl_orients = [Rz,         Rxz,         Ryz]
        pl_fills   = [
            np.array([0.20, 0.50, 0.95, 0.28], dtype=np.float32),  # XY ≈ blue (Z perp)
            np.array([0.15, 0.85, 0.15, 0.28], dtype=np.float32),  # XZ ≈ green (Y perp)
            np.array([0.85, 0.15, 0.15, 0.28], dtype=np.float32),  # YZ ≈ red (X perp)
        ]

        if active is None or active.startswith('plane_'):
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glBindVertexArray(self._plane_vao)
            for name, ori, fill in zip(pl_names, pl_orients, pl_fills):
                if active is not None and active != name:
                    continue
                model = self._build_model(obj, sf, ori)
                self._set_model(model)
                # filled quad
                hover_fill = np.array([1.0, 0.9, 0.0, 0.55], dtype=np.float32)
                self._set_color(hover_fill if self.is_handle_active_or_hovered(name) else fill)
                gl.glDrawElements(gl.GL_TRIANGLES, self._PLANE_TRIS, gl.GL_UNSIGNED_INT, None)
                # outline
                outline = self.COL_HOVER if self.is_handle_active_or_hovered(name) else np.array([*fill[:3], 1.0], dtype=np.float32)
                self._set_color(outline)
                gl.glDrawArrays(gl.GL_LINE_LOOP, 0, 4)
            gl.glDisable(gl.GL_BLEND)

    def _draw_scale(self, obj, sf, Rx, Ry, Rz, TIP):
        ax_names   = ['axis_x', 'axis_y', 'axis_z']
        ax_orients = [Rx, Ry, Rz]
        ax_cols    = [self.COL_X, self.COL_Y, self.COL_Z]
        active     = self.active_handle

        for name, ori, col in zip(ax_names, ax_orients, ax_cols):
            if active is not None and active != name:
                continue
            # shaft
            gl.glBindVertexArray(self._line_vao)
            self._set_color(self._active_col(name, col))
            self._set_model(self._build_model(obj, sf, ori))
            gl.glDrawArrays(gl.GL_LINES, 0, 2)
            # cube tip
            gl.glBindVertexArray(self._cube_vao)
            self._set_color(self._active_col(name, col))
            self._set_model(self._build_model(obj, sf, ori, TIP))
            gl.glDrawElements(gl.GL_TRIANGLES, self._CUBE_TRIS, gl.GL_UNSIGNED_INT, None)

        # centre uniform-scale cube
        if active is None or active == 'center':
            gl.glBindVertexArray(self._cube_vao)
            self._set_color(self._active_col('center', self.COL_WHITE))
            self._set_model(self._build_model(obj, sf))
            gl.glDrawElements(gl.GL_TRIANGLES, self._CUBE_TRIS, gl.GL_UNSIGNED_INT, None)

    def _draw_rotate(self, obj, camera, sf, Rx, Ry, Rz):
        ring_names   = ['ring_x', 'ring_y', 'ring_z']
        ring_orients = [Rx, Ry, Rz]
        ring_cols    = [self.COL_X, self.COL_Y, self.COL_Z]
        active       = self.active_handle

        gl.glBindVertexArray(self._circle_vao)
        for name, ori, col in zip(ring_names, ring_orients, ring_cols):
            if active is not None and active != name:
                continue
            self._set_color(self._active_col(name, col))
            self._set_model(self._build_model(obj, sf, ori))
            gl.glDrawArrays(gl.GL_LINE_LOOP, 0, 64)

        # camera-facing outer ring (always global orientation)
        if active is None or active == 'ring_outer':
            outer_col = self._active_col('ring_outer', np.array([0.9, 0.9, 0.9, 0.85], dtype=np.float32))
            self._set_color(outer_col)
            self._set_model(self._outer_ring_model(obj, camera, sf))
            gl.glDrawArrays(gl.GL_LINE_LOOP, 0, 64)

    def _outer_ring_model(self, obj, camera, sf) -> np.ndarray:
        """Build a camera-facing ring model (global space, scale 1.15)."""
        eye = camera.get_position()
        fwd = eye - np.asarray(obj.position, dtype=np.float64)
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-6:
            return self._build_model(obj, sf * 1.15, force_global=True)
        fwd /= fwd_len

        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, fwd);  right /= np.linalg.norm(right)
        up2   = np.cross(fwd, right)

        # 4×4 rotation matrix in trimesh column-major style (columns = basis vectors)
        cam_rot = np.eye(4, dtype=np.float64)
        cam_rot[:3, 0] = right
        cam_rot[:3, 1] = up2
        cam_rot[:3, 2] = fwd

        return self._build_model(obj, sf * 1.15, cam_rot, force_global=True)

    # -------------------------------------------------------- hit testing

    def project_local_to_screen(self, local_pt, gizmo_matrix, scale_factor,
                                 vp_mat, width, height):
        """Project a gizmo-local point (pre-scaled by scale_factor) to screen pixels."""
        p4 = np.array([
            local_pt[0] * scale_factor,
            local_pt[1] * scale_factor,
            local_pt[2] * scale_factor,
            1.0,
        ], dtype=np.float32)
        world_pt = gizmo_matrix @ p4
        clip = pyrr.matrix44.apply_to_vector(vp_mat, world_pt)
        if abs(clip[3]) < 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        sx = (ndc[0] + 1.0) * 0.5 * width
        sy = (1.0 - ndc[1]) * 0.5 * height
        return np.array([sx, sy], dtype=np.float32)

    def check_hover(self, mx: float, my: float, obj, camera,
                    width: float, height: float):
        if obj is None:
            return None

        mouse  = np.array([mx, my], dtype=np.float32)
        aspect = width / max(height, 1.0)
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect)
        vp_mat   = pyrr.matrix44.multiply(view_mat, proj_mat)

        _, _, gizmo_matrix = self.get_gizmo_transforms(obj)
        sf = self.get_scale_factor(camera)

        center_s = self.project_local_to_screen([0, 0, 0], gizmo_matrix, sf, vp_mat, width, height)
        if center_s is None:
            return None

        def scr(lp):
            return self.project_local_to_screen(lp, gizmo_matrix, sf, vp_mat, width, height)

        def dist_seg(M, A, B):
            ab = B - A
            t  = np.clip(np.dot(M - A, ab) / max(np.dot(ab, ab), 1e-10), 0.0, 1.0)
            return np.linalg.norm(M - (A + t * ab))

        def in_quad(p, q0, q1, q2, q3):
            def sgn(a, b, c):
                return (a[0]-c[0])*(b[1]-c[1]) - (b[0]-c[0])*(a[1]-c[1])
            s = [sgn(p,q0,q1), sgn(p,q1,q2), sgn(p,q2,q3), sgn(p,q3,q0)]
            return not (any(x < 0 for x in s) and any(x > 0 for x in s))

        if self.operation == "translate":
            # Plane quads first
            for name, corners in [
                ('plane_xy', [[.15,.15,0],[.35,.15,0],[.35,.35,0],[.15,.35,0]]),
                ('plane_yz', [[0,.15,.15],[0,.35,.15],[0,.35,.35],[0,.15,.35]]),
                ('plane_xz', [[.15,0,.15],[.35,0,.15],[.35,0,.35],[.15,0,.35]]),
            ]:
                pts = [scr(c) for c in corners]
                if all(p is not None for p in pts) and in_quad(mouse, *pts):
                    return name
            # Axis arrows
            best, best_d = None, 12.0
            for name, tip in [('axis_x',[1,0,0]),('axis_y',[0,1,0]),('axis_z',[0,0,1])]:
                t = scr(tip)
                if t is None: continue
                d = dist_seg(mouse, center_s, t)
                if d < best_d:
                    best_d, best = d, name
            return best

        elif self.operation == "scale":
            if np.linalg.norm(mouse - center_s) < 15.0:
                return 'center'
            best, best_d = None, 12.0
            for name, tip in [('axis_x',[1,0,0]),('axis_y',[0,1,0]),('axis_z',[0,0,1])]:
                t = scr(tip)
                if t is None: continue
                d = dist_seg(mouse, center_s, t)
                if d < best_d:
                    best_d, best = d, name
            return best

        elif self.operation == "rotate":
            # Outer ring
            eye = camera.get_position()
            fwd = eye - obj.position
            if np.linalg.norm(fwd) > 1e-5:
                fwd = fwd / np.linalg.norm(fwd)
                up  = np.array([0,0,1], dtype=np.float32)
                if abs(np.dot(fwd, up)) > 0.99: up = np.array([0,1,0], dtype=np.float32)
                rr  = np.cross(up, fwd);  rr /= np.linalg.norm(rr)
                uu  = np.cross(fwd, rr)
                pts = []
                for i in range(65):
                    a  = 2.0 * np.pi * i / 64
                    wp = obj.position + (rr * np.cos(a) + uu * np.sin(a)) * sf * 1.15
                    sp = self.project_local_to_screen([0,0,0],
                                                     np.eye(4, dtype=np.float32), 1.0,
                                                     vp_mat, width, height)
                    # project world point directly
                    c4 = np.array([wp[0], wp[1], wp[2], 1.0], dtype=np.float32)
                    cl = pyrr.matrix44.apply_to_vector(vp_mat, c4)
                    if abs(cl[3]) > 1e-6:
                        nd = cl[:3] / cl[3]
                        pts.append(np.array([(nd[0]+1)*.5*width,(1-nd[1])*.5*height]))
                d_outer = min((dist_seg(mouse, pts[i], pts[i+1]) for i in range(len(pts)-1)), default=999)
                if d_outer < 10.0:
                    return 'ring_outer'

            # XYZ rings
            ring_local = {
                'ring_x': lambda t: [0.0, np.cos(t), np.sin(t)],
                'ring_y': lambda t: [np.cos(t), 0.0, np.sin(t)],
                'ring_z': lambda t: [np.cos(t), np.sin(t), 0.0],
            }
            best, best_d = None, 10.0
            for name, fn in ring_local.items():
                pts = [scr(fn(2*np.pi*i/64)) for i in range(65)]
                d = min((dist_seg(mouse, pts[i], pts[i+1])
                         for i in range(len(pts)-1)
                         if pts[i] is not None and pts[i+1] is not None), default=999)
                if d < best_d:
                    best_d, best = d, name
            return best

        return None

    # ---------------------------------------------------------- drag API

    def begin_drag(self, mx: float, my: float, obj, camera,
                   width: float, height: float) -> bool:
        hit = self.check_hover(mx, my, obj, camera, width, height)
        if hit is None:
            return False

        self.active_handle  = hit
        self.hovered_handle = hit
        self._drag_start_pos   = np.copy(obj.position)
        self._drag_start_rot   = np.copy(obj.rotation)
        self._drag_start_scale = np.copy(obj.scale)
        self._drag_start_mouse = np.array([mx, my], dtype=np.float32)

        aspect   = width / max(height, 1.0)
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect)
        vp_mat   = pyrr.matrix44.multiply(view_mat, proj_mat)

        _, r_mat, _ = self.get_gizmo_transforms(obj)
        axes         = self.get_axes_vectors(r_mat)
        ray_o, ray_d = self.get_ray_from_mouse(mx, my, vp_mat, width, height)

        if self.operation == "translate":
            if hit.startswith('axis_'):
                idx = {'axis_x':0,'axis_y':1,'axis_z':2}[hit]
                self._drag_t_start = self.closest_point_on_line(ray_o, ray_d, obj.position, axes[idx])
            elif hit.startswith('plane_'):
                n_idx = {'plane_yz':0,'plane_xz':1,'plane_xy':2}[hit]
                t = self.ray_plane_intersection(ray_o, ray_d, obj.position, axes[n_idx])
                if t is not None:
                    self._drag_plane_start = ray_o + t * ray_d

        elif self.operation == "scale":
            if hit.startswith('axis_'):
                idx = {'axis_x':0,'axis_y':1,'axis_z':2}[hit]
                self._drag_t_start = self.closest_point_on_line(ray_o, ray_d, obj.position, axes[idx])

        elif self.operation == "rotate":
            if hit == 'ring_outer':
                eye = camera.get_position()
                pn  = eye - obj.position
                pn /= max(np.linalg.norm(pn), 1e-6)
            else:
                pn = axes[{'ring_x':0,'ring_y':1,'ring_z':2}[hit]]
            self._drag_plane_normal = pn
            t = self.ray_plane_intersection(ray_o, ray_d, obj.position, pn)
            if t is not None:
                pt = ray_o + t * ray_d
                v  = pt - obj.position
                up  = np.array([0,0,1], dtype=np.float32)
                if abs(np.dot(pn, up)) > 0.99: up = np.array([0,1,0], dtype=np.float32)
                px  = np.cross(up, pn);  px /= np.linalg.norm(px)
                py  = np.cross(pn, px)
                self._drag_angle_start = np.arctan2(np.dot(v, py), np.dot(v, px))

        return True

    def update_drag(self, mx: float, my: float, obj, camera,
                    width: float, height: float):
        if self.active_handle is None or obj is None:
            return

        aspect   = width / max(height, 1.0)
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect)
        vp_mat   = pyrr.matrix44.multiply(view_mat, proj_mat)

        _, r_mat, _ = self.get_gizmo_transforms(obj)
        axes         = self.get_axes_vectors(r_mat)
        ray_o, ray_d = self.get_ray_from_mouse(mx, my, vp_mat, width, height)
        h = self.active_handle

        if self.operation == "translate":
            if h.startswith('axis_'):
                idx   = {'axis_x':0,'axis_y':1,'axis_z':2}[h]
                t_now = self.closest_point_on_line(ray_o, ray_d, self._drag_start_pos, axes[idx])
                delta = t_now - self._drag_t_start
                if self.use_snapping:
                    delta = round(delta / self.snap_translation) * self.snap_translation
                obj.position = self._drag_start_pos + axes[idx] * delta
            elif h.startswith('plane_'):
                n_idx = {'plane_yz':0,'plane_xz':1,'plane_xy':2}[h]
                t = self.ray_plane_intersection(ray_o, ray_d, self._drag_start_pos, axes[n_idx])
                if t is not None and self._drag_plane_start is not None:
                    dp = ray_o + t * ray_d - self._drag_plane_start
                    if self.use_snapping:
                        dp = np.round(dp / self.snap_translation) * self.snap_translation
                    obj.position = self._drag_start_pos + dp

        elif self.operation == "scale":
            sf   = self.get_scale_factor(camera)
            vp2  = vp_mat
            _, _, gmat = self.get_gizmo_transforms(obj)
            c_s  = self.project_local_to_screen([0,0,0], gmat, sf, vp2, width, height)
            dm   = np.array([mx, my]) - self._drag_start_mouse

            if h == 'center':
                proj = dm[0] - dm[1]
                mul  = max(0.01, 1.0 + proj * 0.01)
                if self.use_snapping:
                    mul = max(0.01, round(mul / self.snap_scale) * self.snap_scale)
                obj.scale = self._drag_start_scale * mul

            elif h.startswith('axis_'):
                idx     = {'axis_x':0,'axis_y':1,'axis_z':2}[h]
                tip_lp  = [[1,0,0],[0,1,0],[0,0,1]][idx]
                tip_s   = self.project_local_to_screen(tip_lp, gmat, sf, vp2, width, height)
                if tip_s is not None and c_s is not None:
                    ad = tip_s - c_s
                    ad_len = np.linalg.norm(ad)
                    if ad_len > 0:
                        proj = np.dot(dm, ad / ad_len)
                        mul  = max(0.01, 1.0 + proj * 0.01)
                        if self.use_snapping:
                            mul = max(0.01, round(mul / self.snap_scale) * self.snap_scale)
                        obj.scale = np.copy(self._drag_start_scale)
                        obj.scale[idx] = self._drag_start_scale[idx] * mul

        elif self.operation == "rotate":
            pn = self._drag_plane_normal
            if pn is None:
                return
            t = self.ray_plane_intersection(ray_o, ray_d, self._drag_start_pos, pn)
            if t is None:
                return
            pt = ray_o + t * ray_d
            v  = pt - self._drag_start_pos
            up  = np.array([0,0,1], dtype=np.float32)
            if abs(np.dot(pn, up)) > 0.99: up = np.array([0,1,0], dtype=np.float32)
            px  = np.cross(up, pn);  px /= np.linalg.norm(px)
            py  = np.cross(pn, px)

            cur   = np.arctan2(np.dot(v, py), np.dot(v, px))
            delta = cur - self._drag_angle_start
            deg   = np.degrees(delta)
            if self.use_snapping:
                deg = round(deg / self.snap_rotation) * self.snap_rotation
            delta = np.radians(deg)

            r_delta = trimesh.transformations.rotation_matrix(delta, pn)
            rx = np.radians(-self._drag_start_rot[0])
            ry = np.radians(-self._drag_start_rot[1])
            rz = np.radians(-self._drag_start_rot[2])
            r_start = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')

            if self.space == "local":
                r_new = r_start @ r_delta
            else:
                r_new = r_delta @ r_start

            ex, ey, ez = trimesh.transformations.euler_from_matrix(r_new, 'sxyz')
            obj.rotation = -np.degrees([ex, ey, ez]).astype(np.float32)

    def end_drag(self):
        self.active_handle     = None
        self._drag_plane_start = None
        self._drag_plane_normal = None

    # ------------------------------------------------------- ray utilities

    def get_ray_from_mouse(self, mx, my, vp_mat, width, height):
        ndc_x =  2.0 * mx / width  - 1.0
        ndc_y =  1.0 - 2.0 * my / height

        inv_vp = pyrr.matrix44.inverse(vp_mat)
        near = pyrr.matrix44.apply_to_vector(inv_vp, np.array([ndc_x, ndc_y, -1, 1], dtype=np.float32))
        far  = pyrr.matrix44.apply_to_vector(inv_vp, np.array([ndc_x, ndc_y,  1, 1], dtype=np.float32))
        near = near[:3] / near[3]
        far  = far[:3]  / far[3]
        d    = far - near
        d   /= np.linalg.norm(d)
        return near, d

    def closest_point_on_line(self, ray_o, ray_d, line_p, line_d) -> float:
        w0 = line_p - ray_o
        a  = np.dot(line_d, line_d)
        b  = np.dot(line_d, ray_d)
        c  = np.dot(ray_d,  ray_d)
        d  = np.dot(line_d, w0)
        e  = np.dot(ray_d,  w0)
        denom = a * c - b * b
        if abs(denom) < 1e-8:
            return 0.0
        return (b * e - c * d) / denom

    def ray_plane_intersection(self, ray_o, ray_d, plane_p, plane_n):
        denom = np.dot(ray_d, plane_n)
        if abs(denom) < 1e-6:
            return None
        t = np.dot(plane_p - ray_o, plane_n) / denom
        return t if t >= 0 else None
