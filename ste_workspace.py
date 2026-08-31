"""
Spatial Texture Engine (STE) - Unified Alignment Workspace
==========================================================

This module implements the unified, CloudCompare-style Spatial Texture Engine
editor experience for ProximaP using the native Mesh Editor Viewport (MeshEditorViewport).

Key Architecture:
-----------------
1. Native Mesh Editor Viewport:
   Reuses ProximaP's MeshEditorViewport (OpenGL 3D canvas with infinite grid floor,
   red/green axis lines, NavGizmo orientation widget, and MeshInspector overlay).
2. Unified Shared Coordinate Space:
   Photogrammetry and LiDAR datasets co-exist in the same shared scene graph.
3. Independent Dataset Controls:
   Independent visibility toggles for Photogrammetry (textured mesh) and LiDAR
   (points, reconstructed surface, and baked surface).
4. In-Viewport Control-Point Placement:
   Place and adjust paired markers on Photogrammetry and LiDAR geometry directly
   within the same shared 3D viewport.
5. Non-Destructive Alignment Preview:
   Visually applies the scale-aware Horn / ICP transformation matrix to the LiDAR
   model matrix without mutating stored native LiDAR coordinates.
6. Production Pipeline Integration:
   Seamlessly integrates with tested backend services: CCCoreLib registration,
   alignment quality gate, LiDAR surface reconstruction, target UV parameterization,
   texture projection, and production texture baking.
"""

import os
import sys
import time
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Union, Callable

import numpy as np
import open3d as o3d
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer, QSize, QPoint, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QIcon, QCursor, QMouseEvent, QKeyEvent, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFrame, QScrollArea, QFileDialog,
    QMessageBox, QButtonGroup, QRadioButton, QCheckBox
)

import OpenGL.GL as gl
import pyrr
import trimesh

from mesh_editor.viewport import MeshEditorViewport, GizmoPivot
from mesh_editor.scene import Camera, Scene, Object, Mesh

from ste_cccorelib_registration import is_cccorelib_available, register_point_pairs, refine_icp
from ste_alignment import (
    STEControlPointManager,
    STEAlignmentService,
    STEAlignmentResult,
    STEICPRefinementService,
    STEICPRefinementSettings,
    STEICPRefinementResult
)
from ste_alignment_quality import (
    STEAlignmentQualityGate,
    STEAlignmentQualityReport,
    AlignmentQualityRating,
    TextureTransferReadiness
)
from ste_texture_projection import (
    STETextureProjectionService,
    STETextureProjectionSettings,
    STETextureProjectionResult
)
from ste_lidar_uv import (
    STELiDARUVService,
    STELiDARUVSettings,
    STELiDARUVResult
)
from ste_texture_baking import (
    STETextureBakingService,
    STETextureBakingWorker,
    TextureBakeResult
)


class STEWorkflowState(str, Enum):
    NO_DATA = "NO_DATA"
    PHOTOGRAMMETRY_READY = "PHOTOGRAMMETRY_READY"
    LIDAR_READY = "LIDAR_READY"
    CONTROL_POINTS_READY = "CONTROL_POINTS_READY"
    ALIGNMENT_READY = "ALIGNMENT_READY"
    SURFACE_READY = "SURFACE_READY"
    UV_READY = "UV_READY"
    PROJECTION_READY = "PROJECTION_READY"
    BAKE_READY = "BAKE_READY"
    BAKED = "BAKED"


class STELiDARRepresentation(str, Enum):
    POINTS = "POINTS"
    SURFACE = "SURFACE"
    BAKED = "BAKED"


class STESurfaceReconstructionWorker(QThread):
    """Asynchronous worker for LiDAR surface reconstruction."""
    progress = Signal(float, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, points: np.ndarray, depth: int = 8, parent=None):
        super().__init__(parent)
        self.points = points.copy()
        self.depth = depth

    def run(self):
        try:
            self.progress.emit(0.1, "Analyzing LiDAR point distribution...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)

            self.progress.emit(0.3, "Estimating surface normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

            self.progress.emit(0.6, f"Running Screened Poisson Reconstruction (depth={self.depth})...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.depth
            )

            self.progress.emit(0.9, "Filtering surface density artifacts...")
            densities_arr = np.asarray(densities)
            density_threshold = np.quantile(densities_arr, 0.05)
            vertices_to_remove = densities_arr < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            self.progress.emit(1.0, "Surface Reconstruction Complete")
            self.finished.emit(mesh)
        except Exception as e:
            self.error.emit(str(e))


class STELiDARObject(Object):
    """
    Scene object representing LiDAR representations with dynamic alignment preview matrix.
    """
    def __init__(self, name: str, mesh: Mesh):
        super().__init__(name, mesh)
        self.preview_matrix: Optional[np.ndarray] = None  # 4x4 row-major transformation

    def get_model_matrix(self) -> np.ndarray:
        if self.preview_matrix is not None:
            # Row-major matrix returned transposed for OpenGL column-major convention
            return self.preview_matrix.T.astype(np.float32)
        return super().get_model_matrix()


def _create_mesh_from_data(
    verts: np.ndarray,
    tris: np.ndarray,
    texcoords: Optional[np.ndarray] = None,
    texture_img: Optional[Union[np.ndarray, Image.Image]] = None
) -> Mesh:
    """Helper to construct a MeshEditor Mesh container from numpy arrays."""
    vertices = np.asarray(verts, dtype=np.float32)
    indices = np.asarray(tris, dtype=np.uint32).ravel()

    # Compute normals using trimesh
    import trimesh
    tm = trimesh.Trimesh(vertices=vertices, faces=np.asarray(tris, dtype=np.uint32), process=False)
    normals = np.asarray(tm.vertex_normals, dtype=np.float32)

    local_min = np.min(vertices, axis=0) if len(vertices) > 0 else np.zeros(3, dtype=np.float32)
    local_max = np.max(vertices, axis=0) if len(vertices) > 0 else np.zeros(3, dtype=np.float32)

    pil_img = None
    if texture_img is not None:
        if isinstance(texture_img, Image.Image):
            pil_img = texture_img
        elif isinstance(texture_img, np.ndarray):
            pil_img = Image.fromarray(texture_img)

    uv_arr = None
    if texcoords is not None and len(texcoords) > 0:
        if len(texcoords) == len(vertices):
            uv_arr = np.asarray(texcoords, dtype=np.float32)

    return Mesh(
        vertices=vertices,
        indices=indices,
        normals=normals,
        local_aabb_min=local_min,
        local_aabb_max=local_max,
        texcoords=uv_arr,
        texture_data=pil_img
    )


class STEUnifiedAlignmentViewport(MeshEditorViewport):
    """
    Unified 3D Alignment Viewport built directly on top of the Mesh Editor Viewport.
    Integrates ProximaP's Blender-style infinite grid, red/green axis lines, NavGizmo
    orientation overlay, MeshInspector card, and shared OpenGL scene rendering.
    """
    point_picked = Signal(str, object)  # Emits (target_mode 'photo'|'lidar', (x, y, z))
    staging_tool_changed = Signal(str)  # Emits active gizmo tool: 'translate', 'rotate', 'scale', 'none'
    staging_target_changed = Signal(str)  # Emits active staging target: 'lidar' or 'photo'

    def __init__(self, parent=None):
        super().__init__(parent)
        # STE Mode: Disable Mesh Editor scene object selection gizmos; use STE dedicated Staging Gizmo
        self.transform_enabled = False
        self.enable_gizmo = False
        self.gizmo.space = "global"  # Force fixed Global Space transforms for STE staging

        self.picking_target: Optional[str] = None  # 'photo', 'lidar', or None
        self.active_cp_id: Optional[str] = None

        self._preview_matrix = np.eye(4, dtype=np.float64)
        self._preview_active: bool = False

        # Interactive Staging Transform (Non-destructive interactive display transform)
        self.staging_enabled: bool = False
        self.staging_target: str = "lidar"  # 'lidar' or 'photo'

        # LiDAR Staging State
        self._lidar_staging_center: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._lidar_staging_delta_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._lidar_staging_rotation: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # degrees (rx, ry, rz)
        self._lidar_staging_scale: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Photogrammetry Staging State
        self._photo_staging_center: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._photo_staging_delta_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._photo_staging_rotation: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # degrees (rx, ry, rz)
        self._photo_staging_scale: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self._drag_pivot = None
        self._drag_start_pivot_pos = None
        self._drag_start_delta_pos = None
        self._drag_start_pivot_scale = None
        self._drag_start_scale = None
        self._drag_start_pivot_r_mat = None
        self._drag_start_rotation = None

        # Raw Source Data (Native Coordinates Preserved)
        self._photo_verts: Optional[np.ndarray] = None
        self._photo_faces: Optional[np.ndarray] = None
        self._photo_texture: Optional[np.ndarray] = None
        self._photo_alignment_cloud: Optional[np.ndarray] = None

        self._lidar_points: Optional[np.ndarray] = None
        self._lidar_surface_verts: Optional[np.ndarray] = None
        self._lidar_surface_faces: Optional[np.ndarray] = None
        self._lidar_baked_verts: Optional[np.ndarray] = None
        self._lidar_baked_faces: Optional[np.ndarray] = None
        self._lidar_baked_texture: Optional[np.ndarray] = None

        # Visibility Flags & Display Modes
        self.photo_visible = True
        self.lidar_visible = True
        self.lidar_display_mode = STELiDARRepresentation.POINTS

        self._photo_markers: Dict[str, np.ndarray] = {}
        self._lidar_markers: Dict[str, np.ndarray] = {}

        # Scene Objects
        self.obj_photo: Optional[Object] = None
        self.obj_lidar_surf: Optional[STELiDARObject] = None
        self.obj_lidar_baked: Optional[STELiDARObject] = None

        # Point Cloud Buffers (for GL_POINTS rendering with RGB point colors)
        self._photo_pcd_vao = None
        self._photo_pcd_vbo = None
        self._photo_pcd_cbo = None
        self._photo_pcd_count = 0
        self._photo_points: Optional[np.ndarray] = None
        self._photo_colors: Optional[np.ndarray] = None

        self._pcd_vao = None
        self._pcd_vbo = None
        self._pcd_cbo = None
        self._pcd_count = 0
        self._lidar_colors: Optional[np.ndarray] = None
        self._pcd_program = None

        self._marker_vao = None
        self._marker_vbo = None

    class _VisualProxy:
        def __init__(self, getter):
            self._getter = getter
        @property
        def visible(self) -> bool:
            return bool(self._getter())

    @property
    def visual_photo_cloud(self):
        return self._VisualProxy(lambda: self.photo_visible and (self._photo_points is not None and len(self._photo_points) > 0))

    @property
    def visual_photo_mesh(self):
        return self._VisualProxy(lambda: False)

    @property
    def visual_lidar_points(self):
        return self._VisualProxy(lambda: self.lidar_visible and (self.lidar_display_mode == STELiDARRepresentation.POINTS))

    @property
    def visual_lidar_surface(self):
        return self._VisualProxy(lambda: self.lidar_visible and (self.lidar_display_mode == STELiDARRepresentation.SURFACE))

    @property
    def visual_lidar_baked(self):
        return self._VisualProxy(lambda: self.lidar_visible and (self.lidar_display_mode == STELiDARRepresentation.BAKED))

    def set_picking_mode(self, target: Optional[str], cp_id: Optional[str] = None):
        """Set active 3D point picking mode: 'photo', 'lidar', or None."""
        self.picking_target = target
        self.active_cp_id = cp_id
        if target is not None:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _safe_setup_mesh(self, gl_mesh: Mesh):
        if self.isValid():
            try:
                self.makeCurrent()
                gl_mesh.setup_buffers()
                self.doneCurrent()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Data Setters (Native Coordinates Preserved)
    # -------------------------------------------------------------------------
    def set_photogrammetry_data(self, mesh: o3d.geometry.TriangleMesh, texture: Optional[np.ndarray] = None, point_cloud: Optional[np.ndarray] = None, colors: Optional[np.ndarray] = None):
        self._photo_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
        self._photo_faces = np.asarray(mesh.triangles, dtype=np.int32).copy()
        self._photo_texture = texture
        if len(self._photo_verts) > 0:
            self._photo_staging_center = np.mean(self._photo_verts, axis=0).astype(np.float32)

        pts = point_cloud if point_cloud is not None else self._photo_verts
        self.set_photogrammetry_point_cloud(pts, colors=colors)

    def set_photogrammetry_point_cloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """Set or update the dense photogrammetry alignment point cloud."""
        self._photo_points = np.ascontiguousarray(points, dtype=np.float64)
        self._photo_alignment_cloud = self._photo_points
        self._photo_pcd_count = len(self._photo_points)
        if len(self._photo_points) > 0:
            self._photo_staging_center = np.mean(self._photo_points, axis=0).astype(np.float32)
        if colors is not None and len(colors) == len(self._photo_points):
            self._photo_colors = np.ascontiguousarray(colors, dtype=np.float32)
        else:
            self._photo_colors = np.full((len(self._photo_points), 3), 0.85, dtype=np.float32)

        if self.isValid():
            try:
                self.makeCurrent()
                self._setup_photo_pcd_gl_buffers()
                self.doneCurrent()
            except Exception:
                pass

        self._apply_render()
        if self._lidar_points is None:
            self.reset_view()

    def _setup_photo_pcd_gl_buffers(self):
        if self._photo_points is None or self._photo_pcd_count == 0:
            return
        if self._photo_pcd_vao is None:
            self._photo_pcd_vao = gl.glGenVertexArrays(1)
            self._photo_pcd_vbo = gl.glGenBuffers(1)
            self._photo_pcd_cbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._photo_pcd_vao)

        # Positions (layout = 0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._photo_pcd_vbo)
        pts_float32 = self._photo_points.astype(np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, pts_float32.nbytes, pts_float32, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # RGB Colors (layout = 1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._photo_pcd_cbo)
        if self._photo_colors is None or len(self._photo_colors) != self._photo_pcd_count:
            self._photo_colors = np.full((self._photo_pcd_count, 3), 0.85, dtype=np.float32)
        colors_float32 = self._photo_colors.astype(np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors_float32.nbytes, colors_float32, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindVertexArray(0)

    def set_lidar_points(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        self._lidar_points = np.ascontiguousarray(points, dtype=np.float64)
        self._pcd_count = len(self._lidar_points)
        if len(self._lidar_points) > 0:
            self._lidar_staging_center = np.mean(self._lidar_points, axis=0).astype(np.float32)
        if colors is not None and len(colors) == len(self._lidar_points):
            self._lidar_colors = np.ascontiguousarray(colors, dtype=np.float32)
        else:
            self._lidar_colors = np.full((len(self._lidar_points), 3), 0.85, dtype=np.float32)
        self.lidar_display_mode = STELiDARRepresentation.POINTS

        if self.isValid():
            try:
                self.makeCurrent()
                self._setup_pcd_gl_buffers()
                self.doneCurrent()
            except Exception:
                pass

        self._apply_render()
        if self._photo_points is None and self._photo_verts is None:
            self.reset_view()

    def _setup_pcd_gl_buffers(self):
        if self._lidar_points is None or self._pcd_count == 0:
            return
        if self._pcd_vao is None:
            self._pcd_vao = gl.glGenVertexArrays(1)
            self._pcd_vbo = gl.glGenBuffers(1)
            self._pcd_cbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._pcd_vao)

        # Positions (layout = 0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._pcd_vbo)
        pts_float32 = self._lidar_points.astype(np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, pts_float32.nbytes, pts_float32, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # RGB Colors (layout = 1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._pcd_cbo)
        if self._lidar_colors is None or len(self._lidar_colors) != self._pcd_count:
            self._lidar_colors = np.full((self._pcd_count, 3), 0.85, dtype=np.float32)
        colors_float32 = self._lidar_colors.astype(np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors_float32.nbytes, colors_float32, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindVertexArray(0)

    def set_lidar_surface(self, mesh: o3d.geometry.TriangleMesh):
        self._lidar_surface_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
        self._lidar_surface_faces = np.asarray(mesh.triangles, dtype=np.int32).copy()
        if len(self._lidar_surface_verts) > 0 and self._lidar_points is None:
            self._lidar_staging_center = np.mean(self._lidar_surface_verts, axis=0).astype(np.float32)

        gl_mesh = _create_mesh_from_data(self._lidar_surface_verts, self._lidar_surface_faces)
        self._safe_setup_mesh(gl_mesh)

        if self.obj_lidar_surf in self.scene.objects:
            self.scene.remove_object(self.obj_lidar_surf)

        self.obj_lidar_surf = STELiDARObject("LiDAR_Surface", gl_mesh)
        active_mat = self.get_active_lidar_transform()
        self.obj_lidar_surf.preview_matrix = active_mat if not np.allclose(active_mat, np.eye(4)) else None
        self.scene.add_object(self.obj_lidar_surf)

        self.lidar_display_mode = STELiDARRepresentation.SURFACE
        self._apply_render()

    def set_lidar_baked(self, mesh: o3d.geometry.TriangleMesh, texture: Optional[np.ndarray] = None):
        self._lidar_baked_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
        self._lidar_baked_faces = np.asarray(mesh.triangles, dtype=np.int32).copy()
        self._lidar_baked_texture = texture

        gl_mesh = _create_mesh_from_data(self._lidar_baked_verts, self._lidar_baked_faces, texture_img=texture)
        self._safe_setup_mesh(gl_mesh)

        if self.obj_lidar_baked in self.scene.objects:
            self.scene.remove_object(self.obj_lidar_baked)

        self.obj_lidar_baked = STELiDARObject("LiDAR_Baked", gl_mesh)
        active_mat = self.get_active_lidar_transform()
        self.obj_lidar_baked.preview_matrix = active_mat if not np.allclose(active_mat, np.eye(4)) else None
        self.scene.add_object(self.obj_lidar_baked)

        self.lidar_display_mode = STELiDARRepresentation.BAKED
        self._apply_render()

    def set_control_markers(self, photo_markers: Dict[str, np.ndarray], lidar_markers: Dict[str, np.ndarray], active_id: Optional[str] = None):
        self._photo_markers = {k: np.array(v, dtype=np.float64) for k, v in photo_markers.items()}
        self._lidar_markers = {k: np.array(v, dtype=np.float64) for k, v in lidar_markers.items()}
        self.active_cp_id = active_id
        self.update()

    # -------------------------------------------------------------------------
    # Interactive Staging Transformation & Non-Destructive Alignment Preview
    # -------------------------------------------------------------------------
    def get_staging_matrix(self, target: Optional[str] = None) -> np.ndarray:
        """
        Returns 4x4 staging transform matrix relative to initial centroid of the target dataset.
        Allows interactive visual orientation/scaling while guaranteeing zero loss
        or modification to the underlying native coordinates.
        """
        tgt = target if target is not None else self.staging_target
        if tgt == "photo":
            if self._photo_points is None and self._photo_verts is None:
                return np.eye(4, dtype=np.float64)
            C = self._photo_staging_center.astype(np.float64)
            delta_t = self._photo_staging_delta_pos.astype(np.float64)
            s = self._photo_staging_scale.astype(np.float64)
            rot = self._photo_staging_rotation
        else:
            if self._lidar_points is None and self._lidar_surface_verts is None:
                return np.eye(4, dtype=np.float64)
            C = self._lidar_staging_center.astype(np.float64)
            delta_t = self._lidar_staging_delta_pos.astype(np.float64)
            s = self._lidar_staging_scale.astype(np.float64)
            rot = self._lidar_staging_rotation

        # T_neg_center (translate to origin)
        T_neg = np.eye(4, dtype=np.float64)
        T_neg[:3, 3] = -C

        # Scale matrix
        S = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float64)

        # Rotation matrix (Euler XYZ, inverted sign for OpenGL camera orientation convention)
        rx = np.radians(-float(rot[0]))
        ry = np.radians(-float(rot[1]))
        rz = np.radians(-float(rot[2]))
        R = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')

        # T_pos_center (translate back from origin plus staging translation offset)
        T_pos = np.eye(4, dtype=np.float64)
        T_pos[:3, 3] = C + delta_t

        return T_pos @ R @ S @ T_neg

    def get_active_lidar_transform(self) -> np.ndarray:
        """Returns active preview matrix if alignment preview is active, else LiDAR staging matrix."""
        if self._preview_active and not np.allclose(self._preview_matrix, np.eye(4)):
            return self._preview_matrix
        return self.get_staging_matrix("lidar")

    def get_active_photo_transform(self) -> np.ndarray:
        """Returns active Photogrammetry staging matrix."""
        return self.get_staging_matrix("photo")

    def set_staging_tool(self, tool: str):
        """Set active staging gizmo tool: 'translate', 'rotate', 'scale', or 'none'."""
        if tool in ("translate", "rotate", "scale"):
            self.staging_enabled = True
            self.enable_gizmo = True
            self.gizmo.operation = tool
            self.staging_tool_changed.emit(tool)
        else:
            self.staging_enabled = False
            self.enable_gizmo = False
            self.gizmo.operation = "translate"
            self.staging_tool_changed.emit("none")
        self.update()

    def set_staging_target(self, target: str):
        """Set active staging target dataset: 'lidar' or 'photo'."""
        if target in ("lidar", "photo"):
            self.staging_target = target
            self.staging_target_changed.emit(target)
            self.update()

    def reset_staging_transform(self, target: Optional[str] = None):
        """Resets interactive staging transformation back to identity for target or both."""
        tgt = target if target is not None else self.staging_target
        if tgt in ("lidar", "all"):
            self._lidar_staging_delta_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._lidar_staging_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._lidar_staging_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if tgt in ("photo", "all"):
            self._photo_staging_delta_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._photo_staging_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._photo_staging_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._apply_staging_update()
        self.update()

    def _apply_staging_update(self):
        active_mat = self.get_active_lidar_transform()
        mat_or_none = active_mat if not np.allclose(active_mat, np.eye(4)) else None
        if self.obj_lidar_surf:
            self.obj_lidar_surf.preview_matrix = mat_or_none
        if self.obj_lidar_baked:
            self.obj_lidar_baked.preview_matrix = mat_or_none
        self.update()

    def _get_gizmo_pivot(self):
        if not self.staging_enabled:
            return None
        if self.staging_target == "photo":
            if self._photo_points is None and self._photo_verts is None:
                return None
            pivot_pos = (self._photo_staging_center + self._photo_staging_delta_pos).astype(np.float32)
            return GizmoPivot(pivot_pos, self._photo_staging_rotation.copy(), self._photo_staging_scale.copy())
        else:
            if self._lidar_points is None and self._lidar_surface_verts is None:
                return None
            pivot_pos = (self._lidar_staging_center + self._lidar_staging_delta_pos).astype(np.float32)
            return GizmoPivot(pivot_pos, self._lidar_staging_rotation.copy(), self._lidar_staging_scale.copy())

    def set_preview_transform(self, matrix_4x4: np.ndarray):
        """Visually transforms the LiDAR representation into photogrammetry space."""
        self._preview_matrix = np.ascontiguousarray(matrix_4x4, dtype=np.float64)
        self._preview_active = True
        self._apply_staging_update()

    def reset_preview_transform(self):
        """Reverts visual rendering immediately to native LiDAR coordinate system (or staging)."""
        self._preview_matrix = np.eye(4, dtype=np.float64)
        self._preview_active = False
        self._apply_staging_update()

    def _apply_render(self):
        """Synchronize object visibility with active toggles."""
        # Note: Textured photogrammetry mesh is preserved internally as texture source and NOT rendered in alignment view
        if self.obj_photo and self.obj_photo in self.scene.objects:
            self.scene.remove_object(self.obj_photo)

        show_lidar_surf = self.lidar_visible and (self.lidar_display_mode == STELiDARRepresentation.SURFACE)
        show_lidar_bake = self.lidar_visible and (self.lidar_display_mode == STELiDARRepresentation.BAKED)

        if self.obj_lidar_surf:
            self.obj_lidar_surf.visible = show_lidar_surf
        if self.obj_lidar_baked:
            self.obj_lidar_baked.visible = show_lidar_bake

        for obj in [self.obj_lidar_surf, self.obj_lidar_baked]:
            if obj is not None:
                if obj.visible and obj not in self.scene.objects:
                    self.scene.add_object(obj)
                elif not obj.visible and obj in self.scene.objects:
                    self.scene.remove_object(obj)

        self.update()

    def reset_view(self):
        """Fit camera to all active visible geometry."""
        all_pts = []
        if self.photo_visible:
            if self._photo_points is not None and len(self._photo_points) > 0:
                photo_mat = self.get_active_photo_transform()
                if not np.allclose(photo_mat, np.eye(4)):
                    pts_h = np.hstack([self._photo_points, np.ones((len(self._photo_points), 1))])
                    all_pts.append((pts_h @ photo_mat.T)[:, :3])
                else:
                    all_pts.append(self._photo_points)
            elif self._photo_alignment_cloud is not None and len(self._photo_alignment_cloud) > 0:
                all_pts.append(self._photo_alignment_cloud)
            elif self._photo_verts is not None and len(self._photo_verts) > 0:
                all_pts.append(self._photo_verts)

        if self.lidar_visible:
            if self._lidar_points is not None and len(self._lidar_points) > 0:
                active_mat = self.get_active_lidar_transform()
                if not np.allclose(active_mat, np.eye(4)):
                    pts_h = np.hstack([self._lidar_points, np.ones((len(self._lidar_points), 1))])
                    all_pts.append((pts_h @ active_mat.T)[:, :3])
                else:
                    all_pts.append(self._lidar_points)

        if all_pts:
            combined = np.vstack(all_pts)
            center = np.mean(combined, axis=0)
            extent = np.max(combined, axis=0) - np.min(combined, axis=0)
            max_ext = max(1.0, float(np.max(extent)))
            self.camera.target = np.array(center, dtype=np.float32)
            self.camera.distance = max_ext * 2.2
            self.update()

    # -------------------------------------------------------------------------
    # OpenGL Rendering Integration (Mesh Editor Viewport + STE Overlays)
    # -------------------------------------------------------------------------
    def paintGL(self):
        # 1. Ensure any uninitialized mesh buffers are set up
        for obj in self.scene.objects:
            if obj.mesh and obj.mesh.vao is None:
                try:
                    obj.mesh.setup_buffers()
                except Exception:
                    pass

        # 2. Execute Mesh Editor Viewport rendering (infinite grid, lighting, shaders, gizmos)
        super().paintGL()

        # 3. Render Photogrammetry Point Cloud (when photo_visible is True)
        if self.photo_visible and self._photo_points is not None and len(self._photo_points) > 0:
            self._render_photogrammetry_point_cloud()

        # 4. Render LiDAR Point Cloud (when in POINTS mode and lidar_visible is True)
        show_lidar_pts = self.lidar_visible and (self.lidar_display_mode == STELiDARRepresentation.POINTS)
        if show_lidar_pts and self._lidar_points is not None and len(self._lidar_points) > 0:
            self._render_lidar_point_cloud()

        # 5. Render Control Point Markers Overlays
        self._render_control_markers()

        # 6. Render Staging Gizmo Overlay (when staging tool is active and preview is not active)
        if self.staging_enabled and not self._preview_active:
            pivot = self._get_gizmo_pivot()
            if pivot is not None:
                w_logical = self.width()
                h_logical = self.height()
                self.gizmo.draw(pivot, self.camera, w_logical, h_logical)

    def _ensure_pcd_program(self):
        if self._pcd_program is not None:
            return
        vert_src = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform float pointSize;

out vec3 vColor;

void main() {
    vColor = aColor;
    gl_PointSize = (pointSize > 0.0) ? pointSize : 4.0;
    gl_Position = proj * view * model * vec4(aPos, 1.0);
}
"""
        frag_src = """#version 330 core
in vec3 vColor;
out vec4 FragColor;

uniform vec4 overrideColor;
uniform bool useOverrideColor;

void main() {
    if (useOverrideColor) {
        FragColor = overrideColor;
    } else {
        FragColor = vec4(vColor, 1.0);
    }
}
"""
        def _compile(src, kind):
            s = gl.glCreateShader(kind)
            gl.glShaderSource(s, src)
            gl.glCompileShader(s)
            return s

        vs = _compile(vert_src, gl.GL_VERTEX_SHADER)
        fs = _compile(frag_src, gl.GL_FRAGMENT_SHADER)
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        self._pcd_program = prog

    def _render_photogrammetry_point_cloud(self):
        try:
            self._ensure_pcd_program()
            if self._photo_pcd_vao is None:
                self._setup_photo_pcd_gl_buffers()

            w_logical = self.width()
            h_logical = self.height()
            aspect = w_logical / max(h_logical, 1.0)
            view_matrix = self.camera.get_view_matrix()
            proj_matrix = self.camera.get_projection_matrix(aspect)

            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glUseProgram(self._pcd_program)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "view"), 1, gl.GL_FALSE, view_matrix)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "proj"), 1, gl.GL_FALSE, proj_matrix)

            # Model matrix (with active Photogrammetry staging transform)
            active_photo_mat = self.get_active_photo_transform()
            model_matrix = active_photo_mat.T.astype(np.float32) if not np.allclose(active_photo_mat, np.eye(4)) else np.eye(4, dtype=np.float32)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "model"), 1, gl.GL_FALSE, model_matrix)
            gl.glUniform1i(gl.glGetUniformLocation(self._pcd_program, "useOverrideColor"), 0)
            gl.glUniform1f(gl.glGetUniformLocation(self._pcd_program, "pointSize"), 4.0)

            gl.glBindVertexArray(self._photo_pcd_vao)
            gl.glDrawArrays(gl.GL_POINTS, 0, self._photo_pcd_count)
            gl.glBindVertexArray(0)
        except Exception:
            pass

    def _render_lidar_point_cloud(self):
        try:
            self._ensure_pcd_program()
            if self._pcd_vao is None:
                self._setup_pcd_gl_buffers()

            w_logical = self.width()
            h_logical = self.height()
            aspect = w_logical / max(h_logical, 1.0)
            view_matrix = self.camera.get_view_matrix()
            proj_matrix = self.camera.get_projection_matrix(aspect)

            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glUseProgram(self._pcd_program)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "view"), 1, gl.GL_FALSE, view_matrix)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "proj"), 1, gl.GL_FALSE, proj_matrix)

            # Model matrix (with active staging or alignment preview)
            active_mat = self.get_active_lidar_transform()
            model_matrix = active_mat.T.astype(np.float32) if not np.allclose(active_mat, np.eye(4)) else np.eye(4, dtype=np.float32)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "model"), 1, gl.GL_FALSE, model_matrix)
            gl.glUniform1i(gl.glGetUniformLocation(self._pcd_program, "useOverrideColor"), 0)
            gl.glUniform1f(gl.glGetUniformLocation(self._pcd_program, "pointSize"), 4.0)

            gl.glBindVertexArray(self._pcd_vao)
            gl.glDrawArrays(gl.GL_POINTS, 0, len(self._lidar_points))
            gl.glBindVertexArray(0)
        except Exception:
            pass

    def _setup_marker_gl_buffers(self):
        if self._marker_vao is None:
            self._marker_vao = gl.glGenVertexArrays(1)
            self._marker_vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._marker_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._marker_vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindVertexArray(0)

    def _render_control_markers(self):
        """Render Photogrammetry and LiDAR 3D markers with distinct OpenGL 3D pins."""
        if not self._photo_markers and not self._lidar_markers:
            return

        try:
            self._ensure_pcd_program()
            if self._marker_vao is None:
                self._setup_marker_gl_buffers()

            w_logical = self.width()
            h_logical = self.height()
            aspect = w_logical / max(h_logical, 1.0)
            view_matrix = self.camera.get_view_matrix()
            proj_matrix = self.camera.get_projection_matrix(aspect)

            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            gl.glDisable(gl.GL_DEPTH_TEST)

            gl.glUseProgram(self._pcd_program)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "view"), 1, gl.GL_FALSE, view_matrix)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "proj"), 1, gl.GL_FALSE, proj_matrix)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._pcd_program, "model"), 1, gl.GL_FALSE, np.eye(4, dtype=np.float32))
            gl.glUniform1i(gl.glGetUniformLocation(self._pcd_program, "useOverrideColor"), 1)

            gl.glBindVertexArray(self._marker_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._marker_vbo)

            # 1. Photogrammetry 3D Markers (White / Silver with outer halo, transformed by Photo staging)
            photo_mat = self.get_active_photo_transform()
            has_photo_trans = not np.allclose(photo_mat, np.eye(4))
            for name, pos in self._photo_markers.items():
                is_active = (self.active_cp_id == name)
                pos_3d = np.array(pos, dtype=np.float64)
                if has_photo_trans:
                    pos_3d = (photo_mat @ np.append(pos_3d, 1.0))[:3]
                pt = np.array(pos_3d, dtype=np.float32)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, pt.nbytes, pt, gl.GL_DYNAMIC_DRAW)

                # Outer halo
                gl.glUniform1f(gl.glGetUniformLocation(self._pcd_program, "pointSize"), 20.0 if is_active else 14.0)
                gl.glUniform4f(gl.glGetUniformLocation(self._pcd_program, "overrideColor"), 0.0, 0.9, 0.46, 1.0 if is_active else 0.4)
                gl.glDrawArrays(gl.GL_POINTS, 0, 1)

                # Inner core
                gl.glUniform1f(gl.glGetUniformLocation(self._pcd_program, "pointSize"), 10.0)
                gl.glUniform4f(gl.glGetUniformLocation(self._pcd_program, "overrideColor"), 1.0, 1.0, 1.0, 1.0)
                gl.glDrawArrays(gl.GL_POINTS, 0, 1)

            # 2. LiDAR 3D Markers (Electric Green, transformed by active LiDAR transform)
            active_mat = self.get_active_lidar_transform()
            has_transform = not np.allclose(active_mat, np.eye(4))
            for name, pos in self._lidar_markers.items():
                is_active = (self.active_cp_id == name)
                pos_3d = np.array(pos, dtype=np.float64)
                if has_transform:
                    pos_3d = (active_mat @ np.append(pos_3d, 1.0))[:3]

                pt = np.array(pos_3d, dtype=np.float32)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, pt.nbytes, pt, gl.GL_DYNAMIC_DRAW)

                # Outer ring
                gl.glUniform1f(gl.glGetUniformLocation(self._pcd_program, "pointSize"), 22.0 if is_active else 16.0)
                gl.glUniform4f(gl.glGetUniformLocation(self._pcd_program, "overrideColor"), 0.0, 0.2, 0.1, 0.95)
                gl.glDrawArrays(gl.GL_POINTS, 0, 1)

                # Inner core
                gl.glUniform1f(gl.glGetUniformLocation(self._pcd_program, "pointSize"), 12.0 if is_active else 10.0)
                gl.glUniform4f(gl.glGetUniformLocation(self._pcd_program, "overrideColor"), 0.0, 0.9, 0.46, 1.0)
                gl.glDrawArrays(gl.GL_POINTS, 0, 1)

            gl.glBindVertexArray(0)
            gl.glUniform1i(gl.glGetUniformLocation(self._pcd_program, "useOverrideColor"), 0)
            gl.glEnable(gl.GL_DEPTH_TEST)
        except Exception:
            pass

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self._photo_markers and not self._lidar_markers:
            return

        w_logical = float(self.width())
        h_logical = float(self.height())
        if w_logical <= 0 or h_logical <= 0:
            return

        aspect = w_logical / max(h_logical, 1.0)
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix(aspect)
        vp_mat = pyrr.matrix44.multiply(view_matrix, proj_matrix)

        def _project(pos_3d):
            p4 = np.array([pos_3d[0], pos_3d[1], pos_3d[2], 1.0], dtype=np.float32)
            clip = p4 @ vp_mat.astype(np.float32)
            if clip[3] <= 0.001:
                return None
            ndc_x = clip[0] / clip[3]
            ndc_y = clip[1] / clip[3]
            if ndc_x < -1.2 or ndc_x > 1.2 or ndc_y < -1.2 or ndc_y > 1.2:
                return None
            sx = (ndc_x + 1.0) * 0.5 * w_logical
            sy = (1.0 - ndc_y) * 0.5 * h_logical
            return sx, sy

        def _get_marker_label(name: str, prefix: str) -> str:
            if name.upper().startswith("CP"):
                num = name[2:]
                return f"{prefix}{num}"
            return f"{prefix}{name}"

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Inter", 8, QFont.Weight.Bold)
        painter.setFont(font)

        # 1. Render Photogrammetry Markers (P1, P2, P3...)
        photo_mat = self.get_active_photo_transform()
        has_photo_trans = not np.allclose(photo_mat, np.eye(4))
        for name, pos in self._photo_markers.items():
            pos_rendered = (photo_mat @ np.append(pos, 1.0))[:3] if has_photo_trans else pos
            screen_pos = _project(pos_rendered)
            if screen_pos is None:
                continue
            sx, sy = screen_pos
            is_active = (self.active_cp_id == name)
            label = _get_marker_label(name, "P")

            # Connector Stem
            badge_y = sy - 22
            painter.setPen(QPen(QColor(0, 230, 118, 220) if is_active else QColor(180, 180, 180, 180), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(sx, sy - 4), QPointF(sx, badge_y + 8))

            # Anchor Point Ring
            painter.setPen(QPen(QColor(0, 0, 0, 220), 2))
            painter.setBrush(QColor(0, 230, 118, 255) if is_active else QColor(255, 255, 255, 240))
            painter.drawEllipse(QPointF(sx, sy), 4.0, 4.0)

            # Label Pill Badge
            badge_w = 30.0
            badge_h = 17.0
            badge_rect = QRectF(sx - badge_w / 2.0, badge_y - badge_h / 2.0, badge_w, badge_h)

            bg_color = QColor(20, 20, 20, 240)
            border_color = QColor(0, 230, 118) if is_active else QColor(200, 200, 200)
            text_color = QColor(0, 230, 118) if is_active else QColor(255, 255, 255)

            painter.setPen(QPen(border_color, 1.8 if is_active else 1.0))
            painter.setBrush(bg_color)
            painter.drawRoundedRect(badge_rect, 4.0, 4.0)

            painter.setPen(QPen(text_color))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, label)

        # 2. Render LiDAR Markers (L1, L2, L3...)
        active_mat = self.get_active_lidar_transform()
        has_transform = not np.allclose(active_mat, np.eye(4))
        for name, pos in self._lidar_markers.items():
            pos_rendered = (active_mat @ np.append(pos, 1.0))[:3] if has_transform else pos

            screen_pos = _project(pos_rendered)
            if screen_pos is None:
                continue
            sx, sy = screen_pos
            is_active = (self.active_cp_id == name)
            label = _get_marker_label(name, "L")

            # Connector Stem
            badge_y = sy - 22
            painter.setPen(QPen(QColor(0, 230, 118, 220), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(sx, sy - 4), QPointF(sx, badge_y + 8))

            # Anchor Point Ring
            painter.setPen(QPen(QColor(0, 0, 0, 220), 2))
            painter.setBrush(QColor(0, 230, 118, 255))
            painter.drawEllipse(QPointF(sx, sy), 4.0, 4.0)

            # Label Pill Badge
            badge_w = 30.0
            badge_h = 17.0
            badge_rect = QRectF(sx - badge_w / 2.0, badge_y - badge_h / 2.0, badge_w, badge_h)

            bg_color = QColor(10, 32, 20, 245)
            border_color = QColor(0, 230, 118)
            text_color = QColor(0, 230, 118)

            painter.setPen(QPen(border_color, 2.0 if is_active else 1.2))
            painter.setBrush(bg_color)
            painter.drawRoundedRect(badge_rect, 4.0, 4.0)

            painter.setPen(QPen(text_color))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.end()

    # -------------------------------------------------------------------------
    # In-Viewport 3D Point Picking & Gizmo Event Handling
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        self.last_mouse_pos = pos.toPoint()

        # 1. Point Picking Mode has top priority if active
        if self.picking_target is not None and event.button() == Qt.MouseButton.LeftButton:
            picked_pt = self._pick_3d_point(pos.x(), pos.y())
            if picked_pt is not None:
                self.point_picked.emit(self.picking_target, picked_pt)
            return

        # 2. Check for Staging Gizmo Handle drag
        if self.staging_enabled and not self._preview_active and event.button() == Qt.MouseButton.LeftButton:
            pivot = self._get_gizmo_pivot()
            if pivot is not None:
                dragged = self.gizmo.begin_drag(pos.x(), pos.y(), pivot, self.camera, self.width(), self.height())
                if dragged:
                    self._gizmo_dragging = True
                    self._drag_pivot = pivot
                    self._drag_start_pivot_pos = np.copy(pivot.position)
                    self._drag_start_pivot_scale = np.copy(pivot.scale)

                    if self.staging_target == "photo":
                        self._drag_start_delta_pos = np.copy(self._photo_staging_delta_pos)
                        self._drag_start_scale = np.copy(self._photo_staging_scale)
                        self._drag_start_rotation = np.copy(self._photo_staging_rotation)
                    else:
                        self._drag_start_delta_pos = np.copy(self._lidar_staging_delta_pos)
                        self._drag_start_scale = np.copy(self._lidar_staging_scale)
                        self._drag_start_rotation = np.copy(self._lidar_staging_rotation)

                    rx = np.radians(-float(pivot.rotation[0]))
                    ry = np.radians(-float(pivot.rotation[1]))
                    rz = np.radians(-float(pivot.rotation[2]))
                    self._drag_start_pivot_r_mat = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
                    self.update()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        dx = pos.x() - self.last_mouse_pos.x()
        dy = pos.y() - self.last_mouse_pos.y()
        if self.invert_y:
            dy = -dy

        if self.staging_enabled and self._gizmo_dragging and self._drag_pivot is not None:
            self.gizmo.update_drag(pos.x(), pos.y(), self._drag_pivot, self.camera, self.width(), self.height())

            if self.gizmo.operation == "translate":
                delta = self._drag_pivot.position - self._drag_start_pivot_pos
                new_delta_pos = self._drag_start_delta_pos + delta
                if self.staging_target == "photo":
                    self._photo_staging_delta_pos = new_delta_pos
                else:
                    self._lidar_staging_delta_pos = new_delta_pos
            elif self.gizmo.operation == "scale":
                ratio = self._drag_pivot.scale / np.maximum(self._drag_start_pivot_scale, 1e-6)
                uniform_ratio = float(np.mean(ratio))
                new_scale = np.maximum(self._drag_start_scale * uniform_ratio, 1e-4).astype(np.float32)
                self._drag_pivot.scale[:] = new_scale
                if self.staging_target == "photo":
                    self._photo_staging_scale = new_scale
                else:
                    self._lidar_staging_scale = new_scale
            elif self.gizmo.operation == "rotate":
                new_rot = np.copy(self._drag_pivot.rotation)
                if self.staging_target == "photo":
                    self._photo_staging_rotation = new_rot
                else:
                    self._lidar_staging_rotation = new_rot

            self._apply_staging_update()
            self.update()
        else:
            if self.staging_enabled and not self._preview_active and self.picking_target is None:
                pivot = self._get_gizmo_pivot()
                if pivot is not None:
                    old_hover = self.gizmo.hovered_handle
                    self.gizmo.hovered_handle = self.gizmo.check_hover(pos.x(), pos.y(), pivot, self.camera, self.width(), self.height())
                    if self.gizmo.hovered_handle != old_hover:
                        self.update()

            super().mouseMoveEvent(event)

        self.last_mouse_pos = pos.toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._gizmo_dragging:
            self.gizmo.end_drag()
            self._gizmo_dragging = False
            self.update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_W:
            self.set_staging_tool("translate")
            event.accept()
            return
        elif event.key() == Qt.Key.Key_E:
            self.set_staging_tool("rotate")
            event.accept()
            return
        elif event.key() == Qt.Key.Key_R:
            self.set_staging_tool("scale")
            event.accept()
            return
        elif event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            if self.picking_target is not None:
                self.set_picking_mode(None, None)
            elif self.staging_enabled:
                self.set_staging_tool("none")
            event.accept()
            return
        super().keyPressEvent(event)

    def _pick_3d_point(self, mouse_x: float, mouse_y: float) -> Optional[np.ndarray]:
        """Projected screen-space point picking directly onto visible point cloud geometry."""
        w_logical = float(self.width())
        h_logical = float(self.height())
        if w_logical <= 0 or h_logical <= 0:
            return None

        aspect = w_logical / max(h_logical, 1.0)
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix(aspect)
        vp_mat = pyrr.matrix44.multiply(view_matrix, proj_matrix)

        if self.picking_target == 'photo':
            candidates = self._photo_alignment_cloud if self._photo_alignment_cloud is not None else self._photo_verts
            active_photo_mat = self.get_active_photo_transform()
            transform_mat = active_photo_mat if not np.allclose(active_photo_mat, np.eye(4)) else None
            return self._pick_closest_point(mouse_x, mouse_y, vp_mat, w_logical, h_logical, candidates, transform_matrix=transform_mat)
        elif self.picking_target == 'lidar':
            lidar_candidates = self._lidar_points if self._lidar_points is not None else self._lidar_surface_verts
            active_mat = self.get_active_lidar_transform()
            transform_mat = active_mat if not np.allclose(active_mat, np.eye(4)) else None
            return self._pick_closest_point(mouse_x, mouse_y, vp_mat, w_logical, h_logical, lidar_candidates, transform_matrix=transform_mat)

        return None

    @staticmethod
    def _pick_closest_point(mouse_x: float, mouse_y: float, vp_mat: np.ndarray,
                            w_logical: float, h_logical: float,
                            candidates: Optional[np.ndarray],
                            transform_matrix: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if candidates is None or len(candidates) == 0:
            return None

        # If a transform (e.g. preview transform or staging transform) is active, apply it to candidates for screen projection
        if transform_matrix is not None and not np.allclose(transform_matrix, np.eye(4)):
            pts_h = np.hstack([candidates, np.ones((len(candidates), 1), dtype=np.float64)])
            pts_world = (pts_h @ transform_matrix.T)[:, :3]
        else:
            pts_world = candidates

        N = len(pts_world)
        # Vectorized projection to screen
        pts_h = np.hstack([pts_world, np.ones((N, 1), dtype=np.float32)])
        clips = pts_h @ vp_mat.astype(np.float32)

        # Points in front of near plane
        valid = clips[:, 3] > 0.001
        if not np.any(valid):
            return None

        valid_indices = np.where(valid)[0]
        clips_val = clips[valid]

        w_clip = clips_val[:, 3]
        ndc_x = clips_val[:, 0] / w_clip
        ndc_y = clips_val[:, 1] / w_clip

        sx = (ndc_x + 1.0) * 0.5 * float(w_logical)
        sy = (1.0 - ndc_y) * 0.5 * float(h_logical)

        dx = sx - float(mouse_x)
        dy = sy - float(mouse_y)
        pixel_dist = np.sqrt(dx * dx + dy * dy)

        min_pixel_dist = np.min(pixel_dist)

        # Empty space rejection if click is too far from any point (> 35 pixels away)
        if min_pixel_dist > 35.0:
            return None

        # Select points in immediate cursor cluster (within min_pixel_dist + 3.0 px)
        cluster_threshold = min_pixel_dist + 3.5
        cluster_mask = pixel_dist <= cluster_threshold

        cluster_indices = valid_indices[cluster_mask]
        cluster_depths = w_clip[cluster_mask]

        # Disambiguate overlapping surface points by front-most camera depth
        best_cluster_local_idx = np.argmin(cluster_depths)
        best_idx = cluster_indices[best_cluster_local_idx]
        return candidates[best_idx]


class STEWorkspace(QWidget):
    """
    Spatial Texture Engine Unified Editor Workspace.
    Hosts the Mesh Editor 3D alignment viewport and the right-hand control dock.
    """
    state_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = STEWorkflowState.NO_DATA

        # Backend Models & Managers
        self.cp_manager = STEControlPointManager()
        self.alignment_result: Optional[STEAlignmentResult] = None
        self.icp_result: Optional[STEICPRefinementResult] = None
        self.quality_result: Optional[STEAlignmentQualityReport] = None

        # Data Models (Native coordinates)
        self.photo_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.photo_verts: Optional[np.ndarray] = None
        self.photo_tris: Optional[np.ndarray] = None
        self.photo_uvs: Optional[np.ndarray] = None
        self.photo_texture_path: Optional[str] = None
        self.photo_texture_img: Optional[np.ndarray] = None
        self.photo_alignment_cloud: Optional[np.ndarray] = None  # Dense point cloud for alignment & picking

        self.lidar_source_points: Optional[np.ndarray] = None
        self.lidar_surface_verts: Optional[np.ndarray] = None
        self.lidar_surface_tris: Optional[np.ndarray] = None
        self.lidar_target_uvs: Optional[np.ndarray] = None
        self.lidar_uv_result: Optional[STELiDARUVResult] = None

        self.projection_result: Optional[STETextureProjectionResult] = None
        self.bake_result: Optional[TextureBakeResult] = None

        self.active_cp_id: Optional[str] = None
        self.preview_active = False

        self._setup_ui()
        self._update_workflow_state()
        self._log("Spatial Texture Engine Editor initialized.")

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 1. Unified 3D Alignment Viewport (Native Mesh Editor Viewport)
        self.viewport = STEUnifiedAlignmentViewport(self)
        self.viewport.point_picked.connect(self._on_viewport_point_picked)

        # Top Control Toolbar
        top_bar = self._create_top_toolbar()
        main_layout.addWidget(top_bar)

        # Main Workspace: Unified Viewport (Left/Center) + Control Panel Dock (Right)
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setStyleSheet("QSplitter::handle { background-color: #282828; width: 4px; }")

        splitter.addWidget(self.viewport)

        # 2. Control Panel Dock (Right)
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter, stretch=1)

    def _create_top_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet("QFrame { background-color: #181818; border: 1px solid #282828; border-radius: 4px; }")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        title = QLabel("Spatial Texture Engine")
        title.setStyleSheet("color: #00E676; font-weight: bold; font-size: 12px; letter-spacing: 0.5px;")
        layout.addWidget(title)

        layout.addSpacing(12)

        # Display Mode Toggles
        self.btn_group_mode = QButtonGroup(self)
        self.btn_mode_points = QPushButton("☁ Points")
        self.btn_mode_surface = QPushButton("◬ Surface")
        self.btn_mode_baked = QPushButton("▣ Baked")

        for b in (self.btn_mode_points, self.btn_mode_surface, self.btn_mode_baked):
            b.setCheckable(True)
            b.setStyleSheet(self._btn_toggle_style())
            self.btn_group_mode.addButton(b)
            layout.addWidget(b)

        self.btn_mode_points.setChecked(True)
        self.btn_mode_points.clicked.connect(lambda: self._set_lidar_display_mode(STELiDARRepresentation.POINTS))
        self.btn_mode_surface.clicked.connect(lambda: self._set_lidar_display_mode(STELiDARRepresentation.SURFACE))
        self.btn_mode_baked.clicked.connect(lambda: self._set_lidar_display_mode(STELiDARRepresentation.BAKED))

        layout.addSpacing(12)

        # Visibility Checkboxes
        self.chk_photo_vis = QCheckBox("Photogrammetry")
        self.chk_photo_vis.setChecked(True)
        self.chk_photo_vis.setStyleSheet(self._checkbox_style())
        self.chk_photo_vis.stateChanged.connect(self._on_visibility_changed)
        layout.addWidget(self.chk_photo_vis)

        self.chk_lidar_vis = QCheckBox("LiDAR")
        self.chk_lidar_vis.setChecked(True)
        self.chk_lidar_vis.setStyleSheet(self._checkbox_style())
        self.chk_lidar_vis.stateChanged.connect(self._on_visibility_changed)
        layout.addWidget(self.chk_lidar_vis)

        layout.addSpacing(12)

        # Transform Gizmo Staging Toolbar
        self.btn_group_gizmo = QButtonGroup(self)
        self.btn_tool_select = QPushButton("👁 View")
        self.btn_tool_move = QPushButton("✛ Move (W)")
        self.btn_tool_rotate = QPushButton("↻ Rotate (E)")
        self.btn_tool_scale = QPushButton("⤢ Scale (R)")

        for b in (self.btn_tool_select, self.btn_tool_move, self.btn_tool_rotate, self.btn_tool_scale):
            b.setCheckable(True)
            b.setStyleSheet(self._btn_toggle_style())
            self.btn_group_gizmo.addButton(b)
            layout.addWidget(b)

        self.btn_tool_select.setChecked(True)

        layout.addSpacing(6)
        # Target Dataset Selector (LiDAR / Photogrammetry)
        self.btn_group_target = QButtonGroup(self)
        self.btn_target_lidar = QPushButton("LiDAR")
        self.btn_target_photo = QPushButton("Photogrammetry")
        for tb in (self.btn_target_lidar, self.btn_target_photo):
            tb.setCheckable(True)
            tb.setStyleSheet(self._btn_toggle_style())
            self.btn_group_target.addButton(tb)
            layout.addWidget(tb)

        self.btn_target_lidar.setChecked(True)
        self.btn_target_lidar.clicked.connect(lambda: self.viewport.set_staging_target("lidar"))
        self.btn_target_photo.clicked.connect(lambda: self.viewport.set_staging_target("photo"))

        self.btn_reset_staging = QPushButton("↺ Reset Staging")
        self.btn_reset_staging.setStyleSheet(self._btn_style_neutral())
        self.btn_reset_staging.setToolTip("Reset staging orientation and scale of active dataset back to native raw coordinates.")
        self.btn_reset_staging.clicked.connect(self._on_reset_staging_clicked)
        layout.addWidget(self.btn_reset_staging)

        self.btn_tool_select.clicked.connect(lambda: self.viewport.set_staging_tool("none"))
        self.btn_tool_move.clicked.connect(lambda: self.viewport.set_staging_tool("translate"))
        self.btn_tool_rotate.clicked.connect(lambda: self.viewport.set_staging_tool("rotate"))
        self.btn_tool_scale.clicked.connect(lambda: self.viewport.set_staging_tool("scale"))

        self.viewport.staging_tool_changed.connect(self._on_viewport_staging_tool_changed)
        self.viewport.staging_target_changed.connect(self._on_viewport_staging_target_changed)

        layout.addStretch()

        # Scale & Status Badge
        self.lbl_top_status_badge = QLabel("Alignment: Unavailable")
        self.lbl_top_status_badge.setStyleSheet("color: #888888; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.lbl_top_status_badge)

        self.btn_frame_all = QPushButton("Reset Camera")
        self.btn_frame_all.setStyleSheet(self._btn_style_neutral())
        self.btn_frame_all.clicked.connect(self.viewport.reset_view)
        layout.addWidget(self.btn_frame_all)

        return bar

    def _on_viewport_staging_tool_changed(self, tool: str):
        if tool == "translate":
            self.btn_tool_move.setChecked(True)
        elif tool == "rotate":
            self.btn_tool_rotate.setChecked(True)
        elif tool == "scale":
            self.btn_tool_scale.setChecked(True)
        else:
            self.btn_tool_select.setChecked(True)

    def _on_viewport_staging_target_changed(self, target: str):
        if target == "photo":
            self.btn_target_photo.setChecked(True)
        else:
            self.btn_target_lidar.setChecked(True)

    def _on_reset_staging_clicked(self):
        target = self.viewport.staging_target
        target_name = "Photogrammetry" if target == "photo" else "LiDAR"
        self.viewport.reset_staging_transform(target)
        self._log(f"{target_name} staging transform reset to native coordinates.")

    def _create_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background-color: #141414; border: 1px solid #282828; border-radius: 4px; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Dataset Loading Section
        layout.addWidget(self._build_datasets_group())

        # 2. Control Points Section
        layout.addWidget(self._build_control_points_group())

        # 3. Alignment Section
        layout.addWidget(self._build_alignment_group())

        # 4. Surface Reconstruction Section
        layout.addWidget(self._build_surface_group())

        # 5. Target UV Parameterization Section
        layout.addWidget(self._build_uv_group())

        # 6. Texture Projection Validation Section
        layout.addWidget(self._build_projection_group())

        # 7. Texture Baking Section
        layout.addWidget(self._build_baking_group())

        # 8. STE Console Feed
        layout.addWidget(self._build_log_group())

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # -------------------------------------------------------------------------
    # Sidebar Control Groups
    # -------------------------------------------------------------------------
    def _build_datasets_group(self) -> QGroupBox:
        group = QGroupBox("DATASETS")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        row_photo = QHBoxLayout()
        self.btn_load_photo = QPushButton("Load Photogrammetry")
        self.btn_load_photo.setStyleSheet(self._btn_style_primary())
        self.btn_load_photo.clicked.connect(self._load_active_session_reconstruction)
        row_photo.addWidget(self.btn_load_photo)
        layout.addLayout(row_photo)

        self.lbl_photo_info = QLabel("Photo: None")
        self.lbl_photo_info.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.lbl_photo_info)

        row_lidar = QHBoxLayout()
        self.btn_import_lidar = QPushButton("Import LiDAR (.ply)")
        self.btn_import_lidar.setStyleSheet(self._btn_style_neutral())
        self.btn_import_lidar.clicked.connect(self._import_lidar_file)
        row_lidar.addWidget(self.btn_import_lidar)
        layout.addLayout(row_lidar)

        self.lbl_lidar_info = QLabel("LiDAR: None")
        self.lbl_lidar_info.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.lbl_lidar_info)

        return group

    def _build_control_points_group(self) -> QGroupBox:
        group = QGroupBox("CONTROL POINTS")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        tb = QHBoxLayout()
        self.btn_add_cp = QPushButton("+ Add Point")
        self.btn_add_cp.setStyleSheet(self._btn_style_primary())
        self.btn_add_cp.clicked.connect(self._add_control_point)

        tb.addWidget(self.btn_add_cp)
        tb.addStretch()
        layout.addLayout(tb)

        self.table_cps = QTableWidget(0, 5)
        self.table_cps.setHorizontalHeaderLabels(["ID", "Photogrammetry", "LiDAR", "Status", ""])
        header = self.table_cps.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        self.table_cps.setColumnWidth(4, 28)
        self.table_cps.setStyleSheet(self._table_style())
        self.table_cps.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_cps.setSelectionMode(QTableWidget.SingleSelection)
        self.table_cps.itemSelectionChanged.connect(self._on_cp_selection_changed)
        self.table_cps.cellClicked.connect(self._on_table_cell_clicked)
        self.table_cps.setFixedHeight(125)
        layout.addWidget(self.table_cps)

        self.lbl_cp_guide = QLabel("Click 'Pick Photogrammetry' / 'Pick LiDAR' on any point row to place 3D coordinates.")
        self.lbl_cp_guide.setStyleSheet("color: #888888; font-size: 11px; font-style: italic;")
        layout.addWidget(self.lbl_cp_guide)

        return group

    def _build_alignment_group(self) -> QGroupBox:
        group = QGroupBox("ALIGNMENT & QUALITY")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        # Scale & Status Badge
        badge_row = QHBoxLayout()
        self.lbl_alignment_status = QLabel("● Alignment Unavailable")
        self.lbl_alignment_status.setStyleSheet("color: #888888; font-weight: bold; font-size: 11px;")
        self.lbl_scale_badge = QLabel("Scale: ---")
        self.lbl_scale_badge.setStyleSheet("color: #00E676; font-weight: bold; font-size: 11px;")
        badge_row.addWidget(self.lbl_alignment_status)
        badge_row.addStretch()
        badge_row.addWidget(self.lbl_scale_badge)
        layout.addLayout(badge_row)

        # Metrics Grid
        grid = QGridLayout()
        grid.setSpacing(2)

        self.lbl_val_cp_rms = QLabel("---")
        self.lbl_val_surf_median = QLabel("---")
        self.lbl_val_surf_p95 = QLabel("---")
        self.lbl_val_overlap = QLabel("---")
        self.lbl_val_quality = QLabel("---")

        for lbl in (self.lbl_val_cp_rms, self.lbl_val_surf_median, self.lbl_val_surf_p95, self.lbl_val_overlap, self.lbl_val_quality):
            lbl.setStyleSheet("color: #E0E0E0; font-weight: bold; font-size: 11px;")

        grid.addWidget(QLabel("CP RMS:"), 0, 0)
        grid.addWidget(self.lbl_val_cp_rms, 0, 1)
        grid.addWidget(QLabel("Median:"), 0, 2)
        grid.addWidget(self.lbl_val_surf_median, 0, 3)

        grid.addWidget(QLabel("P95 Dist:"), 1, 0)
        grid.addWidget(self.lbl_val_surf_p95, 1, 1)
        grid.addWidget(QLabel("BBox IoU:"), 1, 2)
        grid.addWidget(self.lbl_val_overlap, 1, 3)

        grid.addWidget(QLabel("Quality:"), 2, 0)
        grid.addWidget(self.lbl_val_quality, 2, 1)

        layout.addLayout(grid)

        # Actions
        btn_row = QHBoxLayout()
        self.btn_preview_align = QPushButton("Preview Alignment")
        self.btn_preview_align.setStyleSheet(self._btn_style_primary())
        self.btn_preview_align.clicked.connect(self._toggle_alignment_preview)

        self.btn_icp_refine = QPushButton("Refine ICP")
        self.btn_icp_refine.setStyleSheet(self._btn_style_neutral())
        self.btn_icp_refine.clicked.connect(self._run_icp_refinement)

        self.btn_reset_align = QPushButton("Reset")
        self.btn_reset_align.setStyleSheet(self._btn_style_neutral())
        self.btn_reset_align.clicked.connect(self._reset_alignment)

        btn_row.addWidget(self.btn_preview_align)
        btn_row.addWidget(self.btn_icp_refine)
        btn_row.addWidget(self.btn_reset_align)
        layout.addLayout(btn_row)

        return group

    def _build_surface_group(self) -> QGroupBox:
        group = QGroupBox("SURFACE RECONSTRUCTION")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        row = QHBoxLayout()
        self.btn_reconstruct_surface = QPushButton("Reconstruct Surface")
        self.btn_reconstruct_surface.setStyleSheet(self._btn_style_primary())
        self.btn_reconstruct_surface.clicked.connect(self._reconstruct_lidar_surface)
        row.addWidget(self.btn_reconstruct_surface)
        layout.addLayout(row)

        self.lbl_surface_stats = QLabel("Surface: Not generated")
        self.lbl_surface_stats.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.lbl_surface_stats)

        self.progress_surface = QProgressBar()
        self.progress_surface.setStyleSheet(self._prog_style())
        self.progress_surface.setVisible(False)
        layout.addWidget(self.progress_surface)

        return group

    def _build_uv_group(self) -> QGroupBox:
        group = QGroupBox("TARGET UV PARAMETERIZATION")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        row = QHBoxLayout()
        self.btn_generate_uvs = QPushButton("Generate Target UVs")
        self.btn_generate_uvs.setStyleSheet(self._btn_style_primary())
        self.btn_generate_uvs.clicked.connect(self._generate_target_uvs)
        row.addWidget(self.btn_generate_uvs)
        layout.addLayout(row)

        self.lbl_uv_stats = QLabel("UVs: Not generated")
        self.lbl_uv_stats.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.lbl_uv_stats)

        return group

    def _build_projection_group(self) -> QGroupBox:
        group = QGroupBox("TEXTURE PROJECTION")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        row = QHBoxLayout()
        self.btn_validate_proj = QPushButton("Validate Projection")
        self.btn_validate_proj.setStyleSheet(self._btn_style_primary())
        self.btn_validate_proj.clicked.connect(self._validate_texture_projection)
        row.addWidget(self.btn_validate_proj)
        layout.addLayout(row)

        self.lbl_proj_stats = QLabel("Projection: Unverified")
        self.lbl_proj_stats.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.lbl_proj_stats)

        return group

    def _build_baking_group(self) -> QGroupBox:
        group = QGroupBox("TEXTURE BAKING")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Res:"))
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["1024", "2048", "4096", "8192"])
        self.combo_resolution.setCurrentIndex(2)  # 4096
        self.combo_resolution.setStyleSheet(self._combo_style())
        cfg_row.addWidget(self.combo_resolution)

        cfg_row.addWidget(QLabel("Pad:"))
        self.spin_padding = QSpinBox()
        self.spin_padding.setRange(0, 32)
        self.spin_padding.setValue(4)
        self.spin_padding.setStyleSheet(self._spin_style())
        cfg_row.addWidget(self.spin_padding)
        cfg_row.addStretch()
        layout.addLayout(cfg_row)

        self.btn_bake_texture = QPushButton("Bake Texture")
        self.btn_bake_texture.setStyleSheet(self._btn_style_primary())
        self.btn_bake_texture.clicked.connect(self._bake_texture)
        layout.addWidget(self.btn_bake_texture)

        self.lbl_bake_stats = QLabel("Bake: Not performed")
        self.lbl_bake_stats.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.lbl_bake_stats)

        self.progress_bake = QProgressBar()
        self.progress_bake.setStyleSheet(self._prog_style())
        self.progress_bake.setVisible(False)
        layout.addWidget(self.progress_bake)

        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("CONSOLE FEED")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(90)
        self.txt_log.setStyleSheet(
            "QTextEdit { background-color: #0E0E0E; color: #00E676; font-family: monospace; font-size: 10px; border: 1px solid #282828; border-radius: 3px; }"
        )
        layout.addWidget(self.txt_log)
        return group

    # -------------------------------------------------------------------------
    # State & Visibility Handlers
    # -------------------------------------------------------------------------
    def _on_visibility_changed(self):
        self.viewport.photo_visible = self.chk_photo_vis.isChecked()
        self.viewport.lidar_visible = self.chk_lidar_vis.isChecked()
        self.viewport._apply_render()

    def _set_lidar_display_mode(self, mode: STELiDARRepresentation):
        self.viewport.lidar_display_mode = mode
        self.viewport._apply_render()
        self._log(f"LiDAR display mode set to {mode.value}.")

    def _update_workflow_state(self):
        has_photo = self.photo_verts is not None and self.photo_tris is not None
        has_lidar = self.lidar_source_points is not None or self.lidar_surface_verts is not None
        valid_cps = self.cp_manager.complete_count
        has_align = self.alignment_result is not None and self.alignment_result.success
        has_surface = self.lidar_surface_verts is not None and self.lidar_surface_tris is not None
        has_uv = self.lidar_target_uvs is not None and self.lidar_uv_result is not None
        has_proj = self.projection_result is not None and self.projection_result.is_ready_for_baking
        has_bake = self.bake_result is not None and self.bake_result.success

        if has_bake:
            self.state = STEWorkflowState.BAKED
        elif has_proj:
            self.state = STEWorkflowState.BAKE_READY
        elif has_uv and has_align:
            self.state = STEWorkflowState.PROJECTION_READY
        elif has_surface and has_align:
            self.state = STEWorkflowState.UV_READY
        elif has_align:
            self.state = STEWorkflowState.SURFACE_READY
        elif valid_cps >= 3:
            self.state = STEWorkflowState.ALIGNMENT_READY
        elif has_photo and has_lidar:
            self.state = STEWorkflowState.CONTROL_POINTS_READY
        elif has_lidar:
            self.state = STEWorkflowState.LIDAR_READY
        elif has_photo:
            self.state = STEWorkflowState.PHOTOGRAMMETRY_READY
        else:
            self.state = STEWorkflowState.NO_DATA

        # Button states
        self.btn_add_cp.setEnabled(has_photo and has_lidar)
        self.btn_preview_align.setEnabled(valid_cps >= 3)
        self.btn_reset_align.setEnabled(has_align or self.preview_active)
        self.btn_icp_refine.setEnabled(has_align and has_photo and has_lidar)

        self.btn_reconstruct_surface.setEnabled(has_lidar)
        self.btn_generate_uvs.setEnabled(has_surface)
        self.btn_validate_proj.setEnabled(has_uv and has_align and has_photo)
        self.btn_bake_texture.setEnabled(has_proj and has_uv and has_photo)

        self.btn_mode_surface.setEnabled(has_surface)
        self.btn_mode_baked.setEnabled(has_bake)

        # Staging Gizmo tool states
        if hasattr(self, 'btn_tool_select'):
            self.btn_tool_select.setEnabled(True)
            self.btn_tool_move.setEnabled(has_lidar or has_photo)
            self.btn_tool_rotate.setEnabled(has_lidar or has_photo)
            self.btn_tool_scale.setEnabled(has_lidar or has_photo)
            self.btn_target_lidar.setEnabled(has_lidar)
            self.btn_target_photo.setEnabled(has_photo)
            self.btn_reset_staging.setEnabled(has_lidar or has_photo)

        # Top & sidebar badges
        if has_align and self.quality_result is not None:
            q_rating = self.quality_result.rating.value
            median_m = self.quality_result.surface_median_dist
            p95_m = self.quality_result.surface_p95_dist
            overlap_pct = (self.quality_result.overlap_metrics.overlap_ratio * 100.0) if self.quality_result.overlap_metrics else 0.0

            if self.preview_active:
                status_text = f"● Alignment Previewing ({q_rating})"
                self.lbl_alignment_status.setStyleSheet("color: #00E676; font-weight: bold;")
                self.lbl_top_status_badge.setText(f"Preview Active (Scale: {self.alignment_result.scale:.2f}×)")
                self.lbl_top_status_badge.setStyleSheet("color: #00E676; font-weight: bold;")
            else:
                status_text = f"● Alignment {q_rating}"
                self.lbl_alignment_status.setStyleSheet("color: #00E676; font-weight: bold;" if self.quality_result.is_ready else "color: #FFA726; font-weight: bold;")
                self.lbl_top_status_badge.setText(f"Aligned ({q_rating}, Scale: {self.alignment_result.scale:.2f}×)")
                self.lbl_top_status_badge.setStyleSheet("color: #00E676; font-weight: bold;" if self.quality_result.is_ready else "color: #FFA726; font-weight: bold;")

            self.lbl_alignment_status.setText(status_text)
            self.lbl_scale_badge.setText(f"Scale: {self.alignment_result.scale:.2f}×")
            self.lbl_val_cp_rms.setText(f"{self.alignment_result.rms_error:.4f} m")
            self.lbl_val_surf_median.setText(f"{median_m*100.0:.2f} cm" if median_m >= 0 else "---")
            self.lbl_val_surf_p95.setText(f"{p95_m*100.0:.2f} cm" if p95_m >= 0 else "---")
            self.lbl_val_overlap.setText(f"{overlap_pct:.1f}%")
            self.lbl_val_quality.setText(q_rating)
        else:
            self.lbl_alignment_status.setText(f"● {valid_cps}/3 CPs Placed")
            self.lbl_alignment_status.setStyleSheet("color: #888888; font-weight: bold;" if valid_cps < 3 else "color: #FFA726; font-weight: bold;")
            self.lbl_top_status_badge.setText(f"Alignment: {valid_cps}/3 CPs")
            self.lbl_top_status_badge.setStyleSheet("color: #888888; font-weight: bold;" if valid_cps < 3 else "color: #FFA726; font-weight: bold;")
            self.lbl_scale_badge.setText("Scale: ---")
            self.lbl_val_cp_rms.setText("---")
            self.lbl_val_surf_median.setText("---")
            self.lbl_val_surf_p95.setText("---")
            self.lbl_val_overlap.setText("---")
            self.lbl_val_quality.setText("---")

        self.state_changed.emit(self.state.value)

    def _log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] [STE] {message}"
        self.txt_log.append(entry)

    # -------------------------------------------------------------------------
    # Dataset Loading & Import
    # -------------------------------------------------------------------------
    def _read_colmap_points3d_binary(self, bin_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import struct
        points = []
        colors = []
        if not os.path.exists(bin_path):
            return None, None
        try:
            with open(bin_path, "rb") as fid:
                num_points = struct.unpack("<Q", fid.read(8))[0]
                for _ in range(num_points):
                    binary_point_properties = struct.unpack("<QdddBBBd", fid.read(43))
                    x, y, z = binary_point_properties[1:4]
                    r, g, b = binary_point_properties[4:7]
                    track_len = struct.unpack("<Q", fid.read(8))[0]
                    fid.read(track_len * 8)
                    points.append((x, y, z))
                    colors.append((r / 255.0, g / 255.0, b / 255.0))
            if points:
                return np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.float32)
        except Exception:
            pass
        return None, None

    def _load_active_session_reconstruction(self):
        mvs_path = "reconstruction_out/mvs/scene_dense_mesh_texture.obj"
        tex_path = "reconstruction_out/mvs/scene_dense_mesh_texture0.png"
        dense_pcd_path = None
        for candidate in [
            "reconstruction_out/mvs/scene_dense.ply",
            "reconstruction_out/mvs/scene_dense_refcloud.ply",
            "reconstruction_out/dense/fused.ply",
            "reconstruction_out/colmap/sparse/0/points3D.ply",
            "reconstruction_out/colmap/sparse/points3D.ply",
            "reconstruction_out/colmap/points3D.ply",
            "reconstruction_out/colmap/sparse/0/points3D.bin",
            "reconstruction_out/colmap/sparse/points3D.bin",
        ]:
            if os.path.exists(candidate):
                dense_pcd_path = candidate
                break

        if not os.path.exists(mvs_path):
            mvs_path = "reconstruction_out/mvs/scene_dense_mesh.ply"

        if os.path.exists(mvs_path) or dense_pcd_path:
            self._load_photogrammetry_mesh(
                mvs_path if os.path.exists(mvs_path) else None,
                tex_path if os.path.exists(tex_path) else None,
                pcd_path=dense_pcd_path
            )
        else:
            self._log("No active photogrammetry reconstruction found in reconstruction_out.")

    def _import_photogrammetry_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Photogrammetry Mesh / Point Cloud", "", "3D Meshes & Point Clouds (*.obj *.ply *.glb *.gltf *.bin *.pcd)")
        if path:
            if path.endswith((".ply", ".bin", ".pcd", ".xyz")):
                self._load_photogrammetry_mesh(mesh_path=None, tex_path=None, pcd_path=path)
            else:
                self._load_photogrammetry_mesh(path)

    def _load_photogrammetry_mesh(self, mesh_path: Optional[str] = None, tex_path: Optional[str] = None, pcd_path: Optional[str] = None):
        try:
            # 1. Load Textured Mesh (Retained exclusively as Texture Source for projection/baking)
            if mesh_path and os.path.exists(mesh_path):
                self._log(f"Loading photogrammetry texture source mesh: {os.path.basename(mesh_path)}...")
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                self.photo_mesh = mesh
                self.photo_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
                self.photo_tris = np.asarray(mesh.triangles, dtype=np.int32).copy()
                self.photo_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float64).copy() if mesh.has_triangle_uvs() else None
                self.photo_texture_path = tex_path
                if tex_path and os.path.exists(tex_path):
                    self.photo_texture_img = np.array(Image.open(tex_path).convert('RGB'))
                else:
                    self.photo_texture_img = None
            else:
                self.photo_mesh = None
                self.photo_verts = None
                self.photo_tris = None
                self.photo_uvs = None
                self.photo_texture_path = None
                self.photo_texture_img = None

            # 2. Alignment representation: Dense / Sparse point cloud
            pts = None
            colors = None
            if pcd_path and os.path.exists(pcd_path):
                if pcd_path.endswith(".bin"):
                    pts, colors = self._read_colmap_points3d_binary(pcd_path)
                else:
                    pcd = o3d.io.read_point_cloud(pcd_path)
                    if len(pcd.points) > 0:
                        pts = np.asarray(pcd.points, dtype=np.float64).copy()
                        if pcd.has_colors() and len(pcd.colors) == len(pts):
                            colors = np.asarray(pcd.colors, dtype=np.float32).copy()
                if pts is not None and len(pts) > 0:
                    self._log(f"Loaded photogrammetry alignment point cloud: {os.path.basename(pcd_path)} ({len(pts):,} points).")

            # Fallback to mesh vertices if no separate point cloud file was found
            if pts is None or len(pts) == 0:
                if self.photo_verts is not None and len(self.photo_verts) > 0:
                    pts = self.photo_verts.copy()
                    colors = None
                    self._log(f"Exposing photogrammetry mesh vertices as alignment point cloud ({len(pts):,} points).")

            self.photo_alignment_cloud = pts
            self.photo_alignment_colors = colors

            # 3. Configure Unified Viewport with Point Cloud (Textured mesh kept internal as texture source)
            if pts is not None:
                self.viewport.set_photogrammetry_point_cloud(pts, colors=colors)
                mesh_info = f" | Texture: {os.path.basename(mesh_path)}" if (mesh_path and os.path.exists(mesh_path)) else ""
                self.lbl_photo_info.setText(f"Photo Cloud: {len(pts):,} pts{mesh_info}")
            elif self.photo_verts is not None:
                self.viewport.set_photogrammetry_point_cloud(self.photo_verts)
                self.lbl_photo_info.setText(f"Photogrammetry: {len(self.photo_verts):,} pts")

            self._log(f"Photogrammetry attached: alignment point cloud ({len(pts) if pts is not None else 0:,} pts) + internal texture source.")
            self._update_workflow_state()
        except Exception as e:
            self._log(f"Failed to load photogrammetry data: {str(e)}")

    def _import_lidar_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import LiDAR Point Cloud", "", "Point Clouds & Meshes (*.ply *.xyz *.pcd *.obj)")
        if path:
            self.attach_lidar_file(path)

    def attach_lidar_file(self, path: str):
        try:
            self._log(f"Importing LiDAR dataset: {os.path.basename(path)}...")
            self._invalidate_lidar_derived_stages()

            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) > 0:
                self.lidar_source_points = np.asarray(pcd.points, dtype=np.float64).copy()
                colors = None
                if pcd.has_colors() and len(pcd.colors) == len(self.lidar_source_points):
                    colors = np.asarray(pcd.colors, dtype=np.float32).copy()
                else:
                    try:
                        import trimesh
                        tm = trimesh.load(path)
                        if hasattr(tm, 'visual') and hasattr(tm.visual, 'vertex_colors') and tm.visual.vertex_colors is not None:
                            vc = tm.visual.vertex_colors
                            if len(vc) == len(self.lidar_source_points):
                                colors = (vc[:, :3].astype(np.float32) / 255.0)
                    except Exception:
                        pass
                self.viewport.set_lidar_points(self.lidar_source_points, colors=colors)
                self.lbl_lidar_info.setText(f"{os.path.basename(path)} ({self.lidar_source_points.shape[0]:,} pts)")
            else:
                mesh = o3d.io.read_triangle_mesh(path)
                self.lidar_surface_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
                self.lidar_surface_tris = np.asarray(mesh.triangles, dtype=np.int32).copy()
                self.lidar_source_points = self.lidar_surface_verts.copy()
                self.viewport.set_lidar_surface(mesh)
                self.lbl_lidar_info.setText(f"{os.path.basename(path)} ({self.lidar_surface_verts.shape[0]:,} V)")

            self._log(f"LiDAR dataset attached ({len(self.lidar_source_points):,} points).")
            self._update_workflow_state()
        except Exception as e:
            self._log(f"Failed to import LiDAR: {str(e)}")

    def _invalidate_lidar_derived_stages(self):
        self.lidar_surface_verts = None
        self.lidar_surface_tris = None
        self.lidar_target_uvs = None
        self.lidar_uv_result = None
        self.projection_result = None
        self.bake_result = None
        self.lbl_surface_stats.setText("Surface: Not generated")
        self.lbl_uv_stats.setText("UVs: Not generated")
        self.lbl_proj_stats.setText("Projection: Unverified")
        self.lbl_bake_stats.setText("Bake: Not performed")

    def _invalidate_alignment_derived_stages(self):
        self.projection_result = None
        self.bake_result = None
        self.lbl_proj_stats.setText("Projection: Stale")
        self.lbl_bake_stats.setText("Bake: Stale")

    # -------------------------------------------------------------------------
    # Control Points Management & In-Viewport Picking
    # -------------------------------------------------------------------------
    def _add_control_point(self):
        cp = self.cp_manager.create_control_point()
        name = cp.id
        self.active_cp_id = name
        self._refresh_control_points_table()
        self._log(f"Created control point {name}.")
        self._update_workflow_state()

    def _remove_control_point_by_id(self, name: str):
        self.cp_manager.remove_control_point(name)
        if self.active_cp_id == name:
            remaining = list(self.cp_manager._points.keys())
            self.active_cp_id = remaining[0] if remaining else None
        self._refresh_control_points_table()
        self._log(f"Removed control point {name}.")
        self._invalidate_alignment_derived_stages()
        self._update_workflow_state()

    def _remove_selected_control_point(self):
        row = self.table_cps.currentRow()
        if row >= 0:
            item = self.table_cps.item(row, 0)
            if item:
                self._remove_control_point_by_id(item.text())

    def _refresh_control_points_table(self):
        self.table_cps.blockSignals(True)
        self.table_cps.setRowCount(0)
        cps = list(self.cp_manager._points.values())
        active_picking = getattr(self.viewport, 'picking_target', None)
        active_picking_cp = getattr(self.viewport, 'active_cp_id', None)

        for i, cp in enumerate(cps):
            self.table_cps.insertRow(i)
            self.table_cps.setRowHeight(i, 28)

            # Column 0: ID
            item_id = QTableWidgetItem(cp.name)
            item_id.setTextAlignment(Qt.AlignCenter)
            self.table_cps.setItem(i, 0, item_id)

            # Column 1: Photogrammetry Action Button
            btn_photo = QPushButton()
            is_picking_photo = (active_picking == 'photo' and active_picking_cp == cp.id)
            if cp.photo_pos is not None:
                if is_picking_photo:
                    btn_photo.setText("● Repick Photo...")
                    btn_photo.setToolTip(f"Repicking Photogrammetry for {cp.name}.\nClick in viewport to replace ({cp.photo_pos[0]:.2f}, {cp.photo_pos[1]:.2f}, {cp.photo_pos[2]:.2f}), or click here to cancel.")
                    btn_photo.setStyleSheet("font-size: 11px; padding: 2px 4px; background-color: #00E676; color: #121212; border: 1px solid #00E676; border-radius: 3px; font-weight: bold;")
                else:
                    btn_photo.setText(f"✓ ({cp.photo_pos[0]:.2f}, {cp.photo_pos[1]:.2f}, {cp.photo_pos[2]:.2f})")
                    btn_photo.setToolTip(f"Photogrammetry: ({cp.photo_pos[0]:.3f}, {cp.photo_pos[1]:.3f}, {cp.photo_pos[2]:.3f})\nClick to replace / repick coordinate.")
                    btn_photo.setStyleSheet("font-size: 11px; padding: 2px 4px; background-color: #15291E; color: #00E676; border: 1px solid #00B862; border-radius: 3px; font-weight: 500;")
            else:
                if is_picking_photo:
                    btn_photo.setText("● Picking...")
                    btn_photo.setToolTip(f"Picking Photogrammetry for {cp.name}.\nClick in viewport to place coordinate, or click here to cancel.")
                    btn_photo.setStyleSheet("font-size: 11px; padding: 2px 4px; background-color: #00E676; color: #121212; border: 1px solid #00E676; border-radius: 3px; font-weight: bold;")
                else:
                    btn_photo.setText("Pick Photogrammetry")
                    btn_photo.setToolTip("Click to pick 3D coordinate on Photogrammetry surface.")
                    btn_photo.setStyleSheet("font-size: 11px; padding: 2px 4px; background-color: #242424; color: #E0E0E0; border: 1px solid #3A3A3A; border-radius: 3px;")
            btn_photo.clicked.connect(lambda checked=False, id=cp.id: self._toggle_or_set_picking_target('photo', id))
            self.table_cps.setCellWidget(i, 1, btn_photo)

            # Column 2: LiDAR Action Button
            btn_lidar = QPushButton()
            is_picking_lidar = (active_picking == 'lidar' and active_picking_cp == cp.id)
            if cp.lidar_pos is not None:
                if is_picking_lidar:
                    btn_lidar.setText("● Repick LiDAR...")
                    btn_lidar.setToolTip(f"Repicking LiDAR for {cp.name}.\nClick in viewport to replace ({cp.lidar_pos[0]:.2f}, {cp.lidar_pos[1]:.2f}, {cp.lidar_pos[2]:.2f}), or click here to cancel.")
                    btn_lidar.setStyleSheet("font-size: 11px; padding: 2px 4px; background-color: #00E676; color: #121212; border: 1px solid #00E676; border-radius: 3px; font-weight: bold;")
                else:
                    btn_lidar.setText(f"✓ ({cp.lidar_pos[0]:.2f}, {cp.lidar_pos[1]:.2f}, {cp.lidar_pos[2]:.2f})")
                    btn_lidar.setToolTip(f"LiDAR: ({cp.lidar_pos[0]:.3f}, {cp.lidar_pos[1]:.3f}, {cp.lidar_pos[2]:.3f})\nClick to replace / repick coordinate.")
                    btn_lidar.setStyleSheet("font-size: 11px; padding: 2px 4px; background-color: #15291E; color: #00E676; border: 1px solid #00B862; border-radius: 3px; font-weight: 500;")
            else:
                if is_picking_lidar:
                    btn_lidar.setText("● Picking...")
                    btn_lidar.setToolTip(f"Picking LiDAR for {cp.name}.\nClick in viewport to place coordinate, or click here to cancel.")
                    btn_lidar.setStyleSheet("font-size: 11px; padding: 2px 4px; background-color: #00E676; color: #121212; border: 1px solid #00E676; border-radius: 3px; font-weight: bold;")
                else:
                    btn_lidar.setText("Pick LiDAR")
                    btn_lidar.setToolTip("Click to pick 3D coordinate on LiDAR point cloud.")
                    btn_lidar.setStyleSheet("font-size: 11px; padding: 2px 4px; background-color: #242424; color: #E0E0E0; border: 1px solid #3A3A3A; border-radius: 3px;")
            btn_lidar.clicked.connect(lambda checked=False, id=cp.id: self._toggle_or_set_picking_target('lidar', id))
            self.table_cps.setCellWidget(i, 2, btn_lidar)

            # Column 3: Status
            status_str = "Complete" if cp.is_complete else "Incomplete"
            item_status = QTableWidgetItem(status_str)
            item_status.setTextAlignment(Qt.AlignCenter)
            item_status.setForeground(QColor("#00E676") if cp.is_complete else QColor("#888888"))
            self.table_cps.setItem(i, 3, item_status)

            # Column 4: Inline Delete [✕] Button
            btn_del = QPushButton("✕")
            btn_del.setToolTip(f"Delete {cp.name}")
            btn_del.setStyleSheet("font-size: 11px; padding: 1px 2px; background-color: transparent; color: #EF5350; border: 1px solid #552222; border-radius: 3px; font-weight: bold;")
            btn_del.clicked.connect(lambda checked=False, id=cp.id: self._remove_control_point_by_id(id))
            self.table_cps.setCellWidget(i, 4, btn_del)

        self.table_cps.blockSignals(False)

        # Update unified 3D viewport markers
        photo_markers = {cp.name: cp.photo_pos for cp in cps if cp.photo_pos is not None}
        lidar_markers = {cp.name: cp.lidar_pos for cp in cps if cp.lidar_pos is not None}
        self.viewport.set_control_markers(photo_markers, lidar_markers, self.active_cp_id)

    def _on_cp_selection_changed(self):
        row = self.table_cps.currentRow()
        if row >= 0:
            item = self.table_cps.item(row, 0)
            if item:
                self.active_cp_id = item.text()
                photo_markers = {cp.name: cp.photo_pos for cp in self.cp_manager._points.values() if cp.photo_pos is not None}
                lidar_markers = {cp.name: cp.lidar_pos for cp in self.cp_manager._points.values() if cp.lidar_pos is not None}
                self.viewport.set_control_markers(photo_markers, lidar_markers, self.active_cp_id)

    def _on_table_cell_clicked(self, row: int, column: int):
        cps = list(self.cp_manager._points.values())
        if 0 <= row < len(cps):
            cp = cps[row]
            if column == 1:
                self._toggle_or_set_picking_target('photo', cp.id)
            elif column == 2:
                self._toggle_or_set_picking_target('lidar', cp.id)
            elif column == 0 or column == 3:
                self.active_cp_id = cp.id
                photo_markers = {c.name: c.photo_pos for c in cps if c.photo_pos is not None}
                lidar_markers = {c.name: c.lidar_pos for c in cps if c.lidar_pos is not None}
                self.viewport.set_control_markers(photo_markers, lidar_markers, self.active_cp_id)

    def _toggle_or_set_picking_target(self, target: str, cp_id: Optional[str] = None):
        target_cp = cp_id or self.active_cp_id
        if not target_cp:
            if self.cp_manager.count == 0:
                self._add_control_point()
                target_cp = self.active_cp_id
            else:
                target_cp = list(self.cp_manager._points.keys())[0]

        # If already picking this target and point, clicking again toggles picking mode off
        if getattr(self.viewport, 'picking_target', None) == target and getattr(self.viewport, 'active_cp_id', None) == target_cp:
            self.viewport.set_picking_mode(None, None)
            self.lbl_cp_guide.setText("Point picking cancelled.")
            self._log(f"Point picking cancelled for {target_cp}.")
            self._refresh_control_points_table()
            return

        self.active_cp_id = target_cp
        self.viewport.set_picking_mode(target, self.active_cp_id)
        target_name = "Photogrammetry" if target == 'photo' else "LiDAR"
        cp = self.cp_manager._points.get(target_cp)
        existing_pos = cp.photo_pos if target == 'photo' else (cp.lidar_pos if cp else None)
        if existing_pos is not None:
            self.lbl_cp_guide.setText(f"Click in viewport to replace {target_name} coordinate for {self.active_cp_id} (or click button to cancel).")
            self._log(f"Repicking {self.active_cp_id} ({target_name}). Click viewport to replace ({existing_pos[0]:.2f}, {existing_pos[1]:.2f}, {existing_pos[2]:.2f}).")
        else:
            self.lbl_cp_guide.setText(f"Click on {target_name} in viewport to place {self.active_cp_id}.")
            self._log(f"Point picking active for {self.active_cp_id} ({target_name}).")
        self._refresh_control_points_table()

    def _set_picking_target(self, target: str, cp_id: Optional[str] = None):
        self._toggle_or_set_picking_target(target, cp_id)

    def _on_viewport_point_picked(self, target: str, pt: np.ndarray):
        if not self.active_cp_id:
            return

        if target == 'photo':
            self.cp_manager.set_photo_marker(self.active_cp_id, pt)
            self._log(f"Placed {self.active_cp_id} on Photogrammetry: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}).")
        elif target == 'lidar':
            self.cp_manager.set_lidar_marker(self.active_cp_id, pt)
            self._log(f"Placed {self.active_cp_id} on LiDAR: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}).")

        self.viewport.set_picking_mode(None, None)
        self.lbl_cp_guide.setText("Point placed. Select next point or trigger alignment.")
        self._invalidate_alignment_derived_stages()
        self._refresh_control_points_table()
        self._auto_solve_alignment_if_ready()
        self._update_workflow_state()

    # -------------------------------------------------------------------------
    # Scale-Aware Alignment & ICP Execution
    # -------------------------------------------------------------------------
    def _auto_solve_alignment_if_ready(self):
        if self.cp_manager.complete_count >= 3:
            self._solve_alignment()

    def _solve_alignment(self):
        self._log("Solving scale-aware Horn alignment with CCCoreLib...")
        res = STEAlignmentService.solve_from_manager(self.cp_manager, adjust_scale=True)
        if res.success:
            self.alignment_result = res
            self._log(f"Alignment solved: Scale = {res.scale:.2f}×, CP RMS = {res.rms_error:.4f} m.")

            lidar_pts = self.lidar_source_points if self.lidar_source_points is not None else self.lidar_surface_verts
            target_photo_pts = self.photo_alignment_cloud if self.photo_alignment_cloud is not None else self.photo_verts
            if target_photo_pts is not None and lidar_pts is not None:
                q_res = STEAlignmentQualityGate.evaluate(
                    source_lidar_points=lidar_pts,
                    target_photogrammetry_points=target_photo_pts,
                    alignment_result=res
                )
                self.quality_result = q_res
                self._log(f"Quality Gate evaluated: {q_res.rating.value} ({q_res.readiness.value}).")

            if self.preview_active:
                self.viewport.set_preview_transform(res.transformation_matrix)

            self._invalidate_alignment_derived_stages()
            self._update_workflow_state()
        else:
            self._log(f"Alignment failed: {res.status_message}")

    def _toggle_alignment_preview(self):
        if not self.alignment_result or not self.alignment_result.success:
            self._solve_alignment()

        if self.alignment_result and self.alignment_result.success:
            self.preview_active = not self.preview_active
            if self.preview_active:
                self.viewport.set_preview_transform(self.alignment_result.transformation_matrix)
                self.btn_preview_align.setText("Exit Preview")
                self._log("Alignment preview activated in shared viewport.")
            else:
                self.viewport.reset_preview_transform()
                self.btn_preview_align.setText("Preview Alignment")
                self._log("Alignment preview deactivated (native coordinates restored).")
            self._update_workflow_state()

    def _reset_alignment(self):
        self.preview_active = False
        self.alignment_result = None
        self.quality_result = None
        self.viewport.reset_preview_transform()
        self.btn_preview_align.setText("Preview Alignment")
        self._log("Alignment reset to native coordinates.")
        self._invalidate_alignment_derived_stages()
        self._update_workflow_state()

    def _run_icp_refinement(self):
        if not self.alignment_result or not self.alignment_result.success:
            self._log("Initial alignment required before running ICP.")
            return

        lidar_pts = self.lidar_source_points if self.lidar_source_points is not None else self.lidar_surface_verts
        target_photo_pts = self.photo_alignment_cloud if self.photo_alignment_cloud is not None else self.photo_verts
        if lidar_pts is None or target_photo_pts is None:
            self._log("LiDAR and Photogrammetry point cloud geometry required for ICP.")
            return

        self._log("Running CCCoreLib fine ICP refinement on point clouds...")
        icp_settings = STEICPRefinementSettings(adjust_scale=False, max_iterations=50, overlap_ratio=0.80)
        icp_res = STEICPRefinementService.refine(
            source_lidar_points=lidar_pts,
            target_photogrammetry_points=target_photo_pts,
            initial_alignment=self.alignment_result,
            settings=icp_settings,
            cp_manager=self.cp_manager
        )

        if icp_res.success:
            self.icp_result = icp_res
            refined_align = icp_res.to_alignment_result()
            new_quality = STEAlignmentQualityGate.evaluate(
                source_lidar_points=lidar_pts,
                target_photogrammetry_points=target_photo_pts,
                alignment_result=refined_align
            )

            # Check if ICP degraded alignment quality or CP fit
            init_cp_rms = self.alignment_result.rms_error if (self.alignment_result and self.alignment_result.rms_error > 0) else 0.1
            new_cp_rms = refined_align.rms_error if refined_align.rms_error > 0 else 0.1

            if new_cp_rms > (init_cp_rms * 1.5) and (new_cp_rms - init_cp_rms) > 0.10:
                self._log(f"ICP refinement rejected: Control-point RMS error increased ({new_cp_rms*100.0:.1f} cm vs {init_cp_rms*100.0:.1f} cm). Preserving initial alignment.")
            elif (self.quality_result and
                self.quality_result.surface_median_dist >= 0 and
                new_quality.surface_median_dist > (self.quality_result.surface_median_dist * 1.25)):
                self._log("ICP refinement rejected: Surface median error increased. Preserving initial alignment.")
            else:
                self.alignment_result = refined_align
                self.quality_result = new_quality
                self._log(f"ICP accepted: Median distance = {new_quality.surface_median_dist*100.0:.2f} cm, CP RMS = {refined_align.rms_error*100.0:.2f} cm.")
                if self.preview_active:
                    self.viewport.set_preview_transform(self.alignment_result.transformation_matrix)

            self._invalidate_alignment_derived_stages()
            self._update_workflow_state()
        else:
            self._log(f"ICP refinement: {icp_res.status_message}")

    # -------------------------------------------------------------------------
    # Surface Reconstruction, UV, Projection, and Baking
    # -------------------------------------------------------------------------
    def _reconstruct_lidar_surface(self):
        if self.lidar_source_points is None:
            self._log("Cannot reconstruct surface: No LiDAR point cloud loaded.")
            return

        self.progress_surface.setVisible(True)
        self.progress_surface.setValue(10)
        self.btn_reconstruct_surface.setEnabled(False)

        self._surf_worker = STESurfaceReconstructionWorker(self.lidar_source_points, depth=8, parent=self)
        self._surf_worker.progress.connect(lambda pct, msg: self._on_surf_progress(pct, msg))
        self._surf_worker.finished.connect(self._on_surf_finished)
        self._surf_worker.error.connect(self._on_surf_error)
        self._surf_worker.start()

    def _on_surf_progress(self, pct: float, msg: str):
        self.progress_surface.setValue(int(pct * 100))
        self.lbl_surface_stats.setText(msg)

    def _on_surf_finished(self, mesh: o3d.geometry.TriangleMesh):
        self.progress_surface.setVisible(False)
        self.btn_reconstruct_surface.setEnabled(True)
        self.lidar_surface_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
        self.lidar_surface_tris = np.asarray(mesh.triangles, dtype=np.int32).copy()

        self.viewport.set_lidar_surface(mesh)
        self.btn_mode_surface.setChecked(True)

        self.lbl_surface_stats.setText(f"Ready: {self.lidar_surface_verts.shape[0]:,} V, {self.lidar_surface_tris.shape[0]:,} T")
        self._log(f"LiDAR surface reconstructed: {self.lidar_surface_verts.shape[0]:,} vertices, {self.lidar_surface_tris.shape[0]:,} triangles.")
        self._update_workflow_state()

    def _on_surf_error(self, err_msg: str):
        self.progress_surface.setVisible(False)
        self.btn_reconstruct_surface.setEnabled(True)
        self._log(f"Surface reconstruction error: {err_msg}")
        QMessageBox.critical(self, "Surface Reconstruction Error", err_msg)

    def _generate_target_uvs(self):
        if self.lidar_surface_verts is None or self.lidar_surface_tris is None:
            self._log("Reconstructed LiDAR surface required for Target UV generation.")
            return

        self._log("Generating LiDAR target UV parameterization and atlas packing...")
        uv_res = STELiDARUVService.generate_uvs(
            mesh_or_vertices=self.lidar_surface_verts,
            triangles=self.lidar_surface_tris,
            settings=STELiDARUVSettings()
        )

        if uv_res.success:
            self.lidar_uv_result = uv_res
            self.lidar_target_uvs = uv_res.uvs
            self.lbl_uv_stats.setText(
                f"{uv_res.chart_count} charts, {uv_res.uv_utilization:.1f}% util, {uv_res.overlapping_triangle_count} overlaps"
            )
            self._log(f"Target UVs generated: {uv_res.chart_count} charts, {uv_res.uv_utilization:.1f}% utilization.")
            self._update_workflow_state()
        else:
            self._log(f"Target UV generation failed: {uv_res.status_message}")

    def _validate_texture_projection(self):
        if not self.alignment_result or self.lidar_surface_verts is None or self.photo_verts is None:
            self._log("Alignment, LiDAR surface, and Photogrammetry mesh required for projection validation.")
            return

        self._log("Validating geometric correspondence and surface projection...")
        sample_count = min(10000, self.lidar_surface_verts.shape[0])
        idx = np.linspace(0, self.lidar_surface_verts.shape[0] - 1, sample_count, dtype=int)
        sample_pts = self.lidar_surface_verts[idx]

        proj_res = STETextureProjectionService.project(
            lidar_surface_points=sample_pts,
            photogrammetry_vertices=self.photo_verts,
            photogrammetry_triangles=self.photo_tris,
            photogrammetry_uvs=self.photo_uvs if self.photo_uvs is not None else np.zeros((self.photo_tris.shape[0]*3, 2)),
            alignment_result=self.alignment_result
        )

        self.projection_result = proj_res
        if proj_res.is_ready_for_baking:
            self.lbl_proj_stats.setText(f"✓ READY ({proj_res.coverage_ratio*100.0:.1f}% cov, med={proj_res.median_distance*100.0:.2f}cm)")
            self._log(f"Projection verified: {proj_res.coverage_ratio*100.0:.2f}% coverage, {proj_res.median_distance*100.0:.2f} cm median distance.")
        else:
            self.lbl_proj_stats.setText(f"⚠ Partial ({proj_res.coverage_ratio*100.0:.1f}% cov)")
            self._log(f"Projection warning: {proj_res.status_message}")

        self._update_workflow_state()

    def _bake_texture(self):
        if not self.projection_result or self.lidar_target_uvs is None or self.photo_verts is None:
            self._log("Cannot bake: Ensure UVs, alignment, and projection validation are complete.")
            return

        res_val = int(self.combo_resolution.currentText())
        padding_val = self.spin_padding.value()

        self.progress_bake.setVisible(True)
        self.progress_bake.setValue(5)
        self.btn_bake_texture.setEnabled(False)

        source_tex = self.photo_texture_img if self.photo_texture_img is not None else self.photo_texture_path
        if source_tex is None:
            source_tex = np.full((512, 512, 3), 200, dtype=np.uint8)

        self._bake_worker = STETextureBakingWorker(
            photogrammetry_mesh=(self.photo_verts, self.photo_tris, self.photo_uvs if self.photo_uvs is not None else np.zeros((self.photo_tris.shape[0]*3, 2))),
            photogrammetry_texture=source_tex,
            lidar_surface_mesh=(self.lidar_surface_verts, self.lidar_surface_tris),
            lidar_target_uvs=self.lidar_target_uvs,
            alignment_transform=self.alignment_result,
            texture_resolution=res_val,
            texture_padding=padding_val,
            batch_size=100000,
            parent=self
        )
        self._bake_worker.progress.connect(self._on_bake_progress)
        self._bake_worker.finished.connect(self._on_bake_finished)
        self._bake_worker.error.connect(self._on_bake_error)
        self._bake_worker.start()

    def _on_bake_progress(self, pct: float, msg: str):
        self.progress_bake.setValue(int(pct * 100))
        self.lbl_bake_stats.setText(msg)

    def _on_bake_finished(self, res: TextureBakeResult):
        self.progress_bake.setVisible(False)
        self.btn_bake_texture.setEnabled(True)

        if res.success:
            self.bake_result = res
            self.lbl_bake_stats.setText(f"✓ Complete ({res.texture_width}x{res.texture_height}, {res.coverage_ratio*100.0:.1f}% cov)")
            self._log(f"Texture Baking complete: {res.texture_width}x{res.texture_height}, {res.valid_texture_pixels:,} pixels ({res.coverage_ratio*100.0:.1f}% coverage).")

            if res.output_mesh is not None:
                self.viewport.set_lidar_baked(res.output_mesh, res.output_texture)
                self.btn_mode_baked.setChecked(True)

            self._update_workflow_state()
        else:
            self.lbl_bake_stats.setText(f"Failed: {res.status}")
            self._log(f"Texture baking failed: {res.status_message}")
            QMessageBox.warning(self, "Baking Warning", res.status_message)

    def _on_bake_error(self, err_msg: str):
        self.progress_bake.setVisible(False)
        self.btn_bake_texture.setEnabled(True)
        self._log(f"Texture baking worker error: {err_msg}")
        QMessageBox.critical(self, "Baking Error", err_msg)

    # -------------------------------------------------------------------------
    # Style Definitions (ProximaP Visual Language: Green, Dark Gray/Black, White/Gray)
    # -------------------------------------------------------------------------
    def _group_style(self) -> str:
        return """
            QGroupBox {
                color: #00E676;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 0.5px;
                border: 1px solid #282828;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 8px;
                background-color: #181818;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                color: #00E676;
                font-weight: bold;
            }
        """

    def _btn_style_primary(self) -> str:
        return """
            QPushButton {
                background-color: #1A3A2A;
                color: #00E676;
                border: 1px solid #00E676;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #244C37;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #181818;
                color: #555555;
                border: 1px solid #282828;
            }
        """

    def _btn_style_neutral(self) -> str:
        return """
            QPushButton {
                background-color: #202020;
                color: #DDDDDD;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border-color: #00E676;
            }
            QPushButton:disabled {
                background-color: #141414;
                color: #444444;
                border: 1px solid #222222;
            }
        """

    def _btn_style_destructive(self) -> str:
        return """
            QPushButton {
                background-color: #2A1A1A;
                color: #FF5252;
                border: 1px solid #5A2A2A;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3A1A1A;
                border-color: #FF5252;
            }
            QPushButton:disabled {
                background-color: #141414;
                color: #444444;
                border: 1px solid #222222;
            }
        """

    def _btn_toggle_style(self) -> str:
        return """
            QPushButton {
                background-color: #1A1A1A;
                color: #AAAAAA;
                border: 1px solid #282828;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:checked {
                background-color: #1E3A2B;
                color: #00E676;
                font-weight: bold;
                border: 1px solid #00E676;
            }
            QPushButton:hover:!checked {
                background-color: #242424;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #141414;
                color: #444444;
                border: 1px solid #222222;
            }
        """

    def _checkbox_style(self) -> str:
        return """
            QCheckBox {
                color: #E0E0E0;
                font-weight: bold;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                background-color: #141414;
                border: 1px solid #333333;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #00E676;
                border-color: #00E676;
            }
        """

    def _table_style(self) -> str:
        return """
            QTableWidget {
                background-color: #121212;
                color: #E0E0E0;
                gridline-color: #242424;
                border: 1px solid #282828;
                border-radius: 4px;
                font-size: 10px;
            }
            QHeaderView::section {
                background-color: #1A1A1A;
                color: #888888;
                padding: 3px;
                border: 1px solid #282828;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background-color: #1E3A2B;
                color: #00E676;
            }
        """

    def _combo_style(self) -> str:
        return """
            QComboBox {
                background-color: #1A1A1A;
                color: #00E676;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox:hover {
                border-color: #00E676;
            }
        """

    def _spin_style(self) -> str:
        return """
            QSpinBox {
                background-color: #1A1A1A;
                color: #00E676;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 2px 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QSpinBox:focus {
                border-color: #00E676;
            }
        """

    def _prog_style(self) -> str:
        return """
            QProgressBar {
                border: 1px solid #282828;
                border-radius: 3px;
                background-color: #121212;
                text-align: center;
                color: #FFFFFF;
                height: 12px;
                font-size: 9px;
            }
            QProgressBar::chunk {
                background-color: #00E676;
                border-radius: 2px;
            }
        """
