"""
Spatial Texture Engine (STE) - CCCoreLib Alignment & ICP Refinement Subsystem
=============================================================================

This module provides the complete production alignment and fine-registration engine
for the Spatial Texture Engine (STE), powered by CloudCompare's CCCoreLib backend.

Pipeline Architecture:
----------------------
    Control Points (CP1, CP2, CP3...)
              ↓
    CCCoreLib Horn Absolute Orientation (Scale-Aware)
              ↓
    Initial Alignment Transform (T_init: s_cp, R_cp, t_cp)
              ↓
    CCCoreLib ICP Refinement (T_icp delta on pre-transformed LiDAR geometry)
              ↓
    Transformation Composition (T_final = T_icp ∘ T_init)
              ↓
    Non-Destructive Viewport Preview / Accept

Mathematical Convention:
------------------------
    P_photo = s * R * P_lidar + t

Transformation Composition:
---------------------------
    P_init  = s_cp * R_cp * P_lidar + t_cp
    P_final = s_icp * R_icp * P_init + t_icp
    
    When adjust_scale=False in ICP (default, scale locked to control-point estimate):
        s_icp = 1.0
        P_final = (s_cp * (R_icp @ R_cp)) @ P_lidar + (R_icp @ t_cp + t_icp)

    In 4x4 Homogeneous Matrix form:
        T_final = T_icp @ T_init
        s_final = s_icp * s_cp
        R_final = R_icp @ R_cp
        t_final = (s_icp * R_icp @ t_cp) + t_icp
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

import ste_cccorelib_registration as ccr


@dataclass
class STEControlPoint:
    """
    Represents a paired control point across Photogrammetry and LiDAR coordinate spaces.
    """
    id: str
    name: str = ""
    photo_pos: Optional[np.ndarray] = None   # shape (3,), float64
    lidar_pos: Optional[np.ndarray] = None   # shape (3,), float64
    enabled: bool = True

    def __post_init__(self):
        if not self.name:
            self.name = self.id
        if self.photo_pos is not None:
            self.photo_pos = np.asarray(self.photo_pos, dtype=np.float64).ravel()
        if self.lidar_pos is not None:
            self.lidar_pos = np.asarray(self.lidar_pos, dtype=np.float64).ravel()

    @property
    def is_complete(self) -> bool:
        """A control point is complete when both valid positions are assigned."""
        if not self.enabled:
            return False
        if self.photo_pos is None or self.lidar_pos is None:
            return False
        if len(self.photo_pos) != 3 or len(self.lidar_pos) != 3:
            return False
        if not np.all(np.isfinite(self.photo_pos)) or not np.all(np.isfinite(self.lidar_pos)):
            return False
        return True


class STEControlPointManager:
    """
    Manages the hierarchy and lifecycle of STE control points.
    Preserves existing user interaction patterns (CP1, CP2, CP3...)
    """
    def __init__(self):
        self._points: Dict[str, STEControlPoint] = {}
        self._order: List[str] = []

    def create_control_point(self, cp_id: Optional[str] = None, name: Optional[str] = None) -> STEControlPoint:
        """Create a new control point in the manager."""
        if cp_id is None:
            idx = 1
            while f"CP{idx}" in self._points:
                idx += 1
            cp_id = f"CP{idx}"
        
        cp = STEControlPoint(id=cp_id, name=name or cp_id)
        self._points[cp_id] = cp
        if cp_id not in self._order:
            self._order.append(cp_id)
        return cp

    def set_photo_marker(self, cp_id: str, pos: np.ndarray) -> STEControlPoint:
        """Set or update the photogrammetry marker position for a control point."""
        if cp_id not in self._points:
            self.create_control_point(cp_id=cp_id)
        cp = self._points[cp_id]
        cp.photo_pos = np.asarray(pos, dtype=np.float64).copy().ravel()
        return cp

    def set_lidar_marker(self, cp_id: str, pos: np.ndarray) -> STEControlPoint:
        """Set or update the LiDAR marker position for a control point."""
        if cp_id not in self._points:
            self.create_control_point(cp_id=cp_id)
        cp = self._points[cp_id]
        cp.lidar_pos = np.asarray(pos, dtype=np.float64).copy().ravel()
        return cp

    def get_point(self, cp_id: str) -> Optional[STEControlPoint]:
        """Get a control point by ID."""
        return self._points.get(cp_id)

    def remove_control_point(self, cp_id: str) -> bool:
        """Remove a control point by ID."""
        if cp_id in self._points:
            del self._points[cp_id]
            if cp_id in self._order:
                self._order.remove(cp_id)
            return True
        return False

    def clear(self):
        """Clear all control points."""
        self._points.clear()
        self._order.clear()

    def get_complete_pairs(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract all complete (Photogrammetry + LiDAR) control point coordinate pairs.
        
        Returns:
            lidar_pts: (N, 3) float64 array of source LiDAR coordinates
            photo_pts: (N, 3) float64 array of target Photogrammetry coordinates
            ids: List of N control point IDs
        """
        lidar_list = []
        photo_list = []
        ids = []

        for cp_id in self._order:
            cp = self._points[cp_id]
            if cp.is_complete:
                lidar_list.append(cp.lidar_pos)
                photo_list.append(cp.photo_pos)
                ids.append(cp.id)

        if not lidar_list:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64), []

        return (
            np.vstack(lidar_list).astype(np.float64),
            np.vstack(photo_list).astype(np.float64),
            ids
        )

    @property
    def count(self) -> int:
        return len(self._points)

    @property
    def complete_count(self) -> int:
        return sum(1 for cp in self._points.values() if cp.is_complete)


@dataclass
class STEAlignmentResult:
    """
    Result of an STE alignment calculation (Control-Point or Refined).
    """
    success: bool
    status: str
    status_message: str

    rotation: np.ndarray                    # 3x3 orthonormal rotation matrix R
    translation: np.ndarray                 # 3D translation vector t, shape (3,)
    scale: float                            # Uniform scale factor s (s > 0)
    transformation_matrix: np.ndarray       # 4x4 matrix: [ [s*R, t], [0, 1] ]

    rms_error: float                        # Root Mean Square error (m / units)
    residuals: List[float]                  # Per-point residual Euclidean distances
    residual_vectors: List[np.ndarray] = field(default_factory=list) # Per-point 3D residual vectors
    control_point_count: int = 0            # Number of complete pairs used
    control_point_ids: List[str] = field(default_factory=list) # Control point IDs used
    alignment_method: str = "cccorelib_scaled"  # "cccorelib_scaled", "cccorelib_rigid", "cccorelib_icp"

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Transform LiDAR coordinates into Photogrammetry space:
            P_photo = s * (P_lidar @ R.T) + t
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1 and pts.shape[0] == 3:
            return self.scale * (self.rotation @ pts) + self.translation
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points shape (N, 3), got {pts.shape}")
        if pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return (pts @ (self.scale * self.rotation).T) + self.translation

    def inverse_apply(self, points: np.ndarray) -> np.ndarray:
        """
        Transform Photogrammetry coordinates into LiDAR space:
            P_lidar = (1 / s) * ((P_photo - t) @ R)
        """
        pts = np.asarray(points, dtype=np.float64)
        if self.scale <= 1e-12:
            raise ValueError("Cannot invert transformation with zero or negative scale.")
        if pts.ndim == 1 and pts.shape[0] == 3:
            return (self.rotation.T @ (pts - self.translation)) / self.scale
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points shape (N, 3), got {pts.shape}")
        if pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return ((pts - self.translation) @ self.rotation) / self.scale


@dataclass
class STEICPRefinementSettings:
    """
    Settings for CCCoreLib ICP Refinement.
    """
    adjust_scale: bool = False               # Lock scale by default to preserve control-point scale
    min_rms_decrease: float = 1e-5           # Convergence threshold
    max_iterations: int = 40                 # Maximum ICP iterations
    sampling_limit: int = 50000              # Maximum points sampled per entity
    overlap_ratio: float = 0.80              # Trimmed overlap ratio (0.0 - 1.0, 0.80 trims 20% outliers)
    max_allowed_translation_drift: float = 5.0 # Max allowed ICP translation delta (m) to reject runaway fits
    roi_margin_ratio: float = 0.35           # Margin around LiDAR bounding box to focus Photogrammetry reference points
    max_allowed_cp_degradation_factor: float = 2.0 # Max allowed factor by which Control Point RMS can increase


@dataclass
class STEICPRefinementResult:
    """
    Detailed result of CCCoreLib ICP fine registration.
    """
    success: bool
    status: str
    status_message: str

    initial_rms: float                       # RMS error before ICP (control-point fit)
    final_rms: float                         # Final ICP surface RMS error
    
    scale: float                             # Total combined scale s_final
    rotation: np.ndarray                     # Total combined rotation R_final
    translation: np.ndarray                  # Total combined translation t_final
    transformation_matrix: np.ndarray        # Total 4x4 transform T_final = T_icp @ T_init

    icp_delta_transform: np.ndarray          # Incremental 4x4 transform computed by ICP
    icp_scale_delta: float                   # Incremental scale computed by ICP
    iterations: int = 0                      # Completed iterations
    registered_point_count: int = 0          # Number of samples registered

    initial_cp_residuals: List[float] = field(default_factory=list) # CP residuals before ICP
    final_cp_residuals: List[float] = field(default_factory=list)   # CP residuals after ICP
    control_point_ids: List[str] = field(default_factory=list)

    def to_alignment_result(self) -> STEAlignmentResult:
        """Convert ICP refinement result into a standard STEAlignmentResult."""
        cp_rms = float(np.sqrt(np.mean(np.square(self.final_cp_residuals)))) if len(self.final_cp_residuals) > 0 else self.final_rms
        return STEAlignmentResult(
            success=self.success,
            status=self.status,
            status_message=self.status_message,
            rotation=self.rotation,
            translation=self.translation,
            scale=self.scale,
            transformation_matrix=self.transformation_matrix,
            rms_error=cp_rms,
            residuals=self.final_cp_residuals,
            control_point_count=len(self.control_point_ids),
            control_point_ids=self.control_point_ids,
            alignment_method="cccorelib_icp"
        )


class STEAlignmentService:
    """
    Production alignment service for Spatial Texture Engine using CCCoreLib.
    Supports scale-aware similarity alignment (primary mode) and rigid alignment.
    """

    @staticmethod
    def check_geometric_degeneracy(points: np.ndarray, tolerance: float = 1e-4) -> Tuple[bool, str]:
        """
        Numerically robust test for geometric degeneracy:
        - Rejects < 3 points
        - Rejects non-finite values (NaN / Inf)
        - Rejects coincident points
        - Rejects collinear points (rank 1 configurations)
        - Accepts planar configurations (3 non-collinear points or coplanar points)
        - Accepts general 3D configurations
        """
        pts = np.asarray(points, dtype=np.float64)
        N = pts.shape[0]
        if N < 3:
            return False, f"At least 3 non-collinear points are required. Provided: {N}."

        if not np.all(np.isfinite(pts)):
            return False, "Points contain non-finite values (NaN or Inf)."

        # Check coincident points (pairwise distances)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-7:
                    return False, f"Degenerate configuration: points at index {i} and {j} are coincident (distance={d:.2e})."

        # Center points
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid

        # Compute SVD of centered points
        _, s, _ = np.linalg.svd(centered)

        span = s[0]
        if span < 1e-7:
            return False, "Degenerate configuration: points have near-zero spatial spread."

        if (s[1] / span) < tolerance:
            return False, f"Degenerate configuration: points are collinear (1D line ratio={s[1]/span:.2e})."

        return True, "Valid non-collinear geometric configuration."

    @classmethod
    def solve(
        cls,
        lidar_points: np.ndarray,
        photogrammetry_points: np.ndarray,
        adjust_scale: bool = True,
        control_point_ids: Optional[List[str]] = None
    ) -> STEAlignmentResult:
        """
        Calculate transformation mapping LiDAR control points to Photogrammetry control points:
            P_photo = s * R * P_lidar + t
        """
        pts_lidar = np.ascontiguousarray(lidar_points, dtype=np.float64)
        pts_photo = np.ascontiguousarray(photogrammetry_points, dtype=np.float64)
        method_name = "cccorelib_scaled" if adjust_scale else "cccorelib_rigid"

        if pts_lidar.ndim != 2 or pts_lidar.shape[1] != 3 or pts_photo.ndim != 2 or pts_photo.shape[1] != 3:
            return STEAlignmentResult(
                success=False,
                status="invalid_dimensions",
                status_message=f"Control point arrays must be (N, 3). Got {pts_lidar.shape} and {pts_photo.shape}.",
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
                scale=1.0,
                transformation_matrix=np.eye(4, dtype=np.float64),
                rms_error=-1.0,
                residuals=[],
                alignment_method=method_name
            )

        if pts_lidar.shape[0] != pts_photo.shape[0]:
            return STEAlignmentResult(
                success=False,
                status="count_mismatch",
                status_message=f"Point count mismatch: {pts_lidar.shape[0]} LiDAR vs {pts_photo.shape[0]} Photogrammetry points.",
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
                scale=1.0,
                transformation_matrix=np.eye(4, dtype=np.float64),
                rms_error=-1.0,
                residuals=[],
                alignment_method=method_name
            )

        count = pts_lidar.shape[0]
        if count < 3:
            return STEAlignmentResult(
                success=False,
                status="insufficient_points",
                status_message=f"At least 3 complete control-point pairs are required for alignment. Provided: {count}.",
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
                scale=1.0,
                transformation_matrix=np.eye(4, dtype=np.float64),
                rms_error=-1.0,
                residuals=[],
                control_point_count=count,
                alignment_method=method_name
            )

        valid_lidar, msg_lidar = cls.check_geometric_degeneracy(pts_lidar)
        if not valid_lidar:
            return STEAlignmentResult(
                success=False,
                status="degenerate_lidar_points",
                status_message=f"LiDAR control points invalid: {msg_lidar}",
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
                scale=1.0,
                transformation_matrix=np.eye(4, dtype=np.float64),
                rms_error=-1.0,
                residuals=[],
                control_point_count=count,
                alignment_method=method_name
            )

        valid_photo, msg_photo = cls.check_geometric_degeneracy(pts_photo)
        if not valid_photo:
            return STEAlignmentResult(
                success=False,
                status="degenerate_photo_points",
                status_message=f"Photogrammetry control points invalid: {msg_photo}",
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
                scale=1.0,
                transformation_matrix=np.eye(4, dtype=np.float64),
                rms_error=-1.0,
                residuals=[],
                control_point_count=count,
                alignment_method=method_name
            )

        cc_res = ccr.register_point_pairs(
            p_aligned=pts_lidar,
            p_ref=pts_photo,
            adjust_scale=adjust_scale
        )

        if not cc_res.success:
            return STEAlignmentResult(
                success=False,
                status="cccorelib_error",
                status_message=f"CCCoreLib registration failed: {cc_res.error_message}",
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
                scale=1.0,
                transformation_matrix=np.eye(4, dtype=np.float64),
                rms_error=-1.0,
                residuals=[],
                control_point_count=count,
                alignment_method=method_name
            )

        R = cc_res.rotation
        t = cc_res.translation
        s = cc_res.scale if adjust_scale else 1.0

        if not np.isfinite(s) or s <= 0:
            return STEAlignmentResult(
                success=False,
                status="invalid_scale",
                status_message=f"Recovered invalid scale factor: {s}",
                rotation=R,
                translation=t,
                scale=s,
                transformation_matrix=cc_res.transform,
                rms_error=-1.0,
                residuals=[],
                control_point_count=count,
                alignment_method=method_name
            )

        transform_4x4 = np.eye(4, dtype=np.float64)
        transform_4x4[:3, :3] = s * R
        transform_4x4[:3, 3] = t

        predicted_photo = (pts_lidar @ (s * R).T) + t
        residual_vectors = pts_photo - predicted_photo
        raw_residual_distances = np.linalg.norm(residual_vectors, axis=1)
        rms_error = float(np.sqrt(np.mean(raw_residual_distances ** 2)))

        cp_ids = control_point_ids if control_point_ids is not None else [f"CP{i+1}" for i in range(count)]

        return STEAlignmentResult(
            success=True,
            status="complete",
            status_message="Alignment solved successfully via CCCoreLib.",
            rotation=R,
            translation=t,
            scale=s,
            transformation_matrix=transform_4x4,
            rms_error=rms_error,
            residuals=[float(r) for r in raw_residual_distances],
            residual_vectors=[residual_vectors[i] for i in range(count)],
            control_point_count=count,
            control_point_ids=cp_ids,
            alignment_method=method_name
        )

    @classmethod
    def solve_from_manager(
        cls,
        cp_manager: STEControlPointManager,
        adjust_scale: bool = True
    ) -> STEAlignmentResult:
        """
        Solve alignment directly from an STEControlPointManager instance.
        """
        pts_lidar, pts_photo, ids = cp_manager.get_complete_pairs()
        return cls.solve(
            lidar_points=pts_lidar,
            photogrammetry_points=pts_photo,
            adjust_scale=adjust_scale,
            control_point_ids=ids
        )


class STEICPRefinementService:
    """
    Dedicated ICP Refinement Service for the Spatial Texture Engine.
    Refines an initial control-point alignment using CCCoreLib ICP.
    """

    @classmethod
    def refine(
        cls,
        source_lidar_points: np.ndarray,
        target_photogrammetry_points: np.ndarray,
        initial_alignment: Optional[STEAlignmentResult],
        settings: Optional[STEICPRefinementSettings] = None,
        cp_manager: Optional[STEControlPointManager] = None
    ) -> STEICPRefinementResult:
        """
        Perform ICP fine registration on LiDAR points against Photogrammetry geometry,
        starting strictly from the initial control-point transformation.

        Args:
            source_lidar_points: (N, 3) raw LiDAR points / surface vertices.
            target_photogrammetry_points: (M, 3) Photogrammetry dense cloud / vertices.
            initial_alignment: Valid STEAlignmentResult from the control-point stage.
            settings: STEICPRefinementSettings configuration.
            cp_manager: Optional STEControlPointManager to compute per-marker residuals before & after.

        Returns:
            STEICPRefinementResult with composed transform and diagnostic metrics.
        """
        if settings is None:
            settings = STEICPRefinementSettings()

        # Step 1: Validate initial alignment existence
        if initial_alignment is None or not initial_alignment.success:
            return STEICPRefinementResult(
                success=False,
                status="no_initial_alignment",
                status_message="Perform control-point alignment before ICP refinement.",
                initial_rms=initial_alignment.rms_error if initial_alignment else -1.0,
                final_rms=-1.0,
                scale=1.0,
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
                transformation_matrix=np.eye(4, dtype=np.float64),
                icp_delta_transform=np.eye(4, dtype=np.float64),
                icp_scale_delta=1.0
            )

        # Step 2: Validate input point geometries
        pts_lidar_raw = np.ascontiguousarray(source_lidar_points, dtype=np.float64)
        pts_photo = np.ascontiguousarray(target_photogrammetry_points, dtype=np.float64)

        if pts_lidar_raw.ndim != 2 or pts_lidar_raw.shape[1] != 3 or pts_photo.ndim != 2 or pts_photo.shape[1] != 3:
            return STEICPRefinementResult(
                success=False,
                status="invalid_point_dimensions",
                status_message=f"Point cloud arrays must be (N, 3). Got {pts_lidar_raw.shape} and {pts_photo.shape}.",
                initial_rms=initial_alignment.rms_error,
                final_rms=-1.0,
                scale=initial_alignment.scale,
                rotation=initial_alignment.rotation,
                translation=initial_alignment.translation,
                transformation_matrix=initial_alignment.transformation_matrix,
                icp_delta_transform=np.eye(4, dtype=np.float64),
                icp_scale_delta=1.0
            )

        if pts_lidar_raw.shape[0] < 10 or pts_photo.shape[0] < 10:
            return STEICPRefinementResult(
                success=False,
                status="insufficient_points",
                status_message=f"Insufficient points for ICP: {pts_lidar_raw.shape[0]} LiDAR vs {pts_photo.shape[0]} Photogrammetry points.",
                initial_rms=initial_alignment.rms_error,
                final_rms=-1.0,
                scale=initial_alignment.scale,
                rotation=initial_alignment.rotation,
                translation=initial_alignment.translation,
                transformation_matrix=initial_alignment.transformation_matrix,
                icp_delta_transform=np.eye(4, dtype=np.float64),
                icp_scale_delta=1.0
            )

        if not np.all(np.isfinite(pts_lidar_raw)) or not np.all(np.isfinite(pts_photo)):
            return STEICPRefinementResult(
                success=False,
                status="non_finite_points",
                status_message="Point clouds contain non-finite coordinates (NaN or Inf).",
                initial_rms=initial_alignment.rms_error,
                final_rms=-1.0,
                scale=initial_alignment.scale,
                rotation=initial_alignment.rotation,
                translation=initial_alignment.translation,
                transformation_matrix=initial_alignment.transformation_matrix,
                icp_delta_transform=np.eye(4, dtype=np.float64),
                icp_scale_delta=1.0
            )

        # Step 3: Apply initial control-point transformation to LiDAR points (source/moving data)
        # P_init = s_init * R_init * P_lidar + t_init
        pts_lidar_initial = initial_alignment.apply(pts_lidar_raw)

        # Step 4: Focus reference Photogrammetry points to LiDAR Region of Interest (ROI)
        # Prevents distant background/ground/room geometry from starving the object of samples
        min_b = np.min(pts_lidar_initial, axis=0)
        max_b = np.max(pts_lidar_initial, axis=0)
        extent = max_b - min_b
        pad = np.maximum(extent * settings.roi_margin_ratio, 0.5)
        roi_min = min_b - pad
        roi_max = max_b + pad

        in_roi = np.all((pts_photo >= roi_min) & (pts_photo <= roi_max), axis=1)
        if np.sum(in_roi) >= 100:
            pts_photo_roi = pts_photo[in_roi]
        else:
            pts_photo_roi = pts_photo

        # Step 5: Subsample if necessary for memory and performance efficiency
        if pts_lidar_initial.shape[0] > settings.sampling_limit:
            idx_l = np.random.choice(pts_lidar_initial.shape[0], settings.sampling_limit, replace=False)
            pts_data = pts_lidar_initial[idx_l]
        else:
            pts_data = pts_lidar_initial

        if pts_photo_roi.shape[0] > settings.sampling_limit:
            idx_p = np.random.choice(pts_photo_roi.shape[0], settings.sampling_limit, replace=False)
            pts_model = pts_photo_roi[idx_p]
        else:
            pts_model = pts_photo_roi

        # Step 6: Execute CCCoreLib ICP
        icp_res = ccr.refine_icp(
            model_pts=pts_model,
            data_pts=pts_data,
            adjust_scale=settings.adjust_scale,
            min_rms_decrease=settings.min_rms_decrease,
            max_iterations=settings.max_iterations,
            sampling_limit=settings.sampling_limit,
            overlap_ratio=settings.overlap_ratio
        )

        if not icp_res.success:
            return STEICPRefinementResult(
                success=False,
                status="icp_failed",
                status_message=f"CCCoreLib ICP failed: {icp_res.error_message}. Preserving initial control-point alignment.",
                initial_rms=initial_alignment.rms_error,
                final_rms=-1.0,
                scale=initial_alignment.scale,
                rotation=initial_alignment.rotation,
                translation=initial_alignment.translation,
                transformation_matrix=initial_alignment.transformation_matrix,
                icp_delta_transform=np.eye(4, dtype=np.float64),
                icp_scale_delta=1.0
            )

        # Step 7: Extract incremental ICP transformation
        R_delta = icp_res.rotation
        t_delta = icp_res.translation
        s_delta = icp_res.scale if settings.adjust_scale else 1.0
        T_delta = icp_res.transform

        # Step 8: Mathematically compose transformations
        # P_final = s_delta * R_delta * (s_init * R_init * P_lidar + t_init) + t_delta
        # T_final = T_delta @ T_init
        T_init = initial_alignment.transformation_matrix
        T_final = T_delta @ T_init

        s_final = s_delta * initial_alignment.scale
        R_final = R_delta @ initial_alignment.rotation
        t_final = (s_delta * (R_delta @ initial_alignment.translation)) + t_delta

        # Step 9: Bad result protection & validation
        # 9a. Check determinant of rotation matrix (must be +1, proper rotation)
        det_R = np.linalg.det(R_final)
        if not np.isfinite(det_R) or abs(det_R - 1.0) > 0.05:
            return STEICPRefinementResult(
                success=False,
                status="invalid_rotation",
                status_message=f"ICP resulted in invalid rotation matrix (det={det_R:.4f}). Preserving initial alignment.",
                initial_rms=initial_alignment.rms_error,
                final_rms=-1.0,
                scale=initial_alignment.scale,
                rotation=initial_alignment.rotation,
                translation=initial_alignment.translation,
                transformation_matrix=initial_alignment.transformation_matrix,
                icp_delta_transform=T_delta,
                icp_scale_delta=s_delta
            )

        # 9b. Check scale positivity and finiteness
        if not np.isfinite(s_final) or s_final <= 0:
            return STEICPRefinementResult(
                success=False,
                status="invalid_scale",
                status_message=f"ICP resulted in invalid scale factor ({s_final}). Preserving initial alignment.",
                initial_rms=initial_alignment.rms_error,
                final_rms=-1.0,
                scale=initial_alignment.scale,
                rotation=initial_alignment.rotation,
                translation=initial_alignment.translation,
                transformation_matrix=initial_alignment.transformation_matrix,
                icp_delta_transform=T_delta,
                icp_scale_delta=s_delta
            )

        # 9c. Check translation runaway drift
        t_drift = np.linalg.norm(t_delta)
        if t_drift > settings.max_allowed_translation_drift:
            return STEICPRefinementResult(
                success=False,
                status="excessive_drift",
                status_message=f"ICP drifted excessively ({t_drift:.2f}m > {settings.max_allowed_translation_drift:.2f}m). Preserving initial alignment.",
                initial_rms=initial_alignment.rms_error,
                final_rms=-1.0,
                scale=initial_alignment.scale,
                rotation=initial_alignment.rotation,
                translation=initial_alignment.translation,
                transformation_matrix=initial_alignment.transformation_matrix,
                icp_delta_transform=T_delta,
                icp_scale_delta=s_delta
            )

        # 9d. Check final RMS validity
        final_rms = icp_res.rms
        if not np.isfinite(final_rms) or final_rms < 0:
            return STEICPRefinementResult(
                success=False,
                status="invalid_rms",
                status_message="ICP produced invalid final RMS error. Preserving initial alignment.",
                initial_rms=initial_alignment.rms_error,
                final_rms=-1.0,
                scale=initial_alignment.scale,
                rotation=initial_alignment.rotation,
                translation=initial_alignment.translation,
                transformation_matrix=initial_alignment.transformation_matrix,
                icp_delta_transform=T_delta,
                icp_scale_delta=s_delta
            )

        # Step 10: Recalculate control-point residuals with final composed transform
        initial_cp_residuals = initial_alignment.residuals
        final_cp_residuals = []
        cp_ids = initial_alignment.control_point_ids

        if cp_manager is not None:
            lidar_cps, photo_cps, cp_ids = cp_manager.get_complete_pairs()
            if lidar_cps.shape[0] > 0:
                p_final_cps = (lidar_cps @ (s_final * R_final).T) + t_final
                raw_final_res = np.linalg.norm(photo_cps - p_final_cps, axis=1)
                final_cp_residuals = [float(r) for r in raw_final_res]
        elif len(initial_alignment.control_point_ids) > 0 and len(initial_alignment.residuals) > 0:
            final_cp_residuals = initial_alignment.residuals.copy()

        # Step 11: Control point semantic anchor protection
        # If ICP drifted away from semantic control points (e.g. slid along planar surfaces), reject it
        if initial_alignment.rms_error > 0 and len(final_cp_residuals) > 0:
            final_cp_rms = float(np.sqrt(np.mean(np.square(final_cp_residuals))))
            max_allowed_rms = max(initial_alignment.rms_error * settings.max_allowed_cp_degradation_factor, initial_alignment.rms_error + 0.15)
            if final_cp_rms > max_allowed_rms:
                return STEICPRefinementResult(
                    success=False,
                    status="control_points_drifted",
                    status_message=f"ICP rejected: Control-point error degraded from {initial_alignment.rms_error*100.0:.1f} cm to {final_cp_rms*100.0:.1f} cm. Preserving initial Horn alignment.",
                    initial_rms=initial_alignment.rms_error,
                    final_rms=final_rms,
                    scale=initial_alignment.scale,
                    rotation=initial_alignment.rotation,
                    translation=initial_alignment.translation,
                    transformation_matrix=initial_alignment.transformation_matrix,
                    icp_delta_transform=T_delta,
                    icp_scale_delta=s_delta,
                    initial_cp_residuals=initial_cp_residuals,
                    final_cp_residuals=final_cp_residuals,
                    control_point_ids=cp_ids
                )

        return STEICPRefinementResult(
            success=True,
            status="complete",
            status_message="ICP refinement completed successfully.",
            initial_rms=initial_alignment.rms_error,
            final_rms=final_rms,
            scale=s_final,
            rotation=R_final,
            translation=t_final,
            transformation_matrix=T_final,
            icp_delta_transform=T_delta,
            icp_scale_delta=s_delta,
            iterations=settings.max_iterations,
            registered_point_count=icp_res.point_count,
            initial_cp_residuals=initial_cp_residuals,
            final_cp_residuals=final_cp_residuals,
            control_point_ids=cp_ids
        )


class STEAlignmentState:
    """
    Non-destructive alignment state manager.
    Preserves raw original LiDAR geometry and coordinates byte-for-byte.
    Manages preview transforms and committed alignment state for downstream STE stages.
    """
    def __init__(self):
        self._initial_result: Optional[STEAlignmentResult] = None
        self._icp_result: Optional[STEICPRefinementResult] = None
        self._current_result: Optional[STEAlignmentResult] = None
        self._preview_active: bool = False
        self._accepted: bool = False
        self._mode: str = "scale_aware"  # "scale_aware" or "rigid"

    @property
    def initial_result(self) -> Optional[STEAlignmentResult]:
        return self._initial_result

    @property
    def icp_result(self) -> Optional[STEICPRefinementResult]:
        return self._icp_result

    @property
    def result(self) -> Optional[STEAlignmentResult]:
        return self._current_result

    @property
    def preview_active(self) -> bool:
        return self._preview_active

    @property
    def is_accepted(self) -> bool:
        return self._accepted

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        if mode not in ("scale_aware", "rigid"):
            raise ValueError(f"Invalid alignment mode: {mode}. Must be 'scale_aware' or 'rigid'.")
        self._mode = mode

    def set_initial_result(self, result: STEAlignmentResult):
        """Set initial control-point alignment result."""
        self._initial_result = result
        self._icp_result = None
        self._current_result = result
        if result.success:
            self._preview_active = True

    def set_icp_result(self, icp_res: STEICPRefinementResult):
        """Set ICP refinement result and update active transform."""
        self._icp_result = icp_res
        if icp_res.success:
            self._current_result = icp_res.to_alignment_result()
            self._preview_active = True

    def reset(self):
        """
        Reset alignment state:
        - Clears initial and ICP results
        - Reverts active transform to Identity 4x4
        - Clears preview flag
        - Clears accepted flag
        - Retains raw underlying geometry unmodified
        """
        self._initial_result = None
        self._icp_result = None
        self._current_result = None
        self._preview_active = False
        self._accepted = False

    def accept(self) -> bool:
        """
        Commit the current alignment state for downstream STE operations (e.g. texture transfer).
        Geometry is NOT destructively modified.
        """
        if self._current_result and self._current_result.success:
            self._accepted = True
            return True
        return False

    def get_preview_transform(self) -> np.ndarray:
        """
        Get the current 4x4 preview transformation matrix.
        Returns Identity 4x4 if preview is not active or result is not valid.
        """
        if self._preview_active and self._current_result and self._current_result.success:
            return self._current_result.transformation_matrix.copy()
        return np.eye(4, dtype=np.float64)

    def get_committed_transform(self) -> np.ndarray:
        """
        Get the committed 4x4 transformation matrix.
        Returns Identity 4x4 if not accepted.
        """
        if self._accepted and self._current_result and self._current_result.success:
            return self._current_result.transformation_matrix.copy()
        return np.eye(4, dtype=np.float64)
