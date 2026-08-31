"""
Spatial Texture Engine (STE) - Alignment Quality Gate & Diagnostics
===================================================================

This module provides a read-only quality-analysis layer for evaluating whether
an alignment (Control-Point or ICP-refined) between LiDAR and Photogrammetry geometry
is of sufficient geometric precision to proceed to texture transfer.

Key Capabilities:
-----------------
1. Geometry Overlap & AABB Analysis (center distance, 1D overlaps, 3D IoU/overlap ratio).
2. Surface Distance Analysis (min, mean, median, P75, P90, P95, P99, max, cumulative tolerance distributions).
3. Directional / Spatial Binning Error Analysis (detects anisotropic drift, floor vs wall mismatches).
4. Control Point vs Surface Agreement Contrast (guards against overfitted/inaccurate manual markers).
5. Failure Mode Detection (POSSIBLE_SCALE_MISMATCH, LOW_SURFACE_OVERLAP, ICP_DEGRADATION, etc.).
6. Configurable Quality Rating (EXCELLENT, GOOD, ACCEPTABLE, POOR, VERY_POOR, INVALID).
7. Texture Projection Readiness Decision (READY_FOR_TEXTURE_TRANSFER vs NOT_READY_FOR_TEXTURE_TRANSFER).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ste_alignment import STEAlignmentResult, STEICPRefinementResult


class AlignmentQualityRating(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    VERY_POOR = "VERY_POOR"
    INVALID = "INVALID"


class TextureTransferReadiness(str, Enum):
    READY_FOR_TEXTURE_TRANSFER = "READY_FOR_TEXTURE_TRANSFER"
    NOT_READY_FOR_TEXTURE_TRANSFER = "NOT_READY_FOR_TEXTURE_TRANSFER"


@dataclass
class QualityGateThresholds:
    """
    Centralized, configurable thresholds for alignment quality classification.
    """
    # Overlap thresholds (3D Bounding Box / Spatial IoU)
    min_acceptable_overlap: float = 0.30     # 30% min bbox intersection
    good_overlap: float = 0.60               # 60% good bbox intersection
    excellent_overlap: float = 0.80          # 80% excellent bbox intersection

    # Surface distance thresholds (in meters)
    excellent_median_dist: float = 0.025     # <= 2.5 cm
    good_median_dist: float = 0.060          # <= 6.0 cm
    acceptable_median_dist: float = 0.120    # <= 12.0 cm
    poor_median_dist: float = 0.250          # <= 25.0 cm

    excellent_p95_dist: float = 0.080        # <= 8.0 cm
    good_p95_dist: float = 0.180             # <= 18.0 cm
    acceptable_p95_dist: float = 0.350       # <= 35.0 cm

    # Percentage within key tolerance (0.05m = 5cm)
    min_pct_within_5cm_for_ready: float = 65.0  # 65% of surface must be within 5cm

    # Control point vs Surface discrepancy ratio threshold
    max_cp_to_surface_discrepancy: float = 4.0  # Flag if surface median is > 4x CP RMS


@dataclass
class OverlapMetrics:
    """Bounding box and volumetric intersection metrics."""
    photo_aabb_min: np.ndarray
    photo_aabb_max: np.ndarray
    lidar_aabb_min: np.ndarray
    lidar_aabb_max: np.ndarray
    center_to_center_distance: float
    overlap_x: float
    overlap_y: float
    overlap_z: float
    intersection_volume: float
    union_volume: float
    overlap_ratio: float  # IoU of bounding boxes (0.0 to 1.0)


@dataclass
class SurfaceDistanceMetrics:
    """Statistical distribution of LiDAR-to-Photogrammetry surface distances."""
    sample_count: int
    min_dist: float
    mean_dist: float
    median_dist: float
    p75_dist: float
    p90_dist: float
    p95_dist: float
    p99_dist: float
    max_dist: float
    pct_within_0_01m: float   # 1 cm
    pct_within_0_025m: float  # 2.5 cm
    pct_within_0_05m: float   # 5 cm
    pct_within_0_10m: float   # 10 cm
    pct_within_0_25m: float   # 25 cm
    pct_within_0_50m: float   # 50 cm


@dataclass
class SpatialRegionError:
    """Error metrics in a specific 3D spatial quadrant / bin."""
    region_id: str
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    point_count: int
    mean_dist: float
    median_dist: float
    p95_dist: float
    max_dist: float


@dataclass
class STEAlignmentQualityReport:
    """
    Comprehensive, non-destructive alignment quality evaluation.
    """
    rating: AlignmentQualityRating
    readiness: TextureTransferReadiness

    # High-level summary strings
    summary: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)

    # Detailed metrics
    overlap_metrics: Optional[OverlapMetrics] = None
    surface_metrics: Optional[SurfaceDistanceMetrics] = None
    spatial_errors: List[SpatialRegionError] = field(default_factory=list)

    # Comparison metrics
    control_point_rms: float = -1.0
    surface_median_dist: float = -1.0
    surface_p95_dist: float = -1.0
    scale: float = 1.0

    # Before vs After ICP comparison (if applicable)
    icp_improved_surface: Optional[bool] = None
    surface_dist_change_pct: float = 0.0

    @property
    def is_ready(self) -> bool:
        return self.readiness == TextureTransferReadiness.READY_FOR_TEXTURE_TRANSFER


class STEAlignmentQualityGate:
    """
    Read-only quality gate and diagnostic service for STE alignments.
    """

    @staticmethod
    def compute_overlap_metrics(
        pts_photo: np.ndarray,
        pts_lidar_aligned: np.ndarray
    ) -> OverlapMetrics:
        """
        Compute AABB bounds, center distance, and 3D intersection volume.
        """
        photo_min = np.min(pts_photo, axis=0)
        photo_max = np.max(pts_photo, axis=0)
        lidar_min = np.min(pts_lidar_aligned, axis=0)
        lidar_max = np.max(pts_lidar_aligned, axis=0)

        photo_center = (photo_min + photo_max) / 2.0
        lidar_center = (lidar_min + lidar_max) / 2.0
        center_dist = float(np.linalg.norm(photo_center - lidar_center))

        # 1D Overlap lengths
        inter_min = np.maximum(photo_min, lidar_min)
        inter_max = np.minimum(photo_max, lidar_max)
        overlap_dims = np.maximum(0.0, inter_max - inter_min)

        overlap_x = float(overlap_dims[0])
        overlap_y = float(overlap_dims[1])
        overlap_z = float(overlap_dims[2])

        inter_vol = float(overlap_x * overlap_y * overlap_z)
        vol_photo = float(np.prod(np.maximum(1e-6, photo_max - photo_min)))
        vol_lidar = float(np.prod(np.maximum(1e-6, lidar_max - lidar_min)))
        union_vol = vol_photo + vol_lidar - inter_vol

        overlap_ratio = float(inter_vol / union_vol) if union_vol > 1e-9 else 0.0

        return OverlapMetrics(
            photo_aabb_min=photo_min,
            photo_aabb_max=photo_max,
            lidar_aabb_min=lidar_min,
            lidar_aabb_max=lidar_max,
            center_to_center_distance=center_dist,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
            overlap_z=overlap_z,
            intersection_volume=inter_vol,
            union_volume=union_vol,
            overlap_ratio=overlap_ratio
        )

    @staticmethod
    def compute_surface_distances(
        pts_photo: np.ndarray,
        pts_lidar_aligned: np.ndarray,
        max_samples: int = 10000
    ) -> Tuple[SurfaceDistanceMetrics, np.ndarray, np.ndarray]:
        """
        Compute surface-to-surface distance distribution using KDTree.
        Returns SurfaceDistanceMetrics, sampled points, and distances array.
        """
        import open3d as o3d

        N = pts_lidar_aligned.shape[0]
        if N > max_samples:
            indices = np.random.choice(N, max_samples, replace=False)
            sampled_lidar = pts_lidar_aligned[indices]
        else:
            sampled_lidar = pts_lidar_aligned

        pcd_photo = o3d.geometry.PointCloud()
        pcd_photo.points = o3d.utility.Vector3dVector(pts_photo)
        kdtree = o3d.geometry.KDTreeFlann(pcd_photo)

        distances = []
        for i in range(len(sampled_lidar)):
            [k, idx, dist_sq] = kdtree.search_knn_vector_3d(sampled_lidar[i], 1)
            if k >= 1:
                distances.append(np.sqrt(dist_sq[0]))
            else:
                distances.append(999.0)

        dists = np.array(distances, dtype=np.float64)
        count = len(dists)

        metrics = SurfaceDistanceMetrics(
            sample_count=count,
            min_dist=float(np.min(dists)),
            mean_dist=float(np.mean(dists)),
            median_dist=float(np.median(dists)),
            p75_dist=float(np.percentile(dists, 75)),
            p90_dist=float(np.percentile(dists, 90)),
            p95_dist=float(np.percentile(dists, 95)),
            p99_dist=float(np.percentile(dists, 99)),
            max_dist=float(np.max(dists)),
            pct_within_0_01m=float(np.mean(dists <= 0.01) * 100.0),
            pct_within_0_025m=float(np.mean(dists <= 0.025) * 100.0),
            pct_within_0_05m=float(np.mean(dists <= 0.05) * 100.0),
            pct_within_0_10m=float(np.mean(dists <= 0.10) * 100.0),
            pct_within_0_25m=float(np.mean(dists <= 0.25) * 100.0),
            pct_within_0_50m=float(np.mean(dists <= 0.50) * 100.0)
        )

        return metrics, sampled_lidar, dists

    @staticmethod
    def compute_spatial_bins(
        sampled_lidar: np.ndarray,
        dists: np.ndarray,
        grid_bins_per_axis: int = 2
    ) -> List[SpatialRegionError]:
        """
        Divide LiDAR bounding volume into spatial bins to detect localized / directional error drift.
        """
        if sampled_lidar.shape[0] == 0:
            return []

        min_bound = np.min(sampled_lidar, axis=0)
        max_bound = np.max(sampled_lidar, axis=0)
        extent = max_bound - min_bound + 1e-6
        bin_size = extent / grid_bins_per_axis

        region_errors = []
        for ix in range(grid_bins_per_axis):
            for iy in range(grid_bins_per_axis):
                for iz in range(grid_bins_per_axis):
                    b_min = min_bound + np.array([ix, iy, iz]) * bin_size
                    b_max = b_min + bin_size
                    mask = np.all((sampled_lidar >= b_min) & (sampled_lidar < b_max), axis=1)
                    count = int(np.sum(mask))
                    if count >= 5:
                        bin_dists = dists[mask]
                        region_errors.append(SpatialRegionError(
                            region_id=f"Bin_{ix}{iy}{iz}",
                            bounds_min=b_min,
                            bounds_max=b_max,
                            point_count=count,
                            mean_dist=float(np.mean(bin_dists)),
                            median_dist=float(np.median(bin_dists)),
                            p95_dist=float(np.percentile(bin_dists, 95)),
                            max_dist=float(np.max(bin_dists))
                        ))
        return region_errors

    @classmethod
    def evaluate(
        cls,
        source_lidar_points: np.ndarray,
        target_photogrammetry_points: np.ndarray,
        alignment_result: STEAlignmentResult,
        initial_alignment_result: Optional[STEAlignmentResult] = None,
        thresholds: Optional[QualityGateThresholds] = None,
        max_samples: int = 10000
    ) -> STEAlignmentQualityReport:
        """
        Comprehensive quality assessment of an alignment.
        Does NOT modify geometry. Returns diagnostic report.
        """
        if thresholds is None:
            thresholds = QualityGateThresholds()

        reasons = []
        warnings = []
        failure_modes = []

        # 1. Basic Validity Checks
        if not alignment_result.success:
            return STEAlignmentQualityReport(
                rating=AlignmentQualityRating.INVALID,
                readiness=TextureTransferReadiness.NOT_READY_FOR_TEXTURE_TRANSFER,
                summary="Alignment computation failed or has invalid status.",
                reasons=["Alignment result is marked as unsuccessful."],
                failure_modes=["DEGENERATE_ALIGNMENT"]
            )

        if not np.isfinite(alignment_result.scale) or alignment_result.scale <= 0:
            return STEAlignmentQualityReport(
                rating=AlignmentQualityRating.INVALID,
                readiness=TextureTransferReadiness.NOT_READY_FOR_TEXTURE_TRANSFER,
                summary="Alignment has invalid non-positive or non-finite scale factor.",
                reasons=[f"Invalid scale factor: {alignment_result.scale}"],
                failure_modes=["POSSIBLE_SCALE_MISMATCH"]
            )

        pts_lidar = np.ascontiguousarray(source_lidar_points, dtype=np.float64)
        pts_photo = np.ascontiguousarray(target_photogrammetry_points, dtype=np.float64)

        if pts_lidar.shape[0] < 10 or pts_photo.shape[0] < 10:
            return STEAlignmentQualityReport(
                rating=AlignmentQualityRating.INVALID,
                readiness=TextureTransferReadiness.NOT_READY_FOR_TEXTURE_TRANSFER,
                summary="Insufficient point samples in geometry for alignment validation.",
                reasons=["Point count in LiDAR or Photogrammetry cloud is too low (<10)."],
                failure_modes=["INSUFFICIENT_CONTROL_POINTS"]
            )

        # 2. Transform LiDAR points into Photogrammetry coordinate space
        pts_lidar_aligned = alignment_result.apply(pts_lidar)

        # 3. Overlap Analysis
        overlap = cls.compute_overlap_metrics(pts_photo, pts_lidar_aligned)

        # 4. Surface Distance Distribution
        surface_dist, sampled_lidar, dists = cls.compute_surface_distances(
            pts_photo=pts_photo,
            pts_lidar_aligned=pts_lidar_aligned,
            max_samples=max_samples
        )

        # 5. Spatial Regional Error Drift
        spatial_bins = cls.compute_spatial_bins(sampled_lidar, dists)

        # 6. Failure Mode Diagnostics
        # 6a. Overlap check
        if overlap.overlap_ratio < thresholds.min_acceptable_overlap:
            failure_modes.append("LOW_SURFACE_OVERLAP")
            warnings.append(f"Surface bounding box overlap is low ({overlap.overlap_ratio*100.0:.1f}% < {thresholds.min_acceptable_overlap*100.0:.1f}%).")
        else:
            reasons.append(f"Surface bounding box overlap is sufficient ({overlap.overlap_ratio*100.0:.1f}%).")

        # 6b. Large surface distance check
        if surface_dist.median_dist > thresholds.acceptable_median_dist:
            failure_modes.append("LARGE_SURFACE_DISTANCE")
            warnings.append(f"Median surface distance is high ({surface_dist.median_dist*100.0:.1f} cm > {thresholds.acceptable_median_dist*100.0:.1f} cm).")
        else:
            reasons.append(f"Median surface distance is within acceptable tolerance ({surface_dist.median_dist*100.0:.1f} cm).")

        # 6c. Control Point vs Surface Discrepancy (overfitting / inaccurate markers)
        cp_rms = alignment_result.rms_error
        if cp_rms > 0.0:
            if surface_dist.median_dist > (cp_rms * thresholds.max_cp_to_surface_discrepancy) and surface_dist.median_dist > 0.05:
                failure_modes.append("LOCAL_ALIGNMENT_FAILURE")
                warnings.append(
                    f"Discrepancy detected: Control-point RMS is {cp_rms*100.0:.1f} cm but surface median distance is {surface_dist.median_dist*100.0:.1f} cm."
                )

        # 6d. Spatial Regional Variation (Anisotropic drift detection)
        if len(spatial_bins) >= 2:
            bin_medians = [b.median_dist for b in spatial_bins]
            min_bin_med = min(bin_medians)
            max_bin_med = max(bin_medians)
            if max_bin_med > (min_bin_med * 3.5) and max_bin_med > 0.10:
                failure_modes.append("LOCAL_ALIGNMENT_FAILURE")
                warnings.append(
                    f"Anisotropic error drift: regional error varies from {min_bin_med*100.0:.1f} cm to {max_bin_med*100.0:.1f} cm across spatial bins."
                )

        # 6e. Scale plausibility check
        if alignment_result.scale < 0.01 or alignment_result.scale > 100.0:
            failure_modes.append("POSSIBLE_SCALE_MISMATCH")
            warnings.append(f"Unusually extreme scale factor recovered: {alignment_result.scale:.4f}x.")

        # 6f. Check if ICP degraded alignment compared to initial
        icp_improved = None
        change_pct = 0.0
        if initial_alignment_result is not None and initial_alignment_result.success:
            pts_lidar_init = initial_alignment_result.apply(pts_lidar)
            init_surf, _, _ = cls.compute_surface_distances(pts_photo, pts_lidar_init, max_samples=min(2000, max_samples))
            if init_surf.median_dist > 1e-6:
                change_pct = float(((surface_dist.median_dist - init_surf.median_dist) / init_surf.median_dist) * 100.0)
            elif surface_dist.median_dist > 0.01:
                change_pct = 100.0

            icp_improved = (surface_dist.median_dist <= (init_surf.median_dist + 1e-4))
            if not icp_improved and (change_pct > 15.0 or surface_dist.median_dist > (init_surf.median_dist + 0.02)):
                failure_modes.append("ICP_DEGRADATION")
                warnings.append(f"ICP refinement degraded surface median distance from {init_surf.median_dist*100.0:.1f} cm to {surface_dist.median_dist*100.0:.1f} cm.")

        # 7. Quality Rating Classification
        if (surface_dist.median_dist <= thresholds.excellent_median_dist and
            surface_dist.p95_dist <= thresholds.excellent_p95_dist and
            overlap.overlap_ratio >= thresholds.good_overlap):
            rating = AlignmentQualityRating.EXCELLENT
            summary = "Excellent spatial alignment across geometry."
        elif (surface_dist.median_dist <= thresholds.good_median_dist and
              surface_dist.p95_dist <= thresholds.good_p95_dist and
              overlap.overlap_ratio >= thresholds.min_acceptable_overlap):
            rating = AlignmentQualityRating.GOOD
            summary = "Good spatial alignment, ready for texture transfer."
        elif (surface_dist.median_dist <= thresholds.acceptable_median_dist and
              surface_dist.p95_dist <= thresholds.acceptable_p95_dist and
              overlap.overlap_ratio >= thresholds.min_acceptable_overlap):
            rating = AlignmentQualityRating.ACCEPTABLE
            summary = "Acceptable spatial alignment with minor regional deviations."
        elif surface_dist.median_dist <= thresholds.poor_median_dist:
            rating = AlignmentQualityRating.POOR
            summary = "Poor spatial alignment. Geometry exhibits notable offsets."
        else:
            rating = AlignmentQualityRating.VERY_POOR
            summary = "Very poor alignment. Severe spatial mismatch between LiDAR and Photogrammetry."

        # 8. Readiness Decision
        if (rating in (AlignmentQualityRating.EXCELLENT, AlignmentQualityRating.GOOD, AlignmentQualityRating.ACCEPTABLE) and
            surface_dist.pct_within_0_05m >= thresholds.min_pct_within_5cm_for_ready and
            overlap.overlap_ratio >= thresholds.min_acceptable_overlap and
            "LOW_SURFACE_OVERLAP" not in failure_modes and
            "LARGE_SURFACE_DISTANCE" not in failure_modes and
            "POSSIBLE_SCALE_MISMATCH" not in failure_modes and
            "ICP_DEGRADATION" not in failure_modes):
            readiness = TextureTransferReadiness.READY_FOR_TEXTURE_TRANSFER
            reasons.append("Geometry meets all criteria for accurate texture projection.")
        else:
            readiness = TextureTransferReadiness.NOT_READY_FOR_TEXTURE_TRANSFER
            if surface_dist.pct_within_0_05m < thresholds.min_pct_within_5cm_for_ready:
                warnings.append(f"Only {surface_dist.pct_within_0_05m:.1f}% of surface is within 5cm tolerance (required >= {thresholds.min_pct_within_5cm_for_ready:.1f}%).")

        return STEAlignmentQualityReport(
            rating=rating,
            readiness=readiness,
            summary=summary,
            reasons=reasons,
            warnings=warnings,
            failure_modes=failure_modes,
            overlap_metrics=overlap,
            surface_metrics=surface_dist,
            spatial_errors=spatial_bins,
            control_point_rms=cp_rms,
            surface_median_dist=surface_dist.median_dist,
            surface_p95_dist=surface_dist.p95_dist,
            scale=alignment_result.scale,
            icp_improved_surface=icp_improved,
            surface_dist_change_pct=change_pct
        )
