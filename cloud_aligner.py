"""
Cloud Aligner Module for Proximap
Aligns an unscaled external reference point cloud to COLMAP's dense point cloud
using initial scale estimation, FPFH feature-based RANSAC coarse registration,
and Point-to-Plane ICP refinement (CPU-only).
"""

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

@dataclass
class AlignResult:
    transform: np.ndarray             # 4x4 homogenous similarity transformation matrix
    initial_scale: float               # Initial scale factor applied to reference cloud
    final_scale: float                 # Final total scale ratio
    fitness: float                     # Fraction of matching inliers
    inlier_rmse: float                 # Root Mean Squared Error of inliers
    normalized_rmse: float             # RMSE normalized by median point spacing
    overlap_pct: float                 # Estimated overlap percentage
    success: bool                      # Whether alignment met confidence threshold
    warnings: List[str] = field(default_factory=list)


def compute_median_point_spacing(cloud, max_samples: int = 5000) -> float:
    """Computes median nearest-neighbor distance to determine point cloud density."""
    import open3d as o3d
    pts = np.asarray(cloud.points)
    if len(pts) == 0:
        return 0.01

    if len(pts) > max_samples:
        indices = np.random.choice(len(pts), max_samples, replace=False)
        sample_pts = pts[indices]
    else:
        sample_pts = pts

    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(sample_pts)
    
    kdtree = o3d.geometry.KDTreeFlann(pcd_sample)
    distances = []
    for i in range(len(sample_pts)):
        [k, idx, dist_sq] = kdtree.search_knn_vector_3d(sample_pts[i], 2)
        if k >= 2 and dist_sq[1] > 0:
            distances.append(np.sqrt(dist_sq[1]))

    if not distances:
        return 0.01

    return float(np.median(distances))


def align_to_dense(
    ref_cloud,
    dense_cloud,
    log_fn: Optional[Callable[[str], None]] = None
) -> AlignResult:
    """
    Aligns ref_cloud to dense_cloud.
    
    Pipeline:
    1. Estimate bounding-box extent ratio and pre-scale ref_cloud.
    2. Estimate median point spacing to set dynamic voxel sizes.
    3. Extract FPFH features & run RANSAC global registration.
    4. Refine with Point-to-Plane ICP.
    5. Evaluate alignment confidence and return transform matrix.
    """
    import open3d as o3d

    def log(msg: str):
        if log_fn:
            log_fn(msg)

    warnings = []

    if len(ref_cloud.points) < 50 or len(dense_cloud.points) < 50:
        return AlignResult(
            transform=np.identity(4), initial_scale=1.0, final_scale=1.0,
            fitness=0.0, inlier_rmse=999.0, normalized_rmse=999.0, overlap_pct=0.0,
            success=False, warnings=["Insufficient points in clouds for alignment."]
        )

    # -------------------------------------------------------------------------
    # STAGE 0: Initial Scale Estimation & Pre-scaling
    # -------------------------------------------------------------------------
    bbox_ref = ref_cloud.get_axis_aligned_bounding_box().get_extent()
    bbox_dense = dense_cloud.get_axis_aligned_bounding_box().get_extent()

    diag_ref = float(np.linalg.norm(bbox_ref))
    diag_dense = float(np.linalg.norm(bbox_dense))

    if diag_ref <= 0 or diag_dense <= 0:
        initial_scale = 1.0
    else:
        initial_scale = diag_dense / diag_ref

    log(f"[ALIGN] Initial scale estimation: ref diagonal = {diag_ref:.3f}, dense diagonal = {diag_dense:.3f} -> scale = {initial_scale:.4f}")

    # Create scaled working copy of reference cloud and align centroids
    center_ref = np.asarray(ref_cloud.get_center())
    center_dense = np.asarray(dense_cloud.get_center())
    initial_translation = center_dense - (center_ref * initial_scale)

    ref_scaled = copy.deepcopy(ref_cloud)
    ref_scaled.scale(initial_scale, center=center_ref)
    ref_scaled.translate(initial_translation)

    # -------------------------------------------------------------------------
    # STAGE 1: Dynamic Resolution & FPFH Feature RANSAC Coarse Registration
    # -------------------------------------------------------------------------
    d_dense = compute_median_point_spacing(dense_cloud)
    voxel_size = max(d_dense * 2.5, 0.002)

    log(f"[ALIGN] Point spacing: {d_dense:.4f} | Voxel size for coarse alignment: {voxel_size:.4f}")

    # Downsample both clouds
    dense_down = dense_cloud.voxel_down_sample(voxel_size)
    ref_down = ref_scaled.voxel_down_sample(voxel_size)

    # Estimate normals for feature calculation
    radius_normal = voxel_size * 2.5
    dense_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    ref_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH features
    radius_feature = voxel_size * 5.0
    fpfh_dense = o3d.pipelines.registration.compute_fpfh_feature(
        dense_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    fpfh_ref = o3d.pipelines.registration.compute_fpfh_feature(
        ref_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    # RANSAC Coarse Registration
    distance_threshold = voxel_size * 4.0
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        ref_down, dense_down, fpfh_ref, fpfh_dense, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    log(f"[ALIGN] RANSAC Coarse Registration: fitness = {ransac_result.fitness:.3f}, RMSE = {ransac_result.inlier_rmse:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 2: Point-to-Plane ICP Refinement
    # -------------------------------------------------------------------------
    # Estimate normals on full dense cloud if missing
    if not dense_cloud.has_normals():
        dense_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=d_dense * 3.0, max_nn=30))
    if not ref_scaled.has_normals():
        ref_scaled.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=d_dense * 3.0, max_nn=30))

    icp_distance_threshold = max(d_dense * 1.5, 0.003)
    icp_result = o3d.pipelines.registration.registration_icp(
        ref_scaled, dense_cloud, icp_distance_threshold, ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    )

    log(f"[ALIGN] Point-to-Plane ICP: fitness = {icp_result.fitness:.3f}, RMSE = {icp_result.inlier_rmse:.4f}")

    # Compute overall transform matrix combining initial scale + centroid translate + ICP
    T_icp = icp_result.transformation
    S_mat = np.identity(4)
    S_mat[0, 0] = initial_scale
    S_mat[1, 1] = initial_scale
    S_mat[2, 2] = initial_scale

    T_init_trans = np.identity(4)
    T_init_trans[:3, 3] = initial_translation

    t_center = np.identity(4)
    t_center[:3, 3] = -center_ref
    t_uncenter = np.identity(4)
    t_uncenter[:3, 3] = center_ref

    T_full = T_icp @ T_init_trans @ t_uncenter @ S_mat @ t_center

    # -------------------------------------------------------------------------
    # STAGE 3: Spatial Overlap Validation & Confidence Evaluation
    # -------------------------------------------------------------------------
    # Transform reference cloud with full matrix and check bounding box intersection
    ref_transformed = copy.deepcopy(ref_cloud).transform(T_full)
    ref_pts_trans = np.asarray(ref_transformed.points)

    bbox_dense_obj = dense_cloud.get_axis_aligned_bounding_box()
    min_d = bbox_dense_obj.get_min_bound() - d_dense * 5.0
    max_d = bbox_dense_obj.get_max_bound() + d_dense * 5.0

    # Count how many transformed points land inside dense cloud's bounding box
    inside_mask = (ref_pts_trans[:, 0] >= min_d[0]) & (ref_pts_trans[:, 0] <= max_d[0]) & \
                  (ref_pts_trans[:, 1] >= min_d[1]) & (ref_pts_trans[:, 1] <= max_d[1]) & \
                  (ref_pts_trans[:, 2] >= min_d[2]) & (ref_pts_trans[:, 2] <= max_d[2])
    
    spatial_overlap_ratio = float(np.mean(inside_mask))
    norm_rmse = icp_result.inlier_rmse / max(d_dense, 1e-6)
    # Gating logic:
    # fitness >= 0.35 AND norm_rmse <= 3.5 AND spatial_overlap_ratio >= 0.15
    success = True
    if spatial_overlap_ratio < 0.15:
        success = False
        warnings.append(f"Spatial bounding box overlap too low ({spatial_overlap_ratio:.1%} < 15.0%). Clouds do not share physical volume.")
    if icp_result.fitness < 0.35:
        success = False
        warnings.append(f"Low alignment overlap fitness ({icp_result.fitness:.1%} < 35.0%). Clouds may not overlap sufficiently.")
    if norm_rmse > 3.5:
        success = False
        warnings.append(f"High relative alignment error (RMSE/spacing = {norm_rmse:.2f} > 3.5). Surfaces may not match.")

    final_scale = initial_scale
    overlap_pct = float(icp_result.fitness * 100.0)

    if not success:
        log(f"[WARNING] Reference cloud alignment confidence too low: {'; '.join(warnings)}")

    log(
        f"\n{'='*60}\n"
        f"  REFERENCE CLOUD ALIGNMENT SUMMARY\n"
        f"{'='*60}\n"
        f"  Pre-scale factor:     {initial_scale:.4f}\n"
        f"  Point spacing:        {d_dense:.4f} units\n"
        f"  Spatial ROI Overlap:  {spatial_overlap_ratio:.1%}\n"
        f"  ICP Fitness:          {icp_result.fitness:.1%}\n"
        f"  Inlier RMSE:          {icp_result.inlier_rmse:.4f} ({norm_rmse:.2f}x point spacing)\n"
        f"  Alignment Confidence: {'[OK] CONFIDENT' if success else '[WARNING] LOW CONFIDENCE (Fusion Skipped)'}\n"
        f"{'='*60}"
    )

    return AlignResult(
        transform=T_full,
        initial_scale=initial_scale,
        final_scale=final_scale,
        fitness=float(icp_result.fitness),
        inlier_rmse=float(icp_result.inlier_rmse),
        normalized_rmse=float(norm_rmse),
        overlap_pct=overlap_pct,
        success=success,
        warnings=warnings
    )
