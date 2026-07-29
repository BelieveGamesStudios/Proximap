"""
Cloud Fusion Module for Proximap
Merges aligned external reference point clouds with COLMAP dense clouds using
dynamic scale-aware gap filtering, normal orientation validation, and Poisson
surface reconstruction with density-based trimming.
"""

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple


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


def merge_clouds(
    dense_cloud,
    aligned_ref_cloud,
    log_fn: Optional[Callable[[str], None]] = None,
    gap_radius_mult: float = 3.0
):
    """
    Merges aligned_ref_cloud into dense_cloud, filtering out points that are
    redundant with existing dense-cloud coverage using a dynamic gap radius.
    
    Args:
        dense_cloud: Original COLMAP dense point cloud.
        aligned_ref_cloud: Aligned and scaled external reference point cloud.
        log_fn: Optional logging callback.
        gap_radius_mult: Multiplier for median point spacing to set dynamic gap radius.
        
    Returns:
        Merged open3d.geometry.PointCloud
    """
    import open3d as o3d

    def log(msg: str):
        if log_fn:
            log_fn(msg)

    d_dense = compute_median_point_spacing(dense_cloud)
    gap_radius = gap_radius_mult * d_dense

    log(f"[FUSION] Dynamic gap radius: {gap_radius:.4f} (spacing: {d_dense:.4f} x {gap_radius_mult:.1f})")

    # Build KD-Tree on dense cloud
    kdtree_dense = o3d.geometry.KDTreeFlann(dense_cloud)
    ref_pts = np.asarray(aligned_ref_cloud.points)
    ref_colors = np.asarray(aligned_ref_cloud.colors) if aligned_ref_cloud.has_colors() else None

    gap_pts = []
    gap_colors = []

    for i in range(len(ref_pts)):
        pt = ref_pts[i]
        [k, idx, dist_sq] = kdtree_dense.search_radius_vector_3d(pt, gap_radius)
        if k == 0:
            # Point has no dense-cloud neighbor within gap_radius -> Genuine Gap-fill point!
            gap_pts.append(pt)
            if ref_colors is not None and len(ref_colors) == len(ref_pts):
                gap_colors.append(ref_colors[i])

    log(f"[FUSION] Reference cloud gap-filtering: {len(ref_pts):,} total pts -> {len(gap_pts):,} gap-filling points injected ({len(ref_pts) - len(gap_pts):,} redundant points filtered)")

    if not gap_pts:
        log("[WARNING] No gap-filling points found. Using original dense cloud.")
        return dense_cloud

    # Create point cloud for gap points
    gap_cloud = o3d.geometry.PointCloud()
    gap_cloud.points = o3d.utility.Vector3dVector(np.asarray(gap_pts))
    if gap_colors:
        gap_cloud.colors = o3d.utility.Vector3dVector(np.asarray(gap_colors))

    # Combine dense cloud and gap cloud
    merged = dense_cloud + gap_cloud

    # Voxel downsample to unify point density
    voxel_size = max(d_dense, 0.001)
    merged_down = merged.voxel_down_sample(voxel_size)

    log(f"[FUSION] Merged cloud point count: {len(merged.points):,} raw -> {len(merged_down.points):,} voxel-downsampled @ {voxel_size:.4f}")

    return merged_down


def generate_mesh(
    merged_cloud,
    log_fn: Optional[Callable[[str], None]] = None,
    poisson_depth: int = 9,
    density_threshold_pct: float = 5.0
):
    """
    Generates a surface mesh from the merged point cloud using Poisson Surface Reconstruction
    with density-based trimming and normal orientation validation.
    
    Args:
        merged_cloud: Merged point cloud (COLMAP + Reference Cloud gap-fill points).
        log_fn: Optional logging callback.
        poisson_depth: Octree depth for Poisson reconstruction (default: 9).
        density_threshold_pct: Percentile threshold for trimming low-density Poisson vertices.
        
    Returns:
        Cleaned open3d.geometry.TriangleMesh
    """
    import open3d as o3d

    def log(msg: str):
        if log_fn:
            log_fn(msg)

    d_spacing = compute_median_point_spacing(merged_cloud)

    log(f"[FUSION] Estimating and orienting normals for Poisson reconstruction...")

    # Estimate normals if missing or compute fresh consistently
    radius_normal = d_spacing * 3.5
    merged_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    merged_cloud.orient_normals_consistent_tangent_plane(k=15)

    # Validate normal orientation consistency
    normals = np.asarray(merged_cloud.normals)
    if len(normals) > 100:
        # Check dot product of random 1000 pairs
        kdtree = o3d.geometry.KDTreeFlann(merged_cloud)
        pts = np.asarray(merged_cloud.points)
        sample_indices = np.random.choice(len(pts), min(1000, len(pts)), replace=False)
        flipped_count = 0
        total_checked = 0

        for idx in sample_indices:
            [k, neighbor_indices, _] = kdtree.search_knn_vector_3d(pts[idx], 3)
            if k >= 2:
                for n_idx in neighbor_indices[1:]:
                    dot_val = np.dot(normals[idx], normals[n_idx])
                    if dot_val < -0.3:
                        flipped_count += 1
                    total_checked += 1

        if total_checked > 0 and (flipped_count / total_checked) > 0.05:
            log(f"[WARNING] Normal orientation validation: {flipped_count / total_checked:.1%} of local normal pairs show orientation inconsistency. Poisson mesh may contain local artifacts.")
        else:
            log("[FUSION] Normal orientation validation passed (consistent orientation).")

    log(f"[FUSION] Running Poisson Surface Reconstruction (octree depth={poisson_depth})...")

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            merged_cloud, depth=poisson_depth, linear_fit=True
        )

    log(f"[FUSION] Raw Poisson mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")

    # Density-based trimming: Remove low-evidence vertices
    densities_np = np.asarray(densities)
    if len(densities_np) > 0:
        cutoff_density = np.percentile(densities_np, density_threshold_pct)
        vertices_to_remove = densities_np < cutoff_density
        mesh.remove_vertices_by_mask(vertices_to_remove)
        log(f"[FUSION] Density trimming: removed lower {density_threshold_pct}% low-evidence vertices (< {cutoff_density:.2f})")

    # Clean small disconnected components
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles()
        )
    
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) > 1:
        total_faces = len(mesh.triangles)
        min_cluster_size = max(50, int(total_faces * 0.005))
        triangles_to_remove = cluster_n_triangles[np.asarray(triangle_clusters)] < min_cluster_size
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        log(f"[FUSION] Removed small floating mesh fragments (< {min_cluster_size} faces)")

    mesh.compute_vertex_normals()
    log(f"[FUSION] Final trimmed Poisson mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")

    return mesh
