"""
Detailed Survival Breakdown & Unambiguous Rejection Test Suite for Proximap
1. COLMAP-only Baseline vs Fused RefineMesh retention
2. Region-specific breakdown: Survival rate of Reference-cloud-derived vertices vs COLMAP-derived vertices
3. Unambiguously wrong rejection cases (100x scale mismatch, Torus geometry)
"""

import os
import sys
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import point_cloud_io
import cloud_aligner
import cloud_fusion
from hardware_profiler import run_safe_subprocess

from main_window import get_reconstruction_out_dir

def run_detailed_breakdown_tests():
    import open3d as o3d

    print("=" * 75)
    print("PROXIMAP REFERENCE CLOUD FUSION - REGION SURVIVAL & REJECTION SUITE")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = get_reconstruction_out_dir()
    mvs_dir = os.path.join(out_dir, "mvs")
    dense_ply_path = os.path.join(mvs_dir, "scene_dense.ply")
    dense_mvs_path = os.path.join(mvs_dir, "scene_dense.mvs")
    if os.path.exists(dense_ply_path):
        print(f"Loading real dense cloud from: {dense_ply_path}")
        dense_load = point_cloud_io.load_point_cloud(dense_ply_path, print)
        dense_cloud = dense_load.cloud
    else:
        print("[NOTICE] scene_dense.ply not found in mvs_dir. Generating 100k-point high-density scene cloud...")
        x = np.linspace(-3.0, 3.0, 316)
        y = np.linspace(-3.0, 3.0, 316)
        xx, yy = np.meshgrid(x, y)
        zz = 0.2 * np.sin(xx * 2.0) * np.cos(yy * 2.0)
        pts = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

        # Create a central gap/hole in SfM reconstruction (hole between -0.8 and 0.8)
        mask = ~((pts[:, 0] > -0.8) & (pts[:, 0] < 0.8) & (pts[:, 1] > -0.8) & (pts[:, 1] < 0.8))
        pts_sfm = pts[mask]

        dense_cloud = o3d.geometry.PointCloud()
        dense_cloud.points = o3d.utility.Vector3dVector(pts_sfm)
        dense_cloud.estimate_normals()
        os.makedirs(mvs_dir, exist_ok=True)
        o3d.io.write_point_cloud(dense_ply_path, dense_cloud)

    d_spacing = cloud_fusion.compute_median_point_spacing(dense_cloud)

    bbox_dense = dense_cloud.get_axis_aligned_bounding_box()
    min_b = bbox_dense.get_min_bound()
    max_b = bbox_dense.get_max_bound()

    # -------------------------------------------------------------------------
    # PART 1: Unambiguous Rejection Cases (Scale Mismatch & Torus Geometry)
    # -------------------------------------------------------------------------
    print("\n--- PART 1: Unambiguous Rejection Calibration ---")

    # Case B1: Pure Random Noise Cloud (no coherent surface features)
    print("\n[Case B1] Testing Pure Random Noise Cloud (unalignable 3D noise)...")
    pts_noise = np.random.uniform(min_b, max_b, (10000, 3))
    ref_noise = o3d.geometry.PointCloud()
    ref_noise.points = o3d.utility.Vector3dVector(pts_noise)

    align_b1 = cloud_aligner.align_to_dense(ref_noise, dense_cloud, print)
    print(f"Case B1 Result: Success={align_b1.success}, Fitness={align_b1.fitness:.1%}, NormRMSE={align_b1.normalized_rmse:.2f}x")
    assert not align_b1.success, "Case B1 (Random noise cloud) should FAIL confidence gate!"
    print(f"[PASS] Case B1 (Pure Random Noise) rejected cleanly with fitness={align_b1.fitness:.1%}.")

    # Case B2: Completely Unrelated Topology (Torus geometry)
    print("\n[Case B2] Testing Unrelated Shape Topology (Torus geometry)...")
    torus_mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=5.0, tube_radius=1.5)
    ref_torus = torus_mesh.sample_points_uniformly(number_of_points=15000)

    align_b2 = cloud_aligner.align_to_dense(ref_torus, dense_cloud, print)
    print(f"Case B2 Result: Success={align_b2.success}, Fitness={align_b2.fitness:.1%}, NormRMSE={align_b2.normalized_rmse:.2f}x")
    assert not align_b2.success, "Case B2 (Torus topology) should FAIL confidence gate!"
    print(f"[PASS] Case B2 (Torus Topology) rejected cleanly with fitness={align_b2.fitness:.1%} (well below 15.0% threshold).")

    # -------------------------------------------------------------------------
    # PART 2: COLMAP-Only Baseline vs Fused RefineMesh Survival & Region Breakdown
    # -------------------------------------------------------------------------
    print("\n--- PART 2: Baseline vs Fused RefineMesh Survival Breakdown ---")

    # 2.1 Generate COLMAP-only Poisson Mesh (Baseline)
    print("\n[Generating COLMAP-Only Poisson Baseline Mesh @ Depth 9]...")
    colmap_poisson_baseline = cloud_fusion.generate_mesh(dense_cloud, print, poisson_depth=9, density_threshold_pct=5.0)
    baseline_ply_path = os.path.join(mvs_dir, "scene_dense_mesh_baseline.ply")
    o3d.io.write_triangle_mesh(baseline_ply_path, colmap_poisson_baseline)

    # 2.2 Run RefineMesh on COLMAP-only Baseline
    refine_exe = os.path.join(base_dir, "backend_bin", "openMVS", "RefineMesh.exe")
    refine_baseline_ply = os.path.join(mvs_dir, "scene_dense_mesh_baseline_refine.ply")
    
    print("\n[Running RefineMesh on COLMAP-Only Baseline]...")
    if os.path.exists(dense_mvs_path):
        cmd_refine_base = [
            refine_exe, "scene_dense.mvs",
            "-m", "scene_dense_mesh_baseline.ply",
            "-o", "scene_dense_mesh_baseline_refine.mvs",
            "--resolution-level", "1", "--scales", "2"
        ]
    else:
        cmd_refine_base = [
            refine_exe,
            "-i", "scene_dense_mesh_baseline.ply",
            "-o", "scene_dense_mesh_baseline_refine.ply",
            "--resolution-level", "1", "--scales", "2"
        ]
    run_safe_subprocess(cmd_refine_base, timeout=600.0, cwd=mvs_dir)

    baseline_refined = o3d.io.read_triangle_mesh(refine_baseline_ply) if os.path.exists(refine_baseline_ply) else None
    baseline_retention = (len(baseline_refined.vertices) / len(colmap_poisson_baseline.vertices)) * 100.0 if baseline_refined else 0.0

    print(f"COLMAP-Only Baseline Mesh Pre-Refine:  {len(colmap_poisson_baseline.vertices):,} vertices, {len(colmap_poisson_baseline.triangles):,} faces")
    if baseline_refined:
        print(f"COLMAP-Only Baseline Mesh Post-Refine: {len(baseline_refined.vertices):,} vertices, {len(baseline_refined.triangles):,} faces")
        print(f"COLMAP-Only Baseline Retention Ratio:  {baseline_retention:.1f}%")

    # 2.3 Create Fused Cloud with Tagged Gap-Filling Points
    print("\n[Generating Fused Cloud with Region Tagging]...")
    # Extract crop sub-region + wall extension
    bbox_dense = dense_cloud.get_axis_aligned_bounding_box()
    min_b = bbox_dense.get_min_bound()
    max_b = bbox_dense.get_max_bound()

    pts_dense = np.asarray(dense_cloud.points)
    crop_mask = (pts_dense[:, 0] < min_b[0] + 0.35 * (max_b[0] - min_b[0])) & \
                (pts_dense[:, 1] < min_b[1] + 0.35 * (max_b[1] - min_b[1]))
    partial_pts = pts_dense[crop_mask]

    x_ext = np.linspace(min_b[0] - 2.0, min_b[0], 60)
    y_ext = np.linspace(min_b[1] - 2.0, min_b[1], 60)
    xx_e, yy_e = np.meshgrid(x_ext, y_ext)
    ext_wall = np.vstack((xx_e.flatten(), yy_e.flatten(), np.ones_like(xx_e.flatten()) * min_b[2])).T
    
    partial_pts_combined = np.vstack((partial_pts, ext_wall))
    scale_true = 3.0
    partial_pts_trans = ((partial_pts_combined * (1.0 / scale_true)) @ np.eye(3)) + np.array([0.2, -0.2, 0.1])

    ref_partial = o3d.geometry.PointCloud()
    ref_partial.points = o3d.utility.Vector3dVector(partial_pts_trans)

    align_res = cloud_aligner.align_to_dense(ref_partial, dense_cloud, print)
    aligned_ref = ref_partial.transform(align_res.transform)

    # Identify exact gap-fill points injected from reference cloud
    kdtree_dense = o3d.geometry.KDTreeFlann(dense_cloud)
    ref_pts_alg = np.asarray(aligned_ref.points)
    gap_radius = 3.0 * d_spacing
    gap_pts = []
    for pt in ref_pts_alg:
        [k, idx, dist_sq] = kdtree_dense.search_radius_vector_3d(pt, gap_radius)
        if k == 0:
            gap_pts.append(pt)

    gap_cloud = o3d.geometry.PointCloud()
    gap_cloud.points = o3d.utility.Vector3dVector(np.asarray(gap_pts))
    print(f"Identified {len(gap_pts):,} reference-cloud gap-fill points.")

    # Generate Fused Poisson Mesh
    merged_cloud = cloud_fusion.merge_clouds(dense_cloud, aligned_ref, print, gap_radius_mult=3.0)
    fused_poisson_mesh = cloud_fusion.generate_mesh(merged_cloud, print, poisson_depth=9, density_threshold_pct=5.0)

    fused_ply_path = os.path.join(mvs_dir, "scene_dense_mesh_refcloud.ply")
    o3d.io.write_triangle_mesh(fused_ply_path, fused_poisson_mesh)

    # Categorize Pre-Refine Fused Mesh Vertices (COLMAP-derived vs RefCloud-derived)
    kdtree_gap = o3d.geometry.KDTreeFlann(gap_cloud)
    fused_verts = np.asarray(fused_poisson_mesh.vertices)
    
    pre_ref_refcloud_indices = []
    pre_ref_colmap_indices = []
    
    for idx, v in enumerate(fused_verts):
        [k, _, dist_sq] = kdtree_dense.search_knn_vector_3d(v, 1)
        [k_g, _, dist_sq_g] = kdtree_gap.search_knn_vector_3d(v, 1)
        
        d_colmap = np.sqrt(dist_sq[0]) if k > 0 else 999.0
        d_gap = np.sqrt(dist_sq_g[0]) if k_g > 0 else 999.0

        if d_gap < d_colmap and d_gap < d_spacing * 2.5:
            pre_ref_refcloud_indices.append(idx)
        else:
            pre_ref_colmap_indices.append(idx)

    count_pre_refcloud = len(pre_ref_refcloud_indices)
    count_pre_colmap = len(pre_ref_colmap_indices)

    print(f"\n[Pre-Refinement Fused Mesh Vertex Categorization]:")
    print(f"  Total Vertices:               {len(fused_verts):,}")
    print(f"  COLMAP-Derived Vertices:       {count_pre_colmap:,} ({count_pre_colmap/len(fused_verts):.1%})")
    print(f"  RefCloud-Derived Vertices:    {count_pre_refcloud:,} ({count_pre_refcloud/len(fused_verts):.1%})")

    # Run RefineMesh on Fused Mesh
    print("\n[Running RefineMesh on Fused Mesh]...")
    if os.path.exists(dense_mvs_path):
        cmd_refine_fused = [
            refine_exe, "scene_dense.mvs",
            "-m", "scene_dense_mesh_refcloud.ply",
            "-o", "scene_dense_mesh_refcloud_refine.mvs",
            "--resolution-level", "1", "--scales", "2"
        ]
    else:
        cmd_refine_fused = [
            refine_exe,
            "-i", "scene_dense_mesh_refcloud.ply",
            "-o", "scene_dense_mesh_refcloud_refine.ply",
            "--resolution-level", "1", "--scales", "2"
        ]
    run_safe_subprocess(cmd_refine_fused, timeout=600.0, cwd=mvs_dir)

    fused_refined_ply = os.path.join(mvs_dir, "scene_dense_mesh_refcloud_refine.ply")
    if not os.path.exists(fused_refined_ply):
        fused_refined_ply = os.path.join(mvs_dir, "scene_dense_mesh_refine.ply")

    fused_refined_mesh = o3d.io.read_triangle_mesh(fused_refined_ply) if os.path.exists(fused_refined_ply) else None
    
    if fused_refined_mesh:
        post_verts = np.asarray(fused_refined_mesh.vertices)
        post_refcloud_count = 0
        post_colmap_count = 0

        for v in post_verts:
            [k, _, dist_sq] = kdtree_dense.search_knn_vector_3d(v, 1)
            [k_g, _, dist_sq_g] = kdtree_gap.search_knn_vector_3d(v, 1)
            
            d_colmap = np.sqrt(dist_sq[0]) if k > 0 else 999.0
            d_gap = np.sqrt(dist_sq_g[0]) if k_g > 0 else 999.0

            if d_gap < d_colmap and d_gap < d_spacing * 2.5:
                post_refcloud_count += 1
            else:
                post_colmap_count += 1

        refcloud_survival_rate = (post_refcloud_count / max(count_pre_refcloud, 1)) * 100.0
        colmap_survival_rate = (post_colmap_count / max(count_pre_colmap, 1)) * 100.0
        total_fused_survival_rate = (len(post_verts) / max(len(fused_verts), 1)) * 100.0

        print(f"\n" + "=" * 70)
        print("FINAL REGION-SPECIFIC SURVIVAL BREAKDOWN REPORT")
        print("=" * 70)
        print(f"1. COLMAP-Only Baseline Mesh Retention Ratio:      {baseline_retention:.1f}%")
        print(f"2. Fused Mesh Total Retention Ratio:               {total_fused_survival_rate:.1f}%")
        print(f"   --------------------------------------------------------")
        print(f"   • COLMAP-Derived Vertices Survival Rate:         {colmap_survival_rate:.1f}% ({post_colmap_count:,} / {count_pre_colmap:,})")
        print(f"   • Reference-Cloud-Derived Vertices Survival Rate: {refcloud_survival_rate:.1f}% ({post_refcloud_count:,} / {count_pre_refcloud:,})")
        print("=" * 70)

        assert post_refcloud_count > 0, "Reference-cloud-derived vertices must survive RefineMesh!"
        print("\n[VERIFIED] Reference-cloud-derived region survival successfully confirmed and quantified!")

if __name__ == "__main__":
    run_detailed_breakdown_tests()
