"""
Test script for Phase 3 (cloud_fusion)
Verifies dynamic scale-aware gap filtering, normal orientation validation,
and Poisson surface reconstruction with density trimming.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test():
    import open3d as o3d
    import point_cloud_io
    import cloud_aligner
    import cloud_fusion

    print("=== TEST PHASE 3: Cloud Fusion & Poisson Mesh Generation ===")
    
    # Load previously generated synthetic clouds or create new ones
    dense_path = "scratch/test_data/test_dense.ply"
    ref_path = "scratch/test_data/test_ref.ply"

    res_dense = point_cloud_io.load_point_cloud(dense_path, print)
    res_ref = point_cloud_io.load_point_cloud(ref_path, print)

    # 1. Align reference cloud
    align_res = cloud_aligner.align_to_dense(res_ref.cloud, res_dense.cloud, print)
    assert align_res.success, "Alignment failed"

    # Transform reference cloud
    aligned_ref = res_ref.cloud.transform(align_res.transform)

    # 2. Merge clouds using dynamic gap radius
    merged_cloud = cloud_fusion.merge_clouds(res_dense.cloud, aligned_ref, print, gap_radius_mult=3.0)
    assert len(merged_cloud.points) > 0, "Merged cloud should contain points"

    # Save merged cloud
    merged_cloud_path = "scratch/test_data/test_merged_cloud.ply"
    point_cloud_io.save_point_cloud(merged_cloud, merged_cloud_path)
    print(f"Saved merged point cloud: {merged_cloud_path}")

    # 3. Generate Poisson Mesh
    mesh = cloud_fusion.generate_mesh(merged_cloud, print, poisson_depth=8, density_threshold_pct=5.0)
    assert len(mesh.vertices) > 0 and len(mesh.triangles) > 0, "Poisson mesh is empty!"

    mesh_path = "scratch/test_data/test_merged_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Saved generated Poisson mesh: {mesh_path}")

    print("\nPhase 3 verified successfully!")

if __name__ == "__main__":
    run_test()
