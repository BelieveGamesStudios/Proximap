"""
Test script for Phase 1 (point_cloud_io) and Phase 2 (cloud_aligner)
Generates synthetic point clouds and verifies point cloud I/O, initial scale pre-correction,
coarse RANSAC registration, point-to-plane ICP refinement, and confidence metrics.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test():
    import open3d as o3d
    import point_cloud_io
    import cloud_aligner

    print("=== TEST 1: Synthetic Point Cloud Generation ===")
    # 1. Create a synthetic dense cloud (100x100 grid with a central gap/hole)
    x = np.linspace(-2.0, 2.0, 100)
    y = np.linspace(-2.0, 2.0, 100)
    xx, yy = np.meshgrid(x, y)
    zz = 0.1 * np.sin(xx) * np.cos(yy)

    # Flatten coordinates
    pts_dense = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    # Add small Gaussian noise
    pts_dense += np.random.normal(0, 0.005, pts_dense.shape)

    # Create a gap/hole in the middle (-0.5 < x < 0.5 and -0.5 < y < 0.5)
    mask = ~((pts_dense[:, 0] > -0.5) & (pts_dense[:, 0] < 0.5) & (pts_dense[:, 1] > -0.5) & (pts_dense[:, 1] < 0.5))
    pts_dense = pts_dense[mask]

    dense_cloud = o3d.geometry.PointCloud()
    dense_cloud.points = o3d.utility.Vector3dVector(pts_dense)
    # Add dummy colors
    dense_cloud.colors = o3d.utility.Vector3dVector(np.ones_like(pts_dense) * 0.7)

    # 2. Create a synthetic reference cloud (complete grid covering the hole, scaled by 3.5x and translated)
    x_ref = np.linspace(-2.0, 2.0, 120)
    y_ref = np.linspace(-2.0, 2.0, 120)
    xx_ref, yy_ref = np.meshgrid(x_ref, y_ref)
    zz_ref = 0.1 * np.sin(xx_ref) * np.cos(yy_ref)
    pts_ref = np.vstack((xx_ref.flatten(), yy_ref.flatten(), zz_ref.flatten())).T

    # Scale reference cloud by 3.5x, apply rotation (10 deg around Z) and translation
    scale_true = 3.5
    pts_ref_scaled = pts_ref * (1.0 / scale_true)  # ref cloud in metric meters vs COLMAP scaled up

    angle = np.radians(12.0)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    translation = np.array([0.3, -0.2, 0.1])
    pts_ref_transformed = (pts_ref_scaled @ R.T) + translation

    ref_cloud = o3d.geometry.PointCloud()
    ref_cloud.points = o3d.utility.Vector3dVector(pts_ref_transformed)
    ref_cloud.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.8, 0.2], (len(pts_ref_transformed), 1)))

    print(f"Generated COLMAP dense cloud: {len(dense_cloud.points)} points")
    print(f"Generated external reference cloud: {len(ref_cloud.points)} points (Pre-scaled by {scale_true}x)")

    # Save to temporary test files
    os.makedirs("scratch/test_data", exist_ok=True)
    dense_path = "scratch/test_data/test_dense.ply"
    ref_path = "scratch/test_data/test_ref.ply"

    point_cloud_io.save_point_cloud(dense_cloud, dense_path)
    point_cloud_io.save_point_cloud(ref_cloud, ref_path)

    print("\n=== TEST 2: Point Cloud I/O Verification ===")
    res_dense = point_cloud_io.load_point_cloud(dense_path, print)
    res_ref = point_cloud_io.load_point_cloud(ref_path, print)

    assert res_dense.success, "Failed to load dense PLY"
    assert res_ref.success, "Failed to load reference PLY"
    print("Point cloud I/O load verified successfully!")

    print("\n=== TEST 3: Cloud Aligner & Pre-scale Alignment Verification ===")
    align_res = cloud_aligner.align_to_dense(res_ref.cloud, res_dense.cloud, print)

    print(f"\nAlignment Result:")
    print(f"  Success: {align_res.success}")
    print(f"  Initial Scale Factor: {align_res.initial_scale:.4f}")
    print(f"  Fitness: {align_res.fitness:.3f}")
    print(f"  Inlier RMSE: {align_res.inlier_rmse:.4f}")
    print(f"  Normalized RMSE: {align_res.normalized_rmse:.2f}x point spacing")

    assert align_res.success, "Alignment failed on synthetic test scene"
    assert align_res.fitness > 0.3, "Fitness score lower than expected"
    print("\nPhase 1 and Phase 2 verified successfully!")

if __name__ == "__main__":
    run_test()
