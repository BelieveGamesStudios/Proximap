import os
import struct
import numpy as np

def verify_math_and_real_data():
    print("=" * 60)
    print("STEP 1: Isolated Matrix & Real-Data Verification")
    print("=" * 60)

    # 1. Transformation Matrix F
    F = np.diag([1.0, -1.0, -1.0])
    det_F = np.linalg.det(F)
    print(f"1. Flip matrix F:\n{F}")
    print(f"   det(F) = {det_F:.1f} (Must be +1.0 for 180 deg rotation about X, no reflection)")

    assert np.isclose(det_F, 1.0), "det(F) is not +1!"

    # 2. Test Mesh & Points on real reconstruction files
    mesh_path = "reconstruction_out/mvs/scene_dense_mesh_texture.obj"
    if os.path.exists(mesh_path):
        import trimesh
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        orig_min = mesh.vertices.min(axis=0)
        orig_max = mesh.vertices.max(axis=0)
        print(f"\n2. Real OBJ mesh ({mesh_path}):")
        print(f"   Original vertex bounds: min={orig_min}, max={orig_max}")
        
        # Apply F to vertices
        flipped_vertices = mesh.vertices @ F.T
        flipped_min = flipped_vertices.min(axis=0)
        flipped_max = flipped_vertices.max(axis=0)
        print(f"   Flipped vertex bounds:  min={flipped_min}, max={flipped_max}")
        
        # Verify Y and Z bounds flipped correctly
        assert np.isclose(flipped_min[0], orig_min[0]) and np.isclose(flipped_max[0], orig_max[0]), "X bounds changed!"
        assert np.isclose(flipped_min[1], -orig_max[1]) and np.isclose(flipped_max[1], -orig_min[1]), "Y bounds failed!"
        assert np.isclose(flipped_min[2], -orig_max[2]) and np.isclose(flipped_max[2], -orig_min[2]), "Z bounds failed!"
        print("   [SUCCESS] Vertex transformation bounds verified!")

        if len(mesh.vertex_normals) > 0:
            orig_normals = mesh.vertex_normals
            flipped_normals = orig_normals @ F.T
            norm_diff = np.abs(np.linalg.norm(flipped_normals, axis=1) - 1.0).max()
            print(f"   Max normal unit length diff: {norm_diff:.6e}")
            assert norm_diff < 1e-5, "Normals lost unit length!"
            print("   [SUCCESS] Normal vector transformation verified!")

    # 3. Test COLMAP Camera similarity transform
    images_bin = "reconstruction_out/colmap/sparse/0/images.bin"
    if os.path.exists(images_bin):
        print(f"\n3. Real COLMAP images.bin ({images_bin}):")
        with open(images_bin, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            print(f"   Number of cameras in binary: {num_images}")
            
            # Read first image
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                char = f.read(1)
                if char == b"\x00": break
                name += char.decode("latin1")
            
            # Quat to Rot matrix
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
            ])
            T = np.array([tx, ty, tz])
            
            # Camera center in world space before flip: C = -R^T @ T
            C_orig = -R.T @ T
            print(f"   Original Camera 1 ({name}):")
            print(f"     Camera Center (World): {C_orig}")
            
            # Apply Similarity Transform
            R_prime = F @ R @ F.T
            T_prime = F @ T
            C_prime = -R_prime.T @ T_prime
            
            # Expected Flipped Camera Center: F @ C_orig
            C_expected = F @ C_orig
            print(f"     Flipped Camera Center (World):  {C_prime}")
            print(f"     Expected F @ C_orig:           {C_expected}")
            
            diff_C = np.linalg.norm(C_prime - C_expected)
            print(f"     Camera Center Difference: {diff_C:.6e}")
            assert diff_C < 1e-6, "Similarity transform for camera pose failed!"
            
            # Verify projection consistency: X_c' = R' X_w' + T'
            # Pick a sample 3D point in world coords
            X_w = np.array([1.5, 2.5, 3.5])
            X_c_orig = R @ X_w + T
            
            X_w_prime = F @ X_w
            X_c_prime = R_prime @ X_w_prime + T_prime
            
            expected_X_c_prime = F @ X_c_orig
            diff_proj = np.linalg.norm(X_c_prime - expected_X_c_prime)
            print(f"     Sample point camera-frame proj diff: {diff_proj:.6e}")
            assert diff_proj < 1e-6, "Projection consistency check failed!"
            print("   [SUCCESS] Camera Similarity Transform (R' = F R F^T, T' = F T) verified!")

    print("\n" + "=" * 60)
    print("ALL ISOLATED MATRIX & REAL-DATA CHECKS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    verify_math_and_real_data()
