import os
import sys
import numpy as np
import open3d as o3d
import time

# Ensure workspace root is in path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from PySide6.QtCore import QCoreApplication
from standalone_reconstruction import StandaloneReconstructionWorker

def create_synthetic_cloud(file_path):
    print(f"Creating colored synthetic sphere point cloud at {file_path}...")
    # Create a sphere point cloud
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    pcd = o3d.geometry.PointCloud()
    pcd.points = sphere.vertices
    
    # Assign vertex colors (e.g. rainbow gradient)
    colors = np.zeros((len(pcd.points), 3))
    pts = np.asarray(pcd.points)
    colors[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min() + 1e-6)
    colors[:, 1] = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-6)
    colors[:, 2] = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Clear normals to simulate a raw scanner import
    pcd.normals = o3d.utility.Vector3dVector([])
    
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Synthetic point cloud created with {len(pcd.points)} points. Colors present: {pcd.has_colors()}, Normals present: {pcd.has_normals()}")
    return file_path

def main():
    # We need a QCoreApplication to run signals / slots
    app = QCoreApplication.instance()
    if not app:
        app = QCoreApplication(sys.argv)

    out_dir = os.path.join(base_dir, "reconstruction_out")
    os.makedirs(out_dir, exist_ok=True)
    
    cloud_path = os.path.join(out_dir, "synthetic_colored_sphere.ply")
    create_synthetic_cloud(cloud_path)
    
    print("\nInitializing StandaloneReconstructionWorker...")
    worker = StandaloneReconstructionWorker(
        cloud_path=cloud_path,
        output_dir=out_dir,
        include_colors=True,
        poisson_depth=8
    )
    
    finished_flag = {"success": False, "msg": ""}
    
    # Signal handlers
    def on_progress(val):
        print(f"[TEST PROGRESS] {val}%")
        
    def on_status(status):
        print(f"[TEST STATUS] {status}")
        
    def on_log(msg):
        print(f"[TEST LOG] {msg}")
        
    def on_finished(success, msg):
        print(f"[TEST FINISHED] Success={success} | Msg={msg}")
        finished_flag["success"] = success
        finished_flag["msg"] = msg
        app.quit()
        
    worker.progress_changed.connect(on_progress)
    worker.status_changed.connect(on_status)
    worker.log_message.connect(on_log)
    worker.finished.connect(on_finished)
    
    print("\nStarting worker thread...")
    worker.start()
    
    # Run event loop until finished is called
    app.exec()
    
    # Check outputs
    mvs_dir = os.path.join(out_dir, "mvs")
    poisson_mesh_path = os.path.join(mvs_dir, "scene_dense_mesh.ply")
    refined_mesh_path = os.path.join(mvs_dir, "scene_dense_mesh_refine.ply")
    
    print("\n--- Verifying Output Files ---")
    assert finished_flag["success"], f"Worker failed: {finished_flag['msg']}"
    assert os.path.exists(poisson_mesh_path), f"Poisson mesh not found at {poisson_mesh_path}"
    assert os.path.exists(refined_mesh_path), f"Refined mesh not found at {refined_mesh_path}"
    print("[PASS] Both Poisson and RefineMesh output files exist.")
    
    # Load mesh and check vertex colors
    mesh = o3d.io.read_triangle_mesh(refined_mesh_path)
    print(f"Refined mesh contains {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    assert len(mesh.vertices) > 0, "Mesh contains 0 vertices!"
    assert mesh.has_vertex_colors(), "Mesh does not have vertex colors, but include_colors was True!"
    print("[PASS] Output mesh contains vertices and retains vertex colors.")
    
    # Test trimesh conversions
    print("\n--- Testing Trimesh Conversions (GLB, OBJ) ---")
    import trimesh
    mesh_tri = trimesh.load(refined_mesh_path, force="mesh")
    
    obj_path = os.path.join(out_dir, "test_export.obj")
    glb_path = os.path.join(out_dir, "test_export.glb")
    
    mesh_tri.export(obj_path, file_type="obj")
    mesh_tri.export(glb_path, file_type="glb")
    
    assert os.path.exists(obj_path), "Failed to export OBJ"
    assert os.path.exists(glb_path), "Failed to export GLB"
    print("[PASS] Trimesh exports to OBJ and GLB completed successfully.")
    
    # Cleanup temp test files
    for p in [cloud_path, obj_path, glb_path]:
        if os.path.exists(p):
            os.remove(p)
            
    print("\nALL STANDALONE RECONSTRUCTION PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    main()
