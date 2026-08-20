import os
import sys
import numpy as np

# Ensure current working directory is in sys.path
sys.path.insert(0, "/home/believestudios/ProximaXR Projects/Proximap")

def run_end_to_end_verification():
    print("=" * 60)
    print("STEP 6: End-to-End Verification of Ingestion, Alignment, & Export")
    print("=" * 60)

    # Test file paths
    obj_path = "reconstruction_out/mvs/scene_dense_mesh_texture.obj"
    ply_path = "reconstruction_out/mvs/scene_dense_mesh.ply"

    if not os.path.exists(obj_path):
        print(f"[SKIP] Test file {obj_path} does not exist.")
        return

    # 1. Test main_window.py photogrammetry ingestion
    from main_window import MainWindow, _read_ply_static
    print("\n1. Testing main_window.py OBJ & PLY Loaders...")
    
    # Static PLY loader
    points_ply_static, _, _ = _read_ply_static(ply_path)
    print(f"   _read_ply_static ({os.path.basename(ply_path)}):")
    print(f"     Point count: {len(points_ply_static)}")
    print(f"     Bounds: min={points_ply_static.min(axis=0)}, max={points_ply_static.max(axis=0)}")

    # OBJ loader (simulated without full Qt GUI instantiation)
    import trimesh
    from point_cloud_io import apply_photogrammetry_coordinate_flip
    raw_obj_mesh = trimesh.load(obj_path)
    if isinstance(raw_obj_mesh, trimesh.Scene):
        raw_obj_mesh = raw_obj_mesh.dump(concatenate=True)
    
    raw_verts = raw_obj_mesh.vertices
    flipped_obj_verts, _, _, _ = apply_photogrammetry_coordinate_flip(points=raw_verts)
    
    print(f"   main_window _read_obj ({os.path.basename(obj_path)}):")
    print(f"     Flipped OBJ bounds: min={flipped_obj_verts.min(axis=0)}, max={flipped_obj_verts.max(axis=0)}")

    # 2. Test Mesh Editor ingestion (load_mesh_file)
    print("\n2. Testing mesh_editor/scene.py load_mesh_file...")
    from mesh_editor.scene import load_mesh_file, export_scene_to_file, Scene, Object
    
    editor_meshes = load_mesh_file(obj_path)
    print(f"   load_mesh_file returned {len(editor_meshes)} mesh(es):")
    for name, mesh in editor_meshes:
        verts = mesh.vertices.reshape(-1, 3)
        print(f"     Mesh '{name}': {len(verts)} vertices")
        print(f"     Bounds: min={verts.min(axis=0)}, max={verts.max(axis=0)}")

    # 3. Cross-Tab Alignment Check
    print("\n3. Cross-Tab Alignment Verification:")
    diff_min = np.abs(flipped_obj_verts.min(axis=0) - verts.min(axis=0)).max()
    diff_max = np.abs(flipped_obj_verts.max(axis=0) - verts.max(axis=0)).max()
    print(f"   Max min-bound diff between main_window and mesh_editor: {diff_min:.6e}")
    print(f"   Max max-bound diff between main_window and mesh_editor: {diff_max:.6e}")
    assert diff_min < 1e-4 and diff_max < 1e-4, "Reconstruction and Mesh Editor bounds misaligned!"
    print("   [SUCCESS] 3D Reconstruction tab and Mesh Editor tab orientations match 100%!")

    # 4. Test Export Inverse Flip
    print("\n4. Testing Mesh Editor Export Inverse Flip (pipeline_native_coords)...")
    test_scene = Scene()
    obj_item = Object(name="TestExportObj", mesh=editor_meshes[0][1])
    test_scene.objects.append(obj_item)

    export_native_path = "scratch/exported_pipeline_native.obj"
    export_blender_path = "scratch/exported_blender_yup.obj"

    # Export with pipeline_native_coords=True (default)
    export_scene_to_file(test_scene, export_native_path, pipeline_native_coords=True)
    native_reloaded = trimesh.load(export_native_path)
    if isinstance(native_reloaded, trimesh.Scene):
        native_reloaded = native_reloaded.dump(concatenate=True)
    
    native_verts = native_reloaded.vertices
    print(f"   Exported Native Bounds: min={native_verts.min(axis=0)}, max={native_verts.max(axis=0)}")
    print(f"   Raw Input File Bounds:  min={raw_verts.min(axis=0)}, max={raw_verts.max(axis=0)}")

    diff_export_native = np.abs(native_verts - raw_verts).max()
    print(f"   Max diff between raw input file and pipeline exported file: {diff_export_native:.6e}")
    assert diff_export_native < 1e-4, "Export inverse matrix transformation failed to revert to COLMAP/OpenMVS coordinates!"
    print("   [SUCCESS] Exported mesh coordinates correctly reverted to native COLMAP/OpenMVS convention!")

    # Export with pipeline_native_coords=False
    export_scene_to_file(test_scene, export_blender_path, pipeline_native_coords=False)
    blender_reloaded = trimesh.load(export_blender_path)
    if isinstance(blender_reloaded, trimesh.Scene):
        blender_reloaded = blender_reloaded.dump(concatenate=True)
    
    blender_verts = blender_reloaded.vertices
    diff_export_blender = np.abs(blender_verts - verts).max()
    print(f"   Max diff between in-memory Y-up mesh and Blender exported file: {diff_export_blender:.6e}")
    assert diff_export_blender < 1e-4, "Blender Y-up export failed!"
    print("   [SUCCESS] Standalone export preserved standard Y-up viewport coordinates!")

    print("\n" + "=" * 60)
    print("ALL END-TO-END VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_end_to_end_verification()
