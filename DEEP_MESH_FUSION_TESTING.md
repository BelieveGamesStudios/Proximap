# Testing Deep Mesh Fusion

Deep Mesh Fusion is the replacement for the visible Spatial Texture Engine
workflow. It treats LiDAR as the geometry authority and registered
photogrammetry cameras as appearance observations.

## 1. Run the automated test suite

From the repository root, using the Python environment that contains the
packages in `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

The suite uses synthetic scan passes, COLMAP text and binary models, camera
images, damaged meshes, missing texture, and altered-source scenarios. It does
not modify production scan or image files.

To run only the end-to-end preparation milestones:

```bash
python -m unittest tests.test_deep_mesh_fusion -v
python -m unittest tests.test_photogrammetry_preparation -v
python -m unittest tests.test_texture_baking -v
python -m unittest tests.test_final_repair -v
python -m unittest tests.test_quality_gate -v
python -m unittest tests.test_deep_mesh_fusion_ui -v
```

## 2. Test from the desktop application

1. Start Proximap:

   ```bash
   python main_window.py
   ```

2. Open the **Deep Mesh Fusion** tab.
3. Drag at least two independent `.ply` scans from the file explorer directly
   onto the viewport, or select **Add PLY Passes**. The green drop overlay should
   appear while dragging. Each scan should then appear immediately in the 3D
   viewport, and the camera should fit all visible passes.
4. Exercise the scan controls: click a row to select it, toggle visibility,
   rename it, and remove/re-add it. Source files remain unchanged.
5. Select **Continue**. Per-pass diagnostics run and Proximap unlocks **Scan
   alignment**. Future stages remain locked until their prerequisites finish.
6. On **Scan alignment**, leave **Auto-select best reference** enabled or choose
   a reference scan, then select **Align Point Clouds**. Inspect fitness and
   excluded-pass results. Use **Export Unified Point Cloud** to test both the
   lossless and voxel-downsampled PLY exports. Select **Continue**.
7. On **Surface fusion**, choose a preset (and optionally expand its exact
   controls), then select **Fuse Point Clouds**. The viewport must show one
   evidence-filtered point cloud—not a solid mesh. Change a parameter and use
   **Rerun Point Fusion** to confirm downstream stages are invalidated. Select
   **Continue** when satisfied.
8. On **Surface validation**, choose the architecture up axis and Screened
   Poisson octree depth, then select **Reconstruct Surface**. The viewport now
   shows the architecture-aware hybrid solid surface. Select **Continue**.
9. On **Cleanup**, try **Cleanup & Reduce Mesh**, **Repair Non-Manifold Edges**,
   **Close Holes**, **Merge Close Vertices**, and **Smooth Mesh** individually.
   Each result is cumulative and immediately visible; **Reset Cleanup** restores
   the original reconstructed surface. **Proceed** revalidates the working mesh.
10. If no photogrammetry reconstruction exists, select **Run Reconstruction**
   and confirm Proximap opens its existing **3D Reconstruction** tab. Complete
   the normal reconstruction there, then return to **Deep Mesh Fusion**.
11. On **Texture**, confirm the recovered reconstruction loads automatically.
    Run **Register Photogrammetry**, inspect its result, then run **Bake Texture**.
    These are deliberately separate operations.
12. Select **Continue** to open **Final quality**, then **Open Quality Report**.

The top stages are clickable only after their prerequisites have completed.
Completed stages remain available for inspection and reruns. The resize handle
above the console should keep the viewport dominant while still allowing a
full diagnostics view.

Enable **Allow provisional review** only when testing imperfect data. This
permits expert inspection, but it does not turn a failed gate into `TOUR READY`.

## 3. Test from the command line with real data

Use a new workspace path for each clean run:

```bash
python -m deep_mesh_fusion.cli \
  /absolute/path/ApartmentFusion \
  /absolute/path/Apartment/Pass_01.ply \
  /absolute/path/Apartment/Pass_02.ply \
  /absolute/path/Apartment/Pass_03.ply \
  --voxel-size 0.03 \
  --up-axis y \
  --photogrammetry-model /absolute/path/reconstruction_out/colmap/sparse/0 \
  --image-root /absolute/path/reconstruction_out/input_images \
  --dense-photogrammetry-cloud /absolute/path/reconstruction_out/mvs/scene_dense.ply \
  --tour-readiness
```

`--tour-readiness` runs photogrammetry preparation, texture baking, final
surface/texture repair, and the final quality gate. For a geometry-only test,
omit the photogrammetry paths and `--tour-readiness`.

## 4. Inspect the outputs

Important derived artifacts are:

```text
ApartmentFusion/
├── workspace.json
├── derived/
│   ├── fused_point_cloud.ply
│   ├── architecture_mesh_repaired.ply
│   ├── validated_lidar_surface.ply
│   └── cleanup/
│       ├── reconstructed_original.ply
│       └── cleanup_working.ply
├── analysis/
│   ├── spatial_evidence_map.json
│   ├── geometry_validation.json
│   └── geometry_quality.ply
├── photogrammetry/
│   ├── photogrammetry_registration.json
│   └── texture_coverage.ply
├── texture/
│   ├── textured_environment.obj
│   ├── environment_albedo.png
│   └── texture_confidence.png
├── final/
│   ├── final_environment.obj
│   ├── final_albedo.png
│   └── final_asset_validation.json
└── quality/
    ├── tour_readiness.json
    ├── tour_readiness.html
    └── tour_asset_manifest.json
```

The source `.ply`, COLMAP model, and source images should retain their original
SHA-256 hashes. `tour_asset_manifest.json` records the final handoff hashes.

## 5. Interpreting a failed gate

`NOT TOUR READY` is a useful test outcome. Open `quality/tour_readiness.html`
and address blocking issues in their originating stage:

- registration conflict: adjust alignment or provide a verified manual transform;
- geometry gaps: add another scan pass or review gap recovery;
- texture-confidence regions: add sharper/less-obstructed camera coverage;
- source integrity: restore the original referenced source;
- artifact integrity: rerun finalization in a new workspace.

Do not edit derived JSON to force readiness; the next run will regenerate it
from the measured evidence.
