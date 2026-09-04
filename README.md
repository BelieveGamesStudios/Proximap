<img width="1918" height="1078" alt="Screenshot 2026-06-20 080015" src="https://github.com/user-attachments/assets/62a2b602-45c5-4e54-855d-76c43b8c07e0" />

<h1>Proximap — Open Source Photogrammetry</h1>

Proximap is an intuitive desktop application built in Python and PySide6 that provides a step-by-step wizard dashboard for 3D photogrammetry reconstruction. It acts as a graphical pipeline coordinator that automates **COLMAP** (for Structure-from-Motion / camera registration) and **OpenMVS** (for dense point clouds, meshing, mesh refinement, and texturing) to turn flat images into detailed 3D scenes.

---

<h2>Key Features</h2>

* **Drag-and-Drop Loader**: Simply drag a folder or selection of image files (`.png`, `.jpg`, `.jpeg`, `.tiff`) into the dashboard.
* **Camera Detection**: Autodetects your camera sensor make/model from image EXIF metadata.
* **Hardware Profiler**: Automatically profiles local system resources (RAM size, discrete GPU availability) and sets fallback limits for low-resource environments.
* **Reconstruction Quality Presets**: Select from Preview, Medium, High, or Ultra configuration presets to balance speed and fidelity.
* **Interactive Embedded 3D Viewer**: Interactively view reconstructed stages (Sparse Cloud & Cameras, Dense Cloud, and Textured Mesh) embedded directly inside the GUI application.
* **Multi-Format Export**: Export final textured 3D models directly as `.obj`, `.glb`, or `.ply`.

---

## Downloads

### macOS Client

Optimized for Apple Silicon (M1/M2/M3) and compatible with Intel Macs on macOS 11 or later. The macOS package includes the Proximap desktop client plus bundled COLMAP and OpenMVS reconstruction tools.

[Download Proximap for macOS](https://github.com/BelieveGamesStudios/Proximap/releases/latest/download/Proximap_Mac_Release.zip)
 
 After downloading, unzip `Proximap_Mac_Release.zip`, open `Proximap.app`, and select a folder of overlapping photos to start a reconstruction.
 
 ---
 
 ## Repository Structure
 
 ```text
 Proximap/
 ├── main_window.py          # PySide6 desktop GUI entry point
 ├── pipeline_manager.py     # Multi-stage photogrammetry pipeline coordinator
 ├── hardware_profiler.py    # Hardware checker (RAM and CUDA detection)
 │
 ├── package_app.ps1         # PowerShell automated build and compilation pipeline
 ├── installer.nsi           # NSIS Setup Wizard installer configuration
 ├── toolchain_map.json      # Config mapping GUI calls to C++ binaries
 │
 ├── LICENSE                 # GNU GPL v3 Source code license
 └── THIRD_PARTY_LICENSES.md # License attributions for OpenMVS, COLMAP, etc.
 ```
 
 ---
 
## Getting Started (Developers)

### Deep Mesh Fusion (Milestones 1–11)

The headless Deep Mesh Fusion foundation accepts independent `.ply` scan passes,
computes per-pass diagnostics, performs FPFH/RANSAC plus robust point-to-plane ICP
registration, records explicit quality metrics, and exports a derived voxel-fused
point cloud. Source files are referenced by path and SHA-256 and are never modified.

```bash
python -m deep_mesh_fusion.cli ./ApartmentFusion \
  ./Apartment/Pass_01.ply ./Apartment/Pass_02.ply \
  --voxel-size 0.03
```

The workspace contains `workspace.json`, per-pass transforms under
`registration/transforms/`, and the fused cloud under `derived/`. Registrations
below configured fitness, RMSE, or overlap thresholds are flagged for manual
alignment and excluded from fusion. A UI or other client can submit a corrected
source-to-reference 4×4 matrix through `DeepMeshFusionWorkspace.set_manual_transform`;
manual transforms pass through the same quality gates before fusion.

Milestone 2 adds a localized cross-pass evidence grid. Every occupied region
records observation count, per-pass density, consensus distance, normal
agreement, local surface consistency, potential missing observations, conflicts,
and explainable weighted confidence. The derived `analysis/spatial_evidence_map.json`
retains point-count and source/transform provenance; `analysis/confidence_map.ply`
provides red-to-green confidence cell centers for the 3D viewport. The CLI runs
this analysis after registration and accepts `--analysis-cell-size` when the
default four-times-voxel grid is not appropriate for the scene scale.

Milestone 3 replaces naïve concatenation with local consensus geometry. At the
fusion-cell scale, observations are clustered by cross-pass proximity, scored
for density, local surface quality, and registration reliability, then resolved
through robust Huber-weighted consensus or best-observation selection when
averaging would blur a disputed edge. Unsupported alternatives are suppressed
when a multi-pass cluster exists. The fused PLY propagates normals, colors,
confidence, observation count, and fusion method; the adjacent
`derived/fused_point_cloud.provenance.json` records every contributing pass,
weight, residual, representative point, and source hash.
Use `--fusion-cell-size` to tune the consensus resolution independently of the
coarser Milestone 2 evidence grid.

Milestone 4 adds a persistence-driven transient and artifact stage before
consensus fusion. Pass-specific cells are grouped into connected components and
scored using independent-pass support, other-pass coverage, conflict with
stronger persistent geometry, isolation, and continuity with stable structural
surfaces. This geometric-temporal model suppresses people-like occluders,
temporary or moving objects, curtains, floaters, and scan fragments without
claiming semantic object recognition. Coplanar single-pass wall patches can be
retained when their boundary remains consistent with persistent structure.
Decisions are written to `analysis/artifact_suppression.json`; rejected points
are written in red to `analysis/rejected_artifacts.ply` for viewport inspection.

The application workflow now uses a strict PyMeshLab-only geometry route.
PyMeshLab performs explicit source-to-reference ICP alignment, merges the
aligned layers, selects and removes
point-cloud outliers, merges duplicate/nearby samples, estimates and smooths
normals, and performs Screened Poisson reconstruction. PyMeshLab then removes
small components, closes remaining mesh holes, repairs duplicate/null elements,
reorients faces, and reports topology. A PyMeshLab failure stops the workflow;
there is no Open3D reconstruction fallback in this UI route. The mesh is written
to `derived/architecture_mesh.ply`, and PyMeshLab's processing/topology report is
stored in `analysis/architecture_reconstruction.json`.

Milestone 6 performs evidence-based gap recovery only after consensus fusion and
architecture reconstruction. Bounded mesh gaps are classified in this order:
reuse fused observations when present, infer a continuation only from a closed
high-confidence planar boundary, conservatively interpolate small closed complex
surface loops, or leave the gap open for review. Detected doorways and windows
are protected, and exterior mesh boundaries are never treated as holes. The
completed mesh is written to `derived/architecture_mesh_repaired.ply`; decisions
and repair confidence are recorded in `analysis/gap_recovery.json`. A colored
`analysis/gap_review.ply` marks repaired, unresolved, intentional, and exterior
regions for manual inspection. Use `--gap-max-planar-area` and
`--gap-min-confidence` to control inference conservatism.

Milestone 7 validates the repaired LiDAR-derived mesh before appearance work.
The audit checks holes and boundary loops, non-manifold edges, self-intersections,
degenerate and excessively stretched triangles, disconnected components, bad
normals, unexpected surface discontinuities, unresolved gaps, geometric
confidence, completeness, and consistency with detected architectural planes and
openings. `analysis/geometry_validation.json` contains defect evidence, review
locations, normalized completeness/surface/consistency/confidence scores, and an
explicit appearance-readiness decision. `analysis/geometry_quality.ply` provides
a red-to-green per-vertex quality overlay, while
`derived/validated_lidar_surface.ply` preserves the validated geometry and its
provenance properties. Readiness can be tuned with `--min-geometry-quality`, and
triangle stretching with `--max-triangle-aspect`.

Milestone 8 registers an existing COLMAP/OpenMVS photogrammetry reconstruction
to the validated LiDAR surface using a scale-aware coarse-to-fine similarity
alignment. It validates mutual 3D correspondences, projects mesh faces through
the registered COLMAP cameras with occlusion checks, measures single-view and
multi-view surface coverage, and scores each immutable source image for
sharpness, exposure, contrast, clipping, and resolution. The
`photogrammetry/` workspace folder contains the similarity transform and source
hashes, aligned source cloud, camera validation report, red-to-green texture
coverage mesh, and `texture_preparation.json` readiness summary. Supply a COLMAP
text model and its image root with `--photogrammetry-model` and `--image-root`;
an OpenMVS dense PLY can be selected with `--dense-photogrammetry-cloud`.

Milestone 9 projects the registered camera imagery across the validated fused
surface and produces a confidence-aware texture atlas. Every atlas texel is
visibility- and occlusion-tested, ranked by camera image quality and viewing
angle, filtered for color disagreement, then blended from the strongest
compatible observations. Geometry confidence is propagated into the texture
confidence map. UVs use deterministic non-overlapping face charts; global
camera color normalization, compatible-view blending, and transparent chart
padding reduce seams without inventing appearance for unobserved regions. The
`texture/` workspace folder contains `textured_environment.obj` and MTL,
`environment_albedo.png`, `texture_confidence.png`, and an auditable
`texture_baking.json` camera-selection report. Use `--bake-textures` and
`--texture-atlas-size` to run the stage from the CLI.

Milestone 10 inspects the baked artifact itself and separates safe repairs from
ambiguous appearance failures. It detects UV-chart seams, abnormal texel
density/stretching, missing or black texels, weak discontinuities, poor camera
selection, suspicious projection, and high-geometry/low-texture confidence
mismatches. Moderate supported seams and small bounded texel defects can be
repaired automatically; large, unsupported, or contradictory regions remain
explicit review items. The `final/` workspace folder contains the polished OBJ
and MTL, repaired albedo, final confidence and review maps, plus
`final_asset_validation.json` with before/after issues and geometry, texture,
coverage, consistency, and overall scores. Use `--finalize-asset` to run baking,
inspection, and controlled repair as one CLI workflow.

Milestone 11 is the final production quality gate. It aggregates geometry,
completeness, surface consistency, texture coverage, texture quality, critical
defects, unresolved reviews, registration state, immutable-source hashes, and
final artifact integrity under the configurable `virtual-tour-standard`
profile. A single blocking failure produces `NOT TOUR READY` with the responsible
stage and artifact; advisory exclusions remain visible without silently
overriding valid fused evidence. The `quality/` folder contains
`tour_readiness.json`, a human-readable `tour_readiness.html`, and
`tour_asset_manifest.json` with SHA-256 hashes for the final handoff files. Run
the complete finalization and gate with `--tour-readiness`.

The desktop application exposes this pipeline through the viewport-first
**Deep Mesh Fusion** tab.
PLY passes load into the resizable viewport immediately; six dependency-gated
stages guide preparation, alignment, geometry reconstruction, cleanup, texture,
and final quality. Alignment advances directly to geometry reconstruction while
the PyMeshLab consolidation required by reconstruction remains internal. Detailed metrics remain available in the
resizable diagnostics console. Existing photogrammetry
outputs are detected automatically, and **Run Reconstruction** returns users to
the normal 3D Reconstruction tab when no model exists. See
[`DEEP_MESH_FUSION_TESTING.md`](DEEP_MESH_FUSION_TESTING.md) for automated,
desktop, and real-data CLI testing instructions.

### 1. Prerequisites
 * **Python**: Version 3.9 or higher (64-bit recommended).
 * **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA drivers installed for GPU-accelerated mesh refinement and densification.
 
 ### 2. Installation
 Clone the repository and install dependencies:
 ```bash
git clone https://github.com/BelieveGamesStudios/Proximap.git
cd Proximap
pip install -r requirements.txt
```
*(Dependencies: `PySide6`, `numpy`, `pillow`, `pyinstaller` for packaging)*

### 3. Placing Backend Binaries
To run reconstructions, you must download precompiled C++ binaries for COLMAP and OpenMVS and place them in the `backend_bin` directory (which is ignored by Git to keep the repository lightweight):

1. **COLMAP**: Download the Windows release and extract it to `backend_bin/colmap/`.
2. **OpenMVS**: Download the Windows binary package and extract it to `backend_bin/openMVS/`.

Make sure your directories resemble the following structure:
```text
Proximap/
└── backend_bin/
    ├── colmap/
    │   ├── colmap.exe
    │   └── [Required DLLs...]
    └── openMVS/
        ├── DensifyPointCloud.exe
        ├── Viewer.exe
        └── [Required DLLs...]
```

### 4. Running the Dashboard
Launch the dashboard via Python:
```bash
python main_window.py
```

---

## Standalone Packaging & Distribution

You can bundle Proximap into a standalone zip file or platform installer.

### macOS

Follow the [macOS ARM64 Packaging Guide](PACKAGING_GUIDE_MACOS_ARM64.md) and run:

```bash
bash package_app.sh
```

The macOS package is written to `Proximap_Mac_Release.zip`.

### Windows

1. Place your custom logo as `app_icon.png` in the project root.
2. Run the automated PowerShell script as an Administrator:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\package_app.ps1
   ```
This script will automatically:
* Convert `app_icon.png` into a multi-resolution `.ico` icon using Pillow.
* Compile the Python scripts into a single directory layout using PyInstaller and embed the custom icon inside `Proximap.exe`.
* Strip out large developer libraries and test executables from COLMAP and OpenMVS to save ~450MB of space.
* Compile a Windows wizard installer (`Proximap_Setup.exe`) using NSIS (Nullsoft Scriptable Install System) if installed.
* Package the finalized distribution into a clean `Proximap_Commercial_Release.zip` folder.

---

## License

* The **Proximap** frontend source code is licensed under the **GNU General Public License v3 (GPL v3)**. See the [LICENSE](LICENSE) file for details.
* Native backend dependencies (OpenMVS, COLMAP, etc.) are subject to their own respective open-source licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full attribution.
