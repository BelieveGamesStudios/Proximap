"""
Pipeline Manager Module
Manages the execution of the photogrammetry pipeline (COLMAP and OpenMVS)
in background threads to keep the UI responsive.

Architecture:
  COLMAP Phase (Structure from Motion):
    1. Image Preparation          — downscale working copies to colmap/images/
    2. Feature Extraction         — SIFT via iGPU (OpenGL) or CPU fallback
    3. Feature Matching           — exhaustive matching with guided matching
    4. Mapper (SfM)               — incremental camera pose estimation + BA
       Bundle Adjuster            — optional extra polish (High/Ultra only)
    5. Export to OpenMVS           — InterfaceCOLMAP converts sparse model

  OpenMVS Phase (Multi-View Stereo → Mesh → Texture):
    6. DensifyPointCloud          — depth-map fusion → dense point cloud
    7. ReconstructMesh            — Delaunay surface reconstruction
    8. RefineMesh                 — multi-scale mesh geometry refinement
    9. TextureMesh                — project image textures onto final mesh

  Diagnostic Logging:
    - Real-time subprocess output parsing via line_parser callbacks
    - Post-step summary reports (feature, matching, SfM, dense, mesh)
    - SQLite database querying for precise COLMAP statistics
    - Threshold-based diagnostic warnings (low features, poor registration)
"""

import os
import sys
import json
import time
import shutil
from PySide6.QtCore import QThread, Signal
from hardware_profiler import run_safe_subprocess


def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


class PipelineWorker(QThread):
    """
    Worker thread that executes the photogrammetry toolchain step-by-step.
    Emits progress and logging signals to keep the UI responsive.
    """
    progress_changed = Signal(int)
    status_changed = Signal(str)
    log_message = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, image_dir: str, output_dir: str, quality_preset: str = "medium", gpu_mode: str = "auto", parent=None):
        super().__init__(parent)
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.quality_preset = quality_preset
        self.gpu_mode = gpu_mode
        self.is_running = True
        self.toolchain_map = self._load_toolchain_map()
        self.last_output_lines = []
        self._last_reconstruction_stats = {}

        # Diagnostic tracking (Metashape-style)
        self._feature_counts = []          # Per-image feature counts
        self._match_counts = []            # Per-pair match counts
        self._pairs_tested = 0             # Total image pairs tested
        self._pairs_matched = 0            # Pairs with verified matches
        self._registered_count = 0         # Cameras registered by mapper
        self._total_images = 0             # Total input images
        self._triangulated_points = 0      # 3D points from SfM
        self._mean_reproj_error = 0.0      # Mean reprojection error
        self._using_gpu_sift = True        # Whether GPU SIFT is being used
        self._depth_map_count = 0          # Number of depth maps computed
        self._dense_point_count = 0        # Points in dense cloud
        self._mesh_vertices = 0            # Mesh vertex count
        self._mesh_faces = 0               # Mesh face count
        self._spurious_removed = 0
        self._spikes_removed = 0
        self._holes_closed = 0

    def _load_toolchain_map(self) -> dict:
        """Loads the toolchain mapping config file."""
        map_path = os.path.join(get_base_dir(), "toolchain_map.json")
        if os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.log_message.emit(f"Error reading toolchain_map.json: {e}")
        return {}

    def run(self):
        try:
            self.status_changed.emit("Initializing Pipeline...")
            self.progress_changed.emit(5)
            time.sleep(0.5)

            has_binaries = self._verify_binaries()

            if has_binaries:
                self.log_message.emit("Valid toolchain detected. Running production pipeline...")
                success = self._run_real_pipeline()
            else:
                self.log_message.emit("Toolchain binaries or test data missing. Running pipeline simulation...")
                success = self._run_simulated_pipeline()

            if success:
                self.progress_changed.emit(100)
                self.status_changed.emit("Pipeline Completed Successfully!")
                self.finished.emit(True, "Mesh reconstruction completed.")
            else:
                self.status_changed.emit("Pipeline Failed!")
                self.finished.emit(False, "Pipeline failed or was cancelled.")

        except Exception as e:
            self.status_changed.emit("Pipeline Error!")
            self.log_message.emit(f"Unhandled pipeline exception: {e}")
            self.finished.emit(False, str(e))

    def _verify_binaries(self) -> bool:
        """Checks if all required binaries in toolchain_map.json exist."""
        if not self.toolchain_map:
            return False

        colmap_map = self.toolchain_map.get("colmap", {})
        for name, rel_path in colmap_map.items():
            abs_path = os.path.join(get_base_dir(), rel_path)
            if not os.path.exists(abs_path):
                self.log_message.emit(f"Missing COLMAP binary: {name} ({rel_path})")
                return False

        mvs_map = self.toolchain_map.get("openMVS", {})
        for name, rel_path in mvs_map.items():
            abs_path = os.path.join(get_base_dir(), rel_path)
            if not os.path.exists(abs_path):
                self.log_message.emit(f"Missing OpenMVS binary: {name} ({rel_path})")
                return False

        return True

    def _get_colmap_env(self) -> dict:
        """
        Build a sanitized environment for COLMAP subprocesses.
        COLMAP 4.x is a Qt6 app and needs its own QT_PLUGIN_PATH
        pointing to its bundled plugins/ directory. Without this,
        it inherits Proximap's PyQt5 env vars and crashes with
        'no Qt platform plugin could be initialized'.
        """
        colmap_dir = os.path.join(
            get_base_dir(), 
            os.path.dirname(self.toolchain_map["colmap"]["colmap"])
        )
        colmap_plugins = os.path.join(colmap_dir, "plugins")
        
        env = os.environ.copy()
        env["QT_PLUGIN_PATH"] = colmap_plugins
        # Remove any conflicting Qt env vars from the parent process
        env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        return env

    def _run_process_realtime(self, cmd: list, timeout: float, cwd=None, env=None, line_parser=None) -> bool:
        """
        Runs a subprocess and streams its stdout/stderr to the console log in real time.
        Allows users to see step-by-step progress as it occurs.
        Returns True if the process exited with code 0, False otherwise.
        """
        import subprocess
        import glob
        from hardware_profiler import _active_subprocesses

        self.log_message.emit(f"[RUN] {' '.join(cmd)}")

        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_NO_WINDOW

        # Identify log file if it's an OpenMVS command
        exe_name = os.path.splitext(os.path.basename(cmd[0]))[0]
        # OpenMVS tools: InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, RefineMesh, TextureMesh
        is_openmvs = exe_name in ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                env=env,
                bufsize=1,  # Line-buffered
                creationflags=creationflags
            )
            _active_subprocesses.add(proc)
            start_time = time.time()

            log_handle = None
            if is_openmvs:
                # Wait up to 2 seconds for log file to appear in the working directory
                log_dir = cwd if cwd else os.getcwd()
                log_pattern = os.path.join(log_dir, f"{exe_name}-*.log")
                for _ in range(20):
                    if not self.is_running:
                        break
                    log_files = glob.glob(log_pattern)
                    if log_files:
                        # Pick the newest matching file
                        newest_file = max(log_files, key=os.path.getmtime)
                        try:
                            log_handle = open(newest_file, "r", encoding="utf-8", errors="ignore")
                            break
                        except Exception:
                            pass
                    time.sleep(0.1)

            if log_handle:
                # Log-tailing loop for OpenMVS
                while True:
                    if not self.is_running:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        log_handle.close()
                        return False

                    if time.time() - start_time > timeout:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        log_handle.close()
                        self.log_message.emit(f"[TIMEOUT] Process timed out after {timeout}s")
                        return False

                    line = log_handle.readline()
                    if not line:
                        if proc.poll() is not None:
                            # Process ended, check one last time for remaining lines
                            while True:
                                extra_line = log_handle.readline()
                                if not extra_line:
                                    break
                                clean_line = extra_line.strip()
                                if clean_line:
                                    if line_parser:
                                        parsed = line_parser(clean_line)
                                        if parsed is not None:
                                            self.log_message.emit(parsed)
                                        else:
                                            self.log_message.emit(clean_line)
                                    else:
                                        self.log_message.emit(clean_line)
                                    self.last_output_lines.append(clean_line)
                            break
                        time.sleep(0.05)
                        continue

                    clean_line = line.strip()
                    if clean_line:
                        if line_parser:
                            parsed = line_parser(clean_line)
                            if parsed is not None:
                                self.log_message.emit(parsed)
                            else:
                                self.log_message.emit(clean_line)
                        else:
                            self.log_message.emit(clean_line)
                        self.last_output_lines.append(clean_line)
                log_handle.close()
            else:
                # Standard stdout reading loop for non-OpenMVS tools (or fallback)
                while True:
                    if not self.is_running:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        return False

                    if time.time() - start_time > timeout:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        self.log_message.emit(f"[TIMEOUT] Process timed out after {timeout}s")
                        return False

                    line = proc.stdout.readline()
                    if not line:
                        if proc.poll() is not None:
                            break
                        time.sleep(0.02)
                        continue

                    clean_line = line.strip()
                    if clean_line:
                        if line_parser:
                            parsed = line_parser(clean_line)
                            if parsed is not None:
                                self.log_message.emit(parsed)
                            else:
                                self.log_message.emit(clean_line)
                        else:
                            self.log_message.emit(clean_line)
                        self.last_output_lines.append(clean_line)

            _active_subprocesses.discard(proc)
            return proc.returncode == 0

        except Exception as e:
            self.log_message.emit(f"[ERROR] Failed to run subprocess: {e}")
            return False

    def _run_real_pipeline(self) -> bool:
        """
        Executes the full 9-step COLMAP + OpenMVS photogrammetry pipeline.
        Each step is heavily parameterized for maximum reconstruction quality.
        """
        base_dir = get_base_dir()
        
        # Clean up stale reconstruction directory to prevent legacy files impacting new scans
        if os.path.exists(self.output_dir):
            self.log_message.emit(f"[INFO] Cleaning up stale reconstruction directory: {self.output_dir}")
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                self.log_message.emit(f"[WARNING] Failed to clean output folder: {e}")

        colmap_out = os.path.join(self.output_dir, "colmap")
        mvs_out = os.path.join(self.output_dir, "mvs")
        os.makedirs(colmap_out, exist_ok=True)
        os.makedirs(mvs_out, exist_ok=True)
        os.makedirs(os.path.join(colmap_out, "sparse"), exist_ok=True)

        num_threads = os.cpu_count() or 4

        # --- GPU / CUDA Environment (used mainly by OpenMVS) ---
        env = os.environ.copy()
        if self.gpu_mode == "force_cpu":
            self.log_message.emit("[INFO] Hardware Acceleration: Forcing CPU fallback.")
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif self.gpu_mode == "force_gpu":
            self.log_message.emit("[INFO] Hardware Acceleration: Forcing GPU (CUDA) execution.")
        else:  # auto
            try:
                parent_has_dgpu = self.parent().dgpu_detected
            except Exception:
                parent_has_dgpu = False
            if not parent_has_dgpu:
                self.log_message.emit("[INFO] Hardware Acceleration: No dedicated GPU detected. Falling back to CPU.")
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                self.log_message.emit("[INFO] Hardware Acceleration: Dedicated GPU detected. Using CUDA.")

        # -------------------------------------------------------------------------
        # QUALITY PRESET PARAMETERS
        # -------------------------------------------------------------------------
        if self.quality_preset == "preview":
            max_image_dim  = 1024
            colmap_max_image_size = 1024
            colmap_max_num_features = 4096
            colmap_first_octave = 0
            guided_matching = "0"
            nndr_ratio     = "0.8"
            ba_global_max_refinements = 3
            run_bundle_adjuster = False
            densify_res    = "2"
            densify_views  = "3"
            max_res        = "1920"
            refine_scales  = "1"
            refine_res     = "2"
            texture_res    = "2"
        elif self.quality_preset == "medium":
            max_image_dim  = 2048
            colmap_max_image_size = 2048
            colmap_max_num_features = 8192
            colmap_first_octave = -1
            guided_matching = "0"
            nndr_ratio     = "0.8"
            ba_global_max_refinements = 5
            run_bundle_adjuster = False
            densify_res    = "1"
            densify_views  = "4"
            max_res        = "2560"
            refine_scales  = "2"
            refine_res     = "1"
            texture_res    = "1"
        elif self.quality_preset == "high":
            max_image_dim  = 3200
            colmap_max_image_size = 3200
            colmap_max_num_features = 12288
            colmap_first_octave = -1
            guided_matching = "1"
            nndr_ratio     = "0.8"
            ba_global_max_refinements = 5
            run_bundle_adjuster = True
            densify_res    = "1"
            densify_views  = "5"
            max_res        = "3200"
            refine_scales  = "2"
            refine_res     = "0"
            texture_res    = "1"
        else:  # ultra
            max_image_dim  = None
            colmap_max_image_size = -1
            colmap_max_num_features = 16384
            colmap_first_octave = -1
            guided_matching = "1"
            nndr_ratio     = "0.8"
            ba_global_max_refinements = 5
            run_bundle_adjuster = True
            densify_res    = "0"
            densify_views  = "8"
            max_res        = "4096"
            refine_scales  = "3"
            refine_res     = "0"
            texture_res    = "0"

        # =========================================================================
        # STEP 1/9 — Image Preparation
        # =========================================================================
        self.status_changed.emit("Step 1/9: Preparing Images...")
        working_image_dir = self._prepare_images(
            self.image_dir, self.output_dir, max_image_dim
        )
        try:
            self._total_images = len([
                f for f in os.listdir(working_image_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
            ])
        except Exception:
            self._total_images = 0
        self.progress_changed.emit(10)

        colmap_exe = os.path.join(base_dir, self.toolchain_map["colmap"]["colmap"])
        colmap_env = self._get_colmap_env()
        database_path = os.path.join(colmap_out, "database.db")
        if os.path.exists(database_path):
            try:
                os.remove(database_path)
                self.log_message.emit("[INFO] Cleared stale COLMAP database.")
            except Exception as e:
                self.log_message.emit(f"[WARNING] Failed to clear database: {e}")

        # =========================================================================
        # STEP 2/9 — Feature Extraction
        # =========================================================================
        self.status_changed.emit("Step 2/9: Extracting SIFT Features...")
        cmd_extract_gpu = [
            colmap_exe, "feature_extractor",
            "--database_path", database_path,
            "--image_path", working_image_dir,
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "1",
            "--FeatureExtraction.max_image_size", str(colmap_max_image_size),
            "--SiftExtraction.max_num_features", str(colmap_max_num_features),
            "--SiftExtraction.first_octave", str(colmap_first_octave),
            "--FeatureExtraction.num_threads", str(num_threads),
        ]
        cmd_extract_cpu = [
            colmap_exe, "feature_extractor",
            "--database_path", database_path,
            "--image_path", working_image_dir,
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "0",
            "--FeatureExtraction.max_image_size", str(colmap_max_image_size),
            "--SiftExtraction.max_num_features", str(colmap_max_num_features),
            "--SiftExtraction.first_octave", str(colmap_first_octave),
            "--FeatureExtraction.num_threads", str(num_threads),
        ]
        self._feature_counts = []
        if not self._run_with_gpu_fallback(
            cmd_extract_gpu, cmd_extract_cpu, timeout=3600.0, env=colmap_env,
            line_parser=self._parse_feature_extraction_line
        ):
            return False
        
        # Query database for exact feature counts
        db_stats = self._query_colmap_database_stats(database_path)
        if db_stats["feature_counts"]:
            self._feature_counts = db_stats["feature_counts"]
        self._emit_feature_summary()
        self.progress_changed.emit(25)

        # =========================================================================
        # STEP 3/9 — Feature Matching
        # =========================================================================
        self.status_changed.emit("Step 3/9: Matching SIFT Features...")
        cmd_match_gpu = [
            colmap_exe, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.guided_matching", guided_matching,
            "--SiftMatching.max_ratio", nndr_ratio,
            "--FeatureMatching.num_threads", str(num_threads),
        ]
        cmd_match_cpu = [
            colmap_exe, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", "0",
            "--FeatureMatching.guided_matching", guided_matching,
            "--SiftMatching.max_ratio", nndr_ratio,
            "--FeatureMatching.num_threads", str(num_threads),
        ]
        self._match_counts = []
        if not self._run_with_gpu_fallback(
            cmd_match_gpu, cmd_match_cpu, timeout=3600.0, env=colmap_env,
            line_parser=self._parse_matching_line
        ):
            return False
            
        # Query database for exact matching stats
        db_stats = self._query_colmap_database_stats(database_path)
        if db_stats["num_images"] > 0:
            self._pairs_tested = (db_stats["num_images"] * (db_stats["num_images"] - 1)) // 2
        else:
            self._pairs_tested = (self._total_images * (self._total_images - 1)) // 2 if self._total_images > 1 else 0
        self._pairs_matched = db_stats["num_pairs"]
        if db_stats["match_counts"]:
            self._match_counts = db_stats["match_counts"]
            
        self._emit_matching_summary()
        self.progress_changed.emit(40)

        # =========================================================================
        # STEP 4/9 — Sparse Reconstruction (Mapper)
        # =========================================================================
        self.status_changed.emit("Step 4/9: Estimating Camera Poses (SfM)...")
        sparse_dir = os.path.join(colmap_out, "sparse")
        if os.path.exists(sparse_dir):
            try:
                shutil.rmtree(sparse_dir)
            except Exception as e:
                self.log_message.emit(f"[WARNING] Failed to clean sparse folder: {e}")
        os.makedirs(sparse_dir, exist_ok=True)

        cmd_mapper = [
            colmap_exe, "mapper",
            "--database_path", database_path,
            "--image_path", working_image_dir,
            "--output_path", sparse_dir,
            "--Mapper.ba_global_max_refinements", str(ba_global_max_refinements),
            "--Mapper.ba_local_max_refinements", "3",
            "--Mapper.min_num_matches", "15",
            "--Mapper.init_min_num_inliers", "100",
            "--Mapper.abs_pose_min_num_inliers", "15",
            "--Mapper.abs_pose_min_inlier_ratio", "0.25",
            "--Mapper.num_threads", str(num_threads),
        ]
        self._triangulated_points = 0
        self._registered_count = 0
        if not self._run_process_realtime(cmd_mapper, timeout=3600.0, env=colmap_env, line_parser=self._parse_mapper_line):
            return False

        best_model_dir = self._select_best_sparse_model(sparse_dir)
        if not best_model_dir:
            self.log_message.emit(
                "[FAILED] SfM registered 0 camera poses. Feature matching produced "
                "insufficient geometric correspondences to initialise reconstruction.\n"
                "  Suggestions:\n"
                "  • Try a higher quality preset (Medium or High)\n"
                "  • Ensure images have at least 60% overlap between adjacent shots"
            )
            return False

        target_model_dir = os.path.join(sparse_dir, "0")
        if os.path.abspath(best_model_dir) != os.path.abspath(target_model_dir):
            if os.path.exists(target_model_dir):
                shutil.rmtree(target_model_dir)
            try:
                shutil.move(best_model_dir, target_model_dir)
            except Exception as e:
                self.log_message.emit(f"[WARNING] Failed to move best model folder: {e}")

        # Optional bundle adjuster polish pass
        if run_bundle_adjuster:
            self.log_message.emit("[INFO] Running extra bundle adjuster refinement...")
            cmd_ba = [
                colmap_exe, "bundle_adjuster",
                "--input_path", target_model_dir,
                "--output_path", target_model_dir,
                "--BundleAdjustmentCeres.max_num_iterations", "100",
                "--BundleAdjustment.refine_focal_length", "1",
                "--BundleAdjustment.refine_principal_point", "0",
                "--BundleAdjustment.refine_extra_params", "1",
            ]
            self._run_process_realtime(cmd_ba, timeout=600.0, env=colmap_env, line_parser=self._parse_mapper_line)

        # Get reconstruction statistics
        self._last_reconstruction_stats = self._run_model_analyzer(target_model_dir)
        if "images" in self._last_reconstruction_stats:
            self._registered_count = self._last_reconstruction_stats["images"]
        if "points" in self._last_reconstruction_stats:
            self._triangulated_points = self._last_reconstruction_stats["points"]
        if "mean_error" in self._last_reconstruction_stats:
            self._mean_reproj_error = self._last_reconstruction_stats["mean_error"]

        self._emit_sfm_summary()
        self.progress_changed.emit(60)

        # =========================================================================
        # STEP 5/9 — Export to OpenMVS Format
        # =========================================================================
        self.status_changed.emit("Step 5/9: Exporting Scene to OpenMVS...")

        # Copy sparse model files to the parent sparse directory so InterfaceCOLMAP can find them
        try:
            for filename in os.listdir(target_model_dir):
                src_file = os.path.join(target_model_dir, filename)
                dst_file = os.path.join(sparse_dir, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
            self.log_message.emit("[INFO] Copied sparse model files to parent directory for InterfaceCOLMAP.")
        except Exception as e:
            self.log_message.emit(f"[WARNING] Failed to copy sparse model files to parent: {e}")

        mvs_export_exe = os.path.join(base_dir, self.toolchain_map["openMVS"]["InterfaceCOLMAP"])
        os.makedirs(os.path.join(mvs_out, "images"), exist_ok=True)
        cmd_export = [
            mvs_export_exe,
            "-i", colmap_out,
            "--image-folder", os.path.join(colmap_out, "images"),
            "-o", os.path.join(mvs_out, "scene.mvs"),
        ]
        if not self._run_process_realtime(cmd_export, timeout=300.0):
            return False
        self.progress_changed.emit(70)

        # =========================================================================
        # STEP 6/9 — Dense Point Cloud Generation
        # =========================================================================
        self.status_changed.emit("Step 6/9: Generating Dense Point Cloud...")
        mvs_densify_exe = os.path.join(base_dir, self.toolchain_map["openMVS"]["DensifyPointCloud"])

        sparse_point_count = self._count_scene_points(mvs_out)
        calibrated_count = self._count_calibrated_images(mvs_out)

        actual_densify_views = densify_views
        actual_fuse_views = "2"
        if sparse_point_count < 500 or calibrated_count < 15:
            actual_densify_views = str(min(int(densify_views), max(2, calibrated_count - 1)))
            actual_fuse_views = "1"
            self.log_message.emit(f"[ADAPT] Low sparse data ({sparse_point_count} pts, {calibrated_count} cal imgs). "
                                   f"Reducing --number-views to {actual_densify_views}, --number-views-fuse to {actual_fuse_views}")

        cmd = [
            mvs_densify_exe,
            "scene.mvs",
            "--resolution-level",    densify_res,
            "--max-resolution",      max_res,
            "--number-views",        actual_densify_views,
            "--number-views-fuse",   actual_fuse_views,
            "--geometric-iters",     "2",
            "--estimate-colors",     "2",
            "--estimate-normals",    "2",
        ]
        self._depth_map_count = 0
        self._dense_point_count = 0
        densify_ok = self._run_process_realtime(cmd, timeout=7200.0, cwd=mvs_out, env=env, line_parser=self._parse_densify_line)
        if not densify_ok:
            self.log_message.emit("[WARNING] DensifyPointCloud failed or returned no points! ReconstructMesh will use sparse cloud.")
        else:
            self._emit_dense_summary()
        self.progress_changed.emit(80)

        # =========================================================================
        # STEP 7/9 — Surface Mesh Reconstruction
        # =========================================================================
        self.status_changed.emit("Step 7/9: Reconstructing Surface Mesh...")
        mvs_mesh_exe = os.path.join(base_dir, self.toolchain_map["openMVS"]["ReconstructMesh"])

        dense_mvs = os.path.join(mvs_out, "scene_dense.mvs")
        target_scene = "scene_dense.mvs" if os.path.exists(dense_mvs) else "scene.mvs"
        if target_scene == "scene.mvs":
            self.log_message.emit("[WARNING] scene_dense.mvs not found. Meshing from sparse scene.mvs.")

        cmd = [
            mvs_mesh_exe,
            target_scene,
            "--remove-spurious", "20",
            "--remove-spikes",   "1",
            "--close-holes",     "30",
            "--smooth",          "2",
        ]
        self._mesh_vertices = 0
        self._mesh_faces = 0
        self._spurious_removed = 0
        self._spikes_removed = 0
        self._holes_closed = 0
        if not self._run_process_realtime(cmd, timeout=1800.0, cwd=mvs_out, env=env, line_parser=self._parse_mesh_line):
            return False
        self._emit_mesh_summary()
        self.progress_changed.emit(88)

        # =========================================================================
        # STEP 8/9 — Mesh Geometry Refinement (Multi-Scale)
        # =========================================================================
        self.status_changed.emit("Step 8/9: Refining Mesh Geometry...")
        mvs_refine_exe = os.path.join(base_dir, self.toolchain_map["openMVS"]["RefineMesh"])

        if target_scene == "scene_dense.mvs":
            mesh_input = "scene_dense_mesh.ply"
        else:
            mesh_input = "scene_mesh.ply"

        if not os.path.exists(os.path.join(mvs_out, mesh_input)):
            for candidate in ["scene_dense_mesh.ply", "scene_mesh.ply"]:
                if os.path.exists(os.path.join(mvs_out, candidate)):
                    mesh_input = candidate
                    break

        refine_mvs_output = "scene_dense_mesh_refine.mvs"
        cmd = [
            mvs_refine_exe,
            target_scene,
            "-m",              mesh_input,
            "-o",              refine_mvs_output,
            "--resolution-level", refine_res,
            "--scales",        refine_scales,
            "--gradient-step", "25.05",
            "--max-face-area", "16",
        ]

        refine_ok = self._run_process_realtime(cmd, timeout=7200.0, cwd=mvs_out, env=env)
        refined_mvs_path = os.path.join(mvs_out, refine_mvs_output)
        refined_ply_path = os.path.join(mvs_out, "scene_dense_mesh_refine.ply")

        if refine_ok and (os.path.exists(refined_mvs_path) or os.path.exists(refined_ply_path)):
            texture_input_scene = refine_mvs_output if os.path.exists(refined_mvs_path) else target_scene
            self.log_message.emit(f"[INFO] RefineMesh succeeded. Using {texture_input_scene} for texturing.")
        else:
            texture_input_scene = target_scene
            self.log_message.emit(f"[WARNING] RefineMesh failed or produced no output. Texturing will use {target_scene}.")

        self.progress_changed.emit(94)

        # =========================================================================
        # STEP 9/9 — Texture Projection
        # =========================================================================
        self.status_changed.emit("Step 9/9: Projecting Textures onto Mesh...")
        mvs_texture_exe = os.path.join(base_dir, self.toolchain_map["openMVS"]["TextureMesh"])

        texture_mesh_ply = None
        for candidate in ["scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
            if os.path.exists(os.path.join(mvs_out, candidate)):
                texture_mesh_ply = candidate
                break

        if not texture_mesh_ply:
            self.log_message.emit("[WARNING] No mesh PLY found for texturing. Skipping TextureMesh.")
            self.progress_changed.emit(99)
            return True

        cmd_ply = [
            mvs_texture_exe,
            texture_input_scene,
            "-m",                    texture_mesh_ply,
            "-o",                    "scene_dense_mesh_texture.mvs",
            "--resolution-level",    texture_res,
            "--cost-smoothness-ratio", "0.1",
            "--empty-color",         "0",
            "--local-seam-leveling",  "0",       # Force turns off local patch edge blending
            "--global-seam-leveling", "0",       # Force turns off global color adjustment
        ]
        texture_ply_ok = self._run_process_realtime(cmd_ply, timeout=1800.0, cwd=mvs_out, env=env)
        if not texture_ply_ok:
            self.log_message.emit("[WARNING] TextureMesh MVS/PLY pass failed. Final reconstruction may lack textures.")

        if texture_ply_ok:
            cmd_obj = [
                mvs_texture_exe,
                texture_input_scene,
                "-m",                    texture_mesh_ply,
                "-o",                    "scene_dense_mesh_texture.obj",
                "--export-type",         "obj",
                "--resolution-level",    texture_res,
                "--cost-smoothness-ratio", "0.1",
                "--empty-color",         "0",
                "--local-seam-leveling",  "0",       # Force turns off local patch edge blending
                "--global-seam-leveling", "0",       # Force turns off global color adjustment
            ]
            self._run_process_realtime(cmd_obj, timeout=1800.0, cwd=mvs_out, env=env)
        else:
            self.log_message.emit("[WARNING] TextureMesh PLY pass failed. Skipping OBJ export pass.")

        self.progress_changed.emit(99)
        return True

    def _run_simulated_pipeline(self) -> bool:
        """Runs a visual simulation of the pipeline for testing UI and fallback states."""
        steps = [
            ("Step 1/9: Preparing Images...", 10, 0.8),
            ("Step 2/9: Extracting SIFT Features...", 25, 1.2),
            ("Step 3/9: Matching SIFT Features...", 40, 1.0),
            ("Step 4/9: Estimating Camera Poses (SfM)...", 60, 1.5),
            ("Step 5/9: Exporting Scene to OpenMVS...", 70, 0.6),
            ("Step 6/9: Generating Dense Point Cloud...", 80, 1.5),
            ("Step 7/9: Reconstructing Surface Mesh...", 88, 1.0),
            ("Step 8/9: Refining Mesh Geometry...", 94, 1.0),
            ("Step 9/9: Projecting Textures onto Mesh...", 99, 0.8),
        ]

        for status, progress, duration in steps:
            if not self.is_running:
                return False
            self.status_changed.emit(status)
            self.log_message.emit(f"[SIM] {status.split(':')[0]}...")

            ticks = int(duration * 10)
            for _ in range(ticks):
                if not self.is_running:
                    return False
                time.sleep(0.1)

            self.log_message.emit(f"[SIM] {status.split(':')[0]} — done.")
            self.progress_changed.emit(progress)

        return True

    def stop(self):
        """Request pipeline worker thread termination."""
        self.is_running = False

    def _prepare_images(self, source_dir: str, output_dir: str, max_image_dim: int | None) -> str:
        """
        Creates downscaled working copies of images in output_dir/working_images/.
        Images are resized so that max(width, height) <= max_image_dim, preserving
        aspect ratio and EXIF metadata (critical for focal length calculation).

        If max_image_dim is None (Ultra preset), returns source_dir unchanged.
        If all images already fit within max_image_dim, returns source_dir unchanged.
        """
        from PIL import Image

        if max_image_dim is None:
            self.log_message.emit("[PREP] Ultra preset: using original full-resolution images.")
            return source_dir

        working_dir = os.path.join(output_dir, "colmap", "images")
        os.makedirs(working_dir, exist_ok=True)

        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        try:
            all_files = sorted(
                f for f in os.listdir(source_dir)
                if f.lower().endswith(image_extensions)
            )
        except Exception as e:
            self.log_message.emit(f"[WARNING] Could not list source images: {e}. Using originals.")
            return source_dir

        if not all_files:
            self.log_message.emit("[WARNING] No images found in source dir during preparation. Using originals.")
            return source_dir

        resized_count = 0
        copied_count = 0
        self.log_message.emit(
            f"[PREP] Preparing {len(all_files)} images for {self.quality_preset.upper()} preset "
            f"(max dimension: {max_image_dim}px)..."
        )

        for filename in all_files:
            if not self.is_running:
                return source_dir

            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(working_dir, filename)

            try:
                with Image.open(src_path) as img:
                    w, h = img.size
                    # Preserve EXIF so focal length extraction keeps working
                    try:
                        exif_bytes = img.info.get('exif', b'')
                    except Exception:
                        exif_bytes = b''

                    if max(w, h) > max_image_dim:
                        # Compute scale factor maintaining aspect ratio
                        scale = max_image_dim / max(w, h)
                        new_w = round(w * scale)
                        new_h = round(h * scale)
                        resized = img.resize((new_w, new_h), Image.LANCZOS)
                        # Save with original EXIF preserved
                        save_kwargs = {}
                        if exif_bytes:
                            save_kwargs['exif'] = exif_bytes
                        # JPEG quality 92 — visually lossless, keeps file size reasonable
                        if filename.lower().endswith(('.jpg', '.jpeg')):
                            save_kwargs['quality'] = 92
                            save_kwargs['subsampling'] = 0
                        resized.save(dst_path, **save_kwargs)
                        resized_count += 1
                    else:
                        # Image already fits — copy it as-is to keep the dir consistent
                        import shutil as _shutil
                        _shutil.copy2(src_path, dst_path)
                        copied_count += 1

            except Exception as e:
                self.log_message.emit(f"[WARNING] Failed to process image {filename}: {e}. Copying original.")
                try:
                    import shutil as _shutil
                    _shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e2:
                    self.log_message.emit(f"[ERROR] Could not copy {filename}: {e2}")

        self.log_message.emit(
            f"[PREP] Done: {resized_count} image(s) downscaled to ≤{max_image_dim}px, "
            f"{copied_count} image(s) were already within limit."
        )
        return working_dir

    def _run_with_gpu_fallback(self, cmd_gpu: list, cmd_cpu: list, timeout: float, cwd=None, env=None, line_parser=None) -> bool:
        """Try GPU first, fall back to CPU if GPU fails."""
        self.log_message.emit("[INFO] Attempting GPU-accelerated execution (OpenGL)...")
        self._using_gpu_sift = True
        if self._run_process_realtime(cmd_gpu, timeout=timeout, cwd=cwd, env=env, line_parser=line_parser):
            return True
        self.log_message.emit("[WARNING] GPU execution failed. Falling back to CPU-only mode...")
        self._using_gpu_sift = False
        return self._run_process_realtime(cmd_cpu, timeout=timeout, cwd=cwd, env=env, line_parser=line_parser)

    def _select_best_sparse_model(self, sparse_dir: str) -> str:
        """Find the sparse model subdirectory with the most registered images."""
        best_dir = None
        best_count = 0
        if not os.path.exists(sparse_dir):
            return None
        for subdir in os.listdir(sparse_dir):
            model_dir = os.path.join(sparse_dir, subdir)
            images_bin = os.path.join(model_dir, "images.bin")
            if os.path.exists(images_bin):
                # File size is a rough proxy for number of images
                size = os.path.getsize(images_bin)
                if size > best_count:
                    best_count = size
                    best_dir = model_dir
        return best_dir

    def _run_model_analyzer(self, model_dir: str) -> dict:
        """Runs COLMAP model_analyzer and returns parsed stats."""
        colmap_exe = os.path.join(get_base_dir(), self.toolchain_map["colmap"]["colmap"])
        cmd = [colmap_exe, "model_analyzer", "--path", model_dir]
        colmap_env = self._get_colmap_env()
        
        from hardware_profiler import run_safe_subprocess
        try:
            ret, stdout, stderr = run_safe_subprocess(cmd, timeout=30.0, env=colmap_env)
            if ret == 0:
                output = stdout + "\n" + stderr
                self.log_message.emit("[INFO] Model Analyzer Output:\n" + output)
                stats = {}
                import re
                for line in output.splitlines():
                    if "Images:" in line:
                        match = re.search(r"Images:\s*(\d+)", line)
                        if match:
                            stats["images"] = int(match.group(1))
                    elif "Points:" in line:
                        match = re.search(r"Points:\s*(\d+)", line)
                        if match:
                            stats["points"] = int(match.group(1))
                    elif "Mean reprojection error:" in line:
                        match = re.search(r"Mean reprojection error:\s*([\d.]+)px", line)
                        if match:
                            stats["mean_error"] = float(match.group(1))
                return stats
        except Exception as e:
            self.log_message.emit(f"[WARNING] Model analyzer failed: {e}")
        return {}

    def _parse_sfm_poses(self) -> int:
        """Returns the number of registered camera poses from the last run stats."""
        return self._last_reconstruction_stats.get("images", 0)

    def _count_scene_points(self, mvs_dir: str) -> int:
        """Returns the number of reconstructed points from the last run stats."""
        return self._last_reconstruction_stats.get("points", 9999)

    def _count_calibrated_images(self, mvs_dir: str) -> int:
        """Returns the number of calibrated images from the last run stats."""
        return self._last_reconstruction_stats.get("images", 999)

    def _parse_feature_extraction_line(self, line: str) -> str or None:
        """Parse COLMAP feature_extractor output into Metashape-style format."""
        import re
        
        # Match: "Processed file [N/M]" pattern
        match = re.search(r'Processed file \[(\d+)/(\d+)\]', line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            
            # Extract feature count
            feat_match = re.search(r'num_features=(\d+)', line) 
            if feat_match:
                num_features = int(feat_match.group(1))
                # Track for summary
                self._feature_counts.append(num_features)
                
                # Extract image name
                name_match = re.search(r'name=(\S+)', line)
                img_name = name_match.group(1) if name_match else f"image {current}"
                
                gpu_label = "[iGPU]" if self._using_gpu_sift else "[CPU]"
                return f"{gpu_label} {img_name}: {num_features:,} features ({current}/{total})"
        
        # Match: timing information
        if "Elapsed time:" in line:
            return f"  ⏱ {line}"
        
        return None  # Use raw line

    def _parse_matching_line(self, line: str) -> str or None:
        """Parse COLMAP exhaustive_matcher output."""
        import re
        
        match = re.search(r'Matching block \[(\d+)/(\d+)', line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            return f"Processing match block {current}/{total}..."
        
        # Match pair results
        match = re.search(r'(\d+) matches for image pair', line)
        if match:
            num_matches = int(match.group(1))
            self._match_counts.append(num_matches)
            return None  # Accumulate silently, report in summary
        
        # Geometric verification results
        if "geometrically verified" in line.lower():
            return f"  ✓ {line}"
        
        return None

    def _parse_mapper_line(self, line: str) -> str or None:
        """Parse COLMAP mapper output into Metashape-style camera registration log."""
        import re
        
        # Image registration
        match = re.search(r'Registering image #(\d+) \((\d+)\)', line)
        if match:
            image_id = match.group(1)
            total_registered = match.group(2)
            self._registered_count = int(total_registered)
            return f"Adding camera {image_id} ({total_registered} of {self._total_images})"
        
        # Inlier count for registered image
        match = re.search(r'Image has (\d+)\s*/\s*(\d+) inliers', line)
        if match:
            used = int(match.group(1))
            total = int(match.group(2))
            return f"  → {used} of {total} feature matches used"
        
        # Bundle adjustment iteration
        if "Bundle adjustment" in line:
            return f"  Adjusting..."
        
        # Track statistics
        match = re.search(r'Merged observations: (\d+)', line)
        if match:
            return f"  → Merged {match.group(1)} track observations"
        
        match = re.search(r'Completed observations: (\d+)', line)
        if match:
            return f"  → Completed {match.group(1)} observations"
        
        # Triangulation
        match = re.search(r'Triangulated (\d+) points', line)
        if match:
            points = int(match.group(1))
            self._triangulated_points += points
            return f"  → Triangulated {points:,} new 3D points"
        
        # Filtered observations
        match = re.search(r'Filtered observations: (\d+)', line)
        if match:
            filtered = int(match.group(1))
            return f"  → Filtered {filtered} outlier observations"
        
        # Mean reprojection error
        match = re.search(r'Mean reprojection error: ([\d.]+)px', line)
        if match:
            error = float(match.group(1))
            self._mean_reproj_error = error
            status = "✓ good" if error < 1.0 else "⚠ high" if error < 2.0 else "✗ poor"
            return f"  → Mean reprojection error: {error:.3f}px ({status})"
        
        return None

    def _parse_densify_line(self, line: str) -> str or None:
        """Parse OpenMVS DensifyPointCloud output."""
        import re
        
        # Depth map estimation: "Depth-map for image  31 estimated using  4 images: 768x1024"
        match = re.search(r'Depth-map for image\s+(\d+) estimated using\s+(\d+) images:\s*(\d+)x(\d+)', line)
        if match:
            img_id = match.group(1)
            num_views = match.group(2)
            w = match.group(3)
            h = match.group(4)
            self._depth_map_count += 1
            return f"[CPU] Estimating depth map for image {img_id} ({w}×{h}, {num_views} views)"
        
        # Depth-map fusion: "Depth-maps dense fused and filtered: 20 depth-maps, ... 221235 points"
        match = re.search(r'Depth-maps dense fused.*?(\d+) depth-maps.*?(\d+) points', line)
        if match:
            dm_count = match.group(1)
            pt_count = int(match.group(2))
            self._dense_point_count = pt_count
            return f"  → Fused {dm_count} depth maps → {pt_count:,} candidate points"
        
        # Final point count: "Densifying point-cloud completed: 213311 points"
        match = re.search(r'[Dd]ensif(?:y|ying).*?completed.*?(\d+) points', line)
        if match:
            count = int(match.group(1))
            self._dense_point_count = count
            return f"  → Dense cloud: {count:,} points"
        
        # Point-cloud trimmed to ROI
        match = re.search(r'Point-cloud trimmed.*?(\d+) points removed', line)
        if match:
            removed = int(match.group(1))
            return f"  → Trimmed {removed:,} points outside ROI"
        
        return None

    def _parse_mesh_line(self, line: str) -> str or None:
        """Parse OpenMVS ReconstructMesh output."""
        import re
        
        # Vertex/face counts
        match = re.search(r'(\d+) vertices, (\d+) faces', line)
        if match:
            verts = int(match.group(1))
            faces = int(match.group(2))
            self._mesh_vertices = verts
            self._mesh_faces = faces
            return f"  → Mesh: {verts:,} vertices, {faces:,} faces"
        
        # Cleaning statistics  
        match = re.search(r'[Rr]emoved? (\d+)', line)
        if match:
            count = int(match.group(1))
            if 'spurious' in line.lower():
                self._spurious_removed = count
                return f"  → Cleaned {count} spurious components"
            elif 'spike' in line.lower():
                self._spikes_removed = count
                return f"  → Cleaned {count} spikes"
            else:
                return f"  → Cleaned {count} artifacts"
                
        match = re.search(r'[Cc]losed? (\d+) holes', line)
        if match:
            count = int(match.group(1))
            self._holes_closed = count
            return f"  → Closed {count} holes"
        
        return None

    def _query_colmap_database_stats(self, db_path: str) -> dict:
        """Query COLMAP's SQLite database for feature and match statistics."""
        import sqlite3
        stats = {
            "num_images": 0,
            "feature_counts": [],
            "num_pairs": 0,
            "match_counts": [],
        }
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Count images
            cursor.execute("SELECT COUNT(*) FROM images")
            stats["num_images"] = cursor.fetchone()[0]
            
            # Feature counts per image
            cursor.execute("SELECT image_id, rows FROM keypoints")
            for row in cursor.fetchall():
                stats["feature_counts"].append(row[1])
            
            # Match counts per pair (from two_view_geometries, which has verified matches)
            cursor.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0")
            for row in cursor.fetchall():
                stats["num_pairs"] += 1
                stats["match_counts"].append(row[1])
            
            conn.close()
        except Exception as e:
            self.log_message.emit(f"[WARNING] Could not read COLMAP database: {e}")
        
        return stats

    def _emit_feature_summary(self):
        if not self._feature_counts:
            return
        
        total = len(self._feature_counts)
        avg = sum(self._feature_counts) / total
        min_f = min(self._feature_counts)
        max_f = max(self._feature_counts)
        
        self.log_message.emit(
            f"\n{'='*60}\n"
            f"  FEATURE EXTRACTION SUMMARY\n"
            f"{'='*60}\n"
            f"  Images processed:     {total}\n"
            f"  Features per image:   {avg:,.0f} avg  |  {min_f:,} min  |  {max_f:,} max\n"
            f"  Total features:       {sum(self._feature_counts):,}\n"
            f"  Compute device:       {'iGPU (OpenGL)' if self._using_gpu_sift else 'CPU (VLFeat)'}\n"
            f"{'='*60}"
        )
        
        # DIAGNOSTIC: Warn if features are too low
        if avg < 3000:
            self.log_message.emit(
                "[⚠ DIAGNOSTIC] Average features per image is LOW (<3000). "
                "This may cause poor camera registration. "
                "Consider: higher quality preset, better image overlap, or sharper images."
            )

    def _emit_matching_summary(self):
        self.log_message.emit(
            f"\n{'='*60}\n"
            f"  FEATURE MATCHING SUMMARY\n"
            f"{'='*60}\n"
            f"  Image pairs tested:   {self._pairs_tested}\n"
            f"  Pairs with matches:   {self._pairs_matched}\n"
            f"  Match success rate:   {(self._pairs_matched/max(self._pairs_tested,1))*100:.1f}%\n"
            f"  Avg matches/pair:     {sum(self._match_counts)/max(len(self._match_counts),1):,.0f}\n"
            f"{'='*60}"
        )
        
        if self._pairs_matched < self._pairs_tested * 0.3:
            self.log_message.emit(
                "[⚠ DIAGNOSTIC] Less than 30% of image pairs have matches. "
                "Images may have insufficient overlap or very different viewpoints."
            )

    def _emit_sfm_summary(self):
        pct = (self._registered_count / max(self._total_images, 1)) * 100
        
        self.log_message.emit(
            f"\n{'='*60}\n"
            f"  STRUCTURE FROM MOTION SUMMARY\n"  
            f"{'='*60}\n"
            f"  Total images:         {self._total_images}\n"
            f"  Cameras registered:   {self._registered_count} ({pct:.0f}%)\n"
            f"  Cameras FAILED:       {self._total_images - self._registered_count}\n"
            f"  3D points:            {self._triangulated_points:,}\n"
            f"  Mean reproj. error:   {self._mean_reproj_error:.3f}px\n"
            f"{'='*60}"
        )
        
        if pct < 50:
            self.log_message.emit(
                "[✗ DIAGNOSTIC] CRITICAL: Less than 50% of cameras registered! "
                "Reconstruction will be incomplete. Check image overlap and quality."
            )
        elif pct < 75:
            self.log_message.emit(
                "[⚠ DIAGNOSTIC] Only {pct:.0f}% cameras registered. "
                "Some areas may have gaps. Consider adding more images in weak areas."
            )
        else:
            self.log_message.emit(
                f"[✓ DIAGNOSTIC] Good camera registration ({pct:.0f}%). "
                "Proceeding with dense reconstruction."
            )

    def _emit_dense_summary(self):
        self.log_message.emit(
            f"\n{'='*60}\n"
            f"  DENSE POINT CLOUD SUMMARY\n"
            f"{'='*60}\n"
            f"  Depth maps computed:  {self._depth_map_count}\n"
            f"  Dense points:         {self._dense_point_count:,}\n"
            f"  Points/camera:        {self._dense_point_count // max(self._registered_count, 1):,}\n"
            f"{'='*60}"
        )

    def _emit_mesh_summary(self):
        self.log_message.emit(
            f"\n{'='*60}\n"
            f"  MESH RECONSTRUCTION SUMMARY\n"
            f"{'='*60}\n"
            f"  Vertices:             {self._mesh_vertices:,}\n"
            f"  Faces:                {self._mesh_faces:,}\n"
            f"  Spurious removed:     {self._spurious_removed}\n"
            f"  Spikes removed:       {self._spikes_removed}\n"
            f"  Holes closed:         {self._holes_closed}\n"
            f"{'='*60}"
        )
