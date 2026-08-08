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
import gc
import psutil
from PySide6.QtCore import QThread, Signal
from hardware_profiler import run_safe_subprocess, get_memory_budget, get_recommended_matching_mode



def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def is_valid_faiss_vocab_tree(file_path: str) -> bool:
    """
    Check if a vocabulary tree binary file is in the FAISS format (file_version == 1)
    required by COLMAP 3.10+ (May 2025+).
    Legacy FLANN index files start with uint32 file_version == 32762 (0x7ffa).
    """
    if not file_path or not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if len(header) < 4:
                return False
            import struct
            version = struct.unpack('<I', header)[0]
            return version == 1
    except Exception:
        return False


def get_default_vocab_tree_path() -> str | None:
    base_dir = get_base_dir()
    colmap_dir = os.path.join(base_dir, "backend_bin", "colmap")
    
    if os.path.exists(colmap_dir):
        for f in os.listdir(colmap_dir):
            if f.endswith(".bin"):
                cand = os.path.join(colmap_dir, f)
                if is_valid_faiss_vocab_tree(cand):
                    return cand
    return None


class PipelineWorker(QThread):
    """
    Worker thread that executes the photogrammetry toolchain step-by-step.
    Emits progress and logging signals to keep the UI responsive.
    """
    progress_changed = Signal(int)
    status_changed = Signal(str)
    log_message = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, image_dir: str, output_dir: str, quality_preset: str = "medium", gpu_mode: str = "auto", has_plain_surfaces: bool = False, mapper_mode: str = "incremental", ref_cloud_path: str = None, mesh_mode: str = "default", poisson_depth: int = 9, custom_params: dict = None, resume_from_step: str = None, parent=None):
        super().__init__(parent)
        self.image_dir = os.path.abspath(image_dir) if image_dir else image_dir
        self.output_dir = os.path.abspath(output_dir) if output_dir else output_dir
        self.quality_preset = quality_preset
        self.gpu_mode = gpu_mode
        self.has_plain_surfaces = has_plain_surfaces
        # When plain/smooth surfaces is selected, use GLOMAP global mapper instead of incremental mapper
        if self.has_plain_surfaces:
            self.mapper_mode = "global"
        else:
            self.mapper_mode = mapper_mode
        self.ref_cloud_path = ref_cloud_path
        self.mesh_mode = mesh_mode
        self.poisson_depth = poisson_depth
        self.custom_params = custom_params
        self.resume_from_step = resume_from_step
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

    def _get_throttled_sift_limits(self, total_images: int) -> tuple:
        """
        Dynamically throttles SIFT Max Features and Max Matches based on:
        1. Quality preset baseline (preview, medium, high, ultra).
        2. Dataset size (number of images).
        3. Dynamic RAM & Swap memory consumption (throttling down once >= 95% swap is consumed to prevent OOM shutdown).
        """
        # 1. Preset baseline limits
        if self.quality_preset == "preview":
            base_features = 4096
            base_matches = 16384
        elif self.quality_preset == "medium":
            base_features = 8192
            base_matches = 16384
        elif self.quality_preset == "high":
            base_features = 12288
            base_matches = 32768
        else:  # ultra
            base_features = 16384
            base_matches = 65536

        # 2. Dataset size throttling factor
        if total_images > 500:
            dataset_factor = 0.5
        elif total_images > 250:
            dataset_factor = 0.65
        elif total_images > 100:
            dataset_factor = 0.8
        else:
            dataset_factor = 1.0

        max_features = int(base_features * dataset_factor)
        max_matches = int(base_matches * dataset_factor)

        # 3. Dynamic Swap and Memory Throttling (Keyword: throttle)
        mem_budget = get_memory_budget()
        sw = psutil.swap_memory()
        swap_pct = sw.percent if sw.total > 0 else 0.0

        if swap_pct >= 95.0 or mem_budget.available_gb < 1.5:
            self.log_message.emit(
                f"[THROTTLE] High swap/RAM consumption detected (Swap: {swap_pct:.1f}%, Available RAM: {mem_budget.available_gb:.2f} GB). "
                f"Throttling down Max Matches and SIFT Feature limits to prevent OOM shutdown."
            )
            max_features = min(max_features, 3072)
            max_matches = min(max_matches, 4096)
        elif swap_pct >= 80.0 or mem_budget.pressure_level != "ok":
            self.log_message.emit(
                f"[THROTTLE] System memory pressure detected (Swap: {swap_pct:.1f}%, Available RAM: {mem_budget.available_gb:.2f} GB). "
                f"Throttling feature and match thresholds."
            )
            max_features = min(max_features, 6144)
            max_matches = min(max_matches, 8192)

        if dataset_factor < 1.0:
            self.log_message.emit(
                f"[THROTTLE] Dataset size of {total_images} images detected. "
                f"Throttling limits to SIFT Max Features: {max_features}, Max Matches: {max_matches}."
            )

        # Custom parameter overrides take priority if provided
        if self.custom_params:
            if "colmap_max_num_features" in self.custom_params:
                max_features = self.custom_params["colmap_max_num_features"]
            if "colmap_max_num_matches" in self.custom_params:
                max_matches = self.custom_params["colmap_max_num_matches"]

        return max_features, max_matches
        self._dense_point_count = 0        # Points in dense cloud
        self._mesh_vertices = 0            # Mesh vertex count
        self._mesh_faces = 0               # Mesh face count
        self._spurious_removed = 0
        self._spikes_removed = 0
        self._spikes_removed = 0
        self._holes_closed = 0
        self._image_names_map = {}

    def _get_image_names_from_db(self, db_path: str) -> dict:
        """Retrieves image IDs to names mapping from COLMAP database."""
        import sqlite3
        image_map = {}
        abs_db_path = os.path.abspath(db_path)
        if not os.path.exists(abs_db_path):
            return image_map
        for attempt in range(3):
            try:
                conn = sqlite3.connect(abs_db_path, timeout=10.0)
                cursor = conn.cursor()
                cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
                for row in cursor.fetchall():
                    image_map[row[0]] = row[1]
                conn.close()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.3)
                else:
                    self.log_message.emit(f"[WARNING] Could not read COLMAP database for image names: {e}")
        return image_map

    def _to_colmap_path(self, p: str) -> str:
        """Converts Windows backslashes to forward slashes to prevent COLMAP parsing errors."""
        return p.replace('\\', '/')

    def _load_toolchain_map(self) -> dict:
        """Loads the toolchain mapping config file."""
        map_path = os.path.join(get_base_dir(), "toolchain_map.json")
        if os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    return self._normalize_toolchain_map(json.load(f))
            except Exception as e:
                self.log_message.emit(f"Error reading toolchain_map.json: {e}")
        return {}

    def _normalize_toolchain_map(self, toolchain_map: dict) -> dict:
        """Resolve binary paths appropriately for host platform (Windows vs Linux/macOS)."""
        import shutil
        if sys.platform == "win32":
            return toolchain_map

        normalized = {}
        for group, binaries in toolchain_map.items():
            if not isinstance(binaries, dict):
                normalized[group] = binaries
                continue

            normalized[group] = {}
            for name, rel_path in binaries.items():
                clean_rel_path = rel_path[:-4] if rel_path.lower().endswith(".exe") else rel_path
                clean_abs_path = os.path.join(get_base_dir(), clean_rel_path)

                if os.path.exists(clean_abs_path):
                    normalized[group][name] = clean_abs_path
                elif shutil.which(name):
                    normalized[group][name] = shutil.which(name)
                elif shutil.which(os.path.basename(clean_rel_path)):
                    normalized[group][name] = shutil.which(os.path.basename(clean_rel_path))
                else:
                    normalized[group][name] = clean_abs_path
        return normalized

    def _backup_checkpoint(self, step_name: str):
        try:
            from main_window import get_backup_dir, save_session_metadata, load_session_metadata
            backup_dir = get_backup_dir()

            mvs_out = os.path.join(self.output_dir, "mvs")
            colmap_out = os.path.join(self.output_dir, "colmap")
            backup_mvs = os.path.join(backup_dir, "mvs")
            backup_colmap = os.path.join(backup_dir, "colmap")

            os.makedirs(backup_mvs, exist_ok=True)
            os.makedirs(backup_colmap, exist_ok=True)

            if os.path.exists(mvs_out):
                for item in os.listdir(mvs_out):
                    s_path = os.path.join(mvs_out, item)
                    d_path = os.path.join(backup_mvs, item)
                    if os.path.isfile(s_path):
                        shutil.copy2(s_path, d_path)
                    elif os.path.isdir(s_path):
                        if os.path.exists(d_path):
                            shutil.rmtree(d_path)
                        shutil.copytree(s_path, d_path)

            if os.path.exists(colmap_out):
                for item in os.listdir(colmap_out):
                    s_path = os.path.join(colmap_out, item)
                    d_path = os.path.join(backup_colmap, item)
                    if os.path.isfile(s_path):
                        shutil.copy2(s_path, d_path)
                    elif os.path.isdir(s_path):
                        if os.path.exists(d_path):
                            shutil.rmtree(d_path)
                        shutil.copytree(s_path, d_path)

            existing_meta = load_session_metadata() or {}
            existing_meta["scan_type"] = "photogrammetry"
            existing_meta["last_completed_step"] = step_name
            img_cnt = getattr(self, '_total_images', 0) or len(getattr(self, '_image_names_map', {}))
            if img_cnt > 0:
                existing_meta["image_count"] = img_cnt
            elif "image_count" not in existing_meta:
                existing_meta["image_count"] = 0
            existing_meta["quality_preset"] = self.quality_preset
            existing_meta["gpu_mode"] = self.gpu_mode
            existing_meta["has_plain_surfaces"] = self.has_plain_surfaces
            existing_meta["mapper_mode"] = self.mapper_mode
            existing_meta["mesh_mode"] = self.mesh_mode
            existing_meta["poisson_depth"] = self.poisson_depth
            if self.custom_params:
                existing_meta["custom_params"] = self.custom_params
            save_session_metadata(existing_meta)
            self.log_message.emit(f"[BACKUP] Saved checkpoint for '{step_name}'.")
        except Exception as e:
            self.log_message.emit(f"[WARNING] Failed to write backup checkpoint: {e}")

    def run(self):
        try:
            self.status_changed.emit("Initializing Pipeline...")
            self.log_message.emit(f"[INFO] Plain/Smooth Surfaces optimization: {'Enabled' if self.has_plain_surfaces else 'Disabled'}")
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
        """Checks if all required binaries in toolchain_map.json exist locally or in system PATH."""
        import shutil
        if not self.toolchain_map:
            return False

        for group, binaries in self.toolchain_map.items():
            if not isinstance(binaries, dict):
                continue
            for name, binary_path in binaries.items():
                if not os.path.exists(binary_path) and not shutil.which(binary_path) and not shutil.which(name):
                    self.log_message.emit(
                        f"Missing toolchain binary for {sys.platform}: '{name}' ({binary_path}). "
                        "Please place Linux binary in backend_bin or install system package."
                    )
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
        if os.path.isdir(colmap_plugins):
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

        cmd = self._adapt_colmap_cmd(cmd)
        self.log_message.emit(f"[RUN] {' '.join(cmd)}")

        # Identify log file if it's an OpenMVS command
        exe_name = os.path.splitext(os.path.basename(cmd[0]))[0]
        # OpenMVS tools: InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, RefineMesh, TextureMesh
        is_openmvs = exe_name in ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"]

        # Auto-grant execution permissions on Linux/macOS for local binary files
        if sys.platform != 'win32' and os.path.isfile(cmd[0]):
            try:
                st = os.stat(cmd[0])
                if not (st.st_mode & 0o111):
                    os.chmod(cmd[0], st.st_mode | 0o111)
            except Exception as e:
                self.log_message.emit(f"[WARNING] Could not set +x permissions on {cmd[0]}: {e}")

        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_NO_WINDOW

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
            last_mem_check = time.time()

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
                last_log_time = time.time()
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

                    # Memory watchdog check
                    if time.time() - last_mem_check > 4.0:
                        last_mem_check = time.time()
                        vm = psutil.virtual_memory()
                        if vm.percent > 96.0 or vm.available < (500 * 1024 * 1024):
                            proc.terminate()
                            try:
                                proc.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            log_handle.close()
                            self.log_message.emit(
                                f"[CRITICAL OOM GUARD] System RAM usage reached {vm.percent:.1f}% "
                                f"({vm.available / (1024**2):.0f} MB available). "
                                f"Terminated {exe_name} to prevent system lockup/crash."
                            )
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
                        # Emit a heartbeat every 30s of silence so the UI doesn't look frozen
                        silent_secs = time.time() - last_log_time
                        if silent_secs >= 30.0:
                            elapsed = int(time.time() - start_time)
                            self.log_message.emit(
                                f"[RUNNING] {exe_name} still processing... ({elapsed}s elapsed, pipeline is active)"
                            )
                            last_log_time = time.time()
                        time.sleep(0.05)
                        continue

                    clean_line = line.strip()
                    if clean_line:
                        last_log_time = time.time()  # reset silence timer on real output
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

                    # Memory watchdog check
                    if time.time() - last_mem_check > 4.0:
                        last_mem_check = time.time()
                        vm = psutil.virtual_memory()
                        if vm.percent > 96.0 or vm.available < (500 * 1024 * 1024):
                            proc.terminate()
                            try:
                                proc.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            self.log_message.emit(
                                f"[CRITICAL OOM GUARD] System RAM usage reached {vm.percent:.1f}% "
                                f"({vm.available / (1024**2):.0f} MB available). "
                                f"Terminated {exe_name} to prevent system lockup/crash."
                            )
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
            if sys.platform != 'win32' and cmd[0].endswith('.exe'):
                self.log_message.emit(
                    "[DIAGNOSTIC] Attempted to run a Windows executable (.exe) on Linux.\n"
                    "  Please ensure native Linux binaries (or system package e.g. 'sudo apt install colmap')\n"
                    "  are available."
                )
            return False

    def _run_real_pipeline(self) -> bool:
        """
        Executes the full 9-step COLMAP + OpenMVS photogrammetry pipeline.
        Each step is heavily parameterized for maximum reconstruction quality.
        """
        base_dir = get_base_dir()
        
        resume_requested = bool(self.resume_from_step)
        colmap_dir_check = os.path.join(self.output_dir, "colmap")
        check_db_path = os.path.join(colmap_dir_check, "database.db")
        is_checkpoint_valid = self._is_valid_checkpoint(check_db_path)

        # Clean up stale reconstruction subdirectories to prevent legacy files impacting new scans
        for subdir in ["colmap", "mvs"]:
            if subdir == "colmap" and resume_requested and is_checkpoint_valid:
                self.log_message.emit(f"[RESUME] Preserving valid database checkpoint at: {check_db_path}")
                continue
            sub_path = os.path.join(self.output_dir, subdir)
            if os.path.exists(sub_path):
                self.log_message.emit(f"[INFO] Cleaning up stale reconstruction directory: {sub_path}")
                try:
                    shutil.rmtree(sub_path)
                except Exception as e:
                    self.log_message.emit(f"[WARNING] Failed to clean output folder: {e}")

        colmap_out = self._to_colmap_path(os.path.abspath(os.path.join(self.output_dir, "colmap")))
        mvs_out = self._to_colmap_path(os.path.abspath(os.path.join(self.output_dir, "mvs")))
        os.makedirs(colmap_out, exist_ok=True)
        os.makedirs(mvs_out, exist_ok=True)
        os.makedirs(os.path.join(colmap_out, "sparse"), exist_ok=True)

        mem_budget = get_memory_budget()
        self.log_message.emit(
            f"[MEMORY] Dynamic Available RAM: {mem_budget.available_gb:.2f} GB | "
            f"Swap Used: {mem_budget.swap_used_gb:.2f}/{mem_budget.swap_total_gb:.2f} GB | "
            f"Pressure Level: {mem_budget.pressure_level.upper()}"
        )

        if mem_budget.available_gb < 1.5 and mem_budget.swap_used_gb > (mem_budget.swap_total_gb * 0.85):
            self.log_message.emit(
                "[CRITICAL] Available system memory is below 1.5 GB and swap memory is near exhaustion. "
                "Aborting reconstruction to prevent system crash."
            )
            return False

        pressure_mode = (mem_budget.pressure_level != "ok")
        num_threads = mem_budget.safe_thread_count
        self.log_message.emit(
            f"[MEMORY] Budgeted parallel worker threads: {num_threads} (Host Logical CPUs: {os.cpu_count() or 4})"
        )

        # --- GPU / CUDA Environment (used mainly by OpenMVS) ---
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_threads)
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

        if self.has_plain_surfaces:
            self.mapper_mode = "global"
            self.log_message.emit("[INFO] Plain/Smooth Surfaces option selected: activating GLOMAP global mapper for camera pose estimation.")

        # -------------------------------------------------------------------------
        # QUALITY PRESET PARAMETERS
        # -------------------------------------------------------------------------
        if self.quality_preset == "preview":
            max_image_dim  = 1024
            colmap_max_image_size = 1024
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
            colmap_first_octave = -1
            guided_matching = "1" if self.has_plain_surfaces else "0"
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

        # Override presets with custom parameters if custom overrides checkbox was checked
        if self.custom_params:
            self.log_message.emit("[INFO] Custom parameter overrides enabled. Overriding quality preset configuration.")
            if "guided_matching" in self.custom_params:
                guided_matching = self.custom_params["guided_matching"]
            if "run_bundle_adjuster" in self.custom_params:
                run_bundle_adjuster = self.custom_params["run_bundle_adjuster"]
            if "densify_res" in self.custom_params:
                densify_res = self.custom_params["densify_res"]
            if "densify_views" in self.custom_params:
                densify_views = self.custom_params["densify_views"]
            if "refine_scales" in self.custom_params:
                refine_scales = self.custom_params["refine_scales"]
            if "texture_res" in self.custom_params:
                texture_res = self.custom_params["texture_res"]

        skip_sfm = self.resume_from_step in ["sparse_reconstruction", "dense_reconstruction"]
        skip_dense = self.resume_from_step == "dense_reconstruction"

        if skip_sfm:
            self.log_message.emit(f"[RESUME] Resuming session from checkpoint: '{self.resume_from_step}'. Skipping Steps 1-5 (SfM & Sparse Cloud already complete).")
            self.progress_changed.emit(70)
            working_image_dir = os.path.join(self.output_dir, "input_images")
            if not os.path.exists(working_image_dir) or not any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')) for f in (os.listdir(working_image_dir) if os.path.exists(working_image_dir) else [])):
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
        else:
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

        # Dynamic SIFT parameter throttling based on dataset size, quality preset, and dynamic swap/memory pressure
        colmap_max_num_features, colmap_max_num_matches = self._get_throttled_sift_limits(self._total_images)

        colmap_exe = os.path.join(base_dir, self.toolchain_map["colmap"]["colmap"])
        colmap_env = self._get_colmap_env()
        database_path = self._to_colmap_path(os.path.abspath(os.path.join(colmap_out, "database.db")))
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        working_image_dir = self._to_colmap_path(working_image_dir)
        if os.path.exists(database_path):
            if resume_requested and is_checkpoint_valid:
                self.log_message.emit("[RESUME] Preserving valid COLMAP database checkpoint for reconstruction.")
            else:
                try:
                    os.remove(database_path)
                    self.log_message.emit("[INFO] Cleared stale COLMAP database.")
                except Exception as e:
                    self.log_message.emit(f"[WARNING] Failed to clear database: {e}")

        # =========================================================================
        # STEP 2/9 — Feature Extraction
        # STEP 3/9 — Feature Matching
        # =========================================================================

        # Determine camera-model multiplicity
        single_camera_val = "1"
        try:
            from PIL import Image
            unique_sizes = set()
            image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
            for filename in os.listdir(working_image_dir):
                if filename.lower().endswith(image_extensions):
                    with Image.open(os.path.join(working_image_dir, filename)) as img:
                        unique_sizes.add(img.size)
            if len(unique_sizes) > 1:
                single_camera_val = "0"
                self.log_message.emit(
                    f"[INFO] Mixed image dimensions detected (unique sizes: {list(unique_sizes)}). "
                    "Using per-image camera model for reconstruction."
                )
            else:
                self.log_message.emit("[INFO] All images share identical dimensions. Using single camera model.")
        except Exception as e:
            self.log_message.emit(f"[WARNING] Could not check image dimensions: {e}. Defaulting to single camera model.")

        skip_features_and_matching = resume_requested and is_checkpoint_valid and self.resume_from_step in ["sparse_reconstruction", "dense_reconstruction", "features_matched"]
        skip_features_only = resume_requested and is_checkpoint_valid and self.resume_from_step == "features_extracted"

        if skip_features_and_matching:
            self.log_message.emit(
                "[RESUME] Valid checkpoint database detected! "
                "Skipping Step 2 (Feature Extraction) and Step 3 (Feature Matching)."
            )
            db_stats = self._query_colmap_database_stats(database_path)
            num_registered = db_stats["num_images"]
            self.log_message.emit(
                f"[RESUME] Loaded checkpoint database with {num_registered} registered images and {db_stats['num_pairs']} matched pairs."
            )
            self._image_names_map = self._get_image_names_from_db(database_path)
            self._pairs_tested = (num_registered * (num_registered - 1)) // 2 if num_registered > 1 else 0
            self._pairs_matched = db_stats["num_pairs"]
            if db_stats["match_counts"]:
                self._match_counts = db_stats["match_counts"]
            self._emit_matching_summary(database_path)
            self.progress_changed.emit(40)

        else:
            if skip_features_only:
                self.log_message.emit(
                    "[RESUME] SIFT features already extracted in database! "
                    "Skipping Step 2 (Feature Extraction) and proceeding directly to Step 3 (Feature Matching)."
                )
                db_stats = self._query_colmap_database_stats(database_path)
                if db_stats["feature_counts"]:
                    self._feature_counts = db_stats["feature_counts"]
                self._emit_feature_summary()
                self.progress_changed.emit(25)
            else:
                # -----------------------------------------------------------------
                # SIFT path: existing COLMAP feature_extractor + exhaustive_matcher
                # -----------------------------------------------------------------
                self.status_changed.emit("Step 2/9: Extracting SIFT Features...")

                cmd_extract_gpu = [
                    colmap_exe, "feature_extractor",
                    "--database_path", database_path,
                    "--image_path", working_image_dir,
                    "--ImageReader.camera_model", "PINHOLE",
                    "--ImageReader.single_camera", single_camera_val,
                    "--SiftExtraction.use_gpu", "1",
                    "--SiftExtraction.max_image_size", str(colmap_max_image_size),
                    "--SiftExtraction.num_threads", str(num_threads),
                    "--SiftExtraction.max_num_features", str(colmap_max_num_features),
                    "--SiftExtraction.first_octave", str(colmap_first_octave),
                ]
                cmd_extract_cpu = [
                    colmap_exe, "feature_extractor",
                    "--database_path", database_path,
                    "--image_path", working_image_dir,
                    "--ImageReader.camera_model", "PINHOLE",
                    "--ImageReader.single_camera", single_camera_val,
                    "--SiftExtraction.use_gpu", "0",
                    "--SiftExtraction.max_image_size", str(colmap_max_image_size),
                    "--SiftExtraction.num_threads", str(num_threads),
                    "--SiftExtraction.max_num_features", str(colmap_max_num_features),
                    "--SiftExtraction.first_octave", str(colmap_first_octave),
                ]
                if self.has_plain_surfaces:
                    # Preview quality: lower SIFT thresholds as a best-effort fallback
                    additional_flags = [
                        "--SiftExtraction.peak_threshold", "0.002",
                        "--SiftExtraction.edge_threshold", "15"
                    ]
                    cmd_extract_gpu.extend(additional_flags)
                    cmd_extract_cpu.extend(additional_flags)
                self._feature_counts = []
                if not self._run_with_gpu_fallback(
                    cmd_extract_gpu, cmd_extract_cpu, timeout=14400.0, env=colmap_env,
                    line_parser=self._parse_feature_extraction_line
                ):
                    return False

                db_stats = self._query_colmap_database_stats(database_path)
                if db_stats["feature_counts"]:
                    self._feature_counts = db_stats["feature_counts"]
                self._emit_feature_summary()

                num_registered = db_stats["num_images"]
                if num_registered < 2:
                    if len(self._feature_counts) >= 2:
                        num_registered = len(self._feature_counts)
                    elif self._total_images >= 2:
                        num_registered = self._total_images
                if num_registered < 2:
                    self.log_message.emit(
                        f"[ERROR] Only {num_registered} image(s) successfully registered in the database. "
                        "Reconstruction requires at least 2 registered images. Aborting."
                    )
                    return False
                elif num_registered < self._total_images:
                    self.log_message.emit(
                        f"[WARNING] Only {num_registered} out of {self._total_images} images successfully registered. "
                        "Some images may be skipped or corrupt."
                    )

                self.progress_changed.emit(25)
                self._backup_checkpoint("features_extracted")

            # Step 3: SIFT matching
            self.status_changed.emit("Step 3/9: Matching SIFT Features...")
            os.makedirs(os.path.dirname(database_path), exist_ok=True)
            if not os.path.exists(database_path):
                self.log_message.emit(f"[WARNING] COLMAP database missing prior to matching: {database_path}. Initializing database schema...")
                self._create_colmap_db_schema(database_path)

            self._image_names_map = self._get_image_names_from_db(database_path)

            curr_avail_gb = get_memory_budget().available_gb
            matching_mode = get_recommended_matching_mode(self._total_images, curr_avail_gb)

            # Override matching_mode if user explicitly chose a matcher type in custom_params
            if self.custom_params and "colmap_matcher_type" in self.custom_params:
                user_matcher = self.custom_params["colmap_matcher_type"]
                if user_matcher in ["exhaustive", "sequential", "vocab_tree", "spatial"]:
                    matching_mode = user_matcher

            if matching_mode == "sequential":
                matcher_cmd = "sequential_matcher"
                extra_args = ["--SequentialMatching.overlap", "15", "--SequentialMatching.loop_detection", "0"]
                self.log_message.emit(
                    f"[INFO] Using sequential_matcher for {self._total_images} images."
                )

            elif matching_mode == "vocab_tree":
                vocab_path = self.custom_params.get("vocab_tree_path", "") if self.custom_params else ""
                if not vocab_path or not os.path.exists(vocab_path):
                    vocab_path = get_default_vocab_tree_path() or ""

                if vocab_path and os.path.exists(vocab_path):
                    if is_valid_faiss_vocab_tree(vocab_path):
                        matcher_cmd = "vocab_tree_matcher"
                        extra_args = ["--VocabTreeMatching.vocab_tree_path", vocab_path]
                        self.log_message.emit(f"[INFO] Using vocab_tree_matcher with FAISS vocabulary tree: {vocab_path}")
                    else:
                        self.log_message.emit(
                            f"[WARNING] Vocabulary tree file '{os.path.basename(vocab_path)}' is in legacy FLANN index format (COLMAP 3.10+ requires FAISS index). "
                            "Automatically falling back to exhaustive_matcher."
                        )
                        matcher_cmd = "exhaustive_matcher"
                        extra_args = []
                else:
                    self.log_message.emit(
                        "[WARNING] Vocabulary tree file not found or invalid FAISS path. Falling back to exhaustive_matcher."
                    )
                    matcher_cmd = "exhaustive_matcher"
                    extra_args = []

            elif matching_mode == "spatial":
                matcher_cmd = "spatial_matcher"
                extra_args = []
                self.log_message.emit("[INFO] Using spatial_matcher for GPS-based camera pose matching.")

            elif matching_mode == "exhaustive_blocked":
                matcher_cmd = "exhaustive_matcher"
                block_size_val = "20"
                if self.custom_params and "colmap_block_size" in self.custom_params:
                    block_size_val = str(self.custom_params["colmap_block_size"])
                extra_args = ["--ExhaustiveMatching.block_size", block_size_val]
                self.log_message.emit(
                    f"[MEMORY OPTIMIZATION] Using exhaustive_matcher with reduced block size ({block_size_val}) "
                    f"to prevent RAM saturation during matrix matching."
                )
            else:
                matcher_cmd = "exhaustive_matcher"
                if self.custom_params and "colmap_block_size" in self.custom_params:
                    block_size_val = str(self.custom_params["colmap_block_size"])
                    extra_args = ["--ExhaustiveMatching.block_size", block_size_val]
                    self.log_message.emit(f"[INFO] Using exhaustive_matcher with custom block size ({block_size_val}).")
                else:
                    extra_args = []

            cmd_match_gpu = [
                colmap_exe, matcher_cmd,
                "--database_path", database_path,
                "--SiftMatching.use_gpu", "1",
                "--SiftMatching.guided_matching", guided_matching,
                "--SiftMatching.max_num_matches", str(colmap_max_num_matches),
                "--SiftMatching.num_threads", str(num_threads),
                "--SiftMatching.max_ratio", nndr_ratio,
            ] + extra_args

            cmd_match_cpu = [
                colmap_exe, matcher_cmd,
                "--database_path", database_path,
                "--SiftMatching.use_gpu", "0",
                "--SiftMatching.guided_matching", guided_matching,
                "--SiftMatching.max_num_matches", str(colmap_max_num_matches),
                "--SiftMatching.num_threads", str(num_threads),
                "--SiftMatching.max_ratio", nndr_ratio,
            ] + extra_args

            self._match_counts = []
            if not self._run_with_gpu_fallback(
                cmd_match_gpu, cmd_match_cpu, timeout=14400.0, env=colmap_env,
                line_parser=self._parse_matching_line
            ):
                if matcher_cmd == "vocab_tree_matcher":
                    self.log_message.emit(
                        "[WARNING] vocab_tree_matcher failed during execution. Retrying automatically with exhaustive_matcher..."
                    )
                    matcher_cmd = "exhaustive_matcher"
                    cmd_match_gpu = [arg for arg in cmd_match_gpu if not arg.startswith("--VocabTreeMatching")]
                    cmd_match_cpu = [arg for arg in cmd_match_cpu if not arg.startswith("--VocabTreeMatching")]
                    cmd_match_gpu[1] = "exhaustive_matcher"
                    cmd_match_cpu[1] = "exhaustive_matcher"
                    if not self._run_with_gpu_fallback(
                        cmd_match_gpu, cmd_match_cpu, timeout=14400.0, env=colmap_env,
                        line_parser=self._parse_matching_line
                    ):
                        return False
                else:
                    return False

            db_stats = self._query_colmap_database_stats(database_path)
            if self._total_images > 1 and db_stats["num_pairs"] == 0:
                self.log_message.emit(
                    "[WARNING] Feature matching produced 0 verified pairs. "
                    "Retrying with CPU matcher and relaxed matching thresholds..."
                )
                self._clear_colmap_match_tables(database_path)
                relaxed_cpu_match = list(cmd_match_cpu)
                self._set_colmap_option(relaxed_cpu_match, "--SiftMatching.guided_matching", "1")
                self._set_colmap_option(relaxed_cpu_match, "--SiftMatching.max_ratio", "0.95")
                self._set_colmap_option(relaxed_cpu_match, "--SiftMatching.max_distance", "0.9")
                self._using_gpu_sift = False
                if not self._run_process_realtime(
                    relaxed_cpu_match, timeout=14400.0, env=colmap_env,
                    line_parser=self._parse_matching_line
                ):
                    return False

            db_stats = self._query_colmap_database_stats(database_path)
            if db_stats["num_images"] > 0:
                self._pairs_tested = (db_stats["num_images"] * (db_stats["num_images"] - 1)) // 2
            else:
                self._pairs_tested = (self._total_images * (self._total_images - 1)) // 2 if self._total_images > 1 else 0
            self._pairs_matched = db_stats["num_pairs"]
            if db_stats["match_counts"]:
                self._match_counts = db_stats["match_counts"]
            self._emit_matching_summary(database_path)
            self.progress_changed.emit(40)
            self._backup_checkpoint("features_extracted")


        if not skip_sfm:
            # =========================================================================
            # STEP 4/9 — Sparse Reconstruction (Mapper)
            # =========================================================================
            self._triangulated_points = 0
            self._registered_count = 0
            sparse_dir = self._to_colmap_path(os.path.join(colmap_out, "sparse"))
            if os.path.exists(sparse_dir):
                try:
                    shutil.rmtree(sparse_dir)
                except Exception as e:
                    self.log_message.emit(f"[WARNING] Failed to clean sparse folder: {e}")
            os.makedirs(sparse_dir, exist_ok=True)
            os.makedirs(os.path.dirname(database_path), exist_ok=True)
            if not os.path.exists(database_path):
                self.log_message.emit(f"[WARNING] COLMAP database missing prior to mapping: {database_path}. Initializing database schema...")
                self._create_colmap_db_schema(database_path)

            cmd_incremental = [
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

            if self.mapper_mode == "global":
                self.status_changed.emit("Step 4/9: Estimating Camera Poses (GLOMAP Global SfM)...")
                self.log_message.emit("[INFO] SfM Mapper: GLOMAP global_mapper selected.")
                cmd_global = [
                    colmap_exe, "global_mapper",
                    "--database_path", database_path,
                    "--image_path", working_image_dir,
                    "--output_path", sparse_dir,
                    "--GlobalMapper.min_num_matches", "15",
                    "--GlobalMapper.num_threads", str(num_threads),
                    "--GlobalMapper.ba_num_iterations", str(ba_global_max_refinements),
                ]
                ok = self._run_process_realtime(cmd_global, timeout=14400.0, env=colmap_env, line_parser=self._parse_mapper_line)
                if not ok or not self._select_best_sparse_model(sparse_dir):
                    self.log_message.emit("[WARNING] GLOMAP global_mapper failed or produced no model. Falling back to COLMAP incremental mapper...")
                    if os.path.exists(sparse_dir):
                        try:
                            shutil.rmtree(sparse_dir)
                        except Exception:
                            pass
                    os.makedirs(sparse_dir, exist_ok=True)
                    if not self._run_process_realtime(cmd_incremental, timeout=14400.0, env=colmap_env, line_parser=self._parse_mapper_line):
                        return False
            else:
                self.status_changed.emit("Step 4/9: Estimating Camera Poses (SfM)...")
                if not self._run_process_realtime(cmd_incremental, timeout=14400.0, env=colmap_env, line_parser=self._parse_mapper_line):
                    return False

            best_model_dir = self._to_colmap_path(self._select_best_sparse_model(sparse_dir)) if self._select_best_sparse_model(sparse_dir) else None
            if not best_model_dir:
                self.log_message.emit(
                    "[FAILED] SfM registered 0 camera poses. Feature matching produced "
                    "insufficient geometric correspondences to initialise reconstruction.\n"
                    "  Suggestions:\n"
                    "  • Try a higher quality preset (Medium or High)\n"
                    "  • Ensure images have at least 60% overlap between adjacent shots"
                )
                return False

            target_model_dir = self._to_colmap_path(os.path.join(sparse_dir, "0"))
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
            self._backup_checkpoint("sparse_reconstruction")
            self.progress_changed.emit(70)

        # =========================================================================
        # STEP 6/9 — Dense Point Cloud Generation
        # =========================================================================
        if skip_dense:
            self.log_message.emit("[RESUME] Skipping Step 6 (Dense Point Cloud already complete).")
            self.progress_changed.emit(80)
        else:
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
            self._backup_checkpoint("dense_reconstruction")
            self.progress_changed.emit(80)

        # =========================================================================
        # STEP 6b — Optional Reference Point Cloud Alignment & Fusion
        # =========================================================================
        fused_mesh_name = self._run_reference_cloud_fusion(mvs_out)

        # =========================================================================
        # STEP 7/9 — Surface Mesh Reconstruction
        # =========================================================================
        if self.mesh_mode == "poisson":
            poisson_ok = self._run_poisson_reconstruction(mvs_out, self.poisson_depth)
            if poisson_ok:
                self.log_message.emit("[POISSON] Poisson reconstruction completed successfully. Skipping OpenMVS meshing, refinement, and texturing.")
                self.progress_changed.emit(99)
                return True
            else:
                self.log_message.emit("[WARNING] Poisson reconstruction failed. Falling back to default OpenMVS Delaunay meshing pipeline...")

        self.status_changed.emit("Step 7/9: Reconstructing Surface Mesh...")
        mvs_mesh_exe = os.path.join(base_dir, self.toolchain_map["openMVS"]["ReconstructMesh"])

        dense_mvs = os.path.join(mvs_out, "scene_dense.mvs")
        target_scene = "scene_dense.mvs" if os.path.exists(dense_mvs) else "scene.mvs"
        if target_scene == "scene.mvs":
            self.log_message.emit("[WARNING] scene_dense.mvs not found. Meshing from sparse scene.mvs.")

        if fused_mesh_name and os.path.exists(os.path.join(mvs_out, fused_mesh_name)):
            self.log_message.emit(f"[INFO] Skipping ReconstructMesh (Delaunay) — using Reference Cloud fused Poisson mesh: {fused_mesh_name}")
            mesh_input = fused_mesh_name
        else:
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
            if target_scene == "scene_dense.mvs":
                mesh_input = "scene_dense_mesh.ply"
            else:
                mesh_input = "scene_mesh.ply"

        self.progress_changed.emit(88)

        # =========================================================================
        # STEP 8/9 — Mesh Geometry Refinement (Multi-Scale)
        # =========================================================================
        self.status_changed.emit("Step 8/9: Refining Mesh Geometry...")
        mvs_refine_exe = os.path.join(base_dir, self.toolchain_map["openMVS"]["RefineMesh"])

        if not os.path.exists(os.path.join(mvs_out, mesh_input)):
            for candidate in ["scene_dense_mesh_refcloud.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
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
        for candidate in ["scene_dense_mesh_refine.ply", "scene_dense_mesh_refcloud.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
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
        self._backup_checkpoint("mesh_reconstruction")
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
                    from PIL import ImageOps
                    # Transpose based on EXIF orientation
                    img_transposed = ImageOps.exif_transpose(img)
                    w, h = img_transposed.size
                    
                    try:
                        exif_data = img_transposed.getexif()
                    except Exception:
                        exif_data = None

                    if max(w, h) > max_image_dim:
                        # Compute scale factor maintaining aspect ratio
                        scale = max_image_dim / max(w, h)
                        new_w = round(w * scale)
                        new_h = round(h * scale)
                        processed = img_transposed.resize((new_w, new_h), Image.LANCZOS)
                        resized_count += 1
                    else:
                        processed = img_transposed
                        copied_count += 1

                    # Save with transposed EXIF preserved (orientation tag is automatically cleared)
                    save_kwargs = {}
                    if exif_data:
                        save_kwargs['exif'] = exif_data
                    # JPEG quality 92 — visually lossless, keeps file size reasonable
                    if filename.lower().endswith(('.jpg', '.jpeg')):
                        save_kwargs['quality'] = 92
                        save_kwargs['subsampling'] = 0
                    processed.save(dst_path, **save_kwargs)

            except Exception as e:
                self.log_message.emit(f"[WARNING] Failed to process image {filename}: {e}. Copying original.")
                try:
                    import shutil as _shutil
                    _shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e2:
                    self.log_message.emit(f"[ERROR] Could not copy {filename}: {e2}")

        self.log_message.emit(
            f"[PREP] Done: {resized_count} image(s) downscaled to <={max_image_dim}px, "
            f"{copied_count} image(s) were already within limit."
        )
        return working_dir

    def _run_with_gpu_fallback(self, cmd_gpu: list, cmd_cpu: list, timeout: float, cwd=None, env=None, line_parser=None) -> bool:
        """Try GPU first, fall back to CPU if GPU fails."""
        if self.gpu_mode == "force_cpu":
            self.log_message.emit("[INFO] CPU-only mode selected. Skipping GPU execution.")
            self._using_gpu_sift = False
            return self._run_process_realtime(cmd_cpu, timeout=timeout, cwd=cwd, env=env, line_parser=line_parser)

        self.log_message.emit("[INFO] Attempting GPU-accelerated execution (OpenGL)...")
        self._using_gpu_sift = True
        if self._run_process_realtime(cmd_gpu, timeout=timeout, cwd=cwd, env=env, line_parser=line_parser):
            return True
        self.log_message.emit("[WARNING] GPU execution failed. Falling back to CPU-only mode...")
        self._using_gpu_sift = False
        return self._run_process_realtime(cmd_cpu, timeout=timeout, cwd=cwd, env=env, line_parser=line_parser)

    def _get_colmap_help(self, colmap_exe: str, subcommand: str) -> str:
        """Fetch and cache colmap <subcommand> -h output to inspect supported CLI option flags."""
        if not hasattr(self, "_colmap_help_cache"):
            self._colmap_help_cache = {}
        cache_key = (colmap_exe, subcommand)
        if cache_key in self._colmap_help_cache:
            return self._colmap_help_cache[cache_key]

        try:
            import subprocess
            creationflags = 0
            if sys.platform == 'win32':
                creationflags = subprocess.CREATE_NO_WINDOW
            res = subprocess.run(
                [colmap_exe, subcommand, "-h"],
                capture_output=True,
                text=True,
                timeout=5.0,
                creationflags=creationflags
            )
            help_text = (res.stdout or "") + "\n" + (res.stderr or "")
        except Exception as e:
            help_text = ""

        self._colmap_help_cache[cache_key] = help_text
        return help_text

    def _adapt_colmap_cmd(self, cmd: list) -> list:
        """
        Dynamically adapts COLMAP feature extraction and matching flags
        to match the exact option names supported by the COLMAP binary
        (e.g., COLMAP <3.11 using --SiftExtraction.use_gpu vs COLMAP 3.11+/4.x using --FeatureExtraction.use_gpu).
        """
        if not cmd or len(cmd) < 2:
            return list(cmd)

        colmap_exe = cmd[0]
        exe_basename = os.path.basename(colmap_exe).lower()
        if "colmap" not in exe_basename:
            return list(cmd)

        subcommand = cmd[1]
        help_text = self._get_colmap_help(colmap_exe, subcommand)
        if not help_text:
            return list(cmd)

        adapted_cmd = list(cmd)

        if subcommand == "feature_extractor":
            use_feature_ext = "--FeatureExtraction.use_gpu" in help_text
            use_sift_ext = "--SiftExtraction.use_gpu" in help_text

            mapping = {}
            if use_feature_ext:
                mapping.update({
                    "--SiftExtraction.use_gpu": "--FeatureExtraction.use_gpu",
                    "--SiftExtraction.num_threads": "--FeatureExtraction.num_threads",
                    "--SiftExtraction.max_image_size": "--FeatureExtraction.max_image_size",
                })
            elif use_sift_ext:
                mapping.update({
                    "--FeatureExtraction.use_gpu": "--SiftExtraction.use_gpu",
                    "--FeatureExtraction.num_threads": "--SiftExtraction.num_threads",
                    "--FeatureExtraction.max_image_size": "--SiftExtraction.max_image_size",
                })

            for i, token in enumerate(adapted_cmd):
                if token in mapping:
                    adapted_cmd[i] = mapping[token]

        elif "matcher" in subcommand:
            use_feature_match = "--FeatureMatching.use_gpu" in help_text
            use_sift_match = "--SiftMatching.use_gpu" in help_text

            mapping = {}
            if use_feature_match:
                mapping.update({
                    "--SiftMatching.use_gpu": "--FeatureMatching.use_gpu",
                    "--SiftMatching.num_threads": "--FeatureMatching.num_threads",
                    "--SiftMatching.guided_matching": "--FeatureMatching.guided_matching",
                    "--SiftMatching.max_num_matches": "--FeatureMatching.max_num_matches",
                })
            elif use_sift_match:
                mapping.update({
                    "--FeatureMatching.use_gpu": "--SiftMatching.use_gpu",
                    "--FeatureMatching.num_threads": "--SiftMatching.num_threads",
                    "--FeatureMatching.guided_matching": "--FeatureMatching.guided_matching",
                    "--FeatureMatching.max_num_matches": "--SiftMatching.max_num_matches",
                })

            for i, token in enumerate(adapted_cmd):
                if token in mapping:
                    adapted_cmd[i] = mapping[token]

        return adapted_cmd

    def _set_colmap_option(self, cmd: list, option: str, value: str):
        """Set or append a COLMAP command-line option in a mutable command list."""
        alt_option = None
        if option.startswith("--SiftMatching."):
            alt_option = option.replace("--SiftMatching.", "--FeatureMatching.")
        elif option.startswith("--FeatureMatching."):
            alt_option = option.replace("--FeatureMatching.", "--SiftMatching.")
        elif option.startswith("--SiftExtraction."):
            alt_option = option.replace("--SiftExtraction.", "--FeatureExtraction.")
        elif option.startswith("--FeatureExtraction."):
            alt_option = option.replace("--FeatureExtraction.", "--SiftExtraction.")

        for opt in (option, alt_option):
            if opt and opt in cmd:
                index = cmd.index(opt)
                if index + 1 < len(cmd):
                    cmd[index + 1] = value
                    return

        cmd.extend([option, value])

    def _clear_colmap_match_tables(self, database_path: str):
        """Clear existing COLMAP match rows so a retry recomputes all pairs."""
        try:
            import sqlite3
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM matches")
            cursor.execute("DELETE FROM two_view_geometries")
            conn.commit()
            conn.close()
        except Exception as e:
            self.log_message.emit(f"[WARNING] Could not clear COLMAP match tables before retry: {e}")

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

    def _run_reference_cloud_fusion(self, mvs_out: str) -> str | None:
        """
        Step 6b: Optional reference cloud alignment, gap fusion, and Poisson meshing.
        Returns the filename of the generated fused mesh PLY, or None if skipped/failed.
        """
        if not self.ref_cloud_path or not os.path.exists(self.ref_cloud_path):
            return None

        self.status_changed.emit("Step 6b: Aligning & Fusing Reference Cloud...")
        self.log_message.emit(f"[REF_CLOUD] Starting reference cloud alignment and fusion pipeline...")

        dense_ply = os.path.join(mvs_out, "scene_dense.ply")
        if not os.path.exists(dense_ply):
            dense_ply = os.path.join(mvs_out, "scene.ply")
        if not os.path.exists(dense_ply):
            self.log_message.emit("[WARNING] No dense or sparse PLY cloud found for reference cloud alignment. Skipping fusion.")
            return None

        try:
            import point_cloud_io
            import cloud_aligner
            import cloud_fusion
            import open3d as o3d

            # 1. Load point clouds
            ref_load = point_cloud_io.load_point_cloud(self.ref_cloud_path, self.log_message.emit)
            if not ref_load.success or ref_load.cloud is None:
                self.log_message.emit(f"[WARNING] Failed to load reference cloud: {'; '.join(ref_load.warnings)}. Skipping fusion.")
                return None

            dense_load = point_cloud_io.load_point_cloud(dense_ply, self.log_message.emit)
            if not dense_load.success or dense_load.cloud is None:
                self.log_message.emit(f"[WARNING] Failed to load scene point cloud ({dense_ply}). Skipping fusion.")
                return None

            # 2. Align reference cloud to scene point cloud
            align_res = cloud_aligner.align_to_dense(ref_load.cloud, dense_load.cloud, self.log_message.emit)
            if not align_res.success:
                self.log_message.emit(f"[WARNING] Reference cloud alignment low confidence: {'; '.join(align_res.warnings)}. Continuing without fusion.")
                return None

            # 3. Transform reference cloud using similarity matrix
            aligned_ref = ref_load.cloud.transform(align_res.transform)

            # Save aligned reference cloud for diagnostics
            aligned_ref_path = os.path.join(mvs_out, "scene_refcloud_aligned.ply")
            point_cloud_io.save_point_cloud(aligned_ref, aligned_ref_path)

            # 4. Merge gap-filling points into dense cloud
            merged_cloud = cloud_fusion.merge_clouds(dense_load.cloud, aligned_ref, self.log_message.emit, gap_radius_mult=3.0)

            merged_cloud_path = os.path.join(mvs_out, "scene_dense_refcloud.ply")
            point_cloud_io.save_point_cloud(merged_cloud, merged_cloud_path)

            # 5. Generate Poisson surface mesh
            fused_mesh = cloud_fusion.generate_mesh(merged_cloud, self.log_message.emit, poisson_depth=9, density_threshold_pct=5.0)

            fused_mesh_name = "scene_dense_mesh_refcloud.ply"
            fused_mesh_path = os.path.join(mvs_out, fused_mesh_name)
            o3d.io.write_triangle_mesh(fused_mesh_path, fused_mesh)

            self.log_message.emit(f"[SUCCESS] Reference cloud fusion complete! Fused Poisson mesh saved as {fused_mesh_name}")
            return fused_mesh_name

        except Exception as e:
            self.log_message.emit(f"[WARNING] Reference cloud fusion encountered an error: {e}. Falling back to standard pipeline.")
            return None

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

    def _parse_feature_extraction_line(self, line: str) -> str | None:
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

    def _parse_matching_line(self, line: str) -> str | None:
        """Parse COLMAP exhaustive_matcher output."""
        import re
        
        match = re.search(r'Matching block \[(\d+)/(\d+),\s*(\d+)/(\d+)\]', line)
        if match:
            # We can try to approximate the image names being matched in this block if possible
            # But the block indexing is quite complex (it depends on block size, max_num_matches etc).
            # We'll just show the block progress for now unless we know the exact names.
            current_1 = int(match.group(1))
            total_1 = int(match.group(2))
            current_2 = int(match.group(3))
            total_2 = int(match.group(4))
            return f"Processing match block [{current_1}/{total_1}, {current_2}/{total_2}]..."

        # Alternatively, COLMAP outputs "Matching block [X/Y]" format sometimes:
        match = re.search(r'Matching block \[(\d+)/(\d+)\]', line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            
            # Since exhaustive_matcher processes images in blocks, let's try to map the block index
            # to image names if we have total images available.
            # Usually exhaustive matcher uses block size of 50 by default.
            block_size = 50 
            start_idx = (current - 1) * block_size + 1
            end_idx = min(current * block_size, self._total_images)
            
            start_name = self._image_names_map.get(start_idx, f"Image {start_idx}")
            end_name = self._image_names_map.get(end_idx, f"Image {end_idx}")
            
            if start_idx == end_idx:
                return f"Processing match block {current}/{total} (Image: {start_name})"
            else:
                return f"Processing match block {current}/{total} (Images: {start_name} to {end_name})"
        
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

    def _parse_mapper_line(self, line: str) -> str | None:
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

    def _parse_densify_line(self, line: str) -> str | None:
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

    def _parse_mesh_line(self, line: str) -> str | None:
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

    def _create_colmap_db_schema(self, db_path: str) -> bool:
        """
        Creates the standard COLMAP SQLite database schema directly in Python.
        Ensures compatibility even when 'database_creator' CLI is unavailable or fails.
        """
        import sqlite3
        try:
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except Exception:
                    pass
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    model INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    params BLOB,
                    prior_focal_length INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    name TEXT NOT NULL UNIQUE,
                    camera_id INTEGER NOT NULL,
                    CONSTRAINT fk_images_camera_id FOREIGN KEY (camera_id) REFERENCES cameras (camera_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS keypoints (
                    image_id INTEGER PRIMARY KEY NOT NULL,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB,
                    CONSTRAINT fk_keypoints_image_id FOREIGN KEY (image_id) REFERENCES images (image_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS descriptors (
                    image_id INTEGER PRIMARY KEY NOT NULL,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB,
                    CONSTRAINT fk_descriptors_image_id FOREIGN KEY (image_id) REFERENCES images (image_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS matches (
                    pair_id INTEGER PRIMARY KEY NOT NULL,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                );
                CREATE TABLE IF NOT EXISTS two_view_geometries (
                    pair_id INTEGER PRIMARY KEY NOT NULL,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB,
                    config INTEGER NOT NULL,
                    F BLOB,
                    E BLOB,
                    H BLOB
                );
            """)
            conn.commit()
            conn.close()
            self.log_message.emit("[SP+LG] COLMAP database schema created successfully via Python schema builder.")
            return True
        except Exception as e:
            self.log_message.emit(f"[ERROR] Failed to create COLMAP database schema: {e}")
            return False

    def _is_valid_checkpoint(self, db_path: str) -> bool:
        """Checks if a COLMAP database exists and contains valid camera/image registration or feature matches."""
        abs_db_path = os.path.abspath(db_path)
        if not os.path.exists(abs_db_path):
            return False
        import sqlite3, time
        for attempt in range(3):
            try:
                conn = sqlite3.connect(abs_db_path, timeout=10.0)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM images")
                num_images = cur.fetchone()[0]
                conn.close()
                return num_images >= 2
            except Exception:
                if attempt < 2:
                    time.sleep(0.3)
        return False

    def _query_colmap_database_stats(self, db_path: str) -> dict:
        """Query COLMAP's SQLite database for feature and match statistics."""
        import sqlite3
        stats = {
            "num_images": 0,
            "feature_counts": [],
            "num_pairs": 0,
            "match_counts": [],
        }
        
        abs_db_path = os.path.abspath(db_path)
        if not os.path.exists(abs_db_path):
            self.log_message.emit(f"[WARNING] COLMAP database file does not exist: {abs_db_path}")
            return stats

        for attempt in range(3):
            try:
                conn = sqlite3.connect(abs_db_path, timeout=10.0)
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
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.3)
                else:
                    self.log_message.emit(f"[WARNING] Could not read COLMAP database ({abs_db_path}): {e}")
        
        return stats

    def _emit_feature_summary(self):
        if not self._feature_counts:
            return
        
        total = len(self._feature_counts)
        avg = sum(self._feature_counts) / total
        min_f = min(self._feature_counts)
        max_f = max(self._feature_counts)

        compute_label = "GPU (CUDA)" if self._using_gpu_sift else "CPU Fallback"
        
        self.log_message.emit(
            f"\n{'='*60}\n"
            f"  FEATURE EXTRACTION SUMMARY\n"
            f"{'='*60}\n"
            f"  Images processed:     {total}\n"
            f"  Features per image:   {avg:,.0f} avg  |  {min_f:,} min  |  {max_f:,} max\n"
            f"  Total features:       {sum(self._feature_counts):,}\n"
            f"  Compute device:       {compute_label}\n"
            f"{'='*60}"
        )
        
        if avg < 3000:
            self.log_message.emit(
                "[⚠ DIAGNOSTIC] Average features per image is LOW (<3000). "
                "This may cause poor camera registration. "
                "Consider: higher quality preset, better image overlap, or sharper images."
            )

    def _emit_matching_summary(self, db_path: str = None):
        top_pairs_str = ""
        if db_path and os.path.exists(db_path):
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                # two_view_geometries structure: pair_id, rows, cols, data, config, F, E, H
                # We can't easily join pair_id to image names because pair_id is computed as:
                # image_id1 * 2147483647 + image_id2. Let's just do a simple query if we want, or just get max matches.
                # Since pair_id is complex to decode in SQL without a function, we'll fetch rows and decode in Python.
                cursor.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0 ORDER BY rows DESC LIMIT 5")
                top_pairs = cursor.fetchall()
                if top_pairs and self._image_names_map:
                    top_pairs_str = "\n  Top matched pairs:\n"
                    for pair_id, matches in top_pairs:
                        image_id2 = pair_id % 2147483647
                        image_id1 = pair_id // 2147483647
                        name1 = self._image_names_map.get(image_id1, f"Img {image_id1}")
                        name2 = self._image_names_map.get(image_id2, f"Img {image_id2}")
                        top_pairs_str += f"    {name1} ↔ {name2} : {matches} matches\n"
                conn.close()
            except Exception as e:
                pass

        self.log_message.emit(
            f"\n{'='*60}\n"
            f"  FEATURE MATCHING SUMMARY\n"
            f"{'='*60}\n"
            f"  Image pairs tested:   {self._pairs_tested}\n"
            f"  Pairs with matches:   {self._pairs_matched}\n"
            f"  Match success rate:   {(self._pairs_matched/max(self._pairs_tested,1))*100:.1f}%\n"
            f"  Avg matches/pair:     {sum(self._match_counts)/max(len(self._match_counts),1):,.0f}\n"
            f"{top_pairs_str}"
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

    def _run_poisson_reconstruction(self, mvs_out: str, depth: int) -> bool:
        """
        Step 7P: Poisson Surface Reconstruction using Open3D and cloud_fusion module.
        Reads the dense point cloud, reconstructs surface, trims low-density components,
        and saves PLY, OBJ, and GLB mesh outputs.
        """
        self.log_message.emit("[POISSON] Starting Poisson surface reconstruction...")
        
        # 1. Find the input dense point cloud PLY file
        dense_ply = os.path.join(mvs_out, "scene_dense.ply")
        if not os.path.exists(dense_ply):
            dense_ply = os.path.join(mvs_out, "scene.ply")
            
        if not os.path.exists(dense_ply):
            self.log_message.emit("[ERROR] No dense or sparse PLY cloud found for Poisson reconstruction.")
            return False

        try:
            import open3d as o3d
            import cloud_fusion
            import trimesh
            import numpy as np

            # Load the point cloud
            self.log_message.emit(f"[POISSON] Loading point cloud: {os.path.basename(dense_ply)}...")
            pcd = o3d.io.read_point_cloud(dense_ply)
            if pcd is None or len(pcd.points) == 0:
                self.log_message.emit("[ERROR] Point cloud is empty or failed to load.")
                return False

            self.status_changed.emit("Step 7P: Reconstructing Poisson Surface...")
            self.progress_changed.emit(83)

            # Generate Poisson mesh using the cloud_fusion module's helper
            # This handles normal estimation, orientation consistency check, density trimming, and cluster fragment cleaning.
            mesh = cloud_fusion.generate_mesh(
                pcd, 
                log_fn=self.log_message.emit, 
                poisson_depth=depth, 
                density_threshold_pct=5.0
            )

            if mesh is None or len(mesh.vertices) == 0:
                self.log_message.emit("[ERROR] Poisson reconstruction produced an empty mesh.")
                return False

            self.status_changed.emit("Step 7P: Exporting Mesh Formats...")
            self.progress_changed.emit(92)

            # Output paths
            ply_path = os.path.normpath(os.path.join(mvs_out, "scene_dense_mesh_texture.ply"))
            obj_path = os.path.normpath(os.path.join(mvs_out, "scene_dense_mesh_texture.obj"))
            glb_path = os.path.normpath(os.path.join(mvs_out, "scene_dense_mesh_texture.glb"))

            # Write PLY using Open3D (includes vertex colors)
            o3d.io.write_triangle_mesh(ply_path, mesh)
            self.log_message.emit(f"[POISSON] Successfully saved PLY mesh to: {os.path.basename(ply_path)}")

            # Load with trimesh to convert to OBJ and GLB (keeps vertex colors)
            self.log_message.emit("[POISSON] Exporting mesh to OBJ and GLB formats...")
            tri_mesh = trimesh.load(ply_path, force="mesh")
            tri_mesh.export(obj_path)
            self.log_message.emit(f"[POISSON] Successfully saved OBJ mesh to: {os.path.basename(obj_path)}")

            tri_mesh.export(glb_path)
            self.log_message.emit(f"[POISSON] Successfully saved GLB mesh to: {os.path.basename(glb_path)}")

            # Update stats for display in the summary
            self._mesh_vertices = len(mesh.vertices)
            self._mesh_faces = len(mesh.triangles)

            self.progress_changed.emit(98)
            return True

        except Exception as e:
            self.log_message.emit(f"[WARNING] Poisson reconstruction failed with error: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            return False


