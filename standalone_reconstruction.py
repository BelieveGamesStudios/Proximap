import os
import sys
import time
import json
import numpy as np
import open3d as o3d
from PySide6.QtCore import QThread, Signal
import point_cloud_io

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

class StandaloneReconstructionWorker(QThread):
    progress_changed = Signal(int)
    status_changed = Signal(str)
    log_message = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, cloud_path: str, output_dir: str, include_colors: bool = True, poisson_depth: int = 9, parent=None):
        super().__init__(parent)
        self.cloud_path = cloud_path
        self.output_dir = output_dir
        self.include_colors = include_colors
        self.poisson_depth = poisson_depth
        self.is_running = True
        self.toolchain_map = self._load_toolchain_map()

    def _load_toolchain_map(self) -> dict:
        map_path = os.path.join(get_base_dir(), "toolchain_map.json")
        if os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    return self._normalize_toolchain_map(json.load(f))
            except Exception as e:
                self.log_message.emit(f"Error reading toolchain_map.json: {e}")
        return {}

    def _normalize_toolchain_map(self, toolchain_map: dict) -> dict:
        if sys.platform == "win32":
            return toolchain_map
        normalized = {}
        for group, binaries in toolchain_map.items():
            if not isinstance(binaries, dict):
                normalized[group] = binaries
                continue
            normalized[group] = {}
            for name, rel_path in binaries.items():
                mac_rel_path = rel_path[:-4] if rel_path.lower().endswith(".exe") else rel_path
                mac_abs_path = os.path.join(get_base_dir(), mac_rel_path)
                normalized[group][name] = mac_rel_path if os.path.exists(mac_abs_path) else rel_path
        return normalized

    def cancel(self):
        self.is_running = False

    def run(self):
        try:
            self.status_changed.emit("Step 1/5: Loading Point Cloud...")
            self.progress_changed.emit(5)
            self.log_message.emit(f"[STANDALONE] Loading point cloud: {self.cloud_path}")
            
            # 1. Ingest
            load_result = point_cloud_io.load_point_cloud(self.cloud_path, self.log_message.emit)
            if not load_result.success or load_result.cloud is None:
                raise RuntimeError(f"Failed to load point cloud: {', '.join(load_result.warnings)}")

            pcd = load_result.cloud
            self.progress_changed.emit(20)

            if not self.is_running:
                raise InterruptedError("Canceled by user.")

            # 2. Preprocess (Outlier Removal & Normals)
            self.status_changed.emit("Step 2/5: Preprocessing (Outlier Removal & Normals)...")
            self.log_message.emit("[STANDALONE] Removing statistical outliers (nb_neighbors=20, std_ratio=2.0)...")
            
            # Statistical outlier removal
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd_cleaned = pcd.select_by_index(ind)
            removed = len(pcd.points) - len(pcd_cleaned.points)
            self.log_message.emit(f"[STANDALONE] Outlier removal complete. Removed {removed:,} outlier points.")
            
            if not self.is_running:
                raise InterruptedError("Canceled by user.")

            # Normal estimation
            self.log_message.emit("[STANDALONE] Estimating normal vectors...")
            # Compute median spacing for dynamic scale
            pts = np.asarray(pcd_cleaned.points)
            if len(pts) == 0:
                raise RuntimeError("Cleaned point cloud has 0 points.")
                
            # Compute nearest neighbor distance median
            pcd_cleaned_sample = pcd_cleaned
            if len(pts) > 5000:
                indices = np.random.choice(len(pts), 5000, replace=False)
                sample_pts = pts[indices]
                pcd_cleaned_sample = o3d.geometry.PointCloud()
                pcd_cleaned_sample.points = o3d.utility.Vector3dVector(sample_pts)
                
            kdtree = o3d.geometry.KDTreeFlann(pcd_cleaned_sample)
            distances = []
            for i in range(len(pcd_cleaned_sample.points)):
                [k, idx, dist_sq] = kdtree.search_knn_vector_3d(pcd_cleaned_sample.points[i], 2)
                if k >= 2 and dist_sq[1] > 0:
                    distances.append(np.sqrt(dist_sq[1]))
            d_spacing = float(np.median(distances)) if distances else 0.01

            radius_normal = d_spacing * 3.5
            pcd_cleaned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            pcd_cleaned.orient_normals_consistent_tangent_plane(k=15)
            self.log_message.emit("[STANDALONE] Normals estimated and oriented consistently.")
            self.progress_changed.emit(50)

            if not self.is_running:
                raise InterruptedError("Canceled by user.")

            # 3. Poisson reconstruction
            self.status_changed.emit("Step 3/5: Generating Poisson Surface Mesh...")
            self.log_message.emit(f"[STANDALONE] Running Poisson Surface Reconstruction (depth={self.poisson_depth})...")
            
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd_cleaned, depth=self.poisson_depth, linear_fit=True
                )
            self.log_message.emit(f"[STANDALONE] Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces.")

            # Trimming low-evidence vertices
            densities_np = np.asarray(densities)
            if len(densities_np) > 0:
                cutoff_density = np.percentile(densities_np, 5.0)
                vertices_to_remove = densities_np < cutoff_density
                mesh.remove_vertices_by_mask(vertices_to_remove)
                self.log_message.emit(f"[STANDALONE] Trimmed lower 5% low-evidence vertices (< {cutoff_density:.2f}).")

            # Clean small disconnected components
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            if len(cluster_n_triangles) > 1:
                total_faces = len(mesh.triangles)
                min_cluster_size = max(50, int(total_faces * 0.005))
                triangles_to_remove = cluster_n_triangles[np.asarray(triangle_clusters)] < min_cluster_size
                mesh.remove_triangles_by_mask(triangles_to_remove)
                mesh.remove_unreferenced_vertices()
                self.log_message.emit(f"[STANDALONE] Removed small floating fragments (< {min_cluster_size} faces).")

            mesh.compute_vertex_normals()
            
            # Carry over colors if toggle enabled, otherwise strip/clear
            if self.include_colors and load_result.has_colors:
                self.log_message.emit("[STANDALONE] Vertex colors preserved in the reconstructed mesh.")
            else:
                mesh.vertex_colors = o3d.utility.Vector3dVector([])
                self.log_message.emit("[STANDALONE] Exporting bare geometry (vertex colors stripped).")

            # Write intermediate Poisson mesh to scene_dense_mesh.ply
            mvs_dir = os.path.join(self.output_dir, "mvs")
            os.makedirs(mvs_dir, exist_ok=True)
            poisson_mesh_path = os.path.join(mvs_dir, "scene_dense_mesh.ply")
            o3d.io.write_triangle_mesh(poisson_mesh_path, mesh)
            self.log_message.emit(f"[STANDALONE] Saved Poisson mesh to {poisson_mesh_path}")
            self.progress_changed.emit(75)

            if not self.is_running:
                raise InterruptedError("Canceled by user.")

            # 4. RefineMesh decimation/cleanup
            self.status_changed.emit("Step 4/5: Refining Mesh Geometry...")
            
            refine_exe = None
            if "openMVS" in self.toolchain_map and "RefineMesh" in self.toolchain_map["openMVS"]:
                refine_exe = os.path.join(get_base_dir(), self.toolchain_map["openMVS"]["RefineMesh"])

            if refine_exe and os.path.exists(refine_exe):
                refined_mesh_name = "scene_dense_mesh_refine.ply"
                refined_mesh_path = os.path.join(mvs_dir, refined_mesh_name)
                self.log_message.emit(f"[STANDALONE] Found RefineMesh tool. Running refinement subprocess...")
                
                cmd = [
                    refine_exe,
                    "-i", "scene_dense_mesh.ply",
                    "-o", refined_mesh_name,
                    "--resolution-level", "1",
                    "--scales", "2"
                ]
                
                self.log_message.emit(f"[RUN] {' '.join(cmd)}")
                
                creationflags = 0
                if sys.platform == 'win32':
                    import subprocess
                    creationflags = subprocess.CREATE_NO_WINDOW
                
                import subprocess
                from hardware_profiler import _active_subprocesses
                
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=mvs_dir,
                    creationflags=creationflags
                )
                _active_subprocesses.add(proc)
                
                while proc.poll() is None:
                    if not self.is_running:
                        proc.terminate()
                        proc.wait(timeout=2.0)
                        raise InterruptedError("Canceled by user.")
                    
                    line = proc.stdout.readline()
                    if line:
                        stripped_line = line.strip()
                        if stripped_line:
                            self.log_message.emit(f"[RefineMesh] {stripped_line}")
                    else:
                        time.sleep(0.05)
                
                # Read any remaining output
                for line in proc.stdout.readlines():
                    self.log_message.emit(f"[RefineMesh] {line.strip()}")
                
                if proc.returncode == 0 and os.path.exists(refined_mesh_path):
                    self.log_message.emit(f"[STANDALONE] RefineMesh completed successfully → {refined_mesh_path}")
                else:
                    self.log_message.emit(f"[WARNING] RefineMesh exited with code {proc.returncode} or produced no output. Falling back to Poisson mesh.")
                    import shutil
                    shutil.copy2(poisson_mesh_path, refined_mesh_path)
            else:
                self.log_message.emit("[WARNING] RefineMesh binary not found. Bypassing refinement. Falling back to Poisson mesh.")
                refined_mesh_path = os.path.join(mvs_dir, "scene_dense_mesh_refine.ply")
                import shutil
                shutil.copy2(poisson_mesh_path, refined_mesh_path)

            self.progress_changed.emit(95)

            # 5. Finalize
            self.status_changed.emit("Step 5/5: Finalizing Reconstruction...")
            self.progress_changed.emit(100)
            time.sleep(0.5)
            self.finished.emit(True, "Standalone point cloud reconstruction completed successfully!")

        except InterruptedError as ie:
            self.log_message.emit(f"[CANCELED] {ie}")
            self.finished.emit(False, str(ie))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log_message.emit(f"[ERROR] Standalone reconstruction failed: {e}\n{tb}")
            self.finished.emit(False, str(e))
