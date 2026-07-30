import os
import sys
import time
import json
import numpy as np
from PySide6.QtCore import QThread, Signal
import point_cloud_io

HAS_OPEN3D = False
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None

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
                    normalized[group][name] = clean_rel_path
                elif shutil.which(name):
                    normalized[group][name] = shutil.which(name)
                elif shutil.which(os.path.basename(clean_rel_path)):
                    normalized[group][name] = shutil.which(os.path.basename(clean_rel_path))
                else:
                    normalized[group][name] = clean_rel_path
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

            mvs_dir = os.path.join(self.output_dir, "mvs")
            os.makedirs(mvs_dir, exist_ok=True)
            poisson_mesh_path = os.path.join(mvs_dir, "scene_dense_mesh.ply")

            if HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
                # Open3D pipeline path
                self.status_changed.emit("Step 2/5: Preprocessing (Outlier Removal & Normals)...")
                self.log_message.emit("[STANDALONE] Removing statistical outliers (nb_neighbors=20, std_ratio=2.0)...")
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                pcd_cleaned = pcd.select_by_index(ind)
                removed = len(pcd.points) - len(pcd_cleaned.points)
                self.log_message.emit(f"[STANDALONE] Outlier removal complete. Removed {removed:,} outlier points.")
                
                if not self.is_running:
                    raise InterruptedError("Canceled by user.")

                self.log_message.emit("[STANDALONE] Estimating normal vectors...")
                pts = np.asarray(pcd_cleaned.points)
                if len(pts) == 0:
                    raise RuntimeError("Cleaned point cloud has 0 points.")
                    
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

                self.status_changed.emit("Step 3/5: Generating Surface Mesh...")
                self.log_message.emit(f"[STANDALONE] Running Poisson Surface Reconstruction (depth={self.poisson_depth})...")
                
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd_cleaned, depth=self.poisson_depth, linear_fit=True
                    )
                self.log_message.emit(f"[STANDALONE] Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces.")

                densities_np = np.asarray(densities)
                if len(densities_np) > 0:
                    cutoff_density = np.percentile(densities_np, 5.0)
                    vertices_to_remove = densities_np < cutoff_density
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                    self.log_message.emit(f"[STANDALONE] Trimmed lower 5% low-evidence vertices (< {cutoff_density:.2f}).")

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
                
                if self.include_colors and load_result.has_colors:
                    self.log_message.emit("[STANDALONE] Vertex colors preserved in the reconstructed mesh.")
                else:
                    mesh.vertex_colors = o3d.utility.Vector3dVector([])
                    self.log_message.emit("[STANDALONE] Exporting bare geometry (vertex colors stripped).")

                o3d.io.write_triangle_mesh(poisson_mesh_path, mesh)
            else:
                # SciPy + NumPy + Trimesh fallback pipeline
                self.status_changed.emit("Step 2/5: Preprocessing (Outlier Removal & Normals via SciPy)...")
                self.log_message.emit("[STANDALONE] Running SciPy statistical outlier removal...")
                
                pts = np.asarray(pcd.points, dtype=np.float32)
                colors = np.asarray(pcd.colors) if hasattr(pcd, 'colors') and pcd.colors is not None else None
                normals = np.asarray(pcd.normals) if hasattr(pcd, 'normals') and pcd.normals is not None else None

                import scipy.spatial
                tree = scipy.spatial.cKDTree(pts)
                dists, _ = tree.query(pts, k=min(21, len(pts)), workers=-1)
                if dists.ndim == 2 and dists.shape[1] > 1:
                    mean_dists = dists[:, 1:].mean(axis=1)
                    g_mean, g_std = float(mean_dists.mean()), float(mean_dists.std())
                    inliers = mean_dists < (g_mean + 2.0 * g_std)
                    removed = len(pts) - int(np.sum(inliers))
                    pts = pts[inliers]
                    if colors is not None and len(colors) == len(inliers): colors = colors[inliers]
                    if normals is not None and len(normals) == len(inliers): normals = normals[inliers]
                    self.log_message.emit(f"[STANDALONE] Outlier removal complete. Removed {removed:,} outlier points.")

                if not self.is_running:
                    raise InterruptedError("Canceled by user.")

                tree = scipy.spatial.cKDTree(pts)
                if normals is None or len(normals) != len(pts):
                    self.log_message.emit("[STANDALONE] Estimating normal vectors via SciPy cKDTree SVD...")
                    k_nn = min(20, len(pts))
                    _, idxs = tree.query(pts, k=k_nn, workers=-1)
                    neighbors = pts[idxs] - pts[:, np.newaxis, :]
                    covs = np.einsum('nki,nkj->nij', neighbors, neighbors) / float(k_nn)
                    evals, evecs = np.linalg.eigh(covs)
                    normals = evecs[:, :, 0]
                    centroid = pts.mean(axis=0)
                    dots = np.einsum('ni,ni->n', normals, pts - centroid)
                    normals[dots < 0] *= -1.0
                    self.log_message.emit("[STANDALONE] Normals estimated successfully.")

                self.progress_changed.emit(40)

                if not self.is_running:
                    raise InterruptedError("Canceled by user.")

                self.status_changed.emit("Step 3/5: Generating Surface Mesh (SDF + Marching Cubes)...")
                self.log_message.emit(f"[STANDALONE] Computing Signed Distance Field grid (depth={self.poisson_depth})...")
                
                min_b = pts.min(axis=0) - 0.05
                max_b = pts.max(axis=0) + 0.05
                res_map = {5: 64, 6: 96, 7: 128, 8: 160, 9: 192, 10: 224}
                res = res_map.get(self.poisson_depth, 140)

                x_pts = np.linspace(min_b[0], max_b[0], res)
                y_pts = np.linspace(min_b[1], max_b[1], res)
                z_pts = np.linspace(min_b[2], max_b[2], res)
                
                sdf_grid = np.zeros((res, res, res), dtype=np.float32)
                num_slices = 8
                slice_size = res // num_slices
                
                for s in range(num_slices):
                    if not self.is_running:
                        raise InterruptedError("Canceled by user.")
                    z_start = s * slice_size
                    z_end = (s + 1) * slice_size if s < num_slices - 1 else res
                    z_sub = z_pts[z_start:z_end]
                    
                    gx, gy, gz = np.meshgrid(x_pts, y_pts, z_sub, indexing='ij')
                    sub_pts = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T
                    
                    dists, idxs = tree.query(sub_pts, k=1, workers=-1)
                    vecs = sub_pts - pts[idxs]
                    sdfs = np.einsum('ij,ij->i', vecs, normals[idxs])
                    sdf_grid[:, :, z_start:z_end] = sdfs.reshape((res, res, len(z_sub)))
                    
                    prog = 40 + int((s + 1) / num_slices * 30)
                    self.progress_changed.emit(prog)

                self.log_message.emit("[STANDALONE] Extracting isosurface mesh via Marching Cubes...")
                import trimesh
                pitch = (max_b - min_b) / (res - 1)
                mesh = trimesh.voxel.ops.matrix_to_marching_cubes(sdf_grid, pitch=pitch)
                mesh.vertices += min_b

                if self.include_colors and colors is not None and len(colors) > 0:
                    _, v_idxs = tree.query(mesh.vertices, k=1, workers=-1)
                    v_colors = colors[v_idxs]
                    if v_colors.ndim == 2 and v_colors.shape[1] == 3:
                        v_colors = np.column_stack([v_colors, np.full(len(v_colors), 255, dtype=np.uint8)])
                    mesh.visual.vertex_colors = v_colors
                    self.log_message.emit("[STANDALONE] Vertex colors preserved in final mesh.")
                else:
                    self.log_message.emit("[STANDALONE] Exporting mesh geometry.")

                mesh.export(poisson_mesh_path)

            self.log_message.emit(f"[STANDALONE] Saved surface mesh to {poisson_mesh_path}")
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
                
                for line in proc.stdout.readlines():
                    self.log_message.emit(f"[RefineMesh] {line.strip()}")
                
                if proc.returncode == 0 and os.path.exists(refined_mesh_path):
                    self.log_message.emit(f"[STANDALONE] RefineMesh completed successfully → {refined_mesh_path}")
                else:
                    self.log_message.emit(f"[WARNING] RefineMesh exited with code {proc.returncode} or produced no output. Falling back to base mesh.")
                    import shutil
                    shutil.copy2(poisson_mesh_path, refined_mesh_path)
            else:
                self.log_message.emit("[WARNING] RefineMesh binary not found. Bypassing refinement. Falling back to base surface mesh.")
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
