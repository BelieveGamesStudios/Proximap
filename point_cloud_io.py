"""
Point Cloud I/O Module for Proximap
Handles loading, saving, and validating point clouds in PLY, LAS/LAZ, and XYZ formats.
Supports both Open3D and pure NumPy / SciPy fallbacks when Open3D is unavailable.
"""

import os
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable

@dataclass
class LoadResult:
    cloud: Optional[object]  # open3d.geometry.PointCloud or SimplePointCloud
    point_count: int = 0
    has_colors: bool = False
    has_normals: bool = False
    bbox_diagonal: float = 0.0
    warnings: List[str] = field(default_factory=list)
    success: bool = False


class SimplePointCloud:
    """Lightweight pure-NumPy point cloud representation used when Open3D is not installed."""
    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None, normals: Optional[np.ndarray] = None):
        self.points = np.asarray(points, dtype=np.float32)
        self.colors = np.asarray(colors, dtype=np.uint8) if colors is not None and len(colors) > 0 else None
        self.normals = np.asarray(normals, dtype=np.float32) if normals is not None and len(normals) > 0 else None

    def has_colors(self) -> bool:
        return self.colors is not None and len(self.colors) == len(self.points)

    def has_normals(self) -> bool:
        return self.normals is not None and len(self.normals) == len(self.points)


def load_point_cloud(file_path: str, log_fn: Optional[Callable[[str], None]] = None) -> LoadResult:
    """
    Loads a point cloud from PLY, LAS, LAZ, or XYZ files into an Open3D PointCloud
    or SimplePointCloud object.
    """
    def log(msg: str):
        if log_fn:
            log_fn(msg)

    if not os.path.exists(file_path):
        return LoadResult(cloud=None, warnings=[f"File not found: {file_path}"], success=False)

    ext = os.path.splitext(file_path)[1].lower()
    log(f"[REF_CLOUD] Loading point cloud: {os.path.basename(file_path)} ({ext})")
    warnings = []

    # 1. Try Open3D if available
    try:
        import open3d as o3d
        cloud = o3d.geometry.PointCloud()
        if ext in ['.las', '.laz']:
            cloud = _load_las_laz_o3d(file_path, warnings, log)
        elif ext in ['.ply', '.xyz', '.pts', '.txt']:
            cloud = o3d.io.read_point_cloud(file_path)
            if len(cloud.points) == 0 and ext in ['.xyz', '.pts', '.txt']:
                cloud = _load_xyz_numpy(file_path, warnings, log)
        
        if cloud is not None and len(cloud.points) > 0:
            cloud = cloud.remove_non_finite_points(remove_nan=True, remove_infinite=True)
            pt_count = len(cloud.points)
            if pt_count >= 100:
                has_colors = cloud.has_colors()
                has_normals = cloud.has_normals()
                bbox = cloud.get_axis_aligned_bounding_box()
                bbox_diag = float(np.linalg.norm(bbox.get_extent()))
                log(f"[REF_CLOUD] Loaded {pt_count:,} points | Colors: {has_colors} | Normals: {has_normals} | BBox Diag: {bbox_diag:.3f}")
                return LoadResult(
                    cloud=cloud,
                    point_count=pt_count,
                    has_colors=has_colors,
                    has_normals=has_normals,
                    bbox_diagonal=bbox_diag,
                    warnings=warnings,
                    success=True
                )
    except Exception as e:
        log(f"[INFO] Open3D load unavailable ({e}), using NumPy backend...")

    # 2. Pure NumPy Fallback
    try:
        spc = _load_numpy_fallback(file_path, warnings, log)
        if spc is not None and len(spc.points) > 0:
            # Filter non-finite points
            valid_mask = np.isfinite(spc.points).all(axis=1)
            if not np.all(valid_mask):
                spc.points = spc.points[valid_mask]
                if spc.colors is not None: spc.colors = spc.colors[valid_mask]
                if spc.normals is not None: spc.normals = spc.normals[valid_mask]

            pt_count = len(spc.points)
            if pt_count < 100:
                warnings.append(f"Point cloud contains too few points ({pt_count} points). Minimum 100 required.")
                return LoadResult(cloud=spc, point_count=pt_count, warnings=warnings, success=False)

            extent = np.ptp(spc.points, axis=0)
            bbox_diag = float(np.linalg.norm(extent))
            has_colors = spc.has_colors()
            has_normals = spc.has_normals()
            log(f"[REF_CLOUD] Loaded {pt_count:,} points (NumPy) | Colors: {has_colors} | Normals: {has_normals} | BBox Diag: {bbox_diag:.3f}")
            return LoadResult(
                cloud=spc,
                point_count=pt_count,
                has_colors=has_colors,
                has_normals=has_normals,
                bbox_diagonal=bbox_diag,
                warnings=warnings,
                success=True
            )
    except Exception as e:
        warnings.append(f"Failed to load point cloud file with NumPy backend: {e}")

    return LoadResult(cloud=None, warnings=warnings, success=False)


def peek_has_colors(file_path: str) -> bool:
    """
    Quickly checks whether a point cloud file contains vertex color data
    by reading only the file header — NOT loading the full point cloud.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.ply':
            with open(file_path, 'rb') as f:
                header = b''
                while len(header) < 4096:
                    line = f.readline()
                    if not line: break
                    header += line
                    if b'end_header' in line: break
            header_str = header.decode('utf-8', errors='ignore').lower()
            return any(
                f'property {dtype} {ch}' in header_str
                for dtype in ('uchar', 'uint8', 'float', 'float32')
                for ch in ('red', 'green', 'blue', 'r', 'g', 'b')
            )
        elif ext in ('.las', '.laz'):
            try:
                import laspy
                las = laspy.read(file_path)
                fmt_id = int(getattr(las.header.point_format, 'id', 0))
                return fmt_id in (2, 3, 5, 7, 8, 10)
            except Exception:
                return True
        else:
            with open(file_path, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        return len(line.split()) >= 6
            return False
    except Exception:
        return True


def _load_numpy_fallback(file_path: str, warnings: List[str], log: Callable[[str], None]) -> Optional[SimplePointCloud]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.ply':
        pts, colors, _ = _read_ply_pure_numpy(file_path)
        return SimplePointCloud(pts, colors)
    elif ext in ('.las', '.laz'):
        try:
            import laspy
            las = laspy.read(file_path)
            pts = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
            colors = None
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                r = np.asarray(las.red, dtype=np.float32)
                g = np.asarray(las.green, dtype=np.float32)
                b = np.asarray(las.blue, dtype=np.float32)
                max_val = max(np.max(r), np.max(g), np.max(b))
                scale = 255.0 if max_val <= 255.0 else 65535.0
                colors = (np.vstack((r, g, b)).T / scale * 255.0).astype(np.uint8)
            return SimplePointCloud(pts, colors)
        except Exception as e:
            warnings.append(f"laspy error: {e}")
            return None
    elif ext in ('.xyz', '.pts', '.txt'):
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError("XYZ file must have at least 3 columns (X, Y, Z)")
        pts = data[:, :3].astype(np.float32)
        colors = None
        if data.shape[1] >= 6:
            c = data[:, 3:6]
            if np.max(c) <= 1.0: c *= 255.0
            colors = c.astype(np.uint8)
        return SimplePointCloud(pts, colors)
    return None


def _read_ply_pure_numpy(path: str):
    if not os.path.exists(path):
        return np.zeros((0, 3), np.float32), None, None
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        num_vertices = num_faces = 0
        format_type = None
        vertex_properties = []
        element_type = None
        for line in header_lines:
            parts = line.split()
            if not parts: continue
            if parts[0] == 'format': format_type = parts[1]
            elif parts[0] == 'element':
                element_type = parts[1]
                if element_type == 'vertex': num_vertices = int(parts[2])
                elif element_type == 'face': num_faces = int(parts[2])
            elif parts[0] == 'property' and element_type == 'vertex':
                if parts[1] == 'list':
                    vertex_properties.append((parts[4], 'list', True, parts[2], parts[3]))
                else:
                    vertex_properties.append((parts[2], parts[1], False, None, None))

        type_map = {
            'char': (np.int8, 1), 'uchar': (np.uint8, 1), 'short': (np.int16, 2), 'ushort': (np.uint16, 2),
            'int': (np.int32, 4), 'uint': (np.uint32, 4), 'float': (np.float32, 4), 'double': (np.float64, 8),
            'int8': (np.int8, 1), 'uint8': (np.uint8, 1), 'int16': (np.int16, 2), 'uint16': (np.uint16, 2),
            'int32': (np.int32, 4), 'uint32': (np.uint32, 4), 'float32': (np.float32, 4), 'float64': (np.float64, 8)
        }
        points = np.zeros((num_vertices, 3), dtype=np.float32)
        prop_names = [p[0] for p in vertex_properties]
        has_color = all(c in prop_names for c in ('red', 'green', 'blue')) or all(c in prop_names for c in ('r', 'g', 'b'))
        colors = np.zeros((num_vertices, 3), dtype=np.uint8) if has_color else None

        if 'binary' in (format_type or ''):
            fixed_props = [p for p in vertex_properties if not p[2]]
            stride = sum(type_map[p[1]][1] for p in fixed_props if p[1] in type_map)
            raw = f.read(stride * num_vertices)
            dt_fields = []
            for p in fixed_props:
                if p[1] in type_map:
                    dt_fields.append((p[0], type_map[p[1]][0]))
            if dt_fields:
                dt = np.dtype(dt_fields)
                arr = np.frombuffer(raw, dtype=dt)
                if 'x' in arr.dtype.names: points[:, 0] = arr['x'].astype(np.float32)
                if 'y' in arr.dtype.names: points[:, 1] = arr['y'].astype(np.float32)
                if 'z' in arr.dtype.names: points[:, 2] = arr['z'].astype(np.float32)
                if has_color and colors is not None:
                    r_key = 'red' if 'red' in arr.dtype.names else 'r'
                    g_key = 'green' if 'green' in arr.dtype.names else 'g'
                    b_key = 'blue' if 'blue' in arr.dtype.names else 'b'
                    if r_key in arr.dtype.names: colors[:, 0] = arr[r_key]
                    if g_key in arr.dtype.names: colors[:, 1] = arr[g_key]
                    if b_key in arr.dtype.names: colors[:, 2] = arr[b_key]
        else:
            for i in range(num_vertices):
                vals = f.readline().decode('utf-8', errors='ignore').split()
                if len(vals) >= 3:
                    points[i] = [float(vals[0]), float(vals[1]), float(vals[2])]
                    if has_color and colors is not None and len(vals) >= 6:
                        try:
                            colors[i] = [int(float(vals[3])), int(float(vals[4])), int(float(vals[5]))]
                        except: pass
    return points, colors, None


def _load_las_laz_o3d(file_path: str, warnings: List[str], log: Callable[[str], None]):
    import open3d as o3d
    import laspy
    las = laspy.read(file_path)
    coords = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(coords)
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = np.asarray(las.red, dtype=np.float64)
        g = np.asarray(las.green, dtype=np.float64)
        b = np.asarray(las.blue, dtype=np.float64)
        max_val = max(np.max(r), np.max(g), np.max(b))
        scale = 65535.0 if max_val > 255.0 else 255.0
        if scale > 0:
            colors = np.vstack((r / scale, g / scale, b / scale)).transpose()
            cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _load_xyz_numpy(file_path: str, warnings: List[str], log: Callable[[str], None]):
    import open3d as o3d
    data = np.loadtxt(file_path)
    if data.ndim == 1: data = data.reshape(1, -1)
    if data.shape[1] < 3: raise ValueError("XYZ file must have at least 3 columns")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))
    if data.shape[1] >= 6:
        colors = data[:, 3:6].astype(np.float64)
        if np.max(colors) > 1.0: colors /= 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def save_point_cloud(cloud, file_path: str) -> bool:
    """Saves a PointCloud object to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    try:
        import open3d as o3d
        if isinstance(cloud, o3d.geometry.PointCloud):
            return o3d.io.write_point_cloud(file_path, cloud)
    except ImportError:
        pass
    
    if hasattr(cloud, 'points'):
        pts = np.asarray(cloud.points)
        cols = np.asarray(cloud.colors) if hasattr(cloud, 'colors') and cloud.colors is not None else None
        with open(file_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if cols is not None and len(cols) == len(pts):
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(len(pts)):
                if cols is not None and len(cols) == len(pts):
                    f.write(f"{pts[i][0]:.6f} {pts[i][1]:.6f} {pts[i][2]:.6f} {int(cols[i][0])} {int(cols[i][1])} {int(cols[i][2])}\n")
                else:
                    f.write(f"{pts[i][0]:.6f} {pts[i][1]:.6f} {pts[i][2]:.6f}\n")
        return True
    return False
