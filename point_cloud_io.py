"""
Point Cloud I/O Module for Proximap
Handles loading, saving, and validating point clouds in PLY format.
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


def get_photogrammetry_flip_matrix() -> np.ndarray:
    """Returns the 3x3 diagonal matrix F = diag(1, -1, -1) for 180 deg rotation about X axis."""
    return np.diag([1.0, -1.0, -1.0])


def apply_photogrammetry_coordinate_flip(
    points: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    camera_R: Optional[np.ndarray] = None,
    camera_T: Optional[np.ndarray] = None
):
    """
    Applies the 180 deg X-axis rotation (Y -> -Y, Z -> -Z) to convert OpenCV/COLMAP/OpenMVS
    coordinate space (Y-down) to standard graphics viewport space (Y-up).

    Transforms:
      - points: P' = P @ F  (P[:, 1] *= -1, P[:, 2] *= -1)
      - normals: N' = N @ F (N[:, 1] *= -1, N[:, 2] *= -1)
      - camera_R: R' = F @ R @ F^T
      - camera_T: T' = F @ T

    Since F = F^T = F^-1, this function is self-inverting (applying it twice returns to original).
    """
    F = get_photogrammetry_flip_matrix()

    ret_points = None
    if points is not None and len(points) > 0:
        pts = np.asarray(points).copy()
        pts[:, 1] *= -1.0
        pts[:, 2] *= -1.0
        ret_points = pts

    ret_normals = None
    if normals is not None and len(normals) > 0:
        nrm = np.asarray(normals).copy()
        nrm[:, 1] *= -1.0
        nrm[:, 2] *= -1.0
        ret_normals = nrm

    ret_R = None
    if camera_R is not None:
        ret_R = F @ camera_R @ F

    ret_T = None
    if camera_T is not None:
        ret_T = F @ camera_T

    return ret_points, ret_normals, ret_R, ret_T



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
    Loads a point cloud from a PLY file into an Open3D PointCloud
    or SimplePointCloud object.
    """
    def log(msg: str):
        if log_fn:
            log_fn(msg)

    if not os.path.exists(file_path):
        return LoadResult(cloud=None, warnings=[f"File not found: {file_path}"], success=False)

    ext = os.path.splitext(file_path)[1].lower()
    if ext != '.ply':
        return LoadResult(cloud=None, warnings=[f"Unsupported point cloud format '{ext}'. Only .ply format is supported."], success=False)

    log(f"[REF_CLOUD] Loading point cloud: {os.path.basename(file_path)} ({ext})")
    warnings = []

    # 1. Try Open3D if available
    try:
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(file_path)
        
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
    Quickly checks whether a PLY point cloud file contains vertex color data
    by reading only the file header — NOT loading the full point cloud.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext != '.ply':
        return False
    try:
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
    except Exception:
        return True


def _load_numpy_fallback(file_path: str, warnings: List[str], log: Callable[[str], None]) -> Optional[SimplePointCloud]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.ply':
        pts, colors, _ = _read_ply_pure_numpy(file_path)
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
        type_char_map = {
            'char': 'b', 'uchar': 'B', 'short': 'h', 'ushort': 'H',
            'int': 'i', 'uint': 'I', 'float': 'f', 'double': 'd',
            'int8': 'b', 'uint8': 'B', 'int16': 'h', 'uint16': 'H',
            'int32': 'i', 'uint32': 'I', 'float32': 'f', 'float64': 'd'
        }
        type_sizes = {'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'f': 4, 'd': 8}
        has_list = any(p[2] for p in vertex_properties)

        points = np.zeros((num_vertices, 3), dtype=np.float32)
        prop_names = [p[0] for p in vertex_properties]
        has_color = all(c in prop_names for c in ('red', 'green', 'blue')) or all(c in prop_names for c in ('r', 'g', 'b'))
        colors = np.zeros((num_vertices, 3), dtype=np.uint8) if has_color else None

        if 'binary' in (format_type or ''):
            if has_list:
                fixed_properties = []
                list_properties = []
                for p in vertex_properties:
                    if p[2]: list_properties.append(p)
                    else:
                        if not list_properties: fixed_properties.append(p)

                fmt_chars = [type_char_map[t] for name, t, _, _, _ in fixed_properties if t in type_char_map]
                fixed_size = sum(type_sizes[c] for c in fmt_chars)
                endian_flag = '>' if 'big' in (format_type or '') else '<'
                fixed_struct = struct.Struct(endian_flag + ''.join(fmt_chars))

                names = [p[0] for p in fixed_properties]
                x_idx = names.index('x') if 'x' in names else -1
                y_idx = names.index('y') if 'y' in names else -1
                z_idx = names.index('z') if 'z' in names else -1

                r_name = 'red' if 'red' in names else ('r' if 'r' in names else None)
                g_name = 'green' if 'green' in names else ('g' if 'g' in names else None)
                b_name = 'blue' if 'blue' in names else ('b' if 'b' in names else None)

                r_idx = names.index(r_name) if r_name else -1
                g_idx = names.index(g_name) if g_name else -1
                b_idx = names.index(b_name) if b_name else -1

                data = f.read()
                offset = 0

                for i in range(num_vertices):
                    val = fixed_struct.unpack_from(data, offset)
                    if x_idx != -1: points[i, 0] = val[x_idx]
                    if y_idx != -1: points[i, 1] = val[y_idx]
                    if z_idx != -1: points[i, 2] = val[z_idx]

                    if has_color and colors is not None:
                        if r_idx != -1: colors[i, 0] = val[r_idx]
                        if g_idx != -1: colors[i, 1] = val[g_idx]
                        if b_idx != -1: colors[i, 2] = val[b_idx]

                    offset += fixed_size

                    for name, _, _, count_type, item_type in list_properties:
                        c_char = type_char_map[count_type]
                        c_size = type_sizes[c_char]
                        count = struct.unpack_from(endian_flag + c_char, data, offset)[0]
                        offset += c_size

                        i_char = type_char_map[item_type]
                        i_size = type_sizes[i_char]
                        offset += count * i_size
            else:
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
