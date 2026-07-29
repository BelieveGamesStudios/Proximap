"""
Point Cloud I/O Module for Proximap
Handles loading, saving, and validating point clouds in PLY, LAS/LAZ, and XYZ formats.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable

@dataclass
class LoadResult:
    cloud: Optional[object]  # open3d.geometry.PointCloud
    point_count: int = 0
    has_colors: bool = False
    has_normals: bool = False
    bbox_diagonal: float = 0.0
    warnings: List[str] = field(default_factory=list)
    success: bool = False


def load_point_cloud(file_path: str, log_fn: Optional[Callable[[str], None]] = None) -> LoadResult:
    """
    Loads a point cloud from PLY, LAS, LAZ, or XYZ files into an Open3D PointCloud object.
    
    Args:
        file_path: Absolute path to the point cloud file.
        log_fn: Optional logging callback.
        
    Returns:
        LoadResult containing the Open3D PointCloud and metadata.
    """
    def log(msg: str):
        if log_fn:
            log_fn(msg)

    if not os.path.exists(file_path):
        return LoadResult(cloud=None, warnings=[f"File not found: {file_path}"], success=False)

    ext = os.path.splitext(file_path)[1].lower()
    log(f"[REF_CLOUD] Loading point cloud: {os.path.basename(file_path)} ({ext})")

    try:
        import open3d as o3d
    except ImportError:
        return LoadResult(cloud=None, warnings=["Open3D is not installed in the Python environment."], success=False)

    cloud = o3d.geometry.PointCloud()
    warnings = []

    try:
        if ext in ['.las', '.laz']:
            cloud = _load_las_laz(file_path, warnings, log)
        elif ext in ['.ply', '.xyz', '.pts', '.txt']:
            cloud = o3d.io.read_point_cloud(file_path)
            if len(cloud.points) == 0 and ext in ['.xyz', '.pts', '.txt']:
                # Fallback numpy loader for plain text xyz
                cloud = _load_xyz_numpy(file_path, warnings, log)
        else:
            warnings.append(f"Unsupported file extension: {ext}")
            return LoadResult(cloud=None, warnings=warnings, success=False)

    except Exception as e:
        warnings.append(f"Failed to parse point cloud file: {e}")
        return LoadResult(cloud=None, warnings=warnings, success=False)

    if cloud is None or len(cloud.points) == 0:
        warnings.append("Point cloud file contains no points or failed to read.")
        return LoadResult(cloud=None, warnings=warnings, success=False)

    # Clean non-finite values (NaN / Inf)
    cloud = cloud.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    pt_count = len(cloud.points)

    if pt_count < 100:
        warnings.append(f"Point cloud contains too few points ({pt_count} points). Minimum 100 points required.")
        return LoadResult(cloud=cloud, point_count=pt_count, warnings=warnings, success=False)

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


def peek_has_colors(file_path: str) -> bool:
    """
    Quickly checks whether a point cloud file contains vertex color data
    by reading only the file header — NOT loading the full point cloud.
    Supports PLY (binary & ASCII) and LAS/LAZ headers.
    Falls back to True (assume colors present) for unsupported/unknown formats.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.ply':
            # Read only the PLY ASCII header (max 4 KB) and look for RGB property declarations
            with open(file_path, 'rb') as f:
                header = b''
                while len(header) < 4096:
                    line = f.readline()
                    if not line:
                        break
                    header += line
                    if b'end_header' in line:
                        break
            header_str = header.decode('utf-8', errors='ignore').lower()
            return any(
                f'property {dtype} {ch}' in header_str
                for dtype in ('uchar', 'uint8', 'float', 'float32')
                for ch in ('red', 'green', 'blue', 'r', 'g', 'b')
            )
        elif ext in ('.las', '.laz'):
            # laspy reads just the header without loading point data
            try:
                import laspy
                las = laspy.read(file_path)
                fmt_id = int(getattr(las.header.point_format, 'id', 0))
                return fmt_id in (2, 3, 5, 7, 8, 10)
            except Exception:
                return True  # assume colors on failure
        else:
            # XYZ / PTS / TXT: read first non-comment data line, check column count >= 6
            with open(file_path, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        return len(line.split()) >= 6  # X Y Z R G B
            return False
    except Exception:
        return True  # safe fallback — show toggle, let user decide


def _load_las_laz(file_path: str, warnings: List[str], log: Callable[[str], None]):

    import open3d as o3d
    try:
        import laspy
    except ImportError:
        raise RuntimeError("laspy package is required for loading .las / .laz files.")

    las = laspy.read(file_path)
    # las.x, las.y, las.z give scaled float coordinates
    coords = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(coords)

    # Check for color attributes
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = np.asarray(las.red, dtype=np.float64)
        g = np.asarray(las.green, dtype=np.float64)
        b = np.asarray(las.blue, dtype=np.float64)

        # Scale 16-bit or 8-bit color channels to 0.0 - 1.0
        max_val = max(np.max(r), np.max(g), np.max(b))
        scale = 65535.0 if max_val > 255.0 else 255.0
        if scale > 0:
            colors = np.vstack((r / scale, g / scale, b / scale)).transpose()
            cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


def _load_xyz_numpy(file_path: str, warnings: List[str], log: Callable[[str], None]):
    import open3d as o3d
    data = np.loadtxt(file_path)
    if data.shape[1] < 3:
        raise ValueError("XYZ file must have at least 3 columns (X, Y, Z)")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))

    if data.shape[1] >= 6:
        colors = data[:, 3:6].astype(np.float64)
        if np.max(colors) > 1.0:
            colors /= 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


def save_point_cloud(cloud, file_path: str) -> bool:
    """Saves an Open3D PointCloud object to disk."""
    import open3d as o3d
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    return o3d.io.write_point_cloud(file_path, cloud)
