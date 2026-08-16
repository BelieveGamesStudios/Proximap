"""
mesh_cleanup.py — PyMeshLab Mesh Repair and Decimation Module for Proximap

This module provides automated mesh cleanup and decimation (Step 8.5) in the Proximap
photogrammetry pipeline using PyMeshLab.

Target Environment Note:
- Production packaging and standalone executable builds (PyInstaller) across all supported
  platforms (Windows, Linux, macOS) are pinned to Python 3.10 (CPython 3.10).
- The pre-bundled wheel binaries in `backend_bin/PymeshLab/` target Python 3.10 (`cp310`).
- If running in a local development environment with a different Python runtime (e.g. Python 3.14),
  this module detects the ABI mismatch and gracefully logs a warning while allowing the pipeline
  to continue without breaking.
"""

import os
import sys
import glob
import subprocess
import logging

logger = logging.getLogger(__name__)

def _log(msg: str, callback=None):
    if callback:
        try:
            callback(msg)
        except Exception:
            pass
    logger.info(msg)

def ensure_pymeshlab_installed(log_callback=None):
    """
    Attempts to import pymeshlab. In frozen packaged apps (PyInstaller), PyMeshLab is pre-bundled.
    In dev mode, if missing, attempts installation from backend_bin/PymeshLab/ wheel, falling back to PyPI.

    Returns:
        tuple: (success: bool, pymeshlab_module or None)
    """
    try:
        import pymeshlab
        return True, pymeshlab
    except ImportError:
        pass

    is_frozen = getattr(sys, 'frozen', False)
    if is_frozen:
        _log("[WARNING] PyMeshLab module missing from packaged app bundle.", log_callback)
        return False, None

    # --- Dev Environment Auto-Install ---
    py_major, py_minor = sys.version_info.major, sys.version_info.minor

    # If Python 3.10, try local offline wheel first
    if (py_major, py_minor) == (3, 10):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_dirs = [
            os.path.join(base_dir, "backend_bin", "PymeshLab"),
            os.path.join(base_dir, "backend_bin", "pymeshlab"),
            os.path.join(os.path.dirname(base_dir), "backend_bin", "PymeshLab"),
        ]

        wheel_dir = None
        for p in possible_dirs:
            if os.path.isdir(p):
                wheel_dir = p
                break

        if wheel_dir:
            platform_key = "win_amd64" if sys.platform.startswith("win") else ("macosx" if sys.platform == "darwin" else "manylinux")
            wheels = glob.glob(os.path.join(wheel_dir, "*.whl"))
            target_wheel = None
            for w in wheels:
                if platform_key in w.lower():
                    target_wheel = w
                    break

            if not target_wheel and wheels:
                target_wheel = wheels[0]

            if target_wheel:
                _log(f"[CLEANUP] Auto-installing PyMeshLab from offline wheel {os.path.basename(target_wheel)}...", log_callback)
                try:
                    cmd = [sys.executable, "-m", "pip", "install", target_wheel, "--no-index", "--find-links", wheel_dir]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if res.returncode == 0:
                        import pymeshlab
                        _log("[CLEANUP] PyMeshLab installed successfully from offline wheel.", log_callback)
                        return True, pymeshlab
                except Exception as e:
                    _log(f"[WARNING] Exception installing offline wheel: {e}", log_callback)

    # PyMeshLab only publishes wheels for Python 3.10 (cp310).
    # Attempting PyPI for any other version will hang or fail — skip it entirely.
    _log(
        f"[INFO] PyMeshLab wheels require Python 3.10 (current runtime: Python {py_major}.{py_minor}). "
        f"Mesh cleanup will be skipped in this dev environment. "
        f"Production builds (PyInstaller/Python 3.10) include PyMeshLab automatically.",
        log_callback
    )
    if False:  # pragma: no cover
        import pymeshlab
        return True, pymeshlab
    return False, None


def _run_fallback_mesh_cleanup(input_ply_path: str, output_ply_path: str, log_callback=None, cleanup_params: dict = None) -> bool:
    """
    Fallback mesh cleanup & decimation engine using Open3D or Trimesh when PyMeshLab is unavailable
    (e.g., in Python 3.11+ / Python 3.14 dev environments).
    """
    if cleanup_params is None:
        cleanup_params = {}

    target_reduction_pct = float(cleanup_params.get("target_reduction_pct", 50))
    remove_duplicates = bool(cleanup_params.get("remove_duplicates", True))

    # 1. Try Open3D engine first
    try:
        import open3d as o3d
        _log(f"[CLEANUP] Open3D Fallback Engine: Processing {os.path.basename(input_ply_path)}...", log_callback)
        mesh = o3d.io.read_triangle_mesh(input_ply_path)
        if not mesh.has_vertices() or len(mesh.vertices) == 0:
            _log("[WARNING] Open3D failed to load vertices from mesh.", log_callback)
            return False

        init_v = len(mesh.vertices)
        init_f = len(mesh.triangles)
        _log(f"[CLEANUP] Initial mesh: {init_v:,} vertices, {init_f:,} faces", log_callback)

        if remove_duplicates:
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()

        if init_f > 0:
            target_perc = max(0.05, min(0.95, (100.0 - target_reduction_pct) / 100.0))
            target_triangles = max(10, int(init_f * target_perc))
            _log(f"[CLEANUP] Applying {int(target_reduction_pct)}% Quadric Edge Collapse Decimation (target {target_triangles:,} faces)...", log_callback)
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
        o3d.io.write_triangle_mesh(output_ply_path, mesh)

        if os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0:
            final_v = len(mesh.vertices)
            final_f = len(mesh.triangles)
            pct = ((init_f - final_f) / init_f * 100.0) if init_f > 0 else 0.0
            _log(
                f"[CLEANUP] Auto Cleanup complete (Open3D): {final_v:,} vertices, {final_f:,} faces "
                f"({pct:.1f}% face reduction). Saved to {os.path.basename(output_ply_path)}",
                log_callback
            )
            return True
    except Exception as e:
        _log(f"[WARNING] Open3D fallback cleanup failed: {e}. Trying Trimesh fallback...", log_callback)

    # 2. Try Trimesh engine second
    try:
        import trimesh
        _log(f"[CLEANUP] Trimesh Fallback Engine: Processing {os.path.basename(input_ply_path)}...", log_callback)
        mesh = trimesh.load(input_ply_path, force="mesh")
        init_v = len(mesh.vertices)
        init_f = len(mesh.faces)
        _log(f"[CLEANUP] Initial mesh: {init_v:,} vertices, {init_f:,} faces", log_callback)

        if remove_duplicates:
            mesh.merge_vertices()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()

        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
        mesh.export(output_ply_path, file_type="ply")

        if os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0:
            _log(
                f"[CLEANUP] Auto Cleanup complete (Trimesh): {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces. "
                f"Saved to {os.path.basename(output_ply_path)}",
                log_callback
            )
            return True
    except Exception as e:
        _log(f"[WARNING] Trimesh fallback cleanup failed: {e}", log_callback)

    return False


def run_mesh_cleanup(input_ply_path: str, output_ply_path: str, log_callback=None, cleanup_params: dict = None) -> bool:
    """
    Executes automated mesh repair and decimation on input PLY file based on optional cleanup parameters.

    Args:
        input_ply_path (str): Path to input mesh PLY file.
        output_ply_path (str): Path to save cleaned PLY file.
        log_callback (callable, optional): Function for logging progress messages.
        cleanup_params (dict, optional): Custom parameters dict containing reduction % and filter toggles.

    Returns:
        bool: True if cleanup succeeded and output_ply_path was created; False otherwise.
    """
    if not os.path.isfile(input_ply_path):
        _log(f"[WARNING] Auto Cleanup input file not found: {input_ply_path}", log_callback)
        return False

    success, ml = ensure_pymeshlab_installed(log_callback)
    if not success or ml is None:
        _log("[INFO] PyMeshLab unavailable. Falling back to Open3D/Trimesh mesh cleanup engine...", log_callback)
        return _run_fallback_mesh_cleanup(input_ply_path, output_ply_path, log_callback, cleanup_params)

    if cleanup_params is None:
        cleanup_params = {}

    target_reduction_pct = float(cleanup_params.get("target_reduction_pct", 50))
    remove_duplicates = bool(cleanup_params.get("remove_duplicates", True))
    repair_nonmanifold = bool(cleanup_params.get("repair_nonmanifold", True))
    close_holes = bool(cleanup_params.get("close_holes", True))
    max_hole_size = int(cleanup_params.get("max_hole_size", 30))

    _log(f"[CLEANUP] Starting Auto Cleanup on: {os.path.basename(input_ply_path)}", log_callback)
    try:
        ms = ml.MeshSet()
        ms.load_new_mesh(input_ply_path)

        init_mesh = ms.current_mesh()
        init_v = init_mesh.vertex_number()
        init_f = init_mesh.face_number()
        _log(f"[CLEANUP] Initial mesh: {init_v:,} vertices, {init_f:,} faces", log_callback)

        _log("[CLEANUP] Applying mesh repair filters...", log_callback)
        if remove_duplicates:
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_null_faces()
        if repair_nonmanifold:
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
        if close_holes and max_hole_size > 0:
            ms.meshing_close_holes(maxholesize=max_hole_size)

        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=25)
        ms.meshing_re_orient_faces_coherentely()
        ms.meshing_merge_close_vertices()

        target_perc = max(0.05, min(0.95, (100.0 - target_reduction_pct) / 100.0))
        _log(f"[CLEANUP] Applying {int(target_reduction_pct)}% Quadric Edge Collapse Decimation...", log_callback)
        ms.meshing_decimation_quadric_edge_collapse(
            targetperc=target_perc,
            qualitythr=0.3,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True
        )

        final_mesh = ms.current_mesh()
        final_v = final_mesh.vertex_number()
        final_f = final_mesh.face_number()

        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
        ms.save_current_mesh(output_ply_path)

        if os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0:
            pct = ((init_f - final_f) / init_f * 100.0) if init_f > 0 else 0.0
            _log(
                f"[CLEANUP] Auto Cleanup complete: {final_v:,} vertices, {final_f:,} faces "
                f"({pct:.1f}% face reduction). Saved to {os.path.basename(output_ply_path)}",
                log_callback
            )
            return True
        else:
            _log("[WARNING] Cleaned output mesh file was not created or empty.", log_callback)
            return False

    except Exception as e:
        _log(f"[WARNING] Error during PyMeshLab mesh cleanup: {e}", log_callback)
        return False
