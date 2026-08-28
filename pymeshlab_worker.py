#!/usr/bin/env python3
"""
pymeshlab_worker.py — Standalone PyMeshLab mesh processing worker for Proximap.

This script is invoked as a subprocess by mesh_cleanup.py (via PyMeshLabWorkerBackend)
using a bundled or system Python 3.10 interpreter that can load the pymeshlab cp310
extension module.

Protocol:
  argv[1]: JSON-encoded params dict with keys:
    - action               (str)  : "cleanup" (default), "merge_by_distance", "smooth_taubin"
    - input_ply            (str)  : path to input .ply
    - output_ply           (str)  : path to output .ply
    - target_reduction_pct (float, default 50)  [action: cleanup]
    - remove_duplicates     (bool,  default True) [action: cleanup]
    - repair_nonmanifold    (bool,  default True) [action: cleanup]
    - close_holes           (bool,  default True) [action: cleanup]
    - max_hole_size         (int,   default 30)   [action: cleanup]
    - threshold_pct        (float, default 1.0)  [action: merge_by_distance]
    - bbox_diagonal        (float, default 0.0)  [action: merge_by_distance]
    - lambda_factor        (float, default 0.5)  [action: smooth_taubin]
    - iterations           (int,   default 10)   [action: smooth_taubin]

  stdout: newline-separated JSON objects, each with:
    { "log": "<message>" }          -- progress messages
    { "result": true|false }        -- final result (last line)

Exit code: 0 on success, 1 on failure.
"""

import sys
import os
import json
import math


def _find_pymeshlab_dir():
    """Find the pymeshlab_extracted directory relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "backend_bin", "pymeshlab_extracted"),
        os.path.join(script_dir, "..", "backend_bin", "pymeshlab_extracted"),
        os.path.join(script_dir, "pymeshlab_extracted"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "pymeshlab")):
            return os.path.abspath(c)
    return None


def _emit(obj):
    print(json.dumps(obj), flush=True)


def _log(msg):
    _emit({"log": msg})


def _get_taubin_filter(ms, pymeshlab_module):
    """
    Locate the Taubin smoothing filter function on the MeshSet instance.
    Checks apply_coord_taubin_smoothing and meshing_apply_coord_taubin_smoothing.
    Raises RuntimeError if neither is present.
    """
    for func_name in ["apply_coord_taubin_smoothing", "meshing_apply_coord_taubin_smoothing"]:
        if hasattr(ms, func_name):
            return getattr(ms, func_name)
    raise RuntimeError(
        "PyMeshLab Taubin smoothing filter (apply_coord_taubin_smoothing) is not available in the worker environment."
    )


def _get_merge_filter(ms, pymeshlab_module):
    """
    Locate the merge close vertices filter function on the MeshSet instance.
    Checks meshing_merge_close_vertices and apply_coord_merge_close_vertices.
    Raises RuntimeError if neither is present.
    """
    for func_name in ["meshing_merge_close_vertices", "apply_coord_merge_close_vertices"]:
        if hasattr(ms, func_name):
            return getattr(ms, func_name)
    raise RuntimeError(
        "PyMeshLab merge close vertices filter (meshing_merge_close_vertices) is not available in the worker environment."
    )


def _calc_bbox_diagonal(mesh):
    """Calculate axis-aligned bounding box diagonal from a PyMeshLab Mesh."""
    try:
        bbox = mesh.bounding_box()
        dim = bbox.dim_x() ** 2 + bbox.dim_y() ** 2 + bbox.dim_z() ** 2
        return math.sqrt(dim)
    except Exception:
        return 1.0


def handle_cleanup(ms, pymeshlab, params, input_ply, output_ply):
    target_reduction_pct = float(params.get("target_reduction_pct", 50))
    remove_duplicates    = bool(params.get("remove_duplicates", True))
    repair_nonmanifold   = bool(params.get("repair_nonmanifold", True))
    close_holes          = bool(params.get("close_holes", True))
    max_hole_size        = int(params.get("max_hole_size", 30))

    _log("[CLEANUP] Starting PyMeshLab Auto Cleanup on: " + os.path.basename(input_ply))

    init_mesh = ms.current_mesh()
    init_v = init_mesh.vertex_number()
    init_f = init_mesh.face_number()
    _log("[CLEANUP] Initial mesh: {:,} vertices, {:,} faces".format(init_v, init_f))

    _log("[CLEANUP] Applying mesh repair filters...")
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

    merge_fn = _get_merge_filter(ms, pymeshlab)
    merge_fn()

    target_perc = max(0.05, min(0.95, (100.0 - target_reduction_pct) / 100.0))
    _log("[CLEANUP] Applying {}% Quadric Edge Collapse Decimation...".format(int(target_reduction_pct)))
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=target_perc,
        qualitythr=0.3,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
    )

    final_mesh = ms.current_mesh()
    final_v = final_mesh.vertex_number()
    final_f = final_mesh.face_number()

    os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
    ms.save_current_mesh(output_ply)

    if os.path.isfile(output_ply) and os.path.getsize(output_ply) > 0:
        pct = ((init_f - final_f) / init_f * 100.0) if init_f > 0 else 0.0
        _log(
            "[CLEANUP] Auto Cleanup complete: {:,} vertices, {:,} faces "
            "({:.1f}% face reduction). Saved to {}".format(
                final_v, final_f, pct, os.path.basename(output_ply)
            )
        )
        _emit({"result": True})
    else:
        _emit({"result": False, "error": "Cleaned output mesh was not created or is empty."})
        sys.exit(1)


def handle_merge_by_distance(ms, pymeshlab, params, input_ply, output_ply):
    threshold_pct = float(params.get("threshold_pct", 1.0))
    bbox_diagonal = float(params.get("bbox_diagonal", 0.0))

    if bbox_diagonal <= 0.0:
        bbox_diagonal = _calc_bbox_diagonal(ms.current_mesh())

    abs_threshold = (threshold_pct / 100.0) * bbox_diagonal
    _log("[MERGE] PyMeshLab Merge by Distance: threshold_pct={:.2f}%, abs_threshold={:.5f}".format(
        threshold_pct, abs_threshold
    ))

    merge_fn = _get_merge_filter(ms, pymeshlab)
    # PyMeshLab meshing_merge_close_vertices takes threshold as absolute distance or percentage
    try:
        merge_fn(threshold=pymeshlab.PureValue(abs_threshold))
    except Exception:
        try:
            merge_fn(threshold=abs_threshold)
        except Exception as e:
            _log("[WARNING] Threshold parameter fallback: " + str(e))
            merge_fn()

    # Clean up degenerate/duplicate/null faces and unreferenced vertices produced by vertex merging
    try:
        ms.meshing_remove_duplicate_faces()
    except Exception:
        pass
    try:
        ms.meshing_remove_null_faces()
    except Exception:
        pass
    try:
        ms.meshing_remove_duplicate_vertices()
    except Exception:
        pass
    try:
        ms.meshing_remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception:
        pass
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception:
        pass

    os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
    ms.save_current_mesh(output_ply)

    if os.path.isfile(output_ply) and os.path.getsize(output_ply) > 0:
        final_mesh = ms.current_mesh()
        _log("[MERGE] Merge by distance complete: {:,} vertices, {:,} faces. Saved to {}".format(
            final_mesh.vertex_number(), final_mesh.face_number(), os.path.basename(output_ply)
        ))
        _emit({"result": True})
    else:
        _emit({"result": False, "error": "Merged output mesh was not created or is empty."})
        sys.exit(1)


def handle_smooth_taubin(ms, pymeshlab, params, input_ply, output_ply):
    lambda_factor = float(params.get("lambda_factor", 0.5))
    mu_factor     = float(params.get("mu_factor", -(lambda_factor + 0.01)))
    iterations    = int(params.get("iterations", 10))

    _log("[SMOOTH] PyMeshLab Taubin Smoothing: lambda={:.2f}, mu={:.2f}, iterations={}".format(
        lambda_factor, mu_factor, iterations
    ))

    taubin_fn = _get_taubin_filter(ms, pymeshlab)
    try:
        taubin_fn(lambda_val=lambda_factor, mu_val=mu_factor, steps=iterations)
    except TypeError:
        try:
            taubin_fn(lambda_val=lambda_factor, steps=iterations)
        except TypeError:
            try:
                taubin_fn(lambda_filter=lambda_factor, iterations=iterations)
            except TypeError:
                taubin_fn()

    os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
    ms.save_current_mesh(output_ply)

    if os.path.isfile(output_ply) and os.path.getsize(output_ply) > 0:
        final_mesh = ms.current_mesh()
        _log("[SMOOTH] Taubin smoothing complete: {:,} vertices, {:,} faces. Saved to {}".format(
            final_mesh.vertex_number(), final_mesh.face_number(), os.path.basename(output_ply)
        ))
        _emit({"result": True})
    else:
        _emit({"result": False, "error": "Smoothed output mesh was not created or is empty."})
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        _emit({"result": False, "error": "No params JSON argument provided."})
        sys.exit(1)

    try:
        params = json.loads(sys.argv[1])
    except Exception as e:
        _emit({"result": False, "error": "Failed to parse params JSON: " + str(e)})
        sys.exit(1)

    action     = params.get("action", "cleanup")
    input_ply  = params.get("input_ply", "")
    output_ply = params.get("output_ply", "")

    if not input_ply or not os.path.isfile(input_ply):
        _emit({"result": False, "error": "Input file not found: " + input_ply})
        sys.exit(1)

    # Inject pymeshlab path
    ml_dir = _find_pymeshlab_dir()
    if ml_dir and ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
        lib_dir = os.path.join(ml_dir, "pymeshlab", "lib")
        if os.path.isdir(lib_dir):
            old_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + old_ld if old_ld else "")

    try:
        import pymeshlab
        _log("[WORKER] PyMeshLab " + pymeshlab.__version__ + " loaded successfully.")
    except ImportError as e:
        _emit({"result": False, "error": "Cannot import pymeshlab: " + str(e)})
        sys.exit(1)

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_ply)

        if action == "cleanup":
            handle_cleanup(ms, pymeshlab, params, input_ply, output_ply)
        elif action == "merge_by_distance":
            handle_merge_by_distance(ms, pymeshlab, params, input_ply, output_ply)
        elif action == "smooth_taubin":
            handle_smooth_taubin(ms, pymeshlab, params, input_ply, output_ply)
        else:
            _emit({"result": False, "error": "Unknown worker action: " + str(action)})
            sys.exit(1)

    except Exception as e:
        _emit({"result": False, "error": "PyMeshLab worker exception: " + str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
