#!/usr/bin/env python3
"""
pymeshlab_worker.py — Standalone PyMeshLab mesh processing worker for Proximap.

This script is invoked as a subprocess by mesh_cleanup.py (via PyMeshLabWorkerBackend)
using a bundled or system Python 3.10 interpreter that can load the pymeshlab cp310
extension module.

Protocol:
  argv[1]: JSON-encoded params dict with keys:
    - action               (str)  : "cleanup" (default), "merge_by_distance", "smooth_taubin",
                                    "align_point_clouds", "fuse_point_clouds", "screened_poisson"
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
    - poisson_depth        (int,   default 8)    [action: screened_poisson]
    - poisson_scale        (float, default 1.05) [action: screened_poisson]
    - normal_neighbors     (int,   default 30)   [action: screened_poisson]
    - min_component_faces  (int,   default 25)   [action: screened_poisson]
    - samples_per_node     (float, default 3.0)  [action: screened_poisson]
    - support_distance     (float, default 0.0)  [action: screened_poisson]

  stdout: newline-separated JSON objects, each with:
    { "log": "<message>" }          -- progress messages
    { "result": true|false }        -- final result (last line)

Exit code: 0 on success, 1 on failure.
"""

import sys
import os
import json
import math
import numpy as np


def _json_safe(value):
    """Convert PyMeshLab/NumPy return values into protocol-safe JSON data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    full_cleanup         = bool(params.get("full_cleanup", True))

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

    if full_cleanup:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=25)
        ms.meshing_re_orient_faces_coherentely()
        merge_fn = _get_merge_filter(ms, pymeshlab)
        merge_fn()

    if target_reduction_pct > 0:
        target_perc = max(0.05, min(0.95, (100.0 - target_reduction_pct) / 100.0))
        _log("[CLEANUP] Applying {}% Quadric Edge Collapse Decimation...".format(int(target_reduction_pct)))
        ms.meshing_decimation_quadric_edge_collapse(
            targetperc=target_perc, qualitythr=0.3, preserveboundary=True,
            preservenormal=True, preservetopology=True,
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


def handle_screened_poisson(ms, pymeshlab, params, input_ply, output_ply):
    """Reconstruct an oriented point cloud with MeshLab's Screened Poisson filter."""
    depth = max(5, min(12, int(params.get("poisson_depth", 8))))
    scale = max(1.001, float(params.get("poisson_scale", 1.05)))
    normal_neighbors = max(3, int(params.get("normal_neighbors", 30)))
    min_component_faces = max(1, int(params.get("min_component_faces", 25)))
    samples_per_node = max(0.1, float(params.get("samples_per_node", 3.0)))
    support_distance = max(0.0, float(params.get("support_distance", 0.0)))

    source = ms.current_mesh()
    source_mesh_id = ms.current_mesh_id()
    if source.vertex_number() < 3:
        raise RuntimeError("Screened Poisson requires at least three input points.")
    _log("[POISSON] PyMeshLab Screened Poisson: {:,} points, depth={}, scale={:.3f}, samples/node={:.1f}, support={:.6f}".format(
        source.vertex_number(), depth, scale, samples_per_node, support_distance
    ))

    # Proximap normally supplies component-oriented normals. Re-estimate only
    # when the input PLY genuinely has none, so the sidecar does not destroy the
    # orientation established by the evidence preprocessing stage.
    has_normals = False
    try:
        normals = source.vertex_normal_matrix()
        has_normals = bool(len(normals) == source.vertex_number() and
                           any(float(x * x + y * y + z * z) > 1e-12 for x, y, z in normals))
    except Exception:
        pass
    if not has_normals:
        _log("[POISSON] Input has no normals; estimating point-cloud normals in PyMeshLab.")
        ms.compute_normal_for_point_clouds(k=normal_neighbors, smoothiter=0)

    poisson = getattr(ms, "generate_surface_reconstruction_screened_poisson", None)
    if poisson is None:
        raise RuntimeError("PyMeshLab Screened Poisson filter is unavailable.")
    try:
        poisson(depth=depth, scale=scale, samplespernode=samples_per_node, preclean=True)
    except TypeError:
        try:
            poisson(depth=depth, scale=scale, samplespernode=samples_per_node)
        except TypeError:
            poisson(depth=depth)

    result_mesh_id = ms.current_mesh_id()
    unsupported_vertices = 0
    if support_distance > 0:
        try:
            ms.compute_scalar_by_distance_from_point_cloud_per_vertex(
                coloredmesh=result_mesh_id, vertexmesh=source_mesh_id,
                radius=support_distance, sampleradius=False, approximategeodetic=False,
            )
            ms.set_current_mesh(result_mesh_id)
            quality = ms.current_mesh().vertex_scalar_array()
            unsupported_vertices = int(sum(1 for value in quality if value > support_distance))
            if unsupported_vertices:
                ms.compute_selection_by_condition_per_vertex(
                    condselect="q > {:.12g}".format(support_distance)
                )
                ms.meshing_remove_selected_vertices()
                _log("[POISSON] Removed {:,} unsupported vertices farther than {:.6f} from input samples.".format(
                    unsupported_vertices, support_distance
                ))
        except Exception as error:
            _log("[POISSON] Support trimming skipped: " + str(error))

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    try:
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=min_component_faces, removeunref=True
        )
    except TypeError:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_component_faces)
        ms.meshing_remove_unreferenced_vertices()
    # Only close tiny cleanup holes. Large automatic caps recreate the sheets
    # and spikes removed by support trimming.
    try:
        ms.meshing_close_holes(
            maxholesize=max(3, min(30, int(params.get("max_hole_size", 30)))),
            selected=False, selfintersection=False, refinehole=False,
        )
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_remove_unreferenced_vertices()
    except Exception as error:
        _log("[POISSON] Conservative hole closure skipped: " + str(error))
    try:
        ms.meshing_re_orient_faces_coherentely()
    except Exception:
        try:
            ms.meshing_re_orient_faces_coherently()
        except Exception:
            pass
    try:
        topology = _json_safe(ms.get_topological_measures())
    except Exception:
        topology = {}

    result = ms.current_mesh()
    if result.vertex_number() == 0 or result.face_number() == 0:
        raise RuntimeError("PyMeshLab Screened Poisson produced an empty mesh.")
    os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
    ms.save_current_mesh(output_ply, save_vertex_normal=True)
    if not os.path.isfile(output_ply) or os.path.getsize(output_ply) == 0:
        raise RuntimeError("Screened Poisson output mesh was not created.")
    _log("[POISSON] Reconstruction complete: {:,} vertices, {:,} faces.".format(
        result.vertex_number(), result.face_number()
    ))
    _emit({
        "result": True,
        "backend": "pymeshlab-screened-poisson",
        "vertex_count": result.vertex_number(),
        "face_count": result.face_number(),
        "unsupported_vertex_count": unsupported_vertices,
        "support_distance": support_distance,
        "topology": topology,
    })


def handle_align_point_clouds(ms, pymeshlab, params):
    """Align multiple scan layers using MeshLab global alignment/ICP only."""
    inputs = [os.path.abspath(path) for path in params.get("input_plys", [])]
    output_dir = os.path.abspath(params.get("output_dir", ""))
    reference_index = int(params.get("reference_index", 0))
    voxel_size = max(float(params.get("voxel_size", 0.03)), 1e-6)
    quality_distance = voxel_size * max(float(params.get("quality_distance_multiplier", 2.5)), 1.0)
    if len(inputs) < 2 or not output_dir:
        raise RuntimeError("PyMeshLab alignment requires at least two inputs and an output directory.")
    if not 0 <= reference_index < len(inputs):
        raise RuntimeError("Invalid PyMeshLab reference scan index.")
    os.makedirs(output_dir, exist_ok=True)

    mesh_ids = []
    for path in inputs:
        if not os.path.isfile(path):
            raise RuntimeError("Alignment input not found: " + path)
        ms.load_new_mesh(path)
        mesh_ids.append(ms.current_mesh_id())
        try:
            ms.compute_normal_for_point_clouds(k=30, smoothiter=0)
        except Exception:
            pass
    _log("[ALIGN] Loaded {} point-cloud layers; reference layer {}.".format(len(mesh_ids), reference_index + 1))

    # MeshLab's global-alignment filter can legitimately no-op on vertex-only
    # layers. Run explicit source-to-reference ICP for every scan so successful
    # completion always corresponds to an applied PyMeshLab transform.
    pairwise = []
    for index, mesh_id in enumerate(mesh_ids):
        if index == reference_index:
            pairwise.append({"reference": True})
            continue
        available = min(
            ms.mesh(mesh_ids[reference_index]).vertex_number(),
            ms.mesh(mesh_id).vertex_number(),
        )
        sample_count = min(
            max(50, available - 1),
            min(12000, max(2000, int(params.get("sample_count", 12000)))),
        )
        result = ms.compute_matrix_by_icp_between_meshes(
            referencemesh=mesh_ids[reference_index],
            sourcemesh=mesh_id,
            samplenum=sample_count,
            mindistabs=max(voxel_size * 8.0, 0.05),
            trgdistabs=max(voxel_size * 0.5, 0.002),
            maxiternum=100,
            samplemode=False,
            reducefactorperc=0.80,
            passhifilter=0.75,
            matchmode=True,
        )
        pairwise.append(_json_safe(result))
    alignment_details = {"pairwise_icp": pairwise}

    outputs, transforms = [], []
    for index, mesh_id in enumerate(mesh_ids):
        ms.set_current_mesh(mesh_id)
        transform = _json_safe(ms.current_mesh().transform_matrix())
        transforms.append(transform)
        ms.apply_matrix_freeze(alllayers=False)
        output = os.path.join(output_dir, "aligned_{:02d}.ply".format(index + 1))
        ms.save_current_mesh(output)
        outputs.append(output)
    quality = []
    for index, mesh_id in enumerate(mesh_ids):
        if index == reference_index:
            quality.append({"reference": True, "RMS": 0.0, "overlap_ratio": 1.0})
            continue
        sample_count = min(10000, max(100, ms.mesh(mesh_id).vertex_number()))
        try:
            # Evaluate only geometrically plausible correspondences. Measuring
            # every point penalizes useful, non-overlapping scan coverage and
            # made partial room passes look misregistered even after good ICP.
            diagonal = max(_calc_bbox_diagonal(ms.mesh(mesh_id)), 1e-9)
            max_distance_percent = min(100.0, max(0.01, quality_distance / diagonal * 100.0))
            measure = ms.get_hausdorff_distance(
                sampledmesh=mesh_id,
                targetmesh=mesh_ids[reference_index],
                savesample=False,
                samplevert=True,
                sampleedge=False,
                sampleface=False,
                samplenum=sample_count,
                maxdist=pymeshlab.PercentageValue(max_distance_percent),
            )
            measure = _json_safe(measure)
            inlier_count = int(measure.get("n_samples", 0) or 0)
            measure["requested_samples"] = sample_count
            measure["overlap_ratio"] = min(1.0, inlier_count / max(sample_count, 1))
            measure["distance_threshold"] = quality_distance
            quality.append(measure)
        except Exception as error:
            quality.append({"error": str(error)})
    _emit({
        "result": True,
        "backend": "pymeshlab-pairwise-icp",
        "outputs": outputs,
        "transforms": transforms,
        "details": alignment_details,
        "quality": quality,
    })


def handle_fuse_point_clouds(ms, pymeshlab, params, output_ply):
    """Merge, de-duplicate, reject outliers, and orient normals in PyMeshLab."""
    inputs = [os.path.abspath(path) for path in params.get("input_plys", [])]
    voxel_size = max(float(params.get("voxel_size", 0.03)), 1e-6)
    normal_neighbors = max(3, int(params.get("normal_neighbors", 30)))
    sensor_origins = params.get("sensor_origins") or []
    if len(inputs) < 2:
        raise RuntimeError("PyMeshLab fusion requires at least two aligned scans.")
    mesh_ids = []
    for index, path in enumerate(inputs):
        if not os.path.isfile(path):
            raise RuntimeError("Fusion input not found: " + path)
        ms.load_new_mesh(path)
        mesh_ids.append(ms.current_mesh_id())
        origin = sensor_origins[index] if index < len(sensor_origins) else [0.0, 0.0, 0.0]
        ms.compute_normal_for_point_clouds(
            k=normal_neighbors, smoothiter=1, flipflag=True,
            viewpos=np.asarray(origin[:3], dtype=np.float64),
        )
    input_vertices = sum(ms.mesh(mesh_id).vertex_number() for mesh_id in mesh_ids)
    _log("[FUSION] Merging {} aligned PyMeshLab layers ({:,} input points).".format(len(inputs), input_vertices))
    ms.generate_by_merging_visible_meshes(
        mergevisible=True, deletelayer=True, mergevertices=True, alsounreferenced=True
    )
    rejected_output = params.get("rejected_output_ply", "")
    try:
        merged_mesh_id = ms.current_mesh_id()
        ms.compute_selection_point_cloud_outliers(
            propthreshold=float(params.get("outlier_probability", 0.80)),
            knearest=max(8, int(params.get("outlier_neighbors", 32))),
        )
        if rejected_output:
            try:
                ms.generate_from_selected_vertices(deleteoriginal=False)
                os.makedirs(os.path.dirname(os.path.abspath(rejected_output)), exist_ok=True)
                ms.save_current_mesh(rejected_output)
                ms.set_current_mesh(merged_mesh_id)
            except Exception as export_error:
                ms.set_current_mesh(merged_mesh_id)
                _log("[FUSION] Could not export selected outliers: " + str(export_error))
        ms.meshing_remove_selected_vertices()
    except Exception as error:
        _log("[FUSION] PyMeshLab outlier filter skipped: " + str(error))
    try:
        ms.meshing_merge_close_vertices(threshold=pymeshlab.PureValue(voxel_size * 0.35))
    except Exception:
        ms.meshing_merge_close_vertices()
    try:
        ms.apply_normal_point_cloud_smoothing(k=max(3, normal_neighbors // 3), usedist=True)
    except Exception:
        pass
    try:
        ms.apply_normal_normalization_per_vertex()
    except Exception:
        pass
    fused = ms.current_mesh()
    os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
    ms.save_current_mesh(output_ply, save_vertex_normal=True)
    if fused.vertex_number() == 0 or not os.path.isfile(output_ply):
        raise RuntimeError("PyMeshLab fusion produced an empty point cloud.")
    _emit({
        "result": True,
        "backend": "pymeshlab-point-fusion",
        "input_point_count": input_vertices,
        "fused_point_count": fused.vertex_number(),
        "rejected_point_count": max(0, input_vertices - fused.vertex_number()),
        "rejected_output": rejected_output if rejected_output and os.path.isfile(rejected_output) else None,
        "output": output_ply,
    })


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

    multi_input_actions = {"align_point_clouds", "fuse_point_clouds"}
    if action not in multi_input_actions and (not input_ply or not os.path.isfile(input_ply)):
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
        version = getattr(pymeshlab, "__version__", None)
        _log("[WORKER] PyMeshLab{} loaded successfully.".format(" " + version if version else ""))
    except ImportError as e:
        _emit({"result": False, "error": "Cannot import pymeshlab: " + str(e)})
        sys.exit(1)

    try:
        ms = pymeshlab.MeshSet()
        if action not in multi_input_actions:
            ms.load_new_mesh(input_ply)

        if action == "cleanup":
            handle_cleanup(ms, pymeshlab, params, input_ply, output_ply)
        elif action == "merge_by_distance":
            handle_merge_by_distance(ms, pymeshlab, params, input_ply, output_ply)
        elif action == "smooth_taubin":
            handle_smooth_taubin(ms, pymeshlab, params, input_ply, output_ply)
        elif action == "screened_poisson":
            handle_screened_poisson(ms, pymeshlab, params, input_ply, output_ply)
        elif action == "align_point_clouds":
            handle_align_point_clouds(ms, pymeshlab, params)
        elif action == "fuse_point_clouds":
            handle_fuse_point_clouds(ms, pymeshlab, params, output_ply)
        else:
            _emit({"result": False, "error": "Unknown worker action: " + str(action)})
            sys.exit(1)

    except Exception as e:
        _emit({"result": False, "error": "PyMeshLab worker exception: " + str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
