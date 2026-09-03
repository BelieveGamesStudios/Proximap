from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .models import DeepMeshFusionConfig
from .workspace import DeepMeshFusionWorkspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register, fuse, validate, and prepare LiDAR/photogrammetry observations")
    parser.add_argument("workspace", help="Directory for derived fusion artifacts")
    parser.add_argument("passes", nargs="+", help="Source .PLY scan passes (never modified)")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="Scene-unit voxel size (default: 0.03)")
    parser.add_argument("--analysis-cell-size", type=float, help="Spatial evidence cell size (default: 4x voxel size)")
    parser.add_argument("--fusion-cell-size", type=float, help="Consensus geometry cell size (default: voxel size)")
    parser.add_argument("--artifact-cell-size", type=float, help="Transient/artifact detection cell size (default: fusion cell size)")
    parser.add_argument("--artifact-threshold", type=float, default=0.65, help="Suppression score threshold (default: 0.65)")
    parser.add_argument("--up-axis", choices=("x", "y", "z"), default="y", help="Architectural up axis (default: y, matching Proximap viewport space)")
    parser.add_argument("--architecture-grid-size", type=float, help="Planar mesh/opening grid size (default: fusion cell size)")
    parser.add_argument("--plane-distance", type=float, help="Maximum architectural plane inlier distance")
    parser.add_argument("--gap-max-planar-area", type=float, default=1.5, help="Maximum inferred planar repair area in square scene units")
    parser.add_argument("--gap-min-confidence", type=float, default=0.78, help="Minimum plane confidence for inferred repair")
    parser.add_argument("--max-triangle-aspect", type=float, default=12.0, help="Maximum accepted triangle aspect ratio")
    parser.add_argument("--min-geometry-quality", type=float, default=0.82, help="Minimum overall geometry quality for appearance readiness")
    parser.add_argument("--photogrammetry-model", help="COLMAP model directory containing a text or binary cameras/images/points3D file set")
    parser.add_argument("--image-root", help="Root directory containing COLMAP source images")
    parser.add_argument("--dense-photogrammetry-cloud", help="Optional OpenMVS dense PLY used instead of sparse COLMAP points for alignment")
    parser.add_argument("--allow-geometry-review", action="store_true", help="Analyze photogrammetry even if geometry validation still requires review")
    parser.add_argument("--bake-textures", action="store_true", help="Bake a confidence-aware texture atlas after photogrammetry preparation")
    parser.add_argument("--texture-atlas-size", type=int, default=2048, help="Square texture atlas resolution (default: 2048)")
    parser.add_argument("--allow-texture-review", action="store_true", help="Create a provisional bake even if photogrammetry preparation requires review")
    parser.add_argument("--finalize-asset", action="store_true", help="Inspect and conservatively repair the baked final asset")
    parser.add_argument("--tour-readiness", action="store_true", help="Run the final virtual-tour production quality gate")
    parser.add_argument("--reference", help="Generated pass id to use as the registration reference")
    parser.add_argument("--output", default="fused_point_cloud.ply", help="Derived .PLY filename")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    workspace = DeepMeshFusionWorkspace(
        args.workspace,
        DeepMeshFusionConfig(
            voxel_size=args.voxel_size,
            analysis_cell_size=args.analysis_cell_size,
            fusion_cell_size=args.fusion_cell_size,
            artifact_cell_size=args.artifact_cell_size,
            artifact_suppression_threshold=args.artifact_threshold,
            architecture_up_axis=args.up_axis,
            architecture_grid_size=args.architecture_grid_size,
            architecture_plane_distance=args.plane_distance,
            gap_max_planar_area=args.gap_max_planar_area,
            gap_min_plane_confidence=args.gap_min_confidence,
            validation_max_triangle_aspect_ratio=args.max_triangle_aspect,
            validation_min_overall_quality=args.min_geometry_quality,
            texture_atlas_size=args.texture_atlas_size,
        ),
        print,
    )
    for path in args.passes:
        workspace.add_pass(path)
    workspace.analyze_passes()
    workspace.register_passes(args.reference)
    analysis = workspace.analyze_cross_passes()
    fusion = workspace.fuse_registered(args.output)
    preparation = None
    if args.photogrammetry_model or args.image_root:
        if not args.photogrammetry_model or not args.image_root:
            raise ValueError("--photogrammetry-model and --image-root must be provided together")
        preparation = workspace.prepare_photogrammetry(
            args.photogrammetry_model, args.image_root, args.dense_photogrammetry_cloud,
            allow_geometry_review=args.allow_geometry_review,
        )
    payload = {"cross_pass_analysis": asdict(analysis), "fusion": asdict(fusion)}
    if preparation is not None:
        payload["photogrammetry_preparation"] = asdict(preparation)
    if args.bake_textures or args.finalize_asset or args.tour_readiness:
        if preparation is None:
            raise ValueError("Texture baking/finalization requires --photogrammetry-model and --image-root")
        baking = workspace.bake_textures(allow_texture_review=args.allow_texture_review)
        payload["texture_baking"] = asdict(baking)
    if args.finalize_asset or args.tour_readiness:
        final_asset = workspace.finalize_asset(allow_texture_review=args.allow_texture_review)
        payload["final_asset"] = asdict(final_asset)
    if args.tour_readiness:
        tour_readiness = workspace.evaluate_tour_readiness()
        payload["tour_readiness"] = asdict(tour_readiness)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
