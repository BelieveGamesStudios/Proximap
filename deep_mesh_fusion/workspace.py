from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from point_cloud_io import load_point_cloud

from .analysis import DeepMeshFusionAnalysisService
from .artifacts import TransientArtifactSuppressionService
from .diagnostics import ScanDiagnosticsService
from .fusion import DeepMeshFusionService
from .gaps import EvidenceBasedGapRepairService, GapRepairOutput
from .models import ArchitectureReconstructionResult, ArtifactSuppressionResult, CrossPassAnalysisResult, DeepMeshFusionConfig, DeepMeshFusionResult, FinalAssetResult, GapRepairResult, GeometryValidationResult, PassDiagnostics, PhotogrammetryPreparationResult, PointFusionResult, RegistrationMetrics, ScanPass, TextureBakeResult, TourReadinessResult
from .photogrammetry import PhotogrammetryPreparationService
from .reconstruction import ArchitectureMeshOutput, DeepMeshFusionReconstructionService
from .registration import DeepMeshFusionRegistrationService
from .validation import GeometryValidationService
from .texture_baking import IntelligentTextureBakingService
from .final_repair import FinalSurfaceTextureRepairService
from .quality_gate import TourReadinessQualityGate


class DeepMeshFusionWorkspace:
    """Orchestrates immutable scan sources and derived fusion artifacts."""

    MANIFEST_VERSION = 11

    @classmethod
    def load(cls, root: str, log_fn: Optional[Callable[[str], None]] = None) -> "DeepMeshFusionWorkspace":
        """Restore immutable pass metadata and workflow settings from a workspace manifest."""
        root_path = Path(root).resolve(); payload = json.loads((root_path / "workspace.json").read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(DeepMeshFusionConfig)}
        config = DeepMeshFusionConfig(**{key: value for key, value in payload.get("config", {}).items() if key in allowed})
        workspace = cls(str(root_path), config, log_fn)
        for item in payload.get("passes", []):
            values = dict(item)
            if values.get("diagnostics"): values["diagnostics"] = PassDiagnostics(**values["diagnostics"])
            if values.get("registration"): values["registration"] = RegistrationMetrics(**values["registration"])
            workspace.passes.append(ScanPass(**values))
        return workspace

    def save_workflow_state(self, state: dict) -> None:
        self._save_manifest(workflow_state=state)

    def __init__(self, root: str, config: Optional[DeepMeshFusionConfig] = None, log_fn: Optional[Callable[[str], None]] = None):
        self.root = Path(root).resolve()
        self.config = config or DeepMeshFusionConfig()
        self.config.validate()
        self.log = log_fn or (lambda _message: None)
        self.passes: List[ScanPass] = []
        self._clouds: Dict[str, object] = {}
        self.registration = DeepMeshFusionRegistrationService(self.config, self.log)
        self.diagnostics = ScanDiagnosticsService(self.config)
        self.cross_pass_analysis = DeepMeshFusionAnalysisService(self.config)
        self.fusion = DeepMeshFusionService(self.config)
        self.artifact_suppression = TransientArtifactSuppressionService(self.config)
        self.reconstruction = DeepMeshFusionReconstructionService(self.config)
        self.gap_repair = EvidenceBasedGapRepairService(self.config)
        self.geometry_validation = GeometryValidationService(self.config)
        self.photogrammetry_preparation = PhotogrammetryPreparationService(self.config)
        self.texture_baking = IntelligentTextureBakingService(self.config)
        self.final_repair = FinalSurfaceTextureRepairService(self.config)
        self.tour_readiness = TourReadinessQualityGate(self.config)
        self._last_texture_output = None
        self._last_geometry_confidence = None
        self._last_fused_output = None
        self._last_evidence_map = None
        self._last_point_fusion_result = None
        self._last_gap_output = None
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "registration" / "transforms").mkdir(parents=True, exist_ok=True)
        (self.root / "derived").mkdir(parents=True, exist_ok=True)
        (self.root / "analysis").mkdir(parents=True, exist_ok=True)

    def update_appearance_settings(self, *, texture_atlas_size: int) -> None:
        """Update live appearance settings without rebuilding registered geometry."""
        self.config.texture_atlas_size = int(texture_atlas_size)
        self.config.validate()
        self._save_manifest()

    def add_pass(self, source_path: str, name: Optional[str] = None) -> ScanPass:
        path = Path(source_path).resolve()
        if path.suffix.lower() != ".ply":
            raise ValueError(f"Only .ply scan passes are supported: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = self._sha256(path)
        if any(item.source_sha256 == digest for item in self.passes):
            raise ValueError(f"This scan pass is already in the workspace: {path.name}")

        result = load_point_cloud(str(path), self.log)
        if not result.success or result.cloud is None:
            raise ValueError("; ".join(result.warnings) or f"Could not load {path.name}")
        pass_id = self._next_id(name or path.stem)
        scan_pass = ScanPass(
            pass_id=pass_id,
            name=name or path.stem,
            source_path=str(path),
            source_sha256=digest,
            source_size=path.stat().st_size,
        )
        self.passes.append(scan_pass)
        self._clouds[pass_id] = result.cloud
        self._save_manifest(invalidate_derived=True)
        return scan_pass

    def analyze_passes(self) -> List[ScanPass]:
        for scan_pass in self.passes:
            if scan_pass.enabled:
                scan_pass.diagnostics = self.diagnostics.analyze(self._cloud(scan_pass))
        self._save_manifest()
        return self.passes

    def register_passes(self, reference_pass_id: Optional[str] = None) -> List[ScanPass]:
        enabled = [item for item in self.passes if item.enabled]
        if not enabled:
            raise ValueError("No enabled scan passes are available")
        if any(item.diagnostics is None for item in enabled):
            self.analyze_passes()
        reference = next((item for item in enabled if item.pass_id == reference_pass_id), None)
        if reference_pass_id is not None and reference is None:
            raise KeyError(f"Unknown or disabled reference scan pass: {reference_pass_id}")
        reference = reference or enabled[0]
        reference.registration = RegistrationMetrics(
            reference_pass_id=reference.pass_id,
            initial_overlap=1.0,
            overlap_ratio=1.0,
            fitness=1.0,
            inlier_rmse=0.0,
            correspondence_count=reference.diagnostics.point_count if reference.diagnostics else 0,
            method="reference",
            accepted=True,
            message="Reference pass defines common environment space",
        )
        target = self._as_open3d(self._cloud(reference))
        for scan_pass in enabled:
            if scan_pass.pass_id == reference.pass_id:
                continue
            source = self._as_open3d(self._cloud(scan_pass))
            scan_pass.registration = self.registration.register(source, target, scan_pass.pass_id, reference.pass_id)
            self._write_transform(scan_pass)
        self._write_transform(reference)
        self._save_manifest(invalidate_derived=True)
        return self.passes

    def set_manual_transform(self, pass_id: str, transform, reference_pass_id: Optional[str] = None) -> RegistrationMetrics:
        """Apply and quality-check a manual source-to-reference alignment."""
        source_pass = next((item for item in self.passes if item.pass_id == pass_id), None)
        if source_pass is None:
            raise KeyError(f"Unknown scan pass: {pass_id}")
        inferred_reference = reference_pass_id
        if inferred_reference is None and source_pass.registration is not None:
            inferred_reference = source_pass.registration.reference_pass_id
        if inferred_reference is None:
            inferred_reference = self.passes[0].pass_id if self.passes else None
        reference = next((item for item in self.passes if item.pass_id == inferred_reference), None)
        if reference is None:
            raise KeyError(f"Unknown reference scan pass: {inferred_reference}")
        if source_pass.pass_id == reference.pass_id:
            raise ValueError("The reference pass does not require a manual transform")
        metrics = self.registration.evaluate_transform(
            self._as_open3d(self._cloud(source_pass)),
            self._as_open3d(self._cloud(reference)),
            transform,
            reference.pass_id,
        )
        source_pass.registration = metrics
        self._write_transform(source_pass)
        self._save_manifest(invalidate_derived=True)
        return metrics

    def analyze_cross_passes(
        self,
        evidence_name: str = "spatial_evidence_map.json",
        visualization_name: str = "confidence_map.ply",
    ) -> CrossPassAnalysisResult:
        """Compare accepted observations in common space and persist the spatial confidence map."""
        for name, suffix in ((evidence_name, ".json"), (visualization_name, ".ply")):
            candidate = Path(name)
            if candidate.name != name or candidate.suffix.lower() != suffix:
                raise ValueError(f"{name!r} must be a {suffix} filename without directories")
        accepted = [
            item for item in self.passes
            if item.enabled and item.registration is not None and item.registration.accepted
        ]
        if len(accepted) < 2:
            raise ValueError("Cross-pass analysis requires at least two accepted registered passes")
        inputs = [(item, self._as_open3d(self._cloud(item))) for item in accepted]
        evidence_map = self.cross_pass_analysis.analyze(inputs)
        evidence_path = (self.root / "analysis" / evidence_name).resolve()
        visualization_path = (self.root / "analysis" / visualization_name).resolve()
        self.cross_pass_analysis.export(evidence_map, str(evidence_path), str(visualization_path))
        result = CrossPassAnalysisResult(
            evidence_map_path=str(evidence_path),
            confidence_cloud_path=str(visualization_path),
            region_count=evidence_map.summary.total_regions,
            conflict_region_count=evidence_map.summary.conflict_regions,
            missing_observation_region_count=evidence_map.summary.missing_observation_regions,
            mean_confidence=evidence_map.summary.mean_confidence,
        )
        self._save_manifest(analysis_result=result)
        return result

    def export_registered_cloud(self, output_path: str, *, voxel_downsampled: bool = False) -> str:
        """Export accepted passes in common space without applying consensus filtering."""
        import open3d as o3d

        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".ply":
            raise ValueError("Registered point-cloud export must use a .ply filename")
        accepted = [item for item in self.passes if item.enabled and item.registration and item.registration.accepted]
        if len(accepted) < 2:
            raise ValueError("At least two accepted registered passes are required for export")
        unified = o3d.geometry.PointCloud()
        for scan_pass in accepted:
            cloud = self._as_open3d(self._cloud(scan_pass))
            transformed = o3d.geometry.PointCloud(cloud)
            transformed.transform(np.asarray(scan_pass.registration.transform, dtype=float))
            unified += transformed
        if voxel_downsampled:
            unified = unified.voxel_down_sample(self.config.voxel_size)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if unified.is_empty() or not o3d.io.write_point_cloud(str(destination), unified, write_ascii=False):
            raise ValueError("Open3D could not write the registered point cloud")
        self.log(f"[DEEP_FUSION] Exported {'voxel-downsampled' if voxel_downsampled else 'lossless'} registered cloud: {destination}")
        return str(destination)

    def fuse_registered(
        self,
        output_name: str = "fused_point_cloud.ply",
        provenance_name: Optional[str] = None,
    ) -> DeepMeshFusionResult:
        """Compatibility wrapper that runs point fusion then surface reconstruction."""
        self.fuse_points_registered(output_name, provenance_name)
        return self.reconstruct_fused_surface()

    def fuse_points_registered(
        self,
        output_name: str = "fused_point_cloud.ply",
        provenance_name: Optional[str] = None,
    ) -> PointFusionResult:
        """Generate the inspectable evidence-supported point cloud without meshing it."""
        accepted = [item for item in self.passes if item.enabled and item.registration and item.registration.accepted]
        if len(accepted) < 2:
            raise ValueError("Consensus fusion requires at least two accepted registered passes")
        provenance_name = provenance_name or f"{Path(output_name).stem}.provenance.json"
        for name, suffix in ((output_name, ".ply"), (provenance_name, ".json")):
            candidate = Path(name)
            if candidate.name != name or not name.lower().endswith(suffix):
                raise ValueError(f"{name!r} must be a {suffix} filename without directories")
        inputs = [(item, self._as_open3d(self._cloud(item))) for item in accepted]
        evidence_map = self.cross_pass_analysis.analyze(inputs)
        evidence_path = (self.root / "analysis" / "spatial_evidence_map.json").resolve()
        confidence_path = (self.root / "analysis" / "confidence_map.ply").resolve()
        self.cross_pass_analysis.export(evidence_map, str(evidence_path), str(confidence_path))
        analysis_result = CrossPassAnalysisResult(
            evidence_map_path=str(evidence_path),
            confidence_cloud_path=str(confidence_path),
            region_count=evidence_map.summary.total_regions,
            conflict_region_count=evidence_map.summary.conflict_regions,
            missing_observation_region_count=evidence_map.summary.missing_observation_regions,
            mean_confidence=evidence_map.summary.mean_confidence,
        )
        artifact_output = self.artifact_suppression.suppress(inputs)
        artifact_report_path = (self.root / "analysis" / "artifact_suppression.json").resolve()
        rejected_geometry_path = (self.root / "analysis" / "rejected_artifacts.ply").resolve()
        artifact_result = self.artifact_suppression.export(
            artifact_output, str(artifact_report_path), str(rejected_geometry_path)
        )
        fused = self.fusion.fuse(artifact_output.filtered_clouds, evidence_map)
        output_path = (self.root / "derived" / output_name).resolve()
        provenance_path = (self.root / "derived" / provenance_name).resolve()
        self.fusion.export(fused, str(output_path), str(provenance_path), accepted)
        consensus_count = int(np.sum(fused.method_codes == self.fusion.METHOD_CODES["consensus"]))
        best_count = int(np.sum(fused.method_codes == self.fusion.METHOD_CODES["best-observation"]))
        single_count = int(np.sum(fused.method_codes == self.fusion.METHOD_CODES["single-observation"]))
        warnings = [f"{item.name} requires alignment review and was excluded" for item in self.passes if item.enabled and (not item.registration or not item.registration.accepted)]
        result = PointFusionResult(
            fused_cloud_path=str(output_path), provenance_path=str(provenance_path),
            evidence_map_path=str(evidence_path), confidence_cloud_path=str(confidence_path),
            artifact_report_path=artifact_result.report_path,
            rejected_geometry_path=artifact_result.rejected_geometry_path,
            source_pass_count=len([item for item in self.passes if item.enabled]),
            registered_pass_count=len(accepted), fused_point_count=len(fused.points),
            consensus_point_count=consensus_count, best_observation_point_count=best_count,
            single_observation_point_count=single_count,
            suppressed_observation_count=fused.suppressed_observation_count,
            artifact_suppressed_point_count=artifact_result.summary.suppressed_point_count,
            region_count=analysis_result.region_count,
            conflict_region_count=analysis_result.conflict_region_count,
            mean_confidence=float(np.mean(fused.confidence)), warnings=warnings,
        )
        self._last_fused_output = fused
        self._last_evidence_map = evidence_map
        self._last_point_fusion_result = result
        self._save_manifest(analysis_result=analysis_result, artifact_result=artifact_result,
                            fused_cloud_path=str(output_path), fused_point_count=len(fused.points))
        return result

    def reconstruct_fused_surface(self) -> DeepMeshFusionResult:
        """Run hybrid architecture-aware reconstruction, gap recovery, and validation."""
        if self._last_fused_output is None or self._last_evidence_map is None or self._last_point_fusion_result is None:
            self.fuse_points_registered()
        fused = self._last_fused_output
        evidence_map = self._last_evidence_map
        point_result = self._last_point_fusion_result
        reconstructed = self.reconstruction.reconstruct(fused)
        complex_report = self.reconstruction.last_complex_report
        self.log(
            "[DEEP_FUSION] Screened Poisson residual reconstruction: "
            f"{complex_report['cleaned_point_count']:,}/{complex_report['input_point_count']:,} points retained, "
            f"{complex_report['generated_face_count']:,} faces, watertight={complex_report['watertight']}"
        )
        reconstructed_mesh_path = (self.root / "derived" / "architecture_mesh.ply").resolve()
        reconstruction_report_path = (self.root / "analysis" / "architecture_reconstruction.json").resolve()
        reconstruction_result = self.reconstruction.export(
            reconstructed, str(reconstructed_mesh_path), str(reconstruction_report_path)
        )
        gap_output = self.gap_repair.recover(reconstructed, fused, evidence_map)
        self._last_gap_output = gap_output
        repaired_mesh_path = (self.root / "derived" / "architecture_mesh_repaired.ply").resolve()
        gap_report_path = (self.root / "analysis" / "gap_recovery.json").resolve()
        gap_review_path = (self.root / "analysis" / "gap_review.ply").resolve()
        gap_result = self.gap_repair.export(
            gap_output, str(repaired_mesh_path), str(gap_report_path), str(gap_review_path)
        )
        validation_output = self.geometry_validation.validate(gap_output)
        validated_mesh_path = (self.root / "derived" / "validated_lidar_surface.ply").resolve()
        validation_report_path = (self.root / "analysis" / "geometry_validation.json").resolve()
        quality_map_path = (self.root / "analysis" / "geometry_quality.ply").resolve()
        validation_result = self.geometry_validation.export(
            validation_output, str(validated_mesh_path), str(validation_report_path), str(quality_map_path)
        )
        result = DeepMeshFusionResult(
            fused_cloud_path=point_result.fused_cloud_path,
            provenance_path=point_result.provenance_path,
            artifact_report_path=point_result.artifact_report_path,
            rejected_geometry_path=point_result.rejected_geometry_path,
            reconstructed_mesh_path=reconstruction_result.mesh_path,
            reconstruction_report_path=reconstruction_result.report_path,
            repaired_mesh_path=gap_result.repaired_mesh_path,
            gap_report_path=gap_result.report_path,
            gap_review_path=gap_result.review_path,
            validated_mesh_path=validation_result.validated_mesh_path,
            validation_report_path=validation_result.report_path,
            quality_map_path=validation_result.quality_map_path,
            manifest_path=str(self.root / "workspace.json"),
            source_pass_count=point_result.source_pass_count,
            registered_pass_count=point_result.registered_pass_count,
            fused_point_count=len(fused.points),
            consensus_point_count=point_result.consensus_point_count,
            best_observation_point_count=point_result.best_observation_point_count,
            single_observation_point_count=point_result.single_observation_point_count,
            suppressed_observation_count=point_result.suppressed_observation_count,
            artifact_suppressed_point_count=point_result.artifact_suppressed_point_count,
            reconstructed_vertex_count=reconstruction_result.summary.vertex_count,
            reconstructed_face_count=reconstruction_result.summary.face_count,
            repaired_gap_count=gap_result.summary.repaired_gap_count,
            unresolved_gap_count=gap_result.summary.unresolved_gap_count,
            validation_review_region_count=validation_result.summary.review_region_count,
            geometry_ready=validation_result.summary.ready_for_appearance_processing,
            overall_geometry_quality=validation_result.summary.scores.overall,
            mean_confidence=float(np.mean(fused.confidence)),
            warnings=point_result.warnings,
        )
        self._save_manifest(
            reconstruction_result=reconstruction_result,
            gap_result=gap_result,
            validation_result=validation_result,
            fusion_result=result,
        )
        return result

    def validate_cleaned_mesh(self, mesh_path: str) -> GeometryValidationResult:
        """Revalidate a cumulative cleanup result and promote it as canonical geometry."""
        if self._last_gap_output is None:
            raise ValueError("The reconstructed surface must be available before cleanup validation")
        import open3d as o3d
        from scipy.spatial import cKDTree

        loaded = o3d.io.read_triangle_mesh(str(Path(mesh_path).resolve()))
        if loaded.is_empty() or not loaded.has_triangles():
            raise ValueError("Cleanup produced an empty or non-triangular mesh")
        loaded.remove_degenerate_triangles(); loaded.remove_duplicated_triangles(); loaded.remove_unreferenced_vertices(); loaded.compute_vertex_normals()
        vertices = np.asarray(loaded.vertices, dtype=float); faces = np.asarray(loaded.triangles, dtype=np.int64)
        original = self._last_gap_output.mesh
        nearest = cKDTree(np.asarray(original.vertices, dtype=float)).query(vertices, k=1)[1]
        colors = np.asarray(loaded.vertex_colors, dtype=float) if loaded.has_vertex_colors() else np.asarray(original.colors)[nearest]
        boundary, nonmanifold, components, area = self.reconstruction._mesh_metrics(vertices, faces)
        summary_values = asdict(original.summary); summary_values.update(vertex_count=len(vertices), face_count=len(faces),
            boundary_edge_count=boundary, nonmanifold_edge_count=nonmanifold,
            connected_component_count=components, surface_area=area)
        mesh = ArchitectureMeshOutput(vertices=vertices, faces=faces, normals=np.asarray(loaded.vertex_normals), colors=colors,
            confidence=np.asarray(original.confidence)[nearest], class_codes=np.asarray(original.class_codes)[nearest],
            surface_ids=np.asarray(original.surface_ids)[nearest], planes=original.planes, edges=original.edges,
            corners=original.corners, summary=type(original.summary)(**summary_values))
        cleaned_gap = GapRepairOutput(mesh=mesh, gaps=self._last_gap_output.gaps, summary=self._last_gap_output.summary,
            review_points=self._last_gap_output.review_points, review_codes=self._last_gap_output.review_codes)
        validation = self.geometry_validation.validate(cleaned_gap)
        result = self.geometry_validation.export(validation,
            str(self.root / "derived" / "validated_lidar_surface.ply"),
            str(self.root / "analysis" / "geometry_validation.json"),
            str(self.root / "analysis" / "geometry_quality.ply"))
        self._save_manifest(validation_result=result)
        return result

    def verify_sources_unchanged(self) -> Dict[str, bool]:
        return {item.pass_id: Path(item.source_path).is_file() and self._sha256(Path(item.source_path)) == item.source_sha256 for item in self.passes}

    def prepare_photogrammetry(
        self,
        model_path: str,
        image_root: str,
        dense_cloud_path: Optional[str] = None,
        manual_transform=None,
        allow_geometry_review: bool = False,
    ) -> PhotogrammetryPreparationResult:
        """Register an immutable COLMAP/OpenMVS dataset to the validated LiDAR mesh."""
        validation_path = self.root / "analysis" / "geometry_validation.json"
        mesh_path = self.root / "derived" / "validated_lidar_surface.ply"
        if not validation_path.is_file() or not mesh_path.is_file():
            raise ValueError("Geometry validation must complete before photogrammetry preparation")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        geometry_ready = bool(validation.get("summary", {}).get("ready_for_appearance_processing"))
        if not geometry_ready and not allow_geometry_review:
            raise ValueError("Validated geometry still requires review; pass allow_geometry_review=True to inspect photogrammetry alignment without declaring readiness")
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty() or not mesh.has_triangles():
            raise ValueError("Validated LiDAR surface could not be loaded as a triangle mesh")
        mesh.compute_vertex_normals()
        output = self.photogrammetry_preparation.prepare(
            model_path, image_root, np.asarray(mesh.vertices), np.asarray(mesh.triangles),
            np.asarray(mesh.vertex_normals), dense_cloud_path, manual_transform,
        )
        result = self.photogrammetry_preparation.export(output, str(self.root / "photogrammetry"))
        self._save_manifest(photogrammetry_result=result)
        return result

    def bake_textures(self, allow_texture_review: bool = False) -> TextureBakeResult:
        """Bake registered photogrammetry appearance onto validated fused geometry."""
        registration_path = self.root / "photogrammetry" / "photogrammetry_registration.json"
        validation_path = self.root / "analysis" / "geometry_validation.json"
        mesh_path = self.root / "derived" / "validated_lidar_surface.ply"
        if not registration_path.is_file() or not validation_path.is_file() or not mesh_path.is_file():
            raise ValueError("Photogrammetry preparation and geometry validation must complete before texture baking")
        registration_payload = json.loads(registration_path.read_text(encoding="utf-8"))
        manifest_path = self.root / "workspace.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        prepared = manifest.get("photogrammetry_preparation") or {}
        prepared_ready = bool(prepared.get("summary", {}).get("texture_ready"))
        if not prepared_ready and not allow_texture_review:
            raise ValueError("Photogrammetry preparation still requires review; pass allow_texture_review=True to create an inspectable provisional bake")
        dataset = registration_payload.get("dataset", {})
        if not dataset.get("model_path") or not dataset.get("image_root"):
            raise ValueError("Photogrammetry registration artifact does not contain source dataset provenance")
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(mesh_path)); mesh.compute_vertex_normals()
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        geometry_confidence = self._read_vertex_property(mesh_path, "confidence", len(vertices), default=1.0)
        transform = np.asarray(registration_payload["registration"]["transform"], dtype=float)
        preparation = self.photogrammetry_preparation.prepare(
            dataset["model_path"], dataset["image_root"], vertices, faces,
            np.asarray(mesh.vertex_normals), dataset.get("dense_cloud_path"), transform,
        )
        baked = self.texture_baking.bake(preparation, geometry_confidence)
        result = self.texture_baking.export(baked, str(self.root / "texture"))
        self._last_texture_output = baked
        self._last_geometry_confidence = geometry_confidence
        self._save_manifest(texture_result=result)
        return result

    def finalize_asset(self, allow_texture_review: bool = False) -> FinalAssetResult:
        """Inspect and conservatively repair the actual textured asset."""
        if self._last_texture_output is None:
            self.bake_textures(allow_texture_review=allow_texture_review)
        repaired = self.final_repair.repair(self._last_texture_output, self._last_geometry_confidence)
        result = self.final_repair.export(repaired, str(self.root / "final"))
        self._save_manifest(final_result=result)
        return result

    def evaluate_tour_readiness(self) -> TourReadinessResult:
        """Run the final production quality gate and create handoff reports."""
        final_path = self.root / "final" / "final_asset_validation.json"
        if not final_path.is_file():
            raise ValueError("Final surface and texture repair must complete before tour readiness evaluation")
        output = self.tour_readiness.evaluate(str(self.root))
        result = self.tour_readiness.export(output, str(self.root / "quality"))
        self._save_manifest(tour_result=result)
        return result

    def _cloud(self, scan_pass: ScanPass):
        if scan_pass.pass_id not in self._clouds:
            result = load_point_cloud(scan_pass.source_path, self.log)
            if not result.success or result.cloud is None:
                raise ValueError(f"Could not reload source pass {scan_pass.name}")
            self._clouds[scan_pass.pass_id] = result.cloud
        return self._clouds[scan_pass.pass_id]

    @staticmethod
    def _as_open3d(cloud):
        import open3d as o3d
        if isinstance(cloud, o3d.geometry.PointCloud):
            return cloud
        result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(cloud.points, dtype=float)))
        if getattr(cloud, "colors", None) is not None:
            colors = np.asarray(cloud.colors, dtype=float)
            if colors.max(initial=0) > 1:
                colors /= 255.0
            result.colors = o3d.utility.Vector3dVector(colors)
        if getattr(cloud, "normals", None) is not None:
            result.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals, dtype=float))
        return result

    def _save_manifest(
        self,
        fused_cloud_path: Optional[str] = None,
        fused_point_count: Optional[int] = None,
        analysis_result: Optional[CrossPassAnalysisResult] = None,
        artifact_result: Optional[ArtifactSuppressionResult] = None,
        reconstruction_result: Optional[ArchitectureReconstructionResult] = None,
        gap_result: Optional[GapRepairResult] = None,
        validation_result: Optional[GeometryValidationResult] = None,
        photogrammetry_result: Optional[PhotogrammetryPreparationResult] = None,
        texture_result: Optional[TextureBakeResult] = None,
        final_result: Optional[FinalAssetResult] = None,
        tour_result: Optional[TourReadinessResult] = None,
        fusion_result: Optional[DeepMeshFusionResult] = None,
        workflow_state: Optional[dict] = None,
        invalidate_derived: bool = False,
    ) -> None:
        manifest_path = self.root / "workspace.json"
        previous = {}
        if manifest_path.exists():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        artifact = previous.get("fused_artifact")
        if invalidate_derived:
            artifact = None
        if fused_cloud_path is not None:
            artifact = {"path": fused_cloud_path, "point_count": fused_point_count, "derived": True}
        if fusion_result is not None:
            artifact = {**asdict(fusion_result), "derived": True}
        analysis_artifact = previous.get("cross_pass_analysis")
        if invalidate_derived:
            analysis_artifact = None
        if analysis_result is not None:
            analysis_artifact = {**asdict(analysis_result), "derived": True}
        artifact_suppression = previous.get("artifact_suppression")
        if invalidate_derived:
            artifact_suppression = None
        if artifact_result is not None:
            artifact_suppression = {**asdict(artifact_result), "derived": True}
        reconstruction_artifact = previous.get("architecture_reconstruction")
        if invalidate_derived:
            reconstruction_artifact = None
        if reconstruction_result is not None:
            reconstruction_artifact = {**asdict(reconstruction_result), "derived": True}
        gap_artifact = previous.get("gap_recovery")
        if invalidate_derived:
            gap_artifact = None
        if gap_result is not None:
            gap_artifact = {**asdict(gap_result), "derived": True}
        validation_artifact = previous.get("geometry_validation")
        if invalidate_derived:
            validation_artifact = None
        if validation_result is not None:
            validation_artifact = {**asdict(validation_result), "derived": True}
        photogrammetry_artifact = previous.get("photogrammetry_preparation")
        geometry_changed = fused_cloud_path is not None or fusion_result is not None or reconstruction_result is not None or validation_result is not None
        if invalidate_derived or geometry_changed:
            photogrammetry_artifact = None
        if photogrammetry_result is not None:
            photogrammetry_artifact = {**asdict(photogrammetry_result), "derived": True}
        texture_artifact = previous.get("texture_baking")
        if invalidate_derived or geometry_changed or photogrammetry_result is not None:
            texture_artifact = None
        if texture_result is not None:
            texture_artifact = {**asdict(texture_result), "derived": True}
        final_artifact = previous.get("final_asset")
        if invalidate_derived or geometry_changed or photogrammetry_result is not None or texture_result is not None:
            final_artifact = None
        if final_result is not None:
            final_artifact = {**asdict(final_result), "derived": True}
        tour_artifact = previous.get("tour_readiness")
        if invalidate_derived or geometry_changed or photogrammetry_result is not None or texture_result is not None or final_result is not None:
            tour_artifact = None
        if tour_result is not None:
            tour_artifact = {**asdict(tour_result), "derived": True}
        payload = {
            "schema_version": self.MANIFEST_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "source_policy": "immutable-reference",
            "passes": [asdict(item) for item in self.passes],
            "cross_pass_analysis": analysis_artifact,
            "artifact_suppression": artifact_suppression,
            "architecture_reconstruction": reconstruction_artifact,
            "gap_recovery": gap_artifact,
            "geometry_validation": validation_artifact,
            "photogrammetry_preparation": photogrammetry_artifact,
            "texture_baking": texture_artifact,
            "final_asset": final_artifact,
            "tour_readiness": tour_artifact,
            "fused_artifact": artifact,
            "workflow_state": workflow_state if workflow_state is not None else previous.get("workflow_state"),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(manifest_path)

    def _write_transform(self, scan_pass: ScanPass) -> None:
        path = self.root / "registration" / "transforms" / f"{scan_pass.pass_id}.json"
        path.write_text(json.dumps(asdict(scan_pass.registration), indent=2, allow_nan=False), encoding="utf-8")

    @staticmethod
    def _read_vertex_property(path: Path, property_name: str, vertex_count: int, default: float):
        with path.open("r", encoding="ascii", errors="replace") as handle:
            properties, declared_vertices, in_vertices = [], None, False
            for line in handle:
                values = line.strip().split()
                if values[:2] == ["element", "vertex"]:
                    declared_vertices, in_vertices = int(values[2]), True
                elif values[:2] == ["element", "face"]:
                    in_vertices = False
                elif values[:1] == ["property"] and in_vertices and len(values) >= 3:
                    properties.append(values[-1])
                elif values[:1] == ["end_header"]:
                    break
            if declared_vertices != vertex_count or property_name not in properties:
                return np.full(vertex_count, default, dtype=float)
            property_index = properties.index(property_name)
            result = []
            for _ in range(vertex_count):
                line = handle.readline()
                if not line: return np.full(vertex_count, default, dtype=float)
                values = line.split(); result.append(float(values[property_index]))
            return np.asarray(result, dtype=float)

    def _next_id(self, name: str) -> str:
        stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "pass"
        candidate = stem
        number = 2
        existing = {item.pass_id for item in self.passes}
        while candidate in existing:
            candidate = f"{stem}-{number}"
            number += 1
        return candidate

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
