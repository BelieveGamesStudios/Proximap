from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .confidence import DeepMeshFusionConfidenceService
from .models import (
    DeepMeshFusionConfig,
    PassRegionEvidence,
    ScanPass,
    SpatialEvidenceMap,
    SpatialEvidenceSummary,
    SpatialRegionEvidence,
)


_NEIGHBOR_OFFSETS = tuple(
    (x, y, z)
    for x in (-1, 0, 1)
    for y in (-1, 0, 1)
    for z in (-1, 0, 1)
    if (x, y, z) != (0, 0, 0)
)


class DeepMeshFusionAnalysisService:
    """Builds a localized, explainable evidence map from registered scan passes."""

    SCHEMA_VERSION = 1

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config
        self.confidence = DeepMeshFusionConfidenceService(config)

    def analyze(self, registered_clouds: Sequence[Tuple[ScanPass, object]]) -> SpatialEvidenceMap:
        if len(registered_clouds) < 2:
            raise ValueError("Cross-pass analysis requires at least two accepted registered passes")
        cell_size = self.config.effective_analysis_cell_size()
        prepared = [self._prepare(scan_pass, cloud, cell_size) for scan_pass, cloud in registered_clouds]
        global_min = np.min([item[2][0] for item in prepared], axis=0)
        origin = np.floor(global_min / cell_size) * cell_size

        pass_cells: Dict[str, Dict[Tuple[int, int, int], dict]] = {}
        pass_bounds = {}
        all_counts = []
        for scan_pass, cloud, bounds in prepared:
            cells = self._aggregate_pass(cloud, origin, cell_size)
            pass_cells[scan_pass.pass_id] = cells
            pass_bounds[scan_pass.pass_id] = bounds
            all_counts.extend(item["point_count"] for item in cells.values())

        pass_ids = [item[0].pass_id for item in prepared]
        scan_by_id = {item[0].pass_id: item[0] for item in prepared}
        all_keys = sorted({key for cells in pass_cells.values() for key in cells})
        density_target = max(float(np.median(all_counts)) if all_counts else 1.0, 1.0)
        regions = [
            self._build_region(
                key, origin, cell_size, pass_ids, scan_by_id, pass_cells, pass_bounds, density_target
            )
            for key in all_keys
        ]
        total = len(regions)
        per_pass_coverage = {
            pass_id: float(sum(pass_id in region.provenance for region in regions) / total) if total else 0.0
            for pass_id in pass_ids
        }
        summary = SpatialEvidenceSummary(
            total_regions=total,
            multi_pass_regions=sum(region.observation_count >= 2 for region in regions),
            conflict_regions=sum(region.conflict for region in regions),
            missing_observation_regions=sum(bool(region.missing_pass_ids) for region in regions),
            high_confidence_regions=sum(region.confidence >= 0.80 and not region.conflict for region in regions),
            overlap_ratio=float(sum(region.observation_count >= 2 for region in regions) / total) if total else 0.0,
            mean_confidence=float(np.mean([region.confidence for region in regions])) if regions else 0.0,
            per_pass_coverage=per_pass_coverage,
        )
        provenance = {
            scan_pass.pass_id: {
                "name": scan_pass.name,
                "source_path": scan_pass.source_path,
                "source_sha256": scan_pass.source_sha256,
                "registration_method": scan_pass.registration.method,
                "registration_fitness": scan_pass.registration.fitness,
                "transform_sha256": self._transform_hash(scan_pass.registration.transform),
            }
            for scan_pass, _cloud in registered_clouds
        }
        return SpatialEvidenceMap(
            schema_version=self.SCHEMA_VERSION,
            cell_size=cell_size,
            grid_origin=origin.tolist(),
            registered_pass_ids=pass_ids,
            source_provenance=provenance,
            scoring_weights=self.config.confidence_weights(),
            summary=summary,
            regions=regions,
        )

    def export(self, evidence_map: SpatialEvidenceMap, json_path: str, confidence_ply_path: str) -> None:
        json_target = Path(json_path)
        json_target.parent.mkdir(parents=True, exist_ok=True)
        temporary = json_target.with_suffix(json_target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(evidence_map), indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(json_target)
        self._write_confidence_ply(evidence_map, Path(confidence_ply_path))

    def _prepare(self, scan_pass: ScanPass, cloud, cell_size: float):
        import open3d as o3d

        derived = o3d.geometry.PointCloud(cloud)
        derived.transform(np.asarray(scan_pass.registration.transform, dtype=float))
        points = np.asarray(derived.points)
        return scan_pass, derived, (points.min(axis=0), points.max(axis=0))

    @staticmethod
    def _aggregate_pass(cloud, origin: np.ndarray, cell_size: float) -> Dict[Tuple[int, int, int], dict]:
        points = np.asarray(cloud.points, dtype=np.float64)
        normals = np.asarray(cloud.normals, dtype=np.float64) if cloud.has_normals() else None
        indices = np.floor((points - origin) / cell_size).astype(np.int64)
        unique, inverse = np.unique(indices, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=len(unique))
        offsets = np.concatenate(([0], np.cumsum(counts)))
        cells = {}
        for group, raw_key in enumerate(unique):
            selected = order[offsets[group]:offsets[group + 1]]
            local = points[selected]
            centroid = local.mean(axis=0)
            mean_normal, consistency = DeepMeshFusionAnalysisService._local_surface(local, normals[selected] if normals is not None else None)
            key = tuple(int(value) for value in raw_key)
            cells[key] = {
                "point_count": int(len(selected)),
                "centroid": centroid,
                "mean_normal": mean_normal,
                "surface_consistency": consistency,
            }
        return cells

    @staticmethod
    def _local_surface(points: np.ndarray, normals: Optional[np.ndarray]):
        if len(points) >= 3:
            covariance = np.cov(points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            total = max(float(np.sum(np.maximum(eigenvalues, 0.0))), 1e-12)
            variation = float(max(eigenvalues[0], 0.0) / total)
            normal = eigenvectors[:, 0]
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            return normal, float(np.clip(1.0 - 3.0 * variation, 0.0, 1.0))
        if normals is not None and len(normals):
            aligned = normals.copy()
            aligned[np.sum(aligned * aligned[0], axis=1) < 0] *= -1
            normal = aligned.mean(axis=0)
            length = float(np.linalg.norm(normal))
            if length > 1e-12:
                return normal / length, 0.5
        return None, 0.25

    def _build_region(
        self, key, origin, cell_size, pass_ids, scan_by_id, pass_cells, pass_bounds, density_target
    ) -> SpatialRegionEvidence:
        observations = {pass_id: pass_cells[pass_id][key] for pass_id in pass_ids if key in pass_cells[pass_id]}
        centroids = np.asarray([item["centroid"] for item in observations.values()])
        consensus = np.median(centroids, axis=0)
        pair_distances = [
            float(np.linalg.norm(centroids[i] - centroids[j]))
            for i in range(len(centroids)) for j in range(i + 1, len(centroids))
        ]
        mean_distance = float(np.mean(pair_distances)) if pair_distances else 0.0
        max_distance = max(pair_distances, default=0.0)
        normals = [item["mean_normal"] for item in observations.values() if item["mean_normal"] is not None]
        normal_pairs = [
            abs(float(np.dot(normals[i], normals[j])))
            for i in range(len(normals)) for j in range(i + 1, len(normals))
        ]
        region_normal_agreement = float(np.mean(normal_pairs)) if normal_pairs else None
        missing = self._missing_passes(key, pass_ids, observations, pass_cells, pass_bounds, origin, cell_size)
        reasons = []
        if len(observations) >= 2 and max_distance > cell_size * self.config.conflict_distance_multiplier:
            reasons.append("cross-pass-distance")
        if region_normal_agreement is not None and region_normal_agreement < self.config.min_normal_agreement:
            reasons.append("normal-disagreement")
        counts = [item["point_count"] for item in observations.values()]
        if len(counts) >= 2 and max(counts) >= 8 and min(counts) / max(counts) < 0.15:
            reasons.append("density-disparity")
        if len(observations) == 1 and missing:
            reasons.append("single-pass-only")
        conflict = bool(reasons)

        pass_evidence = []
        for pass_id, item in observations.items():
            distance = float(np.linalg.norm(item["centroid"] - consensus))
            tolerance = max(cell_size * self.config.cross_pass_distance_tolerance_multiplier, 1e-12)
            distance_score = float(np.exp(-((distance / tolerance) ** 2))) if len(observations) >= 2 else 0.5
            other_normals = [other["mean_normal"] for other_id, other in observations.items() if other_id != pass_id and other["mean_normal"] is not None]
            normal_score = float(np.mean([abs(float(np.dot(item["mean_normal"], other))) for other in other_normals])) if item["mean_normal"] is not None and other_normals else 0.5
            components = {
                "observation": len(observations) / len(pass_ids),
                "density": min(1.0, item["point_count"] / density_target),
                "distance": distance_score,
                "normal": normal_score,
                "surface": item["surface_consistency"],
                "registration": float(np.clip(scan_by_id[pass_id].registration.fitness, 0.0, 1.0)),
            }
            score = self.confidence.score(components, conflict)
            pass_evidence.append(PassRegionEvidence(
                pass_id=pass_id,
                point_count=item["point_count"],
                density=float(item["point_count"] / (cell_size ** 3)),
                centroid=item["centroid"].tolist(),
                mean_normal=item["mean_normal"].tolist() if item["mean_normal"] is not None else None,
                distance_to_consensus=distance,
                distance_agreement=distance_score,
                normal_agreement=normal_score,
                surface_consistency=item["surface_consistency"],
                registration_quality=components["registration"],
                confidence=score,
                score_components=components,
            ))
        pass_evidence.sort(key=lambda item: item.pass_id)
        region_components = {
            name: float(np.mean([item.score_components[name] for item in pass_evidence]))
            for name in self.config.confidence_weights()
        }
        confidence = self.confidence.score(region_components, conflict)
        lower = origin + np.asarray(key) * cell_size
        upper = lower + cell_size
        return SpatialRegionEvidence(
            region_id=f"{key[0]}:{key[1]}:{key[2]}",
            grid_index=list(key),
            bounds_min=lower.tolist(),
            bounds_max=upper.tolist(),
            center=((lower + upper) * 0.5).tolist(),
            observation_count=len(observations),
            observation_ratio=len(observations) / len(pass_ids),
            total_point_count=sum(counts),
            mean_density=float(np.mean([item.density for item in pass_evidence])),
            mean_cross_pass_distance=mean_distance,
            max_cross_pass_distance=max_distance,
            normal_agreement=region_normal_agreement,
            local_surface_consistency=float(np.mean([item["surface_consistency"] for item in observations.values()])),
            confidence=confidence,
            agreement=self.confidence.agreement_label(confidence, conflict),
            conflict=conflict,
            conflict_reasons=reasons,
            missing_pass_ids=missing,
            pass_evidence=pass_evidence,
            provenance={item.pass_id: item.point_count for item in pass_evidence},
        )

    def _missing_passes(self, key, pass_ids, observations, pass_cells, pass_bounds, origin, cell_size):
        center = origin + (np.asarray(key) + 0.5) * cell_size
        missing = []
        for pass_id in pass_ids:
            if pass_id in observations:
                continue
            lower, upper = pass_bounds[pass_id]
            if not np.all(center >= lower - cell_size * 0.5) or not np.all(center <= upper + cell_size * 0.5):
                continue
            neighbor_count = sum(
                tuple(key[i] + offset[i] for i in range(3)) in pass_cells[pass_id]
                for offset in _NEIGHBOR_OFFSETS
            )
            if neighbor_count >= self.config.missing_neighbor_count:
                missing.append(pass_id)
        return missing

    @staticmethod
    def _transform_hash(transform) -> str:
        canonical = json.dumps(np.round(np.asarray(transform), 12).tolist(), separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _write_confidence_ply(evidence_map: SpatialEvidenceMap, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"comment Proximap spatial confidence cell_size {evidence_map.cell_size:.9g}\n")
            handle.write(f"element vertex {len(evidence_map.regions)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property float confidence\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("property ushort observation_count\nproperty uchar conflict\nend_header\n")
            for region in evidence_map.regions:
                red, green, blue = DeepMeshFusionAnalysisService._confidence_color(region.confidence)
                handle.write(
                    f"{region.center[0]:.7g} {region.center[1]:.7g} {region.center[2]:.7g} "
                    f"{region.confidence:.6f} {red} {green} {blue} {region.observation_count} {int(region.conflict)}\n"
                )

    @staticmethod
    def _confidence_color(confidence: float):
        value = float(np.clip(confidence, 0.0, 1.0))
        if value < 0.5:
            return 255, int(round(510 * value)), 0
        return int(round(510 * (1.0 - value))), 255, 0
