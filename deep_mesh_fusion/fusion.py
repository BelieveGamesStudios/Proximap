from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analysis import DeepMeshFusionAnalysisService, _NEIGHBOR_OFFSETS
from .confidence import DeepMeshFusionConfidenceService
from .models import (
    ConsensusPointProvenance,
    DeepMeshFusionConfig,
    GeometryProvenanceContribution,
    ScanPass,
    SpatialEvidenceMap,
)


@dataclass
class ConsensusFusionOutput:
    points: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    observation_counts: np.ndarray
    method_codes: np.ndarray
    provenance: List[ConsensusPointProvenance]
    suppressed_observation_count: int


class DeepMeshFusionService:
    """Generates evidence-supported geometry instead of concatenating registered points."""

    METHOD_CODES = {"consensus": 0, "best-observation": 1, "single-observation": 2}

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config
        self.confidence = DeepMeshFusionConfidenceService(config)

    def fuse(
        self,
        registered_clouds: Sequence[Tuple[ScanPass, object]],
        evidence_map: Optional[SpatialEvidenceMap] = None,
    ) -> ConsensusFusionOutput:
        registered_clouds = [item for item in registered_clouds if len(item[1].points) > 0]
        if len(registered_clouds) < 2:
            raise ValueError("Consensus fusion requires at least two accepted registered passes")
        cell_size = self.config.effective_fusion_cell_size()
        prepared = [self._prepare(scan_pass, cloud) for scan_pass, cloud in registered_clouds]
        origin = np.floor(np.min([item[2][0] for item in prepared], axis=0) / cell_size) * cell_size
        pass_cells = {
            scan_pass.pass_id: self._aggregate_pass(cloud, origin, cell_size, scan_pass.pass_id)
            for scan_pass, cloud, _bounds in prepared
        }
        scans = {scan_pass.pass_id: scan_pass for scan_pass, _cloud, _bounds in prepared}
        all_keys = sorted({key for cells in pass_cells.values() for key in cells})
        all_counts = [item["point_count"] for cells in pass_cells.values() for item in cells.values()]
        density_target = max(float(np.median(all_counts)) if all_counts else 1.0, 1.0)
        region_lookup = self._region_lookup(evidence_map)

        points, normals, colors, confidences, observation_counts, method_codes = [], [], [], [], [], []
        provenance: List[ConsensusPointProvenance] = []
        suppressed = 0
        for key in all_keys:
            observations = [pass_cells[pass_id][key] for pass_id in scans if key in pass_cells[pass_id]]
            for item in observations:
                item["density_score"] = min(1.0, item["point_count"] / density_target)
                item["registration"] = float(np.clip(scans[item["pass_id"]].registration.fitness, 0.0, 1.0))
                item["quality"] = self._observation_quality(item, scans[item["pass_id"]], density_target)
            clusters = self._cluster_observations(observations, cell_size)
            supported = [cluster for cluster in clusters if len({item["pass_id"] for item in cluster}) >= self.config.min_consensus_observations]
            candidates = supported
            if not supported and clusters:
                strongest = max(clusters, key=lambda cluster: max(item["quality"] for item in cluster))
                representative = max(strongest, key=lambda item: item["quality"])
                region = self._region_for_point(representative["point"], evidence_map, region_lookup)
                if self._retain_single(key, representative, pass_cells, region):
                    candidates = [[representative]]
                suppressed += sum(len(cluster) for cluster in clusters) - len(candidates)
            elif supported:
                supported_ids = {id(cluster) for cluster in supported}
                suppressed += sum(len(cluster) for cluster in clusters if id(cluster) not in supported_ids)

            for cluster in candidates:
                fused = self._fuse_cluster(cluster, len(scans), cell_size, evidence_map, region_lookup)
                output_index = len(points)
                points.append(fused["point"])
                normals.append(fused["normal"])
                colors.append(fused["color"])
                confidences.append(fused["confidence"])
                observation_counts.append(len({item["pass_id"] for item in cluster}))
                method_codes.append(self.METHOD_CODES[fused["method"]])
                provenance.append(ConsensusPointProvenance(
                    output_index=output_index,
                    grid_index=list(key),
                    fusion_method=fused["method"],
                    confidence=fused["confidence"],
                    region_id=fused["region_id"],
                    contributions=fused["contributions"],
                ))

        if not points:
            raise ValueError("No evidence-supported geometry survived consensus fusion")
        return ConsensusFusionOutput(
            points=np.asarray(points, dtype=np.float64),
            normals=np.asarray(normals, dtype=np.float64),
            colors=np.asarray(colors, dtype=np.float64),
            confidence=np.asarray(confidences, dtype=np.float64),
            observation_counts=np.asarray(observation_counts, dtype=np.uint16),
            method_codes=np.asarray(method_codes, dtype=np.uint8),
            provenance=provenance,
            suppressed_observation_count=suppressed,
        )

    def export(
        self,
        output: ConsensusFusionOutput,
        cloud_path: str,
        provenance_path: str,
        registered_passes: Sequence[ScanPass],
    ) -> None:
        self._write_fused_ply(output, Path(cloud_path))
        method_counts = {
            name: int(np.sum(output.method_codes == code)) for name, code in self.METHOD_CODES.items()
        }
        per_pass_contributions = {scan_pass.pass_id: 0 for scan_pass in registered_passes}
        for point in output.provenance:
            for contribution in point.contributions:
                if contribution.weight > 0:
                    per_pass_contributions[contribution.pass_id] += 1
        payload = {
            "schema_version": 1,
            "source_policy": "immutable-reference",
            "fusion_cell_size": self.config.effective_fusion_cell_size(),
            "method_codes": self.METHOD_CODES,
            "summary": {
                "output_point_count": len(output.points),
                "method_counts": method_counts,
                "suppressed_observation_count": output.suppressed_observation_count,
                "mean_confidence": float(np.mean(output.confidence)),
                "per_pass_contributions": per_pass_contributions,
            },
            "sources": {
                scan_pass.pass_id: {
                    "name": scan_pass.name,
                    "source_path": scan_pass.source_path,
                    "source_sha256": scan_pass.source_sha256,
                    "registration_transform": scan_pass.registration.transform,
                    "registration_fitness": scan_pass.registration.fitness,
                }
                for scan_pass in registered_passes
            },
        }
        target = Path(provenance_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        # Stream point records so large scans do not require a second in-memory copy.
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), allow_nan=False)[:-1])
            handle.write(',"points":[')
            for index, item in enumerate(output.provenance):
                if index:
                    handle.write(",")
                handle.write(json.dumps(asdict(item), separators=(",", ":"), allow_nan=False))
            handle.write("]}")
        temporary.replace(target)

    @staticmethod
    def _prepare(scan_pass: ScanPass, cloud):
        import open3d as o3d

        derived = o3d.geometry.PointCloud(cloud)
        derived.transform(np.asarray(scan_pass.registration.transform, dtype=float))
        points = np.asarray(derived.points)
        return scan_pass, derived, (points.min(axis=0), points.max(axis=0))

    @staticmethod
    def _aggregate_pass(cloud, origin, cell_size, pass_id: str) -> Dict[Tuple[int, int, int], dict]:
        points = np.asarray(cloud.points, dtype=np.float64)
        colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
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
            normal, surface = DeepMeshFusionAnalysisService._local_surface(
                local, normals[selected] if normals is not None else None
            )
            key = tuple(int(value) for value in raw_key)
            cells[key] = {
                "pass_id": pass_id,
                "point_count": int(len(selected)),
                "point": np.median(local, axis=0),
                "normal": normal,
                "color": np.median(colors[selected], axis=0) if colors is not None else np.asarray([0.5, 0.5, 0.5]),
                "surface": surface,
            }
        return cells

    def _cluster_observations(self, observations, cell_size):
        if not observations:
            return []
        radius = cell_size * self.config.correspondence_distance_multiplier
        parent = list(range(len(observations)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            a, b = find(left), find(right)
            if a != b:
                parent[b] = a

        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                if np.linalg.norm(observations[i]["point"] - observations[j]["point"]) <= radius:
                    union(i, j)
        groups = {}
        for index, observation in enumerate(observations):
            groups.setdefault(find(index), []).append(observation)
        return list(groups.values())

    @staticmethod
    def _observation_quality(observation, scan_pass: ScanPass, density_target: float) -> float:
        density = min(1.0, observation["point_count"] / density_target)
        registration = float(np.clip(scan_pass.registration.fitness, 0.0, 1.0))
        return float(np.clip(0.45 * registration + 0.35 * observation["surface"] + 0.20 * density, 0.0, 1.0))

    def _fuse_cluster(self, cluster, total_passes, cell_size, evidence_map, region_lookup):
        locations = np.asarray([item["point"] for item in cluster])
        center = np.median(locations, axis=0)
        residuals = np.linalg.norm(locations - center, axis=1)
        delta = max(cell_size * self.config.consensus_huber_delta_multiplier, 1e-12)
        huber = np.minimum(1.0, delta / np.maximum(residuals, 1e-12))
        quality = np.asarray([item["quality"] for item in cluster])
        weights = quality * huber
        weights /= max(float(np.sum(weights)), 1e-12)
        normal_agreement = self._normal_agreement(cluster)
        ordered_quality = np.sort(quality)[::-1]
        dominant = len(quality) > 1 and ordered_quality[0] - ordered_quality[1] >= self.config.best_observation_margin
        spread = float(np.max(residuals)) if len(residuals) else 0.0
        choose_best = normal_agreement < self.config.min_normal_agreement or dominant or spread > cell_size * 0.5
        if len(cluster) == 1:
            method = "single-observation"
            chosen = 0
            point = locations[0]
            effective_weights = np.asarray([1.0])
        elif choose_best:
            method = "best-observation"
            chosen = int(np.argmax(quality))
            point = locations[chosen]
            effective_weights = np.zeros(len(cluster))
            effective_weights[chosen] = 1.0
        else:
            method = "consensus"
            point = np.sum(locations * weights[:, None], axis=0)
            effective_weights = weights
        normal = self._weighted_normal(cluster, effective_weights)
        color = np.sum(np.asarray([item["color"] for item in cluster]) * effective_weights[:, None], axis=0)
        region = self._region_for_point(point, evidence_map, region_lookup)
        distance_score = float(np.exp(-((float(np.mean(residuals)) / max(delta, 1e-12)) ** 2))) if len(cluster) > 1 else 0.5
        components = {
            "observation": len({item["pass_id"] for item in cluster}) / total_passes,
            "density": float(np.mean([item["density_score"] for item in cluster])),
            "distance": distance_score,
            "normal": normal_agreement if len(cluster) > 1 else 0.5,
            "surface": float(np.mean([item["surface"] for item in cluster])),
            "registration": float(np.mean([item["registration"] for item in cluster])),
        }
        local_confidence = self.confidence.score(components, False)
        if region is not None:
            blend = self.config.fusion_region_confidence_weight
            confidence = blend * region.confidence + (1.0 - blend) * local_confidence
            region_id = region.region_id
        else:
            confidence = local_confidence
            region_id = None
        contributions = [GeometryProvenanceContribution(
            pass_id=item["pass_id"],
            source_point_count=item["point_count"],
            representative_point=item["point"].tolist(),
            weight=float(effective_weights[index]),
            residual=float(np.linalg.norm(item["point"] - point)),
            observation_quality=item["quality"],
        ) for index, item in enumerate(cluster)]
        return {
            "point": point,
            "normal": normal,
            "color": np.clip(color, 0.0, 1.0),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "method": method,
            "region_id": region_id,
            "contributions": contributions,
        }

    def _retain_single(self, key, observation, pass_cells, region) -> bool:
        if not self.config.retain_single_pass_geometry:
            return False
        neighbors = sum(
            tuple(key[i] + offset[i] for i in range(3)) in pass_cells[observation["pass_id"]]
            for offset in _NEIGHBOR_OFFSETS
        )
        region_confidence = region.confidence if region is not None else 0.5
        return neighbors >= self.config.single_pass_min_neighbors and region_confidence >= self.config.min_single_pass_region_confidence

    @staticmethod
    def _normal_agreement(cluster) -> float:
        normals = [item["normal"] for item in cluster if item["normal"] is not None]
        if len(normals) < 2:
            return 0.5
        values = [abs(float(np.dot(normals[i], normals[j]))) for i in range(len(normals)) for j in range(i + 1, len(normals))]
        return float(np.clip(np.mean(values), 0.0, 1.0))

    @staticmethod
    def _weighted_normal(cluster, weights):
        available = [(index, item["normal"]) for index, item in enumerate(cluster) if item["normal"] is not None]
        if not available:
            return np.zeros(3)
        reference = available[0][1]
        result = np.zeros(3)
        for index, normal in available:
            aligned = normal if np.dot(normal, reference) >= 0 else -normal
            result += weights[index] * aligned
        length = float(np.linalg.norm(result))
        return result / length if length > 1e-12 else reference

    @staticmethod
    def _region_lookup(evidence_map):
        if evidence_map is None:
            return {}
        return {tuple(region.grid_index): region for region in evidence_map.regions}

    @staticmethod
    def _region_for_point(point, evidence_map, lookup):
        if evidence_map is None:
            return None
        key = tuple(np.floor((np.asarray(point) - np.asarray(evidence_map.grid_origin)) / evidence_map.cell_size).astype(int))
        return lookup.get(key)

    @staticmethod
    def _write_fused_ply(output: ConsensusFusionOutput, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii") as handle:
            handle.write("ply\nformat ascii 1.0\ncomment Proximap evidence-supported consensus geometry\n")
            handle.write(f"element vertex {len(output.points)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property float nx\nproperty float ny\nproperty float nz\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("property float confidence\nproperty ushort observation_count\nproperty uchar fusion_method\nend_header\n")
            rgb = np.clip(np.rint(output.colors * 255.0), 0, 255).astype(np.uint8)
            for index, point in enumerate(output.points):
                normal = output.normals[index]
                handle.write(
                    f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                    f"{normal[0]:.7g} {normal[1]:.7g} {normal[2]:.7g} "
                    f"{rgb[index, 0]} {rgb[index, 1]} {rgb[index, 2]} "
                    f"{output.confidence[index]:.6f} {output.observation_counts[index]} {output.method_codes[index]}\n"
                )
