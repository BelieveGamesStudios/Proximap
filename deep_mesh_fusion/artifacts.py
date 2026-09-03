from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .analysis import DeepMeshFusionAnalysisService, _NEIGHBOR_OFFSETS
from .models import (
    ArtifactComponentReport,
    ArtifactSuppressionResult,
    ArtifactSuppressionSummary,
    DeepMeshFusionConfig,
    ScanPass,
)


@dataclass
class ArtifactSuppressionOutput:
    filtered_clouds: List[Tuple[ScanPass, object]]
    reports: List[ArtifactComponentReport]
    summary: ArtifactSuppressionSummary
    rejected_points: np.ndarray
    rejected_pass_indices: np.ndarray
    source_provenance: Dict[str, dict]


class TransientArtifactSuppressionService:
    """Suppresses non-persistent connected geometry while preserving structural continuity."""

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config

    def suppress(self, registered_clouds: Sequence[Tuple[ScanPass, object]]) -> ArtifactSuppressionOutput:
        if len(registered_clouds) < 2:
            raise ValueError("Artifact suppression requires at least two accepted registered passes")
        if self.config.artifact_min_persistent_passes > len(registered_clouds):
            raise ValueError("artifact_min_persistent_passes exceeds the number of registered passes")
        cell_size = self.config.effective_artifact_cell_size()
        prepared = [self._prepare(scan_pass, cloud) for scan_pass, cloud in registered_clouds]
        origin = np.floor(np.min([item[3][0] for item in prepared], axis=0) / cell_size) * cell_size
        pass_cells = {
            scan_pass.pass_id: self._aggregate(points, normals, origin, cell_size, scan_pass.pass_id)
            for scan_pass, _cloud, points, _bounds, normals in prepared
        }
        pass_bounds = {scan_pass.pass_id: bounds for scan_pass, _cloud, _points, bounds, _normals in prepared}
        pass_ids = [scan_pass.pass_id for scan_pass, _cloud in registered_clouds]
        all_keys = sorted({key for cells in pass_cells.values() for key in cells})
        persistent_by_key: Dict[Tuple[int, int, int], List[dict]] = {}
        candidates_by_pass = {pass_id: {} for pass_id in pass_ids}

        for key in all_keys:
            observations = [pass_cells[pass_id][key] for pass_id in pass_ids if key in pass_cells[pass_id]]
            clusters = self._cluster(observations, cell_size)
            for cluster in clusters:
                support = len({item["pass_id"] for item in cluster})
                for item in cluster:
                    item["support"] = support
                if support >= self.config.artifact_min_persistent_passes:
                    persistent_by_key.setdefault(key, []).append(self._persistent_descriptor(cluster))
                else:
                    for item in cluster:
                        candidates_by_pass[item["pass_id"]][key] = item

        reports = []
        suppressed_indices = {pass_id: set() for pass_id in pass_ids}
        for pass_id in pass_ids:
            for component_number, component_keys in enumerate(self._components(set(candidates_by_pass[pass_id])), 1):
                observations = [candidates_by_pass[pass_id][key] for key in component_keys]
                report = self._classify_component(
                    pass_id,
                    component_number,
                    component_keys,
                    observations,
                    pass_ids,
                    pass_bounds,
                    persistent_by_key,
                    origin,
                    cell_size,
                )
                reports.append(report)
                if report.suppressed:
                    for item in observations:
                        suppressed_indices[pass_id].update(int(index) for index in item["indices"])

        import open3d as o3d

        filtered_clouds = []
        rejected_points, rejected_pass_indices = [], []
        per_pass_suppressed = {}
        input_count = 0
        for pass_index, (scan_pass, source_cloud, transformed_points, _bounds, _normals) in enumerate(prepared):
            count = len(source_cloud.points)
            input_count += count
            rejected = np.asarray(sorted(suppressed_indices[scan_pass.pass_id]), dtype=np.int64)
            keep = np.ones(count, dtype=bool)
            keep[rejected] = False
            filtered_clouds.append((scan_pass, source_cloud.select_by_index(np.flatnonzero(keep).tolist())))
            per_pass_suppressed[scan_pass.pass_id] = int(len(rejected))
            if len(rejected):
                rejected_points.append(transformed_points[rejected])
                rejected_pass_indices.append(np.full(len(rejected), pass_index, dtype=np.uint16))
        rejected_array = np.vstack(rejected_points) if rejected_points else np.empty((0, 3), dtype=float)
        rejected_pass_array = np.concatenate(rejected_pass_indices) if rejected_pass_indices else np.empty(0, dtype=np.uint16)
        classification_counts = {}
        for report in reports:
            classification_counts[report.classification] = classification_counts.get(report.classification, 0) + 1
        suppressed_count = sum(per_pass_suppressed.values())
        summary = ArtifactSuppressionSummary(
            input_point_count=input_count,
            retained_point_count=input_count - suppressed_count,
            suppressed_point_count=suppressed_count,
            candidate_component_count=len(reports),
            suppressed_component_count=sum(report.suppressed for report in reports),
            retained_uncertain_component_count=sum(not report.suppressed for report in reports),
            per_pass_suppressed_points=per_pass_suppressed,
            classification_counts=classification_counts,
        )
        return ArtifactSuppressionOutput(
            filtered_clouds=filtered_clouds,
            reports=reports,
            summary=summary,
            rejected_points=rejected_array,
            rejected_pass_indices=rejected_pass_array,
            source_provenance={
                scan_pass.pass_id: {
                    "source_pass_index": index,
                    "name": scan_pass.name,
                    "source_path": scan_pass.source_path,
                    "source_sha256": scan_pass.source_sha256,
                    "registration_transform": scan_pass.registration.transform,
                    "registration_fitness": scan_pass.registration.fitness,
                }
                for index, (scan_pass, _cloud) in enumerate(registered_clouds)
            },
        )

    def export(self, output: ArtifactSuppressionOutput, report_path: str, rejected_geometry_path: str) -> ArtifactSuppressionResult:
        report_target = Path(report_path)
        report_target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "principle": "persistence-across-independent-passes",
            "parameters": {
                "cell_size": self.config.effective_artifact_cell_size(),
                "min_persistent_passes": self.config.artifact_min_persistent_passes,
                "suppression_threshold": self.config.artifact_suppression_threshold,
                "structural_continuity_threshold": self.config.artifact_structural_continuity_threshold,
            },
            "summary": asdict(output.summary),
            "sources": output.source_provenance,
            "components": [asdict(report) for report in output.reports],
        }
        temporary = report_target.with_suffix(report_target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(report_target)
        self._write_rejected_ply(output, Path(rejected_geometry_path))
        return ArtifactSuppressionResult(
            report_path=str(report_target),
            rejected_geometry_path=str(Path(rejected_geometry_path)),
            summary=output.summary,
        )

    @staticmethod
    def _prepare(scan_pass, cloud):
        import open3d as o3d

        source = o3d.geometry.PointCloud(cloud)
        transformed = o3d.geometry.PointCloud(source)
        transformed.transform(np.asarray(scan_pass.registration.transform, dtype=float))
        points = np.asarray(transformed.points, dtype=np.float64)
        normals = np.asarray(transformed.normals, dtype=np.float64) if transformed.has_normals() else None
        return scan_pass, source, points, (points.min(axis=0), points.max(axis=0)), normals

    @staticmethod
    def _aggregate(points, normals, origin, cell_size, pass_id):
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
                "key": key,
                "pass_id": pass_id,
                "indices": selected,
                "point_count": int(len(selected)),
                "point": np.median(local, axis=0),
                "normal": normal,
                "surface": surface,
            }
        return cells

    def _cluster(self, observations, cell_size):
        if not observations:
            return []
        radius = cell_size * self.config.artifact_correspondence_distance_multiplier
        parent = list(range(len(observations)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left in range(len(observations)):
            for right in range(left + 1, len(observations)):
                if np.linalg.norm(observations[left]["point"] - observations[right]["point"]) <= radius:
                    a, b = find(left), find(right)
                    if a != b:
                        parent[b] = a
        groups = {}
        for index, item in enumerate(observations):
            groups.setdefault(find(index), []).append(item)
        return list(groups.values())

    @staticmethod
    def _persistent_descriptor(cluster):
        points = np.asarray([item["point"] for item in cluster])
        normals = [item["normal"] for item in cluster if item["normal"] is not None]
        normal = None
        if normals:
            aligned = [value if np.dot(value, normals[0]) >= 0 else -value for value in normals]
            normal = np.mean(aligned, axis=0)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
        return {
            "point": np.median(points, axis=0),
            "normal": normal,
            "pass_ids": {item["pass_id"] for item in cluster},
        }

    @staticmethod
    def _components(keys):
        remaining = set(keys)
        while remaining:
            start = remaining.pop()
            component = {start}
            stack = [start]
            while stack:
                current = stack.pop()
                for offset in _NEIGHBOR_OFFSETS:
                    neighbor = tuple(current[index] + offset[index] for index in range(3))
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            yield component

    def _classify_component(
        self, pass_id, number, keys, observations, pass_ids, pass_bounds, persistent_by_key, origin, cell_size
    ):
        key_array = np.asarray(list(keys), dtype=np.int64)
        bounds_min = origin + key_array.min(axis=0) * cell_size
        bounds_max = origin + (key_array.max(axis=0) + 1) * cell_size
        support = float(np.mean([item["support"] / len(pass_ids) for item in observations]))
        coverage_counts = []
        for item in observations:
            center = item["point"]
            coverage_counts.append(sum(
                other_id != pass_id
                and np.all(center >= pass_bounds[other_id][0] - cell_size)
                and np.all(center <= pass_bounds[other_id][1] + cell_size)
                for other_id in pass_ids
            ))
        mean_coverage_count = float(np.mean(coverage_counts)) if coverage_counts else 0.0
        coverage = mean_coverage_count / max(len(pass_ids) - 1, 1)
        compatible = 0
        conflicts = 0
        for item in observations:
            key = item["key"]
            same_cell = persistent_by_key.get(key, [])
            if same_cell:
                conflicts += 1
            neighbor_match = False
            for offset in _NEIGHBOR_OFFSETS:
                neighbor = tuple(key[index] + offset[index] for index in range(3))
                for persistent in persistent_by_key.get(neighbor, []):
                    if item["normal"] is None or persistent["normal"] is None:
                        continue
                    if abs(float(np.dot(item["normal"], persistent["normal"]))) >= self.config.artifact_structural_normal_agreement:
                        neighbor_match = True
                        break
                if neighbor_match:
                    break
            compatible += int(neighbor_match)
        continuity = compatible / len(observations)
        conflict = conflicts / len(observations)
        isolation = min(1.0, self.config.artifact_isolated_component_cells / max(len(keys), 1))
        score = float(np.clip(
            0.35 * (1.0 - support)
            + 0.25 * coverage
            + 0.20 * (1.0 - continuity)
            + 0.10 * isolation
            + 0.10 * conflict,
            0.0,
            1.0,
        ))
        isolated = len(keys) <= self.config.artifact_isolated_component_cells
        covered = mean_coverage_count >= self.config.artifact_min_other_pass_coverage
        stronger_conflict = conflict >= 0.5 and covered
        suppressed = isolated or stronger_conflict or (covered and score >= self.config.artifact_suppression_threshold)
        if stronger_conflict:
            classification = "conflicting-pass-specific-geometry"
        elif isolated:
            classification = "floating-or-isolated-fragment"
        elif suppressed:
            classification = "non-persistent-object"
        elif continuity >= self.config.artifact_structural_continuity_threshold:
            classification = "retained-structural-single-pass"
        else:
            classification = "retained-insufficient-coverage"
        return ArtifactComponentReport(
            component_id=f"{pass_id}:{number}",
            pass_id=pass_id,
            cell_count=len(keys),
            source_point_count=sum(item["point_count"] for item in observations),
            bounds_min=bounds_min.tolist(),
            bounds_max=bounds_max.tolist(),
            observation_support=support,
            other_pass_coverage=coverage,
            structural_continuity=continuity,
            conflict_proximity=conflict,
            isolation_score=isolation,
            artifact_score=score,
            classification=classification,
            suppressed=suppressed,
        )

    @staticmethod
    def _write_rejected_ply(output, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii") as handle:
            handle.write("ply\nformat ascii 1.0\ncomment Proximap rejected transient and artifact geometry\n")
            handle.write(f"element vertex {len(output.rejected_points)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("property ushort source_pass_index\nend_header\n")
            for point, pass_index in zip(output.rejected_points, output.rejected_pass_indices):
                handle.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} 255 32 32 {pass_index}\n")
