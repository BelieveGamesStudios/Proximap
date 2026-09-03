from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .models import (
    DeepMeshFusionConfig,
    FinalAssetQuality,
    FinalAssetResult,
    FinalAssetSummary,
    FinalTextureIssue,
)
from .texture_baking import IntelligentTextureBakingService, TextureBakeOutput


@dataclass
class FinalSurfaceRepairOutput:
    texture: TextureBakeOutput
    initial_issues: List[FinalTextureIssue]
    remaining_issues: List[FinalTextureIssue]
    summary: FinalAssetSummary
    review_map: np.ndarray


class FinalSurfaceTextureRepairService:
    """Inspect a baked asset and apply only bounded, evidence-preserving texture repairs."""

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config
        self.baker = IntelligentTextureBakingService(config)

    def repair(self, baked: TextureBakeOutput, geometry_confidence=None) -> FinalSurfaceRepairOutput:
        geometry = np.ones(len(baked.mesh_vertices), dtype=float) if geometry_confidence is None else np.asarray(geometry_confidence, dtype=float)
        if len(geometry) != len(baked.mesh_vertices):
            raise ValueError("geometry_confidence must have one value per mesh vertex")
        atlas = baked.atlas.copy()
        confidence = baked.texture_confidence.copy()
        valid = baked.valid_texels.copy()
        initial, details = self._inspect(baked, atlas, confidence, valid, geometry)
        repaired_ids = set()
        for issue in initial:
            detail = details.get(issue.issue_id, {})
            if not issue.auto_repairable:
                continue
            if issue.category == "texture-seam" and self._repair_seam(atlas, confidence, detail):
                repaired_ids.add(issue.issue_id)
            elif issue.category in {"missing-texture", "black-region"} and self._repair_bounded_region(atlas, confidence, valid, detail):
                repaired_ids.add(issue.issue_id)
        initial = [replace(issue, repaired=issue.issue_id in repaired_ids) for issue in initial]
        repaired_texture = replace(baked, atlas=atlas, texture_confidence=confidence, valid_texels=valid)
        remaining, _remaining_details = self._inspect(repaired_texture, atlas, confidence, valid, geometry)
        review_map = self._review_map(atlas.shape[:2], initial, remaining)
        summary = self._summary(repaired_texture, geometry, initial, remaining)
        return FinalSurfaceRepairOutput(repaired_texture, initial, remaining, summary, review_map)

    def _inspect(self, baked, atlas, confidence, valid, geometry):
        issues, details = [], {}
        face_pixels = [self._face_pixels(baked.face_uvs[index], atlas.shape[0]) for index in range(len(baked.mesh_faces))]
        issue_number = 1

        world_area = self._face_areas(baked.mesh_vertices, baked.mesh_faces)
        uv_area = self._uv_areas(baked.face_uvs, atlas.shape[0])
        density = np.divide(uv_area, world_area, out=np.zeros_like(uv_area), where=world_area > 1e-12)
        median_density = float(np.median(density[density > 0])) if np.any(density > 0) else 0.0

        for face_index, ((ys, xs), face) in enumerate(zip(face_pixels, baked.mesh_faces)):
            if not len(xs):
                continue
            face_valid = valid[ys, xs]
            coverage = float(np.mean(face_valid))
            bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            geometry_score = float(np.mean(geometry[face]))
            face_confidence = float(np.mean(confidence[ys[face_valid], xs[face_valid]])) if np.any(face_valid) else 0.0
            if 1.0 - coverage > self.config.final_missing_face_fraction:
                missing = ~face_valid
                fraction = float(np.mean(missing))
                repairable = fraction <= self.config.final_max_auto_repair_fraction and self._bounded_mask(missing)
                issue = FinalTextureIssue(f"issue-{issue_number:04d}", "missing-texture", "error", [face_index], bounds, fraction, "Face contains texels with no supported camera observation", repairable)
                issues.append(issue); details[issue.issue_id] = {"ys": ys, "xs": xs, "mask": missing}; issue_number += 1
            if np.any(face_valid):
                colors = atlas[ys[face_valid], xs[face_valid], :3].astype(float) / 255.0
                luminance = colors @ np.asarray([0.2126, 0.7152, 0.0722])
                black = luminance <= self.config.final_black_luminance
                black_fraction = float(np.mean(black))
                if black_fraction > self.config.final_black_region_fraction:
                    pixel_mask = np.zeros(len(xs), dtype=bool); pixel_mask[np.flatnonzero(face_valid)[black]] = True
                    repairable = black_fraction <= self.config.final_max_auto_repair_fraction and face_confidence < self.config.final_min_texture_confidence and self._bounded_mask(pixel_mask)
                    issue = FinalTextureIssue(f"issue-{issue_number:04d}", "black-region", "warning", [face_index], bounds, black_fraction, "Suspicious near-black texture region detected", repairable)
                    issues.append(issue); details[issue.issue_id] = {"ys": ys, "xs": xs, "mask": pixel_mask}; issue_number += 1
                gradient = self._face_gradient(atlas, ys, xs, face_valid)
                if gradient > self.config.final_discontinuity_gradient and face_confidence < 0.60:
                    issue = FinalTextureIssue(f"issue-{issue_number:04d}", "texture-discontinuity", "warning", [face_index], bounds, gradient, "Abrupt texture change is weakly supported by camera evidence", False)
                    issues.append(issue); issue_number += 1
            if median_density > 0 and density[face_index] > 0:
                stretch = max(density[face_index] / median_density, median_density / density[face_index])
                if stretch > self.config.final_texture_stretch_ratio:
                    issue = FinalTextureIssue(f"issue-{issue_number:04d}", "stretched-texture", "warning", [face_index], bounds, float(stretch), "UV texel density differs excessively from the mesh median", False)
                    issues.append(issue); issue_number += 1
            if geometry_score >= self.config.texture_high_confidence_threshold and face_confidence < self.config.final_min_texture_confidence:
                category = "wrong-projection" if coverage > 0.80 else "texture-geometry-mismatch"
                message = "High-confidence geometry has a weak or inconsistent projected appearance"
                issue = FinalTextureIssue(f"issue-{issue_number:04d}", category, "error", [face_index], bounds, 1.0 - face_confidence, message, False)
                issues.append(issue); issue_number += 1

        for edge, owners in self._shared_edges(baked.mesh_faces).items():
            if len(owners) != 2:
                continue
            first, second = owners
            samples = self._seam_samples(edge, first, second, baked.mesh_faces, baked.face_uvs, atlas.shape[0])
            first_xy, second_xy = samples
            first_valid = valid[first_xy[:, 1], first_xy[:, 0]]; second_valid = valid[second_xy[:, 1], second_xy[:, 0]]
            supported = first_valid & second_valid
            if not np.any(supported):
                continue
            first_colors = atlas[first_xy[supported, 1], first_xy[supported, 0], :3].astype(float) / 255.0
            second_colors = atlas[second_xy[supported, 1], second_xy[supported, 0], :3].astype(float) / 255.0
            delta = float(np.mean(np.linalg.norm(first_colors - second_colors, axis=1) / np.sqrt(3.0)))
            if delta > self.config.final_seam_color_delta:
                combined = np.vstack((first_xy, second_xy)); bounds = [int(combined[:, 0].min()), int(combined[:, 1].min()), int(combined[:, 0].max()), int(combined[:, 1].max())]
                seam_confidence = float(np.mean(np.r_[confidence[first_xy[supported, 1], first_xy[supported, 0]], confidence[second_xy[supported, 1], second_xy[supported, 0]]]))
                repairable = delta <= self.config.final_max_repairable_seam_delta and seam_confidence >= self.config.final_min_texture_confidence
                issue = FinalTextureIssue(f"issue-{issue_number:04d}", "texture-seam", "warning", [first, second], bounds, delta, "Adjacent UV charts disagree along a shared mesh edge", repairable)
                issues.append(issue); details[issue.issue_id] = {"first": first_xy[supported], "second": second_xy[supported]}; issue_number += 1

        for usage in baked.camera_usage:
            if usage.selected_texel_count and usage.mean_selection_score < self.config.texture_min_observation_score:
                issue = FinalTextureIssue(f"issue-{issue_number:04d}", "poor-camera-selection", "error", [], [0, 0, atlas.shape[1] - 1, atlas.shape[0] - 1], 1.0 - usage.mean_selection_score, f"Camera {usage.image_name} was selected despite weak evidence", False)
                issues.append(issue); issue_number += 1
        return issues, details

    @staticmethod
    def _repair_seam(atlas, confidence, detail):
        first, second = detail.get("first"), detail.get("second")
        if first is None or not len(first): return False
        first_color = atlas[first[:, 1], first[:, 0], :3].astype(float)
        second_color = atlas[second[:, 1], second[:, 0], :3].astype(float)
        average = np.rint((first_color + second_color) * 0.5).astype(np.uint8)
        atlas[first[:, 1], first[:, 0], :3] = average; atlas[second[:, 1], second[:, 0], :3] = average
        combined_confidence = np.minimum(confidence[first[:, 1], first[:, 0]], confidence[second[:, 1], second[:, 0]])
        confidence[first[:, 1], first[:, 0]] = combined_confidence; confidence[second[:, 1], second[:, 0]] = combined_confidence
        return True

    def _repair_bounded_region(self, atlas, confidence, valid, detail):
        ys, xs, mask = detail.get("ys"), detail.get("xs"), detail.get("mask")
        if ys is None or not np.any(mask): return False
        target_y, target_x = ys[mask], xs[mask]
        source = (~mask) & valid[ys, xs]
        if not np.any(source): return False
        source_coordinates = np.column_stack((ys[source], xs[source]))
        distances, nearest = cKDTree(source_coordinates).query(np.column_stack((target_y, target_x)), k=1)
        if np.max(distances) > self.config.final_max_repair_radius: return False
        nearest_y, nearest_x = source_coordinates[nearest].T
        atlas[target_y, target_x] = atlas[nearest_y, nearest_x]
        confidence[target_y, target_x] = confidence[nearest_y, nearest_x] * 0.85
        valid[target_y, target_x] = True
        return True

    def _summary(self, baked, geometry, initial, remaining):
        interior = baked.valid_texels
        texture_values = baked.texture_confidence[interior]
        texture_score = float(np.mean(texture_values)) if len(texture_values) else 0.0
        geometry_score = float(np.mean(np.clip(geometry, 0.0, 1.0)))
        coverage = float(baked.summary.texture_coverage)
        affected_faces = {face for issue in remaining for face in issue.face_indices}
        consistency = float(np.clip(1.0 - len(affected_faces) / max(len(baked.mesh_faces), 1), 0.0, 1.0))
        overall = float(0.25 * geometry_score + 0.30 * texture_score + 0.25 * coverage + 0.20 * consistency)
        geometry_valid = geometry_score >= self.config.texture_min_geometry_confidence
        texture_valid = texture_score >= self.config.final_min_texture_confidence and not any(issue.severity == "error" for issue in remaining)
        coverage_status = "GOOD" if coverage >= 0.90 else "WARNING" if coverage >= 0.70 else "POOR"
        review_count = len({tuple(issue.atlas_bounds) for issue in remaining})
        ready = bool(geometry_valid and texture_valid and coverage_status == "GOOD" and not remaining)
        warnings = [f"{review_count} texture regions require review"] if review_count else []
        return FinalAssetSummary(
            geometry_valid=geometry_valid, texture_valid=texture_valid, coverage_status=coverage_status,
            detected_issue_count=len(initial), repaired_issue_count=sum(issue.repaired for issue in initial),
            review_region_count=review_count, remaining_issue_count=len(remaining), polished_asset_ready=ready,
            quality=FinalAssetQuality(geometry_score, texture_score, coverage, consistency, overall), warnings=warnings,
        )

    @staticmethod
    def _face_pixels(uv, size):
        points = np.column_stack((uv[:, 0] * size - 0.5, (1.0 - uv[:, 1]) * size - 0.5))
        xmin, ymin = np.floor(points.min(axis=0)).astype(int); xmax, ymax = np.ceil(points.max(axis=0)).astype(int)
        xmin, ymin = max(0, xmin), max(0, ymin); xmax, ymax = min(size - 1, xmax), min(size - 1, ymax)
        yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        p = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
        a, b, c = points
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) <= 1e-12: return np.empty(0, dtype=int), np.empty(0, dtype=int)
        w1 = ((b[1] - c[1]) * (p[:, 0] - c[0]) + (c[0] - b[0]) * (p[:, 1] - c[1])) / denominator
        w2 = ((c[1] - a[1]) * (p[:, 0] - c[0]) + (a[0] - c[0]) * (p[:, 1] - c[1])) / denominator
        inside = (w1 >= -1e-6) & (w2 >= -1e-6) & (w1 + w2 <= 1.0 + 1e-6)
        return yy.ravel()[inside], xx.ravel()[inside]

    @staticmethod
    def _bounded_mask(mask):
        indices = np.flatnonzero(mask)
        return bool(len(indices) and indices.min() > 0 and indices.max() < len(mask) - 1)

    @staticmethod
    def _face_gradient(atlas, ys, xs, face_valid):
        if np.count_nonzero(face_valid) < 3: return 0.0
        colors = atlas[ys[face_valid], xs[face_valid], :3].astype(float) / 255.0
        return float(np.percentile(np.linalg.norm(np.diff(colors, axis=0), axis=1), 90)) if len(colors) > 1 else 0.0

    @staticmethod
    def _face_areas(vertices, faces):
        tri = vertices[faces]
        return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5

    @staticmethod
    def _uv_areas(face_uvs, size):
        tri = face_uvs * size
        first, second = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        return np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) * 0.5

    @staticmethod
    def _shared_edges(faces):
        edges: Dict[Tuple[int, int], List[int]] = {}
        for face_index, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_index)
        return edges

    @staticmethod
    def _seam_samples(edge, first, second, faces, face_uvs, size):
        samples = []
        for face_index in (first, second):
            face = list(faces[face_index]); uv = face_uvs[face_index]
            endpoints = [uv[face.index(edge[0])], uv[face.index(edge[1])]]
            length = np.linalg.norm((endpoints[1] - endpoints[0]) * size)
            count = max(3, int(np.ceil(length)))
            values = np.linspace(0.05, 0.95, count)
            coordinates = endpoints[0][None, :] * (1 - values[:, None]) + endpoints[1][None, :] * values[:, None]
            x = np.clip(np.rint(coordinates[:, 0] * size - 0.5), 0, size - 1).astype(int)
            y = np.clip(np.rint((1.0 - coordinates[:, 1]) * size - 0.5), 0, size - 1).astype(int)
            samples.append(np.column_stack((x, y)))
        count = min(len(samples[0]), len(samples[1]))
        return samples[0][:count], samples[1][:count]

    @staticmethod
    def _review_map(shape, initial, remaining):
        image = np.zeros((*shape, 4), dtype=np.uint8)
        remaining_keys = {(issue.category, tuple(issue.face_indices), tuple(issue.atlas_bounds)) for issue in remaining}
        for issue in initial:
            x0, y0, x1, y1 = issue.atlas_bounds
            key = (issue.category, tuple(issue.face_indices), tuple(issue.atlas_bounds))
            color = [255, 40, 40, 220] if key in remaining_keys else [40, 220, 80, 180]
            image[y0:y1 + 1, x0:x1 + 1] = np.maximum(image[y0:y1 + 1, x0:x1 + 1], color)
        return image

    def export(self, output: FinalSurfaceRepairOutput, root: str) -> FinalAssetResult:
        target = Path(root).resolve(); target.mkdir(parents=True, exist_ok=True)
        texture_path = target / "final_albedo.png"; confidence_path = target / "final_texture_confidence.png"
        review_path = target / "final_review_map.png"; obj_path = target / "final_environment.obj"
        material_path = target / "final_environment.mtl"; report_path = target / "final_asset_validation.json"
        Image.fromarray(output.texture.atlas).save(texture_path)
        Image.fromarray(self.baker._confidence_heatmap(output.texture.texture_confidence, output.texture.valid_texels)).save(confidence_path)
        Image.fromarray(output.review_map).save(review_path)
        self.baker._write_obj(output.texture, obj_path, material_path, texture_path.name)
        payload = {
            "schema_version": 1,
            "summary": asdict(output.summary),
            "initial_issues": [asdict(issue) for issue in output.initial_issues],
            "remaining_issues": [asdict(issue) for issue in output.remaining_issues],
            "repair_policy": "bounded evidence-preserving repairs only; ambiguous projection defects require review",
        }
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"); temporary.replace(report_path)
        return FinalAssetResult(str(obj_path), str(material_path), str(texture_path), str(confidence_path), str(review_path), str(report_path), output.summary)


FinalAssetRepairService = FinalSurfaceTextureRepairService
