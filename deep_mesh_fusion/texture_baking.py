from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from scipy import ndimage

from .models import DeepMeshFusionConfig, TextureBakeResult, TextureBakeSummary, TextureCameraUsage
from .photogrammetry import PhotogrammetryPreparationOutput, project_camera_points


@dataclass
class TextureBakeOutput:
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    face_uvs: np.ndarray
    atlas: np.ndarray
    texture_confidence: np.ndarray
    valid_texels: np.ndarray
    camera_usage: List[TextureCameraUsage]
    summary: TextureBakeSummary
    source_hashes: dict


class IntelligentTextureBakingService:
    """Visibility-driven, confidence-aware multi-camera texture selection and baking."""

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config

    def bake(self, preparation: PhotogrammetryPreparationOutput, geometry_confidence=None) -> TextureBakeOutput:
        import open3d as o3d

        vertices = np.asarray(preparation.mesh_vertices, dtype=float)
        faces = np.asarray(preparation.mesh_faces, dtype=np.int64)
        confidence = np.ones(len(vertices), dtype=float) if geometry_confidence is None else np.asarray(geometry_confidence, dtype=float)
        if len(confidence) != len(vertices):
            raise ValueError("geometry_confidence must have one value per mesh vertex")
        atlas_size = self.config.texture_atlas_size
        grid = int(np.ceil(np.sqrt(len(faces))))
        cell = atlas_size // max(grid, 1)
        padding = self.config.texture_atlas_padding
        if cell <= padding * 2 + 2:
            raise ValueError("Texture atlas is too small for the mesh face count and configured padding")

        atlas = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)
        texture_confidence = np.zeros((atlas_size, atlas_size), dtype=np.float32)
        valid_texels = np.zeros((atlas_size, atlas_size), dtype=bool)
        face_uvs = np.zeros((len(faces), 3, 2), dtype=float)
        images, gains = self._load_images(preparation)
        inverse = np.linalg.inv(np.asarray(preparation.registration.transform))
        usage_selected = np.zeros(len(preparation.dataset.cameras), dtype=np.int64)
        usage_blended = np.zeros(len(preparation.dataset.cameras), dtype=np.int64)
        usage_score_sum = np.zeros(len(preparation.dataset.cameras), dtype=float)

        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces.astype(np.int32))
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        cameras = preparation.dataset.cameras
        quality = preparation.texture_quality
        relationship = preparation.camera_validation
        transformed_centers = preparation.aligned_camera_centers
        epsilon = self.config.effective_photogrammetry_voxel_size() * self.config.photogrammetry_visibility_epsilon_multiplier
        blended_texels = 0

        for face_index, face in enumerate(faces):
            row, column = divmod(face_index, grid)
            x0, y0 = column * cell + padding, row * cell + padding
            x1, y1 = (column + 1) * cell - padding - 1, (row + 1) * cell - padding - 1
            face_uvs[face_index] = np.asarray([
                [(x0 + 0.5) / atlas_size, 1.0 - (y0 + 0.5) / atlas_size],
                [(x1 + 0.5) / atlas_size, 1.0 - (y0 + 0.5) / atlas_size],
                [(x0 + 0.5) / atlas_size, 1.0 - (y1 + 0.5) / atlas_size],
            ])
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            bary_u = (xx - x0 + 0.5) / max(x1 - x0 + 1, 1)
            bary_v = (yy - y0 + 0.5) / max(y1 - y0 + 1, 1)
            inside = bary_u + bary_v <= 1.0
            if not np.any(inside):
                continue
            pixel_x, pixel_y = xx[inside], yy[inside]
            u, v = bary_u[inside], bary_v[inside]
            weights = np.column_stack((1.0 - u - v, u, v))
            world_points = weights @ vertices[face]
            geometry_score = np.clip(weights @ confidence[face], 0.0, 1.0)
            photo_points = self._transform_points(world_points, inverse)
            triangle = vertices[face]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal /= max(np.linalg.norm(normal), 1e-12)
            camera_colors, camera_scores = [], []
            for camera_index, camera in enumerate(cameras):
                if images[camera_index] is None or not relationship[camera_index].relationship_valid:
                    camera_colors.append(np.zeros((len(world_points), 3))); camera_scores.append(np.zeros(len(world_points))); continue
                rotation = np.asarray(camera.rotation_world_to_camera); translation = np.asarray(camera.translation_world_to_camera)
                camera_space = photo_points @ rotation.T + translation
                image_x, image_y, depth = project_camera_points(camera, camera_space)
                in_frame = (depth > 1e-6) & (image_x >= 0) & (image_x < camera.width - 1) & (image_y >= 0) & (image_y < camera.height - 1)
                visible = np.zeros(len(world_points), dtype=bool)
                indices = np.flatnonzero(in_frame)
                if len(indices):
                    vectors = world_points[indices] - transformed_centers[camera_index]
                    distances = np.linalg.norm(vectors, axis=1)
                    directions = np.divide(vectors, distances[:, None], out=np.zeros_like(vectors), where=distances[:, None] > 1e-12)
                    rays = np.hstack((np.repeat(transformed_centers[camera_index][None, :], len(indices), axis=0), directions)).astype(np.float32)
                    hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
                    visible[indices] = np.isfinite(hits) & (np.abs(hits - distances) <= epsilon)
                view_vectors = transformed_centers[camera_index] - world_points
                view_vectors /= np.maximum(np.linalg.norm(view_vectors, axis=1, keepdims=True), 1e-12)
                angle_score = np.clip((np.abs(view_vectors @ normal) - 0.10) / 0.90, 0.0, 1.0)
                score = quality[camera_index].score * (0.20 + 0.80 * angle_score) * visible
                colors = self._bilinear(images[camera_index], image_x, image_y) * gains[camera_index]
                camera_colors.append(np.clip(colors, 0.0, 1.0)); camera_scores.append(score)
            colors = np.stack(camera_colors, axis=0)
            scores = np.stack(camera_scores, axis=0)
            baked, texel_confidence, selected, blended = self._select_and_blend(colors, scores, geometry_score)
            observed = selected[:, 0] >= 0
            rgba = np.zeros((len(world_points), 4), dtype=np.uint8)
            rgba[observed, :3] = np.rint(np.clip(baked[observed], 0.0, 1.0) * 255).astype(np.uint8)
            rgba[observed, 3] = 255
            atlas[pixel_y, pixel_x] = rgba
            texture_confidence[pixel_y, pixel_x] = texel_confidence
            valid_texels[pixel_y, pixel_x] = observed
            blended_texels += int(np.count_nonzero(blended))
            for camera_index in range(len(cameras)):
                primary = selected[:, 0] == camera_index
                participating = np.any(selected == camera_index, axis=1)
                usage_selected[camera_index] += np.count_nonzero(primary)
                usage_blended[camera_index] += np.count_nonzero(participating & blended)
                usage_score_sum[camera_index] += float(np.sum(scores[camera_index, primary]))
            self._pad_tile(atlas, texture_confidence, valid_texels, row * cell, column * cell, cell, padding)

        interior_texels = int(np.count_nonzero(valid_texels))
        alpha_texels = int(np.count_nonzero(atlas[:, :, 3]))
        uncovered = max(0, self._triangle_texel_capacity(len(faces), cell, padding) - interior_texels)
        confidence_values = texture_confidence[valid_texels]
        mean_confidence = float(np.mean(confidence_values)) if len(confidence_values) else 0.0
        high_fraction = float(np.mean(confidence_values >= self.config.texture_high_confidence_threshold)) if len(confidence_values) else 0.0
        low_threshold = max(self.config.texture_min_observation_score, self.config.texture_min_geometry_confidence)
        low_fraction = float(np.mean(confidence_values < low_threshold)) if len(confidence_values) else 1.0
        coverage = interior_texels / max(interior_texels + uncovered, 1)
        seams = self._seam_count(faces)
        warnings = []
        if uncovered: warnings.append("Some UV texels have no reliable visible camera observation and remain transparent")
        if low_fraction > 0.20: warnings.append("More than 20% of textured texels have low combined geometry/texture confidence")
        if not preparation.summary.texture_ready: warnings.append("Milestone 8 photogrammetry preparation was not texture-ready")
        texture_ready = bool(preparation.summary.texture_ready and coverage >= self.config.photogrammetry_min_camera_mesh_coverage and mean_confidence >= self.config.texture_min_observation_score)
        summary = TextureBakeSummary(
            face_count=len(faces), atlas_size=atlas_size,
            atlas_utilization=alpha_texels / float(atlas_size * atlas_size),
            textured_texel_count=interior_texels, blended_texel_count=blended_texels,
            uncovered_texel_count=uncovered, texture_coverage=coverage, seam_count=seams,
            mean_texture_confidence=mean_confidence, high_confidence_fraction=high_fraction,
            low_confidence_fraction=low_fraction, texture_ready=texture_ready, warnings=warnings,
        )
        camera_usage = [TextureCameraUsage(
            image_id=camera.image_id, image_name=camera.image_name,
            selected_texel_count=int(usage_selected[index]), blended_texel_count=int(usage_blended[index]),
            mean_selection_score=float(usage_score_sum[index] / max(usage_selected[index], 1)),
        ) for index, camera in enumerate(cameras)]
        return TextureBakeOutput(vertices, faces, face_uvs, atlas, texture_confidence, valid_texels, camera_usage, summary, preparation.dataset.source_hashes)

    def _select_and_blend(self, colors, scores, geometry_confidence):
        pixel_count = scores.shape[1]
        order = np.argsort(scores, axis=0)[::-1]
        selected = np.full((pixel_count, self.config.texture_max_blend_cameras), -1, dtype=np.int32)
        baked = np.zeros((pixel_count, 3), dtype=float)
        confidence = np.zeros(pixel_count, dtype=float)
        blended = np.zeros(pixel_count, dtype=bool)
        for pixel in range(pixel_count):
            best = int(order[0, pixel])
            if scores[best, pixel] < self.config.texture_min_observation_score:
                continue
            chosen = [best]
            for candidate in order[1:, pixel]:
                candidate = int(candidate)
                if len(chosen) >= self.config.texture_max_blend_cameras or scores[candidate, pixel] < self.config.texture_min_observation_score:
                    break
                disagreement = np.linalg.norm(colors[candidate, pixel] - colors[best, pixel]) / np.sqrt(3.0)
                if disagreement <= self.config.texture_color_disagreement:
                    chosen.append(candidate)
            selected[pixel, :len(chosen)] = chosen
            camera_scores = scores[chosen, pixel]
            blend_weights = camera_scores ** 2; blend_weights /= max(np.sum(blend_weights), 1e-12)
            selected_colors = colors[chosen, pixel]
            baked[pixel] = np.sum(selected_colors * blend_weights[:, None], axis=0)
            observation = 1.0 - float(np.prod(1.0 - np.clip(camera_scores, 0.0, 1.0)))
            agreement = 1.0 - float(np.clip(np.sum(np.linalg.norm(selected_colors - baked[pixel], axis=1) * blend_weights) / np.sqrt(3.0), 0.0, 1.0))
            confidence[pixel] = geometry_confidence[pixel] * (0.75 * observation + 0.25 * agreement)
            blended[pixel] = len(chosen) > 1
        return baked, confidence, selected, blended

    @staticmethod
    def _load_images(preparation):
        arrays, means = [], []
        for camera in preparation.dataset.cameras:
            path = Path(camera.image_path)
            if not path.is_file(): arrays.append(None); means.append(None); continue
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=float) / 255.0
            arrays.append(array); means.append(np.mean(array.reshape((-1, 3)), axis=0))
        available = [mean for mean in means if mean is not None]
        target = np.median(np.asarray(available), axis=0) if available else np.ones(3)
        gains = [np.ones(3) if mean is None else np.clip(target / np.maximum(mean, 1e-3), 0.75, 1.25) for mean in means]
        return arrays, gains

    @staticmethod
    def _bilinear(image, x, y):
        if image is None: return np.zeros((len(x), 3))
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x0 = np.clip(x0, 0, image.shape[1] - 1); y0 = np.clip(y0, 0, image.shape[0] - 1)
        x1 = np.clip(x0 + 1, 0, image.shape[1] - 1); y1 = np.clip(y0 + 1, 0, image.shape[0] - 1)
        wx, wy = (x - x0)[:, None], (y - y0)[:, None]
        return (image[y0, x0] * (1 - wx) * (1 - wy) + image[y0, x1] * wx * (1 - wy) + image[y1, x0] * (1 - wx) * wy + image[y1, x1] * wx * wy)

    @staticmethod
    def _pad_tile(atlas, confidence, valid, top, left, cell, padding):
        tile_valid = valid[top:top + cell, left:left + cell]
        if not np.any(tile_valid): return
        distances, indices = ndimage.distance_transform_edt(~tile_valid, return_indices=True)
        fill = (~tile_valid) & (distances <= padding)
        tile_atlas = atlas[top:top + cell, left:left + cell]
        tile_confidence = confidence[top:top + cell, left:left + cell]
        tile_atlas[fill] = tile_atlas[indices[0][fill], indices[1][fill]]
        tile_confidence[fill] = tile_confidence[indices[0][fill], indices[1][fill]]

    @staticmethod
    def _triangle_texel_capacity(face_count, cell, padding):
        side = max(cell - padding * 2, 1)
        return face_count * side * (side + 1) // 2

    @staticmethod
    def _seam_count(faces):
        edges = {}
        for face_index, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = tuple(sorted((int(a), int(b)))); edges.setdefault(edge, []).append(face_index)
        return sum(len(owners) == 2 for owners in edges.values())

    @staticmethod
    def _transform_points(points, transform):
        return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]

    def export(self, output: TextureBakeOutput, root: str) -> TextureBakeResult:
        target = Path(root).resolve(); target.mkdir(parents=True, exist_ok=True)
        atlas_path = target / "environment_albedo.png"
        confidence_path = target / "texture_confidence.png"
        obj_path = target / "textured_environment.obj"
        material_path = target / "textured_environment.mtl"
        report_path = target / "texture_baking.json"
        Image.fromarray(output.atlas, mode="RGBA").save(atlas_path)
        heat = self._confidence_heatmap(output.texture_confidence, output.valid_texels)
        Image.fromarray(heat, mode="RGBA").save(confidence_path)
        self._write_obj(output, obj_path, material_path, atlas_path.name)
        payload = {
            "schema_version": 1,
            "strategy": "confidence-aware-multi-camera-projection",
            "uv_strategy": "non-overlapping-per-face-atlas",
            "seam_strategy": ["cross-camera-color-normalization", "compatible-view-blending", "transparent-atlas-padding"],
            "hole_treatment": "unobserved texels remain transparent and are not synthesized",
            "summary": asdict(output.summary),
            "camera_usage": [asdict(item) for item in output.camera_usage],
            "source_hashes": output.source_hashes,
        }
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(report_path)
        return TextureBakeResult(str(obj_path), str(material_path), str(atlas_path), str(confidence_path), str(report_path), output.summary)

    @staticmethod
    def _confidence_heatmap(confidence, valid):
        normalized = np.clip(confidence, 0.0, 1.0)
        rgba = np.zeros((*confidence.shape, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.rint((1.0 - normalized) * 255).astype(np.uint8)
        rgba[:, :, 1] = np.rint(normalized * 255).astype(np.uint8)
        rgba[:, :, 3] = valid.astype(np.uint8) * 255
        return rgba

    @staticmethod
    def _write_obj(output, obj_path, material_path, atlas_name):
        material_path.write_text(
            "newmtl ProximapEnvironment\nKa 1 1 1\nKd 1 1 1\nd 1\nillum 2\nmap_Kd " + atlas_name + "\n",
            encoding="utf-8",
        )
        with obj_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"mtllib {material_path.name}\nusemtl ProximapEnvironment\n")
            for face in output.mesh_faces:
                for vertex in output.mesh_vertices[face]:
                    handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
            for face_uv in output.face_uvs:
                for uv in face_uv:
                    handle.write(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")
            for face_index in range(len(output.mesh_faces)):
                base = face_index * 3 + 1
                handle.write(f"f {base}/{base} {base + 1}/{base + 1} {base + 2}/{base + 2}\n")


DeepMeshFusionTextureBakingService = IntelligentTextureBakingService
