from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .models import (
    CameraMeshValidation,
    DeepMeshFusionConfig,
    PhotogrammetryCamera,
    PhotogrammetryPreparationResult,
    PhotogrammetryPreparationSummary,
    PhotogrammetryRegistrationMetrics,
    TextureSourceQuality,
)


@dataclass
class PhotogrammetryDataset:
    model_path: str
    image_root: str
    dense_cloud_path: Optional[str]
    points: np.ndarray
    colors: np.ndarray
    cameras: List[PhotogrammetryCamera]
    source_hashes: Dict[str, str]


@dataclass
class PhotogrammetryPreparationOutput:
    dataset: PhotogrammetryDataset
    registration: PhotogrammetryRegistrationMetrics
    aligned_points: np.ndarray
    aligned_camera_centers: np.ndarray
    camera_validation: List[CameraMeshValidation]
    texture_quality: List[TextureSourceQuality]
    face_view_counts: np.ndarray
    face_quality: np.ndarray
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    summary: PhotogrammetryPreparationSummary


def project_camera_points(camera: PhotogrammetryCamera, camera_points: np.ndarray):
    """Project camera-space points with the COLMAP model's distortion convention."""
    points = np.asarray(camera_points, dtype=float)
    depth = points[:, 2]
    x = np.divide(points[:, 0], depth, out=np.zeros_like(depth), where=depth > 1e-12)
    y = np.divide(points[:, 1], depth, out=np.zeros_like(depth), where=depth > 1e-12)
    model, params = camera.camera_model, camera.camera_parameters
    radius2 = x * x + y * y
    if model == "SIMPLE_RADIAL":
        radial = 1.0 + params[3] * radius2; x, y = x * radial, y * radial
    elif model == "RADIAL":
        radial = 1.0 + params[3] * radius2 + params[4] * radius2 ** 2; x, y = x * radial, y * radial
    elif model in {"OPENCV", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        k1, k2, p1, p2 = params[4:8]
        if model == "FULL_OPENCV" and len(params) >= 12:
            numerator = 1.0 + k1 * radius2 + k2 * radius2 ** 2 + params[8] * radius2 ** 3
            denominator = 1.0 + params[9] * radius2 + params[10] * radius2 ** 2 + params[11] * radius2 ** 3
            radial = numerator / np.maximum(denominator, 1e-12)
        else:
            radial = 1.0 + k1 * radius2 + k2 * radius2 ** 2
        xy = x * y
        x, y = x * radial + 2 * p1 * xy + p2 * (radius2 + 2 * x * x), y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * xy
    elif model in {"OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        radius = np.sqrt(radius2); theta = np.arctan(radius); theta2 = theta * theta
        if model == "OPENCV_FISHEYE": coefficients = params[4:8]
        elif model == "SIMPLE_RADIAL_FISHEYE": coefficients = [params[3]]
        else: coefficients = params[3:5]
        distorted = theta.copy(); power = theta * theta2
        for coefficient in coefficients:
            distorted += coefficient * power; power *= theta2
        scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
        x, y = x * scale, y * scale
    elif model == "FOV":
        omega = params[4]; radius = np.sqrt(radius2)
        distorted = np.arctan(2.0 * radius * np.tan(omega * 0.5)) / max(omega, 1e-12)
        scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
        x, y = x * scale, y * scale
    return camera.fx * x + camera.cx, camera.fy * y + camera.cy, depth


class ColmapTextModelLoader:
    """Read an immutable COLMAP text model and its referenced source images."""

    def load(self, model_path: str, image_root: str, dense_cloud_path: Optional[str] = None) -> PhotogrammetryDataset:
        model = Path(model_path).resolve()
        images_root = Path(image_root).resolve()
        text_sources = [model / "cameras.txt", model / "images.txt", model / "points3D.txt"]
        binary_sources = [model / "cameras.bin", model / "images.bin", model / "points3D.bin"]
        if all(path.is_file() for path in text_sources):
            required = text_sources
            camera_models = self._read_cameras(required[0])
            cameras = self._read_images(required[1], camera_models, images_root)
            sparse_points, sparse_colors = self._read_points(required[2])
        elif all(path.is_file() for path in binary_sources):
            required = binary_sources
            camera_models = self._read_cameras_binary(required[0])
            cameras = self._read_images_binary(required[1], camera_models, images_root)
            sparse_points, sparse_colors = self._read_points_binary(required[2])
        else:
            raise FileNotFoundError("COLMAP model must contain a complete cameras/images/points3D text or binary file set")
        points, colors = sparse_points, sparse_colors
        sources = required.copy()
        if dense_cloud_path is not None:
            dense = Path(dense_cloud_path).resolve()
            if not dense.is_file():
                raise FileNotFoundError(dense)
            points, colors = self._read_ply(dense)
            sources.append(dense)
        if len(points) < 3:
            raise ValueError("Photogrammetry source contains fewer than three 3D points")
        hashes = {str(path): self._sha256(path) for path in sources}
        for camera in cameras:
            path = Path(camera.image_path)
            if path.is_file():
                camera.image_sha256 = self._sha256(path)
                hashes[str(path)] = camera.image_sha256
        return PhotogrammetryDataset(str(model), str(images_root), str(Path(dense_cloud_path).resolve()) if dense_cloud_path else None, points, colors, cameras, hashes)

    @staticmethod
    def _data_lines(path: Path):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    yield stripped

    def _read_cameras(self, path: Path):
        result = {}
        for line in self._data_lines(path):
            values = line.split()
            camera_id, model, width, height = int(values[0]), values[1], int(values[2]), int(values[3])
            params = [float(value) for value in values[4:]]
            result[camera_id] = (width, height, *self._intrinsics(model, params), model, params)
        return result

    def _read_cameras_binary(self, path: Path):
        models = {
            0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
            3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
            6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
            9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
        }
        result = {}
        with path.open("rb") as handle:
            count = self._unpack(handle, "<Q")[0]
            for _ in range(count):
                camera_id, model_id, width, height = self._unpack(handle, "<iiQQ")
                if model_id not in models:
                    raise ValueError(f"Unsupported COLMAP binary camera model id: {model_id}")
                model, parameter_count = models[model_id]
                params = self._unpack(handle, "<" + "d" * parameter_count)
                result[camera_id] = (width, height, *self._intrinsics(model, params), model, list(params))
        return result

    def _read_images(self, path: Path, camera_models, image_root: Path):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle if not line.lstrip().startswith("#")]
        result = []
        index = 0
        while index < len(lines):
            if not lines[index]:
                index += 1
                continue
            values = lines[index].split()
            index += 1
            if len(values) < 10:
                continue
            image_id = int(values[0]); qvec = np.asarray([float(value) for value in values[1:5]])
            translation = np.asarray([float(value) for value in values[5:8]])
            camera_id = int(values[8]); image_name = " ".join(values[9:])
            if camera_id not in camera_models:
                raise ValueError(f"Image {image_id} references missing camera {camera_id}")
            observation_count = 0
            if index < len(lines):
                observations = lines[index].split()
                index += 1
                observation_count = sum(int(float(observations[offset + 2])) >= 0 for offset in range(0, len(observations) - 2, 3))
            result.append(self._make_camera(image_id, camera_id, image_name, qvec, translation, camera_models, image_root, observation_count))
        if not result:
            raise ValueError("COLMAP model contains no registered images")
        return result

    def _read_images_binary(self, path: Path, camera_models, image_root: Path):
        result = []
        with path.open("rb") as handle:
            count = self._unpack(handle, "<Q")[0]
            for _ in range(count):
                values = self._unpack(handle, "<i7di")
                image_id, qvec, translation, camera_id = values[0], np.asarray(values[1:5]), np.asarray(values[5:8]), values[8]
                name_bytes = bytearray()
                while True:
                    value = handle.read(1)
                    if not value:
                        raise ValueError("Unexpected end of COLMAP images.bin while reading image name")
                    if value == b"\x00": break
                    name_bytes.extend(value)
                image_name = name_bytes.decode("utf-8", errors="replace")
                observation_count = self._unpack(handle, "<Q")[0]
                registered = 0
                for _point in range(observation_count):
                    _x, _y, point_id = self._unpack(handle, "<ddq")
                    registered += point_id >= 0
                result.append(self._make_camera(image_id, camera_id, image_name, qvec, translation, camera_models, image_root, registered))
        if not result:
            raise ValueError("COLMAP model contains no registered images")
        return result

    def _read_points(self, path: Path):
        points, colors = [], []
        for line in self._data_lines(path):
            values = line.split()
            if len(values) >= 8:
                points.append([float(value) for value in values[1:4]])
                colors.append([int(value) / 255.0 for value in values[4:7]])
        return np.asarray(points, dtype=float), np.asarray(colors, dtype=float)

    def _read_points_binary(self, path: Path):
        points, colors = [], []
        with path.open("rb") as handle:
            count = self._unpack(handle, "<Q")[0]
            for _ in range(count):
                values = self._unpack(handle, "<Q3d3BdQ")
                points.append(values[1:4]); colors.append(np.asarray(values[4:7], dtype=float) / 255.0)
                track_length = values[8]
                handle.seek(track_length * 8, 1)
        return np.asarray(points, dtype=float), np.asarray(colors, dtype=float)

    @staticmethod
    def _intrinsics(model, params):
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
            return float(params[0]), float(params[0]), float(params[1]), float(params[2])
        if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
            return tuple(float(value) for value in params[:4])
        raise ValueError(f"Unsupported COLMAP camera model for texture preparation: {model}")

    def _make_camera(self, image_id, camera_id, image_name, qvec, translation, camera_models, image_root, observation_count):
        if camera_id not in camera_models:
            raise ValueError(f"Image {image_id} references missing camera {camera_id}")
        width, height, fx, fy, cx, cy, model, params = camera_models[camera_id]
        rotation = self._qvec_to_rotation(qvec); center = -rotation.T @ translation
        return PhotogrammetryCamera(
            image_id=image_id, camera_id=camera_id, camera_model=model, camera_parameters=list(params), image_name=image_name,
            image_path=str((image_root / image_name).resolve()), image_sha256=None,
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
            rotation_world_to_camera=rotation.tolist(), translation_world_to_camera=np.asarray(translation).tolist(),
            center=center.tolist(), registered_observation_count=int(observation_count),
        )

    @staticmethod
    def _unpack(handle, format_string):
        size = struct.calcsize(format_string); data = handle.read(size)
        if len(data) != size:
            raise ValueError("Unexpected end of COLMAP binary model")
        return struct.unpack(format_string, data)

    @staticmethod
    def _read_ply(path: Path):
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=float)
        colors = np.asarray(cloud.colors, dtype=float) if cloud.has_colors() else np.full((len(points), 3), 0.7)
        return points, colors

    @staticmethod
    def _qvec_to_rotation(qvec):
        norm = np.linalg.norm(qvec)
        if norm <= 1e-12:
            raise ValueError("Invalid zero-length COLMAP camera quaternion")
        w, x, y, z = qvec / norm
        return np.asarray([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ])

    @staticmethod
    def _sha256(path: Path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class PhotogrammetryPreparationService:
    """Register COLMAP/OpenMVS observations to validated LiDAR geometry and assess texture readiness."""

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config
        self.loader = ColmapTextModelLoader()

    def prepare(
        self,
        model_path: str,
        image_root: str,
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        mesh_normals: Optional[np.ndarray] = None,
        dense_cloud_path: Optional[str] = None,
        manual_transform: Optional[np.ndarray] = None,
    ) -> PhotogrammetryPreparationOutput:
        dataset = self.loader.load(model_path, image_root, dense_cloud_path)
        vertices, faces = np.asarray(mesh_vertices, dtype=float), np.asarray(mesh_faces, dtype=np.int64)
        if len(vertices) < 3 or len(faces) < 1:
            raise ValueError("Photogrammetry preparation requires a validated triangle mesh")
        registration, aligned = self._register(dataset.points, vertices, mesh_normals, manual_transform)
        aligned_centers = self._transform_points(np.asarray([camera.center for camera in dataset.cameras]), np.asarray(registration.transform))
        qualities = [self._image_quality(camera) for camera in dataset.cameras]
        camera_validation, face_views, face_quality = self._camera_mesh_analysis(
            dataset.cameras, qualities, np.asarray(registration.transform), vertices, faces, aligned_centers
        )
        areas = np.linalg.norm(np.cross(vertices[faces, 1] - vertices[faces, 0], vertices[faces, 2] - vertices[faces, 0]), axis=1) * 0.5
        total_area = max(float(np.sum(areas)), 1e-12)
        covered = face_views > 0
        multi = face_views >= self.config.photogrammetry_min_views_per_face
        coverage = float(np.sum(areas[covered]) / total_area)
        multi_coverage = float(np.sum(areas[multi]) / total_area)
        mean_quality = float(np.mean([quality.score for quality in qualities])) if qualities else 0.0
        valid_cameras = sum(item.relationship_valid for item in camera_validation)
        warnings = []
        if not registration.accepted: warnings.append("Photogrammetry-to-LiDAR registration requires manual alignment")
        if coverage < self.config.photogrammetry_min_camera_mesh_coverage: warnings.append("Camera coverage is below the configured texture threshold")
        if mean_quality < self.config.photogrammetry_min_image_quality: warnings.append("Mean texture-source quality is below threshold")
        if valid_cameras < 2: warnings.append("Fewer than two cameras have a valid mesh relationship")
        texture_ready = bool(registration.accepted and coverage >= self.config.photogrammetry_min_camera_mesh_coverage and mean_quality >= self.config.photogrammetry_min_image_quality and valid_cameras >= 2)
        summary = PhotogrammetryPreparationSummary(
            camera_count=len(dataset.cameras), valid_camera_count=valid_cameras,
            source_point_count=len(dataset.points), registered=registration.accepted,
            correspondence_count=registration.correspondence_count, mesh_face_count=len(faces),
            covered_face_count=int(np.count_nonzero(covered)), multi_view_face_count=int(np.count_nonzero(multi)),
            coverage=coverage, multi_view_coverage=multi_coverage,
            mean_views_per_covered_face=float(np.mean(face_views[covered])) if np.any(covered) else 0.0,
            mean_texture_source_quality=mean_quality, texture_ready=texture_ready, warnings=warnings,
        )
        return PhotogrammetryPreparationOutput(
            dataset, registration, aligned, aligned_centers, camera_validation, qualities,
            face_views, face_quality, vertices, faces, summary,
        )

    def _register(self, source, target, target_normals, manual_transform):
        import open3d as o3d
        if hasattr(o3d.utility, "random"):
            o3d.utility.random.seed(self.config.random_seed)
        voxel = self.config.effective_photogrammetry_voxel_size()
        if manual_transform is not None:
            transform = np.asarray(manual_transform, dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("manual photogrammetry transform must be a finite 4x4 matrix")
            method = "manual-similarity"
        else:
            source_center, target_center = np.median(source, axis=0), np.median(target, axis=0)
            source_radius = np.percentile(np.linalg.norm(source - source_center, axis=1), 75)
            target_radius = np.percentile(np.linalg.norm(target - target_center, axis=1), 75)
            scale = float(target_radius / max(source_radius, 1e-12))
            initial = np.eye(4); initial[:3, :3] *= scale; initial[:3, 3] = target_center - scale * source_center
            scaled = self._transform_points(source, initial)
            source_cloud = self._cloud(scaled, voxel)
            target_cloud = self._cloud(target, voxel, target_normals)
            source_feature = o3d.pipelines.registration.compute_fpfh_feature(
                source_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
            )
            target_feature = o3d.pipelines.registration.compute_fpfh_feature(
                target_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
            )
            coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_cloud, target_cloud, source_feature, target_feature, True, voxel * 3.0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 3.0)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
            fine = o3d.pipelines.registration.registration_icp(
                source_cloud, target_cloud, voxel * self.config.photogrammetry_registration_distance_multiplier,
                coarse.transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
            )
            final = o3d.pipelines.registration.registration_icp(
                source_cloud, target_cloud, voxel * self.config.photogrammetry_registration_distance_multiplier,
                fine.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
            )
            transform = np.asarray(final.transformation) @ initial
            method = "scale-normalized-fpfh-ransac-point-to-plane-icp"
        aligned = self._transform_points(source, transform)
        scale = float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))
        threshold = voxel * self.config.photogrammetry_registration_distance_multiplier
        source_tree, target_tree = cKDTree(aligned), cKDTree(target)
        distances, target_indices = target_tree.query(aligned, k=1)
        target_distances, source_indices = source_tree.query(target, k=1)
        inliers = distances <= threshold
        mutual = np.zeros(len(source), dtype=bool)
        inlier_indices = np.flatnonzero(inliers)
        mutual[inlier_indices] = source_indices[target_indices[inlier_indices]] == inlier_indices
        selected = distances[inliers]
        fitness = float(np.mean(inliers))
        median = float(np.median(selected)) if len(selected) else float(np.median(distances))
        p95 = float(np.percentile(selected, 95)) if len(selected) else float(np.percentile(distances, 95))
        rmse = float(np.sqrt(np.mean(selected ** 2))) if len(selected) else float(np.sqrt(np.mean(distances ** 2)))
        accepted = bool(fitness >= self.config.photogrammetry_min_registration_fitness and median <= voxel * self.config.photogrammetry_max_median_error_multiplier and np.count_nonzero(inliers) >= 3)
        message = "Photogrammetry source registered to validated LiDAR geometry" if accepted else "Registration quality is insufficient; provide a manual similarity transform"
        metrics = PhotogrammetryRegistrationMetrics(
            transform=transform.tolist(), scale=scale, fitness=fitness, inlier_rmse=rmse,
            median_correspondence_error=median, p95_correspondence_error=p95,
            correspondence_count=int(np.count_nonzero(inliers)),
            mutual_correspondence_ratio=float(np.mean(mutual[inliers])) if np.any(inliers) else 0.0,
            accepted=accepted, requires_manual_alignment=not accepted, method=method, message=message,
        )
        return metrics, aligned

    @staticmethod
    def _cloud(points, voxel, normals=None):
        import open3d as o3d
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if normals is not None and len(normals) == len(points):
            cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
        cloud = cloud.voxel_down_sample(voxel)
        if not cloud.has_normals():
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=50))
        return cloud

    def _camera_mesh_analysis(self, cameras, qualities, transform, vertices, faces, aligned_centers):
        import open3d as o3d
        centers = vertices[faces].mean(axis=1)
        area = np.linalg.norm(np.cross(vertices[faces, 1] - vertices[faces, 0], vertices[faces, 2] - vertices[faces, 0]), axis=1) * 0.5
        total_area = max(float(np.sum(area)), 1e-12)
        inverse = np.linalg.inv(transform)
        face_views = np.zeros(len(faces), dtype=np.uint16)
        face_quality = np.zeros(len(faces), dtype=float)
        legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces.astype(np.int32)))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        validation = []
        for camera, quality, lidar_center in zip(cameras, qualities, aligned_centers):
            photo_centers = self._transform_points(centers, inverse)
            rotation = np.asarray(camera.rotation_world_to_camera); translation = np.asarray(camera.translation_world_to_camera)
            camera_points = photo_centers @ rotation.T + translation
            u, v, depth = project_camera_points(camera, camera_points)
            projected = (depth > 1e-6) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
            indices = np.flatnonzero(projected)
            visible = np.zeros(len(faces), dtype=bool)
            if len(indices):
                vectors = centers[indices] - lidar_center
                distances = np.linalg.norm(vectors, axis=1)
                directions = np.divide(vectors, distances[:, None], out=np.zeros_like(vectors), where=distances[:, None] > 1e-12)
                rays = np.hstack((np.repeat(lidar_center[None, :], len(indices), axis=0), directions)).astype(np.float32)
                hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
                epsilon = self.config.effective_photogrammetry_voxel_size() * self.config.photogrammetry_visibility_epsilon_multiplier
                visible[indices] = np.isfinite(hits) & (np.abs(hits - distances) <= epsilon)
            face_views[visible] = np.minimum(face_views[visible] + 1, np.iinfo(np.uint16).max)
            face_quality[visible] = np.maximum(face_quality[visible], quality.score)
            visible_area = float(np.sum(area[visible]))
            transformed_rotation = transform[:3, :3] / max(float(np.cbrt(abs(np.linalg.det(transform[:3, :3])))), 1e-12)
            forward = transformed_rotation @ (rotation.T @ np.asarray([0., 0., 1.]))
            warnings = []
            if not quality.available: warnings.append("Referenced image is missing")
            if not np.any(projected): warnings.append("Camera frustum does not intersect the mesh")
            if np.any(projected) and not np.any(visible): warnings.append("Projected mesh faces are occluded or inconsistent with the camera pose")
            valid = bool(quality.available and np.any(visible) and np.isfinite(lidar_center).all())
            validation.append(CameraMeshValidation(
                image_id=camera.image_id, image_name=camera.image_name, center=lidar_center.tolist(),
                forward=forward.tolist(), projected_face_count=int(np.count_nonzero(projected)),
                visible_face_count=int(np.count_nonzero(visible)), visible_area=visible_area,
                mesh_coverage=visible_area / total_area, relationship_valid=valid, warnings=warnings,
            ))
        return validation, face_views, face_quality

    @staticmethod
    def _image_quality(camera):
        from PIL import Image
        path = Path(camera.image_path)
        if not path.is_file():
            return TextureSourceQuality(camera.image_id, camera.image_name, False, camera.width, camera.height, camera.width * camera.height / 1e6, 0., 0., 0., 0., 1., 0., ["Image file is unavailable"])
        try:
            with Image.open(path) as image:
                actual_width, actual_height = image.size
                image = image.convert("L")
                image.thumbnail((1600, 1600))
                pixels = np.asarray(image, dtype=float)
            laplacian_variance = float(np.var(ndimage.laplace(pixels)))
            megapixels = actual_width * actual_height / 1e6
            resolution = float(np.clip(megapixels / 2.0, 0., 1.))
            sharpness = float(1.0 - np.exp(-laplacian_variance / 500.0))
            mean = float(np.mean(pixels)); exposure = float(np.clip(1.0 - abs(mean - 127.5) / 127.5, 0., 1.))
            contrast = float(np.clip(np.std(pixels) / 64.0, 0., 1.))
            clipped = float(np.mean((pixels <= 5) | (pixels >= 250)))
            score = float(np.clip(0.40 * sharpness + 0.20 * exposure + 0.15 * contrast + 0.15 * (1.0 - clipped) + 0.10 * resolution, 0., 1.))
            warnings = []
            if (actual_width, actual_height) != (camera.width, camera.height): warnings.append("Image dimensions differ from the COLMAP camera model")
            if resolution < 0.25: warnings.append("Image resolution is low for texture projection")
            if sharpness < 0.3: warnings.append("Image appears soft or motion-blurred")
            if exposure < 0.4: warnings.append("Image exposure is poor")
            if clipped > 0.2: warnings.append("Large image regions are clipped")
            return TextureSourceQuality(camera.image_id, camera.image_name, True, actual_width, actual_height, megapixels, resolution, sharpness, exposure, contrast, clipped, score, warnings)
        except OSError as error:
            return TextureSourceQuality(camera.image_id, camera.image_name, False, camera.width, camera.height, camera.width * camera.height / 1e6, 0., 0., 0., 0., 1., 0., [f"Image could not be decoded: {error}"])

    def export(self, output: PhotogrammetryPreparationOutput, root: str) -> PhotogrammetryPreparationResult:
        target = Path(root).resolve(); target.mkdir(parents=True, exist_ok=True)
        registration_path = target / "photogrammetry_registration.json"
        aligned_path = target / "photogrammetry_aligned.ply"
        camera_path = target / "photogrammetry_cameras.json"
        coverage_path = target / "texture_coverage.ply"
        report_path = target / "texture_preparation.json"
        self._write_points(aligned_path, output.aligned_points, output.dataset.colors)
        self._write_coverage(coverage_path, output)
        self._write_json(registration_path, {
            "schema_version": 1,
            "dataset": {"model_path": output.dataset.model_path, "image_root": output.dataset.image_root, "dense_cloud_path": output.dataset.dense_cloud_path},
            "registration": asdict(output.registration), "source_hashes": output.dataset.source_hashes,
        })
        self._write_json(camera_path, {"schema_version": 1, "cameras": [asdict(item) for item in output.dataset.cameras], "validation": [asdict(item) for item in output.camera_validation], "texture_quality": [asdict(item) for item in output.texture_quality]})
        self._write_json(report_path, {"schema_version": 1, "summary": asdict(output.summary), "registration": asdict(output.registration), "face_view_counts": output.face_view_counts.tolist(), "face_quality": output.face_quality.tolist()})
        return PhotogrammetryPreparationResult(str(registration_path), str(aligned_path), str(camera_path), str(coverage_path), str(report_path), output.summary)

    @staticmethod
    def _transform_points(points, transform):
        return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]

    @staticmethod
    def _write_json(path, payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _write_points(path, points, colors):
        rgb = np.rint(np.clip(colors, 0., 1.) * 255).astype(np.uint8)
        with path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for point, color in zip(points, rgb): handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {color[0]} {color[1]} {color[2]}\n")

    @staticmethod
    def _write_coverage(path, output):
        vertex_views = np.zeros(len(output.mesh_vertices), dtype=float); vertex_quality = np.zeros(len(output.mesh_vertices), dtype=float); counts = np.zeros(len(output.mesh_vertices), dtype=float)
        for face, views, quality in zip(output.mesh_faces, output.face_view_counts, output.face_quality):
            vertex_views[face] += views; vertex_quality[face] += quality; counts[face] += 1
        vertex_views = np.divide(vertex_views, counts, out=np.zeros_like(vertex_views), where=counts > 0)
        vertex_quality = np.divide(vertex_quality, counts, out=np.zeros_like(vertex_quality), where=counts > 0)
        normalized = np.clip(vertex_views / max(1, output.summary.camera_count), 0., 1.)
        rgb = np.rint(np.column_stack((1. - normalized, normalized, np.zeros(len(normalized)))) * 255).astype(np.uint8)
        with path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(output.mesh_vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float view_count\nproperty float texture_quality\n")
            handle.write(f"element face {len(output.mesh_faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            for point, color, views, quality in zip(output.mesh_vertices, rgb, vertex_views, vertex_quality): handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {color[0]} {color[1]} {color[2]} {views:.6f} {quality:.6f}\n")
            for face in output.mesh_faces: handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


PhotogrammetryRegistrationService = PhotogrammetryPreparationService
