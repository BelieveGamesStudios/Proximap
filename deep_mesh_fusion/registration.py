from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .models import DeepMeshFusionConfig, RegistrationMetrics


def axis_aligned_overlap(bounds_a, bounds_b) -> float:
    """Intersection volume divided by the smaller bounding-box volume."""
    a0, a1 = np.asarray(bounds_a[0]), np.asarray(bounds_a[1])
    b0, b1 = np.asarray(bounds_b[0]), np.asarray(bounds_b[1])
    intersection = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    intersection_volume = float(np.prod(intersection))
    volume_a = float(np.prod(np.maximum(a1 - a0, 1e-12)))
    volume_b = float(np.prod(np.maximum(b1 - b0, 1e-12)))
    return intersection_volume / max(min(volume_a, volume_b), 1e-12)


class DeepMeshFusionRegistrationService:
    """Open3D FPFH/RANSAC coarse registration followed by robust point-to-plane ICP."""

    def __init__(self, config: DeepMeshFusionConfig, log_fn: Optional[Callable[[str], None]] = None):
        self.config = config
        self.log = log_fn or (lambda _message: None)

    def register(self, source, target, source_pass_id: str, reference_pass_id: str) -> RegistrationMetrics:
        import open3d as o3d

        voxel = self.config.voxel_size
        if hasattr(o3d.utility, "random"):
            o3d.utility.random.seed(self.config.random_seed)
        source_down = source.voxel_down_sample(voxel)
        target_down = target.voxel_down_sample(voxel)
        if len(source_down.points) < 20 or len(target_down.points) < 20:
            return self._failed(reference_pass_id, "Insufficient downsampled points for registration")

        self._estimate_normals(source_down)
        self._estimate_normals(target_down)
        initial_overlap = axis_aligned_overlap(self._bounds(source), self._bounds(target))

        identity_eval = o3d.pipelines.registration.evaluate_registration(
            source_down, target_down, voxel * self.config.coarse_distance_multiplier, np.eye(4)
        )
        if identity_eval.fitness >= max(self.config.min_registration_fitness, 0.65):
            initial = np.eye(4)
            method = "aligned+point-to-plane-icp"
        else:
            source_fpfh = self._features(source_down)
            target_fpfh = self._features(target_down)
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                True,
                voxel * self.config.coarse_distance_multiplier,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * self.config.coarse_distance_multiplier),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
            )
            initial = result.transformation
            method = "fpfh-ransac+point-to-plane-icp"

        source_full = source.voxel_down_sample(max(voxel * 0.5, 1e-6))
        target_full = target.voxel_down_sample(max(voxel * 0.5, 1e-6))
        self._estimate_normals(source_full)
        self._estimate_normals(target_full)
        robust_loss = o3d.pipelines.registration.TukeyLoss(k=voxel * self.config.coarse_distance_multiplier)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(robust_loss)
        refined = o3d.pipelines.registration.registration_icp(
            source_full,
            target_full,
            voxel * self.config.fine_distance_multiplier,
            initial,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
        )
        transformed_bounds = self._transformed_bounds(source, refined.transformation)
        target_bounds = self._bounds(target)
        overlap = axis_aligned_overlap(transformed_bounds, target_bounds)
        correspondence_count = len(refined.correspondence_set)
        rmse = float(refined.inlier_rmse) if np.isfinite(refined.inlier_rmse) else None
        accepted = (
            refined.fitness >= self.config.min_registration_fitness
            and rmse is not None
            and rmse <= voxel * self.config.max_registration_rmse_multiplier
            and overlap >= self.config.min_overlap_ratio
        )
        message = "Registration accepted" if accepted else "Registration below configured quality thresholds"
        self.log(f"[DEEP_FUSION] {source_pass_id}: fitness={refined.fitness:.3f}, rmse={refined.inlier_rmse:.4f}, overlap={overlap:.3f}")
        return RegistrationMetrics(
            reference_pass_id=reference_pass_id,
            transform=np.asarray(refined.transformation).tolist(),
            initial_overlap=initial_overlap,
            overlap_ratio=overlap,
            fitness=float(refined.fitness),
            inlier_rmse=rmse,
            correspondence_count=correspondence_count,
            method=method,
            accepted=accepted,
            requires_manual_alignment=not accepted,
            message=message,
        )

    def evaluate_transform(self, source, target, transform, reference_pass_id: str) -> RegistrationMetrics:
        """Score a user-provided source-to-reference transform with the same gates as automatic registration."""
        import open3d as o3d

        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("A manual transform must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("A manual transform must have homogeneous bottom row [0, 0, 0, 1]")
        source_down = source.voxel_down_sample(max(self.config.voxel_size * 0.5, 1e-6))
        target_down = target.voxel_down_sample(max(self.config.voxel_size * 0.5, 1e-6))
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_down,
            target_down,
            self.config.voxel_size * self.config.fine_distance_multiplier,
            matrix,
        )
        overlap = axis_aligned_overlap(self._transformed_bounds(source, matrix), self._bounds(target))
        rmse = float(evaluation.inlier_rmse) if np.isfinite(evaluation.inlier_rmse) else None
        accepted = (
            evaluation.fitness >= self.config.min_registration_fitness
            and rmse is not None
            and rmse <= self.config.voxel_size * self.config.max_registration_rmse_multiplier
            and overlap >= self.config.min_overlap_ratio
        )
        return RegistrationMetrics(
            reference_pass_id=reference_pass_id,
            transform=matrix.tolist(),
            initial_overlap=axis_aligned_overlap(self._bounds(source), self._bounds(target)),
            overlap_ratio=overlap,
            fitness=float(evaluation.fitness),
            inlier_rmse=rmse,
            correspondence_count=len(evaluation.correspondence_set),
            method="manual",
            accepted=accepted,
            requires_manual_alignment=not accepted,
            message="Manual transform accepted" if accepted else "Manual transform below configured quality thresholds",
        )

    def _estimate_normals(self, cloud) -> None:
        import open3d as o3d
        radius = self.config.voxel_size * self.config.normal_radius_multiplier
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
        cloud.normalize_normals()

    def _features(self, cloud):
        import open3d as o3d
        radius = self.config.voxel_size * self.config.feature_radius_multiplier
        return o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
        )

    @staticmethod
    def _bounds(cloud):
        box = cloud.get_axis_aligned_bounding_box()
        return np.asarray(box.min_bound), np.asarray(box.max_bound)

    @staticmethod
    def _transformed_bounds(cloud, transform):
        box = cloud.get_axis_aligned_bounding_box()
        corners = np.asarray(box.get_box_points())
        moved = corners @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]
        return moved.min(axis=0), moved.max(axis=0)

    @staticmethod
    def _failed(reference_pass_id: str, message: str) -> RegistrationMetrics:
        return RegistrationMetrics(
            reference_pass_id=reference_pass_id,
            accepted=False,
            requires_manual_alignment=True,
            message=message,
        )
