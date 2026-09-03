from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from .models import DeepMeshFusionConfig, PassDiagnostics


class ScanDiagnosticsService:
    """Computes bounded-cost geometric diagnostics for one independent pass."""

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config

    def analyze(self, cloud) -> PassDiagnostics:
        points = np.asarray(cloud.points, dtype=np.float64)
        if len(points) < 3:
            raise ValueError("A scan pass must contain at least three finite points")

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points) < 3:
            raise ValueError("A scan pass must contain at least three finite points")

        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        dimensions = bounds_max - bounds_min
        diagonal = float(np.linalg.norm(dimensions))
        volume = float(np.prod(np.maximum(dimensions, self.config.voxel_size)))

        sample = self._deterministic_sample(points, 100_000)
        tree = cKDTree(sample)
        distances, _ = tree.query(sample, k=min(7, len(sample)))
        neighbor_distance = distances[:, -1] if distances.ndim == 2 else distances
        median_distance = float(np.median(neighbor_distance))
        mad = float(np.median(np.abs(neighbor_distance - median_distance)))
        sparse_cutoff = median_distance + 3.0 * max(mad, np.finfo(float).eps)
        sparse_fraction = float(np.mean(neighbor_distance > sparse_cutoff))

        mean_d = np.mean(distances[:, 1:], axis=1) if distances.ndim == 2 and distances.shape[1] > 1 else neighbor_distance
        outlier_cutoff = float(np.mean(mean_d) + self.config.outlier_std_ratio * np.std(mean_d))
        outlier_fraction = float(np.mean(mean_d > outlier_cutoff))
        component_count, largest_fraction = self._components(sample, tree)
        normal_consistency, surface_variation = self._surface_metrics(cloud, sample)

        warnings = []
        if sparse_fraction > 0.10:
            warnings.append("Large sparse-point fraction detected")
        if outlier_fraction > 0.05:
            warnings.append("Potential outliers or floating geometry detected")
        if largest_fraction < 0.90:
            warnings.append("Multiple significant disconnected components detected")

        colors = getattr(cloud, "colors", None)
        normals = getattr(cloud, "normals", None)
        return PassDiagnostics(
            point_count=int(len(points)),
            bounds_min=bounds_min.tolist(),
            bounds_max=bounds_max.tolist(),
            dimensions=dimensions.tolist(),
            centroid=points.mean(axis=0).tolist(),
            bbox_diagonal=diagonal,
            density_points_per_volume=float(len(points) / volume),
            median_neighbor_distance=median_distance,
            sparse_fraction=sparse_fraction,
            outlier_fraction=outlier_fraction,
            component_count=component_count,
            largest_component_fraction=largest_fraction,
            has_colors=colors is not None and len(colors) == len(cloud.points),
            has_normals=normals is not None and len(normals) == len(cloud.points),
            normal_consistency=normal_consistency,
            local_surface_variation=surface_variation,
            warnings=warnings,
        )

    @staticmethod
    def _deterministic_sample(points: np.ndarray, limit: int) -> np.ndarray:
        if len(points) <= limit:
            return points
        return points[np.linspace(0, len(points) - 1, limit, dtype=np.int64)]

    def _components(self, points: np.ndarray, tree: cKDTree) -> Tuple[int, float]:
        # Voxel occupancy gives a stable approximation without quadratic clustering.
        size = max(self.config.voxel_size * 2.0, np.finfo(float).eps)
        voxels = np.unique(np.floor(points / size).astype(np.int64), axis=0)
        occupied = {tuple(v) for v in voxels}
        seen = set()
        sizes = []
        directions = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
        for start in occupied:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            count = 0
            while stack:
                current = stack.pop()
                count += 1
                for delta in directions:
                    neighbor = tuple(current[i] + delta[i] for i in range(3))
                    if neighbor in occupied and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            sizes.append(count)
        return len(sizes), float(max(sizes) / len(voxels)) if sizes else 0.0

    def _surface_metrics(self, cloud, sample: np.ndarray):
        normals = getattr(cloud, "normals", None)
        if normals is None or len(normals) != len(cloud.points):
            return None, None
        normals = np.asarray(normals, dtype=np.float64)
        normals = normals[np.isfinite(np.asarray(cloud.points)).all(axis=1)]
        if len(normals) > len(sample):
            normals = normals[np.linspace(0, len(normals) - 1, len(sample), dtype=np.int64)]
        tree = cKDTree(sample)
        _, indices = tree.query(sample, k=min(7, len(sample)))
        if indices.ndim < 2:
            return None, None
        unit = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        agreement = np.abs(np.sum(unit[:, None, :] * unit[indices[:, 1:]], axis=2))
        consistency = float(np.clip(np.mean(agreement), 0.0, 1.0))
        return consistency, float(1.0 - consistency)
