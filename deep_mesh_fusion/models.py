from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _identity() -> List[List[float]]:
    return np.eye(4, dtype=float).tolist()


@dataclass
class DeepMeshFusionConfig:
    """Parameter-driven fusion settings, expressed in scene units."""

    voxel_size: float = 0.03
    normal_radius_multiplier: float = 2.5
    feature_radius_multiplier: float = 5.0
    coarse_distance_multiplier: float = 2.5
    fine_distance_multiplier: float = 0.75
    min_registration_fitness: float = 0.20
    max_registration_rmse_multiplier: float = 1.5
    min_overlap_ratio: float = 0.05
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    random_seed: int = 7
    analysis_cell_size: Optional[float] = None
    analysis_cell_size_multiplier: float = 4.0
    cross_pass_distance_tolerance_multiplier: float = 0.35
    conflict_distance_multiplier: float = 0.55
    min_normal_agreement: float = 0.70
    missing_neighbor_count: int = 2
    conflict_penalty: float = 0.70
    confidence_observation_weight: float = 0.25
    confidence_density_weight: float = 0.15
    confidence_distance_weight: float = 0.20
    confidence_normal_weight: float = 0.15
    confidence_surface_weight: float = 0.15
    confidence_registration_weight: float = 0.10
    fusion_cell_size: Optional[float] = None
    correspondence_distance_multiplier: float = 0.85
    consensus_huber_delta_multiplier: float = 0.35
    min_consensus_observations: int = 2
    best_observation_margin: float = 0.20
    retain_single_pass_geometry: bool = True
    single_pass_min_neighbors: int = 3
    min_single_pass_region_confidence: float = 0.35
    fusion_region_confidence_weight: float = 0.50
    artifact_cell_size: Optional[float] = None
    artifact_min_persistent_passes: int = 2
    artifact_correspondence_distance_multiplier: float = 0.50
    artifact_min_other_pass_coverage: int = 1
    artifact_structural_continuity_threshold: float = 0.55
    artifact_structural_normal_agreement: float = 0.80
    artifact_isolated_component_cells: int = 2
    artifact_suppression_threshold: float = 0.65
    architecture_up_axis: str = "y"
    architecture_plane_distance: Optional[float] = None
    architecture_plane_min_points: int = 80
    architecture_plane_min_ratio: float = 0.015
    architecture_max_planes: int = 48
    architecture_orientation_threshold: float = 0.88
    architecture_grid_size: Optional[float] = None
    architecture_grid_closing_iterations: int = 0
    opening_min_width: float = 0.35
    opening_min_height: float = 0.35
    opening_min_area: float = 0.15
    doorway_floor_tolerance: float = 0.20
    complex_reconstruction_min_points: int = 40
    complex_reconstruction_method: str = "screened-poisson"
    complex_poisson_depth: int = 8
    complex_poisson_scale: float = 1.05
    complex_poisson_density_quantile: float = 0.0
    complex_min_confidence: float = 0.20
    complex_outlier_neighbors: int = 20
    complex_outlier_std_ratio: float = 2.0
    complex_normal_radius_multiplier: float = 3.0
    complex_alpha_multiplier: float = 3.0
    mesh_merge_tolerance_multiplier: float = 0.15
    gap_min_boundary_confidence: float = 0.72
    gap_min_plane_confidence: float = 0.78
    gap_max_planar_area: float = 1.5
    gap_max_complex_perimeter: float = 0.80
    gap_max_complex_area: float = 0.20
    gap_intentional_opening_overlap: float = 0.50
    gap_confidence_penalty: float = 0.15
    validation_degenerate_area_ratio: float = 1e-5
    validation_max_triangle_aspect_ratio: float = 12.0
    validation_normal_alignment: float = 0.50
    validation_discontinuity_angle_degrees: float = 35.0
    validation_review_cell_size: Optional[float] = None
    validation_self_intersection_pair_limit: int = 2_000_000
    validation_min_completeness: float = 0.85
    validation_min_surface_quality: float = 0.80
    validation_min_consistency: float = 0.80
    validation_min_confidence: float = 0.70
    validation_min_overall_quality: float = 0.82
    photogrammetry_voxel_size: Optional[float] = None
    photogrammetry_registration_distance_multiplier: float = 2.0
    photogrammetry_min_registration_fitness: float = 0.20
    photogrammetry_max_median_error_multiplier: float = 2.0
    photogrammetry_min_camera_mesh_coverage: float = 0.60
    photogrammetry_min_views_per_face: int = 2
    photogrammetry_min_image_quality: float = 0.45
    photogrammetry_visibility_epsilon_multiplier: float = 0.50
    texture_atlas_size: int = 2048
    texture_atlas_padding: int = 2
    texture_max_blend_cameras: int = 3
    texture_min_observation_score: float = 0.25
    texture_color_disagreement: float = 0.35
    texture_min_geometry_confidence: float = 0.20
    texture_high_confidence_threshold: float = 0.75
    final_seam_color_delta: float = 0.18
    final_max_repairable_seam_delta: float = 0.40
    final_texture_stretch_ratio: float = 4.0
    final_missing_face_fraction: float = 0.01
    final_black_luminance: float = 0.025
    final_black_region_fraction: float = 0.01
    final_discontinuity_gradient: float = 0.35
    final_min_texture_confidence: float = 0.35
    final_max_repair_radius: int = 4
    final_max_auto_repair_fraction: float = 0.05
    tour_quality_profile: str = "virtual-tour-standard"
    tour_min_geometry: float = 0.85
    tour_min_completeness: float = 0.90
    tour_min_surface_consistency: float = 0.85
    tour_min_texture_coverage: float = 0.85
    tour_min_texture_quality: float = 0.75
    tour_max_critical_defects: int = 0
    tour_max_review_regions: int = 0
    tour_require_source_integrity: bool = True

    def validate(self) -> None:
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be greater than zero")
        if self.outlier_neighbors < 2:
            raise ValueError("outlier_neighbors must be at least 2")
        if not 0 <= self.min_registration_fitness <= 1:
            raise ValueError("min_registration_fitness must be between 0 and 1")
        if self.analysis_cell_size is not None and self.analysis_cell_size <= 0:
            raise ValueError("analysis_cell_size must be greater than zero")
        if self.analysis_cell_size_multiplier <= 0:
            raise ValueError("analysis_cell_size_multiplier must be greater than zero")
        if self.cross_pass_distance_tolerance_multiplier <= 0 or self.conflict_distance_multiplier <= 0:
            raise ValueError("cross-pass distance multipliers must be greater than zero")
        if not 0 <= self.min_normal_agreement <= 1:
            raise ValueError("min_normal_agreement must be between 0 and 1")
        if not 0 <= self.conflict_penalty <= 1:
            raise ValueError("conflict_penalty must be between 0 and 1")
        if self.missing_neighbor_count < 1 or self.missing_neighbor_count > 26:
            raise ValueError("missing_neighbor_count must be between 1 and 26")
        weights = self.confidence_weights()
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("confidence weights must sum to 1.0")
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("confidence weights cannot be negative")
        if self.fusion_cell_size is not None and self.fusion_cell_size <= 0:
            raise ValueError("fusion_cell_size must be greater than zero")
        if self.correspondence_distance_multiplier <= 0 or self.consensus_huber_delta_multiplier <= 0:
            raise ValueError("consensus distance multipliers must be greater than zero")
        if self.min_consensus_observations < 2:
            raise ValueError("min_consensus_observations must be at least 2")
        if not 0 <= self.best_observation_margin <= 1:
            raise ValueError("best_observation_margin must be between 0 and 1")
        if self.single_pass_min_neighbors < 0 or self.single_pass_min_neighbors > 26:
            raise ValueError("single_pass_min_neighbors must be between 0 and 26")
        if not 0 <= self.min_single_pass_region_confidence <= 1:
            raise ValueError("min_single_pass_region_confidence must be between 0 and 1")
        if not 0 <= self.fusion_region_confidence_weight <= 1:
            raise ValueError("fusion_region_confidence_weight must be between 0 and 1")
        if self.artifact_cell_size is not None and self.artifact_cell_size <= 0:
            raise ValueError("artifact_cell_size must be greater than zero")
        if self.artifact_min_persistent_passes < 2:
            raise ValueError("artifact_min_persistent_passes must be at least 2")
        if self.artifact_correspondence_distance_multiplier <= 0:
            raise ValueError("artifact_correspondence_distance_multiplier must be greater than zero")
        if self.artifact_min_other_pass_coverage < 1:
            raise ValueError("artifact_min_other_pass_coverage must be at least 1")
        if not 0 <= self.artifact_structural_continuity_threshold <= 1:
            raise ValueError("artifact_structural_continuity_threshold must be between 0 and 1")
        if not 0 <= self.artifact_structural_normal_agreement <= 1:
            raise ValueError("artifact_structural_normal_agreement must be between 0 and 1")
        if self.artifact_isolated_component_cells < 1:
            raise ValueError("artifact_isolated_component_cells must be at least 1")
        if not 0 <= self.artifact_suppression_threshold <= 1:
            raise ValueError("artifact_suppression_threshold must be between 0 and 1")
        if self.architecture_up_axis not in {"x", "y", "z"}:
            raise ValueError("architecture_up_axis must be x, y, or z")
        if self.architecture_plane_distance is not None and self.architecture_plane_distance <= 0:
            raise ValueError("architecture_plane_distance must be greater than zero")
        if self.architecture_plane_min_points < 3 or self.architecture_max_planes < 1:
            raise ValueError("architecture plane limits are invalid")
        if not 0 < self.architecture_plane_min_ratio <= 1:
            raise ValueError("architecture_plane_min_ratio must be between 0 and 1")
        if not 0 < self.architecture_orientation_threshold <= 1:
            raise ValueError("architecture_orientation_threshold must be between 0 and 1")
        if self.architecture_grid_size is not None and self.architecture_grid_size <= 0:
            raise ValueError("architecture_grid_size must be greater than zero")
        if self.architecture_grid_closing_iterations < 0:
            raise ValueError("architecture_grid_closing_iterations cannot be negative")
        if min(self.opening_min_width, self.opening_min_height, self.opening_min_area) <= 0:
            raise ValueError("opening dimensions must be greater than zero")
        if self.doorway_floor_tolerance < 0:
            raise ValueError("doorway_floor_tolerance cannot be negative")
        if self.complex_reconstruction_min_points < 4 or self.complex_alpha_multiplier <= 0:
            raise ValueError("complex reconstruction parameters are invalid")
        if self.complex_reconstruction_method not in {"screened-poisson", "none"}:
            raise ValueError("complex_reconstruction_method must be screened-poisson or none")
        if not 5 <= self.complex_poisson_depth <= 12 or self.complex_poisson_scale <= 1:
            raise ValueError("complex Poisson depth/scale parameters are invalid")
        if not 0 <= self.complex_poisson_density_quantile < 0.5:
            raise ValueError("complex_poisson_density_quantile must be in [0, 0.5)")
        if not 0 <= self.complex_min_confidence <= 1:
            raise ValueError("complex_min_confidence must be between 0 and 1")
        if self.complex_outlier_neighbors < 2 or self.complex_outlier_std_ratio <= 0 or self.complex_normal_radius_multiplier <= 0:
            raise ValueError("complex residual cleaning parameters are invalid")
        if self.mesh_merge_tolerance_multiplier <= 0:
            raise ValueError("mesh_merge_tolerance_multiplier must be greater than zero")
        if not 0 <= self.gap_min_boundary_confidence <= 1 or not 0 <= self.gap_min_plane_confidence <= 1:
            raise ValueError("gap confidence thresholds must be between 0 and 1")
        if min(self.gap_max_planar_area, self.gap_max_complex_perimeter, self.gap_max_complex_area) <= 0:
            raise ValueError("gap size limits must be greater than zero")
        if not 0 <= self.gap_intentional_opening_overlap <= 1:
            raise ValueError("gap_intentional_opening_overlap must be between 0 and 1")
        if not 0 <= self.gap_confidence_penalty <= 1:
            raise ValueError("gap_confidence_penalty must be between 0 and 1")
        if self.validation_degenerate_area_ratio <= 0 or self.validation_max_triangle_aspect_ratio <= 1:
            raise ValueError("validation triangle thresholds are invalid")
        if not 0 <= self.validation_normal_alignment <= 1:
            raise ValueError("validation_normal_alignment must be between 0 and 1")
        if not 0 < self.validation_discontinuity_angle_degrees < 180:
            raise ValueError("validation_discontinuity_angle_degrees must be between 0 and 180")
        if self.validation_review_cell_size is not None and self.validation_review_cell_size <= 0:
            raise ValueError("validation_review_cell_size must be greater than zero")
        if self.validation_self_intersection_pair_limit < 1:
            raise ValueError("validation_self_intersection_pair_limit must be positive")
        validation_scores = (
            self.validation_min_completeness,
            self.validation_min_surface_quality,
            self.validation_min_consistency,
            self.validation_min_confidence,
            self.validation_min_overall_quality,
        )
        if any(value < 0 or value > 1 for value in validation_scores):
            raise ValueError("validation readiness thresholds must be between 0 and 1")
        if self.photogrammetry_voxel_size is not None and self.photogrammetry_voxel_size <= 0:
            raise ValueError("photogrammetry_voxel_size must be greater than zero")
        if self.photogrammetry_registration_distance_multiplier <= 0 or self.photogrammetry_max_median_error_multiplier <= 0:
            raise ValueError("photogrammetry registration distance multipliers must be positive")
        if not 0 <= self.photogrammetry_min_registration_fitness <= 1:
            raise ValueError("photogrammetry_min_registration_fitness must be between 0 and 1")
        if not 0 <= self.photogrammetry_min_camera_mesh_coverage <= 1:
            raise ValueError("photogrammetry_min_camera_mesh_coverage must be between 0 and 1")
        if self.photogrammetry_min_views_per_face < 1:
            raise ValueError("photogrammetry_min_views_per_face must be positive")
        if not 0 <= self.photogrammetry_min_image_quality <= 1:
            raise ValueError("photogrammetry_min_image_quality must be between 0 and 1")
        if self.photogrammetry_visibility_epsilon_multiplier <= 0:
            raise ValueError("photogrammetry_visibility_epsilon_multiplier must be positive")
        if self.texture_atlas_size < 64:
            raise ValueError("texture_atlas_size must be at least 64 pixels")
        if self.texture_atlas_padding < 1:
            raise ValueError("texture_atlas_padding must be positive")
        if self.texture_max_blend_cameras < 1:
            raise ValueError("texture_max_blend_cameras must be positive")
        if not 0 <= self.texture_min_observation_score <= 1 or not 0 <= self.texture_color_disagreement <= 1:
            raise ValueError("texture selection thresholds must be between 0 and 1")
        if not 0 <= self.texture_min_geometry_confidence <= 1 or not 0 <= self.texture_high_confidence_threshold <= 1:
            raise ValueError("texture confidence thresholds must be between 0 and 1")
        final_unit_thresholds = (
            self.final_seam_color_delta, self.final_max_repairable_seam_delta,
            self.final_black_luminance, self.final_black_region_fraction,
            self.final_discontinuity_gradient, self.final_min_texture_confidence,
            self.final_max_auto_repair_fraction,
        )
        if any(value < 0 or value > 1 for value in final_unit_thresholds):
            raise ValueError("final repair thresholds must be between 0 and 1")
        if self.final_max_repairable_seam_delta < self.final_seam_color_delta:
            raise ValueError("final_max_repairable_seam_delta cannot be below the seam detection threshold")
        if self.final_texture_stretch_ratio <= 1:
            raise ValueError("final_texture_stretch_ratio must be greater than one")
        if not 0 <= self.final_missing_face_fraction <= 1:
            raise ValueError("final_missing_face_fraction must be between 0 and 1")
        if self.final_max_repair_radius < 1:
            raise ValueError("final_max_repair_radius must be positive")
        tour_thresholds = (
            self.tour_min_geometry, self.tour_min_completeness,
            self.tour_min_surface_consistency, self.tour_min_texture_coverage,
            self.tour_min_texture_quality,
        )
        if any(value < 0 or value > 1 for value in tour_thresholds):
            raise ValueError("tour readiness thresholds must be between 0 and 1")
        if self.tour_max_critical_defects < 0 or self.tour_max_review_regions < 0:
            raise ValueError("tour readiness defect limits cannot be negative")
        if not self.tour_quality_profile.strip():
            raise ValueError("tour_quality_profile cannot be empty")

    def effective_analysis_cell_size(self) -> float:
        return self.analysis_cell_size or self.voxel_size * self.analysis_cell_size_multiplier

    def effective_fusion_cell_size(self) -> float:
        return self.fusion_cell_size or self.voxel_size

    def effective_artifact_cell_size(self) -> float:
        return self.artifact_cell_size or self.effective_fusion_cell_size()

    def effective_architecture_plane_distance(self) -> float:
        return self.architecture_plane_distance or self.effective_fusion_cell_size() * 0.45

    def effective_architecture_grid_size(self) -> float:
        return self.architecture_grid_size or self.effective_fusion_cell_size()

    def effective_validation_review_cell_size(self) -> float:
        return self.validation_review_cell_size or self.effective_analysis_cell_size()

    def effective_photogrammetry_voxel_size(self) -> float:
        return self.photogrammetry_voxel_size or self.voxel_size * 2.0

    def confidence_weights(self) -> Dict[str, float]:
        return {
            "observation": self.confidence_observation_weight,
            "density": self.confidence_density_weight,
            "distance": self.confidence_distance_weight,
            "normal": self.confidence_normal_weight,
            "surface": self.confidence_surface_weight,
            "registration": self.confidence_registration_weight,
        }


@dataclass
class PassDiagnostics:
    point_count: int
    bounds_min: List[float]
    bounds_max: List[float]
    dimensions: List[float]
    centroid: List[float]
    bbox_diagonal: float
    density_points_per_volume: float
    median_neighbor_distance: float
    sparse_fraction: float
    outlier_fraction: float
    component_count: int
    largest_component_fraction: float
    has_colors: bool
    has_normals: bool
    normal_consistency: Optional[float] = None
    local_surface_variation: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class RegistrationMetrics:
    reference_pass_id: str
    transform: List[List[float]] = field(default_factory=_identity)
    initial_overlap: float = 0.0
    overlap_ratio: float = 0.0
    fitness: float = 0.0
    inlier_rmse: Optional[float] = None
    correspondence_count: int = 0
    method: str = "unregistered"
    accepted: bool = False
    requires_manual_alignment: bool = False
    message: str = "Not registered"


@dataclass
class ScanPass:
    pass_id: str
    name: str
    source_path: str
    source_sha256: str
    source_size: int
    diagnostics: Optional[PassDiagnostics] = None
    registration: Optional[RegistrationMetrics] = None
    enabled: bool = True


@dataclass
class DeepMeshFusionResult:
    fused_cloud_path: str
    provenance_path: str
    artifact_report_path: str
    rejected_geometry_path: str
    reconstructed_mesh_path: str
    reconstruction_report_path: str
    repaired_mesh_path: str
    gap_report_path: str
    gap_review_path: str
    validated_mesh_path: str
    validation_report_path: str
    quality_map_path: str
    manifest_path: str
    source_pass_count: int
    registered_pass_count: int
    fused_point_count: int
    consensus_point_count: int = 0
    best_observation_point_count: int = 0
    single_observation_point_count: int = 0
    suppressed_observation_count: int = 0
    artifact_suppressed_point_count: int = 0
    reconstructed_vertex_count: int = 0
    reconstructed_face_count: int = 0
    repaired_gap_count: int = 0
    unresolved_gap_count: int = 0
    validation_review_region_count: int = 0
    geometry_ready: bool = False
    overall_geometry_quality: float = 0.0
    mean_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class PointFusionResult:
    """Inspectable point-only output produced before surface reconstruction."""

    fused_cloud_path: str
    provenance_path: str
    evidence_map_path: str
    confidence_cloud_path: str
    artifact_report_path: str
    rejected_geometry_path: str
    source_pass_count: int
    registered_pass_count: int
    fused_point_count: int
    consensus_point_count: int = 0
    best_observation_point_count: int = 0
    single_observation_point_count: int = 0
    suppressed_observation_count: int = 0
    artifact_suppressed_point_count: int = 0
    region_count: int = 0
    conflict_region_count: int = 0
    mean_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class GeometryProvenanceContribution:
    pass_id: str
    source_point_count: int
    representative_point: List[float]
    weight: float
    residual: float
    observation_quality: float


@dataclass
class ConsensusPointProvenance:
    output_index: int
    grid_index: List[int]
    fusion_method: str
    confidence: float
    region_id: Optional[str]
    contributions: List[GeometryProvenanceContribution]


@dataclass
class ArtifactComponentReport:
    component_id: str
    pass_id: str
    cell_count: int
    source_point_count: int
    bounds_min: List[float]
    bounds_max: List[float]
    observation_support: float
    other_pass_coverage: float
    structural_continuity: float
    conflict_proximity: float
    isolation_score: float
    artifact_score: float
    classification: str
    suppressed: bool


@dataclass
class ArtifactSuppressionSummary:
    input_point_count: int
    retained_point_count: int
    suppressed_point_count: int
    candidate_component_count: int
    suppressed_component_count: int
    retained_uncertain_component_count: int
    per_pass_suppressed_points: Dict[str, int]
    classification_counts: Dict[str, int]


@dataclass
class ArtifactSuppressionResult:
    report_path: str
    rejected_geometry_path: str
    summary: ArtifactSuppressionSummary


@dataclass
class ArchitecturalOpening:
    opening_id: str
    plane_id: str
    classification: str
    width: float
    height: float
    area: float
    confidence: float
    corners: List[List[float]]


@dataclass
class ArchitecturalPlane:
    plane_id: str
    classification: str
    equation: List[float]
    normal: List[float]
    centroid: List[float]
    basis_u: List[float]
    basis_v: List[float]
    projected_bounds: List[float]
    bounds_min: List[float]
    bounds_max: List[float]
    inlier_point_count: int
    area: float
    confidence: float
    openings: List[ArchitecturalOpening] = field(default_factory=list)


@dataclass
class ArchitecturalEdge:
    edge_id: str
    plane_ids: List[str]
    classification: str
    start: List[float]
    end: List[float]
    length: float


@dataclass
class ArchitecturalCorner:
    corner_id: str
    plane_ids: List[str]
    position: List[float]
    confidence: float


@dataclass
class ArchitectureReconstructionSummary:
    input_point_count: int
    plane_count: int
    wall_count: int
    floor_count: int
    ceiling_count: int
    doorway_count: int
    window_count: int
    edge_count: int
    corner_count: int
    complex_point_count: int
    vertex_count: int
    face_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    connected_component_count: int
    surface_area: float


@dataclass
class ArchitectureReconstructionResult:
    mesh_path: str
    report_path: str
    summary: ArchitectureReconstructionSummary


@dataclass
class GapRegion:
    gap_id: str
    surface_id: int
    plane_id: Optional[str]
    classification: str
    decision: str
    area: float
    perimeter: float
    boundary_confidence: float
    repair_confidence: float
    observed_point_count: int
    evidence_observation_count: int
    bounds_min: List[float]
    bounds_max: List[float]
    repaired_face_count: int
    review_required: bool
    reason: str


@dataclass
class GapRepairSummary:
    detected_gap_count: int
    repaired_gap_count: int
    observed_geometry_repair_count: int
    planar_continuation_count: int
    surface_interpolation_count: int
    intentional_opening_count: int
    exterior_boundary_count: int
    unresolved_gap_count: int
    added_vertex_count: int
    added_face_count: int
    final_vertex_count: int
    final_face_count: int


@dataclass
class GapRepairResult:
    repaired_mesh_path: str
    report_path: str
    review_path: str
    summary: GapRepairSummary


@dataclass
class GeometryIssue:
    issue_id: str
    category: str
    severity: str
    count: int
    message: str
    review_required: bool
    sample_locations: List[List[float]] = field(default_factory=list)
    surface_ids: List[int] = field(default_factory=list)


@dataclass
class GeometryQualityScores:
    completeness: float
    surface_quality: float
    consistency: float
    confidence: float
    architectural_consistency: float
    overall: float


@dataclass
class GeometryValidationSummary:
    vertex_count: int
    face_count: int
    surface_area: float
    boundary_edge_count: int
    boundary_loop_count: int
    unclassified_hole_count: int
    nonmanifold_edge_count: int
    self_intersection_count: int
    self_intersection_pairs_tested: int
    self_intersection_audit_complete: bool
    degenerate_triangle_count: int
    stretched_triangle_count: int
    bad_normal_count: int
    surface_discontinuity_count: int
    disconnected_component_count: int
    tiny_component_count: int
    unresolved_gap_count: int
    review_region_count: int
    ready_for_appearance_processing: bool
    scores: GeometryQualityScores


@dataclass
class GeometryValidationResult:
    validated_mesh_path: str
    report_path: str
    quality_map_path: str
    summary: GeometryValidationSummary


@dataclass
class PhotogrammetryCamera:
    image_id: int
    camera_id: int
    camera_model: str
    camera_parameters: List[float]
    image_name: str
    image_path: str
    image_sha256: Optional[str]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation_world_to_camera: List[List[float]]
    translation_world_to_camera: List[float]
    center: List[float]
    registered_observation_count: int


@dataclass
class TextureSourceQuality:
    image_id: int
    image_name: str
    available: bool
    width: int
    height: int
    megapixels: float
    resolution: float
    sharpness: float
    exposure: float
    contrast: float
    clipped_fraction: float
    score: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class PhotogrammetryRegistrationMetrics:
    transform: List[List[float]]
    scale: float
    fitness: float
    inlier_rmse: float
    median_correspondence_error: float
    p95_correspondence_error: float
    correspondence_count: int
    mutual_correspondence_ratio: float
    accepted: bool
    requires_manual_alignment: bool
    method: str
    message: str


@dataclass
class CameraMeshValidation:
    image_id: int
    image_name: str
    center: List[float]
    forward: List[float]
    projected_face_count: int
    visible_face_count: int
    visible_area: float
    mesh_coverage: float
    relationship_valid: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class PhotogrammetryPreparationSummary:
    camera_count: int
    valid_camera_count: int
    source_point_count: int
    registered: bool
    correspondence_count: int
    mesh_face_count: int
    covered_face_count: int
    multi_view_face_count: int
    coverage: float
    multi_view_coverage: float
    mean_views_per_covered_face: float
    mean_texture_source_quality: float
    texture_ready: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class PhotogrammetryPreparationResult:
    registration_path: str
    aligned_source_path: str
    camera_report_path: str
    coverage_map_path: str
    report_path: str
    summary: PhotogrammetryPreparationSummary


@dataclass
class TextureCameraUsage:
    image_id: int
    image_name: str
    selected_texel_count: int
    blended_texel_count: int
    mean_selection_score: float


@dataclass
class TextureBakeSummary:
    face_count: int
    atlas_size: int
    atlas_utilization: float
    textured_texel_count: int
    blended_texel_count: int
    uncovered_texel_count: int
    texture_coverage: float
    seam_count: int
    mean_texture_confidence: float
    high_confidence_fraction: float
    low_confidence_fraction: float
    texture_ready: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class TextureBakeResult:
    textured_obj_path: str
    material_path: str
    texture_atlas_path: str
    confidence_map_path: str
    selection_report_path: str
    summary: TextureBakeSummary


@dataclass
class FinalTextureIssue:
    issue_id: str
    category: str
    severity: str
    face_indices: List[int]
    atlas_bounds: List[int]
    score: float
    message: str
    auto_repairable: bool
    repaired: bool = False


@dataclass
class FinalAssetQuality:
    geometry: float
    texture: float
    coverage: float
    consistency: float
    overall: float


@dataclass
class FinalAssetSummary:
    geometry_valid: bool
    texture_valid: bool
    coverage_status: str
    detected_issue_count: int
    repaired_issue_count: int
    review_region_count: int
    remaining_issue_count: int
    polished_asset_ready: bool
    quality: FinalAssetQuality
    warnings: List[str] = field(default_factory=list)


@dataclass
class FinalAssetResult:
    final_obj_path: str
    final_material_path: str
    final_texture_path: str
    final_confidence_path: str
    review_map_path: str
    report_path: str
    summary: FinalAssetSummary


@dataclass
class TourQualityCheck:
    check_id: str
    label: str
    score: Optional[float]
    threshold: Optional[float]
    passed: bool
    blocking: bool
    message: str


@dataclass
class TourReadinessIssue:
    issue_id: str
    category: str
    severity: str
    blocking: bool
    source_stage: str
    message: str
    region_count: int
    artifact_path: Optional[str] = None


@dataclass
class TourReadinessSummary:
    profile: str
    geometry: float
    completeness: float
    surface_consistency: float
    texture_coverage: float
    texture_quality: float
    critical_defect_count: int
    review_region_count: int
    blocking_issue_count: int
    advisory_issue_count: int
    source_integrity_verified: bool
    artifact_integrity_verified: bool
    tour_ready: bool


@dataclass
class TourReadinessResult:
    report_path: str
    html_report_path: str
    asset_manifest_path: str
    summary: TourReadinessSummary


@dataclass
class PassRegionEvidence:
    pass_id: str
    point_count: int
    density: float
    centroid: List[float]
    mean_normal: Optional[List[float]]
    distance_to_consensus: float
    distance_agreement: float
    normal_agreement: Optional[float]
    surface_consistency: float
    registration_quality: float
    confidence: float
    score_components: Dict[str, float] = field(default_factory=dict)


@dataclass
class SpatialRegionEvidence:
    region_id: str
    grid_index: List[int]
    bounds_min: List[float]
    bounds_max: List[float]
    center: List[float]
    observation_count: int
    observation_ratio: float
    total_point_count: int
    mean_density: float
    mean_cross_pass_distance: float
    max_cross_pass_distance: float
    normal_agreement: Optional[float]
    local_surface_consistency: float
    confidence: float
    agreement: str
    conflict: bool
    conflict_reasons: List[str]
    missing_pass_ids: List[str]
    pass_evidence: List[PassRegionEvidence]
    provenance: Dict[str, int]


@dataclass
class SpatialEvidenceSummary:
    total_regions: int
    multi_pass_regions: int
    conflict_regions: int
    missing_observation_regions: int
    high_confidence_regions: int
    overlap_ratio: float
    mean_confidence: float
    per_pass_coverage: Dict[str, float]


@dataclass
class SpatialEvidenceMap:
    schema_version: int
    cell_size: float
    grid_origin: List[float]
    registered_pass_ids: List[str]
    source_provenance: Dict[str, Dict[str, Any]]
    scoring_weights: Dict[str, float]
    summary: SpatialEvidenceSummary
    regions: List[SpatialRegionEvidence]


@dataclass
class CrossPassAnalysisResult:
    evidence_map_path: str
    confidence_cloud_path: str
    region_count: int
    conflict_region_count: int
    missing_observation_region_count: int
    mean_confidence: float


def to_json_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)
