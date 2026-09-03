"""Headless multi-pass LiDAR fusion services for Proximap."""

from .models import (
    ArtifactComponentReport,
    ArtifactSuppressionResult,
    ArchitectureReconstructionResult,
    ArchitecturalPlane,
    GapRepairResult,
    GapRegion,
    GeometryIssue,
    GeometryQualityScores,
    GeometryValidationResult,
    PhotogrammetryCamera,
    PhotogrammetryPreparationResult,
    PhotogrammetryRegistrationMetrics,
    TextureSourceQuality,
    TextureBakeResult,
    FinalAssetResult,
    FinalAssetSummary,
    FinalTextureIssue,
    TourReadinessIssue,
    TourReadinessResult,
    CrossPassAnalysisResult,
    ConsensusPointProvenance,
    DeepMeshFusionConfig,
    DeepMeshFusionResult,
    PassDiagnostics,
    PointFusionResult,
    RegistrationMetrics,
    ScanPass,
    SpatialEvidenceMap,
)
from .analysis import DeepMeshFusionAnalysisService
from .confidence import DeepMeshFusionConfidenceService
from .fusion import DeepMeshFusionService
from .artifacts import TransientArtifactSuppressionService
from .reconstruction import DeepMeshFusionReconstructionService
from .gaps import EvidenceBasedGapRepairService, GapRepairService
from .validation import GeometryQualityValidationService, GeometryValidationService
from .photogrammetry import ColmapTextModelLoader, PhotogrammetryPreparationService, PhotogrammetryRegistrationService
from .texture_baking import DeepMeshFusionTextureBakingService, IntelligentTextureBakingService
from .final_repair import FinalAssetRepairService, FinalSurfaceTextureRepairService
from .quality_gate import TourReadinessQualityGate, TourReadinessService
from .workspace import DeepMeshFusionWorkspace

__all__ = [
    "ArtifactComponentReport",
    "ArtifactSuppressionResult",
    "ArchitecturalPlane",
    "ArchitectureReconstructionResult",
    "ConsensusPointProvenance",
    "CrossPassAnalysisResult",
    "DeepMeshFusionAnalysisService",
    "DeepMeshFusionConfig",
    "DeepMeshFusionConfidenceService",
    "DeepMeshFusionService",
    "DeepMeshFusionReconstructionService",
    "DeepMeshFusionResult",
    "DeepMeshFusionWorkspace",
    "PassDiagnostics",
    "PointFusionResult",
    "RegistrationMetrics",
    "ScanPass",
    "SpatialEvidenceMap",
    "TransientArtifactSuppressionService",
    "EvidenceBasedGapRepairService",
    "GapRepairResult",
    "GapRepairService",
    "GapRegion",
    "GeometryIssue",
    "GeometryQualityScores",
    "GeometryQualityValidationService",
    "GeometryValidationResult",
    "GeometryValidationService",
    "ColmapTextModelLoader",
    "PhotogrammetryCamera",
    "PhotogrammetryPreparationResult",
    "PhotogrammetryPreparationService",
    "PhotogrammetryRegistrationMetrics",
    "PhotogrammetryRegistrationService",
    "TextureSourceQuality",
    "TextureBakeResult",
    "DeepMeshFusionTextureBakingService",
    "IntelligentTextureBakingService",
    "FinalAssetRepairService",
    "FinalAssetResult",
    "FinalAssetSummary",
    "FinalSurfaceTextureRepairService",
    "FinalTextureIssue",
    "TourReadinessIssue",
    "TourReadinessQualityGate",
    "TourReadinessResult",
    "TourReadinessService",
]
