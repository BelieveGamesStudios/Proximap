from __future__ import annotations

from typing import Dict

import numpy as np

from .models import DeepMeshFusionConfig


class DeepMeshFusionConfidenceService:
    """Explainable weighted confidence scoring shared by every spatial region."""

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config
        self.weights = config.confidence_weights()

    def score(self, components: Dict[str, float], conflict: bool = False) -> float:
        normalized = {name: float(np.clip(components.get(name, 0.0), 0.0, 1.0)) for name in self.weights}
        score = sum(self.weights[name] * normalized[name] for name in self.weights)
        if conflict:
            score *= self.config.conflict_penalty
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def agreement_label(confidence: float, conflict: bool) -> str:
        if conflict:
            return "CONFLICT"
        if confidence >= 0.80:
            return "HIGH"
        if confidence >= 0.55:
            return "MEDIUM"
        return "LOW"
