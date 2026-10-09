"""
Context-Aware Image Quality Assessment (IQA) - Pipeline Module

Encapsulates Phase 4 multimodal inference, fusing semantic localization,
local technical features, and global context representations with defect
penalty modeling and uncertainty quantification.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from PIL import Image

from src.semantic_localizer import SemanticLocalizer
from src.local_features import LocalFeatureExtractor
from src.global_features import GlobalFeatureExtractor


@dataclass
class PipelineConfig:
    """Configuration hyperparameters for model fusion and decision logic."""
    w_local: float = 0.65
    w_global: float = 0.35
    threshold: float = 55.0
    ablation_mode: str = "full"  # "full", "local_only", "global_only", "no_penalty"


@dataclass
class PipelineResult:
    """Structured container holding all inference outputs and metrics."""
    unified_score: float
    base_score: float
    classification: str  # "Keep" vs "Review"
    s_local: float
    s_global: float
    framing_penalty_pts: float
    face_penalty_pts: float
    motion_penalty_pts: float
    total_defect_penalty: float
    overall_confidence_pct: float
    uncertainty_level: str
    conf_tier: str  # "high", "med", "low"
    margin_of_error: float
    dist_to_boundary: float
    localization: Dict[str, Any] = field(default_factory=dict)
    local_features: Dict[str, Any] = field(default_factory=dict)
    global_features: Dict[str, Any] = field(default_factory=dict)


class Phase4Pipeline:
    """
    Phase 4 inference orchestrator coordinating deep learning localization,
    local/global feature extractors, and calibrated penalty modeling.
    """

    def __init__(
        self,
        localizer: Optional[SemanticLocalizer] = None,
        local_extractor: Optional[LocalFeatureExtractor] = None,
        global_extractor: Optional[GlobalFeatureExtractor] = None,
    ):
        self.localizer = localizer or SemanticLocalizer()
        self.local_extractor = local_extractor or LocalFeatureExtractor()
        self.global_extractor = global_extractor or GlobalFeatureExtractor()

    def run(self, img: Image.Image, config: Optional[PipelineConfig] = None) -> PipelineResult:
        """
        Executes complete Phase 4 assessment on a PIL image.
        """
        if config is None:
            config = PipelineConfig()

        rgb_img = img.convert("RGB")

        # 1. Semantic Localization
        loc = self.localizer.localize(rgb_img)

        # 2. Local Technical Features
        local_feat = self.local_extractor.extract(rgb_img, loc)

        # 3. Global Context Features
        global_feat = self.global_extractor.extract(rgb_img, loc)

        # 4. Multimodal Fusion & Defect Penalties
        s_local = float(local_feat["local_technical_score"])
        s_global = float(global_feat["global_context_score"])
        framing_defect = bool(local_feat["framing_defect_detected"])
        has_face = bool(local_feat["face_present"])
        motion_metric = float(local_feat["motion_blur_metric"])
        framing_anomaly = loc.get("framing_anomaly", "NONE")

        base_score = (config.w_local * s_local) + (config.w_global * s_global)

        framing_penalty_pts = 0.0
        face_penalty_pts = 0.0
        motion_penalty_pts = 0.0

        if config.ablation_mode == "local_only":
            unified_score = s_local
        elif config.ablation_mode == "global_only":
            unified_score = s_global
        elif config.ablation_mode == "no_penalty":
            unified_score = base_score
        else:
            # Full context-aware penalty engine
            score_curr = base_score

            # Anomaly Penalty 1: Head/Face cropped out
            if framing_defect or framing_anomaly == "HEAD_CROPPED":
                capped_score = min(score_curr * 0.35, 25.0)
                framing_penalty_pts = score_curr - capped_score
                score_curr = capped_score

            # Anomaly Penalty 2: Turned away face / Face not visible in portrait
            elif not has_face:
                discounted_score = score_curr * 0.60
                face_penalty_pts = score_curr - discounted_score
                score_curr = discounted_score

            # Anomaly Penalty 3: Severe motion blur
            if motion_metric > 0.45:
                motion_penalty = (motion_metric - 0.45) * 45.0
                motion_penalty_pts = motion_penalty
                score_curr -= motion_penalty

            unified_score = max(0.0, min(100.0, score_curr))

        total_defect_penalty = round(base_score - unified_score, 2)
        classification = "Keep" if unified_score >= config.threshold else "Review"

        # 5. Calibrated Confidence & Uncertainty Quantification
        dist_to_boundary = abs(unified_score - config.threshold)
        c_margin = min(1.0, dist_to_boundary / 15.0)  # saturation at 15 points distance
        c_subj = float(loc.get("subject_confidence", 0.8))
        c_face = float(loc.get("face_confidence", 0.7)) if has_face else 0.5
        c_detect = 0.6 * c_subj + 0.4 * c_face

        overall_confidence_pct = round((0.70 * c_margin + 0.30 * c_detect) * 100.0, 1)

        if overall_confidence_pct >= 75.0:
            uncertainty_level = "High Confidence (Low Uncertainty)"
            conf_tier = "high"
        elif overall_confidence_pct >= 50.0:
            uncertainty_level = "Moderate Confidence (Borderline Margin)"
            conf_tier = "med"
        else:
            uncertainty_level = "Low Confidence (High Uncertainty - Near Boundary)"
            conf_tier = "low"

        margin_of_error = round(max(0.5, (100.0 - overall_confidence_pct) * 0.08), 1)

        return PipelineResult(
            unified_score=round(unified_score, 2),
            base_score=round(base_score, 2),
            classification=classification,
            s_local=round(s_local, 2),
            s_global=round(s_global, 2),
            framing_penalty_pts=round(framing_penalty_pts, 2),
            face_penalty_pts=round(face_penalty_pts, 2),
            motion_penalty_pts=round(motion_penalty_pts, 2),
            total_defect_penalty=total_defect_penalty,
            overall_confidence_pct=overall_confidence_pct,
            uncertainty_level=uncertainty_level,
            conf_tier=conf_tier,
            margin_of_error=margin_of_error,
            dist_to_boundary=round(dist_to_boundary, 2),
            localization=loc,
            local_features=local_feat,
            global_features=global_feat,
        )
