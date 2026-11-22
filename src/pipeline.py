"""
Context-Aware Image Quality Assessment (IQA) - Pipeline Module

Encapsulates Phase 4 multimodal inference, fusing semantic localization,
local technical features, and global context representations with defect
penalty modeling and uncertainty quantification.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from PIL import Image, ImageOps
from src.scene_context import resolve_context
from src.quality_policy import apply_policy

from src.semantic_localizer import SemanticLocalizer
from src.local_features import LocalFeatureExtractor
from src.global_features import GlobalFeatureExtractor


@dataclass
class PipelineConfig:
    """Configuration hyperparameters for model fusion and decision logic."""
    w_local: float = 0.55
    w_global: float = 0.45
    threshold: float = 52.0
    face_discount: float = 0.60
    motion_thresh: float = 0.45
    motion_scale: float = 45.0
    framing_cap: float = 25.0
    ablation_mode: str = "full"  # "full", "local_only", "global_only", "no_penalty"

    shot_type: str = "auto"
    motion_intent: str = "auto"
    subject_quality_threshold: float = 35.0
    decision_mode: str = "auto"  # auto uses a compatible human-trained model when present
    model_path: Optional[str] = None

    @classmethod
    def from_tuned(cls, config_path: Optional[str] = None) -> "PipelineConfig":
        """Loads optimal hyperparameters from tuned_config.json if available."""
        import json
        from pathlib import Path
        p = Path(config_path) if config_path else Path(__file__).resolve().parent.parent / "configs" / "tuned_config.json"
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("policy_version") != 2:
                    return cls()  # Portrait-era calibration does not transfer to this policy.
                best = data.get("best_config", {})
                return cls(
                    w_local=float(best.get("w_local", 0.55)),
                    w_global=float(best.get("w_global", 0.45)),
                    threshold=float(best.get("threshold", 52.0)),
                    face_discount=float(best.get("face_discount", 0.60)),
                    motion_thresh=float(best.get("motion_thresh", 0.45)),
                    motion_scale=float(best.get("motion_scale", 45.0)),
                    framing_cap=float(best.get("framing_cap", 25.0)),
                )
            except Exception:
                pass
        return cls()


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
    assessment_context: Dict[str, Any] = field(default_factory=dict)
    review_reasons: list = field(default_factory=list)
    localization: Dict[str, Any] = field(default_factory=dict)
    local_features: Dict[str, Any] = field(default_factory=dict)
    global_features: Dict[str, Any] = field(default_factory=dict)
    decision_source: str = "rules"
    probability_keep: Optional[float] = None
    decision_threshold: Optional[float] = None
    rule_score: Optional[float] = None
    assessment_notes: list = field(default_factory=list)
    model_evidence: list = field(default_factory=list)
    model_identifier: str = ""
    creative_style: str = "Unknown"
    creative_probability: Optional[float] = None
    photographic_profiles: list = field(default_factory=list)
    possible_creative: bool = False


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
        self._classifier_cache = None

    def _classifier(self, config):
        from src.context_classifier import ContextClassifier
        from src.model_registry import active_model_path
        from src.training_data import implementation_fingerprint
        path = active_model_path(config.model_path)
        if not path.is_file():
            if config.decision_mode == "learned":
                raise FileNotFoundError(f"Train a human-labelled context classifier first: {path}")
            return None, None
        try:
            fingerprint = implementation_fingerprint()
            signature = (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size, fingerprint)
            if self._classifier_cache and self._classifier_cache[0] == signature:
                return self._classifier_cache[1], None
            model = ContextClassifier.load(path)
            if model.metadata.get("training_protocol") != "human_event_holdout_v1":
                raise ValueError("Classifier lacks independent-label/event-validation metadata")
            if model.metadata.get("implementation_sha256") != fingerprint:
                raise ValueError("Classifier was trained with a different assessment implementation; retrain it before use")
            import hashlib
            with path.open("rb") as handle:
                model.checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            self._classifier_cache = signature, model
            return model, None
        except Exception as exc:
            if config.decision_mode == "learned":
                raise ValueError(f"Cannot use context classifier: {exc}") from exc
            return None, f"Using rules because the learned model could not be loaded: {exc}"

    def run(self, img: Image.Image, config: Optional[PipelineConfig] = None) -> PipelineResult:
        """
        Executes complete Phase 4 assessment on a PIL image.
        """
        if config is None:
            config = PipelineConfig.from_tuned()
        if config.decision_mode not in ("auto", "rules", "learned"):
            raise ValueError("Unsupported decision_mode")

        rgb_img = ImageOps.exif_transpose(img).convert("RGB")

        # 1. Semantic Localization
        loc = self.localizer.localize(rgb_img)

        context = resolve_context(loc, config.shot_type, config.motion_intent)
        loc["assessment_context"] = context
        review_reasons = list(context["review_reasons"])

        # 2. Local Technical Features
        local_feat = self.local_extractor.extract(rgb_img, loc)

        # 3. Global Context Features
        global_feat = self.global_extractor.extract(rgb_img, loc)

        from src.photographic_intent import inspect_photographic_intent
        try:
            exposure_time = img.getexif().get(33434)
            if exposure_time is None:
                exposure_time = img.getexif().get_ifd(34665).get(33434)
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            exposure_time = None
        intent = inspect_photographic_intent(rgb_img, loc, local_feat, global_feat, exposure_time)
        context["photographic_profiles"] = intent["profiles"]
        context["silhouette_supported"] = intent["supported_silhouette"]

        # 4. Multimodal Fusion & Defect Penalties
        s_local = float(local_feat["local_technical_score"])
        s_global = float(global_feat["global_context_score"])
        # Preserve raw measurements for learned features and diagnostics. These
        # adjustments change only the rule baseline, with explicit evidence.
        if intent["supported_silhouette"]:
            contour_score = min(90., min(r["contrast"] for r in intent["silhouette_evidence"])/2)
            s_local = max(s_local, contour_score)
        if intent["supported_silhouette"] or (intent["low_light"] and intent["sharp_subject"]):
            s_global = min(100., s_global + max(0., (global_feat.get("shadow_underexposure_ratio", 0)-.08)*40))
        if intent["separated_background_blur"]:
            s_global = max(s_global, s_local-10.)
        has_face = bool(local_feat["face_present"])
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
            unified_score, face_penalty_pts, motion_penalty_pts, review_reasons = apply_policy(
                base_score, loc, local_feat, context, config)

        total_defect_penalty = round(base_score - unified_score, 2)
        classification = "Keep" if unified_score >= config.threshold else "Review"
        if review_reasons and config.ablation_mode == "full":
            classification = "Review"

        rule_score = unified_score
        decision_source, probability_keep = "rules", None
        decision_threshold = config.threshold
        notes, model_evidence, model_identifier = list(intent["notes"]), [], ""
        if intent["supported_silhouette"]:
            notes.append("Rules assess silhouette contour separation instead of requiring illuminated facial texture. This is an automated hypothesis.")
        creative_style, creative_probability = "Unknown", None
        if config.decision_mode != "rules" and config.ablation_mode == "full":
            classifier, load_note = self._classifier(config)
            if load_note:
                notes.append(load_note)
            if classifier is not None:
                entry = classifier.prepare_entry(rgb_img, {"localization": loc, "local_features": local_feat,
                                                          "global_features": global_feat})
                prediction = classifier.predict(entry)
                decision_source = f"learned_{classifier.model_type}"
                model_identifier = classifier.checkpoint_sha256
                probability_keep = prediction["probability_keep"]
                creative_style = prediction.get("creative_style", "Unknown")
                creative_probability = prediction.get("creative_probability")
                unified_score = probability_keep * 100.
                decision_threshold = classifier.threshold * 100.
                classification = prediction["classification"]
                model_evidence = prediction["review_evidence"]
                # Quality heuristics must not veto the learned creative decision.
                # Missing evidence is still an explicit request for inspection.
                from src.context_classifier import missing_evidence_reasons
                review_reasons = missing_evidence_reasons(loc, local_feat)
                if review_reasons:
                    classification = "Review"
                    decision_source = "learned_with_evidence_check"
                elif classification == "Review":
                    review_reasons.append("The learned classifier flags this photo for inspection.")
                    if model_evidence:
                        names = ", ".join(item["feature"].replace("_", " ") for item in model_evidence)
                        review_reasons.append(f"Model evidence toward Review: {names}. These are learned associations, not confirmed defects.")
                notes.append("The learned score estimates Keep suitability; it is not a calibrated artistic-quality rating.")
        if classification == "Review":
            # Evidence-based observations, not claims of certain artistic failure.
            if local_feat.get("local_technical_score", 100.) < config.subject_quality_threshold:
                review_reasons.append("Low subject/detail sharpness; inspect focus at full resolution.")
            if motion_penalty_pts > 0 and decision_source == "rules":
                review_reasons.append("Directional blur is present in the assessed subject/detail region.")
            if local_feat.get("subject_highlight_clipping", 0.) > .15:
                review_reasons.append("Substantial highlight clipping in the assessed subject/detail region.")
            if local_feat.get("subject_shadow_clipping", 0.) > .15 and not intent["supported_silhouette"]:
                review_reasons.append("Substantial shadow clipping in the assessed subject/detail region.")
            if not review_reasons:
                review_reasons.append("Combined technical evidence is below the Keep threshold; inspect focus, exposure and contrast.")
        if (global_feat.get("background_assessable") and
                global_feat.get("background_motion_metric", 0.) > config.motion_thresh and
                local_feat.get("motion_blur_metric", 1.) <= config.motion_thresh):
            notes.append("Background blur is stronger than subject blur; background movement alone is not a subject-focus defect.")
        review_reasons = list(dict.fromkeys(review_reasons))

        # 5. Heuristic confidence indicator
        dist_to_boundary = abs(unified_score - decision_threshold)
        c_margin = min(1.0, dist_to_boundary / 15.0)  # saturation at 15 points distance
        c_subj = float(loc.get("subject_confidence", 0.8))
        c_face = float(loc.get("face_confidence", 0.7)) if has_face else 0.5
        c_detect = 0.6 * c_subj + 0.4 * c_face
        if not loc.get("subject_detected") and loc.get("person_detector_status") == "ok":
            c_detect = 0.8
        metrics = local_feat.get("subject_metrics", [])
        if context["faces_required"] and metrics:
            c_detect *= local_feat.get("assessable_face_count", 0)/len(metrics)

        overall_confidence_pct = round((0.70 * c_margin + 0.30 * c_detect) * 100.0, 1)

        # This is a heuristic confidence indicator, not a calibrated probability.
        if review_reasons:
            overall_confidence_pct = min(overall_confidence_pct, 49.0)
        if overall_confidence_pct >= 75.0:
            uncertainty_level = "High Confidence (Low Uncertainty)"
            conf_tier = "high"
        elif overall_confidence_pct >= 50.0:
            uncertainty_level = "Moderate Confidence (Borderline Margin)"
            conf_tier = "med"
        else:
            uncertainty_level = "Low Confidence (Insufficient Evidence or Near Boundary)"
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
            assessment_context=context,
            review_reasons=review_reasons,
            localization=loc,
            local_features=local_feat,
            global_features=global_feat,
            decision_source=decision_source,
            probability_keep=probability_keep,
            decision_threshold=decision_threshold,
            rule_score=round(rule_score, 2),
            assessment_notes=notes,
            model_evidence=model_evidence,
            model_identifier=model_identifier,
            creative_style=creative_style,
            creative_probability=creative_probability,
            photographic_profiles=intent["profiles"],
            possible_creative=intent["possible_creative"],
        )
