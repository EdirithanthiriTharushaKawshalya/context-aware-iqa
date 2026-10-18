"""Deterministic scene regressions; no model downloads required."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from src.pipeline import Phase4Pipeline, PipelineConfig
from src.semantic_localizer import SemanticLocalizer
from src.local_features import LocalFeatureExtractor
from src.scene_context import resolve_context


def localizer_for(count, faces=True):
    localizer = SemanticLocalizer.__new__(SemanticLocalizer)
    localizer.device = torch.device("cpu")
    localizer.conf_thresh = .5
    boxes = [[i*60, 10, i*60+55, 190] for i in range(count)]
    localizer._subject_detector = lambda _: [{"boxes": torch.tensor(boxes).reshape(-1, 4),
        "scores": torch.full((count,), .95), "labels": torch.ones(count, dtype=torch.int64)}]
    rows = [[i*60+10, 20, i*60+45, 60, .95] for i in range(count)] if faces else []
    localizer._get_face_detector = lambda: SimpleNamespace(detect_faces=lambda _: rows)
    return localizer


@pytest.mark.parametrize("count,expected", [(0,"scene"),(1,"portrait"),(2,"couple"),(3,"group"),(20,"group")])
def test_all_people_and_faces_retained(count, expected):
    loc = localizer_for(count).localize(Image.new("RGB", (max(60*count, 200), 200)))
    assert len(loc["subjects"]) == count
    assert sum(s["face_detected"] for s in loc["subjects"]) == count
    assert len({f["subject_id"] for f in loc["faces"]}) == count
    assert resolve_context(loc)["shot_type"] == expected


def test_missing_face_is_unknown_and_not_head_crop():
    loc = localizer_for(1, False).localize(Image.new("RGB", (200, 200)))
    assert not loc["face_detected"]
    assert loc["face_orientation"] == "UNKNOWN"
    assert loc["framing_anomaly"] == "NONE"


def test_other_person_face_is_not_assigned_to_primary():
    localizer = localizer_for(2)
    localizer._get_face_detector = lambda: SimpleNamespace(
        detect_faces=lambda _: [[70, 20, 105, 60, .99]])
    loc = localizer.localize(Image.new("RGB", (200,200)))
    assert not loc["subjects"][0]["face_detected"]
    assert loc["subjects"][1]["face_detected"]
    assert not loc["face_detected"]


def test_detail_ignores_incidental_people_for_local_score():
    image = Image.new("RGB", (200,200), "gray")
    loc = localizer_for(1).localize(image)
    loc["assessment_context"] = resolve_context(loc, "detail", "freeze")
    result = LocalFeatureExtractor().extract(image, loc)
    assert result["subject_metrics"] == []
    assert not result["face_present"]


class FixedFeatures:
    def __init__(self, **values):
        self.values = values

    def extract(self, *_):
        return dict(self.values)


def pipeline_for(count=0, faces=True, motion=.2):
    return Phase4Pipeline(localizer=localizer_for(count, faces),
        local_extractor=FixedFeatures(local_technical_score=80., face_present=faces and count>0,
            framing_defect_detected=False, motion_blur_metric=motion),
        global_extractor=FixedFeatures(global_context_score=80.))


def test_scene_and_candid_do_not_require_faces():
    img = Image.new("RGB", (200,200))
    for count, shot in [(0,"scene"),(1,"candid"),(0,"detail")]:
        result = pipeline_for(count, False).run(img, PipelineConfig(shot_type=shot))
        assert result.classification == "Keep"
        assert result.face_penalty_pts == 0


def test_posed_missing_face_requires_review():
    result = pipeline_for(2, False).run(Image.new("RGB", (200,200)), PipelineConfig(shot_type="couple"))
    assert result.classification == "Review"
    assert result.review_reasons


def test_intentional_motion_avoids_motion_penalty():
    pipeline = pipeline_for(motion=.8)
    img = Image.new("RGB", (200,200))
    normal = pipeline.run(img, PipelineConfig(motion_intent="freeze"))
    deliberate = pipeline.run(img, PipelineConfig(motion_intent="intentional"))
    assert normal.motion_penalty_pts > 0
    assert deliberate.motion_penalty_pts == 0
    assert deliberate.unified_score == 80


def test_detector_unavailable_does_not_invent_people():
    localizer = localizer_for(0)
    localizer._subject_detector = None
    result = Phase4Pipeline(localizer=localizer).run(Image.new("RGB", (200,200)))
    assert result.assessment_context["subject_count"] == 0
    assert result.classification == "Review"
    assert result.overall_confidence_pct < 50


def test_blurred_partner_reduces_couple_quality():
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(40, 210, (200, 200, 3), dtype=np.uint8))
    loc = localizer_for(2).localize(image)
    loc["assessment_context"] = resolve_context(loc, "couple", "freeze")
    extractor = LocalFeatureExtractor()
    sharp = extractor.extract(image, loc)
    mixed = image.copy()
    mixed.paste(image.crop((60,0,120,200)).filter(ImageFilter.GaussianBlur(8)), (60,0))
    blurred = extractor.extract(mixed, loc)
    assert blurred["local_technical_score"] < sharp["local_technical_score"] - 10
    assert len(blurred["subject_metrics"]) == 2


def test_tiny_faces_require_review_for_posed_group():
    image = Image.new("RGB", (300,200))
    loc = localizer_for(3).localize(image)
    for subject in loc["subjects"]:
        subject["face_bbox"] = [10,20,20,30]
    pipeline = Phase4Pipeline(localizer=SimpleNamespace(localize=lambda _: loc))
    result = pipeline.run(image, PipelineConfig(shot_type="group", motion_intent="intentional"))
    assert result.local_features["unassessable_subject_count"] == 3
    assert result.review_reasons


def test_invalid_intent_rejected():
    with pytest.raises(ValueError):
        resolve_context({}, "wedding")


def test_one_weak_member_in_twenty_requires_review():
    from src.quality_policy import apply_policy
    metrics = [{"subject_id": i, "face_detected": True, "face_assessable": True,
                "local_technical_score": 10. if i == 19 else 85.} for i in range(20)]
    context = resolve_context({"subjects": [{}]*20}, "group", "freeze")
    _, _, _, reasons = apply_policy(85., {}, {"subject_metrics": metrics}, context, PipelineConfig())
    assert any("19" in reason for reason in reasons)


def test_optimizer_and_pipeline_share_policy():
    from src.optimizer import HyperparameterOptimizer
    pipeline = Phase4Pipeline(localizer=localizer_for(2, False))
    result = pipeline.run(Image.new("RGB", (200,200)), PipelineConfig(shot_type="couple"))
    entry = {"localization": result.localization, "local_features": result.local_features,
             "global_features": result.global_features}
    score = HyperparameterOptimizer()._compute_score(entry, .55, .45)
    assert round(score, 2) == result.unified_score


def test_failed_person_detector_returns_unknown_evidence():
    localizer = localizer_for(0, False)
    def fail(_):
        raise RuntimeError("simulated inference failure")
    localizer._subject_detector = fail
    loc = localizer.localize(Image.new("RGB", (200,200)))
    assert loc["person_detector_status"] == "failed"
    assert not loc["subject_detected"]
