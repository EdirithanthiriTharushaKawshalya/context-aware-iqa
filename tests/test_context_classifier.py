"""Research-path regressions with synthetic labels; no pretrained downloads."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.context_classifier import ContextClassifier, classification_metrics, feature_vector
from src.global_features import GlobalFeatureExtractor
from src.local_features import LocalFeatureExtractor
from src.pipeline import Phase4Pipeline, PipelineConfig
from src.training_data import load_human_partitions
from src.training_data import implementation_fingerprint


def entry(sharpness, motion=.1, background_motion=.8):
    return {"localization": {"subject_detected": True, "person_detector_status": "ok",
                "assessment_context": {"shot_type": "candid", "motion_intent": "auto",
                                       "subject_count": 1, "faces_required": False, "review_reasons": []}},
            "local_features": {"subject_sharpness": sharpness, "face_sharpness": sharpness,
                "face_tenengrad": sharpness, "motion_blur_metric": motion,
                "local_technical_score": 80 if sharpness > 100 else 15,
                "face_present": True, "assessable_face_count": 1},
            "global_features": {"background_motion_metric": background_motion,
                "background_assessable": True, "global_context_score": 30.}}


def pipeline_for(sample):
    return Phase4Pipeline(localizer=SimpleNamespace(localize=lambda _: sample["localization"]),
        local_extractor=SimpleNamespace(extract=lambda *_: sample["local_features"]),
        global_extractor=SimpleNamespace(extract=lambda *_: sample["global_features"]))


def save_constant_model(path, keep=True):
    model = ContextClassifier()
    with torch.no_grad():
        model.model.weight.zero_()
        model.model.bias.fill_(4. if keep else -4.)
    model.metadata = {"training_protocol": "human_event_holdout_v1",
                      "implementation_sha256": implementation_fingerprint()}
    model.save(path)


def test_learns_subject_defects_without_rejecting_background_motion(tmp_path):
    # Background movement occurs in both classes. Only subject quality predicts
    # these synthetic labels; this does not establish real-photograph accuracy.
    samples, labels = [], []
    for background in (.1, .8):
        for sharpness, label in [(2., 0), (5., 0), (400., 1), (800., 1)]:
            samples.append(entry(sharpness, .8 if label == 0 else .1, background))
            labels.append(label)
    model = ContextClassifier().fit(samples, labels, epochs=120)
    validation = [entry(3., .75, .4), entry(600., .12, .4)]
    model.tune_threshold(validation, [0, 1])
    assert model.predict(entry(500., .1, .9))["classification"] == "Keep"
    assert model.predict(entry(4., .85, .1))["classification"] == "Review"
    path = tmp_path/"model.pt"
    model.save(path)
    loaded = ContextClassifier.load(path)
    assert loaded.predict(validation[0]) == model.predict(validation[0])


def test_normal_pipeline_uses_learned_decision_instead_of_motion_heuristic(tmp_path):
    path = tmp_path/"model.pt"
    save_constant_model(path)
    sample = entry(500., motion=.8)
    pipeline = pipeline_for(sample)
    image = Image.new("RGB", (100, 100))
    rules = pipeline.run(image, PipelineConfig(decision_mode="rules"))
    learned = pipeline.run(image, PipelineConfig(decision_mode="learned", model_path=str(path)))
    assert rules.classification == "Review"
    assert learned.classification == "Keep"
    assert learned.decision_source == "learned_logistic"
    assert not learned.review_reasons
    assert learned.probability_keep > .9
    assert learned.rule_score == rules.unified_score


def test_learned_keep_does_not_hide_missing_detector_evidence(tmp_path):
    path = tmp_path/"model.pt"
    save_constant_model(path)
    sample = entry(500.)
    sample["localization"]["person_detector_status"] = "failed"
    pipeline = pipeline_for(sample)
    result = pipeline.run(Image.new("RGB", (100,100)), PipelineConfig(model_path=str(path)))
    assert result.classification == "Review"
    assert result.decision_source == "learned_with_evidence_check"
    assert any("unavailable" in reason for reason in result.review_reasons)


def test_learned_keep_requires_review_when_faces_have_no_assessed_person(tmp_path):
    path = tmp_path/"model.pt"
    save_constant_model(path)
    sample = entry(500.)
    sample["localization"]["unmatched_face_count"] = 2
    result = pipeline_for(sample).run(Image.new("RGB", (100,100)), PipelineConfig(model_path=str(path)))
    assert result.probability_keep > .9
    assert result.classification == "Review"
    assert result.decision_source == "learned_with_evidence_check"
    assert any("coverage is incomplete" in reason for reason in result.review_reasons)


@pytest.mark.parametrize("fingerprint", [None, "outdated-implementation"])
def test_unverified_implementation_falls_back_or_fails_in_strict_mode(tmp_path, fingerprint):
    path = tmp_path/"model.pt"
    save_constant_model(path)
    model = ContextClassifier.load(path)
    model.metadata["implementation_sha256"] = fingerprint
    model.save(path)
    pipeline = pipeline_for(entry(500.))
    result = pipeline.run(Image.new("RGB", (100,100)), PipelineConfig(model_path=str(path)))
    assert result.decision_source == "rules"
    assert any("retrain" in note for note in result.assessment_notes)
    with pytest.raises(ValueError, match="retrain"):
        pipeline.run(Image.new("RGB", (100,100)), PipelineConfig(decision_mode="learned", model_path=str(path)))


def test_cached_model_is_rechecked_when_implementation_changes(tmp_path, monkeypatch):
    path = tmp_path/"model.pt"
    save_constant_model(path)
    pipeline = pipeline_for(entry(500.))
    config = PipelineConfig(model_path=str(path))
    assert pipeline.run(Image.new("RGB", (100,100)), config).decision_source == "learned_logistic"
    monkeypatch.setattr("src.training_data.implementation_fingerprint", lambda: "changed-implementation")
    assert pipeline.run(Image.new("RGB", (100,100)), config).decision_source == "rules"


def test_model_compatibility_works_with_bundled_implementation_files(tmp_path, monkeypatch):
    import src.training_data as training_data
    from pathlib import Path
    path = tmp_path/"model.pt"
    save_constant_model(path)
    # PyInstaller keeps these source files beside the frozen modules for hashing.
    bundle = tmp_path/"_internal"/"src"
    bundle.mkdir(parents=True)
    source = Path(training_data.__file__).parent
    for name in training_data.IMPLEMENTATION_FILES:
        (bundle/name).write_bytes((source/name).read_bytes())
    monkeypatch.setattr(training_data, "__file__", str(bundle/"training_data.py"))
    pipeline = pipeline_for(entry(500.))
    config = PipelineConfig(model_path=str(path))
    assert pipeline.run(Image.new("RGB", (100,100)), config).decision_source == "learned_logistic"
    (bundle/"local_features.py").unlink()
    result = pipeline.run(Image.new("RGB", (100,100)), config)
    assert result.decision_source == "rules"
    assert any("could not be loaded" in note for note in result.assessment_notes)


def test_incompatible_model_falls_back_visibly_and_strict_mode_raises(tmp_path):
    path = tmp_path/"legacy.pt"
    torch.save({"model_type": "logistic"}, path)
    pipeline = pipeline_for(entry(500.))
    result = pipeline.run(Image.new("RGB", (100,100)), PipelineConfig(model_path=str(path)))
    assert result.decision_source == "rules"
    assert any("could not be loaded" in note for note in result.assessment_notes)
    with pytest.raises(ValueError, match="Cannot use"):
        pipeline.run(Image.new("RGB", (100,100)), PipelineConfig(decision_mode="learned", model_path=str(path)))


def test_review_has_measured_reason_and_model_evidence(tmp_path):
    path = tmp_path/"model.pt"
    save_constant_model(path, keep=False)
    result = pipeline_for(entry(2., .8)).run(Image.new("RGB", (100,100)), PipelineConfig(model_path=str(path)))
    assert result.classification == "Review"
    assert any("sharpness" in reason for reason in result.review_reasons)


def test_face_motion_evidence_excludes_background_inside_body_box():
    rng = np.random.default_rng(5)
    image = Image.new("RGB", (240,240), (110,110,110))
    face = Image.fromarray(rng.integers(30,220,(60,60,3), dtype=np.uint8))
    image.paste(face, (90,30))
    loc = {"subjects": [{"subject_id": 0, "subject_bbox": [40,10,200,230],
            "face_bbox": [90,30,150,90], "face_detected": True}],
            "assessment_context": {"shot_type": "portrait"}}
    result = LocalFeatureExtractor().extract(image, loc)
    subject = result["subject_metrics"][0]
    assert subject["face_assessable"]
    assert result["motion_blur_metric"] == subject["face_motion_metric"]
    assert subject["body_motion_metric"] > result["motion_blur_metric"]


def test_background_measurements_do_not_include_subject_texture():
    rng = np.random.default_rng(7)
    image = Image.new("RGB", (240,240), (110,110,110))
    image.paste(Image.fromarray(rng.integers(30,220,(120,100,3), dtype=np.uint8)), (70,60))
    result = GlobalFeatureExtractor().extract(image, {"subject_detected": True,
        "subjects": [{"subject_bbox": [70,60,170,180]}]})
    assert result["background_assessable"]
    assert result["background_gradient_energy"] == 0


def write_manifest(tmp_path, change=None):
    rows = []
    for split in ("train", "validation", "test"):
        for label in ("Keep", "Review"):
            image_id = split+label
            path = tmp_path/(image_id+".jpg")
            Image.new("RGB", (10,10)).save(path)
            rows.append(dict(image_id=image_id, file_path=str(path), event_id=split,
                expert_score_usability=label, label_source="human", labeler_id="photographer",
                review_reason="subject out of focus" if label == "Review" else "",
                split_category=split))
    if change:
        change(rows)
    path = tmp_path/"annotations.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_human_partitions_and_provenance_enforced(tmp_path):
    assert len(load_human_partitions(write_manifest(tmp_path), tmp_path)["train"]) == 2
    bad = write_manifest(tmp_path, lambda rows: rows[0].update(label_source="pipeline"))
    with pytest.raises(ValueError, match="independent human"):
        load_human_partitions(bad, tmp_path)


def test_whole_events_cannot_leak_across_splits(tmp_path):
    path = write_manifest(tmp_path, lambda rows: rows[2].update(event_id="train"))
    with pytest.raises(ValueError, match="Event leakage"):
        load_human_partitions(path, tmp_path)


def test_always_keep_catches_zero_review_cases():
    report = classification_metrics([1,1,1,0], [1,1,1,1])
    assert report["accuracy"] == .75
    assert report["review_recall"] == 0
    assert report["balanced_accuracy"] == .5


def test_invalid_numeric_features_rejected():
    sample = entry(float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        feature_vector(sample)


def test_training_and_frozen_test_workflow(tmp_path, monkeypatch):
    import json
    import sys
    import scripts.train_context_classifier as trainer
    manifest = write_manifest(tmp_path)
    # Distinct contents allow the real duplicate audit to run. Synthetic feature
    # extraction keeps this end-to-end training test independent of downloads.
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    for i, row in enumerate(rows):
        Image.new("RGB", (10,10), (i*30+10, 0, 0)).save(row["file_path"])
        row["creative_style"] = "Creative" if i % 2 == 0 else "Standard"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    class SyntheticPipeline:
        def run(self, image, config):
            index = round((image.getpixel((0,0))[0]-10)/30)
            sample = entry(800. if index % 2 == 0 else 2., .1 if index % 2 == 0 else .8)
            return SimpleNamespace(**sample, classification="Keep")
    monkeypatch.setattr(trainer, "Phase4Pipeline", SyntheticPipeline)
    model_path, report_path = tmp_path/"candidate.pt", tmp_path/"validation.json"
    monkeypatch.setattr(sys, "argv", ["train", "--manifest", str(manifest), "--output-model", str(model_path),
                                    "--report-json", str(report_path), "--epochs", "80"])
    trainer.main()
    report = json.loads(report_path.read_text())
    assert "test" not in report
    assert report["validation"]["application_with_evidence_checks"]["review_recall"] == 1
    assert report["metadata"]["creative_training"]["trained"]
    assert ContextClassifier.load(model_path).creative_classifier is not None
    assert report["validation"]["creative_photos"]["creative_keep_recall"] == 1
    original_checkpoint = model_path.read_bytes()
    def unexpected_training(*args, **kwargs):
        raise AssertionError("Frozen test evaluation retrained the model")
    monkeypatch.setattr(ContextClassifier, "fit", unexpected_training)
    monkeypatch.setattr(sys, "argv", ["train", "--manifest", str(manifest), "--evaluate-model", str(model_path),
                                    "--report-json", str(tmp_path/"test.json")])
    trainer.main()
    test_report = json.loads((tmp_path/"test.json").read_text())
    assert test_report["test"]["application_with_evidence_checks"]["macro_f1"] == 1
    assert test_report["test"]["creative_photos"]["style_accuracy"] == 1
    assert model_path.read_bytes() == original_checkpoint


def test_duplicate_contents_rejected_before_model_loading(tmp_path, monkeypatch):
    import sys
    import scripts.train_context_classifier as trainer
    manifest = write_manifest(tmp_path)  # Six identical images with distinct paths.
    monkeypatch.setattr(sys, "argv", ["train", "--manifest", str(manifest)])
    monkeypatch.setattr(trainer, "Phase4Pipeline", lambda: pytest.fail("Should validate before loading models"))
    with pytest.raises(ValueError, match="Duplicate image contents"):
        trainer.main()
