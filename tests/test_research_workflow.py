"""Integration checks for blind labels, CNN deployment and frozen comparisons."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.prepare_human_labels import FIELDS
from src.annotations import AnnotationStore
from src.cnn_features import FrozenResNetEncoder
from src.context_classifier import ContextClassifier
from src.pipeline import Phase4Pipeline, PipelineConfig
from tests.test_context_classifier import entry, pipeline_for, write_manifest
from src.training_data import implementation_fingerprint


@pytest.fixture
def blank_annotations(tmp_path):
    path = tmp_path/"human.csv"
    rows = []
    for i in range(2):
        image_path = tmp_path/f"photo{i}.png"
        Image.new("RGB", (50,40), (100+i,90,80)).save(image_path)
        row = dict.fromkeys(FIELDS, "")
        row.update(image_id=f"image{i}", file_path=str(image_path), event_id="event", burst_id="burst")
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_blind_labels_validate_reasons_and_assign_whole_events(blank_annotations):
    store = AnnotationStore(blank_annotations)
    with pytest.raises(ValueError, match="reason"):
        store.save_label("image0", "Review", "photographer")
    store = AnnotationStore(blank_annotations)
    store.save_label("image0", "Keep", "photographer", creative_notes="sharp subject, moving background")
    saved = AnnotationStore(blank_annotations)
    assert saved.rows[0]["label_source"] == "human"
    assert not saved.rows[1]["expert_score_usability"]
    saved.assign_event("event", "validation")
    assert {r["split_category"] for r in AnnotationStore(blank_annotations).rows} == {"validation"}


def test_concurrent_annotations_do_not_overwrite_other_reviewer(blank_annotations):
    first, stale = AnnotationStore(blank_annotations), AnnotationStore(blank_annotations)
    first.save_label("image0", "Keep", "first")
    with pytest.raises(ValueError, match="another session"):
        stale.save_label("image0", "Review", "second", "focus")
    assert AnnotationStore(blank_annotations).rows[0]["labeler_id"] == "first"
    lock = blank_annotations.with_suffix(".csv.lock")
    lock.write_text("other process")
    with pytest.raises(ValueError, match="in progress"):
        AnnotationStore(blank_annotations).save_label("image1", "Keep", "second")
    assert lock.exists(), "A failed lock attempt must not remove another session's lock"


def test_repeated_annotation_saves_keep_other_session_protection(blank_annotations):
    store = AnnotationStore(blank_annotations)
    store.save_label("image0", "Keep", "first")
    store.save_label("image1", "Review", "first", "subject focus")
    store.assign_event("event", "validation")
    reloaded = AnnotationStore(blank_annotations)
    assert store.digest == reloaded.digest
    assert [row["expert_score_usability"] for row in reloaded.rows] == ["Keep", "Review"]
    assert {row["split_category"] for row in reloaded.rows} == {"validation"}
    reloaded.save_label("image0", "Review", "second", "focus")
    with pytest.raises(ValueError, match="another session"):
        store.save_label("image1", "Keep", "first")


def test_blind_annotation_screen_saves_without_predictions(blank_annotations, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CONTEXT_IQA_LABELS", str(blank_annotations))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"label_photos.py")).run()
    assert not app.exception
    assert not any("prediction" in e.value.lower() for e in app.markdown)
    app.text_input[0].set_value("independent-photographer")
    app.radio[0].set_value("Keep")
    next(control for control in app.selectbox if control.label == "Photographic style").set_value("Creative")
    next(button for button in app.button if button.label == "Save judgment").click().run()
    assert not app.exception
    store = AnnotationStore(blank_annotations)
    assert store.rows[0]["expert_score_usability"] == "Keep"
    assert store.rows[0]["labeler_id"] == "independent-photographer"
    assert store.rows[0]["creative_style"] == "Creative"


def test_cnn_checkpoint_runs_in_normal_pipeline_without_downloading(tmp_path, monkeypatch):
    # Random initial encoder weights exercise serialization only. This fixture
    # is not a pretrained or production research model and stays in test temp.
    torch.manual_seed(9)
    encoder = FrozenResNetEncoder(image_size=224, pretrained=False)
    image = Image.fromarray(np.random.default_rng(3).integers(0,255,(240,320,3), dtype=np.uint8))
    sample = entry(500.)
    sample["localization"]["subjects"] = [{"subject_bbox": [100,20,230,220],
                                           "face_bbox": [130,40,180,100], "face_detected": True}]
    model = ContextClassifier("cnn", encoder=encoder)
    model.creative_classifier = ContextClassifier("cnn", encoder=encoder)
    prepared = model.prepare_entry(image, sample)
    assert len(prepared["cnn_features"]["image"]) == 512
    assert len(prepared["cnn_features"]["subject"]) == 512
    assert not np.allclose(prepared["cnn_features"]["image"], prepared["cnn_features"]["subject"])
    model.metadata = {"training_protocol": "human_event_holdout_v1",
                      "implementation_sha256": implementation_fingerprint()}
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.zero_()
        model.model[-1].bias.fill_(3.)
        for parameter in model.creative_classifier.model.parameters():
            parameter.zero_()
        model.creative_classifier.model[-1].bias.fill_(3.)
    model.model.eval()
    path = tmp_path/"cnn.pt"
    model.save(path)
    monkeypatch.setattr("src.cnn_features.pretrained_weights_path", lambda *a, **kw: pytest.fail("Inference tried to download weights"))
    restored = ContextClassifier.load(path)
    assert restored.predict(prepared)["classification"] == "Keep"
    assert restored.creative_classifier.encoder is restored.encoder
    assert restored.predict(prepared)["creative_style"] == "Creative"
    result = pipeline_for(sample).run(image, PipelineConfig(decision_mode="learned", model_path=str(path)))
    assert result.decision_source == "learned_cnn"
    assert result.classification == "Keep"
    assert result.creative_style == "Creative"
    assert len(result.model_identifier) == 64


def test_context_ablations_exclude_other_visual_regions():
    sample = entry(500.)
    sample["cnn_features"] = {"image": [1.]*512, "subject": [2.]*512}
    global_model = ContextClassifier("cnn", "global")
    subject_model = ContextClassifier("cnn", "subject")
    global_original, subject_original = global_model._vector(sample), subject_model._vector(sample)
    sample["local_features"]["subject_sharpness"] = 1
    sample["cnn_features"]["subject"] = [90.]*512
    assert np.array_equal(global_original, global_model._vector(sample))
    sample["local_features"]["subject_sharpness"] = 500.
    sample["cnn_features"]["subject"] = [2.]*512
    sample["global_features"]["global_contrast"] = 99
    sample["cnn_features"]["image"] = [90.]*512
    assert np.array_equal(subject_original, subject_model._vector(sample))


def test_research_comparison_freezes_selection_and_never_trains_on_test(tmp_path, monkeypatch):
    import scripts.run_research_experiment as experiment
    manifest = write_manifest(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for i, row in enumerate(rows):
        Image.new("RGB", (20,20), (i*30+10,0,0)).save(row["file_path"])
    observed = []
    def features(image):
        index = round((image.getpixel((0,0))[0]-10)/30)
        observed.append(index)
        return entry(800. if index % 2 == 0 else 2., .1 if index % 2 == 0 else .8)
    def pipeline():
        return Phase4Pipeline(localizer=SimpleNamespace(localize=lambda image: features(image)["localization"]),
            local_extractor=SimpleNamespace(extract=lambda image, *_: features(image)["local_features"]),
            global_extractor=SimpleNamespace(extract=lambda image, *_: features(image)["global_features"]))
    monkeypatch.setattr(experiment, "Phase4Pipeline", pipeline)
    directory = tmp_path/"run"
    monkeypatch.setattr(sys, "argv", ["run", "--manifest", str(manifest), "--run-dir", str(directory),
                                    "--logistic-only", "--epochs", "80"])
    experiment.main()
    assert max(observed) <= 3, "Training accessed test image features"
    selection_before = (directory/"selection.json").read_bytes()
    selection = json.loads(selection_before)
    assert len(selection["candidates"]) == 3
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Frozen evaluation retrained a candidate")
    monkeypatch.setattr(ContextClassifier, "fit", unexpected_fit)
    monkeypatch.setattr(sys, "argv", ["run", "--manifest", str(manifest), "--evaluate-run", str(directory)])
    experiment.main()
    assert (directory/"selection.json").read_bytes() == selection_before
    report = json.loads((directory/"test_results.json").read_text())
    assert set(report["comparisons"]) == set(selection["candidates"]) | {"always_keep", "current_rules"}
    assert report["metadata"]["selected_before_test"] == selection["selected"]
    assert (directory/"test_predictions.csv").exists()
    assert report["comparisons"]["logistic_combined"]["runtime"]["images"] == 2
    with pytest.raises(ValueError, match="already exist"):
        experiment.main()


def test_cnn_head_learns_quality_with_frozen_visual_features():
    entries, labels = [], []
    for background in (.1, .9):
        for sharpness, label in [(2., 0), (5., 0), (400., 1), (800., 1)]:
            sample = entry(sharpness, .8 if label == 0 else .1, background)
            sample["cnn_features"] = {"image": [background]*512, "subject": [float(label)]*512}
            entries.append(sample)
            labels.append(label)
    model = ContextClassifier("cnn").fit(entries, labels, epochs=80,
        validation_entries=[entries[1], entries[-1]], validation_labels=[0, 1])
    model.tune_threshold([entries[1], entries[-1]], [0, 1])
    assert model.predict(entries[-1])["classification"] == "Keep"
    assert model.predict(entries[0])["classification"] == "Review"
    assert model.training_summary["early_stopping"]


def test_model_change_refreshes_machine_assessment_and_preserves_human_choices():
    from src.desktop.project import Project, Photo
    photo = Photo("creative.jpg", 1, 1, status="Review", score=30., override="Keep", must_include=True)
    project = Project("photos", photos=[photo], assessment_signature="old-engine")
    project.prepare_assessment("new-engine")
    assert photo.status == "Pending"
    assert photo.override == "Keep" and photo.must_include
    photo.status, photo.score = "Keep", 85.
    project.prepare_assessment("new-engine")
    assert photo.status == "Keep" and photo.score == 85.


def test_desktop_receives_new_engine_and_inventory_atomically():
    from dataclasses import asdict
    from src.desktop.app import DesktopApp
    from src.desktop.project import Project, Photo
    app = DesktopApp.__new__(DesktopApp)
    app.project = Project("photos", photos=[Photo("one.jpg", 1, 1, status="Keep")], assessment_signature="old")
    snapshots = []
    app.checkpoint = lambda: snapshots.append((app.project.assessment_signature, app.project.photos[0].status))
    app.refresh, app.current = lambda: None, lambda: None
    app.consume([("engine_inventory", {"signature": "new", "photos": [asdict(Photo("one.jpg", 1, 1))]})])
    assert snapshots == [("new", "Pending")]


def test_musiq_threshold_uses_validation_and_accepts_creative_high_scores(monkeypatch):
    from src.musiq_baseline import MusiqBaseline
    metric = lambda batch: batch.mean().reshape(1)*100
    monkeypatch.setitem(sys.modules, "pyiqa", SimpleNamespace(create_metric=lambda *a, **kw: metric))
    scorer = MusiqBaseline()
    assert scorer.score(Image.new("RGB", (20,20), "white")) == 100.
    scorer.tune([10., 20., 80., 90.], [0, 0, 1, 1])
    assert 20 < scorer.threshold <= 80


def test_independent_reviewer_agreement_and_disagreements(blank_annotations, tmp_path):
    from scripts.compare_annotations import compare
    second_path = tmp_path/"second.csv"
    second_path.write_bytes(blank_annotations.read_bytes())
    AnnotationStore(blank_annotations).save_label("image0", "Keep", "first")
    AnnotationStore(blank_annotations).save_label("image1", "Review", "first", "subject focus")
    AnnotationStore(second_path).save_label("image0", "Keep", "second")
    AnnotationStore(second_path).save_label("image1", "Keep", "second")
    result = compare(blank_annotations, second_path)
    assert result["shared_photos"] == 2
    assert result["agreement"] == .5
    assert result["disagreements"][0]["image_id"] == "image1"


def test_musiq_snapshot_restores_frozen_weights_and_threshold(tmp_path, monkeypatch):
    from src.musiq_baseline import MusiqBaseline
    calls = []
    class MeanMetric(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(100.))
        def forward(self, batch):
            return batch.mean().reshape(1)*self.scale
    def create(*args, **kwargs):
        calls.append(kwargs)
        return MeanMetric()
    monkeypatch.setitem(sys.modules, "pyiqa", SimpleNamespace(create_metric=create))
    scorer = MusiqBaseline()
    scorer.tune([10., 90.], [0, 1])
    path = tmp_path/"musiq.pt"
    scorer.save(path)
    restored = MusiqBaseline.load(path)
    assert restored.threshold == scorer.threshold
    assert restored.score(Image.new("RGB", (20,20), "white")) == 100.
    assert calls[-1]["pretrained"] is False


def test_detail_context_ignores_incidental_people_for_background_scoring():
    from src.global_features import GlobalFeatureExtractor
    image = Image.fromarray(np.random.default_rng(8).integers(30,220,(200,200,3), dtype=np.uint8))
    extractor = GlobalFeatureExtractor()
    context = {"assessment_context": {"shot_type": "detail"}}
    plain = extractor.extract(image, context)
    people = extractor.extract(image, dict(context, subject_detected=True,
        subjects=[{"subject_bbox": [20,20,130,180]}]))
    assert plain == people
