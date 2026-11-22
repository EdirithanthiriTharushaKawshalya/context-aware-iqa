"""Synthetic behavioral tests; these do not establish photographic accuracy."""
from types import SimpleNamespace
import csv
import json
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from src.desktop.project import Photo, Project
from src.photographic_intent import inspect_photographic_intent
from src.pipeline import Phase4Pipeline, PipelineConfig


def silhouette(blur=0, background=180):
    image = Image.new("RGB", (240, 240), (background, background, background))
    draw = ImageDraw.Draw(image)
    draw.ellipse((95, 35, 145, 85), fill=(4, 4, 4))
    draw.rectangle((85, 80, 155, 210), fill=(4, 4, 4))
    return image.filter(ImageFilter.GaussianBlur(blur)) if blur else image


def localization():
    return dict(subject_detected=True, person_detector_status="ok", face_detector_status="ok",
                subjects=[dict(subject_id=0, subject_detected=True, subject_bbox=[80, 30, 160, 215],
                               face_detected=False, face_bbox=None)],
                assessment_context={"shot_type": "portrait"})


def inspect(image, local=None, global_features=None, exposure=None):
    return inspect_photographic_intent(image, localization(), local or {}, global_features or {}, exposure)


def test_silhouette_requires_bright_surround_and_crisp_shape():
    assert inspect(silhouette())["supported_silhouette"]
    assert not inspect(silhouette(blur=8))["supported_silhouette"]
    assert not inspect(silhouette(background=25))["supported_silhouette"]
    assert not inspect(Image.new("RGB", (240, 240), "black"))["possible_creative"]


def test_black_clothing_with_an_illuminated_face_is_not_a_silhouette():
    image = silhouette()
    ImageDraw.Draw(image).ellipse((95,35,145,85), fill=(180,150,120))
    assert not inspect(image)['silhouette_evidence']


def test_silhouette_rules_do_not_require_illuminated_face():
    pipeline = Phase4Pipeline(localizer=SimpleNamespace(localize=lambda image: localization()))
    config = PipelineConfig(shot_type="portrait", decision_mode="rules")
    sharp = pipeline.run(silhouette(), config)
    soft = pipeline.run(silhouette(blur=8), config)
    assert sharp.classification == "Keep", sharp.review_reasons
    assert sharp.possible_creative and "Possible silhouette" in sharp.photographic_profiles
    assert sharp.face_penalty_pts == sharp.motion_penalty_pts == 0
    assert soft.classification == "Review"
    assert sharp.creative_style == "Unknown"  # No invented photographer style label.


def test_background_softness_does_not_establish_subject_motion():
    detail = dict(local_technical_score=75, motion_blur_metric=.2)
    background = dict(background_assessable=True, background_motion_metric=.9)
    result = inspect(silhouette(background=60), detail, background)
    assert result["separated_background_blur"]
    assert not inspect(silhouette(), dict(detail, motion_blur_metric=.8), background)["separated_background_blur"]


def test_darkness_and_slow_exposure_are_not_automatic_artistic_approval():
    result = inspect(Image.new("RGB", (240,240), (20,20,20)), exposure=.25)
    assert result["low_light"] and result["slow_shutter_metadata"]
    assert result["possible_creative"] and not result["supported_silhouette"]
    pipeline = Phase4Pipeline(localizer=SimpleNamespace(localize=lambda image: localization()))
    assert pipeline.run(Image.new("RGB", (240,240), "black"), PipelineConfig(decision_mode="rules")).classification == "Review"


def test_low_key_sharp_subject_recovers_shadow_deduction_only():
    rng = np.random.default_rng(12)
    pixels = np.zeros((240,240,3), dtype=np.uint8)
    pixels[30:215,80:160] = rng.integers(30,160,(185,80,3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    pipeline = Phase4Pipeline(localizer=SimpleNamespace(localize=lambda image: localization()))
    result = pipeline.run(image, PipelineConfig(decision_mode="rules"))
    assert "Low-light / low-key" in result.photographic_profiles
    assert result.classification == "Keep", result.review_reasons


def test_automated_candidates_survive_cleanup_but_explicit_marks_work(tmp_path):
    candidate = Photo("dream.jpg", 1, 1, status="Review", possible_creative=True,
                      photographic_profiles=["Possible slow-shutter / dreamy motion"])
    regular = Photo("ordinary.jpg", 1, 1, status="Review")
    project = Project(str(tmp_path), photos=[candidate, regular])
    assert project.cleanup_candidates() == [regular]
    project.save(tmp_path/"project.json")
    loaded = Project.load(tmp_path/"project.json")
    assert loaded.photos[0].possible_creative
    candidate.override = "Excluded"
    assert candidate in project.cleanup_candidates()
    project.prepare_assessment("new")
    assert not candidate.possible_creative and not candidate.photographic_profiles


def test_exposure_metadata_reaches_automatic_candidate_detection():
    image = Image.new('RGB', (240,240), (20,20,20))
    image.getexif()[33434] = .25
    pipeline = Phase4Pipeline(localizer=SimpleNamespace(localize=lambda image: localization()))
    result = pipeline.run(image, PipelineConfig(decision_mode='rules'))
    assert 'Possible slow-shutter / dreamy motion' in result.photographic_profiles
    assert result.creative_probability is None


def test_machine_inventory_resumes_after_interruption_without_touching_human_labels(tmp_path, monkeypatch):
    import scripts.assess_photographic_candidates as batch
    manifest = tmp_path/'labels.csv'
    paths = [tmp_path/f'{i}.png' for i in range(3)]
    for i, path in enumerate(paths):
        Image.new('RGB', (30,30), (30+i,30,30)).save(path)
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_id', 'file_path', 'label_source', 'expert_score_usability'])
        writer.writeheader()
        writer.writerows(dict(image_id=str(i), file_path=str(path), label_source='', expert_score_usability='')
                         for i, path in enumerate(paths))
    original = manifest.read_bytes()
    calls = []
    def run(image, config):
        calls.append(image.getpixel((0,0))[0])
        if len(calls) == 2:
            raise KeyboardInterrupt('terminal closed')
        return SimpleNamespace(classification='Review', unified_score=30., photographic_profiles=['Low-light / low-key'],
                               possible_creative=False, review_reasons=['uncertain'], assessment_notes=[])
    monkeypatch.setattr(batch, 'Phase4Pipeline', lambda: SimpleNamespace(run=run))
    output = tmp_path/'machine.jsonl'
    monkeypatch.setattr(sys, 'argv', ['batch', '--manifest', str(manifest), '--output', str(output)])
    with pytest.raises(KeyboardInterrupt):
        batch.main()
    assert len(output.read_text().splitlines()) == 1
    batch.main()
    assert calls == [30,31,31,32]  # Completed first photo was reused.
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 3 and all(not row['is_human_judgment'] for row in records)
    assert {r['label_source'] for r in records} == {'automated_rules'}
    assert manifest.read_bytes() == original
    summary = json.loads(output.with_suffix('.summary.json').read_text())
    assert summary['human_labels_added'] == 0 and summary['errors'] == 0
    Image.new('RGB', (30,30), (99,30,30)).save(paths[2])
    monkeypatch.setattr(sys, 'argv', ['batch', '--manifest', str(manifest), '--output', str(output), '--workers', '2'])
    batch.main()
    assert calls == [30,31,31,32,99]  # Changed source is reassessed, cached peers survive.
    assert manifest.read_bytes() == original
    monkeypatch.setattr(sys, 'argv', ['batch', '--manifest', str(manifest), '--output', str(manifest)])
    with pytest.raises(SystemExit):
        batch.main()
    assert manifest.read_bytes() == original


def test_machine_snapshot_retries_a_transient_windows_file_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.assess_photographic_candidates import replace_snapshot
    output, temporary = tmp_path/'result.jsonl', tmp_path/'result.tmp'
    output.write_text('old complete snapshot')
    temporary.write_text('new complete snapshot')
    original = Path.replace
    attempts = []
    def locked_once(path, target):
        attempts.append(1)
        if len(attempts) == 1:
            assert output.read_text() == 'old complete snapshot'
            raise PermissionError('reader briefly holds file')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', locked_once)
    replace_snapshot(temporary, output)
    assert len(attempts) == 2 and output.read_text() == 'new complete snapshot'
