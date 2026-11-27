"""Collection safety and selection tests without model downloads or a GUI."""
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.desktop.project import Photo, Project


def collection(tmp_path):
    source = tmp_path / 'originals'
    source.mkdir()
    (source / 'nested').mkdir()
    for name in ['one.jpg', 'two.jpg', 'nested/one.jpg']:
        Image.new('RGB', (24, 24), 'white').save(source / name)
    (source / 'camera.nef').write_bytes(b'raw')
    project = Project(str(source), target=2)
    project.scan()
    return project


def test_inventory_resume_and_changed_file_reset(tmp_path):
    project = collection(tmp_path)
    assert len(project.photos) == 4
    assert next(p for p in project.photos if p.relative.endswith('.nef')).status == 'Pending'
    photo = next(p for p in project.photos if p.relative == 'one.jpg')
    photo.status, photo.override = 'Keep', 'Review'
    saved = tmp_path / 'project.iqa.json'
    project.save(saved)
    restored = Project.load(saved)
    restored.scan()
    assert next(p for p in restored.photos if p.relative == 'one.jpg').decision == 'Review'
    (Path(project.source) / 'one.jpg').write_bytes(b'changed')
    restored.scan()
    assert next(p for p in restored.photos if p.relative == 'one.jpg').status == 'Pending'


def test_selection_diversity_override_pins_and_shortfall():
    project = Project('.', target=2, photos=[
        Photo('a.jpg', 0, 0, 'Keep', 95, group='burst'),
        Photo('b.jpg', 0, 0, 'Keep', 94, group='burst'),
        Photo('c.jpg', 0, 0, 'Keep', 80, group='other'),
        Photo('d.jpg', 0, 0, 'Review', 99),
    ])
    assert [p.relative for p in project.selection()] == ['a.jpg', 'c.jpg']
    project.style = 'Strict quality'
    assert [p.relative for p in project.selection()] == ['a.jpg', 'b.jpg']
    project.photos[3].override = 'Keep'
    project.photos[3].must_include = True
    assert project.selection()[0].relative == 'd.jpg'
    project.target = 10
    assert len(project.selection()) == 4
    project.target = 1
    project.photos[0].must_include = True
    with pytest.raises(ValueError, match='Must-include'):
        project.selection()


def test_analysis_pause_errors_and_resumption(tmp_path):
    project = collection(tmp_path)
    (Path(project.source) / 'two.jpg').write_bytes(b'corrupt')
    stop = threading.Event()
    calls = []
    pipeline = SimpleNamespace(run=lambda image, config: SimpleNamespace(
        unified_score=85, classification='Keep', review_reasons=[]))
    def pause(photo):
        calls.append(photo.relative)
        stop.set()
    project.analyze(pipeline, None, stop, pause)
    assert len(calls) == 1
    stop.clear()
    project.analyze(pipeline, None, stop, lambda p: calls.append(p.relative))
    assert len(calls) == 4
    assert next(p for p in project.photos if p.relative == 'two.jpg').status == 'Error'
    assert all(p.status != 'Pending' for p in project.photos)


def test_export_never_overwrites_sources_and_preserves_names(tmp_path):
    project = collection(tmp_path)
    for photo in project.photos:
        if photo.status == 'Pending':
            photo.status = 'Keep'
    project.target = 3
    before = {p.relative: project.path_for(p).read_bytes() for p in project.photos}
    with pytest.raises(ValueError, match='outside'):
        project.export(Path(project.source) / 'export')
    destination = tmp_path / 'export'
    assert project.export(destination) == 3
    assert (destination / 'photos/nested/one.jpg').is_file()
    assert (destination / 'photos/one.jpg').is_file()
    with pytest.raises(FileExistsError):
        project.export(destination)
    assert before == {p.relative: project.path_for(p).read_bytes() for p in project.photos}
    (Path(project.source) / 'one.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source changed'):
        project.export(tmp_path / 'another-export')
    assert not (tmp_path / 'another-export').exists()


def test_project_cannot_escape_selected_folder(tmp_path):
    project = Project(str(tmp_path), photos=[Photo('../outside.jpg', 0, 0)])
    with pytest.raises(ValueError, match='outside'):
        project.path_for(project.photos[0])


def test_creative_photos_are_not_automatic_removal_suggestions():
    project = Project('.', target=1, photos=[
        Photo('standard.jpg', 0, 0, 'Keep', 95),
        Photo('creative.jpg', 0, 0, 'Keep', 80, creative_style='Creative'),
        Photo('uncertain-creative.jpg', 0, 0, 'Review', 35, creative_style='Creative'),
        Photo('review.jpg', 0, 0, 'Review', 20),
    ])
    assert [p.relative for p in project.cleanup_candidates()] == ['review.jpg']
    assert project.summary()['creative_keeps'] == 1
    creative = project.photos[1]
    creative.override = 'Excluded'  # The photographer can still explicitly remove one.
    assert [p.relative for p in project.cleanup_candidates()] == ['creative.jpg', 'review.jpg']


def test_analysis_persists_machine_style_and_preserves_photographer_creative_override(tmp_path):
    project = collection(tmp_path)
    photo = next(p for p in project.photos if p.relative.endswith('.jpg'))
    photo.override, photo.creative_override = 'Keep', 'Creative'
    pipeline = SimpleNamespace(run=lambda image, config: SimpleNamespace(
        unified_score=35, classification='Review', review_reasons=['Possible blur'],
        s_local=40., s_global=30., creative_style='Standard', creative_probability=.2))
    project.analyze(pipeline, None, threading.Event(), lambda _: None)
    assert photo.status == 'Review' and photo.creative_keep
    assert photo.creative_style == 'Standard' and photo.style_label == 'Creative'
    assert photo.technical_score == 40. and photo.context_score == 30.
    path = tmp_path/'creative.iqa.json'
    project.save(path)
    restored = Project.load(path)
    kept = next(p for p in restored.photos if p.relative == photo.relative)
    assert kept.creative_keep and kept.creative_probability == .2
    restored.prepare_assessment('new-engine')
    assert kept.status == 'Pending' and kept.style_label == 'Creative'
    assert kept.creative_style == 'Unknown' and kept.creative_probability is None
