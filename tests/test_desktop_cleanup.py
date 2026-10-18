"""Source cleanup requires exact approval and preserves recoverable outcomes."""
import threading

import pytest
from PIL import Image

from src.desktop.project import Photo, Project


def collection(tmp_path, target=None):
    root = tmp_path / 'source'
    root.mkdir()
    for name in ('keep.jpg', 'review.jpg', 'spare.jpg'):
        Image.new('RGB', (24, 24), 'white').save(root / name)
    project = Project(str(root), target=target)
    project.scan()
    for photo in project.photos:
        photo.status = 'Review' if photo.relative == 'review.jpg' else 'Keep'
        photo.score = 90 if photo.relative == 'keep.jpg' else 60
    return project


def fake_trash(tmp_path):
    destination = tmp_path / 'fake-recycle-bin'
    destination.mkdir()
    return lambda path: path.rename(destination / path.name)


def test_optional_count_keeps_every_approved_photo_and_roundtrips_old_projects(tmp_path):
    project = collection(tmp_path)
    assert {p.relative for p in project.selection()} == {'keep.jpg', 'spare.jpg'}
    assert [p.relative for p in project.cleanup_candidates()] == ['review.jpg']
    project.target = 1
    assert {p.relative for p in project.cleanup_candidates()} == {'review.jpg', 'spare.jpg'}
    assert all(p.override == '' for p in project.photos)
    project.target = None
    path = tmp_path / 'project.iqa.json'
    project.save(path)
    assert Project.load(path).target is None
    # Previously saved project records had neither cleanup history nor recycled.
    import json
    data = json.loads(path.read_text())
    data.pop('cleanup_history')
    for photo in data['photos']:
        photo.pop('recycled')
    data['target'] = 2
    path.write_text(json.dumps(data))
    assert len(Project.load(path).selection()) == 2


def test_suggestions_or_target_limit_do_not_authorize_removal(tmp_path):
    project = collection(tmp_path, target=1)
    calls = []
    with pytest.raises(ValueError, match='explicitly marked'):
        project.recycle(['spare.jpg'], tmp_path / 'project.json', trash=calls.append)
    assert calls == []
    assert all(project.path_for(p).exists() for p in project.photos)


def test_exact_marked_files_are_recycled_and_survivors_remain_in_place(tmp_path):
    project = collection(tmp_path)
    photo = next(p for p in project.photos if p.relative == 'review.jpg')
    photo.override = 'Excluded'
    saved = tmp_path / 'project.json'
    report = project.recycle(['review.jpg'], saved, trash=fake_trash(tmp_path))
    assert report['recycled'] == ['review.jpg']
    assert not report['errors']
    assert (tmp_path / 'source/keep.jpg').is_file()
    assert (tmp_path / 'source/spare.jpg').is_file()
    assert photo.recycled and photo.decision == 'Recycled'
    restored = Project.load(saved)
    restored.scan()
    assert restored.cleanup_history[0]['files'][0]['outcome'] == 'recycled'
    assert next(p for p in restored.photos if p.relative == 'review.jpg').recycled
    (tmp_path / 'fake-recycle-bin/review.jpg').rename(tmp_path / 'source/review.jpg')
    restored.scan()
    assert not next(p for p in restored.photos if p.relative == 'review.jpg').recycled


def test_changed_or_pinned_file_blocks_whole_cleanup_before_first_move(tmp_path):
    project = collection(tmp_path)
    for photo in project.photos:
        photo.override = 'Excluded'
    calls = []
    last = project.photos[-1]
    last.must_include = True
    with pytest.raises(ValueError, match='explicitly marked'):
        project.recycle([p.relative for p in project.photos], tmp_path / 'project.json', trash=calls.append)
    assert not calls
    last.must_include = False
    project.path_for(last).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source changed'):
        project.recycle([p.relative for p in project.photos], tmp_path / 'project.json', trash=calls.append)
    assert not calls


def test_partial_failure_records_success_and_stops_remaining_moves(tmp_path):
    project = collection(tmp_path)
    for photo in project.photos:
        photo.override = 'Excluded'
    move = fake_trash(tmp_path)
    calls = []
    def fail_second(path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError('Recycle Bin unavailable')
        move(path)
    saved = tmp_path / 'project.json'
    report = project.recycle([p.relative for p in project.photos], saved, trash=fail_second)
    assert len(report['recycled']) == 1 and len(report['errors']) == 1
    assert len(calls) == 2
    assert project.path_for(project.photos[2]).exists()
    history = Project.load(saved).cleanup_history[-1]['files']
    assert [entry['outcome'] for entry in history] == ['recycled', 'error', 'not_attempted']


def test_cancel_before_cleanup_does_not_change_source_files(tmp_path):
    project = collection(tmp_path)
    for photo in project.photos:
        photo.override = 'Excluded'
    stop = threading.Event()
    stop.set()
    calls = []
    report = project.recycle([p.relative for p in project.photos], tmp_path / 'project.json', trash=calls.append, stop=stop)
    assert report['cancelled'] and not calls
    assert all(project.path_for(p).exists() for p in project.photos)


def test_failed_journal_write_prevents_source_changes(tmp_path, monkeypatch):
    project = collection(tmp_path)
    project.photos[0].override = 'Excluded'
    calls = []
    def failed_save(path):
        raise OSError('Disk full')
    monkeypatch.setattr(project, 'save', failed_save)
    with pytest.raises(OSError, match='Disk full'):
        project.recycle([project.photos[0].relative], tmp_path / 'project.json', trash=calls.append)
    assert not calls


def test_cleanup_rejects_outside_paths(tmp_path):
    project = Project(str(tmp_path), photos=[Photo('../outside.jpg', 1, 1, 'Review', override='Excluded')])
    with pytest.raises(ValueError, match='relative'):
        project.recycle(['../outside.jpg'], tmp_path.parent / 'project.json', trash=lambda path: None)


def test_cleanup_refuses_paths_redirected_to_other_photos(tmp_path, monkeypatch):
    project = collection(tmp_path)
    review = next(p for p in project.photos if p.relative == 'review.jpg')
    review.override = 'Excluded'
    keep = project.path_for(next(p for p in project.photos if p.relative == 'keep.jpg'))
    monkeypatch.setattr(project, 'path_for', lambda photo: keep)
    calls = []
    with pytest.raises(ValueError, match='links or junctions'):
        project.recycle(['review.jpg'], tmp_path / 'project.json', trash=calls.append)
    assert not calls and keep.exists()


@pytest.mark.parametrize('approved', [[], ['review.jpg', 'review.jpg']])
def test_cleanup_requires_distinct_nonempty_approval(tmp_path, approved):
    project = collection(tmp_path)
    calls = []
    with pytest.raises(ValueError, match='distinct'):
        project.recycle(approved, tmp_path / 'project.json', trash=calls.append)
    assert not calls
