"""Real Tk burst filtering, comparison choices and cached-worker upgrades."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import time

from PIL import Image
import pytest

from tests.test_desktop_gallery import app
from src.desktop.analysis import assess_project
from src.desktop.duplicates import describe
from src.desktop.project import Project
from src.model_registry import engine_signature


def add_duplicates(app):
    a,b = app.project.photos[:2]
    source = app.project.path_for(a)
    copy = app.project.path_for(b)
    copy.write_bytes(source.read_bytes())
    b.size,b.modified = copy.stat().st_size,copy.stat().st_mtime_ns
    for photo in (a,b):
        with Image.open(app.project.path_for(photo)) as original:
            photo.duplicate_features = describe(original.convert('RGB'))
    app.project.update_duplicate_groups()
    app.refresh()
    return a,b


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def test_burst_filters_and_default_single_recommendation_are_visible(app):
    a,b = add_duplicates(app)
    assert app.project.selection()[0] is a
    assert b not in app.project.selection()
    app.view.set('Duplicates / bursts')
    app.refresh()
    assert len(app.gallery.rows) == 2
    app.view.set('Best in bursts')
    app.refresh()
    assert [p for _,p in app.gallery.rows] == [a]
    app.view.set('Burst alternatives')
    app.refresh()
    assert [p for _,p in app.gallery.rows] == [b]
    app.select_gallery_photo('1')
    assert 'Exact visual/file copies' in app.details.get()
    assert str(app.burst_button.cget('state')) == 'normal'


def test_comparison_can_replace_recommended_photo_and_keep_multiple_without_removal(app):
    a,b = add_duplicates(app)
    app.view.set('Duplicates / bursts')
    app.refresh()
    app.select_gallery_photo('0')
    window = app.compare_burst()
    gallery = next(child for child in window.winfo_children() if child.__class__.__name__ == 'PhotoGallery')
    gallery.on_select('1')
    use = next(child for child in descendants(window) if child.winfo_class() == 'TButton' and
               str(child.cget('text')) == 'Use selected shot for this burst')
    use.invoke()
    assert b in app.project.selection() and a not in app.project.selection()
    assert a.decision == 'Keep' and b.selection_choice == 'include'
    app.target.set('3')
    app.update()
    gallery.on_toggle('0')
    assert a in app.project.selection() and b in app.project.selection()
    assert len(list(Path(app.project.source).glob('*.jpg'))) == 5
    assert not app.project.cleanup_history
    window.destroy()


def test_turning_off_single_burst_selection_restores_eligible_alternates(app):
    a,b = add_duplicates(app)
    assert b not in app.project.selection()
    app.one_per_burst.set(False)
    app.update()
    assert a in app.project.selection() and b in app.project.selection()
    stored = Project.load(app.project_file)
    assert stored.one_per_burst is False


def test_comparison_opens_the_actual_alternate_at_full_resolution(app, monkeypatch):
    a,b = add_duplicates(app)
    app.select_gallery_photo('0')
    window = app.compare_burst()
    gallery = next(child for child in window.winfo_children() if child.__class__.__name__ == 'PhotoGallery')
    opened = []
    monkeypatch.setattr(app,'zoom',lambda photo=None,parent=None: opened.append((photo,parent)))
    gallery.on_open('1')
    assert opened == [(b,window)]
    window.destroy()


def test_comparison_fits_minimum_window_and_save_failure_restores_choices(app, monkeypatch):
    a,b = add_duplicates(app)
    app.select_gallery_photo('0')
    window = app.compare_burst()
    window.geometry('800x560')
    app.root.update()
    gallery = next(child for child in window.winfo_children() if child.__class__.__name__ == 'PhotoGallery')
    assert gallery.canvas.winfo_height() >= gallery.CARD_HEIGHT+gallery.GAP
    use = next(child for child in descendants(window) if child.winfo_class() == 'TButton' and
               str(child.cget('text')) == 'Use selected shot for this burst')
    assert use.winfo_rootx()-window.winfo_rootx()+use.winfo_width() <= window.winfo_width()
    gallery.on_select('1')
    messages = []
    monkeypatch.setattr('src.desktop.burst_review.messagebox.showinfo',lambda *a,**kw:messages.append(a))
    def failing_checkpoint():
        raise OSError('Disk full')
    monkeypatch.setattr(app,'checkpoint',failing_checkpoint)
    use.invoke()
    assert messages and b.selection_choice == a.selection_choice == ''
    assert a in app.project.selection() and b not in app.project.selection()
    window.destroy()


def test_cached_assessments_gain_groups_in_worker_without_loading_quality_models(app, monkeypatch):
    app.project.assessment_signature = engine_signature()
    before = {p.relative:p.score for p in app.project.photos}
    records = []
    def forbidden():
        pytest.fail('A duplicate-only upgrade tried to reload the quality engine')
    monkeypatch.setattr('src.pipeline.Phase4Pipeline',forbidden)
    channel = SimpleNamespace(send=records.append,close=lambda:None)
    assess_project(channel,asdict(app.project))
    inventory = next(detail for event,detail in records if event=='inventory')
    assert all(p['duplicate_features'] for p in inventory)
    assert {p['relative']:p['score'] for p in inventory} == before
    assert records[-1][0] == 'done'


def test_duplicate_upgrade_progress_does_not_look_like_completed_quality_analysis(app):
    app.analysis_stage = 'Checking duplicates: photo0.jpg'
    app.stage_started = time.monotonic()
    assert app.project.duplicates_pending()
    app.update_progress()
    assert 'Checking duplicates: photo0.jpg' in app.status.get()
