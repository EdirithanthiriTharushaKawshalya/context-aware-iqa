"""Folder-first workflow, automatic selection, finishing, and recovery."""
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.desktop.app import DesktopApp
from src.desktop.project import Project
from src.desktop.sessions import SessionStore


@pytest.fixture
def app(tmp_path, monkeypatch, desktop_tk):
    root = tk.Toplevel(desktop_tk)
    root.withdraw()
    destroy = root.destroy
    application = DesktopApp(root, session_store=SessionStore(tmp_path / 'recovery'))
    application.errors = []
    monkeypatch.setattr('src.desktop.app.messagebox.showerror', lambda title, message: application.errors.append(message))
    yield application
    if application.session:
        application.session.close()
    for callback in desktop_tk.tk.splitlist(desktop_tk.tk.call('after', 'info')):
        root.after_cancel(callback)
    try:
        destroy()
    except tk.TclError:
        pass  # The Done action may already have closed this application window.


def source(tmp_path, name='photos'):
    folder = tmp_path / name
    folder.mkdir()
    for name in ('a.jpg', 'b.jpg', 'c.jpg'):
        Image.new('RGB', (30, 30), 'white').save(folder / name)
    return folder


def assessed(app, folder):
    assert app.start_folder(folder)
    for index, photo in enumerate(app.project.photos):
        photo.status, photo.score = 'Keep', 90 - index * 5
        photo.group = 'burst' if index < 2 else 'other'
    app.checkpoint()
    app.refresh()


def pump(app, condition):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        app.root.update()
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(app.status.get())


def test_choose_folder_creates_recovery_without_a_save_dialog(app, tmp_path, monkeypatch):
    folder = source(tmp_path)
    monkeypatch.setattr('src.desktop.app.filedialog.askdirectory', lambda: str(folder))
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: pytest.fail('Save dialog appeared before Finish'))
    app.choose_folder()
    assert app.project.source == str(folder)
    assert len(app.project.photos) == 3
    assert app.project_file.is_file()
    assert not app.project_file.is_relative_to(folder)
    assert app.saved_project_file is None and app.dirty
    assert Project.load(app.project_file).photos[0].status == 'Pending'


def test_count_and_style_change_selection_automatically(app, tmp_path):
    assessed(app, source(tmp_path))
    app.target.set('2')
    pump(app, lambda: app.project.target == 2)
    assert [p.relative for p in app.project.selection()] == ['a.jpg', 'c.jpg']
    app.style.set('Strict quality')
    pump(app, lambda: app.project.style == 'Strict quality')
    assert [p.relative for p in app.project.selection()] == ['a.jpg', 'b.jpg']
    app.target.set('')
    pump(app, lambda: app.project.target is None)
    assert len(app.project.selection()) == 3
    app.target.set('invalid')
    pump(app, lambda: not app.preferences_valid)
    assert app.project.target is None
    assert not app.errors  # Invalid text is explained inline, without modal spam.


def test_protection_checkbox_keeps_essential_photo_in_limited_selection(app, tmp_path):
    assessed(app, source(tmp_path))
    app.table.selection_set('2')
    app.preview()
    app.protection_control.invoke()
    assert app.project.photos[2].must_include
    app.target.set('1')
    pump(app, lambda: app.project.target == 1)
    assert app.project.selection()[0].relative == 'c.jpg'
    assert 'c.jpg' not in [p.relative for p in app.project.cleanup_candidates()]


def test_finish_saves_named_project_and_summary_then_keeps_new_edits_private(app, tmp_path, monkeypatch):
    folder = source(tmp_path)
    assessed(app, folder)
    output = tmp_path / 'finished.iqa.json'
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: str(output))
    summaries = []
    monkeypatch.setattr(app, 'show_finish_summary', summaries.append)
    assert app.finish()
    saved = Project.load(output)
    assert saved.finished_at and saved.finish_summary['remaining'] == 3
    assert saved.finish_summary['selected'] == 3
    assert summaries[0]['pending'] == 0
    assert app.saved_project_file == output and not app.dirty
    assert app.project_file != output
    assert len(list(folder.iterdir())) == 3  # Finish does not copy or delete photos.
    app.table.selection_set('0')
    app.decide('Review')
    assert app.dirty and app.project.finished_at is None
    assert Project.load(output).photos[0].decision == 'Keep'
    assert Project.load(app.project_file).photos[0].decision == 'Review'


def test_finish_cancel_keeps_work_and_partial_finish_is_saved_for_later(app, tmp_path, monkeypatch):
    assert app.start_folder(source(tmp_path))
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: '')
    assert not app.finish(show_summary=False)
    assert app.dirty and app.project_file.exists()
    output = tmp_path / 'partial.iqa.json'
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: str(output))
    assert app.finish(show_summary=False)
    saved = Project.load(output)
    assert saved.finished_at is None
    assert saved.finish_summary['pending'] == 3
    assert app.session_store.unfinished()


def test_open_saved_project_reuses_assessments_without_initial_save_prompt(app, tmp_path, monkeypatch):
    folder = source(tmp_path)
    project = Project(str(folder))
    project.scan()
    for photo in project.photos:
        photo.status, photo.score = 'Keep', 88
    path = tmp_path / 'prior.iqa.json'
    project.finish(path)
    monkeypatch.setattr('src.desktop.app.filedialog.askopenfilename', lambda **kwargs: str(path))
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: pytest.fail('Opening asked for a save location'))
    app.open()
    assert app.saved_project_file == path
    assert all(p.status == 'Keep' and p.score == 88 for p in app.project.photos)
    assert app.project_file != path and not app.dirty


def test_switching_folders_can_leave_a_recoverable_unfinished_review(app, tmp_path, monkeypatch):
    assessed(app, source(tmp_path))
    original = app.project_file
    other = source(tmp_path, 'another-event')
    monkeypatch.setattr('src.desktop.app.filedialog.askdirectory', lambda: str(other))
    monkeypatch.setattr('src.desktop.app.messagebox.askyesnocancel', lambda *args, **kwargs: False)
    app.choose_folder()
    assert app.project.source == str(other)
    assert any(path == original for path, project in app.session_store.unfinished())
    assert Project.load(original).photos[0].score == 90


def test_finish_rejects_source_folder_and_failed_save_preserves_recovery(app, tmp_path, monkeypatch):
    folder = source(tmp_path)
    assessed(app, folder)
    forbidden = folder / 'project.iqa.json'
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: str(forbidden))
    assert not app.finish(show_summary=False)
    assert not forbidden.exists() and app.project_file.exists()
    assert app.project.finished_at is None and app.dirty
    assert 'outside' in app.errors[-1]


def test_new_folder_shot_preferences_apply_without_creating_another_project(app, tmp_path, monkeypatch):
    assert app.start_folder(source(tmp_path))
    app.shot.set('group')
    captured = []
    def worker(project):
        captured.append(project.shot_type)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr('src.desktop.app.AnalysisSession', worker)
    app.analyze()
    assert captured == ['group'] and not app.errors


def test_session_store_refuses_checkpoints_inside_photo_folder(tmp_path):
    folder = source(tmp_path)
    store = SessionStore(folder / 'recovery')
    with pytest.raises(ValueError, match='recovery directory'):
        store.create(Project(str(folder)))


def test_finish_after_cleanup_reports_remaining_and_removed_photos(app, tmp_path, monkeypatch):
    folder = source(tmp_path)
    assessed(app, folder)
    app.project.photos[0].override = 'Excluded'
    bin_folder = tmp_path / 'test-bin'
    bin_folder.mkdir()
    app.project.recycle(['a.jpg'], app.project_file, trash=lambda path: path.rename(bin_folder / path.name))
    output = tmp_path / 'finished.iqa.json'
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: str(output))
    assert app.finish(show_summary=False)
    result = Project.load(output)
    assert result.finish_summary['remaining'] == 2
    assert result.finish_summary['recycled'] == 1
    assert result.cleanup_history[-1]['files'][0]['outcome'] == 'recycled'
    assert len(list(folder.iterdir())) == 2


def test_finish_save_failure_does_not_claim_completion_or_lose_recovery(app, tmp_path, monkeypatch):
    assessed(app, source(tmp_path))
    output = tmp_path / 'finished.iqa.json'
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: str(output))
    save = app.project.save
    def failing_save(path):
        if Path(path) == output:
            raise OSError('Destination disk is full')
        save(path)
    monkeypatch.setattr(app.project, 'save', failing_save)
    assert not app.finish(show_summary=False)
    assert app.dirty and app.project.finished_at is None
    assert app.project_file.exists() and not output.exists()
    assert 'disk is full' in app.errors[-1]


def test_close_then_cancel_save_keeps_review_open(app, tmp_path, monkeypatch):
    assessed(app, source(tmp_path))
    monkeypatch.setattr('src.desktop.app.messagebox.askyesnocancel', lambda *args, **kwargs: True)
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: '')
    app.close()
    assert app.root.winfo_exists() and app.dirty
    assert app.project_file.exists()


def test_finish_summary_has_done_action_that_closes_saved_project(app, tmp_path, monkeypatch):
    assessed(app, source(tmp_path))
    output = tmp_path / 'finished.iqa.json'
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: str(output))
    closed = []
    destroy = app.root.destroy
    def close():
        closed.append(True)
        destroy()
    monkeypatch.setattr(app.root, 'destroy', close)
    assert app.finish()
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    button = next(widget for widget in descendants(app.root)
                  if widget.winfo_class() == 'TButton' and widget.cget('text') == 'Done')
    button.invoke()
    assert closed == [True] and output.is_file()


def test_early_finish_can_save_unanalyzed_preferences_for_later(app, tmp_path, monkeypatch):
    assert app.start_folder(source(tmp_path))
    app.shot.set('group')
    output = tmp_path / 'later.iqa.json'
    monkeypatch.setattr('src.desktop.app.filedialog.asksaveasfilename', lambda **kwargs: str(output))
    assert app.finish(show_summary=False)
    saved = Project.load(output)
    assert saved.shot_type == 'group' and saved.finished_at is None
    assert saved.finish_summary['pending'] == 3


def test_reanalysis_with_changed_intent_preserves_manual_decisions(app, tmp_path, monkeypatch):
    assessed(app, source(tmp_path))
    app.project.photos[0].override = 'Keep'
    app.project.photos[0].must_include = True
    app.shot.set('group')
    monkeypatch.setattr('src.desktop.app.messagebox.askyesno', lambda *args, **kwargs: True)
    monkeypatch.setattr('src.desktop.app.AnalysisSession', lambda project: SimpleNamespace(close=lambda: None))
    app.analyze()
    assert all(p.status == 'Pending' for p in app.project.photos)
    assert app.project.photos[0].override == 'Keep' and app.project.photos[0].must_include
    assert app.project.shot_type == 'group'
