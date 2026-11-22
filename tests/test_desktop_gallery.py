"""Delivery selection, virtual thumbnails and exact final-folder approval."""
import time
import tkinter as tk
from pathlib import Path

import pytest
from PIL import Image

from src.desktop.app import DesktopApp
from src.desktop.project import Photo, Project
from src.desktop.sessions import SessionStore


def project_at(tmp_path, count=5, target=2):
    source = tmp_path/'originals'
    source.mkdir()
    for i in range(count):
        Image.new('RGB', (160,120), (40+i*10,90,120)).save(source/f'photo{i}.jpg')
    project = Project(str(source), target=target, style='Strict quality')
    project.scan()
    for i, photo in enumerate(project.photos):
        photo.status, photo.score = ('Review' if i == count-1 else 'Keep'), 90-i*10
    return project


@pytest.fixture
def app(tmp_path, monkeypatch, desktop_tk):
    window = tk.Toplevel(desktop_tk)
    window.geometry('1400x960')
    application = DesktopApp(window, SessionStore(tmp_path/'recovery'))
    project = project_at(tmp_path)
    application.activate(project, application.session_store.create(project))
    monkeypatch.setattr('src.desktop.app.messagebox.showinfo', lambda *a, **k: None)
    monkeypatch.setattr('src.desktop.app.messagebox.showerror', lambda *a, **k: pytest.fail(str(a)))
    yield application
    for callback in desktop_tk.tk.splitlist(desktop_tk.tk.call('after', 'info')):
        window.after_cancel(callback)
    if window.winfo_exists():
        window.destroy()


def pump(app, condition):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        app.root.update()
        if condition():
            return
        time.sleep(.02)
    pytest.fail('UI did not reach the expected state')


def test_review_photo_replaces_automatic_choice_and_survives_save(tmp_path):
    project = project_at(tmp_path)
    review = project.photos[-1]
    project.choose_photo(review.relative, True)
    assert [p.relative for p in project.selection()] == [review.relative, 'photo0.jpg']
    assert review.status == 'Review' and review.decision == 'Keep'
    project.choose_photo('photo0.jpg', False)
    assert {p.relative for p in project.selection()} == {review.relative, 'photo1.jpg'}
    assert project.photos[0].decision == 'Keep'  # Delivery membership differs from quality.
    path = tmp_path/'saved.json'
    project.save(path)
    restored = Project.load(path)
    assert {p.relative for p in restored.selection()} == {review.relative, 'photo1.jpg'}


def test_explicit_choices_and_protection_are_never_silently_displaced(tmp_path):
    project = project_at(tmp_path, target=1)
    first, last = project.photos[0], project.photos[-1]
    project.choose_photo(first.relative, True)
    with pytest.raises(ValueError, match='explicitly chosen'):
        project.choose_photo(last.relative, True)
    assert last.override == last.selection_choice == ''
    first.must_include = True
    with pytest.raises(ValueError, match='Unprotect'):
        project.choose_photo(first.relative, False)
    first.must_include = False
    project.choose_photo(first.relative, False)
    project.choose_photo(last.relative, True)
    assert project.selection() == [last]


def test_finalization_plan_excludes_selected_protected_and_unreadable_files(tmp_path):
    project = project_at(tmp_path)
    project.photos[2].status = 'Error'
    project.photos[3].status = 'Unsupported'
    project.photos[2].must_include = True
    project.photos[-1].possible_creative = True
    plan = project.finalization_plan()
    assert {p.relative for p in plan['chosen']} == {'photo0.jpg','photo1.jpg'}
    assert plan['remove'] == [project.photos[-1]]  # Explicit plan can include creative candidates.
    assert len(plan['retained']) == 2
    assert all(p.override == '' for p in project.photos)
    assert all(project.path_for(p).exists() for p in project.photos)


def test_gallery_is_default_and_filter_and_choice_update_together(app):
    app.root.update()
    assert app.library_tabs.select() == str(app.gallery)
    app.view.set('Review')
    app.refresh()
    assert [i for i, _ in app.gallery.rows] == ['4']
    app.toggle_chosen('4')
    app.view.set('Final selection')
    app.refresh()
    assert {p.relative for _, p in app.gallery.rows} == {'photo0.jpg','photo4.jpg'}
    assert app.project.photos[-1].status == 'Review'
    app.view.set('Not chosen')
    app.refresh()
    assert len(app.gallery.rows) == 3
    pump(app, lambda: bool(app.gallery.photo_images))


def test_large_gallery_only_draws_visible_cards(app):
    template = app.project.photos[0]
    app.project.photos = [Photo(f'missing-{i}.jpg', template.size, template.modified, status='Keep', score=80.) for i in range(1500)]
    app.refresh()
    app.root.update()
    assert len(app.gallery.rows) == 1500
    assert len(app.gallery.visible_indices) < 50
    assert len(app.gallery.canvas.find_all()) < 500
    assert len(app.gallery.pending) <= 24
    app.gallery.scroll('moveto', 1)
    assert max(app.gallery.visible_indices) == 1499


def test_gallery_and_primary_actions_fit_the_smallest_supported_window(app):
    app.root.geometry('1120x760')
    app.root.update()
    assert not app.advanced_settings_open
    assert app.gallery.canvas.winfo_width() >= app.gallery.CARD_WIDTH
    assert app.gallery.canvas.winfo_height() >= app.gallery.CARD_HEIGHT+app.gallery.GAP
    for name in ('analyze', 'pause', 'finish'):
        button = app.action_buttons[name]
        right = button.winfo_rootx()-app.root.winfo_rootx()+button.winfo_width()
        assert right <= app.root.winfo_width()
    assert app.gallery.winfo_ismapped()


def test_finalization_cancel_does_not_mark_or_move_any_files(app):
    window = app.finalize()
    assert window and window.winfo_exists()
    window.destroy()
    assert all(p.override == '' for p in app.project.photos)
    assert len(list(Path(app.project.source).glob('*.jpg'))) == 5


def test_declining_the_exact_cleanup_confirmation_leaves_gallery_open(app, monkeypatch):
    monkeypatch.setattr('src.desktop.finalize.messagebox.askyesno', lambda *a, **k: False)
    window = app.finalize()
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    button = next(widget for widget in descendants(window)
                  if widget.winfo_class() == 'TButton' and str(widget.cget('text')).startswith('Recycle '))
    button.invoke()
    assert window.winfo_exists() and not app.busy
    assert all(p.override == '' for p in app.project.photos)
    assert len(list(Path(app.project.source).glob('*.jpg'))) == 5
    window.destroy()


def test_changed_source_blocks_final_cleanup_before_marking(app):
    plan = app.project.finalization_plan()
    app.project.path_for(plan['remove'][-1]).write_bytes(b'changed since preview')
    with pytest.raises(ValueError, match='Source changed'):
        app.start_final_cleanup([p.relative for p in plan['chosen']], [p.relative for p in plan['remove']])
    assert all(p.override == '' for p in app.project.photos)
    assert not app.busy


def test_saved_approval_cannot_remove_an_old_selection_after_choices_change(app):
    plan = app.project.finalization_plan()
    app.project.choose_photo('photo4.jpg', True)
    with pytest.raises(ValueError, match='chosen photos changed'):
        app.start_final_cleanup([p.relative for p in plan['chosen']], [p.relative for p in plan['remove']])
    assert not app.busy and all(app.project.path_for(p).exists() for p in app.project.photos)


def test_new_photo_after_preview_blocks_folder_finalization(app):
    plan = app.project.finalization_plan()
    Image.new('RGB', (50,50), 'white').save(Path(app.project.source)/'new.jpg')
    with pytest.raises(ValueError, match='source folder changed'):
        app.start_final_cleanup([p.relative for p in plan['chosen']], [p.relative for p in plan['remove']])
    assert not app.busy
    assert len(list(Path(app.project.source).glob('*.jpg'))) == 6


def test_checkpoint_failure_rolls_back_all_removal_marks(app, monkeypatch):
    plan = app.project.finalization_plan()
    def fail():
        raise OSError('Disk full')
    monkeypatch.setattr(app, 'checkpoint', fail)
    with pytest.raises(OSError, match='Disk full'):
        app.start_final_cleanup([p.relative for p in plan['chosen']], [p.relative for p in plan['remove']])
    assert all(p.override == '' for p in app.project.photos)
    assert not app.busy


def test_final_cleanup_moves_only_unchosen_and_then_opens_project_save(app, tmp_path, monkeypatch):
    project = app.project
    project.choose_photo('photo4.jpg', True)
    plan = project.finalization_plan()
    recycle = project.recycle
    fake_bin = tmp_path/'disposable-test-bin'
    fake_bin.mkdir()
    def recycle_disposable(approved, project_file, **kwargs):
        return recycle(approved, project_file, trash=lambda path: path.rename(fake_bin/path.name), **kwargs)
    monkeypatch.setattr(project, 'recycle', recycle_disposable)
    finished = []
    monkeypatch.setattr(app, 'finish', lambda: finished.append(True))
    app.start_final_cleanup([p.relative for p in plan['chosen']], [p.relative for p in plan['remove']])
    pump(app, lambda: (app.poll_cleanup() or not app.cleaning))
    assert finished == [True]
    assert {p.name for p in Path(project.source).glob('*.jpg')} == {'photo0.jpg','photo4.jpg'}
    assert len(list(fake_bin.glob('*.jpg'))) == 3
    assert len(Project.load(app.project_file).cleanup_history[-1]['files']) == 3
