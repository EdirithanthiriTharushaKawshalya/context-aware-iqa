"""Regression checks for responsiveness, cache reuse, and saved review fidelity."""
import json
import threading
import time
import tkinter as tk
from dataclasses import asdict
from unittest.mock import Mock

from PIL import Image
import pytest

from src.desktop.app import DesktopApp
from src.desktop.previews import PreviewLoader, read_preview
from src.desktop.project import Photo, Project
from src.desktop.welcome import Preferences


def pump(root, condition):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(.005)
    pytest.fail('Background display work did not complete')


@pytest.fixture
def app(desktop_tk, tmp_path):
    root = tk.Toplevel(desktop_tk)
    application = DesktopApp(root, preferences=Preferences(tmp_path/'preferences.json'))
    source = tmp_path/'photos'
    source.mkdir()
    for index in range(3):
        Image.new('RGB', (80, 60), (index*70, 80, 120)).save(source/f'{index}.jpg')
    project = Project(str(source))
    project.scan()
    for photo in project.photos:
        photo.status, photo.score = 'Keep', 90.
    application.activate(project, tmp_path/'recovery.json')
    root.update()
    yield application
    for timer in root.tk.splitlist(root.tk.call('after', 'info')):
        root.after_cancel(timer)
    root.destroy()


def test_latest_preview_wins_without_blocking_events(desktop_tk):
    entered, release = threading.Event(), threading.Event()
    decoded, displayed, ticks = [], [], []
    main_thread = threading.get_ident()
    def decode(path, size):
        assert threading.get_ident() != main_thread
        decoded.append(path)
        if path == 'first':
            entered.set()
            release.wait(5)
        return Image.new('RGB', (30, 30))
    loader = PreviewLoader(desktop_tk, decoder=decode, max_pixels=1000)
    try:
        callback = lambda key, image, error: displayed.append(key)
        loader.request('first', 'first', (30, 30), callback)
        assert entered.wait(2)
        desktop_tk.after(0, lambda: ticks.append('responsive'))
        desktop_tk.update()
        assert ticks == ['responsive'] and not displayed
        loader.request('obsolete', 'obsolete', (30, 30), callback)
        loader.request('latest', 'latest', (30, 30), callback)
        release.set()
        pump(desktop_tk, lambda: displayed == ['latest'])
        assert decoded == ['first', 'latest']
        assert loader.cache_pixels <= 1000
        loader.request('latest', 'latest', (30, 30), callback)
        assert displayed == ['latest', 'latest'] and decoded == ['first', 'latest']
    finally:
        release.set()
        loader.close()


def test_display_preview_honors_orientation_and_full_resolution(tmp_path):
    path = tmp_path/'rotated.jpg'
    image = Image.new('RGB', (800, 400), 'green')
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    assert read_preview(path, (100, 200)).size == (100, 200)
    assert read_preview(path).size == (400, 800)


def test_unchanged_refresh_does_not_rewrite_table_rows(app, monkeypatch):
    item = Mock(wraps=app.table.item)
    monkeypatch.setattr(app.table, 'item', item)
    app.refresh()
    assert item.call_count == 0
    app.project.photos[1].score = 42.
    app.refresh()
    assert item.call_count == 1
    assert item.call_args.args[0] == '1'


def test_gallery_reuses_canvas_items_and_tk_images(app, monkeypatch):
    gallery = app.gallery
    pump(app.root, lambda: not gallery.pending)
    gallery.render()
    images = tuple(str(image) for image in gallery.photo_images)
    items = gallery.canvas.find_all()
    import src.desktop.gallery as gallery_module
    constructor = Mock(wraps=gallery_module.ImageTk.PhotoImage)
    monkeypatch.setattr(gallery_module.ImageTk, 'PhotoImage', constructor)
    gallery.render()
    assert gallery.canvas.find_all() == items
    gallery.selected = '1'
    gallery.render()
    assert constructor.call_count == 0
    assert tuple(str(image) for image in gallery.photo_images) == images


def test_switching_back_to_a_loaded_photo_cancels_a_pending_preview(app):
    app.select_gallery_photo('0')
    pump(app.root, lambda: app._preview_displayed == app._preview_requested)
    first = app._preview_displayed
    original_request = app.preview_loader.request
    pending = []
    app.preview_loader.request = lambda *args: pending.append(args)
    app.select_gallery_photo('1')
    assert app._preview_requested != first
    app.preview_loader.request = original_request
    app.select_gallery_photo('0')
    assert app._preview_displayed == first
    key, _, _, callback = pending[-1]
    callback(key, Image.new('RGB', (20, 20), 'red'), None)
    assert app._preview_displayed == first


def test_preview_refits_after_resizing_the_photo_panel(app):
    app.select_gallery_photo('0')
    pump(app.root, lambda: app._preview_displayed is not None)
    before = app._preview_displayed
    body = app.gallery.master.master.master
    body.sashpos(0, body.sashpos(0)-120)
    pump(app.root, lambda: app._preview_displayed is not None and app._preview_displayed != before)
    assert app._preview_displayed[-1] == (max(32, app.preview_label.winfo_width()-16),
                                        max(32, app.preview_label.winfo_height()-16))


def test_compact_checkpoint_roundtrips_all_public_fields(tmp_path):
    photo = Photo('photo.jpg', 12, 34, status='Keep', override='Keep', must_include=True,
                  creative_override='Creative', selection_choice='include',
                  duplicate_features={'version': 1, 'subjects': [{'box': [0, 0, 1, 1], 'color': 'abc'}]})
    photo._temporary_cache = 'must not be persisted'
    project = Project(str(tmp_path/'source'), photos=[photo],
                      cleanup_history=[{'files': [{'outcome': 'recycled'}]}])
    path = tmp_path/'project.json'
    project.save(path)
    assert json.loads(path.read_text()) == asdict(project)
    assert asdict(Project.load(path)) == asdict(project)


def test_scan_reuses_unchanged_reviews_and_filters_nonphotos(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'a.jpg').write_bytes(b'photo')
    (source/'notes.txt').write_text('notes')
    (source/'nested').mkdir()
    (source/'nested'/'b.NEF').write_bytes(b'raw')
    project = Project(str(source))
    project.scan()
    assert [photo.relative for photo in project.photos] == ['a.jpg', 'nested/b.NEF']
    first = project.photos[0]
    first.override = 'Keep'
    project.scan()
    assert project.photos[0] is first and first.override == 'Keep'
    project.recursive = False
    project.scan()
    assert project.photos == [first]
