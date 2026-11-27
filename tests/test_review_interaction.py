"""Wheel routing, fit-first inspection, visible reasons, and RAW focus evidence."""
from types import SimpleNamespace
from unittest.mock import Mock
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageFilter
import pytest

from src.desktop.app import DesktopApp
from src.desktop.project import Photo, Project
from src.desktop.raw_quality import RawFocusExtractor, RawAwareLocalizer, brighten
from src.desktop.review import explanation
from src.desktop.welcome import Preferences
from src.local_features import LocalFeatureExtractor


def pump(root, condition):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(.01)
    pytest.fail('Interaction did not reach its expected state')


@pytest.fixture
def app(desktop_tk, tmp_path):
    root = tk.Toplevel(desktop_tk)
    application = DesktopApp(root, preferences=Preferences(tmp_path/'preferences.json'))
    source = tmp_path/'photos'
    source.mkdir()
    for index in range(12):
        Image.new('RGB', (1600, 1100) if index == 0 else (80, 60), 'teal').save(source/f'{index:02d}.jpg')
    project = Project(str(source))
    project.scan()
    for photo in project.photos:
        photo.status, photo.score = 'Keep', 90.
    application.activate(project, tmp_path/'recovery.json')
    root.geometry('1200x840')
    root.update()
    yield application
    if getattr(application, 'viewer', None) and application.viewer.window.winfo_exists():
        application.viewer.close()
    for timer in root.tk.splitlist(root.tk.call('after', 'info')):
        root.tk.call('after', 'cancel', timer)
    root.destroy()


def test_wheel_over_gallery_scrolls_even_when_entry_has_focus(app, monkeypatch):
    canvas = app.gallery.canvas
    host = app.root._root()
    monkeypatch.setattr(host, 'winfo_containing', lambda x, y: canvas)
    app.count_control.focus_force()
    app.root.update()
    assert 'IQA.GalleryWheel' in app.count_control.bindtags()
    before = canvas.yview()[0]
    observed = Mock(wraps=app.gallery.wheel)
    monkeypatch.setattr(app.gallery, 'wheel', observed)
    app.count_control.event_generate('<MouseWheel>', delta=-120, rootx=100, rooty=100)
    assert observed.call_count == 1
    assert canvas.yview()[0] > before
    canvas.yview_moveto(0)
    for _ in range(3):
        app.count_control.event_generate('<MouseWheel>', delta=-30, rootx=100, rooty=100)
    assert canvas.yview()[0] == 0
    app.count_control.event_generate('<MouseWheel>', delta=-30, rootx=100, rooty=100)
    assert canvas.yview()[0] > 0
    before = canvas.yview()[0]
    monkeypatch.setattr(host, 'winfo_containing', lambda x, y: app.count_control)
    app.count_control.event_generate('<MouseWheel>', delta=-120, rootx=100, rooty=100)
    assert canvas.yview()[0] == before
    canvas.event_generate('<MouseWheel>', delta=-120, rootx=100, rooty=100)
    assert canvas.yview()[0] == before


def test_reason_panel_scrolls_by_pointer_without_moving_the_library(app, monkeypatch):
    photo = app.project.photos[0]
    photo.status = 'Review'
    photo.review_reasons = [f'Review observation {i}' for i in range(20)]
    app.refresh()
    app.select_gallery_photo('0')
    app.root.update()
    monkeypatch.setattr(app.root._root(), 'winfo_containing', lambda x, y: app.review_text)
    before = app.gallery.canvas.yview()
    app.count_control.event_generate('<MouseWheel>', delta=-120, rootx=100, rooty=100)
    assert app.review_text.yview()[0] > 0
    assert app.gallery.canvas.yview() == before


def test_double_click_opens_fit_then_explicit_100_percent(app):
    app.open_gallery_photo('0')
    viewer = app.viewer
    assert viewer.mode == 'fit'
    pump(app.root, lambda: viewer.displayed is not None)
    assert viewer.canvas.photo.width() <= viewer.canvas.winfo_width()
    assert viewer.canvas.photo.height() <= viewer.canvas.winfo_height()
    viewer.set_mode('actual')
    pump(app.root, lambda: viewer.displayed is not None)
    assert viewer.canvas.photo.width() == 1600 and viewer.canvas.photo.height() == 1100
    viewer.set_mode('fit')
    pump(app.root, lambda: viewer.displayed is not None)
    assert viewer.canvas.photo.width() <= viewer.canvas.winfo_width()
    viewer.window.geometry('620x460')
    before = viewer.displayed
    pump(app.root, lambda: viewer.displayed is not None and viewer.displayed != before)
    assert viewer.canvas.photo.width() <= viewer.canvas.winfo_width()
    assert viewer.canvas.photo.height() <= viewer.canvas.winfo_height()


def test_review_reasons_are_prominent_and_processing_notes_are_separate(app):
    photo = app.project.photos[0]
    photo.status = 'Review'
    photo.review_reasons = ['Possible motion blur; inspect the subject at 100%.']
    photo.reasons = photo.review_reasons+['RAW assessment uses a neutral rendering.']
    app.refresh()
    app.select_gallery_photo('0')
    app.root.update()
    assert app.review_frame.winfo_ismapped()
    assert 'motion blur' in app.review_text.get('1.0', 'end')
    assert 'neutral rendering' not in app.review_text.get('1.0', 'end')
    app.decide('Keep')
    assert 'approved by you' in app.review_heading.get()
    app.select_gallery_photo('1')
    app.root.update()
    assert not app.review_frame.winfo_ismapped()


def test_manual_and_legacy_review_explanations_are_honest():
    photo = Photo('a.jpg', 1, 1, status='Keep', override='Review')
    assert 'by you' in explanation(photo)[0]
    photo.status, photo.override = 'Review', ''
    photo.reasons = ['Possible focus problem.', 'RAW assessment uses a neutral rendering.']
    assert explanation(photo)[1] == '• Possible focus problem.'
    photo.status = 'Error'
    photo.review_reasons = ['Could not decode the RAW file.']
    assert 'decode' in explanation(photo)[1]


def test_raw_detail_compensation_is_bounded_and_does_not_hide_blur_or_clipping():
    pattern = np.where((np.indices((160, 160)).sum(axis=0)//4) % 2, 120, 40).astype('uint8')
    sharp = Image.fromarray(pattern).convert('RGB').point([round(i*.05) for i in range(256)]*3)
    blurred = sharp.filter(ImageFilter.GaussianBlur(4))
    localization = {'subject_detected': False, 'face_detected': False}
    ordinary = LocalFeatureExtractor().extract(sharp, localization)
    sharp.info['raw_render_profile'] = 'test'
    blurred.info['raw_render_profile'] = 'test'
    compensated = RawFocusExtractor().extract(sharp, localization)
    soft = RawFocusExtractor().extract(blurred, localization)
    assert compensated['local_technical_score'] > ordinary['local_technical_score']+15
    assert soft['local_technical_score'] < compensated['local_technical_score']-20
    assert 1 <= compensated['raw_focus_gain'] <= 4
    assert compensated['subject_shadow_clipping'] == ordinary['subject_shadow_clipping']
    assert compensated['local_contrast'] == ordinary['local_contrast']
    jpeg = Image.fromarray(pattern).convert('RGB')
    assert RawFocusExtractor().extract(jpeg, localization) == LocalFeatureExtractor().extract(jpeg, localization)


def test_raw_detection_uses_same_coordinates_and_leaves_source_pixels_unchanged():
    image = Image.new('RGB', (120, 80), (25, 25, 25))
    image.info['raw_render_profile'] = 'test'
    before = image.tobytes()
    localizer = SimpleNamespace(localize=Mock(return_value={'subjects': []}))
    result = RawAwareLocalizer(localizer).localize(image)
    supplied = localizer.localize.call_args.args[0]
    assert supplied.size == image.size and supplied.getpixel((0, 0))[0] > 25
    assert result['raw_detection_gain'] <= 4 and image.tobytes() == before
    highlights = Image.fromarray(np.array([[120, 200, 255]], dtype='uint8'))
    lifted = np.asarray(brighten(highlights, 4.))[0]
    assert lifted[0] < lifted[1] < lifted[2] == 255


def test_raw_compensation_keeps_exposure_review_separate_from_focus():
    from src.pipeline import Phase4Pipeline, PipelineConfig
    pattern = np.where((np.indices((160, 160)).sum(axis=0)//4) % 2, 6, 2).astype('uint8')
    image = Image.fromarray(pattern).convert('RGB')
    image.info['raw_render_profile'] = 'test'
    localizer = SimpleNamespace(localize=lambda image: {
        'subjects': [], 'faces': [], 'subject_detected': False, 'face_detected': False,
        'subject_bbox': [0, 0, image.width, image.height],
        'person_detector_status': 'ok', 'face_detector_status': 'ok', 'framing_anomaly': 'NO_SUBJECT'})
    pipeline = Phase4Pipeline(localizer=localizer, local_extractor=RawFocusExtractor())
    result = pipeline.run(image, PipelineConfig(decision_mode='rules'))
    assert result.classification == 'Review'
    assert any('shadow' in reason.lower() or 'exposure' in reason.lower() for reason in result.review_reasons)
    assert not any('Low subject/detail sharpness' in reason for reason in result.review_reasons)
