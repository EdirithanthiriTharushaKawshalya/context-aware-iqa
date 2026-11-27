"""Genuine DNG sensor decoding, preview policy, migration, and original-file safety."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
from PIL import Image, TiffImagePlugin
import pytest
import rawpy

from src.desktop import image_io
from src.desktop.duplicates import capture_metadata
from src.desktop.gallery import read_thumbnail
from src.desktop.previews import read_preview
from src.desktop.project import Project
from src.desktop.raw_check import verify_raw


@pytest.fixture
def dng(tmp_path):
    source = tmp_path/'camera'
    source.mkdir()
    path = source/'portrait.DNG'
    pixels = np.arange(128*96, dtype=np.uint16).reshape(96, 128) % 3000+400
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    for tag, value in {262: 32803, 271: 'Synthetic', 272: 'Test Bayer', 274: 6,
                       33421: (2, 2), 33422: bytes([0, 1, 1, 2]),
                       50706: bytes([1, 4, 0, 0]), 50707: bytes([1, 1, 0, 0]),
                       50708: 'Synthetic Test Bayer', 50713: (1, 1), 50714: 0,
                       50717: 4095, 50721: (1., 0., 0., 0., 1., 0., 0., 0., 1.),
                       50728: (1., 1., 1.), 50778: 21}.items():
        tags[tag] = value
    Image.fromarray(pixels).save(path, format='TIFF', tiffinfo=tags)
    return path


def pipeline():
    return SimpleNamespace(run=Mock(return_value=SimpleNamespace(
        unified_score=85., classification='Keep', review_reasons=[], s_local=80., s_global=90.)))


def test_real_sensor_decode_and_full_resolution_inspection_preserve_source(dng):
    before = dng.read_bytes()
    full = image_io.read_photo(dng)
    assert full.mode == 'RGB' and full.size == (96, 128)
    assert full.info['raw_render_profile'] == image_io.RAW_RENDER_PROFILE
    assert full.getexif().get(274, 1) == 1
    assert read_preview(dng).size == full.size
    assert read_thumbnail(dng).size == (48, 64)  # No embedded preview: half-size fallback.
    fitted = read_preview(dng, (20, 30))
    assert fitted.width <= 20 and fitted.height <= 30
    assert dng.read_bytes() == before
    # Native handles must be released after decoding.
    renamed = dng.with_name('renamed.DNG')
    dng.rename(renamed)
    renamed.rename(dng)


@pytest.mark.parametrize('already_rotated', [False, True])
def test_embedded_preview_is_fast_and_oriented_once(dng, monkeypatch, already_rotated):
    embedded = Image.new('RGB', (60, 100) if already_rotated else (100, 60), 'blue')
    buffer = BytesIO()
    exif = Image.Exif()
    if not already_rotated:
        exif[274] = 6
    embedded.save(buffer, format='JPEG', exif=exif)
    raw = MagicMock()
    raw.__enter__.return_value = raw
    raw.sizes = SimpleNamespace(width=100, height=60, flip=6)
    raw.extract_thumb.return_value = SimpleNamespace(format=rawpy.ThumbFormat.JPEG, data=buffer.getvalue())
    monkeypatch.setattr(rawpy, 'RawPy', lambda: raw)
    image = read_thumbnail(dng)
    assert image.size == (60, 100)
    raw.unpack.assert_not_called()
    raw.postprocess.assert_not_called()
    raw.__exit__.assert_called_once()


def test_corrupt_raw_does_not_fall_back_to_scoring_a_jpeg(dng, monkeypatch):
    raw = MagicMock()
    raw.__enter__.return_value = raw
    raw.sizes = SimpleNamespace(width=128, height=96, flip=6)
    raw.unpack.side_effect = rawpy.LibRawDataError('damaged sensor data')
    monkeypatch.setattr(rawpy, 'RawPy', lambda: raw)
    project = Project(str(dng.parent))
    project.scan()
    model = pipeline()
    project.analyze(model, None, threading.Event(), lambda _: None)
    assert project.photos[0].status == 'Error'
    assert 'RAW' in project.photos[0].reasons[0]
    model.run.assert_not_called()
    raw.extract_thumb.assert_not_called()


def test_legacy_unsupported_raw_is_migrated_without_losing_manual_choices(dng, tmp_path):
    project = Project(str(dng.parent))
    project.scan()
    photo = project.photos[0]
    photo.status, photo.override = 'Unsupported', 'Keep'
    photo.creative_override, photo.selection_choice = 'Creative', 'include'
    photo.must_include = True
    photo.technical_score = 123.
    path = tmp_path/'old-project.json'
    project.save(path)
    record = json.loads(path.read_text())
    record['photos'][0].pop('render_profile')
    path.write_text(json.dumps(record))
    restored = Project.load(path)
    restored.scan()
    migrated = restored.photos[0]
    assert migrated.status == 'Pending' and migrated.technical_score is None
    assert migrated.override == 'Keep' and migrated.must_include
    assert migrated.creative_override == 'Creative' and migrated.selection_choice == 'include'
    model = pipeline()
    restored.analyze(model, None, threading.Event(), lambda _: None)
    assert model.run.call_args.args[0].size == (96, 128)
    assert migrated.status == 'Keep' and migrated.render_profile == image_io.RAW_RENDER_PROFILE
    assert migrated.duplicate_features['file_sha256'] == hashlib.sha256(dng.read_bytes()).hexdigest()
    restored.save(path)
    resumed = Project.load(path)
    resumed.scan()
    assert resumed.photos[0].status == 'Keep'  # Valid RAW results are reused.


def test_export_copies_raw_bytes_and_leaves_sidecars_untouched(dng, tmp_path):
    before = dng.read_bytes()
    sidecar = dng.with_suffix('.xmp')
    sidecar.write_text('photographer edits')
    project = Project(str(dng.parent))
    project.scan()
    project.analyze(pipeline(), None, threading.Event(), lambda _: None)
    assert project.export(tmp_path/'export') == 1
    assert (tmp_path/'export/photos'/dng.name).read_bytes() == before
    assert dng.read_bytes() == before and sidecar.read_text() == 'photographer edits'


def test_raw_metadata_retains_numeric_orientation_original_time_and_shutter(dng, monkeypatch):
    import exifread
    class Tag:
        def __init__(self, value):
            self.values = [value]
            self.value = value
        def __str__(self):
            return str(self.value)
    tags = {key: Tag(value) for key, value in {
        'Image Make': 'SONY', 'Image Model': 'ILCE-7M3', 'Image Orientation': 8,
        'EXIF DateTimeOriginal': '2026:11:27 10:30:00', 'EXIF SubSecTimeOriginal': '125',
        'EXIF OffsetTimeOriginal': '+05:30', 'EXIF ExposureTime': .008}.items()}
    monkeypatch.setattr(exifread, 'process_file', lambda *a, **k: tags)
    image_io._raw_tags.cache_clear()
    try:
        exif = image_io.raw_exif(dng)
        assert exif[274] == 8 and float(exif[33434]) == .008
        assert capture_metadata(exif)['captured_at'] is not None
        assert capture_metadata(exif)['camera'] == 'SONY|ILCE-7M3'
    finally:
        image_io._raw_tags.cache_clear()


def test_headless_raw_check_verifies_native_decoder(dng, tmp_path):
    report = verify_raw(dng, tmp_path/'raw-check.json')
    assert report['full_resolution_size'] == [96, 128]
    assert report['source_unchanged']
    with pytest.raises(ValueError, match='separate'):
        verify_raw(dng, dng)


def test_confirmed_raw_cleanup_releases_handles_and_leaves_sidecars(dng, tmp_path):
    sidecar = dng.with_suffix('.xmp')
    sidecar.write_text('edits')
    project = Project(str(dng.parent))
    project.scan()
    project.analyze(pipeline(), None, threading.Event(), lambda _: None)
    read_thumbnail(dng)
    project.photos[0].override = 'Excluded'
    trash = tmp_path/'fake-trash'
    trash.mkdir()
    report = project.recycle([dng.name], tmp_path/'recovery.json',
                             trash=lambda path: path.rename(trash/path.name))
    assert report['recycled'] == [dng.name] and not report['errors']
    assert sidecar.read_text() == 'edits'


def test_changed_raw_focus_profile_requeues_old_assessments(dng):
    project = Project(str(dng.parent))
    project.scan()
    photo = project.photos[0]
    photo.status, photo.render_profile = 'Review', 'previous-raw-profile'
    photo.override, photo.must_include = 'Keep', True
    photo.review_reasons = ['Previous detail estimate.']
    project.scan()
    assert photo.status == 'Pending' and photo.review_reasons is None
    assert photo.override == 'Keep' and photo.must_include
