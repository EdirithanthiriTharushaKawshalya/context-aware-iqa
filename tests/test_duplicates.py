"""Synthetic burst behavior, not a real-world recognition accuracy benchmark."""
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter, PngImagePlugin
import pytest

from src.desktop.duplicates import capture_metadata, describe, group_photos, near_match
from src.desktop.project import Photo, Project


def scene(seed=7):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(40,150,(240,320,3),dtype=np.uint8))
    image = image.filter(ImageFilter.GaussianBlur(1))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20+seed%40,30,120,150),fill=(30+seed%20,30,70))
    draw.ellipse((170,100+seed%30,270,210),fill=(190,150,120))
    return image


def photo(name, image=None, captured=100., score=80., status='Keep', local=None):
    image = scene() if image is None else image
    result = Photo(name,1,1,status=status,score=score,technical_score=score)
    result.duplicate_features = describe(image,local,{'captured_at':captured,'camera':'camera-a'})
    return result


def test_exact_visual_copies_match_across_names_folders_and_capture_dates():
    a = photo('folder/a.jpg',captured=1.)
    b = photo('different/export.png',captured=1000000.)
    assert near_match(a,b)
    groups = group_photos([b,a])
    assert len(groups) == 1 and len(groups[0]) == 2
    assert a.duplicate_kind == b.duplicate_kind == 'exact'


def test_recompressed_blurred_burst_groups_and_better_quality_is_selected():
    image = scene()
    a = photo('DSC_001.jpg',image,score=90.)
    b = photo('DSC_002.jpg',image.filter(ImageFilter.GaussianBlur(.7)),captured=101.,score=60.)
    c = photo('DSC_003.jpg',image.filter(ImageFilter.GaussianBlur(1.2)),captured=102.,score=30.,status='Review')
    d = photo('DSC_004.jpg',image.filter(ImageFilter.GaussianBlur(.3)),captured=100.5,score=70.)
    e = photo('DSC_005.jpg',image.filter(ImageFilter.GaussianBlur(.9)),captured=101.5,score=50.)
    assert near_match(a,b) and near_match(a,c)
    project = Project('.',photos=[c,b,a,d,e])
    project.update_duplicate_groups()
    assert len(project.duplicate_recommendations()) == 1
    assert a.duplicate_count == 5 and a.duplicate_label == 'Best of 5'
    assert a.duplicate_best and a.duplicate_kind == 'burst'
    assert project.selection() == [a]
    assert all(p.override == p.selection_choice == '' for p in project.photos)


def test_jpeg_recompression_is_a_similar_candidate_not_an_exact_copy():
    image = scene()
    buffer = BytesIO()
    image.save(buffer,format='JPEG',quality=82)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        a,b = photo('DSC_001.jpg',image),photo('DSC_002.jpg',encoded.convert('RGB'))
    assert a.duplicate_features['pixels_sha256'] != b.duplicate_features['pixels_sha256']
    assert near_match(a,b)
    group_photos([a,b])
    assert a.duplicate_kind == 'burst'


def test_matching_does_not_use_file_copy_time_or_bridge_distant_bursts():
    image = scene()
    a = photo('DSC_001.jpg',image,captured=100.)
    b = photo('DSC_002.jpg',image.filter(ImageFilter.GaussianBlur(.5)),captured=200.)
    a.modified = b.modified = 123456789
    assert not near_match(a,b)
    assert group_photos([a,b]) == []


def test_missing_exif_requires_nearby_camera_filenames():
    a = photo('event/DSC_001.jpg',captured=None)
    b = photo('event/DSC_002.jpg',scene().filter(ImageFilter.GaussianBlur(.4)),captured=None)
    assert near_match(a,b)
    b.relative = 'unrelated/export.jpg'
    assert not near_match(a,b)
    b.relative = 'event/DSC_999.jpg'
    assert not near_match(a,b)


def test_flat_images_different_colors_cameras_and_subject_counts_are_not_near_duplicates():
    assert not near_match(photo('a1.jpg',Image.new('RGB',(300,200),'gray')),
                          photo('a2.jpg',Image.new('RGB',(300,200),(129,129,129))))
    assert not near_match(photo('a1.jpg',scene(1)),photo('a2.jpg',scene(30)))
    a = photo('a1.jpg')
    b = photo('a2.jpg',scene().filter(ImageFilter.GaussianBlur(.5)))
    b.duplicate_features['camera'] = 'camera-b'
    assert not near_match(a,b)
    a.duplicate_features['captured_at'] = b.duplicate_features['captured_at'] = None
    assert not near_match(a,b)  # Camera disagreement also matters without EXIF dates.
    b.duplicate_features['camera'] = 'camera-a'
    a.duplicate_features['subject_count'] = 1
    b.duplicate_features['subject_count'] = 2
    assert not near_match(a,b)


def test_same_background_with_a_changed_subject_region_is_not_grouped():
    image = scene()
    changed = image.copy()
    changed.paste(Image.new('RGB',(60,100),'black'),(130,60))
    regions = {'subjects':[{'subject_bbox':[130,60,190,160]}]}
    a,b = photo('a1.jpg',image,local=regions),photo('a2.jpg',changed,local=regions)
    assert not near_match(a,b)


def test_complete_link_prevents_chaining_a_sequence_of_different_exposures():
    array = np.asarray(scene()).astype(np.int16)
    frames = [photo(f'DSC_{i:03}.jpg',Image.fromarray((array+i*10).astype(np.uint8)),captured=100.+i) for i in range(3)]
    assert near_match(frames[0],frames[1]) and near_match(frames[1],frames[2])
    assert not near_match(frames[0],frames[2])
    groups = group_photos(frames)
    assert len(groups) == 1 and len(groups[0]) == 2
    before = [(p.relative,p.duplicate_group) for p in frames]
    group_photos(list(reversed(frames)))
    assert before == [(p.relative,p.duplicate_group) for p in frames]


def test_selection_keeps_manual_variants_and_never_backfills_with_duplicate_frames():
    a,b = photo('DSC_1.jpg',score=90),photo('DSC_2.jpg',score=70)
    unique = photo('different.jpg',scene(55),score=80)
    project = Project('.',target=5,photos=[a,b,unique])
    project.update_duplicate_groups()
    assert project.selection() == [a,unique]
    project.choose_photo(b.relative,True)
    assert project.selection() == [b,unique]
    a.must_include = True
    assert {p.relative for p in project.selection()} == {a.relative,b.relative,unique.relative}
    project.one_per_burst = False
    assert len(project.selection()) == 3


def test_keep_candidates_rank_before_high_scoring_review_and_face_detail_breaks_ties():
    a = photo('a1.jpg',score=60.)
    b = photo('a2.jpg',score=99.,status='Review')
    c = photo('a3.jpg',score=60.)
    a.burst_detail_score,c.burst_detail_score = 35.,90.
    project = Project('.',photos=[a,b,c])
    project.update_duplicate_groups()
    assert c.duplicate_best and project.selection() == [c]
    a.status = c.status = 'Review'
    assert project.selection() == []
    assert project.duplicate_recommendations() and b.duplicate_best


def test_use_another_burst_frame_is_persisted_without_changing_other_quality_decisions(tmp_path):
    a,b = photo('a1.jpg',score=90.),photo('a2.jpg',score=70.,status='Review')
    project = Project('.',photos=[a,b])
    project.update_duplicate_groups()
    project.choose_burst_photo(b.relative)
    assert project.selection() == [b]
    assert a.decision == 'Keep' and a.selection_choice == 'exclude'
    assert b.status == 'Review' and b.decision == 'Keep'
    path = tmp_path/'project.json'
    project.save(path)
    restored = Project.load(path)
    assert [p.relative for p in restored.selection()] == [b.relative]
    assert restored.one_per_burst
    a.must_include = True
    with pytest.raises(ValueError,match='Unprotect'):
        project.choose_burst_photo(b.relative)


def test_existing_assessments_gain_fingerprints_without_reanalysis_and_changed_files_invalidate(tmp_path):
    root = tmp_path/'photos'
    root.mkdir()
    image = scene()
    image.save(root/'original.png')
    info = PngImagePlugin.PngInfo()
    info.add_text('Export','Copy with different metadata')
    image.save(root/'copy.png',pnginfo=info)
    project = Project(str(root))
    project.scan()
    for i,p in enumerate(project.photos):
        p.status,p.score,p.override = 'Keep',80.+i,'Keep'
    project.refresh_duplicate_fingerprints()
    assert len(project.duplicate_recommendations()) == 1
    assert all(p.override == 'Keep' for p in project.photos)
    image.filter(ImageFilter.GaussianBlur(7)).save(root/'copy.png')
    project.scan()
    assert next(p for p in project.photos if p.relative=='copy.png').status == 'Pending'
    assert not project.duplicate_recommendations()


def test_metadata_preserves_subseconds_and_timezone_without_using_generic_modify_date():
    exif = {36867:'2026:11:22 10:00:00',37521:'250',36881:'+05:30',271:'Camera',272:'Model',42033:'123'}
    parsed = capture_metadata(exif)
    assert parsed['captured_at'] % 1 == .25
    assert parsed['camera'] == 'Camera|Model|123'
    assert capture_metadata({306:'2026:11:22 10:00:00'})['captured_at'] is None
