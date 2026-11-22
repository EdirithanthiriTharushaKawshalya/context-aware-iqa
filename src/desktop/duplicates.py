"""Conservative exact-copy and burst candidates, independent of quality scoring.

Near matches require time/filename proximity AND perceptual, color, structural
and available subject-region agreement. Thresholds are engineering defaults;
they do not establish identity, expression quality or photographic intent.
"""
import base64
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
from pathlib import PurePosixPath
import re

import numpy as np
from PIL import Image

VERSION = 1
BURST_SECONDS = 3.
_DCT = np.cos(np.pi/32 * (np.arange(32)+.5)[None, :] * np.arange(32)[:, None])


def capture_metadata(exif):
    try:
        nested = exif.get_ifd(34665)
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        nested = {}
    def get(key):
        return nested.get(key) or exif.get(key)
    captured = None
    try:
        text = get(36867) or get(36868)  # Original/digitized time, never file-copy time.
        if text:
            stamp = datetime.strptime(str(text), '%Y:%m:%d %H:%M:%S')
            fraction = re.match(r'^[0-9]+', str(get(37521) or ''))
            if fraction:
                stamp = stamp.replace(microsecond=int((fraction[0]+'000000')[:6]))
            offset = str(get(36881) or '')
            zone = timezone.utc
            if re.fullmatch(r'[+-][0-9]{2}:[0-9]{2}', offset):
                minutes = (int(offset[1:3])*60+int(offset[4:6])) * (-1 if offset[0] == '-' else 1)
                zone = timezone(timedelta(minutes=minutes))
            captured = stamp.replace(tzinfo=zone).timestamp()
    except (ValueError, TypeError, OverflowError):
        pass
    camera = '|'.join(str(get(key) or '').strip() for key in (271,272,42033)).strip('|')
    return {'captured_at': captured, 'camera': camera}


def _pack(array):
    return base64.b64encode(np.asarray(array, dtype=np.uint8).tobytes()).decode('ascii')


def _unpack(text, shape):
    return np.frombuffer(base64.b64decode(text, validate=True), dtype=np.uint8).reshape(shape).astype(np.float32)


def describe(image, localization=None, metadata=None, file_sha256=''):
    """Input is already EXIF-oriented RGB. Store bounded visual fingerprints."""
    gray = np.asarray(image.convert('L').resize((32,32), Image.Resampling.LANCZOS))
    color = np.asarray(image.resize((16,16), Image.Resampling.LANCZOS))
    frequency = (_DCT @ gray.astype(np.float64) @ _DCT.T)[:8,:8].reshape(-1)
    bits = frequency > np.median(frequency[1:])
    bits[0] = False
    pixel_hash = hashlib.sha256(f'{image.width}x{image.height}:RGB:'.encode())
    pixel_hash.update(image.tobytes())
    subjects = []
    localization = localization or {}
    people = localization.get('subjects') if localization.get('person_detector_status') not in ('unavailable','failed') else None
    for subject in (people or [])[:16]:
        box = subject.get('subject_bbox')
        if not box or len(box) != 4 or not np.isfinite(box).all():
            continue
        x1,y1,x2,y2 = map(int, box)
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(image.width,x2),min(image.height,y2)
        if min(x2-x1,y2-y1) < 8:
            continue
        subjects.append({'box': [x1/image.width,y1/image.height,x2/image.width,y2/image.height],
                         'color': _pack(np.asarray(image.crop((x1,y1,x2,y2)).resize((8,8), Image.Resampling.LANCZOS)))})
    subjects.sort(key=lambda record: (record['box'][0],record['box'][1]))
    return dict(metadata or {}, version=VERSION, file_sha256=file_sha256,
                pixels_sha256=pixel_hash.hexdigest(), aspect=image.width/image.height,
                phash=f'{sum(int(value)<<i for i,value in enumerate(bits)):016x}',
                gray=_pack(gray), color=_pack(color), texture=float(gray.std()),
                subject_count=len(people) if people is not None else None, subjects=subjects)


def _ssim(a, b):
    mean_a,mean_b = a.mean(),b.mean()
    var_a,var_b = a.var(),b.var()
    covariance = ((a-mean_a)*(b-mean_b)).mean()
    return float(((2*mean_a*mean_b+6.5025)*(2*covariance+58.5225))/
                 ((mean_a**2+mean_b**2+6.5025)*(var_a+var_b+58.5225)))


def _filename(photo):
    path = PurePosixPath(photo.relative)
    match = re.fullmatch(r'(.*?)([0-9]+)', path.stem)
    return (str(path.parent), match[1].casefold(), int(match[2])) if match else None


def exact(a, b):
    for key in ('file_sha256','pixels_sha256'):
        if a.get(key) and a.get(key) == b.get(key):
            return True
    return False


def near_match(left, right):
    a,b = left.duplicate_features,right.duplicate_features
    if exact(a,b):
        return True
    if a.get('version') != VERSION or b.get('version') != VERSION:
        return False
    ta,tb = a.get('captured_at'),b.get('captured_at')
    timed = ta is not None and tb is not None
    if a.get('camera') and b.get('camera') and a['camera'] != b['camera']:
        return False
    if timed:
        if abs(ta-tb) > BURST_SECONDS:
            return False
    else:
        fa,fb = _filename(left),_filename(right)
        if not fa or not fb or fa[:2] != fb[:2] or abs(fa[2]-fb[2]) > 5:
            return False
    if abs(a['aspect']/b['aspect']-1) > .015 or min(a['texture'],b['texture']) < 12:
        return False
    if (int(a['phash'],16)^int(b['phash'],16)).bit_count() > (6 if timed else 4):
        return False
    # Missing region evidence is not a fabricated subject match.
    known = a.get('subject_count') is not None and b.get('subject_count') is not None
    if known and a['subject_count'] != b['subject_count']:
        return False
    threshold = .95 if timed and known else .98
    if _ssim(_unpack(a['gray'],(32,32)),_unpack(b['gray'],(32,32))) < threshold:
        return False
    if np.abs(_unpack(a['color'],(16,16,3))-_unpack(b['color'],(16,16,3))).mean() > (14 if timed else 9):
        return False
    if known:
        if a['subject_count'] > 16 or len(a['subjects']) != a['subject_count'] or len(b['subjects']) != b['subject_count']:
            return False
        for sa,sb in zip(a['subjects'],b['subjects']):
            ax,ay,bx,by = sa['box']
            cx,cy,dx,dy = sb['box']
            intersection = max(0,min(bx,dx)-max(ax,cx))*max(0,min(by,dy)-max(ay,cy))
            union = (bx-ax)*(by-ay)+(dx-cx)*(dy-cy)-intersection
            if union <= 0 or intersection/union < .75:
                return False
            if _ssim(_unpack(sa['color'],(8,8,3)),_unpack(sb['color'],(8,8,3))) < .90:
                return False
    return True


def group_photos(photos):
    """Use candidate indexes and complete-link verification; avoid chain merges."""
    clusters, index, exact_index, visual_index = [],defaultdict(set),{},defaultdict(set)
    def keys(photo):
        features = photo.duplicate_features
        if features.get('captured_at') is not None:
            bucket = int(features['captured_at']//BURST_SECONDS)
            yield ('time',bucket)
        name = _filename(photo)
        if name:
            yield ('name',name[0],name[1],name[2]//6)
    def bands(photo):
        # Any pair within six changed hash bits shares at least one of seven
        # disjoint bands. This prunes candidates without losing valid matches.
        value = int(photo.duplicate_features['phash'],16)
        for band in range(7):
            width = 10 if band == 6 else 9
            yield band,(value>>(band*9)) & ((1<<width)-1)
    for photo in sorted(photos,key=lambda p:p.relative):
        photo.duplicate_group,photo.duplicate_kind,photo.duplicate_count,photo.duplicate_best = '', '',0,False
        features = photo.duplicate_features
        if photo.recycled or photo.status not in ('Keep','Review') or features.get('version') != VERSION:
            continue
        hashes = [(key,features[key]) for key in ('file_sha256','pixels_sha256') if features.get(key)]
        found = next((exact_index[key] for key in hashes if key in exact_index),None)
        visual_candidates = set()
        for band in bands(photo):
            visual_candidates.update(visual_index.get(band,()))
        candidate_ids = set()
        for key in keys(photo):
            for delta in (-1,0,1):
                candidate_ids.update(visual_candidates.intersection(index.get(key[:-1]+(key[-1]+delta,),())))
        if found is None:
            for number in sorted(candidate_ids):
                if all(near_match(photo,member) for member in clusters[number]):
                    found = number
                    break
        if found is None:
            found = len(clusters)
            clusters.append([])
        clusters[found].append(photo)
        for key in hashes:
            exact_index[key] = found
        if features['texture'] >= 12:
            for key in keys(photo):
                index[key].add(found)
            for band in bands(photo):
                visual_index[band].add(found)
    for members in clusters:
        if len(members) < 2:
            continue
        identifier = hashlib.sha256(('duplicates-v1:'+members[0].relative).encode()).hexdigest()[:16]
        kind = 'exact' if all(exact(members[0].duplicate_features,p.duplicate_features) for p in members) else 'burst'
        for photo in members:
            photo.duplicate_group,photo.duplicate_kind,photo.duplicate_count = identifier,kind,len(members)
    return [members for members in clusters if len(members)>1]


def quality_rank(photo):
    """Keep first, then overall/weakest-subject detail; deterministic filename tie."""
    detail = photo.burst_detail_score if photo.burst_detail_score is not None else photo.technical_score
    detail = photo.score if detail is None else detail
    return (-(photo.decision == 'Keep'), -(.7*photo.score+.3*detail), -detail, -photo.score, photo.relative)
