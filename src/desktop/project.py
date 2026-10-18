"""Persistent, non-destructive collection assessment and selection."""
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
RAW = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.raf', '.rw2'}


@dataclass
class Photo:
    relative: str
    size: int
    modified: int
    status: str = 'Pending'
    score: float = 0.0
    reasons: list = field(default_factory=list)
    override: str = ''
    must_include: bool = False
    signature: str = ''
    group: str = ''
    recycled: bool = False

    @property
    def decision(self):
        return 'Recycled' if self.recycled else self.override or self.status


@dataclass
class Project:
    source: str
    target: int | None = None
    recursive: bool = True
    shot_type: str = 'auto'
    motion_intent: str = 'auto'
    style: str = 'Balanced'
    photos: list = field(default_factory=list)
    cleanup_history: list = field(default_factory=list)
    finished_at: str | None = None
    finish_summary: dict = field(default_factory=dict)

    def summary(self):
        active = [photo for photo in self.photos if not photo.recycled]
        return {
            'source': self.source,
            'remaining': len(active),
            'selected': len(self.selection()),
            'review': sum(photo.decision in ('Review', 'Error', 'Unsupported') for photo in active),
            'pending': sum(photo.status == 'Pending' for photo in active),
            'marked': sum(photo.decision == 'Excluded' for photo in active),
            'recycled': sum(photo.recycled for photo in self.photos),
            'target': self.target,
        }

    def finish(self, path):
        """Save a named project, honestly distinguishing partial and finished work."""
        from datetime import datetime, timezone
        path = Path(path)
        source = Path(self.source).resolve()
        if path.absolute().is_relative_to(source) or path.resolve().is_relative_to(source):
            raise ValueError('Save the project file outside the photo folder.')
        summary = self.summary()
        summary['project_file'] = str(path.resolve())
        summary['saved_at'] = datetime.now(timezone.utc).isoformat()
        old_finished, old_summary = self.finished_at, self.finish_summary
        self.finished_at = summary['saved_at'] if summary['pending'] == 0 else None
        self.finish_summary = summary
        try:
            self.save(path)
        except BaseException:
            self.finished_at, self.finish_summary = old_finished, old_summary
            raise
        return summary

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding='utf-8')
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        data['photos'] = [Photo(**p) for p in data['photos']]
        return cls(**data)

    def path_for(self, photo):
        root = Path(self.source).resolve()
        path = (root / photo.relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Photo path is outside the selected folder.')
        return path

    def scan(self):
        root = Path(self.source).resolve(strict=True)
        if not root.is_dir():
            raise ValueError('Select a folder.')
        old = {p.relative: p for p in self.photos}
        photos = []
        for path in sorted(root.rglob('*') if self.recursive else root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED | RAW:
                continue
            if not path.resolve().is_relative_to(root):
                continue
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            previous = old.get(relative)
            if previous and (previous.size, previous.modified) == (stat.st_size, stat.st_mtime_ns):
                previous.recycled = False  # A photo restored from the Recycle Bin.
                photos.append(previous)
            else:
                photo = Photo(relative, stat.st_size, stat.st_mtime_ns)
                if path.suffix.lower() in RAW:
                    photo.status = 'Unsupported'
                    photo.reasons = ['RAW decoding is not supported in this release.']
                photos.append(photo)
        found = {p.relative for p in photos}
        photos.extend(p for p in old.values() if p.recycled and p.relative not in found)
        self.photos = photos

    def selection(self):
        if self.target is not None and (type(self.target) is not int or self.target < 1):
            raise ValueError('Target must be at least one.')
        eligible = [p for p in self.photos if p.decision == 'Keep' and p.status in ('Keep', 'Review')]
        pinned = sorted([p for p in eligible if p.must_include], key=lambda p: p.relative)
        if self.target is not None and len(pinned) > self.target:
            raise ValueError('Must-include count exceeds the target. Increase the target or unpin photos.')
        candidates = sorted([p for p in eligible if not p.must_include], key=lambda p: (-p.score, p.relative))
        if self.style == 'Balanced':
            seen = {p.group for p in pinned if p.group}
            first, remaining = [], []
            for photo in candidates:
                if photo.group and photo.group in seen:
                    remaining.append(photo)
                else:
                    first.append(photo)
                    seen.add(photo.group)
            candidates = first + remaining
        if self.target is None:
            return pinned + candidates
        return pinned + candidates[:max(0, self.target - len(pinned))]

    def cleanup_candidates(self):
        """Suggestions only; explicit approval is required separately."""
        selected = {photo.relative for photo in self.selection()}
        return [photo for photo in self.photos if not photo.recycled and not photo.must_include and (
            photo.override == 'Excluded' or photo.decision == 'Review' or
            self.target is not None and photo.decision == 'Keep' and photo.relative not in selected)]

    def checked_cleanup_path(self, photo):
        relative = Path(photo.relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Cleanup paths must be relative to the selected folder.')
        root = Path(self.source).resolve(strict=True)
        lexical = root / relative
        path = self.path_for(photo)
        if os.path.normcase(str(lexical)) != os.path.normcase(str(path)) or lexical.is_symlink():
            raise ValueError(f'Cleanup does not follow links or junctions: {photo.relative}')
        stat = path.stat()
        if not path.is_file() or (stat.st_size, stat.st_mtime_ns) != (photo.size, photo.modified):
            raise ValueError(f'Source changed; rescan before cleanup: {photo.relative}')
        return path

    def recycle(self, approved, project_file, trash=None, stop=None, progress=None):
        """Recycle only the exact explicitly marked list, with a durable audit trail."""
        from datetime import datetime, timezone
        from uuid import uuid4
        if not approved or len(set(approved)) != len(approved):
            raise ValueError('Select a nonempty list of distinct photos for removal.')
        if any(photo.status == 'Pending' and not photo.recycled for photo in self.photos):
            raise ValueError('Finish photo analysis before cleaning the folder.')
        root = Path(self.source).resolve(strict=True)
        if Path(project_file).absolute().is_relative_to(root) or Path(project_file).resolve().is_relative_to(root):
            raise ValueError('Save the project outside the source folder before cleanup.')
        indexed = {photo.relative: photo for photo in self.photos}
        photos = []
        for relative in approved:
            photo = indexed.get(relative)
            if not photo or photo.recycled or photo.override != 'Excluded' or photo.must_include:
                raise ValueError(f'Photo is not explicitly marked for removal: {relative}')
            self.checked_cleanup_path(photo)
            photos.append(photo)
        if trash is None:
            from .recycle import recycle_file
            trash = recycle_file
        operation = {'id': str(uuid4()), 'source': str(root),
                     'started': datetime.now(timezone.utc).isoformat(),
                     'files': [{'relative': p.relative, 'size': p.size, 'modified': p.modified,
                                'outcome': 'queued'} for p in photos]}
        self.cleanup_history.append(operation)
        self.save(project_file)  # Failure here must prevent all source changes.
        report = {'recycled': [], 'errors': {}, 'cancelled': False}
        for photo, record in zip(photos, operation['files']):
            if stop is not None and stop.is_set():
                report['cancelled'] = True
                break
            try:
                path = self.checked_cleanup_path(photo)
                record['outcome'] = 'moving'
                self.save(project_file)
                trash(path)
                if os.path.lexists(path):
                    raise OSError('The file is still present after the Recycle Bin operation.')
                photo.recycled = True
                record['outcome'] = 'recycled'
                report['recycled'].append(photo.relative)
            except Exception as exc:
                record.update(outcome='error', error=str(exc))
                report['errors'][photo.relative] = str(exc)
                break
            finally:
                self.save(project_file)
            if progress is not None:
                progress(len(report['recycled']), len(photos))
        for record in operation['files']:
            if record['outcome'] == 'queued':
                record['outcome'] = 'not_attempted'
        self.save(project_file)
        return report

    def analyze(self, pipeline, config, stop, progress, before_photo=None):
        """Called by a worker. Completed records can be saved after each callback."""
        for photo in self.photos:
            if stop.is_set():
                break
            if photo.recycled or photo.status != 'Pending':
                continue
            if before_photo is not None:
                before_photo(photo)
            try:
                path = self.path_for(photo)
                with Image.open(path) as original:
                    image = ImageOps.exif_transpose(original).convert('RGB')
                result = pipeline.run(image, config)
                if not math.isfinite(result.unified_score):
                    raise ValueError('Assessment returned an invalid score.')
                photo.score = float(result.unified_score)
                photo.status = result.classification
                photo.reasons = list(result.review_reasons)
                if photo.status == 'Review' and not photo.reasons:
                    photo.reasons = ['Quality score is below the Keep threshold.']
                # Coarse visual similarity heuristic, not identity or event recognition.
                small = image.convert('L').resize((8, 8))
                pixels = [small.getpixel((x, y)) for y in range(8) for x in range(8)]
                mean = sum(pixels) / len(pixels)
                photo.signature = ''.join('1' if p >= mean else '0' for p in pixels)
                photo.group = photo.relative
                for earlier in self.photos:
                    if earlier is photo:
                        break
                    if earlier.signature and sum(a != b for a, b in zip(earlier.signature, photo.signature)) <= 4:
                        photo.group = earlier.group
                        break
            except Exception as exc:
                photo.status = 'Error'
                photo.reasons = [str(exc)]
            progress(photo)

    def export(self, destination):
        """Create a new export folder exclusively; never overwrite or move originals."""
        selected = self.selection()
        if not selected:
            raise ValueError('There are no selected photos to export.')
        destination = Path(destination).resolve()
        source = Path(self.source).resolve()
        if destination == source or destination.is_relative_to(source):
            raise ValueError('Choose an export folder outside the source folder.')
        # Validate the complete selection before copying anything.
        for photo in selected:
            stat = self.path_for(photo).stat()
            if (stat.st_size, stat.st_mtime_ns) != (photo.size, photo.modified):
                raise ValueError(f'Source changed; rescan before export: {photo.relative}')
        destination.mkdir(parents=True, exist_ok=False)
        for photo in selected:
            target = destination / 'photos' / photo.relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with self.path_for(photo).open('rb') as reader, target.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
        (destination / '_selection.json').write_text(json.dumps({
            'requested': self.target, 'exported': len(selected),
            'photos': [asdict(p) for p in selected],
        }, indent=2), encoding='utf-8')
        return len(selected)
