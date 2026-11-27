"""Headless verification of the bundled RAW decoder against an actual camera file."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from .duplicates import capture_metadata
from .image_io import is_raw, read_photo


def verify_raw(path, report_path):
    import rawpy
    path, report_path = Path(path).resolve(), Path(report_path).resolve()
    if not is_raw(path) or path == report_path:
        raise ValueError('Choose a RAW photograph and a separate JSON report path.')
    def digest():
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    before = digest()
    thumb = read_photo(path, (190, 128), thumbnail=True)
    fitted = read_photo(path, (640, 480))
    full = read_photo(path)
    after = digest()
    if before != after:
        raise RuntimeError('Source RAW changed during verification.')
    report = {'filename': path.name, 'rawpy': rawpy.__version__,
              'libraw': list(rawpy.libraw_version), 'thumbnail_size': list(thumb.size),
              'fitted_size': list(fitted.size), 'full_resolution_size': list(full.size),
              'exposure_seconds': float(full.getexif()[33434]) if 33434 in full.getexif() else None,
              'capture_metadata': capture_metadata(full.getexif()),
              'render_profile': full.info.get('raw_render_profile'),
              'source_sha256': before, 'source_unchanged': True}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-smoke-test', required=True, type=Path)
    parser.add_argument('--raw-report', type=Path,
                        default=Path(tempfile.gettempdir())/'context-aware-iqa-raw-check.json')
    args = parser.parse_args(argv)
    verify_raw(args.raw_smoke_test, args.raw_report)


if __name__ == '__main__':
    main()
