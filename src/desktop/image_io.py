"""Read-only raster/RAW decoding shared by assessment and desktop previews."""
from functools import lru_cache
from fractions import Fraction
from io import BytesIO
import math
import struct
from pathlib import Path
import threading

from PIL import Image, ImageOps
from PIL.TiffImagePlugin import IFDRational

SUPPORTED = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
RAW = {'.arw', '.sr2', '.srf', '.cr2', '.cr3', '.crw', '.nef', '.nrw',
       '.dng', '.orf', '.raf', '.rw2', '.rwl', '.pef', '.srw', '.mrw'}
RAW_RENDER_PROFILE = 'libraw-srgb-camera-wb-no-auto-bright-raw-focus-v2'
_RAW_RENDER_SLOT = threading.BoundedSemaphore(1)
_FLIP_ORIENTATION = {0: 1, 3: 3, 5: 8, 6: 6}


def is_raw(path):
    return Path(path).suffix.lower() in RAW


@lru_cache(maxsize=512)
def _raw_tags(path, size, modified):
    """Cache bounded metadata by source identity, without parsing MakerNotes."""
    import exifread
    try:
        with Path(path).open('rb') as handle:
            tags = exifread.process_file(handle, details=False, extract_thumbnail=False)
        # Built-in conversion rewrites EXIF timestamps and orientation into
        # display strings. Retain the source values required by burst grouping.
        selected = {}
        for name in ('Image Make', 'Image Model', 'Image Orientation',
                     'EXIF DateTimeOriginal', 'EXIF DateTimeDigitized',
                     'EXIF SubSecTimeOriginal', 'EXIF OffsetTimeOriginal',
                     'EXIF BodySerialNumber', 'EXIF ExposureTime'):
            tag = tags.get(name)
            if tag is not None:
                if name in ('Image Orientation', 'EXIF ExposureTime'):
                    values = tag.values
                    value = values[0] if isinstance(values, (list, tuple)) else values
                    try:
                        selected[name] = int(value) if name == 'Image Orientation' else float(value)
                    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                        continue
                else:
                    selected[name] = str(tag)
        return selected
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
            ZeroDivisionError, struct.error, NotImplementedError):
        return {}  # Missing metadata never becomes invented burst evidence.


def raw_exif(path):
    stat = Path(path).stat()
    tags = _raw_tags(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
    exif = Image.Exif()
    for number, name in ((271, 'Image Make'), (272, 'Image Model'),
                         (36867, 'EXIF DateTimeOriginal'), (36868, 'EXIF DateTimeDigitized'),
                         (37521, 'EXIF SubSecTimeOriginal'), (36881, 'EXIF OffsetTimeOriginal'),
                         (42033, 'EXIF BodySerialNumber')):
        value = tags.get(name)
        if value is not None:
            exif[number] = str(value)
    for number, name in ((274, 'Image Orientation'), (33434, 'EXIF ExposureTime')):
        value = tags.get(name)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        try:
            numeric = float(value)
            if math.isfinite(numeric) and numeric > 0:
                if number == 274:
                    if numeric.is_integer() and 1 <= numeric <= 8:
                        exif[number] = int(numeric)
                else:
                    exif[number] = IFDRational(Fraction(numeric).limit_denominator(1_000_000_000))
        except (TypeError, ValueError, OverflowError):
            pass
    return exif


def _oriented(image, orientation, exif):
    # LibRaw renders sensor orientation (user_flip=0); apply EXIF exactly once.
    exif[274] = orientation
    image.info['exif'] = exif.tobytes()
    return ImageOps.exif_transpose(image).convert('RGB')


def _decode_raw(path, size, thumbnail):
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError('RAW decoder is unavailable. Install the current desktop setup, or rawpy and ExifRead for source use.') from exc
    try:
        with _RAW_RENDER_SLOT, rawpy.RawPy() as raw:
            try:
                raw.open_file(str(path))
            except rawpy.LibRawIOError:
                # File objects also handle paths unsupported by a native filename API.
                with Path(path).open('rb') as handle:
                    raw.open_buffer(handle)
            exif = raw_exif(path)
            orientation = exif.get(274, _FLIP_ORIENTATION.get(raw.sizes.flip, 1))
            if thumbnail:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        with Image.open(BytesIO(thumb.data)) as embedded:
                            embedded_orientation = embedded.getexif().get(274)
                            image = embedded.convert('RGB')
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        image = Image.fromarray(thumb.data).convert('RGB')
                        embedded_orientation = None
                    else:
                        raise ValueError('No usable embedded preview')
                    if embedded_orientation in range(1, 9):
                        orientation = embedded_orientation
                    elif orientation in (5, 6, 7, 8):
                        # Some cameras store an already rotated JPEG without EXIF.
                        sensor_landscape = raw.sizes.width > raw.sizes.height
                        if (image.width > image.height) != sensor_landscape and image.width != image.height:
                            orientation = 1
                    image = _oriented(image, orientation, Image.Exif())
                    if size:
                        image.thumbnail(size, Image.Resampling.LANCZOS)
                    return image
                except (rawpy.LibRawError, OSError, ValueError):
                    pass  # Decode sensor data if the embedded preview is missing/damaged.
            raw.unpack()
            pixels = raw.postprocess(use_camera_wb=True, use_auto_wb=False,
                                    no_auto_bright=True, output_bps=8,
                                    output_color=rawpy.ColorSpace.sRGB, user_flip=0,
                                    half_size=size is not None, gamma=(2.222, 4.5))
            shutter = getattr(raw.other, 'shutter', 0.)
            if 33434 not in exif and shutter and math.isfinite(shutter) and shutter > 0:
                exif[33434] = IFDRational(Fraction(shutter).limit_denominator(1_000_000_000))
            image = _oriented(Image.fromarray(pixels), orientation, exif)
            image.info['raw_render_profile'] = RAW_RENDER_PROFILE
            if size:
                image.thumbnail(size, Image.Resampling.LANCZOS)
            return image
    except rawpy.LibRawFileUnsupportedError as exc:
        raise ValueError('This RAW camera/encoding is not supported by the bundled decoder, or the file is not a valid RAW photograph.') from exc
    except rawpy.LibRawError as exc:
        raise ValueError(f'Could not decode this RAW photograph: {exc}') from exc


def read_photo(path, size=None, *, thumbnail=False):
    """Return an independent, oriented RGB image; never write the source file.

    size=None always decodes full resolution, including RAW sensor data.
    Embedded camera JPEGs are used only for gallery thumbnails.
    """
    path = Path(path)
    if is_raw(path):
        return _decode_raw(path, size, thumbnail)
    with Image.open(path) as original:
        if size is not None:
            orientation = original.getexif().get(274, 1)
            original.draft('RGB', size[::-1] if orientation in (5, 6, 7, 8) else size)
        image = ImageOps.exif_transpose(original).convert('RGB')
        if size is not None:
            image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
