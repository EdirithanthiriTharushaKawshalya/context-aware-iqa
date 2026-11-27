"""Latest-request image decoding; workers never touch Tk objects."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import queue

from PIL import Image, ImageOps


def read_preview(path, size=None):
    with Image.open(path) as original:
        if size is not None:
            # JPEG decoder reduction is for display only; assessment stays full-size.
            orientation = original.getexif().get(274, 1)
            draft_size = size[::-1] if orientation in (5, 6, 7, 8) else size
            original.draft('RGB', draft_size)
        image = ImageOps.exif_transpose(original).convert('RGB')
        if size is not None:
            image.thumbnail(size, Image.Resampling.LANCZOS)
        return image


class PreviewLoader:
    def __init__(self, root, decoder=read_preview, max_pixels=8_000_000):
        self.root, self.decoder = root, decoder
        self.max_pixels = max_pixels
        self.cache = OrderedDict()
        self.cache_pixels = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='photo-preview')
        self.results = queue.Queue()
        self.pending = {}
        self.desired = None
        self.callback = None
        self.timer = None
        self.closed = False

    def request(self, key, path, size, callback):
        self.desired, self.callback = key, callback
        for obsolete, future in list(self.pending.items()):
            if obsolete != key and future.cancel():
                del self.pending[obsolete]
        if key in self.cache:
            self.cache.move_to_end(key)
            callback(key, self.cache[key], None)
            return
        if key not in self.pending:
            future = self.executor.submit(self.decoder, path, size)
            self.pending[key] = future
            results = self.results
            def finished(task):
                if task.cancelled():
                    return
                try:
                    results.put((key, task.result(), None))
                except Exception as exc:
                    results.put((key, None, str(exc)))
            future.add_done_callback(finished)
        if self.timer is None:
            self.timer = self.root.after(20, self.poll)

    def cancel(self):
        self.desired = self.callback = None
        for key, future in list(self.pending.items()):
            if future.cancel():
                del self.pending[key]

    def poll(self):
        self.timer = None
        if self.closed:
            return
        while True:
            try:
                key, image, error = self.results.get_nowait()
            except queue.Empty:
                break
            self.pending.pop(key, None)
            if image is not None and image.width*image.height <= self.max_pixels:
                self.cache[key] = image
                self.cache_pixels += image.width*image.height
                while len(self.cache) > 10 or self.cache_pixels > self.max_pixels:
                    _, old = self.cache.popitem(last=False)
                    self.cache_pixels -= old.width*old.height
            if key == self.desired and self.callback:
                self.callback(key, image, error)
        if self.pending:
            self.timer = self.root.after(20, self.poll)

    def close(self):
        self.closed = True
        self.cancel()
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        self.cache.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)
