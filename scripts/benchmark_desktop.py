"""Repeatable synthetic desktop timings; no personal photos or ML inference."""
import argparse
import json
import statistics
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.desktop.app import DesktopApp
from src.desktop.project import Photo, Project
from src.desktop.welcome import Preferences


def measure(task, repeats=5):
    values = []
    for _ in range(repeats):
        began = time.perf_counter()
        task()
        values.append((time.perf_counter()-began)*1000)
    return round(statistics.median(values), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='iqa-performance-') as directory:
        folder = Path(directory)
        source = folder/'photos'
        source.mkdir()
        root = tk.Tk()
        root.withdraw()
        app = DesktopApp(root, preferences=Preferences(folder/'preferences.json'))
        root.geometry('1280x900')
        root.deiconify()
        root.update()
        app.gallery.request = lambda photo: None
        photos = [Photo(f'frame-{i:05d}.jpg', 100, i, status='Keep', score=90-i/1000,
                        selection_choice='include' if i < 1000 else '',
                        duplicate_features={'version': 1, 'gray': 'A'*1368, 'color': 'B'*1024}) for i in range(2000)]
        project = Project(str(source), photos=photos, one_per_burst=False)
        app.activate(project, folder/'recovery.json')
        root.update()
        selection_ms = measure(project.selection)
        save_ms = measure(lambda: project.save(folder/'recovery.json'))
        refresh_ms = measure(app.refresh)
        rows = [Photo(f'thumb-{i}.jpg', 10, i, status='Keep') for i in range(60)]
        gallery_project = Project(str(source), photos=rows)
        image = Image.new('RGB', (190, 128), '#8877aa')
        for row in rows:
            app.gallery.cache[(str(source), row.relative, row.size, row.modified)] = image
        app.gallery.set_photos(gallery_project, range(60))
        root.update_idletasks()
        render_ms = measure(app.gallery.render, 20)
        def scroll():
            app.gallery.scroll('moveto', .4)
            app.gallery.scroll('moveto', 0)
        scroll_ms = measure(scroll, 10)
        report = dict(collection_photos=2000, explicit_choices=1000,
                      timings_ms=dict(selection=selection_ms, recovery_save=save_ms,
                                      unchanged_refresh=refresh_ms, unchanged_gallery_render=render_ms,
                                      gallery_scroll_roundtrip=scroll_ms),
                      recovery_file_bytes=(folder/'recovery.json').stat().st_size,
                      note='Median synthetic timings on this machine; not an inference or real-photo speed claim.')
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        root.destroy()


if __name__ == '__main__':
    main()
