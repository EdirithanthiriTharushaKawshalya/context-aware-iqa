"""Capture desktop previews using bundled artwork or an optional RAW sample."""
import shutil
import argparse
import sys
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.desktop.app import DesktopApp
from src.desktop.appearance import apply, preferences
from src.desktop.project import Project
from src.desktop.sessions import SessionStore
from src.desktop.startup import Splash
from src.desktop.welcome import Preferences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-sample', type=Path)
    parser.add_argument('--review-example', action='store_true')
    args = parser.parse_args()
    destination = Path('outputs/raw-verification/desktop-preview' if args.raw_sample else 'outputs/desktop-design-review')
    destination.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    splash = Splash(root)
    splash.attributes('-topmost', True)

    def capture(widget, name):
        widget.update_idletasks()
        if name.startswith('workspace-'):
            # Include the native caption; PrintWindow captures only Tk's client.
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.WinDLL('user32')
            user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            user32.GetAncestor.restype = wintypes.HWND
            hwnd = user32.GetAncestor(widget.winfo_id(), 2)
            bounds = wintypes.RECT()
            dwm = ctypes.WinDLL('dwmapi')
            dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(bounds), ctypes.sizeof(bounds))
            ImageGrab.grab(bbox=(bounds.left, bounds.top, bounds.right, bounds.bottom),
                           include_layered_windows=True).save(destination / f'{name}.png')
            return
        x, y = widget.winfo_rootx(), widget.winfo_rooty()
        ImageGrab.grab(bbox=(x, y, x+widget.winfo_width(), y+widget.winfo_height()),
                       include_layered_windows=True).save(destination / f'{name}.png')

    def workspace():
        capture(splash, 'splash')
        splash.destroy()
        app = DesktopApp(root, SessionStore(destination / 'sessions'), Preferences(destination / 'preferences.json'))
        source = destination / 'sample-photos'
        source.mkdir(exist_ok=True)
        for index in range(6):
            shutil.copyfile('src/images/splash-landscape.png', source / f'Dawn study {index+1:02d}.png')
        if args.raw_sample:
            shutil.copyfile(args.raw_sample, source/('Camera sample'+args.raw_sample.suffix))
        project = Project(str(source.resolve()))
        project.scan()
        for index, photo in enumerate(project.photos):
            photo.status, photo.score = ('Review', 48.) if index == 4 else ('Keep', 91.-index)
            if args.review_example and index == 4:
                photo.review_reasons = ['Possible motion blur; inspect the subject at 100%.',
                                        'Some subject detail is obscured by deep shadows.']
                photo.reasons = list(photo.review_reasons)
        app.activate(project, destination / 'preview.iqa.json')
        root.geometry(f'1280x{min(900, root.winfo_screenheight()-100)}+30+30')
        root.deiconify()
        root.attributes('-topmost', True)
        root.lift()
        selected = next((str(i) for i, photo in enumerate(project.photos)
                         if photo.relative.startswith('Camera sample')), '0')
        if args.review_example:
            selected = '4'
        app.select_gallery_photo(selected)

        def dark():
            capture(root, 'workspace-light')
            app.appearance.set('dark')
            apply(app, 'dark')
            root.after(400, show_preferences)

        def show_preferences():
            capture(root, 'workspace-dark')
            app.open_gallery_photo(selected)
            app.viewer.window.attributes('-topmost', True)
            root.after(100, viewer_fit)

        def viewer_fit():
            if app.viewer.displayed is None:
                root.after(50, viewer_fit)
                return
            capture(app.viewer.window, 'viewer-fit')
            app.viewer.set_mode('actual')
            root.after(100, viewer_actual)

        def viewer_actual():
            if app.viewer.displayed is None:
                root.after(50, viewer_actual)
                return
            capture(app.viewer.window, 'viewer-100')
            app.viewer.close()
            app.menu_bar.open(0)
            app.menu_bar.popup.attributes('-topmost', True)
            root.after(300, menu_preview)

        def menu_preview():
            capture(root, 'menu-dark')
            app.menu_bar.close()
            preferences(app)
            root.after(350, tour)

        def tour():
            capture(app.preferences_window, 'preferences-dark')
            app.preferences_window.destroy()
            app.introduction()
            app.tour.overlay.attributes('-topmost', True)
            app.tour.window.attributes('-topmost', True)
            root.after(350, lambda: next_card(0))

        def next_card(index):
            capture(root, f'tour-{index+1}')
            if index < 3:
                app.tour.next()
                root.after(300, lambda: next_card(index+1))
            else:
                app.tour.dismiss()
                root.after(50, root.destroy)

        def loaded():
            if app._preview_displayed is not None:
                dark()
            else:
                root.after(50, loaded)
        root.after(750, loaded)

    root.after(500, workspace)
    root.after(15000, root.destroy)
    root.mainloop()
    print(f'Previews saved to {destination.resolve()}')


if __name__ == '__main__':
    main()
