"""Live appearance and coaching must preserve the active photo review."""
import json
import tkinter as tk
from types import SimpleNamespace
from tkinter import ttk

from src.desktop.app import DesktopApp
from src.desktop.appearance import apply, preferences
from src.desktop.project import Photo, Project
from src.desktop.welcome import Preferences


def children(widget):
    for child in widget.winfo_children():
        yield child
        yield from children(child)


def dispose(root):
    for timer in root.tk.splitlist(root.tk.call('after', 'info')):
        root.after_cancel(timer)
    root.destroy()


def test_theme_preferences_survive_restart_and_preserve_tour(tmp_path):
    prefs = Preferences(tmp_path / 'settings.json')
    prefs.complete()
    prefs.save(theme='dark')
    assert Preferences(prefs.path).theme() == 'dark'
    assert Preferences(prefs.path).completed()
    prefs.path.write_text(json.dumps({'theme': 'invalid'}))
    assert prefs.theme() == 'light'


def test_live_switch_restores_colors_and_keeps_project(desktop_tk, tmp_path):
    root = tk.Toplevel(desktop_tk)
    prefs = Preferences(tmp_path / 'settings.json')
    app = DesktopApp(root, preferences=prefs)
    try:
        app.project = Project(str(tmp_path))
        active = app.project
        app.target.set('12')
        style = ttk.Style(root)
        initial = app.preview_label.cget('background')
        apply(app, 'dark')
        assert app.preview_label.cget('background') == '#202127'
        assert style.lookup('Primary.TButton', 'foreground', ('!disabled',)) == 'white'
        assert 'SoftDark' in str(style.layout('Rounded.TFrame'))
        apply(app, 'light')
        assert app.preview_label.cget('background') == initial
        assert app.project is active and app.target.get() == '12'
        preferences(app)
        radios = [w for w in children(app.preferences_window) if isinstance(w, ttk.Radiobutton)]
        radios[1].invoke()
        assert prefs.theme() == 'dark' and app.appearance.get() == 'dark'
        radios[0].invoke()
        assert prefs.theme() == 'light'
    finally:
        dispose(root)


def test_menu_guards_and_tour_follow_actual_controls(desktop_tk, tmp_path):
    root = tk.Toplevel(desktop_tk)
    app = DesktopApp(root, preferences=Preferences(tmp_path / 'settings.json'))
    try:
        root.geometry('1120x760')
        root.update()
        labels = [label for label, _ in app.menu_bar.sections]
        assert labels == ['File', 'Edit', 'Photo', 'View', 'Review', 'Help']
        for menu, index, requirement in app.menu_actions:
            if requirement in ('ready', 'photo', 'finish', 'busy'):
                assert menu.entrycget(index, 'state') == 'disabled'
        app.introduction()
        tour = app.tour
        assert tour.targets[0] is app.tour_targets[0]
        for step in range(4):
            root.update()
            assert tour.index == step
            assert tour.window.winfo_rootx() >= root.winfo_rootx()
            assert tour.window.winfo_rooty() >= root.winfo_rooty()
            assert tour.window.winfo_rooty()+tour.window.winfo_height() <= root.winfo_rooty()+root.winfo_height()
            if step < 3:
                tour.next()
        tour.dismiss()
        assert not tour.overlay.winfo_exists()
        assert app.user_preferences.completed()
    finally:
        dispose(root)


def test_themed_menu_navigation_guards_and_dismissal(desktop_tk, tmp_path):
    root = tk.Toplevel(desktop_tk)
    app = DesktopApp(root, preferences=Preferences(tmp_path / 'settings.json'))
    try:
        root.update()
        bar = app.menu_bar
        calls = []
        app.project_menu.entryconfigure(0, command=lambda: calls.append('open'))
        apply(app, 'dark')
        bar.open(0)
        root.update()
        assert bar.popup.cget('background') == bar.palette['border']
        bar.invoke(4)  # Saving is unavailable without a project.
        assert bar.popup is not None and not calls
        bar.current = 2
        bar.move(1)
        assert bar.current == 7  # Skip disabled commands and separators.
        bar.invoke(0)
        assert calls == ['open'] and bar.popup is None
        bar.open(2)
        assert bar.current is None  # No photo selected: every Photo action disabled.
        bar.move(1)
        bar.outside(SimpleNamespace(x_root=root.winfo_rootx()+root.winfo_width()-4,
                                    y_root=root.winfo_rooty()+root.winfo_height()-4))
        assert bar.popup is None
        apply(app, 'light')
        assert bar.cget('background') == '#f4f3f8'
    finally:
        dispose(root)


def test_compact_scrollbar_can_drag_the_photo_canvas(desktop_tk, tmp_path):
    root = tk.Toplevel(desktop_tk)
    app = DesktopApp(root, preferences=Preferences(tmp_path / 'settings.json'))
    try:
        root.update()
        canvas = app.gallery.canvas
        project = Project(str(tmp_path), photos=[Photo(f'photo-{i}.jpg', 1, 1) for i in range(30)])
        app.gallery.set_photos(project, range(30))
        root.update_idletasks()
        scrollbar = next(w for w in app.gallery.winfo_children() if isinstance(w, ttk.Scrollbar))
        x = scrollbar.winfo_width()//2
        thumb = [y for y in range(scrollbar.winfo_height()) if scrollbar.identify(x, y).endswith('thumb')]
        assert thumb
        y = thumb[len(thumb)//2]
        scrollbar.event_generate('<ButtonPress-1>', x=x, y=y)
        scrollbar.event_generate('<B1-Motion>', x=x, y=y+50)
        scrollbar.event_generate('<ButtonRelease-1>', x=x, y=y+50)
        assert canvas.yview()[0] > 0
    finally:
        dispose(root)
