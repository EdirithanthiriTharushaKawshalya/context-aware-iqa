"""Show the branded splash before loading the desktop workspace."""
import tkinter as tk
import math
import time
import os
from tkinter import ttk
from pathlib import Path


def round_window(window, radius=36):
    """Clip the actual Windows window, including its otherwise square corners."""
    if os.name != 'nt':
        return
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HRGN, wintypes.BOOL]
    user32.SetWindowRgn.restype = ctypes.c_int
    gdi32.CreateRoundRectRgn.argtypes = [ctypes.c_int] * 6
    gdi32.CreateRoundRectRgn.restype = wintypes.HRGN
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    handle = user32.GetParent(window.winfo_id()) or window.winfo_id()
    region = gdi32.CreateRoundRectRgn(0, 0, window.winfo_width() + 1,
                                    window.winfo_height() + 1, radius * 2, radius * 2)
    if region and not user32.SetWindowRgn(handle, region, True):
        gdi32.DeleteObject(region)  # Windows owns the region after a successful call.


class Splash(tk.Toplevel):
    def __init__(self, root):
        super().__init__(root)
        self.overrideredirect(True)
        self.configure(background='#202127')
        width, height = min(900, root.winfo_screenwidth()-60), min(520, root.winfo_screenheight()-60)
        self.geometry(f'{width}x{height}+{max(0, (root.winfo_screenwidth()-width)//2)}+{max(0, (root.winfo_screenheight()-height)//2)}')
        from .soft_theme import install
        style = ttk.Style(root)
        style.theme_use('clam')
        install(root, style)
        from PIL import Image, ImageOps, ImageTk
        assets = Path(__file__).resolve().parents[1] / 'images'
        picture_width = int(width * .48)
        with Image.open(assets / 'splash-mirrored-dusk.png') as source:
            self.artwork = ImageTk.PhotoImage(ImageOps.fit(source, (picture_width, height)), master=self)
        tk.Label(self, image=self.artwork, borderwidth=0).pack(side='right', fill='y')
        content = tk.Frame(self, background='#202127')
        content.pack(side='left', fill='both', expand=True, padx=34, pady=34)
        with Image.open(assets / 'icon.png') as source:
            source.thumbnail((74, 56), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(source, master=self)
        tk.Label(content, image=self.logo, background='#202127').pack(anchor='w', pady=(0, 22))
        tk.Label(content, text='Context Aware\nIQA', justify='left', font=('Segoe UI', 28, 'bold'),
                 background='#202127', foreground='#f5f2fb').pack(anchor='w')
        tk.Label(content, text='Every frame. A considered choice.',
                 font=('Segoe UI', 11), background='#202127', foreground='#cbb8f3').pack(anchor='w', pady=(14, 16))
        tk.Label(content, text='Discover the photographs worth keeping.\nAssess detail, compare moments, and\nbuild a selection that feels like you.', justify='left',
                 font=('Segoe UI', 10), background='#202127', foreground='#b4adbf').pack(anchor='w')
        loading = tk.Frame(content, background='#202127')
        loading.pack(side='bottom', fill='x')
        self.status = tk.StringVar(value='Loading application components…')
        tk.Label(loading, textvariable=self.status, anchor='w', font=('Segoe UI', 10),
                 background='#202127', foreground='#ddd5e9', wraplength=330).pack(fill='x', pady=(0, 12))
        tk.Label(loading, text='CONTEXT AWARE IMAGE QUALITY ASSESSMENT',
                 font=('Segoe UI', 7), background='#202127', foreground='#827b90').pack(anchor='w')
        self.update()
        round_window(self)
        self.visible_since = time.monotonic()

    def stage(self, text, progress=None):
        self.status.set(text)
        self.update_idletasks()


def prepare_workspace(root, splash, steps, done, failed):
    """Present each real startup task before running it; keep Tk responsive."""
    def advance(index):
        if index == len(steps):
            if splash:
                splash.stage('Ready — opening your workspace…', 100)
            done()
            return
        message, progress, task = steps[index]
        if splash:
            splash.stage(message, progress)
        began = time.monotonic()
        def perform():
            try:
                task()
            except Exception as exc:
                failed(exc)
                return
            delay = max(0, math.ceil((0.65 - (time.monotonic() - began)) * 1000)) if splash else 0
            root.after(delay, lambda: advance(index + 1))
        root.after(20 if splash else 0, perform)
    initial = max(0, math.ceil((0.65 - (time.monotonic() - splash.visible_since)) * 1000)) if splash else 0
    root.after(initial, lambda: advance(0))


def show_workspace(root, splash, ready):
    """Keep the splash visible for five seconds without blocking Tk events."""
    def reveal():
        if splash:
            splash.destroy()
        root.deiconify()
        ready()
    remaining = max(0, 5 - (time.monotonic() - splash.visible_since)) if splash else 0
    root.after(math.ceil(remaining * 1000), reveal)


def launch(smoke_test=False):
    root = tk.Tk()
    root.withdraw()
    splash = None
    try:
        if not smoke_test:
            splash = Splash(root)
        from .app import main
        main(smoke_test=smoke_test, root=root, splash=splash)
    finally:
        try:
            if root.winfo_exists():
                root.destroy()
        except tk.TclError:
            pass
