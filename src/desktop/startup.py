"""Show the branded splash before loading the desktop workspace."""
import tkinter as tk
import math
import time
from tkinter import ttk
from pathlib import Path


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
            splash = tk.Toplevel(root)
            splash.overrideredirect(True)
            splash.configure(background='#f4f3f8')
            from .soft_theme import install
            style = ttk.Style(root)
            style.theme_use('clam')
            install(root, style)
            width, height = 500, 300
            splash.geometry(f'{width}x{height}+{max(0, (root.winfo_screenwidth()-width)//2)}+{max(0, (root.winfo_screenheight()-height)//2)}')
            card = ttk.Frame(splash, style='Rounded.TFrame', padding=18)
            card.pack(fill='both', expand=True, padx=8, pady=8)
            image = tk.PhotoImage(file=str(Path(__file__).resolve().parents[1] / 'images/icon.png'))
            splash.logo = image.subsample(6)
            tk.Label(card, image=splash.logo, background='#ffffff').pack(pady=(10, 12))
            tk.Label(card, text='Context Aware IQA', font=('Segoe UI', 22, 'bold'), background='#ffffff', foreground='#302442').pack()
            tk.Label(card, text='Preparing your photo workspace…', font=('Segoe UI', 10), background='#ffffff', foreground='#8b75ab').pack(pady=12)
            splash.update()
            splash.visible_since = time.monotonic()
        from .app import main
        main(smoke_test=smoke_test, root=root, splash=splash)
    finally:
        try:
            if root.winfo_exists():
                root.destroy()
        except tk.TclError:
            pass
