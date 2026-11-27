"""A fit-first photo viewer with an explicit, full-resolution 100% mode."""
import tkinter as tk
from tkinter import ttk

from PIL import ImageTk

from .previews import PreviewLoader


class PhotoViewer:
    def __init__(self, root, path, photo, parent=None, fit=True):
        self.path, self.photo, self.parent = path, photo, parent
        self.mode = 'fit' if fit else 'actual'
        self.requested = None
        self.displayed = None
        self.resize_timer = None
        self.window = tk.Toplevel(parent or root)
        self.window.title(photo.relative)
        width = min(1050, root.winfo_screenwidth()-80)
        height = min(760, root.winfo_screenheight()-100)
        self.window.geometry(f'{width}x{height}')
        self.window.minsize(480, 360)
        self.window.rowconfigure(1, weight=1)
        self.window.columnconfigure(0, weight=1)
        if parent:
            self.window.transient(parent)
            self.window.grab_set()
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.bind('<Escape>', lambda _: self.close())
        toolbar = ttk.Frame(self.window, padding=(12, 7))
        toolbar.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.fit_button = ttk.Button(toolbar, text='Fit to window', command=lambda: self.set_mode('fit'))
        self.fit_button.pack(side='left')
        self.actual_button = ttk.Button(toolbar, text='100%', command=lambda: self.set_mode('actual'))
        self.actual_button.pack(side='left', padx=6)
        self.status = tk.StringVar(master=self.window)
        ttk.Label(toolbar, textvariable=self.status).pack(side='right')
        self.canvas = tk.Canvas(self.window, background='#202127', highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky='nsew')
        self.horizontal = ttk.Scrollbar(self.window, orient='horizontal', command=self.canvas.xview)
        self.vertical = ttk.Scrollbar(self.window, orient='vertical', command=self.canvas.yview)
        self.horizontal.grid(row=2, column=0, sticky='ew')
        self.vertical.grid(row=1, column=1, sticky='ns')
        self.canvas.configure(xscrollcommand=self.horizontal.set, yscrollcommand=self.vertical.set,
                              xscrollincrement=24, yscrollincrement=24)
        self.loader = PreviewLoader(self.window)
        self.window.preview_loader = self.loader
        self.canvas.bind('<Configure>', self.resized)
        self.window.bind('<MouseWheel>', self.wheel)
        self.window.bind('<Destroy>', self.destroyed, add='+')
        self.set_mode(self.mode)

    def set_mode(self, mode):
        self.mode = mode
        self.requested = self.displayed = None
        if mode == 'fit':
            self.horizontal.grid_remove()
            self.vertical.grid_remove()
        else:
            self.horizontal.grid()
            self.vertical.grid()
        self.status.set('Fit to window' if mode == 'fit' else '100% · scroll to inspect')
        self.fit_button.state(['pressed'] if mode == 'fit' else ['!pressed'])
        self.actual_button.state(['pressed'] if mode == 'actual' else ['!pressed'])
        self.resized()

    def resized(self, _=None):
        if self.mode == 'actual' and self.requested:
            return
        if self.resize_timer:
            self.window.after_cancel(self.resize_timer)
        self.resize_timer = self.window.after(70, self.load)

    def load(self):
        self.resize_timer = None
        size = (max(32, self.canvas.winfo_width()-24), max(32, self.canvas.winfo_height()-24)) if self.mode == 'fit' else None
        key = (str(self.path), self.photo.size, self.photo.modified, size)
        if self.displayed == key:
            return
        self.requested = key
        self.canvas.delete('all')
        self.canvas.photo = None
        self.canvas.create_text(24, 24, anchor='nw', text='Loading photo…', fill='#b4adbf')
        self.loader.request(key, self.path, size, self.loaded)

    def loaded(self, key, image, error):
        if key != self.requested or not self.canvas.winfo_exists():
            return
        self.canvas.delete('all')
        if error:
            self.canvas.create_text(24, 24, anchor='nw', text=f'Preview unavailable: {error}', fill='#b4adbf')
            return
        self.displayed = key
        self.canvas.photo = ImageTk.PhotoImage(image, master=self.window)
        if self.mode == 'fit':
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
            self.canvas.configure(scrollregion=(0, 0, width, height))
            self.canvas.create_image(width//2, height//2, image=self.canvas.photo)
        else:
            self.canvas.configure(scrollregion=(0, 0, image.width, image.height))
            self.canvas.create_image(0, 0, anchor='nw', image=self.canvas.photo)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def wheel(self, event):
        if self.mode == 'actual':
            steps = -int(event.delta/120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1)
            if event.state & 1:
                self.canvas.xview_scroll(steps*3, 'units')
            else:
                self.canvas.yview_scroll(steps*3, 'units')
        return 'break'

    def destroyed(self, event):
        if event.widget == self.window:
            if self.resize_timer:
                self.window.after_cancel(self.resize_timer)
                self.resize_timer = None
            self.loader.close()

    def close(self):
        self.window.destroy()
        if self.parent and self.parent.winfo_exists():
            self.parent.grab_set()
