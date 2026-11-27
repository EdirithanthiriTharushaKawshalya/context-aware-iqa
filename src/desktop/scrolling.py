"""Route wheel input by pointer location, including when a different control has focus."""
import tkinter as tk
from weakref import WeakSet


def register_gallery(gallery):
    host = gallery._root()
    tag = 'IQA.GalleryWheel'
    if not hasattr(host, '_wheel_galleries'):
        host._wheel_galleries = WeakSet()
        def wheel(event):
            try:
                widget = host.winfo_containing(event.x_root, event.y_root)
                while widget is not None:
                    if widget in host._wheel_galleries and widget.alive:
                        return widget.wheel(event)
                    if widget.winfo_class() in ('Text', 'Treeview'):
                        remainder = getattr(widget, '_wheel_remainder', 0)+event.delta
                        steps = int(remainder/120)
                        widget._wheel_remainder = remainder-steps*120
                        if steps:
                            widget.yview_scroll(-steps*3, 'units')
                        return 'break'
                    widget = widget.master
            except (tk.TclError, KeyError):
                pass
        host.bind_class(tag, '<MouseWheel>', wheel)
        def attach(widget):
            if isinstance(widget, tk.Misc) and tag not in widget.bindtags():
                widget.bindtags((tag, *widget.bindtags()))
        def mapped(event):
            attach(event.widget)
        host.bind_all('<Map>', mapped, add='+')
        def walk(widget):
            attach(widget)
            for child in widget.winfo_children():
                walk(child)
        walk(host)
    host._wheel_galleries.add(gallery)
