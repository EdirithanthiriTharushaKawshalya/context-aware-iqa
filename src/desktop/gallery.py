"""Virtual thumbnail contact sheet. Image decoding stays off the Tk thread."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageOps, ImageTk


def read_thumbnail(path):
    with Image.open(path) as original:
        original.draft('RGB', (420, 320))
        image = ImageOps.exif_transpose(original).convert('RGB')
        image.thumbnail((190, 128), Image.Resampling.LANCZOS)
        return image


class PhotoGallery(ttk.Frame):
    CARD_WIDTH, CARD_HEIGHT, GAP = 210, 224, 12

    def __init__(self, parent, on_select=None, on_toggle=None, on_open=None):
        super().__init__(parent, style='Card.TFrame')
        self.on_select, self.on_toggle, self.on_open = on_select, on_toggle, on_open
        self.canvas = tk.Canvas(self, background='#f7f6fa', highlightthickness=0, takefocus=True)
        scroll = ttk.Scrollbar(self, orient='vertical', command=self.scroll)
        scroll.pack(side='right', fill='y')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.bind('<Configure>', lambda _: self.render())
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<Button-4>', lambda _: self.scroll(-3, 'units'))
        self.canvas.bind('<Button-5>', lambda _: self.scroll(3, 'units'))
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<Double-Button-1>', self.open_current)
        for key, offset in (('Left', -1), ('Right', 1), ('Up', 'up'), ('Down', 'down')):
            self.canvas.bind('<'+key+'>', lambda _, step=offset: self.navigate(step))
        self.canvas.bind('<space>', self.toggle_current)
        self.canvas.bind('<Return>', self.open_current)
        self.bind('<Destroy>', self.close, add='+')
        self.rows, self.chosen, self.selected = [], set(), None
        self.source = ''
        self.editable = False
        self.cache = OrderedDict()
        self.pending = set()
        self.results = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='photo-thumbnail')
        self.alive = True
        self.timer_host = self.winfo_toplevel()
        self.timer = self.timer_host.after(80, self.poll)
        self.signature = None
        self.photo_images = []
        self.visible_indices = []

    def set_photos(self, project, indices, chosen=(), editable=False, selected=None):
        rows = [(str(i), project.photos[int(i)]) for i in indices] if project else []
        signature = (project.source if project else '', tuple((i, p.relative, p.size, p.modified,
                     p.decision, p.status, p.must_include, p.style_label, p.duplicate_label) for i, p in rows),
                     frozenset(chosen), editable, selected)
        if signature == self.signature:
            return
        reset = self.source != (project.source if project else '') or [i for i, _ in rows] != [i for i, _ in self.rows]
        self.signature = signature
        self.source = project.source if project else ''
        self.rows, self.chosen, self.editable, self.selected = rows, set(chosen), editable, selected
        if reset:
            self.canvas.yview_moveto(0)
        self.render()

    def columns(self):
        return max(1, self.canvas.winfo_width() // (self.CARD_WIDTH+self.GAP))

    def key(self, photo):
        return self.source, photo.relative, photo.size, photo.modified

    def scroll(self, *args):
        self.canvas.yview(*args)
        self.render()

    def wheel(self, event):
        self.scroll(-int(event.delta/120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1), 'units')
        return 'break'

    def request(self, photo):
        key = self.key(photo)
        if key in self.pending or len(self.pending) >= 24 or photo.recycled:
            return
        # Never decode a path escaping the selected folder.
        root = Path(self.source).resolve()
        path = (root/photo.relative).resolve()
        if not path.is_relative_to(root):
            self.cache[key] = None
            return
        self.pending.add(key)
        future = self.executor.submit(read_thumbnail, path)
        results = self.results
        def finished(task):
            try:
                result = task.result()
            except Exception:
                result = None
            results.put((key, result))
        future.add_done_callback(finished)

    def render(self):
        if not self.alive:
            return
        canvas = self.canvas
        canvas.delete('all')
        self.photo_images = []
        columns = self.columns()
        height = ((len(self.rows)+columns-1)//columns)*(self.CARD_HEIGHT+self.GAP)+self.GAP
        canvas.configure(scrollregion=(0, 0, max(1, canvas.winfo_width()-2), max(height, canvas.winfo_height())))
        if not self.rows:
            self.visible_indices = []
            canvas.create_text(max(110, canvas.winfo_width()/2), 70, text='No photos in this view.\nChoose another filter or open a photo folder.',
                               fill='#827b90', font=('Segoe UI', 11), justify='center', width=300)
            return
        top = max(0, int(canvas.canvasy(0))//(self.CARD_HEIGHT+self.GAP)-1)
        bottom = int(canvas.canvasy(canvas.winfo_height()))//(self.CARD_HEIGHT+self.GAP)+2
        self.visible_indices = list(range(top*columns, min(len(self.rows), bottom*columns)))
        for position in self.visible_indices:
            iid, photo = self.rows[position]
            x, y = self.GAP+(position % columns)*(self.CARD_WIDTH+self.GAP), self.GAP+(position//columns)*(self.CARD_HEIGHT+self.GAP)
            tags = ('photo:'+iid,)
            chosen = photo.relative in self.chosen
            canvas.create_rectangle(x, y, x+self.CARD_WIDTH, y+self.CARD_HEIGHT,
                                    fill='white', outline='#7659d6' if iid == self.selected else '#ddd8e8',
                                    width=2 if iid == self.selected else 1, tags=tags)
            key = self.key(photo)
            if key in self.cache:
                image = self.cache[key]
                self.cache.move_to_end(key)
                if image is not None:
                    tk_image = ImageTk.PhotoImage(image, master=canvas)
                    self.photo_images.append(tk_image)
                    canvas.create_image(x+self.CARD_WIDTH/2, y+72, image=tk_image, tags=tags)
                else:
                    canvas.create_text(x+105, y+65, text='Preview unavailable', fill='#91879f', tags=tags)
            else:
                canvas.create_text(x+105, y+65, text='Recycled' if photo.recycled else 'Loading preview…', fill='#91879f', tags=tags)
                self.request(photo)
            name = Path(photo.relative).name
            canvas.create_text(x+10, y+146, text=name if len(name) <= 29 else name[:26]+'…',
                               anchor='w', fill='#302442', font=('Segoe UI', 9, 'bold'), tags=tags)
            label = 'Marked for removal' if photo.decision == 'Excluded' else photo.decision
            if photo.override and photo.decision != photo.status and not photo.recycled:
                label += ' · AI '+photo.status
            color = '#24765b' if photo.decision == 'Keep' else '#926528' if photo.decision == 'Review' else '#837b91'
            canvas.create_text(x+10, y+166, text=label, anchor='w', fill=color, font=('Segoe UI', 9), tags=tags)
            if photo.duplicate_label:
                badge = photo.duplicate_label + (' · protected' if photo.must_include else '')
                canvas.create_text(x+10,y+186,text=badge,anchor='w',fill='#24765b' if photo.duplicate_best else '#837b91',
                                   font=('Segoe UI',9),tags=tags)
            elif photo.must_include:
                canvas.create_text(x+10, y+186, text='Protected', anchor='w', fill='#7659d6', font=('Segoe UI', 9), tags=tags)
            elif photo.style_label != 'Unknown':
                canvas.create_text(x+10, y+186, text=photo.style_label, anchor='w', fill='#837b91', font=('Segoe UI', 9), tags=tags)
            toggle_tags = tags+('toggle:'+iid,)
            canvas.create_rectangle(x+7, y+199, x+203, y+219, fill='#e9f4ef' if chosen else '#f3effa',
                                    outline='', tags=toggle_tags)
            text = '✓ Chosen' if chosen else '+ Add to chosen' if self.editable and self.on_toggle else 'Not chosen'
            canvas.create_text(x+105, y+209, text=text, fill='#24765b' if chosen else '#7659d6',
                               font=('Segoe UI', 9, 'bold'), tags=toggle_tags)

    def poll(self):
        changed = False
        try:
            while True:
                key, image = self.results.get_nowait()
                self.pending.discard(key)
                self.cache[key] = image
                while len(self.cache) > 120:
                    self.cache.popitem(last=False)
                changed = True
        except queue.Empty:
            pass
        if self.alive:
            if changed:
                self.render()
            self.timer = self.timer_host.after(80, self.poll)

    def click(self, event):
        self.canvas.focus_set()
        item = self.canvas.find_withtag('current')
        tags = self.canvas.gettags(item[0]) if item else ()
        iid = next((tag[6:] for tag in tags if tag.startswith('photo:')), None)
        if iid is None:
            return
        self.selected = iid
        if self.editable and self.on_toggle and 'toggle:'+iid in tags:
            self.on_toggle(iid)
        elif self.on_select:
            self.on_select(iid)
        self.render()

    def navigate(self, offset):
        ids = [i for i, _ in self.rows]
        if not ids:
            return 'break'
        step = -self.columns() if offset == 'up' else self.columns() if offset == 'down' else offset
        index = max(0, min(len(ids)-1, ids.index(self.selected)+step)) if self.selected in ids else 0
        self.selected = ids[index]
        y = (index//self.columns())*(self.CARD_HEIGHT+self.GAP)
        total = max(1, ((len(ids)+self.columns()-1)//self.columns())*(self.CARD_HEIGHT+self.GAP))
        if y < self.canvas.canvasy(0) or y+self.CARD_HEIGHT > self.canvas.canvasy(self.canvas.winfo_height()):
            self.canvas.yview_moveto(y/total)
        if self.on_select:
            self.on_select(self.selected)
        self.render()
        return 'break'

    def toggle_current(self, _=None):
        if self.editable and self.on_toggle and self.selected is not None:
            self.on_toggle(self.selected)
        return 'break'

    def open_current(self, _=None):
        if self.selected is not None and self.on_open:
            self.on_open(self.selected)
        return 'break'

    def close(self, event):
        if event.widget == self and self.alive:
            self.alive = False
            self.timer_host.after_cancel(self.timer)
            # Break callbacks that capture the owning app/dialog and Tk variables.
            # Otherwise a later background-worker GC can finalize them off-thread.
            self.on_select = self.on_toggle = self.on_open = None
            self.rows = []
            self.photo_images.clear()
            self.cache.clear()
            self.executor.shutdown(wait=False, cancel_futures=True)
