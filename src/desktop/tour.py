"""Modal workspace coaching with a translucent scrim and anchored targets."""
import tkinter as tk
import os
from tkinter import ttk


class Introduction:
    def __init__(self, root, preferences=None, targets=()):
        from .welcome import Preferences
        self.root, self.targets = root, targets
        self.preferences = preferences or Preferences()
        self.index = 0
        self.previous_focus = root.focus_get()
        self.overlay = tk.Toplevel(root)
        self.overlay.withdraw()
        self.overlay.overrideredirect(True)
        self.overlay.transient(root)
        self.overlay.configure(background='#101018')
        self.overlay.attributes('-alpha', .72)
        if os.name == 'nt':
            self.overlay.attributes('-transparentcolor', '#ff00ff')
        self.canvas = tk.Canvas(self.overlay, background='#101018', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.title('Workspace tour')
        self.window.overrideredirect(True)
        self.window.transient(root)
        self.window.resizable(False, False)
        self.window.protocol('WM_DELETE_WINDOW', self.dismiss)
        self.window.bind('<Escape>', lambda _: self.dismiss())
        self.window.bind('<Right>', lambda _: self.next())
        self.window.bind('<Left>', lambda _: self.back())
        self.overlay.bind('<Escape>', lambda _: self.dismiss())
        card = ttk.Frame(self.window, style='Card.TFrame', padding=26)
        card.pack(fill='both', expand=True)
        self.counter = ttk.Label(card, style='Muted.TLabel')
        self.counter.pack(anchor='w', pady=(0, 14))
        self.title = ttk.Label(card, style='Section.TLabel', font=('Segoe UI', 17, 'bold'), wraplength=368)
        self.title.pack(anchor='w')
        self.body = ttk.Label(card, style='Card.TLabel', wraplength=368, justify='left', font=('Segoe UI', 10))
        self.body.pack(anchor='w', pady=16)
        self.note = ttk.Label(card, style='Muted.TLabel', wraplength=368)
        self.note.pack(anchor='w')
        controls = ttk.Frame(card, style='Card.TFrame')
        controls.pack(side='bottom', fill='x', pady=(22, 0))
        ttk.Button(controls, text='Skip tour', command=self.dismiss, style='Card.TButton').pack(side='left')
        self.next_button = ttk.Button(controls, text='Next', style='Card.Primary.TButton', command=self.next)
        self.next_button.pack(side='right')
        self.back_button = ttk.Button(controls, text='Back', command=self.back, style='Card.TButton')
        self.back_button.pack(side='right', padx=6)
        self.resize_binding = root.bind('<Configure>', self.resize, add='+')
        self.window.bind('<Destroy>', self.destroyed, add='+')
        self.render()
        self.window.grab_set()
        self.next_button.focus_set()

    def resize(self, event):
        if event.widget == self.root and self.window.winfo_exists():
            self.position()

    def position(self):
        root = self.root
        width, height = max(460, root.winfo_width()), max(360, root.winfo_height())
        rx, ry = root.winfo_rootx(), root.winfo_rooty()
        self.overlay.geometry(f'{width}x{height}+{rx}+{ry}')
        self.canvas.delete('all')
        cw, ch = min(440, width-24), max(330, self.window.winfo_reqheight())
        x, y = (width-cw)//2, (height-ch)//2
        if self.targets and self.index < len(self.targets):
            target = self.targets[self.index]
            tx, ty = target.winfo_rootx()-rx, target.winfo_rooty()-ry
            tw, th = target.winfo_width(), target.winfo_height()
            self.canvas.create_rectangle(tx-4, ty-4, tx+tw+4, ty+th+4,
                                         fill='#ff00ff' if os.name == 'nt' else '', outline='#d7c4ff', width=4)
            self.canvas.create_text(tx+12, max(12, ty-16), text=f'{self.index+1:02d}',
                                    fill='#ffffff', font=('Segoe UI', 11, 'bold'), anchor='w')
            x = tx if self.index != 2 else tx+tw+18
            y = ty+th+18 if self.index < 2 else ty
            if y+ch > height-12:
                y = ty-ch-18
        x, y = max(12, min(x, width-cw-12)), max(12, min(y, height-ch-12))
        self.window.geometry(f'{cw}x{ch}+{rx+x}+{ry+y}')
        self.overlay.deiconify()
        self.window.deiconify()
        self.overlay.lift()
        self.window.lift()

    def render(self):
        from .welcome import CARDS
        title, body, note = CARDS[self.index]
        self.counter.configure(text=f'WORKSPACE TOUR    {self.index+1:02d} / {len(CARDS):02d}')
        self.title.configure(text=title)
        self.body.configure(text=body)
        self.note.configure(text=note)
        self.back_button.configure(state='disabled' if self.index == 0 else 'normal')
        self.next_button.configure(text='Get started' if self.index == len(CARDS)-1 else 'Next')
        self.window.update_idletasks()
        self.position()

    def next(self):
        from .welcome import CARDS
        if self.index == len(CARDS)-1:
            return self.dismiss()
        self.index += 1
        self.render()

    def back(self):
        self.index = max(0, self.index-1)
        self.render()

    def destroyed(self, event):
        if event.widget == self.window:
            if self.resize_binding:
                self.root.unbind('<Configure>', self.resize_binding)
                self.resize_binding = None
            if self.overlay.winfo_exists():
                self.overlay.destroy()

    def dismiss(self):
        try:
            self.preferences.complete()
        except OSError:
            pass
        self.window.grab_release()
        self.window.destroy()
        if self.previous_focus and self.previous_focus.winfo_exists():
            self.previous_focus.focus_set()
