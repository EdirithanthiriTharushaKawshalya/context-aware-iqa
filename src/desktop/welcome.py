"""First-run guidance with explicit dismissal and a replayable tour."""
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk

CARDS = (
    ('Your photos, thoughtfully selected',
     'Welcome to Context Aware IQA. Turn a full event folder into a considered selection, while keeping every decision in your hands.',
     'Choose a folder  →  Analyze  →  Review  →  Finish'),
    ('Start with your photo folder',
     'Choose the folder you copied from your camera. Include subfolders if needed. Leave the final count blank for all suitable photos, or enter a number for a smaller selection.',
     'Your originals stay in their existing location.'),
    ('Let the assessment guide you',
     'Analyze sorts photos into Keep and Review. Select a photo to inspect its assessment. You can change a decision, protect an essential moment, or pause and resume analysis.',
     'AI proposes. You make the final decision.'),
    ('Clean up, then finish',
     'Mark unwanted photos for removal and open Review cleanup. Only the photos you confirm go to the Recycle Bin. Finish project saves your decisions to a project file you can reopen later.',
     'Automatic recovery protects work between saves.'),
)


class Preferences:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share')) / 'ContextAwareIQA/preferences.json'

    def completed(self):
        try:
            return json.loads(self.path.read_text(encoding='utf-8')).get('introduction_complete') is True
        except (OSError, ValueError, AttributeError):
            return False

    def complete(self):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data['introduction_complete'] = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temporary.replace(self.path)


class Introduction:
    def __init__(self, root, preferences=None):
        self.preferences = preferences or Preferences()
        self.index = 0
        self.window = tk.Toplevel(root)
        self.window.title('Welcome to Context Aware IQA')
        self.window.configure(background='#f4f3f8')
        self.window.resizable(False, False)
        width, height = 660, 440
        self.window.geometry(f'{width}x{height}+{max(0, (root.winfo_screenwidth()-width)//2)}+{max(0, (root.winfo_screenheight()-height)//2)}')
        self.window.transient(root)
        self.window.protocol('WM_DELETE_WINDOW', self.dismiss)
        self.window.bind('<Escape>', lambda _: self.dismiss())
        self.window.bind('<Right>', lambda _: self.next())
        self.window.bind('<Left>', lambda _: self.back())
        card = ttk.Frame(self.window, style='Rounded.TFrame', padding=30)
        card.pack(fill='both', expand=True, padx=16, pady=16)
        self.counter = ttk.Label(card, style='Muted.TLabel')
        self.counter.pack(anchor='w', pady=(0, 20))
        self.title = ttk.Label(card, style='Title.TLabel', wraplength=580)
        self.title.pack(anchor='w')
        self.body = ttk.Label(card, style='Card.TLabel', wraplength=570, justify='left', font=('Segoe UI', 11))
        self.body.pack(anchor='w', pady=20)
        self.note = ttk.Label(card, style='Muted.TLabel', wraplength=570)
        self.note.pack(anchor='w')
        controls = ttk.Frame(card, style='Card.TFrame')
        controls.pack(side='bottom', fill='x', pady=(16, 0))
        ttk.Button(controls, text='Skip introduction', command=self.dismiss, style='Card.TButton').pack(side='left')
        self.next_button = ttk.Button(controls, text='Next', style='Card.Primary.TButton', command=self.next)
        self.next_button.pack(side='right')
        self.back_button = ttk.Button(controls, text='Back', command=self.back, style='Card.TButton')
        self.back_button.pack(side='right', padx=8)
        self.render()
        self.window.grab_set()
        self.next_button.focus_set()

    def render(self):
        title, body, note = CARDS[self.index]
        self.counter.configure(text=f'QUICK START     {self.index + 1} / {len(CARDS)}     ' + '  '.join('●' if i == self.index else '○' for i in range(len(CARDS))))
        self.title.configure(text=title)
        self.body.configure(text=body)
        self.note.configure(text=note)
        self.back_button.configure(state='disabled' if self.index == 0 else 'normal')
        self.next_button.configure(text='Get started' if self.index == len(CARDS) - 1 else 'Next')

    def next(self):
        if self.index == len(CARDS) - 1:
            return self.dismiss()
        self.index += 1
        self.render()

    def back(self):
        self.index = max(0, self.index - 1)
        self.render()

    def dismiss(self):
        try:
            self.preferences.complete()
        except OSError:
            pass  # A read-only preferences location must never block the workspace.
        self.window.destroy()
