"""Presentation layer for the native photo review workspace."""
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

ASSETS = Path(__file__).resolve().parents[1] / 'images'


def hint(widget, text):
    """Show brief contextual help on hover or keyboard focus."""
    popup = None
    timer = None
    def hide(_=None):
        nonlocal popup, timer
        if timer:
            widget.after_cancel(timer)
            timer = None
        if popup:
            popup.destroy()
            popup = None
    def show():
        nonlocal popup, timer
        timer = None
        popup = tk.Toplevel(widget)
        popup.overrideredirect(True)
        popup.geometry(f'+{widget.winfo_rootx()}+{widget.winfo_rooty() + widget.winfo_height() + 6}')
        tk.Label(popup, text=text, background='#30283f', foreground='white', padx=12, pady=8,
                 wraplength=300, justify='left', font=('Segoe UI', 9)).pack()
    def schedule(_=None):
        nonlocal timer
        hide()
        timer = widget.after(650, show)
    for event in ('<Enter>', '<FocusIn>'):
        widget.bind(event, schedule, add='+')
    for event in ('<Leave>', '<FocusOut>', '<ButtonPress>', '<Destroy>'):
        widget.bind(event, hide, add='+')


def build(app):
    root = app.root
    root.configure(background='#f4f3f8')
    root.minsize(1120, 760)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10), background='#f4f3f8', foreground='#242332')
    style.configure('Card.TFrame', background='#ffffff')
    style.configure('Card.TLabel', background='#ffffff')
    style.configure('Muted.TLabel', foreground='#706d80', background='#ffffff', font=('Segoe UI', 9))
    style.configure('Title.TLabel', background='#ffffff', font=('Segoe UI', 20, 'bold'))
    style.configure('Section.TLabel', background='#ffffff', font=('Segoe UI', 11, 'bold'))
    style.configure('TButton', padding=(14, 9), background='#ffffff', bordercolor='#dedbe8', focuscolor='#7659d6')
    style.map('TButton', background=[('active', '#eee9fc'), ('pressed', '#e2d8fa')], foreground=[('disabled', '#aaa6b5')])
    style.configure('Primary.TButton', background='#7252cf', foreground='white', bordercolor='#7252cf', font=('Segoe UI', 10, 'bold'))
    style.map('Primary.TButton', background=[('disabled', '#e4dff0'), ('pressed', '#5334ac'), ('active', '#6241bd')], foreground=[('disabled', '#91879f'), ('!disabled', 'white')])
    style.configure('Danger.TButton', foreground='#a23652', background='#fff5f7', bordercolor='#efdbe1')
    style.configure('TEntry', padding=7, fieldbackground='#faf9fc', bordercolor='#dedbe8')
    style.configure('TCombobox', padding=7, fieldbackground='#faf9fc', background='#eeebf5', bordercolor='#dedbe8', arrowsize=14)
    style.map('TCombobox', fieldbackground=[('readonly', '#faf9fc')], selectbackground=[('readonly', '#faf9fc')], selectforeground=[('readonly', '#242332')])
    style.configure('TCheckbutton', background='#ffffff', padding=4)
    style.map('TCheckbutton', background=[('active', '#f3effc')])
    style.configure('Treeview', background='white', fieldbackground='white', foreground='#393545', rowheight=36, borderwidth=0)
    style.configure('Treeview.Heading', background='#f7f6fa', foreground='#706d80', font=('Segoe UI', 9, 'bold'), padding=(10, 10), relief='flat')
    style.map('Treeview', background=[('selected', '#eae2ff')], foreground=[('selected', '#38206e')])
    style.configure('Horizontal.TProgressbar', background='#8f72e4', troughcolor='#eae6f2', borderwidth=0, thickness=5)
    style.configure('TPanedwindow', background='#f4f3f8')
    style.configure('Vertical.TScrollbar', background='#dfdbe9', troughcolor='#faf9fc', borderwidth=0, arrowsize=12)
    root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 10))
    root.option_add('*TCombobox*Listbox.selectBackground', '#eae2ff')
    root.option_add('*TCombobox*Listbox.selectForeground', '#38206e')
    from .soft_theme import install
    install(root, style)
    with Image.open(ASSETS / 'icon.png') as image:
        app.window_icon = ImageTk.PhotoImage(image, master=root)
        app.brand_icon = ImageTk.PhotoImage(image.resize((48, 36), Image.Resampling.LANCZOS), master=root)
    root.iconphoto(True, app.window_icon)

    header = ttk.Frame(root, style='Card.TFrame', padding=(24, 16))
    header.pack(fill='x')
    ttk.Label(header, image=app.brand_icon, style='Card.TLabel').pack(side='left', padx=(0, 12))
    brand = ttk.Frame(header, style='Card.TFrame')
    brand.pack(side='left')
    ttk.Label(brand, text='Context Aware IQA', style='Title.TLabel').pack(anchor='w')
    ttk.Label(brand, text='A considered selection. Every moment accounted for.', style='Muted.TLabel').pack(anchor='w')
    ttk.Label(header, text='PHOTO REVIEW  /  DESKTOP', style='Muted.TLabel').pack(side='right')
    workflow = ttk.Label(root, textvariable=app.workflow, foreground='#74668f', padding=(24, 12), font=('Segoe UI', 9))
    workflow.pack(fill='x')

    setup = ttk.Frame(root, style='Rounded.TFrame', padding=(18, 14))
    setup.pack(fill='x', padx=20)
    setup.columnconfigure(1, weight=1)
    ttk.Label(setup, text='Photo folder', style='Section.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 16))
    ttk.Entry(setup, textvariable=app.source, state='readonly').grid(row=0, column=1, columnspan=4, sticky='ew')
    app.folder_button = ttk.Button(setup, text='Choose folder…', command=app.choose_folder, style='Card.TButton')
    app.folder_button.grid(row=0, column=5, padx=(12, 0))
    options = ttk.Frame(setup, style='Card.TFrame')
    options.grid(row=1, column=0, columnspan=6, sticky='ew', pady=(14, 0))
    for column, label in enumerate(('Final photo count', 'Selection style', 'Shot intent', 'Motion intent')):
        ttk.Label(options, text=label, style='Muted.TLabel').grid(row=0, column=column, sticky='w', padx=(0, 20))
    app.count_control = ttk.Entry(options, textvariable=app.target, width=17)
    app.count_control.grid(row=1, column=0, sticky='ew', padx=(0, 16), pady=5)
    hint(app.count_control, 'Optional: leave blank for all suitable photos, or enter the size of your final selection.')
    for column, attr, variable, values in (
        (1, 'style_control', app.style, ('Balanced', 'Strict quality')),
        (2, 'shot_control', app.shot, ('auto', 'portrait', 'couple', 'group', 'candid', 'detail', 'scene')),
        (3, 'motion_control', app.motion, ('auto', 'freeze', 'intentional'))):
        control = ttk.Combobox(options, textvariable=variable, state='readonly', values=values, width=17)
        control.grid(row=1, column=column, sticky='ew', padx=(0, 16), pady=5)
        setattr(app, attr, control)
        hint(control, {'style_control': 'Balanced favors variety across similar shots. Strict quality favors the highest scores.',
                       'shot_control': 'Auto adapts to each photo. Choose an intent when the whole folder needs the same assessment.',
                       'motion_control': 'Auto detects motion. Intentional allows creative motion blur; freeze favors sharp subjects.'}[attr])
        if column > 1:
            control.bind('<<ComboboxSelected>>', app.configuration_changed)
        options.columnconfigure(column, weight=1)
    app.recursive_control = ttk.Checkbutton(options, text='Include subfolders', variable=app.recursive, command=app.configuration_changed)
    app.recursive_control.grid(row=1, column=4, sticky='w')
    ttk.Label(options, textvariable=app.selection_hint, style='Muted.TLabel').grid(row=2, column=0, columnspan=5, sticky='w')

    toolbar = ttk.Frame(root, padding=(20, 12))
    toolbar.pack(fill='x')
    app.action_buttons = {}
    for name, label, action, button_style in (
        ('analyze', 'Analyze photos', app.analyze, 'Primary.TButton'),
        ('pause', 'Pause', app.pause, 'TButton'),
        ('cleanup', 'Review cleanup', app.cleanup, 'TButton'),
        ('finish', 'Finish project…', app.finish, 'Primary.TButton')):
        button = ttk.Button(toolbar, text=label, command=action, style=button_style)
        button.pack(side='right' if name == 'finish' else 'left', padx=(0, 8))
        app.action_buttons[name] = button

    app.metrics = {name: tk.StringVar(value='—') for name in ('Photos', 'Keep', 'Review', 'Selected')}
    metrics = ttk.Frame(root)
    metrics.pack(fill='x', padx=20, pady=(0, 10))
    for col, (name, variable) in enumerate(app.metrics.items()):
        metrics.columnconfigure(col, weight=1)
        card = ttk.Frame(metrics, style='Rounded.TFrame', padding=(18, 10))
        card.grid(row=0, column=col, sticky='ew', padx=(0, 8 if col < 3 else 0))
        ttk.Label(card, text=name.upper(), style='Muted.TLabel').pack(side='left')
        ttk.Label(card, textvariable=variable, style='Section.TLabel').pack(side='right')

    body = ttk.Panedwindow(root, orient='horizontal')
    left = ttk.Frame(body, style='Rounded.TFrame', padding=16)
    right = ttk.Frame(body, style='Rounded.TFrame', padding=16)
    body.add(left, weight=2)
    body.add(right, weight=3)
    library = ttk.Frame(left, style='Card.TFrame')
    library.pack(fill='x', pady=(0, 10))
    ttk.Label(library, text='Photo library', style='Section.TLabel').pack(side='left')
    filters = ttk.Combobox(library, textvariable=app.view, state='readonly', width=21, values=('All', 'Keep', 'Review', 'Final selection', 'Other Keep', 'Pending', 'Marked for removal', 'Recycled'))
    filters.pack(side='right')
    filters.bind('<<ComboboxSelected>>', lambda _: app.refresh())
    table_frame = ttk.Frame(left, style='Card.TFrame')
    table_frame.pack(fill='both', expand=True)
    app.table = ttk.Treeview(table_frame, columns=('decision', 'score', 'pin'), selectmode='browse', height=6)
    for col, label, width in (('#0', 'PHOTO', 250), ('decision', 'DECISION', 135), ('score', 'SCORE', 65), ('pin', 'PROTECTED', 95)):
        app.table.heading(col, text=label)
        app.table.column(col, width=width, minwidth=width if col != '#0' else 120, stretch=col == '#0')
    for tag, color in (('Keep', '#24765b'), ('Review', '#926528'), ('Excluded', '#aa3653'), ('Pending', '#827b90'), ('Recycled', '#827b90')):
        app.table.tag_configure(tag, foreground=color)
    scroll = ttk.Scrollbar(table_frame, orient='vertical', command=app.table.yview)
    app.table.configure(yscrollcommand=scroll.set)
    scroll.pack(side='right', fill='y')
    app.table.pack(fill='both', expand=True)
    app.table.bind('<<TreeviewSelect>>', app.preview)
    ttk.Label(right, text='Photo inspection', style='Section.TLabel').pack(anchor='w', pady=(0, 10))
    inspection = ttk.Frame(right, style='Card.TFrame')
    photo_page = ttk.Frame(inspection, style='Card.TFrame', height=1, width=1)
    photo_page.pack_propagate(False)
    assessment_page = ttk.Frame(inspection, style='Card.TFrame', padding=(12, 8))
    assessment_page.pack(side='right', fill='y')
    photo_page.pack(side='left', fill='both', expand=True)
    app.preview_label = tk.Label(photo_page, background='#f6f3fb', foreground='#9b8caf', text='Choose a photo\nto inspect.', font=('Segoe UI', 10), anchor='center')
    app.preview_label.pack(fill='both', expand=True, padx=8, pady=8)
    app.details = tk.StringVar(value='Select a photo to see its score and assessment.')
    ttk.Label(assessment_page, text='ASSESSMENT', style='Muted.TLabel').pack(anchor='w', pady=(0, 12))
    detail = tk.Text(assessment_page, wrap='word', font=('Segoe UI', 10), background='white', foreground='#625c73', relief='flat', width=23, height=1, state='disabled', cursor='arrow', spacing3=6)
    detail_scroll = ttk.Scrollbar(assessment_page, command=detail.yview)
    detail.configure(yscrollcommand=detail_scroll.set)
    detail_scroll.pack(side='right', fill='y')
    detail.pack(fill='both', expand=True)
    def update_details(*_):
        detail.configure(state='normal')
        detail.delete('1.0', 'end')
        detail.insert('1.0', app.details.get())
        detail.configure(state='disabled')
    app.details.trace_add('write', update_details)
    update_details()
    decisions = ttk.Frame(right, style='Card.TFrame')
    decisions.pack(side='bottom', fill='x', pady=(10, 0))
    actions = ttk.Frame(decisions, style='Card.TFrame')
    actions.pack(fill='x')
    app.photo_buttons = []
    for col, (label, decision) in enumerate((('Keep', 'Keep'), ('Review', 'Review'), ('Mark removal', 'Excluded'), ('Reset', ''))):
        actions.columnconfigure(col, weight=1)
        button = ttk.Button(actions, text=label, style='Photo.Danger.TButton' if decision == 'Excluded' else 'Photo.Card.TButton', command=lambda d=decision: app.decide(d))
        button.grid(row=0, column=col, sticky='ew', padx=(0, 4))
        app.photo_buttons.append(button)
        hint(button, {'Keep': 'Approve this photo for selection.', 'Review': 'Leave this photo for manual review.', 'Excluded': 'Mark for cleanup. The file stays in place until you confirm Review cleanup.', '': 'Restore the original assessment and clear your manual decision.'}[decision])
    app.protection_control = ttk.Checkbutton(decisions, text='Protect this photo — always include it', variable=app.protected, command=app.pin)
    app.protection_control.pack(anchor='w', pady=(8, 0))
    hint(app.protection_control, 'Essential moments stay in your selection and are protected from cleanup.')
    ttk.Button(decisions, text='Inspect at 100%', command=app.zoom, style='Card.TButton').pack(anchor='w', pady=(8, 0))
    inspection.pack(fill='both', expand=True)
    footer = ttk.Frame(root, style='Rounded.TFrame', padding=(18, 12))
    footer.pack(side='bottom', fill='x', padx=20, pady=10)
    ttk.Label(footer, textvariable=app.status, style='Card.TLabel', wraplength=1180, font=('Segoe UI', 9), foreground='#625c73').pack(fill='x')
    app.bar = ttk.Progressbar(footer)
    app.bar.pack(fill='x', pady=(8, 6))
    ttk.Label(footer, text='Automatic recovery is on  ·  Originals stay in place until you confirm cleanup', style='Card.TLabel', foreground='#837b91', font=('Segoe UI', 9)).pack(anchor='w')
    body.pack(fill='both', expand=True, padx=20)
    compact = False
    resize_timer = None
    dimensions = None
    def resize(event):
        nonlocal compact, resize_timer, dimensions
        if event.widget != root or dimensions == (event.width, event.height):
            return
        dimensions = event.width, event.height
        small = event.height < 860
        if small != compact:
            compact = small
            if small:
                workflow.pack_forget()
                metrics.pack_forget()
            else:
                workflow.pack(fill='x', before=setup)
                metrics.pack(fill='x', padx=20, pady=(0, 10), before=body)
        if resize_timer:
            root.after_cancel(resize_timer)
        def redraw():
            nonlocal resize_timer
            resize_timer = None
            if app.project and app.current() and not app.busy:
                app.preview()
        resize_timer = root.after(180, redraw)
    root.bind('<Configure>', resize, add='+')
