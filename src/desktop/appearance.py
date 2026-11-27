"""Live desktop appearance, shared by ttk surfaces and painted photo cards."""
import tkinter as tk
from tkinter import ttk


DARK = {
    'white': '#26272e', '#ffffff': '#26272e', '#f4f3f8': '#1c1d23',
    '#f7f6fa': '#202127', '#faf9fc': '#30313a', '#faf8fd': '#30313a',
    '#f6f3fb': '#202127', '#302442': '#eeeaf8', '#242332': '#eeeaf8',
    '#393545': '#e3dfed', '#30283f': '#17181e', '#625c73': '#c4bece',
    '#706d80': '#b4adbf', '#837b91': '#b4adbf', '#827b90': '#b4adbf',
    '#91879f': '#b4adbf', '#9b8caf': '#b4adbf', '#84758f': '#b4adbf',
    '#ddd8e8': '#42434e', '#dedbe8': '#42434e', '#e1dbe9': '#42434e',
    '#e0daeb': '#42434e', '#e6e1ed': '#42434e', '#e7e1ee': '#42434e',
    '#f0eafa': '#383143', '#e3d8f8': '#473759', '#f4f1f8': '#303039',
    '#f4f1f7': '#303039', '#e7e0f3': '#393441', '#f7f1f4': '#303039',
    '#fff2f5': '#422b34', '#fff5f7': '#422b34', '#fbe4eb': '#54303e',
    '#f5d2dd': '#633749', '#eeebf5': '#383143', '#eee9fc': '#383143',
    '#e2d8fa': '#473759', '#f3effc': '#383143', '#eae2ff': '#49356b',
    '#38206e': '#eeeaf8', '#e9f4ef': '#243e35', '#f3effa': '#383143',
    '#24765b': '#79d6ad', '#926528': '#edc17c', '#aa3653': '#f694ad',
    '#a23652': '#f694ad', '#7659d6': '#b29af5', '#7655d3': '#8464df',
    '#eee8f6': '#36313f', '#eee6f2': '#36313f', '#eae6f2': '#36313f',
    '#dfdbe9': '#4c4858', '#e4dff0': '#393441', '#aaa6b5': '#827b90',
    'black': '#eeeaf8', 'systemmenutext': '#eeeaf8', 'systemwindowtext': '#eeeaf8',
    'systemmenu': '#26272e', 'systembuttonface': '#26272e',
}


def color(widget, value):
    return DARK.get(value.lower(), value) if getattr(widget.winfo_toplevel(), '_iqa_theme', 'light') == 'dark' else value


def paint_widget(widget, mode):
    """Remember original explicit colors so repeated switches never drift."""
    if getattr(widget, '_appearance_managed', False):
        return
    if isinstance(widget, (tk.Tk, tk.Toplevel, tk.Frame, tk.Label, tk.Text, tk.Canvas, tk.Menu, tk.Listbox)):
        originals = getattr(widget, '_appearance_colors', {})
        for option in ('background', 'foreground', 'insertbackground', 'selectbackground', 'selectforeground', 'activebackground', 'activeforeground'):
            if option in widget.keys():
                originals.setdefault(option, str(widget.cget(option)))
                value = originals[option]
                widget.configure(**{option: DARK.get(value.lower(), value) if mode == 'dark' else value})
        widget._appearance_colors = originals
    elif isinstance(widget, ttk.Widget):
        originals = getattr(widget, '_appearance_colors', {})
        for option in ('background', 'foreground'):
            if option in widget.keys() and str(widget.cget(option)):
                originals.setdefault(option, str(widget.cget(option)))
                value = originals[option]
                widget.configure(**{option: DARK.get(value.lower(), value) if mode == 'dark' else value})
        widget._appearance_colors = originals
    for child in widget.winfo_children():
        if isinstance(child, tk.Toplevel):
            child._iqa_theme = mode
        paint_widget(child, mode)


def match_surfaces(widget, mode):
    """Give transparent control corners the actual containing panel's color."""
    if getattr(widget, '_appearance_managed', False):
        return
    style = ttk.Style(widget)
    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Combobox)):
        base = getattr(widget, '_surface_base_style', None)
        if base is None:
            base = str(widget.cget('style')) or widget.winfo_class()
            widget._surface_base_style = base
        parent = widget.master
        if isinstance(parent, ttk.Widget):
            parent_style = str(parent.cget('style')) or parent.winfo_class()
            surface = ('#26272e' if mode == 'dark' else '#ffffff') if parent_style == 'Rounded.TFrame' else style.lookup(parent_style, 'background')
        else:
            surface = parent.cget('background')
        if not str(surface).startswith('#'):
            surface = '#1c1d23' if mode == 'dark' else '#f4f3f8'
        name = 'Surface'+surface[1:]+'.'+base
        style.configure(name, background=surface)
        style.map(name, background=[('disabled', surface), ('!disabled', surface)])
        widget.configure(style=name)
    for child in widget.winfo_children():
        match_surfaces(child, mode)


def watch_windows(app):
    """Theme owned dialogs when they map, including their native title bars."""
    host = app.root._root()
    if not hasattr(host, '_iqa_apps'):
        host._iqa_apps = {}
        def mapped(event):
            widget = event.widget
            if not isinstance(widget, tk.Misc):
                return
            owner = widget
            while owner is not None:
                if getattr(owner, '_appearance_managed', False):
                    return
                if owner in host._iqa_apps:
                    application = host._iqa_apps[owner]
                    mode = application.appearance.get()
                    widget.winfo_toplevel()._iqa_theme = mode
                    paint_widget(widget, mode)
                    match_surfaces(widget, mode)
                    if isinstance(widget, (tk.Tk, tk.Toplevel)):
                        from .chrome import titlebar
                        titlebar(widget, mode)
                    return
                owner = owner.master
        host.bind_all('<Map>', mapped, add='+')
    host._iqa_apps[app.root] = app
    def removed(event):
        if event.widget == app.root:
            host._iqa_apps.pop(app.root, None)
    app.root.bind('<Destroy>', removed, add='+')


def apply(app, mode):
    root = app.root
    root._iqa_theme = mode
    style = ttk.Style(root)
    if not hasattr(app, '_appearance_styles'):
        names = ('.', 'TFrame', 'TLabel', 'Card.TFrame', 'Card.TLabel', 'Muted.TLabel',
                 'Title.TLabel', 'Section.TLabel', 'TButton', 'Primary.TButton', 'Danger.TButton',
                 'TEntry', 'TCombobox', 'TCheckbutton', 'Treeview', 'Treeview.Heading',
                 'Horizontal.TProgressbar', 'TPanedwindow', 'Vertical.TScrollbar')
        app._appearance_styles = {name: (style.configure(name) or {}, style.map(name) or {}) for name in names}
    def mapped(value):
        return DARK.get(str(value).lower(), value) if mode == 'dark' else value
    for name, (config, states) in app._appearance_styles.items():
        style.configure(name, **{key: mapped(value) for key, value in config.items()})
        style.map(name, **{key: [(*entry[:-1], mapped(entry[-1])) for entry in entries] for key, entries in states.items()})
    style.configure('Primary.TButton', foreground='white')
    style.map('Primary.TButton', foreground=[('disabled', '#91879f'), ('!disabled', 'white')])
    # These surfaces use inherited defaults in the initial workspace.
    bg, panel, fg = ('#1c1d23', '#26272e', '#eeeaf8') if mode == 'dark' else ('#f4f3f8', '#ffffff', '#242332')
    style.configure('TNotebook', background=panel, borderwidth=0, bordercolor=panel,
                    lightcolor=panel, darkcolor=panel)
    style.configure('TNotebook.Tab', background=bg, foreground=fg, padding=(14, 6),
                    bordercolor=panel, lightcolor=panel, darkcolor=panel)
    style.configure('TCheckbutton', indicatorbackground=panel, indicatorforeground=fg)
    style.map('TCheckbutton', indicatorbackground=[('disabled', bg), ('!disabled', panel)])
    style.map('TNotebook.Tab', background=[('selected', panel)], foreground=[('selected', fg)])
    style.configure('TRadiobutton', background=panel, foreground=fg, padding=8)
    style.map('TRadiobutton', background=[('active', panel)])
    from .soft_theme import install
    install(root, style, mode=mode)
    paint_widget(root, mode)
    match_surfaces(root, mode)
    from .chrome import titlebar
    titlebar(root, mode)
    for child in root.winfo_children():
        if isinstance(child, tk.Toplevel):
            titlebar(child, mode)
    app.menu_bar.set_theme(mode)
    # Preserve the tutorial's intentionally dark translucent scrim.
    tour = getattr(app, 'tour', None)
    if tour and tour.window.winfo_exists():
        tour.overlay.configure(background='#101018')
        tour.canvas.configure(background='#101018')
    root.configure(background=bg)
    for tag, value in (('Keep', '#24765b'), ('Review', '#926528'), ('Excluded', '#aa3653'), ('Pending', '#827b90'), ('Recycled', '#827b90')):
        app.table.tag_configure(tag, foreground=mapped(value))
    app.gallery.render()


def preferences(app):
    existing = getattr(app, 'preferences_window', None)
    if existing and existing.winfo_exists():
        existing.lift()
        return
    window = app.preferences_window = tk.Toplevel(app.root)
    window.title('Preferences')
    window.transient(app.root)
    window.geometry('500x310')
    window.resizable(False, False)
    card = ttk.Frame(window, style='Card.TFrame', padding=28)
    card.pack(fill='both', expand=True)
    ttk.Label(card, text='Appearance', style='Title.TLabel').pack(anchor='w')
    ttk.Label(card, text='Choose a comfortable workspace for your photos.', style='Muted.TLabel').pack(anchor='w', pady=(8, 20))
    choice = tk.StringVar(value=app.appearance.get())
    def change():
        app.appearance.set(choice.get())
        apply(app, choice.get())
        try:
            app.user_preferences.save(theme=choice.get())
            note.configure(text='Saved. Your theme will be restored next time.')
        except OSError:
            note.configure(text='Applied for this session. Preferences could not be saved.')
    for value, title in (('light', 'Light — a clean, bright workspace'), ('dark', 'Dark — keep the focus on your photographs')):
        ttk.Radiobutton(card, text=title, value=value, variable=choice, command=change).pack(anchor='w')
    note = ttk.Label(card, text='Changes apply immediately.', style='Muted.TLabel')
    note.pack(anchor='w', pady=16)
    ttk.Button(card, text='Done', style='Card.Primary.TButton', command=window.destroy).pack(anchor='e')
    window._iqa_theme = app.appearance.get()
    paint_widget(window, app.appearance.get())
    window.bind('<Escape>', lambda _: window.destroy())
