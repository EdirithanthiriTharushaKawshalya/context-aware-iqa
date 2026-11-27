"""Menu command models share the same action guards as workspace controls."""
import tkinter as tk
from tkinter import messagebox


def build_menus(app):
    menu = tk.Menu(app.root, tearoff=False)
    sections = []
    app.menu_actions = []
    def section(label):
        child = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label=label, menu=child)
        sections.append((label, child))
        return child
    def action(parent, label, command, enabled='always', accelerator=''):
        parent.add_command(label=label, command=command, accelerator=accelerator)
        app.menu_actions.append((parent, parent.index('end'), enabled))
    file = section('File')
    action(file, 'Open photo folder…', app.choose_folder, 'idle', 'Ctrl+O')
    action(file, 'Open saved project…', app.open, 'idle', 'Ctrl+Shift+O')
    action(file, 'Recover unfinished session…', app.recover, 'idle')
    file.add_separator()
    action(file, 'Save project…', app.finish, 'ready', 'Ctrl+S')
    action(file, 'Copy selection…', app.export, 'ready')
    file.add_separator()
    action(file, 'Exit', app.close)
    app.project_menu = file
    edit = section('Edit')
    from .appearance import preferences
    action(edit, 'Preferences…', lambda: preferences(app), accelerator='Ctrl+,')
    photo = section('Photo')
    action(photo, 'Keep', lambda: app.decide('Keep'), 'photo')
    action(photo, 'Review', lambda: app.decide('Review'), 'photo')
    action(photo, 'Reset assessment override', lambda: app.decide(''), 'photo')
    photo.add_separator()
    action(photo, 'Keep as creative', app.keep_creative, 'photo')
    action(photo, 'Inspect at 100%', app.zoom, 'photo')
    select = section('View')
    for label, value in (('All photos', 'All'), ('Keep', 'Keep'), ('Review', 'Review'), ('Chosen', 'Final selection'), ('Duplicates / bursts', 'Duplicates / bursts')):
        def show(view=value):
            app.view.set(view)
            app.refresh()
        action(select, label, show, 'project')
    review = section('Review')
    action(review, 'Analyze photos', app.analyze, 'ready')
    action(review, 'Pause analysis', app.pause, 'busy')
    review.add_separator()
    action(review, 'Compare duplicate / burst shots…', app.compare_burst, 'burst')
    action(review, 'Removal suggestions…', app.cleanup, 'finish')
    action(review, 'Review & finish folder…', app.finalize, 'finish')
    help_menu = section('Help')
    action(help_menu, 'Workspace tour…', app.introduction)
    action(help_menu, 'About Context Aware IQA', lambda: messagebox.showinfo(
        'Context Aware IQA', 'Context Aware IQA\n\nThoughtful photo selection, with you in control.\nAssess, compare, and curate your photographs.', parent=app.root))
    app.menu_model = menu
    app.root.configure(menu='')
    from .chrome import MenuBar
    app.menu_bar = MenuBar(app.root, sections)
    for key, command in (('<Control-o>', app.choose_folder), ('<Control-Shift-O>', app.open),
                         ('<Control-s>', app.finish), ('<Control-comma>', lambda: preferences(app))):
        def invoke(_, callback=command):
            if not getattr(app, 'tour', None) or not app.tour.window.winfo_exists():
                callback()
            return 'break'
        app.root.bind(key, invoke)


def sync_menus(app, ready):
    current = app.current() if app.project else None
    finish = ready and str(app.action_buttons['finish'].cget('state')) != 'disabled'
    flags = {'always': True, 'idle': not app.busy, 'ready': ready, 'busy': app.busy,
             'project': app.project is not None, 'photo': ready and current and not current.recycled,
             'burst': ready and current and current.duplicate_group, 'finish': finish}
    for menu, index, requirement in app.menu_actions:
        menu.entryconfigure(index, state='normal' if flags[requirement] else 'disabled')
    app.menu_bar.refresh()
