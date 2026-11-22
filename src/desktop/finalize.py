"""Visual, explicit plan for keeping the delivery set and recycling the rest."""
import tkinter as tk
from tkinter import messagebox, ttk

from .gallery import PhotoGallery


def show_finalization(app):
    if not app.idle() or not app.project or not app.update():
        return
    if app.config_changed():
        messagebox.showinfo('Analyze first', 'Analyze with the changed settings before finalizing.')
        return
    try:
        before = [(p.relative, p.size, p.modified, p.recycled) for p in app.project.photos]
        app.project.scan()
        after = [(p.relative, p.size, p.modified, p.recycled) for p in app.project.photos]
        if before != after:
            app.checkpoint()
            app.refresh()
        plan = app.project.finalization_plan()
    except (OSError, ValueError) as exc:
        messagebox.showinfo('Finish analysis first', str(exc))
        return
    window = tk.Toplevel(app.root)
    app.finalization_window = window
    window.title('Review and finish your photo folder')
    window.geometry('1050x740')
    window.minsize(800, 580)
    window.transient(app.root)
    window.grab_set()
    chosen, remove, retained = plan['chosen'], plan['remove'], plan['retained']
    expected_chosen = tuple(p.relative for p in chosen)
    expected_remove = tuple(p.relative for p in remove)
    target = app.project.target
    short = target is not None and len(chosen) < target
    ttk.Label(window, text='Your final folder', font=('Segoe UI', 19, 'bold'), padding=(20,16)).pack(anchor='w')
    ttk.Label(window, text=f'{len(chosen)} chosen · {len(remove)} unchosen photos available to recycle · '
                           f'{len(retained)} protected or unreadable photos left untouched',
              padding=(20,0), wraplength=980).pack(anchor='w')
    ttk.Label(window, text=f'Original folder: {app.project.source}', padding=(20,8), wraplength=980).pack(anchor='w')
    note = ('Choose more photos to reach your target before removing the rest.' if short else
            'Inspect both lists. Recycling is optional. The chosen originals stay in this folder.')
    if not chosen:
        note = 'Choose at least one photo in the gallery before recycling the rest. You can still save without cleanup.'
    creative = sum(p.possible_creative or p.style_label == 'Creative' for p in remove)
    if creative:
        note += f' The removal list includes {creative} possible or confirmed creative photos.'
    ttk.Label(window, text=note, foreground='#926528', padding=(20,6), wraplength=980).pack(fill='x')
    tabs = ttk.Notebook(window)
    tabs.pack(fill='both', expand=True, padx=20, pady=12)
    index = {p.relative: str(i) for i, p in enumerate(app.project.photos)}
    for label, photos in ((f'Keep in folder ({len(chosen)})', chosen),
                          (f'Move to Recycle Bin ({len(remove)})', remove),
                          (f'Leave untouched ({len(retained)})', retained)):
        gallery = PhotoGallery(tabs)
        tabs.add(gallery, text=label)
        gallery.set_photos(app.project, [index[p.relative] for p in photos], expected_chosen)
    controls = ttk.Frame(window, padding=(20,12))
    controls.pack(fill='x')
    ttk.Button(controls, text='Back to choosing', command=window.destroy).pack(side='left')

    def save_only():
        window.destroy()
        app.finish()

    def apply_cleanup():
        if not messagebox.askyesno('Confirm this exact folder cleanup',
                f'Keep {len(chosen)} chosen photos in:\n{app.project.source}\n\n'
                f'Move the {len(remove)} photos shown in the removal tab to the Windows Recycle Bin?\n'
                f'{len(retained)} protected or unreadable photos will stay untouched.\n\n'
                'You can restore recycled files from the Windows Recycle Bin.', parent=window, default='no'):
            return
        try:
            app.start_final_cleanup(expected_chosen, expected_remove)
        except (OSError, ValueError) as exc:
            messagebox.showerror('Folder unchanged', str(exc), parent=window)
            return
        window.destroy()

    ttk.Button(controls, text='Save project without cleanup…', command=save_only).pack(side='right', padx=(8,0))
    ttk.Button(controls, text=f'Recycle {len(remove)} unchosen & finish…', command=apply_cleanup,
               style='Primary.TButton', state='normal' if chosen and remove and not short else 'disabled').pack(side='right')
    ttk.Label(window, text='After cleanup, choose where to save your project. Automatic recovery also preserves your progress.',
              padding=(20,0,20,12), wraplength=980).pack(fill='x')
    return window
