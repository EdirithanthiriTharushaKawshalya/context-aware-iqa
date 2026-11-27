"""Side-by-side burst contact sheet with explicit delivery choices."""
import tkinter as tk
from tkinter import ttk, messagebox

from .gallery import PhotoGallery
from .duplicates import quality_rank


def show_burst(app):
    if not app.idle() or not app.project or not app.update():
        return
    photo = app.current()
    members = app.project.duplicate_recommendations().get(photo.duplicate_group,[]) if photo else []
    if not members:
        messagebox.showinfo('Compare shots','Select a photo in Duplicates / bursts first.')
        return
    window = tk.Toplevel(app.root)
    app.burst_window = window
    window.title('Compare duplicate and burst shots')
    window.geometry('1080x660')
    window.minsize(800,560)
    window.transient(app.root)
    window.grab_set()
    ttk.Label(window,text=f'{len(members)} '+('exact copies' if photo.duplicate_kind == 'exact' else 'similar burst candidates'),
              font=('Segoe UI',17,'bold'),padding=(18,12)).pack(anchor='w')
    explanation = ttk.Label(window,text='Best uses Keep status, overall quality and weakest assessable subject detail. '
              'Expressions, smiles and closed eyes are not rated. Double-click a card to open the photo; choose 100% to inspect detail.',
              wraplength=1000,padding=(18,0))
    explanation.pack(fill='x')
    selected = tk.StringVar(value='')
    detail = ttk.Label(window,textvariable=selected,wraplength=1000,padding=(18,10))
    detail.pack(fill='x')
    def resize(event):
        if event.widget == window:
            explanation.configure(wraplength=max(300,event.width-36))
            detail.configure(wraplength=max(300,event.width-36))
    window.bind('<Configure>',resize,add='+')
    index = {p.relative:str(i) for i,p in enumerate(app.project.photos)}
    by_id = {index[p.relative]:p for p in members}
    active = [index[photo.relative]]

    def inspect(iid):
        active[0] = iid
        candidate = by_id[iid]
        selected.set(f'{candidate.relative} · AI: {candidate.status} · Decision: {candidate.decision} · '
                     f'Score: {candidate.score:.1f} · {candidate.duplicate_label}')

    def refresh():
        app.project.duplicate_recommendations()
        ordered = sorted(members,key=quality_rank)
        gallery.set_photos(app.project,[index[p.relative] for p in ordered],
                           [p.relative for p in app.project.selection()],editable=True,selected=active[0])
        inspect(active[0])

    def toggle(iid):
        old = [(p,p.override,p.selection_choice) for p in members]
        try:
            chosen = {p.relative for p in app.project.selection()}
            app.project.choose_photo(by_id[iid].relative,by_id[iid].relative not in chosen)
            app.checkpoint()
            app.refresh()
            active[0] = iid
            refresh()
        except (OSError,ValueError) as exc:
            for member,override,choice in old:
                member.override,member.selection_choice = override,choice
            app.refresh()
            refresh()
            messagebox.showinfo('Chosen photos',str(exc),parent=window)

    gallery = PhotoGallery(window,on_select=inspect,on_toggle=toggle,
                           on_open=lambda iid: app.view_photo(photo=by_id[iid],parent=window))
    gallery.pack(fill='both',expand=True,padx=18,pady=(0,10))

    def use_selected():
        old = [(p,p.override,p.selection_choice) for p in members]
        try:
            app.project.choose_burst_photo(by_id[active[0]].relative)
            app.checkpoint()
            app.refresh()
            refresh()
        except (OSError,ValueError) as exc:
            for member,override,choice in old:
                member.override,member.selection_choice = override,choice
            app.refresh()
            refresh()
            messagebox.showinfo('Burst choice',str(exc),parent=window)

    controls = ttk.Frame(window,padding=(18,12))
    controls.pack(fill='x')
    ttk.Label(controls,text='Use the card buttons to keep multiple variations. No originals are moved here.',wraplength=740).pack(anchor='w')
    actions = ttk.Frame(controls)
    actions.pack(fill='x',pady=(8,0))
    ttk.Button(actions,text='Done',command=window.destroy).pack(side='right')
    ttk.Button(actions,text='Use selected shot for this burst',command=use_selected,style='Primary.TButton').pack(side='right',padx=8)
    refresh()
    return window
