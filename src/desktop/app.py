"""Native desktop review, selection and confirmed source-folder cleanup."""
import queue
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from .analysis import AnalysisSession
from .project import Photo, Project
from .sessions import SessionStore


class DesktopApp:
    def __init__(self, root, session_store=None):
        self.root = root
        self.session_store = session_store or SessionStore()
        self.project = None
        self.project_file = None
        self.saved_project_file = None
        self.dirty = False
        self.loading_preferences = False
        self.preference_timer = None
        self.preferences_valid = True
        self.busy = False
        self.session = None
        self.session_outcome = None
        self.analysis_stage = ''
        self.stage_started = time.monotonic()
        self.close_requested = False
        self.cleaning = False
        self.cleanup_stop = threading.Event()
        self.cleanup_events = queue.Queue()
        root.title('Context Aware IQA — Photo Review')
        root.geometry('1400x960')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.source = tk.StringVar()
        self.target = tk.StringVar(value='')
        self.recursive = tk.BooleanVar(value=True)
        self.shot = tk.StringVar(value='auto')
        self.motion = tk.StringVar(value='auto')
        self.style = tk.StringVar(value='Balanced')
        self.view = tk.StringVar(value='All')
        self.protected = tk.BooleanVar(value=False)
        self.selection_hint = tk.StringVar(value='Blank = all suitable photos')
        self.workflow = tk.StringVar(value='Choose folder  →  Analyze  →  Review & clean up  →  Finish & save')
        self.status = tk.StringVar(value='Select a folder. Leave the count blank to keep all suitable photos. Cleanup requires your confirmation.')
        menu = tk.Menu(root)
        self.project_menu = tk.Menu(menu, tearoff=False)
        self.project_menu.add_command(label='Open saved project…', command=self.open)
        self.project_menu.add_command(label='Recover unfinished session…', command=self.recover)
        self.project_menu.add_separator()
        self.project_menu.add_command(label='Copy selection…', command=self.export)
        self.project_menu.add_command(label='Finish & save…', command=self.finish)
        self.project_menu.add_separator()
        self.project_menu.add_command(label='Exit', command=self.close)
        menu.add_cascade(label='Project', menu=self.project_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label='Quick start introduction…', command=self.introduction)
        menu.add_cascade(label='Help', menu=help_menu)
        root.configure(menu=menu)
        from .interface import build
        build(self)
        self.target.trace_add('write', self.schedule_preferences)
        self.style.trace_add('write', self.schedule_preferences)
        self.sync_controls()
        root.after(150, self.poll)

    def introduction(self):
        from .welcome import Introduction
        if getattr(self, 'tour', None) and self.tour.window.winfo_exists():
            self.tour.window.lift()
        else:
            self.tour = Introduction(self.root)

    def idle(self):
        if self.busy:
            messagebox.showinfo('Operation running', 'Click Pause before changing this project.')
            return False
        return True

    def choose_folder(self):
        if self.idle():
            path = filedialog.askdirectory()
            if path and self.before_replace():
                self.start_folder(path)

    def checkpoint(self):
        self.dirty = True
        self.project.finished_at = None
        self.project.finish_summary = {}
        self.project.save(self.project_file)

    def activate(self, project, checkpoint, saved_file=None, dirty=True):
        self.loading_preferences = True
        try:
            self.project, self.project_file = project, Path(checkpoint)
            self.saved_project_file = Path(saved_file) if saved_file else None
            self.dirty = dirty
            self.source.set(project.source)
            self.target.set('' if project.target is None else str(project.target))
            self.recursive.set(project.recursive)
            self.shot.set(project.shot_type)
            self.motion.set(project.motion_intent)
            self.style.set(project.style)
            self.preferences_valid = True
            self.selection_hint.set('Blank = all suitable photos · selection updates automatically')
            self.table.delete(*self.table.get_children())
            self.preview_label.configure(image='', text='Select a photo to inspect it.')
            self.details.set('Your work is saved to automatic recovery. Choose Finish project when ready.')
            self.refresh()
        finally:
            self.loading_preferences = False

    def start_folder(self, path):
        try:
            source = Path(path).resolve(strict=True)
            if not source.is_dir():
                raise ValueError('Choose a photo folder.')
            project = Project(str(source), self.settings(), self.recursive.get(), self.shot.get(), self.motion.get(), self.style.get())
            project.scan()
            checkpoint = self.session_store.create(project)
            self.activate(project, checkpoint)
            if not project.photos:
                self.status.set('No supported photos or recognized RAW files were found in this folder.')
            return True
        except Exception as exc:
            messagebox.showerror('Open photo folder', str(exc))
            return False

    def before_replace(self):
        if not self.project or not self.dirty:
            return True
        answer = messagebox.askyesnocancel('Save current project?',
                'Save your current review to a project file before switching?\n\nYes: choose where to save.\nNo: leave an automatic recovery copy.\nCancel: continue this review.')
        if answer is None:
            return False
        return self.finish(show_summary=False) if answer else True

    def settings(self):
        if not self.target.get().strip():
            return None
        try:
            target = int(self.target.get())
            if target < 1:
                raise ValueError()
        except ValueError:
            raise ValueError('Enter a positive whole number, or leave the count blank to keep all suitable photos.') from None
        return target

    def open(self):
        if not self.idle():
            return
        path = filedialog.askopenfilename(filetypes=[('IQA project', '*.iqa.json')])
        if not path:
            return
        try:
            project = Project.load(path)
            project.scan()
            if self.before_replace():
                checkpoint = self.session_store.create(project)
                self.activate(project, checkpoint, saved_file=path, dirty=False)
        except Exception as exc:
            messagebox.showerror('Open project', str(exc))

    def recover(self):
        if not self.idle():
            return
        entries = [(path, project) for path, project in self.session_store.unfinished()
                   if not self.project_file or path.resolve() != Path(self.project_file).resolve()]
        if not entries:
            messagebox.showinfo('Recover a session', 'There are no other unfinished sessions. Use Open saved project for a project you saved with Finish.')
            return
        window = tk.Toplevel(self.root)
        window.title('Recover unfinished session')
        window.geometry('750x360')
        window.transient(self.root)
        window.grab_set()
        ttk.Label(window, text='Continue work preserved automatically before Finish was used.', padding=10).pack(fill='x')
        listing = ttk.Treeview(window, columns=('progress', 'updated'), selectmode='browse')
        listing.heading('#0', text='Photo folder')
        listing.heading('progress', text='Processed / total')
        listing.heading('updated', text='Last saved')
        listing.column('#0', width=420)
        listing.column('progress', width=110)
        listing.column('updated', width=170)
        listing.pack(fill='both', expand=True, padx=10)
        for index, (path, project) in enumerate(entries):
            active = [p for p in project.photos if not p.recycled]
            progress = f'{sum(p.status != "Pending" for p in active)} / {len(active)}'
            updated = time.strftime('%Y-%m-%d %H:%M', time.localtime(path.stat().st_mtime))
            listing.insert('', 'end', iid=str(index), text=project.source, values=(progress, updated))
        def load():
            chosen = listing.selection()
            if not chosen:
                return
            path = entries[int(chosen[0])][0]
            window.destroy()
            if not self.before_replace():
                return
            try:
                project = Project.load(path)
                project.scan()
                self.activate(project, path, saved_file=project.finish_summary.get('project_file'))
            except Exception as exc:
                messagebox.showerror('Recovery', str(exc))
        ttk.Button(window, text='Continue selected review', command=load).pack(pady=10)

    def schedule_preferences(self, *_):
        if self.loading_preferences:
            return
        if self.preference_timer:
            self.root.after_cancel(self.preference_timer)
        self.preference_timer = self.root.after(400, lambda: self.update(silent=True))

    def configuration_changed(self, _=None):
        if self.project and not self.loading_preferences:
            self.refresh()

    def config_changed(self):
        return self.project is not None and (self.shot.get(), self.motion.get(), self.recursive.get()) != (
            self.project.shot_type, self.project.motion_intent, self.project.recursive)

    def update(self, silent=False):
        if self.preference_timer is not None:
            self.root.after_cancel(self.preference_timer)
        self.preference_timer = None
        if not self.project or self.busy:
            return False
        old = self.project.target, self.project.style
        try:
            self.project.target = self.settings()
            self.project.style = self.style.get()
            self.project.selection()
            self.preferences_valid = True
            self.selection_hint.set('Blank = all suitable photos · selection updates automatically')
            if old != (self.project.target, self.project.style):
                self.checkpoint()
            self.refresh()
            return True
        except Exception as exc:
            self.project.target, self.project.style = old
            self.preferences_valid = False
            self.selection_hint.set(str(exc))
            self.sync_controls()
            if not silent:
                messagebox.showerror('Selection', str(exc))
            return False

    def sync_controls(self):
        ready = self.project is not None and not self.busy
        pending = self.project and any(p.status == 'Pending' and not p.recycled for p in self.project.photos)
        analyzed = self.project and any(p.status in ('Keep', 'Review', 'Error') for p in self.project.photos if not p.recycled)
        changed = self.config_changed()
        label = 'Reanalyze photos' if changed and analyzed else 'Resume analysis' if pending and analyzed else 'Analyze photos' if pending else 'Check for new photos'
        self.action_buttons['analyze'].configure(text=label if self.project else 'Analyze photos', state='normal' if ready else 'disabled')
        self.action_buttons['pause'].configure(state='normal' if self.busy else 'disabled')
        self.action_buttons['cleanup'].configure(state='normal' if ready and not pending and not changed and self.preferences_valid else 'disabled')
        self.action_buttons['finish'].configure(state='normal' if ready else 'disabled')
        self.folder_button.configure(state='disabled' if self.busy else 'normal')
        self.count_control.configure(state='disabled' if self.busy else 'normal')
        self.recursive_control.configure(state='disabled' if self.busy else 'normal')
        for control in (self.shot_control, self.motion_control, self.style_control):
            control.configure(state='disabled' if self.busy else 'readonly')
        for index in (0, 1):
            self.project_menu.entryconfigure(index, state='disabled' if self.busy else 'normal')
        for index in (3, 4):
            self.project_menu.entryconfigure(index, state='normal' if ready else 'disabled')
        photo = self.current() if self.project else None
        for button in self.photo_buttons:
            button.configure(state='normal' if ready and photo and not photo.recycled else 'disabled')
        self.protected.set(bool(photo and photo.must_include))
        self.protection_control.configure(state='normal' if ready and photo and photo.decision == 'Keep' else 'disabled')
        if not self.project:
            stage = 'Choose folder'
        elif self.cleaning:
            stage = 'Review & clean up — cleanup in progress'
        elif self.busy or pending or changed:
            stage = 'Analyze'
        elif not self.dirty and self.saved_project_file:
            stage = 'Project saved — continue reviewing or start another folder'
        else:
            stage = 'Review & clean up — then Finish project'
        self.workflow.set(f'Choose folder  →  Analyze  →  Review & clean up  →  Finish & save     |     {stage}')

    def analyze(self):
        if not self.idle() or not self.project:
            return
        if not self.update():
            return
        changed_intent = (self.shot.get(), self.motion.get()) != (self.project.shot_type, self.project.motion_intent)
        if changed_intent and any(p.status in ('Keep', 'Review', 'Error') for p in self.project.photos if not p.recycled):
            if not messagebox.askyesno('Reassess with new settings?', 'The shot or motion intent has changed. Recalculate quality scores? Your manual decisions and protected photos will be preserved.'):
                return
            for photo in self.project.photos:
                if not photo.recycled and photo.status in ('Keep', 'Review', 'Error'):
                    photo.status, photo.score, photo.reasons = 'Pending', 0.0, []
                    photo.signature = photo.group = ''
        self.project.shot_type, self.project.motion_intent = self.shot.get(), self.motion.get()
        self.project.recursive = self.recursive.get()
        self.checkpoint()
        self.busy = True
        self.session_outcome = None
        self.analysis_stage = 'Starting analysis worker. Photos are processed locally.'
        self.stage_started = time.monotonic()
        self.update_progress()
        try:
            self.session = AnalysisSession(self.project)
        except Exception as exc:
            self.busy = False
            messagebox.showerror('Start analysis', str(exc))

    def pause(self):
        if self.cleaning:
            self.cleanup_stop.set()
            self.status.set('Stopping cleanup after the current Recycle Bin operation…')
            return
        if self.busy and self.session:
            self.session.pause()
            self.status.set('Stopping analysis now. Completed results are preserved; the interrupted photo will be retried.')

    def poll(self):
        if self.cleaning:
            self.poll_cleanup()
            if self.close_requested and not self.cleaning:
                self.close_requested = False
                self.close()
                return
        if self.session:
            try:
                self.consume(self.session.drain())
                if self.session_outcome is not None or self.session.eof or not self.session.alive:
                    # Drain final messages after exit before releasing the pipe.
                    self.consume(self.session.drain())
                    pending = sum(p.status == 'Pending' and not p.recycled for p in self.project.photos)
                    if self.session_outcome and self.session_outcome[0] == 'done' and pending:
                        self.session_outcome = ('error', f'The worker stopped with {pending} photos still pending. Completed results are saved; click Analyze / resume to retry.')
                    cancelled = self.session.cancelled
                    self.session.close()
                    self.session = None
                    self.busy = False
                    self.checkpoint()
                    self.refresh()
                    if self.session_outcome and self.session_outcome[0] == 'error':
                        messagebox.showerror('Analysis', self.session_outcome[1])
                    elif cancelled:
                        self.status.set('Paused. Resume will retry the interrupted photo. ' + self.status.get())
                    elif self.session_outcome:
                        self.status.set(self.session_outcome[1] + ' ' + self.status.get())
                    else:
                        messagebox.showerror('Analysis stopped', 'The worker exited unexpectedly. Completed results are saved; click Analyze / resume to retry.')
                    if self.close_requested:
                        self.close_requested = False
                        self.close()
                        return
                else:
                    self.update_progress()
            except Exception as exc:
                if self.session:
                    self.session.close()
                    self.session = None
                self.busy = False
                self.close_requested = False
                messagebox.showerror('Save analysis results', str(exc))
        self.root.after(150, self.poll)

    def consume(self, messages):
        changed = False
        updated = set()
        index = {photo.relative: i for i, photo in enumerate(self.project.photos)}
        for event, detail in messages:
            if event == 'inventory':
                self.project.photos = [Photo(**photo) for photo in detail]
                index = {photo.relative: i for i, photo in enumerate(self.project.photos)}
                changed = True
            elif event == 'photo':
                photo = Photo(**detail)
                if photo.relative in index:
                    self.project.photos[index[photo.relative]] = photo
                    updated.add(photo.relative)
                    changed = True
            elif event in ('done', 'error'):
                self.session_outcome = (event, detail)
            elif event == 'stage' and self.session and not self.session.cancelled:
                self.analysis_stage = detail
                self.stage_started = time.monotonic()
        if changed:
            self.checkpoint()
            self.refresh()
            selected = self.current()
            if selected and selected.relative in updated:
                self.preview()

    def update_progress(self):
        self.sync_controls()
        active = [p for p in self.project.photos if not p.recycled]
        completed = sum(p.status != 'Pending' for p in active)
        self.bar.configure(maximum=max(1, len(active)), value=completed)
        if self.session and self.session.cancelled:
            self.status.set(f'{completed}/{len(active)} processed. Stopping analysis; completed results are saved.')
            return
        stage = self.analysis_stage
        if active and completed == len(active):
            stage = 'Finalizing results…'
        elapsed = int(time.monotonic() - self.stage_started)
        suffix = f' ({elapsed}s on this step; Pause is available)' if elapsed >= 15 else ''
        self.status.set(f'{completed}/{len(active)} processed · {len(active) - completed} pending. {stage}{suffix}')

    def refresh(self):
        if not self.project or self.cleaning:
            return
        self.sync_controls()
        try:
            chosen = {p.relative for p in self.project.selection()}
            issue = ''
        except ValueError as exc:
            chosen, issue = set(), str(exc)
        for name, value in [('Photos', sum(not p.recycled for p in self.project.photos)), ('Keep', sum(p.decision == 'Keep' for p in self.project.photos)), ('Review', sum(p.decision in ('Review', 'Error', 'Unsupported') for p in self.project.photos)), ('Selected', len(chosen))]:
            self.metrics[name].set(str(value))
        previous = self.table.selection()
        desired = []
        keeps = sum(p.decision == 'Keep' for p in self.project.photos)
        reviews = sum(p.decision in ('Review', 'Error', 'Unsupported') for p in self.project.photos)
        for i, photo in enumerate(self.project.photos):
            view = self.view.get()
            show = (view == 'All' or view == photo.decision or
                    view == 'Marked for removal' and photo.decision == 'Excluded' or
                    view == 'Review' and photo.decision in ('Error', 'Unsupported') or
                    view == 'Final selection' and photo.relative in chosen or
                    view == 'Other Keep' and photo.decision == 'Keep' and photo.relative not in chosen)
            if show:
                iid = str(i)
                desired.append(iid)
                decision = 'Marked for removal' if photo.decision == 'Excluded' else photo.decision
                score = f'{photo.score:.1f}' if photo.status in ('Keep', 'Review') else '—'
                values = (decision, score, 'Yes' if photo.must_include else 'No')
                if self.table.exists(iid):
                    self.table.item(iid, text=photo.relative, values=values, tags=(photo.decision,))
                else:
                    self.table.insert('', 'end', iid=iid, text=photo.relative, values=values, tags=(photo.decision,))
        desired_set = set(desired)
        for iid in self.table.get_children():
            if iid not in desired_set:
                self.table.delete(iid)
        if tuple(desired) != self.table.get_children():
            self.table.set_children('', *desired)
        if previous and self.table.exists(previous[0]) and self.table.selection() != previous:
            self.table.selection_set(previous[0])
            self.table.see(previous[0])
        if self.busy:
            self.update_progress()
            return
        active_photos = [p for p in self.project.photos if not p.recycled]
        done = sum(p.status != 'Pending' for p in active_photos)
        self.bar.configure(maximum=max(1, len(active_photos)), value=done)
        selected = f'{len(chosen)} suitable photos' if self.project.target is None else f'{len(chosen)}/{self.project.target} selected'
        short = self.project.target is not None and len(chosen) < self.project.target
        active = sum(not p.recycled for p in self.project.photos)
        pending = any(p.status == 'Pending' and not p.recycled for p in self.project.photos)
        self.status.set(issue or f'{active} source photos · {keeps} Keep · {reviews} Review · {selected}. ' +
                        ('Preferences changed — analyze to apply them.' if self.config_changed() else
                         'Analysis pending.' if pending else 'More approved photos needed.' if short else 'Review cleanup or choose Finish project.'))

    def current(self):
        selection = self.table.selection()
        if not selection or not self.project:
            return None
        index = int(selection[0])
        return self.project.photos[index] if index < len(self.project.photos) else None

    def preview(self, _=None):
        if self.cleaning:
            return
        photo = self.current()
        self.sync_controls()
        if not photo:
            return
        if photo.recycled:
            self.preview_label.configure(image='', text='Moved to the Recycle Bin. Restore it there, then reopen this project.')
            self.details.set(photo.relative)
            return
        score = f'{photo.score:.1f}' if photo.status in ('Keep', 'Review') else 'Not assessed'
        self.details.set(f'{photo.relative}\nAI: {photo.status} · Score: {score}\nDecision: {photo.decision}\n' + '\n'.join(photo.reasons))
        try:
            with Image.open(self.project.path_for(photo)) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
                image.thumbnail((max(32, self.preview_label.winfo_width() - 16), max(32, self.preview_label.winfo_height() - 16)))
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_image, text='')
        except Exception as exc:
            self.preview_label.configure(image='', text=f'Preview unavailable: {exc}')

    def decide(self, decision):
        if not self.idle():
            return
        photo = self.current()
        if photo:
            if photo.recycled:
                messagebox.showinfo('Recycled photo', 'Restore this file from the Recycle Bin before changing its decision.')
                return
            if decision == 'Excluded' and photo.must_include:
                messagebox.showinfo('Protected photo', 'Clear Protect this photo before marking it for removal.')
                return
            if decision == 'Keep' and photo.status not in ('Keep', 'Review'):
                messagebox.showinfo('Cannot keep this file', 'Only successfully assessed photos can enter the selection.')
                return
            photo.override = decision
            if photo.decision != 'Keep':
                photo.must_include = False
            self.checkpoint()
            self.refresh()

    def pin(self):
        if not self.idle():
            return
        photo = self.current()
        if photo and photo.decision == 'Keep':
            photo.must_include = not photo.must_include
            self.checkpoint()
            self.refresh()

    def zoom(self):
        if self.busy:
            return
        photo = self.current()
        if not photo:
            return
        try:
            with Image.open(self.project.path_for(photo)) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
            window = tk.Toplevel(self.root)
            window.title(f'100% — {photo.relative}')
            window.geometry('950x700')
            canvas = tk.Canvas(window)
            horizontal = ttk.Scrollbar(window, orient='horizontal', command=canvas.xview)
            vertical = ttk.Scrollbar(window, orient='vertical', command=canvas.yview)
            horizontal.pack(side='bottom', fill='x')
            vertical.pack(side='right', fill='y')
            canvas.pack(fill='both', expand=True)
            canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set, scrollregion=(0, 0, image.width, image.height))
            canvas.photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor='nw', image=canvas.photo)
        except Exception as exc:
            messagebox.showerror('Inspect photo', str(exc))

    def export(self):
        if not self.idle() or not self.project:
            return
        try:
            if not self.update():
                return
            if any(p.status == 'Pending' and not p.recycled for p in self.project.photos):
                raise ValueError('Finish analysis before exporting the final selection.')
            selected = self.project.selection()
            if self.project.target is not None and len(selected) < self.project.target and not messagebox.askyesno('Selection shortfall', f'Only {len(selected)} approved photos are available for a target of {self.project.target}. Export this smaller set?'):
                return
            parent = filedialog.askdirectory(title='Choose export parent folder (outside source)')
            if not parent:
                return
            from datetime import datetime
            destination = Path(parent) / datetime.now().strftime('IQA-selection-%Y%m%d-%H%M%S-%f')
            count = self.project.export(destination)
            self.project.save(self.project_file)
            messagebox.showinfo('Export complete', f'Copied {count} originals to:\n{destination}')
        except Exception as exc:
            messagebox.showerror('Export', str(exc))

    def cleanup(self):
        if not self.idle() or not self.project or not self.update():
            return
        if self.config_changed():
            messagebox.showinfo('Apply preferences', 'Analyze with the changed preferences before cleanup.')
            return
        if any(p.status == 'Pending' and not p.recycled for p in self.project.photos):
            messagebox.showinfo('Analysis pending', 'Finish analysis before reviewing folder cleanup.')
            return
        candidates = self.project.cleanup_candidates()
        if not candidates:
            messagebox.showinfo('Folder cleanup', 'No cleanup suggestions or manually marked photos are available.')
            return
        window = tk.Toplevel(self.root)
        window.title('Review source-folder cleanup')
        window.geometry('1100x650')
        window.transient(self.root)
        window.grab_set()
        ttk.Label(window, text=f'Source: {self.project.source}\nSelect the exact photos to remove. Nothing is selected automatically.', wraplength=1050, padding=10).pack(fill='x')
        body = ttk.Panedwindow(window, orient='horizontal')
        body.pack(fill='both', expand=True, padx=10)
        left, right = ttk.Frame(body), ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)
        listing = ttk.Treeview(left, columns=('reason',), selectmode='extended')
        listing.heading('#0', text='Photo (Ctrl/Shift click to select multiple)')
        listing.heading('reason', text='Why suggested')
        listing.column('#0', width=310)
        listing.column('reason', width=200)
        scroll = ttk.Scrollbar(left, command=listing.yview)
        listing.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        listing.pack(fill='both', expand=True)
        preview = ttk.Label(right, anchor='center')
        preview.pack(fill='both', expand=True)
        reason = tk.StringVar(value='Inspect the suggestions, then select only files you want removed.')
        ttk.Label(right, textvariable=reason, wraplength=400).pack(fill='x')
        for i, photo in enumerate(candidates):
            why = 'Manually marked' if photo.override == 'Excluded' else 'Needs review' if photo.decision == 'Review' else 'Outside target selection'
            listing.insert('', 'end', iid=str(i), text=photo.relative, values=(why,))
        count = tk.StringVar(value=f'0 of {len(candidates)} files selected for removal')
        def inspect(_=None):
            chosen = listing.selection()
            count.set(f'{len(chosen)} of {len(candidates)} files selected for removal')
            if not chosen:
                return
            photo = candidates[int(chosen[0])]
            reason.set(f'{photo.relative}\nScore: {photo.score:.1f}\n' + '\n'.join(photo.reasons))
            try:
                with Image.open(self.project.path_for(photo)) as original:
                    image = ImageOps.exif_transpose(original).convert('RGB')
                    image.thumbnail((400, 420))
                preview.photo = ImageTk.PhotoImage(image)
                preview.configure(image=preview.photo, text='')
            except Exception as exc:
                preview.configure(image='', text=f'Preview unavailable: {exc}')
        listing.bind('<<TreeviewSelect>>', inspect)
        def confirm():
            approved = [candidates[int(i)] for i in listing.selection()]
            if not approved:
                messagebox.showinfo('Select photos', 'Select the exact files you want to remove.', parent=window)
                return
            if not messagebox.askyesno('Confirm source-folder removal',
                    f'Move these {len(approved)} selected photos from:\n{self.project.source}\n\nto the Windows Recycle Bin?\nThe highlighted list contains the exact files. Restore them using the Recycle Bin if needed.',
                    parent=window, default='no'):
                return
            try:
                for photo in approved:
                    self.project.checked_cleanup_path(photo)
                for photo in approved:
                    photo.override = 'Excluded'
                self.checkpoint()
            except Exception as exc:
                messagebox.showerror('Cleanup not started', str(exc), parent=window)
                return
            window.destroy()
            self.busy = self.cleaning = True
            self.sync_controls()
            self.cleanup_stop.clear()
            self.status.set('Moving confirmed files to the Recycle Bin…')
            threading.Thread(target=self.cleanup_worker, args=([p.relative for p in approved],), daemon=True).start()
        controls = ttk.Frame(window, padding=10)
        controls.pack(fill='x')
        ttk.Label(controls, textvariable=count).pack(side='left')
        ttk.Button(controls, text='Select all listed', command=lambda: listing.selection_set(listing.get_children())).pack(side='left', padx=5)
        ttk.Button(controls, text='Clear selection', command=lambda: listing.selection_remove(listing.selection())).pack(side='left')
        ttk.Button(controls, text='Move selected to Recycle Bin…', command=confirm).pack(side='right')

    def cleanup_worker(self, approved):
        try:
            report = self.project.recycle(approved, self.project_file, stop=self.cleanup_stop,
                    progress=lambda done, total: self.cleanup_events.put(('progress', (done, total))))
            self.cleanup_events.put(('done', report))
        except Exception as exc:
            self.cleanup_events.put(('error', str(exc)))

    def poll_cleanup(self):
        try:
            while True:
                event, detail = self.cleanup_events.get_nowait()
                if event == 'progress':
                    self.status.set(f'Moved {detail[0]}/{detail[1]} confirmed files to the Recycle Bin.')
                else:
                    self.busy = self.cleaning = False
                    self.refresh()
                    if event == 'error':
                        messagebox.showerror('Cleanup stopped', detail)
                    else:
                        result = f'{len(detail["recycled"])} photos moved to the Recycle Bin. Kept photos remain in the source folder.'
                        if detail['errors']:
                            messagebox.showerror('Cleanup stopped', result + '\n\n' + '\n'.join(f'{name}: {error}' for name, error in detail['errors'].items()))
                        elif not self.close_requested:
                            messagebox.showinfo('Cleanup paused' if detail['cancelled'] else 'Cleanup complete', result)
        except queue.Empty:
            pass

    def close(self):
        if self.busy:
            self.close_requested = True
            self.pause()
        else:
            if self.project and self.dirty:
                answer = messagebox.askyesnocancel('Save project before closing?',
                        'Save your review to a project file?\n\nYes: choose a save location.\nNo: keep an automatic recovery copy.\nCancel: continue reviewing.')
                if answer is None:
                    return
                if answer:
                    self.finish(close_after=True)
                    return
            self.root.destroy()

    def finish(self, close_after=False, show_summary=True):
        if not self.idle() or not self.project or not self.update():
            return False
        if self.config_changed():
            if any(p.status in ('Keep', 'Review', 'Error') and not p.recycled for p in self.project.photos):
                messagebox.showinfo('Apply preferences', 'Analyze with the changed preferences, or restore the previous preferences, before finishing.')
                return False
            self.project.shot_type, self.project.motion_intent = self.shot.get(), self.motion.get()
            self.project.recursive = self.recursive.get()
            self.checkpoint()
        try:
            before = [(p.relative, p.size, p.modified, p.status, p.recycled) for p in self.project.photos]
            self.project.scan()
            after = [(p.relative, p.size, p.modified, p.status, p.recycled) for p in self.project.photos]
            if before != after:
                self.checkpoint()
            self.refresh()
            current = self.saved_project_file
            path = filedialog.asksaveasfilename(
                title='Finish — save project', defaultextension='.iqa.json',
                initialfile=current.name if current else Path(self.project.source).name + '.iqa.json',
                initialdir=str(current.parent) if current else None,
                filetypes=[('IQA project', '*.iqa.json')])
            if not path:
                return False
            summary = self.project.finish(path)
            self.saved_project_file = Path(path).resolve()
            self.dirty = False
            try:
                self.project.save(self.project_file)
            except OSError as exc:
                messagebox.showwarning('Project saved', f'Your project was saved successfully, but the private recovery copy could not be updated:\n{exc}')
            self.refresh()
            self.status.set(f'Project saved: {self.saved_project_file}')
            if close_after:
                self.root.destroy()
            elif show_summary:
                self.show_finish_summary(summary)
            return True
        except Exception as exc:
            messagebox.showerror('Project not saved', str(exc))
            return False

    def show_finish_summary(self, summary):
        window = tk.Toplevel(self.root)
        window.title('Project finished' if not summary['pending'] else 'Project saved for later')
        window.geometry('650x440')
        window.transient(self.root)
        window.grab_set()
        heading = 'Project finished and saved' if not summary['pending'] else 'Progress saved — ready to resume later'
        ttk.Label(window, text=heading, font=('Segoe UI', 15, 'bold'), padding=15).pack(fill='x')
        target = f' (target: {summary["target"]})' if summary['target'] is not None else ''
        rows = [f'Photos remaining in the original folder: {summary["remaining"]}',
                f'Photos in your proposed selection: {summary["selected"]}{target}',
                f'Photos moved to the Recycle Bin: {summary["recycled"]}',
                f'Review photos left in place: {summary["review"]}',
                f'Marked photos not removed: {summary["marked"]}',
                f'Photos awaiting analysis: {summary["pending"]}']
        ttk.Label(window, text='\n'.join(rows), padding=15, justify='left').pack(fill='x')
        ttk.Label(window, text=f'Project file: {self.saved_project_file}\nPhoto folder: {self.project.source}', wraplength=610, padding=15).pack(fill='x')
        ttk.Label(window, text='Unconfirmed photos remain in place. The project file stores your decisions, not copies of the photos.', wraplength=610, padding=10).pack(fill='x')
        controls = ttk.Frame(window, padding=10)
        controls.pack(fill='x')
        def another():
            window.destroy()
            self.choose_folder()
        def done():
            window.destroy()
            self.root.destroy()
        ttk.Button(controls, text='Open photo folder', command=self.open_photo_folder).pack(side='left')
        ttk.Button(controls, text='Continue reviewing', command=window.destroy).pack(side='left', padx=5)
        ttk.Button(controls, text='Another folder', command=another).pack(side='left')
        ttk.Button(controls, text='Done', command=done).pack(side='right')

    def open_photo_folder(self):
        try:
            if os.name != 'nt':
                raise OSError('Open the photo folder using your file manager.')
            os.startfile(self.project.source)
        except OSError as exc:
            messagebox.showerror('Open photo folder', str(exc))


def main(smoke_test=False, root=None, splash=None):
    root = root or tk.Tk()
    root.withdraw()
    if smoke_test:
        DesktopApp(root)
        root.update_idletasks()
        from src.pipeline import PipelineConfig
        PipelineConfig()
        import sys
        if sys.platform == 'win32':
            import pythoncom
            from win32com.shell import shell
            from send2trash.win.IFileOperationProgressSink import FileOperationProgressSink
        # Exercise frozen multiprocessing as well as imports and Tk startup.
        import tempfile
        import time
        with tempfile.TemporaryDirectory(prefix='iqa-startup-') as folder:
            session = AnalysisSession(Project(folder))
            messages = []
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    messages.extend(session.drain())
                    if any(event in ('done', 'error') for event, detail in messages):
                        break
                    time.sleep(0.05)
                errors = [detail for event, detail in messages if event == 'error']
                if errors or not any(event == 'done' for event, detail in messages):
                    raise RuntimeError(f'Analysis worker startup failed: {errors or messages}')
            finally:
                session.close()
        root.update()
        root.destroy()
        return
    from .welcome import Preferences
    from .startup import prepare_workspace, show_workspace
    startup = {}
    def read_preferences():
        startup['introduction'] = not Preferences().completed()
    def build_workspace():
        startup['app'] = DesktopApp(root)
    def ready():
        if startup['introduction']:
            root.after(200, startup['app'].introduction)
    def failed(exc):
        startup['error'] = exc
        root.quit()
    prepare_workspace(root, splash, [
        ('Reading your preferences…', 25, read_preferences),
        ('Preparing review controls…', 55, build_workspace),
        ('Applying workspace layout…', 85, root.update_idletasks),
    ], lambda: show_workspace(root, splash, ready), failed)
    root.mainloop()
    if 'error' in startup:
        raise startup['error']
