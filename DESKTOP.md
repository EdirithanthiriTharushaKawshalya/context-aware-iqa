# Context Aware IQA — Desktop photo selection

The native desktop application is separate from `app.py` and `src/ui/`.
The Streamlit website retains its existing layout and research workflow, with
the shared Context Aware IQA name and logo. Both
interfaces reuse `src.pipeline.Phase4Pipeline`.

Version 0.4.0 introduces a violet-accented desktop workspace, live collection
counts, colored review decisions, contextual help, and a dark photo viewer.
The shared logo stays in `src/images/icon.png`; `icon.ico` supplies Windows
executable and installer icons. The native presentation layer lives in
`src/desktop/interface.py`, separate from assessment and project persistence.

Version 0.5.1 keeps the startup splash visible for at least five seconds. This
includes time spent preparing the workspace; slower startups add no extra wait.

Version 0.5.0 adds a branded startup splash and a four-card introduction on
first launch. Next, Back, Skip, and Get started are available; completing or
skipping the guide is remembered per Windows user. Replay it from
**Help → Quick start introduction**. Preferences live beside Sessions in
`%LOCALAPPDATA%/ContextAwareIQA/preferences.json`.

The desktop uses rounded sections, pill buttons, softened dropdown fields and
progress indicators. Photo inspection has a light background with the
assessment beside the image, without cropping the photo. The installer writes
an explicit logo icon for both desktop and Start menu shortcuts.

## Run

In your existing Python environment:

```powershell
python desktop_app.py
```

Tkinter is included in the standard Windows Python installer. Pillow, NumPy,
PyTorch and torchvision are required by the shared engine. The separate
`requirements-desktop.txt` also includes the optional distribution build tool.

1. **Choose folder…** creates a working session and inventories photos
   automatically. No initial save dialog or separate scan action is needed.
2. Leave the final count **blank to keep all suitable photos**, or enter a
   positive count for a limited selection. Count and selection-style changes
   update the proposal automatically; there is no Update selection button.
   Set shot and motion intent before analysis. Use `auto` for a mixed event.
3. Select **Analyze photos** (or **Resume analysis** for a partly analyzed session).
   Analysis runs in a separate worker process.
   **Pause** cancels the active worker, including a slow photo or model startup,
   without waiting for inference to finish. Completed results are saved by the
   desktop parent; the interrupted photo stays pending and is retried on resume.
   Resume starts a new worker and reloads models, so it still has startup overhead.
   Pausing an initial model download may require downloading it again. Reopening a
   project rescans its source and reuses unchanged assessments. Changed files
   lose stale assessments and overrides and are analyzed again.
4. Browse All, Keep, Review, Final selection and Other Keep. Review includes
   uncertain images, decoding errors and unsupported files. Inspect previews or
   use the scrollable 100% view. Keep/Review override the AI; **Mark removal**
   flags a photo without changing the source file yet. Reset restores the AI decision. A successfully
   assessed Review photo can be manually approved. **Protect this photo — always
   include it** is an optional checkbox for essential moments. Keep makes a photo
   eligible; protection also guarantees inclusion in a limited selection and
   prevents removal. Too many protected photos for a target produce a clear
   selection error. Protection is available for Keep photos.
5. Select **Review cleanup** after analysis finishes. It suggests successfully
   assessed Review photos, manually marked photos, and (when a count is supplied)
   suitable alternatives outside the requested selection. Nothing is selected
   automatically. Ctrl/Shift click individual rows, or deliberately choose
   **Select all listed**. Inspect the photos and click **Move selected to Recycle
   Bin**, then confirm the exact highlighted selection and source folder.
   Confirmed files are moved out of the original folder into the Windows Recycle
   Bin. Kept and unconfirmed files stay in place. Restore removed files through
   the Windows Recycle Bin, then reopen the project. Recycled files are shown in
   the Recycled view; restoration is detected on the next scan. Decisions can
   then be changed using Keep/Review/Reset.
6. Choose **Finish project…**, select where to save the `.iqa.json` file outside
   the photo folder, and review the completion summary. It shows photos remaining,
   photos selected, recycled photos, and outstanding Review/marked/pending photos.
   **Open photo folder** opens the remaining collection. **Done** closes the saved
   project and application. You can also continue reviewing or start another folder.
   A partially analyzed session is labeled **saved for later**, not completed.
7. To return another day, use **Project → Open saved project…**. Scores and manual
   decisions are reused for unchanged files. Changing shot or motion intent after
   assessment requires reanalysis confirmation; manual decisions are preserved.

## Recovery and saving

While you work, private recovery checkpoints are stored automatically in
`%LOCALAPPDATA%/ContextAwareIQA/Sessions` on Windows. These protect progress and
the source-cleanup audit before Finish, without asking you for a project path
at the start. The checkpoint directory must be outside the chosen photo folder.
Use **Project → Recover unfinished session…** after an interruption; sessions are
listed with their folder, processing progress and save time.

Opening a saved project creates a private working copy. Edits update recovery;
the named project file is updated when you use Finish again. Closing or switching
away from an unsaved review offers saving, keeping a recovery copy, or canceling.
Finishing never removes additional images: marked but unconfirmed photos remain
in the original folder. Recovery copies contain metadata and decisions, not
backups of image files, and do not undo confirmed Recycle Bin operations.

Cleanup is the primary workflow; it does not require copying photos elsewhere.
**Copy selection…** remains available as an optional convenience. It creates a
new folder containing `photos/` and `_selection.json`. An optional count is a
selection target, not authorization to remove every other file. Unconfirmed
photos remain in their source folder even if that leaves more than the target.

Balanced mode first chooses the highest-scoring approved representative of each
coarse visual-similarity group, then fills remaining slots by score. Strict quality
uses score order. Neither automatically promotes Review photos. Must-include
photos take precedence in both modes. This heuristic reduces repetition; it does
not establish event chronology, identify important people, or guarantee coverage
of meaningful moments. Photographers should inspect the final selection.

## Privacy and limitations

Photos are processed locally. The existing model loader may download pretrained
weights on first use; provision its caches beforehand for offline use. Detection
can degrade if models are unavailable; inspect the assessment reasons. Decisions
are heuristic and inherit the research engine's documented limitations in
`PHOTO_ASSESSMENT.md`; this is not a validated commercial culling model.

This release decodes JPEG, PNG, TIFF, BMP and WebP through Pillow. RAW extensions
are inventoried as unsupported; RAW decoding and RAW/JPEG pairing are future work.
Other file types are ignored. Analysis is sequential and full-resolution; large
collections require runtime and memory validation. The table stores a row per
photo, project persistence currently rewrites JSON per photo, and
similarity matching is quadratic. SQLite, indexed similarity, thumbnail caching
and background export are needed for production-scale collections. Export runs
synchronously and may temporarily occupy the interface. A failed copy can leave
a partial export folder. Cleanup runs in a background thread and can be stopped
between individual Recycle Bin operations using Pause. It never invokes permanent
deletion as a fallback. Windows may refuse to recycle some locations or files;
cleanup stops and reports that failure. Successfully recycled files remain
recoverable through the Recycle Bin until it is emptied or Windows removes them.
The process inherits the user's OS permissions; analysis reads sources, and
source-folder cleanup requires explicit per-batch approval.

Must-include photos are protected. Unreadable and unsupported files are not
automatically suggested for removal; they require manual marking. Cleanup
validates source containment, rejects redirected paths, and checks size and
modification time before each move. An audit trail in the saved project records
the approved files and outcomes, including partial failures. The working project
checkpoint is automatically saved outside the source folder. If saving the initial audit fails, cleanup does
not begin. Sidecars and RAW/JPEG companions are not removed automatically.

Projects contain source paths, analysis and decisions, not copies of originals.
Keep the source folder in place. Copy SD-card contents to local storage first.
Training data is never updated by the desktop application's decisions.

## Build a Windows distribution

```powershell
python -m pip install -r requirements-desktop.txt
python -m PyInstaller desktop.spec
```

The result is a folder distribution at `dist/ContextAwareIQA/`; distribute the
whole folder, not only its executable. Compile `installer/windows.iss` with
Inno Setup to produce a per-user installer and shortcuts. Packaging configuration
is provided; a clean-machine installation and model-cache check are required
before releasing it. Code signing and automatic updates are not included.

The installer output is `dist/installer/ContextAwareIQA-Setup.exe`.
The bundled application can check its Tk interface and assessment imports using
`ContextAwareIQA.exe --smoke-test`. It exits automatically; failures return exit
code 1 and write `%TEMP%/context-aware-iqa-startup.log`. This check does not run
model inference or establish photo-selection accuracy.

## Verify

```powershell
python -m pytest tests/test_desktop_project.py -q
```

Tests cover inventory and cache invalidation, paused/resumed analysis, corrupt
files, human overrides, pins, selection shortfalls, source containment and
non-destructive exports, without downloading models.

Version 0.1.1 adds real-process pause tests with deliberately blocked model
startup and photo inference. Run them with
`python -m pytest tests/test_desktop_analysis.py -q`.

Version 0.2.0 adds optional counts and confirmed source cleanup. Run
`python -m pytest tests/test_desktop_cleanup.py -q` for count selection,
approval, path, audit, cancellation and partial-failure checks. Unit tests use a
fake recycle destination and do not touch user photos. The native Windows
integration was separately checked with an application-generated disposable file.
Windows integration uses [IFileOperation recycle flags](https://learn.microsoft.com/en-us/windows/win32/api/shobjidl_core/nf-shobjidl_core-ifileoperation-setoperationflags)
and a progress callback that refuses non-recycling deletion.

Version 0.2.1 updates decisions, scores and filtered views as each assessment
arrives. Pending or unassessed files display a dash instead of a misleading zero.
Must include is a manual pin and displays No until you pin an approved photo.
Progress reports processed/pending counts and elapsed time on long steps. A
worker's completion message finishes the interface without waiting for slow
machine-learning library shutdown; pending files cannot be labeled complete.
Run `python -m pytest tests/test_desktop_progress.py -q` for the real Tk/worker
regression tests (requires Tk and a desktop session).

Version 0.3.0 replaces the setup and finishing workflow: automatic folder
inventory and recovery, automatic selection updates, explained protection,
Finish/save/summary/Done, and Open/Recover commands in the Project menu. Run
`python -m pytest tests/test_desktop_workflow.py -q` for workflow and data-safety
checks including canceled/failed saves, partial completion and recovery.
