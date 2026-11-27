# Context Aware IQA — Desktop photo selection

The native desktop application is separate from `app.py` and `src/ui/`.
The Streamlit website retains its existing layout and research workflow, with
the shared Context Aware IQA name and logo. Both
interfaces reuse `src.pipeline.Phase4Pipeline`.

The desktop appearance refresh adds a split startup card with original landscape
artwork, a short product description, and live status messages at the bottom.
The startup card has no progress bar. The analysis progress indicator remains in
the workspace. The extra Photo Review / Desktop label and numbered workflow strip
have been removed.

First launch opens a four-step tour anchored to the photo folder, analysis
toolbar, photo library, and finish action. The workspace is dimmed around a
highlighted target. Use Next, Back, Skip tour, or Escape; finishing or skipping is
remembered. Replay it with **Help > Workspace tour**.

Use **Edit > Preferences** (Ctrl+,) to switch between Light and Dark. Changes
apply immediately to the workspace and photo gallery and are remembered per user;
they do not change the active project or its photo decisions. If preferences
cannot be saved, the dialog reports that the theme is applied for this session.
The themed menu bar groups existing commands under **File, Edit, Photo, View,
Review, Help**. Use File to open/recover/save/copy projects, Photo for decisions
and inspection, and Review for analysis, burst comparison, and finishing.
Ctrl+O opens a folder, Ctrl+Shift+O opens a saved project, and Ctrl+S saves it.

Version 0.9.1 extends the theme to the window caption and application menus.
The caption keeps Windows' own minimize, maximize, close, and resize behavior;
its colors use the documented [DWM window attributes](https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute)
where the Windows version supports them. Newly opened application dialogs also
receive the current theme. The menu strip and command popups follow both themes,
including hover, disabled commands, mouse switching, and keyboard navigation.
Use Alt plus the menu initial, or F10, then arrows, Enter, and Escape.

Compact buttons and fields now use transparent corner images matched to their
containing surface. Panel corners have a smaller radius. Scrollbars are slim,
arrowless controls with draggable thumbs and page scrolling, and library tabs
use an accent underline rather than raised borders. The preview script includes
the complete window caption and a screenshot of an open dark command menu.

Version 0.9.2 reduces work during photo review. The gallery retains unchanged
canvas cards and bounded Tk thumbnail images, and cancels queued thumbnails
outside the viewport. The details list updates only changed rows. Normal photo
previews and 100% inspection decode in background workers; rapid navigation
discards obsolete requests, and a bounded preview cache makes revisits immediate.
JPEG decoder reduction is used only for display previews. Full-resolution
inspection and ML assessment retain their original image resolution.

Folder scanning filters entries before stat calls and reuses Windows directory
metadata. Selection uses filename sets for explicit choices, duplicate comparison
reuses bounded, immutable decoded fingerprint arrays, and recovery saves avoid
recursively copying every fingerprint before JSON encoding. Recovery still writes
an atomic checkpoint; cleanup retains its per-file durable audit and validation.
Project fields and assessment decisions are unchanged.

Run `python scripts/benchmark_desktop.py --output outputs/desktop-performance.json`
to measure a disposable synthetic collection. Local median timings for 2,000
photos with 1,000 explicit choices, before/after this change, were:

| Operation | Before | After |
| --- | ---: | ---: |
| Selection | 304.206 ms | 0.747 ms |
| Recovery checkpoint | 75.908 ms | 32.112 ms |
| Unchanged collection refresh | 268.028 ms | 8.144 ms |
| Unchanged gallery render | 8.008 ms | 0.050 ms |
| Gallery scroll round-trip | 19.574 ms | 3.153 ms |

These are synthetic interface timings on the development machine, not ML
inference benchmarks or guaranteed speedups for every folder. Tests cover stale
preview rejection, UI responsiveness during decoding, cache reuse, orientation,
unchanged-row updates, and complete checkpoint round-trips.

Version 0.10.0 adds RAW decoding to folder analysis, gallery thumbnails, photo
previews, 100% inspection, duplicate comparison, and removal-review previews.
Supported extensions include Sony ARW/SR2/SRF, Canon CR2/CR3/CRW, Nikon NEF/NRW,
DNG, Fujifilm RAF, Olympus ORF, Panasonic RW2, Leica RWL, Pentax PEF, Samsung SRW,
and Minolta MRW. Actual camera/compression support depends on the bundled LibRaw
decoder; damaged or unsupported encodings produce a per-file Error with a reason.

Assessment and 100% inspection decode sensor data at full resolution into sRGB
with camera white balance and automatic brightening disabled. Small display
previews use half-size decoding; gallery thumbnails use the embedded camera
preview when available, with sensor decoding as a fallback. Embedded camera JPEGs
are never substituted for full-resolution RAW assessment. Camera previews can
look different from the neutral rendering. XMP/Lightroom edits are not applied.
Orientation is applied once, and capture time, subseconds, timezone, camera ID,
and shutter speed are retained when available. Rendering is versioned in project
records; old Unsupported RAW entries return to Pending on rescan while manual
choices and protection are preserved.

RAW files are opened read-only and no converted files are written into the source
folder. Export copies the selected original RAW bytes. RAW/JPEG companions are
separate collection entries; this release does not implement automatic pairing or
sidecar export/editing. Cleanup still requires confirmation of the exact original
filenames, and never implicitly removes companions or XMP files.

The installer bundles rawpy 0.28.0, LibRaw 0.22.2, ExifRead, and the native RAW
runtime DLLs. Third-party license files are included in the bundled distribution
metadata. Decoder documentation: [rawpy](https://letmaik.github.io/rawpy/api/rawpy.RawPy.html),
[render parameters](https://letmaik.github.io/rawpy/api/rawpy.Params.html), and
[ExifRead](https://github.com/ianare/exif-py).

For a packaged native-decoder check, run:
`ContextAwareIQA.exe --raw-smoke-test "photo.ARW" --raw-report "raw-check.json"`.
This verifies thumbnail, fitted, and full-resolution decoding, records capture
metadata and decoder versions, and checks that the source SHA-256 is unchanged.
The same flags work with `python desktop_app.py` for source verification.
Public Sony A7 III ARW, Nikon D3S NEF, and Canon 40D sRAW CR2 samples were used for
local verification. Sony sample: [metadata-extractor-images](https://github.com/drewnoakes/metadata-extractor-images/tree/main/arw);
Nikon/Canon samples: [rawpy tests](https://github.com/letmaik/rawpy/tree/main/test).
These sample files are kept outside Git and are not included in the installer.

Version 0.10.1 fixes mouse-wheel scrolling in the photo library. Wheel input is
routed by pointer location even when another control has keyboard focus; partial
Windows wheel deltas accumulate, and the details list and review-reason text also
scroll under the pointer. Dialog galleries use the same routing.

Double-click a photograph (or press Enter on a gallery card) to open it fitted
to the viewer window. Fit to window and 100% controls switch between the complete
composition and full-resolution inspection. The explicit Inspect at 100% action
still opens at actual pixels. Fit mode refits on window resize; loading remains
asynchronous and closing a viewer restores its parent dialog's modal state.

Selected photo now has a prominent Why this photo needs review section, separate
from measurements and RAW-processing notes. Actual review reasons are saved in
project records; manual Review decisions and approvals are identified explicitly.
Use `--review-example` with the desktop preview script to capture the reason card
and both viewer modes using synthetic display decisions.

Dark RAW subject detection and focus measurements use bounded illumination
compensation (at most a 4x gain). The original rendering still supplies contrast,
shadow/highlight clipping, and global exposure measurements. This reduces the
brightness dependence of edge measurements without changing JPEG feature values
or automatically approving dark images. Genuine blur and exposure problems can
still require Review. RAW assessment profiles advance to v2, so reopening a saved
project queues older RAW assessments for analysis while preserving human choices.
These are engineering corrections to the existing heuristic assessment, rather
than a new accuracy claim or a substitute for inspecting the photograph.

`python scripts/preview_desktop_design.py` opens a temporary preview and saves
splash, light/dark workspace, preferences, and tour screenshots under
`outputs/desktop-design-review/`. It uses copies of the bundled artwork as sample
photos and isolated preferences, then closes automatically.
Pass `--raw-sample "photo.ARW"` to use a copy of a camera file for a RAW display
check; those screenshots go under `outputs/raw-verification/desktop-preview/`.

The bundled image at `src/images/splash-landscape.png` was generated with the
built-in image generation tool. Its prompt was: “Create a premium fine-art
landscape photograph for the image half of a desktop photography application's
startup splash card. Portrait aspect ratio, misty layered mountains in Sri Lanka
at dawn, deep forest teal foreground with delicate grasses, soft lavender
mountains and warm peach sky, small glowing sun, cinematic natural light,
sophisticated restrained color grading, actual photographic texture and detail,
quiet contemplative mood. No text, no logos, no UI, no borders. This is an
original full bleed image asset, not a splash screen mockup.” The asset is
included in the PyInstaller bundle.

Version 0.8.0 adds exact-copy and similar-burst recommendations. File SHA-256
and orientation-normalized RGB pixel hashes identify exact file/visual copies,
including copies with different names or image metadata. Similar burst candidates
must pass perceptual-hash, structural, color and available subject-region checks.
Capture timestamps (including subseconds and time-zone offsets) limit near matches
to a three-second window. Different known cameras are not joined into a burst.
Without original capture times, matching requires nearby numbered filenames in
the same folder and stricter visual agreement. File modification times are never
treated as camera capture times. All members must agree, preventing similarity
chains from joining gradually changing scenes. Low-information flat images are
not grouped by approximate visual similarity; exact copies can still match.

**Best shot per burst** is on by default. Automatic selection takes one eligible
Keep from each group, ranked by Keep status and a combination of overall quality
and the weakest assessable subject's detail. It never fills a count shortfall with
extra burst frames. Explicit choices and protected photos remain selected, even
when several belong to one group. Uncheck the option to include all eligible
frames, or explicitly add alternatives. AI Review remains Review until you approve
a frame. Recommendations do not evaluate smiles, closed eyes, expressions or
artistic intention, and similar-burst groups are candidates rather than proof of
duplicate content. Thresholds have synthetic regression coverage, not a claimed
real-world accuracy measurement.

Use **Bursts**, **Best in bursts** or **Burst alternatives** to find groups.
Thumbnail badges show Best of N or Alternate. **Compare duplicate / burst shots**
opens a contact sheet with individual scores, 100% inspection, card controls for
keeping multiple variations, and **Use selected shot for this burst** to replace
that group's delivery choice. Protected alternatives must be unprotected before
replacement. Grouping never marks originals for removal. Unchosen alternatives
can be inspected in the existing final-folder preview and recycled only after
your confirmation. Creative-photo protection still applies to automatic removal
suggestions.

New analysis builds visual fingerprints automatically. On older saved projects,
**Find duplicate / burst shots** upgrades cached assessments in the killable
worker without rerunning the quality models. Fingerprints and the selection
preference persist in project saves; changed images invalidate their cached
fingerprints. Pausing and resuming retains completed work.

Version 0.7.0 makes the thumbnail gallery the default photo library. Use All,
Keep, Review, Chosen and Not chosen to browse the folder. Click a card to inspect
it; its bottom button adds or removes that photo from the delivery set. Adding
a Review photo records an explicit Keep override while retaining its AI Review
assessment. Explicitly chosen and protected photos take priority over automatic
choices, so adding a photo to a full target replaces an automatic choice. If all
target slots are explicit/protected, remove one first or increase the target.
Removing a chosen photo leaves its quality decision and original file intact;
automatic selection fills the available slot from other eligible Keep photos.

Assessment settings are collapsed until needed; the optional delivery count stays
visible. A Details list tab retains scores and filenames. Thumbnails load in the
background, and only visible rows are drawn, with a bounded image cache.

**Review & finish folder** shows three visual lists: chosen originals to keep,
assessed unchosen photos proposed for recycling, and protected/unreadable files
left untouched. This explicit final-folder plan can include creative candidates,
which are identified in its warning. A second confirmation approves the exact
removal list. Source changes or changed choices invalidate the plan. Cleanup
keeps the existing path checks, durable journal, cancellation and Recycle Bin
behavior; it never permanently deletes originals. Successful cleanup leads to
project saving and the finish summary. Saving without cleanup is also available.
The older suggestion-based cleanup remains under Project → Removal suggestions.

Version 0.6.0 adds automatic photographic candidates for low-key scenes,
silhouettes, soft backgrounds and dreamy motion, with protection from automatic
removal suggestions. These observations work without a trained style model.

Version 0.4.0 introduces a violet-accented desktop workspace, live collection
counts, colored review decisions, contextual help, and a dark photo viewer.
The shared logo stays in `src/images/icon.png`; `icon.ico` supplies Windows
executable and installer icons. The native presentation layer lives in
`src/desktop/interface.py`, separate from assessment and project persistence.

Version 0.5.2 rounds the actual splash window and shows a product description
with startup messages for loading components, reading preferences, preparing
controls, and applying the layout. Messages correspond to actual startup tasks.

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

## Creative photos and removal suggestions

Automatic photographic candidates now work without manual labels or a trained
style head: low-light/low-key appearance, possible silhouettes, sharp subjects
against soft backgrounds, and possible slow-shutter/dreamy motion. The inspection
panel explains the detected evidence. A possible effect is not a confirmed
artistic intention or a Creative style judgment. Dark scenes with sharp subjects
can recover the shadow-clipping deduction. Supported silhouettes use clear
contour separation instead of requiring illuminated facial texture. Background
softness alone does not lower an otherwise sharp subject's rule score.

Possible creative photos are excluded from automatic removal suggestions, even
when their delivery decision remains Review. They appear in Creative to inspect.
Explicitly marking an unprotected photo for removal remains available. Ambiguous
dreamy or dark photos remain Review rather than receiving fabricated approval.
The normal Analyze workflow enables this behavior automatically; no annotation
or training step is required. Automatic inference does not identify a beach,
sunset, moving vehicle, or the photographer's artistic intent with certainty.

The delivery decision and photographic style are separate. **Keep** means the
photo is approved for delivery. **Creative** identifies a deliberate effect; it
does not mean every photo using that effect is suitable for delivery. A good
creative shot belongs in **Creative keeps**. A creative shot with unresolved
quality or detection evidence appears in **Creative to inspect**.

The inspection panel shows technical detail and whole-frame context scores.
Neither is a creativity rating. Automatic Creative/Standard recognition is
available only when the installed model includes a style classifier trained on
photographer labels. Otherwise the style is **Unknown**. The application does
not call background blur or motion proof of creativity.

Use **Keep as creative** to approve a readable, assessed photo yourself. This
decision is saved in the project and survives reassessment. **Reset** clears
your manual decision. The Creative keeps filter includes model-approved and
photographer-approved creative photos, and the inspection panel identifies the
source. Use Protect as well if a photo must fit within a limited target count.

The **Removal suggestions** filter shows the same candidates as Review cleanup.
Photos identified as Creative are excluded from automatic removal suggestions,
including creative photos outside the target count and creative photos awaiting
inspection. You may still explicitly Mark removal for an unprotected creative
photo. Nothing moves until you select the exact files in Review cleanup and
confirm the Recycle Bin operation. A Review decision alone is not proof that a
photo is bad.

Automatic recognition is not yet validated on the existing photo collection.
Label delivery suitability and Creative/Standard style independently using
`python -m streamlit run label_photos.py`, then train and evaluate on separate
events as described in PHOTO_ASSESSMENT.md. Saved project approvals are not
silently reused as independent research ground truth.

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

Raster formats use Pillow and RAW formats use LibRaw, as described above.
Other file types are ignored. Analysis is sequential and full-resolution; large
collections require runtime and memory validation. The table stores a row per
photo, project persistence rewrites JSON checkpoints, and similarity matching
uses candidate indexes before detailed verification. Display caches are bounded;
background export and alternative persistence remain future work. Export runs
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
