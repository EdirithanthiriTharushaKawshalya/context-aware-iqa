The assessment now uses photographic shot intent independently of event names.
A wedding and a birthday can both contain all of these shot types.

| Shot type | Assessment |
| --- | --- |
| Auto | Infers scene, portrait, couple or group from detected people; does not assume they are posing. |
| Portrait | Evaluates the detected person and face; missing or tiny faces require review. |
| Couple | Evaluates both people equally and gives extra weight to the weaker score. |
| Group | Evaluates every detected person, combines mean and lower-quintile quality, and checks each assessable subject for possible focus problems. |
| Candid | Evaluates people without requiring frontal or visible faces. |
| Detail | Scores the full frame without requiring people, suitable for cakes, rings and decorations. |
| Scene | Scores the full frame without requiring people, suitable for venues and streets. |

Choose shot type and motion intent in the Streamlit sidebar, or use:

```powershell
python cli_demo.py --image wedding.jpg --shot-type couple --motion-intent freeze
python cli_demo.py --image street.jpg --shot-type scene --motion-intent intentional
```

`intentional` disables the additional motion deduction and blends local quality
with sharp tile evidence for static detail. Exposure and sharpness still affect
the score. In rules mode, `auto` sends suspected motion to review because gradient direction
cannot establish artistic intent. `freeze` applies the motion deduction without
the intent question. None of these modes automatically accepts a blurred image.

All detected people and their associated faces are drawn in the annotated view.
The individual subject table exposes per-person measurements. A review reason
can override the numeric Keep threshold: Review means inspect, not discard.
Border contact is reported as a possible head crop for posed shots, rather than
automatically destroying the score. Profiles and missing faces are never taken
as proof that a subject is turned away.

Detection counts are estimates, not a guarantee that all 20 guests were found.
The existing SSDLite detector still operates at its native small input size;
distant, overlapping people can be missed. RetinaFace is optional; missing model
support produces unknown face evidence, not fabricated faces. Faces smaller
than 24 pixels are not treated as reliable focus measurements. The system does
not identify the bride/groom, detect closed eyes, or infer cake/ring focus regions.
Detail and scene scoring currently use full-frame evidence, so intentionally
shallow-focus details still need photographer validation. Long-exposure tile
scores are heuristics and cannot distinguish every creative effect from shake.
Confidence is a heuristic indicator, not a calibrated probability.

The regression suite covers 0, 1, 2, 3 and 20 subjects, face association,
blurred partners, tiny faces, detector unavailability and motion intent.
Existing portrait-era tuned parameters are no longer loaded by default.
Rebuild feature caches with `scripts/tune_hyperparameters.py --refresh-cache`
before retuning; fresh tuning uses the same scoring policy as inference and
includes all events. Legacy optimizer entries remain reproducible, but are
rejected by the tuning CLI. Existing learned-fusion models and benchmark reports
have not been retrained or regenerated and do not validate this policy.

Before claiming wedding performance, label representative portraits, couples,
large groups, details, venues, candid sequences and creative motion. Split by
event (and keep bursts together), then measure false reviews and missed defects
per shot type on held-out events. The synthetic tests establish behavior, not
real-world detection accuracy or calibrated quality thresholds.

## Learned creative-context decision

### Automatic photographic candidates without manual annotation

The default rules now inspect low-key appearance, silhouette contrast/contour
evidence, sharp-subject/background-softness separation, and possible dreamy
motion. This behavior is implemented in `src/photographic_intent.py` and works
without human labels or training. It is an engineering policy, not a trained
creative classifier. Silhouette contour candidates are measured at up to 1280
pixels on the longest side to bound memory. Texture/gradient measurements cannot
prove shutter speed, moving vehicles, beach/sunset semantics or artistic intent.
Exposure-time metadata, when present, adds a possible slow-shutter candidate;
missing metadata does not imply a fast exposure. Blurred low-key photos and
ambiguous dreamy motion are not automatically approved.

The rule baseline recovers its global shadow deduction only for supported
silhouettes or dark frames with a sharp subject, and assesses supported silhouette
contours instead of illuminated face texture. Raw feature measurements remain
available to the learned classifier. Learned predictions are not replaced by
these rule adjustments. Possible creative candidates are automatically excluded
from desktop removal suggestions and remain distinct from verified Creative
style labels and photographer overrides.

`scripts/assess_photographic_candidates.py` writes resumable, source-hashed
machine assessments to `outputs/automated_photographic_assessments.jsonl` and a
summary beside it. This optional batch inventory uses 640-pixel previews by
default; normal desktop analysis retains its original-resolution input. Batch
results explicitly use `label_source=automated_rules` and
`is_human_judgment=false`. The human annotation sheet is never changed. These
predictions must not be reused as independent training/evaluation labels.
Candidate counts describe software behavior, not measured recognition accuracy.

The normal pipeline supports versioned logistic and ResNet18-based classifiers in
`src/context_classifier.py`. They learn from subject and face sharpness, subject
motion, background motion/detail, exposure clipping, subject/background
separation, missing-evidence flags and shot/motion context. The legacy 12-feature
experiment and its 44-image report remain historical experiments; their weights
are not loaded by the new application path.

An assessable face supplies subject-motion evidence in place of the larger
person box, which can contain moving background objects. Body-motion evidence
is retained in the subject table. This is still a texture-dependent blur
measurement, not proof of motion or artistic intention. Background measurements
exclude all detected person boxes and their immediate boundaries. Bounding boxes
are approximate regions, not segmentation masks.

In learned mode the model makes the quality decision; old motion penalties do
not override it. Explicit missing detector/count/posed-face evidence can still
require Review. Reports distinguish classifier-only predictions from the
application with these evidence checks. Rule scores remain visible as a
baseline. Review explanations are measured observations and learned feature
associations, not verified causal defect diagnoses. A Keep score is not a
calibrated probability of artistic excellence.

The CNN candidate uses an ImageNet-pretrained, frozen ResNet18 to encode the
whole image and detected person/assessable-face crops. Crops are pooled into a
subject representation. A small nonlinear classification head combines these
visual features with technical/context measurements. Letterboxing preserves
composition, with a default 384-pixel square input. This downsampling can still
hide subtle focus defects; validate resolution effects on your photographs.
No pretrained classification category is treated as a Keep/Review answer.

No new production model has been trained or activated on independent labels.
The application uses rules until a compatible trained model is supplied. Code
and synthetic regression tests do not demonstrate real-world improvement on
creative photos. The current annotation sheet still needs human judgments.

### Independent annotation guide

Create the blind CSV with `python scripts/prepare_human_labels.py`. A starter
sheet is provided at `outputs/human_labels.csv`; previous pipeline predictions
are deliberately absent. The command refuses to overwrite existing annotations.

Use the blind annotation screen instead of editing CSV by hand:

```powershell
python -m streamlit run label_photos.py
```

It displays source photos without predictions, records the photographer's
identity, judgment, Review reason and creative notes, and assigns partitions to
whole events. Saves are atomic and concurrent edits are detected. For an
independent second reviewer, create a separate blank sheet and set
`CONTEXT_IQA_LABELS` to that sheet's path before launching the labelling screen.
Compare their shared subset with:

```powershell
python scripts/compare_annotations.py outputs/human_labels.csv outputs/second_reviewer.csv
```

This reports observed agreement, Cohen's kappa where defined, and disagreements
for documented human resolution. It does not automatically replace either
reviewer's judgments.

Label photos by opening the original images without viewing system predictions:

- **Keep**: suitable for delivery for the intended shot, including deliberate
  background motion, bokeh, panning or intentional subject movement when these
  support the photograph. A creative effect is not automatically a defect.
- **Review**: needs inspection because of an unintended focus/motion problem,
  lost subject detail, unintended clipping, problematic framing, or uncertainty
  about suitability. Record a specific reason. Review does not mean discard.
- Assess faces for posed portraits/couples/groups; do not require visible faces
  in candids, details or scenes. Consider all important people in groups.
- Record creative intent in `creative_notes`. Record a context value only when
  it will also be available at inference; do not feed label-derived creative
  notes into the classifier. Keep auto context for automatic-use experiments.
- Set `label_source=human`, `labeler_id`, `expert_score_usability` and, for Review,
  `review_reason`. An independent second photographer should label a shared
  subset, with disagreements resolved and documented separately.
- Assign `split_category` to `train`, `validation` or `test` for entire events.
  Keep bursts and near duplicates together. Use fresh events for test where
  earlier examples influenced the scoring policy. Each partition needs both
  classes. Blank labels are skipped so a smaller independently labelled subset
  can be used. Audit duplicate event IDs and near duplicates manually.

### Creative style and delivery suitability

Technical detail and whole-frame context are measurements, not two independent
judgments of technical quality and artistic merit. Delivery suitability is the
Keep/Review decision. Creative/Standard style is a separate optional prediction.
There is no numerical creativity rating.

In the blind annotation screen, record **Photographic style** separately from
the delivery judgment. Label an intentional effect Creative whether the photo
is Keep or Review. Label conventional shots Standard; leave uncertain style
unspecified. Include successful panning, long exposures, bokeh and deliberate
movement among Keep examples, and accidental blur and failed creative attempts
among Review examples. Do not infer creative labels from blur measurements or
copy pipeline outputs. Old CSV sheets gain the optional `creative_style` column
when saved without losing their existing judgments.

Training fits the delivery classifier from Keep/Review labels. When train and
validation events both contain explicit Creative and Standard annotations, it
also fits a separate style classifier using the same image/context features.
CNN candidates share their frozen encoder between the two heads. No style
labels means Unknown at inference; it never means every photo is Standard.
Sparse style labels that lack either class skip style training and report the
reason. No test events are used to fit either classifier or select thresholds.

A learned Keep decision can accept intentional motion without the rule-based
motion deduction overriding it. A Creative style prediction alone never forces
Keep: unsuccessful artistic shots can still need inspection. Evaluation reports
creative Keep recall, the number of human-approved creative photos sent to
Review, and style accuracy on explicitly labelled examples. Zero creative
examples produce an undefined recall rather than a claim of perfect results.
Validate these results on new events before relying on automatic culling.

The desktop exposes Creative keeps, Creative to inspect and Removal suggestions.
Photographer-approved creative keeps survive reassessment. Creative-tagged
photos are omitted from automatic cleanup suggestions; explicit manual marking
and confirmation can still remove an unprotected photo. No originals are moved
merely because a model returned Review.

### Train, evaluate and use

```powershell
python scripts/train_context_classifier.py --manifest outputs/human_labels.csv
```

This rejects missing human provenance, event overlap, duplicate paths and
byte-identical images. It regenerates features from the current pipeline, fits
normalization and class-balanced logistic regression on training events, chooses
the threshold using validation macro F1, and writes
`outputs/candidate_context_fusion_model.pt` plus
`outputs/context_fusion_evaluation.json`. The report compares learned decisions,
the application with evidence checks, current rules and always-Keep, with Review
recall/precision, macro F1, balanced accuracy, confusion counts, errors, per-event
and per-shot results, and processing time. Declared human provenance is a
research record; the software cannot verify who actually assigned a label.

Use `--model-type cnn` for a single ResNet18 candidate, and `--input-mode global`,
`subject` or `combined` for context ablations. Normalization is fitted on train
only; validation loss supports early stopping and validation macro F1 selects
the decision threshold. CNN checkpoints include the frozen encoder and its
preprocessing configuration, so inference does not download missing weights.

Freeze settings, then evaluate the saved checkpoint without retraining:

```powershell
python scripts/train_context_classifier.py --manifest outputs/human_labels.csv --evaluate-model outputs/candidate_context_fusion_model.pt --report-json outputs/context_fusion_test.json
```

The manifest must match the checkpoint's recorded digest. Do not repeatedly tune
against this test report. Compare the planned CNN and context ablations on the
same frozen event partitions before drawing conclusions about model superiority.

### Complete model/context comparison

Run the readiness check first. It reports actual available labels, source photos,
weights and optional packages without creating or guessing any human labels:

```powershell
python scripts/check_research_readiness.py
```

Once the independent labels and event assignments are complete:

```powershell
python scripts/run_research_experiment.py --run-dir outputs/research_runs/run1
python scripts/run_research_experiment.py --evaluate-run outputs/research_runs/run1 --activate
```

The first command trains logistic regression and CNN candidates with global-only,
subject-only and combined inputs on identical event partitions. It chooses a
candidate using validation application macro F1, then Review recall, then model
simplicity and combined context as tie breakers. It freezes checkpoint hashes,
thresholds, dataset/code fingerprints and the selection before test evaluation.
Test image features never enter training or validation. Global-only excludes
subject-region measurements; subject-only excludes global/background visual
measurements. Shot/motion context is shared where it is available at inference.

The second command verifies the frozen artifacts, evaluates every candidate on
the same unseen test events, and writes JSON/Markdown comparisons and prediction
CSVs. The validation-selected model remains selected regardless of test ranking.
Runtime measurements warm up the pipeline, then time image decoding plus normal
assessment on CPU. Optional `--activate` makes this checkpoint the normal
application default only after evaluation finishes. An evaluation receipt means
the candidate was tested on these annotations; it does not establish superiority
or guarantee artistic judgment on other image populations. Inspect differences
against current rules and always-Keep, particularly missed Review cases.

Use a new run directory for a different training experiment. Frozen test reports
are not silently overwritten. The script verifies image contents as well as the
annotation manifest and relevant implementation files before testing.

Add `--musiq` to the training command to include an optional pretrained MUSIQ
comparison through PyIQA. Its Keep threshold is selected only on validation
events and reused on test; it is a general image-quality score rather than a
project-trained deliverability classifier. Install PyIQA separately if compatible
with your Python environment. A missing optional package is reported explicitly.
See `requirements-research.txt` for the core research environment.

The MUSIQ baseline uses the KonIQ-10k weights and caps original-resolution
patches at 512 to bound memory, in addition to its multiscale inputs. Its weights,
threshold, package version and patch configuration are frozen in the experiment
directory; test evaluation reloads that snapshot without downloading weights.
Report this patch cap when discussing comparisons with published results.

The first CNN run downloads the official ResNet18 weights into
`outputs/pretrained/` if they are not already available. Subtle focus, occlusion,
composition and deliberate motion remain empirical research questions; the
system cannot promise expert creative judgments before this evaluation.

After selecting a validated candidate, make it available to either application:

```powershell
$env:CONTEXT_IQA_MODEL = (Resolve-Path outputs/candidate_context_fusion_model.pt).Path
python desktop_app.py
# Alternatively, in that same terminal: python -m streamlit run app.py
```

The default automatic lookup also accepts `outputs/context_fusion_model.pt`.
An explicit `PipelineConfig(decision_mode="learned", model_path=...)` requires a
compatible model and fails visibly if unavailable; `decision_mode="rules"`
forces the baseline. Automatic mode falls back to rules with a note if an
installed checkpoint is incompatible. Inference rejects checkpoints with missing
or outdated implementation fingerprints;
retrain those models against the current assessment code before using them.
On the next Analyze/resume, desktop projects
invalidate completed machine assessments when the active model or assessment
settings change. Human overrides and protected photos are preserved. No model file
is bundled into the desktop executable by this change; use the environment
variable with an external checkpoint for packaged builds.

Implementation references: [torchvision ResNet18](https://docs.pytorch.org/vision/main/models)
and [PyIQA MUSIQ adapter](https://github.com/chaofengc/IQA-PyTorch). The custom
aspect-preserving preprocessing and learned fusion head are project choices that
must be assessed on this project's independently labelled photographs.
