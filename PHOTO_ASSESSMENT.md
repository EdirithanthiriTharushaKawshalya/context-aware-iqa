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
the score. `auto` sends suspected motion to review because gradient direction
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
