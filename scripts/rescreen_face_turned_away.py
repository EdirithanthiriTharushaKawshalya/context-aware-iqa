"""
Targeted re-screening of images currently flagged 'Face Turned Away'.
Re-runs the pipeline (with the 'no subject -> not turned away' fix) and updates
their labels in dataset_manifest.csv and baseline_results.csv in place.

Ground-truth Event 01 (birthday_shoot) rows are intentionally left untouched.
"""

import csv
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Phase4Pipeline, PipelineConfig


def classify_defect(res):
    if res.framing_penalty_pts > 0:
        return "Head Cropped Out"
    if res.face_penalty_pts > 0:
        return "Face Turned Away"
    if res.motion_penalty_pts > 0:
        return "Motion Blur"
    if res.classification == "Review":
        return "Low Technical Quality"
    return "None"


def main():
    manifest_path = PROJECT_ROOT / "dataset_manifest.csv"
    baseline_path = PROJECT_ROOT / "baseline_results.csv"

    with open(manifest_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        m_fields = reader.fieldnames
        manifest = list(reader)

    with open(baseline_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        b_fields = reader.fieldnames
        baseline = list(reader)

    baseline_by_id = {r["image_id"]: r for r in baseline}

    targets = [
        r for r in manifest
        if r.get("expert_score_usability") == "Review"
        and r.get("event_id") != "birthday_shoot"
    ]
    print(f"[INFO] Re-checking {len(targets)} 'Review' images (excluding Event 01 ground truth).")

    pipeline = Phase4Pipeline()
    config = PipelineConfig(threshold=55.0)

    changed, unchanged, errors = [], [], []
    for i, r in enumerate(targets, 1):
        img_path = PROJECT_ROOT / r["file_path"]
        name = Path(r["file_path"]).name
        try:
            with Image.open(img_path) as img:
                res = pipeline.run(img.convert("RGB"), config)
        except Exception as e:
            errors.append(f"{r['image_id']} ({name}): ERROR {e}")
            continue

        loc = res.localization
        usability = res.classification
        defect = classify_defect(res)
        detail = (
            f"{r['image_id']} ({name}): subject={loc.get('subject_detected')} "
            f"face={loc.get('face_detected')} framing={loc.get('framing_anomaly')} "
            f"-> {usability}/{defect}"
        )

        old = (r["expert_score_usability"], r["local_defect_flag"])
        new = (usability, defect)
        r["expert_score_usability"] = usability
        r["local_defect_flag"] = defect

        b = baseline_by_id.get(r["image_id"])
        if b is not None:
            b["expert_score_usability"] = usability
            b["local_defect_flag"] = defect

        if old != new:
            changed.append(f"{detail}   [was {old[0]}/{old[1]}]")
        else:
            unchanged.append(detail)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=m_fields)
        w.writeheader()
        w.writerows(manifest)

    with open(baseline_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=b_fields)
        w.writeheader()
        w.writerows(baseline)

    print(f"\n[INFO] CHANGED ({len(changed)}):")
    for s in changed:
        print("  " + s)
    print(f"\n[INFO] UNCHANGED ({len(unchanged)}):")
    for s in unchanged:
        print("  " + s)
    if errors:
        print(f"\n[ERRORS] ({len(errors)}):")
        for s in errors:
            print("  " + s)


if __name__ == "__main__":
    main()
