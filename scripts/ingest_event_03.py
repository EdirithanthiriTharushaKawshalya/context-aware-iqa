"""
Dataset Ingestion & Screening Script for event_03_birthday
Integrates both burst_01 (41 images) and burst_02 (182 images) = 223 new photos.
Runs Phase 4 screening, updates dataset_manifest.csv and baseline_results.csv.
"""

import os
import sys
import csv
from pathlib import Path
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Phase4Pipeline, PipelineConfig
from src.run_baseline import FallbackQualityScorer, initialize_iqa_metric, evaluate_image


EVENT_ID = "event_03_birthday"
EVENT_DIR = PROJECT_ROOT / "data" / "raw" / "event_03_birthday"
ID_PREFIX = "bday3"
SPLIT = "test"


def collect_images():
    """Collect all jpg files from both bursts in sorted, chronological order."""
    images = []
    for burst in sorted(EVENT_DIR.iterdir()):
        if burst.is_dir():
            burst_files = [(f, burst.name) for f in sorted(burst.glob("*.jpg"))]
            images.extend(burst_files)
    return images


def main():
    manifest_path = PROJECT_ROOT / "dataset_manifest.csv"
    baseline_path = PROJECT_ROOT / "baseline_results.csv"

    if not EVENT_DIR.is_dir():
        print(f"[ERROR] Event directory not found: {EVENT_DIR}")
        sys.exit(1)

    image_files = collect_images()
    print(f"[INFO] Found {len(image_files)} images in {EVENT_DIR.name} (burst_01 + burst_02).")

    # Load existing manifest to check for duplicates
    existing_manifest = []
    if manifest_path.is_file():
        with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_manifest = list(reader)
    print(f"[INFO] Current manifest has {len(existing_manifest)} records.")

    existing_paths = {row.get("file_path", "").replace("\\", "/") for row in existing_manifest}

    # Initialize models
    print("\n[INFO] Initializing Phase 4 Pipeline for automated screening...")
    pipeline = Phase4Pipeline()
    config = PipelineConfig(threshold=55.0)

    print("[INFO] Initializing Baseline Scorer (BRISQUE)...")
    metric, backend = initialize_iqa_metric("brisque", device="cpu")

    new_manifest_rows = []
    new_baseline_rows = []

    print("\n[INFO] Processing images...\n")
    for idx, (img_path, burst_name) in enumerate(image_files, start=1):
        rel_path = f"data/raw/event_03_birthday/{burst_name}/{img_path.name}"

        if rel_path in existing_paths:
            print(f"[{idx}/{len(image_files)}] Already registered: {img_path.name}")
            continue

        img_id = f"{ID_PREFIX}_{idx:03d}"

        try:
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                res = pipeline.run(rgb_img, config)

            # Defect classification
            defect_flag = "None"
            if res.framing_penalty_pts > 0:
                defect_flag = "Head Cropped Out"
            elif res.face_penalty_pts > 0:
                defect_flag = "Face Turned Away"
            elif res.motion_penalty_pts > 0:
                defect_flag = "Motion Blur"
            elif res.classification == "Review":
                defect_flag = "Low Technical Quality"

            usability = res.classification

            manifest_row = {
                "image_id": img_id,
                "file_path": rel_path,
                "event_id": EVENT_ID,
                "burst_id": burst_name,
                "split_category": SPLIT,
                "rights_status": "consented",
                "expert_score_usability": usability,
                "local_defect_flag": defect_flag,
            }
            new_manifest_rows.append(manifest_row)

            # Baseline BRISQUE score
            b_score, b_status, b_err = evaluate_image(metric, backend, str(img_path))
            baseline_row = {
                "image_id": img_id,
                "file_path": rel_path,
                "event_id": EVENT_ID,
                "burst_id": burst_name,
                "split_category": SPLIT,
                "rights_status": "consented",
                "expert_score_usability": usability,
                "local_defect_flag": defect_flag,
                "baseline_metric": "brisque",
                "predicted_score": f"{b_score:.4f}" if b_score is not None else "",
                "status": b_status,
                "error_message": b_err,
            }
            new_baseline_rows.append(baseline_row)

            print(
                f"[{idx}/{len(image_files)}] {img_id} ({burst_name}/{img_path.name}): "
                f"{usability} (Score: {res.unified_score:.1f}, Baseline: {b_score}, Defect: {defect_flag})"
            )

        except Exception as e:
            print(f"[ERROR] Failed processing {img_path.name}: {e}")

    # Append to dataset_manifest.csv
    if new_manifest_rows:
        fieldnames = [
            "image_id", "file_path", "event_id", "burst_id",
            "split_category", "rights_status", "expert_score_usability", "local_defect_flag"
        ]
        write_mode = "a" if manifest_path.is_file() else "w"
        with open(manifest_path, mode=write_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_mode == "w":
                writer.writeheader()
            writer.writerows(new_manifest_rows)
        print(f"\n[SUCCESS] Appended {len(new_manifest_rows)} rows to {manifest_path.name}")

    # Append to baseline_results.csv
    if new_baseline_rows:
        b_fields = [
            "image_id", "file_path", "event_id", "burst_id", "split_category",
            "rights_status", "expert_score_usability", "local_defect_flag",
            "baseline_metric", "predicted_score", "status", "error_message"
        ]
        write_mode = "a" if baseline_path.is_file() else "w"
        with open(baseline_path, mode=write_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=b_fields)
            if write_mode == "w":
                writer.writeheader()
            writer.writerows(new_baseline_rows)
        print(f"[SUCCESS] Appended {len(new_baseline_rows)} rows to {baseline_path.name}")

    # Summary
    total_manifest = len(existing_manifest) + len(new_manifest_rows)
    keep_count = sum(1 for r in new_manifest_rows if r["expert_score_usability"] == "Keep")
    review_count = sum(1 for r in new_manifest_rows if r["expert_score_usability"] == "Review")
    print(f"\n--- Event 03 Ingestion Summary ---")
    print(f"Total dataset count:       {total_manifest} images")
    print(f"Event 03 newly processed:  {len(new_manifest_rows)} images")
    print(f"  - Keep:   {keep_count}")
    print(f"  - Review: {review_count}")


if __name__ == "__main__":
    main()
