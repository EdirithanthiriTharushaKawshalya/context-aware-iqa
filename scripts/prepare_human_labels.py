"""Create a blind annotation sheet; never copy pipeline labels into ground truth."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["image_id", "file_path", "event_id", "burst_id", "rights_status", "split_category",
          "shot_type", "motion_intent", "label_source", "labeler_id", "expert_score_usability",
          "review_reason", "creative_notes", "creative_style"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT/"dataset_manifest.csv"))
    parser.add_argument("--output", default=str(ROOT/"outputs"/"human_labels.csv"))
    args = parser.parse_args()
    with open(args.manifest, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") if name in FIELDS[:5] else "" for name in FIELDS})
    print(f"Created {len(rows)} unlabelled rows: {args.output}")


if __name__ == "__main__":
    main()
