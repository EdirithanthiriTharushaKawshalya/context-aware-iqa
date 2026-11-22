"""Readable experiment tables beside the full machine-readable audit results."""
import csv
import json
from pathlib import Path


def write_comparison(directory, stage, reports, metadata):
    directory = Path(directory)
    payload = {"stage": stage, "metadata": metadata, "comparisons": reports}
    (directory/f"{stage}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"# Context-aware IQA: {stage}", "", "All candidates use the same independently labelled event partitions.", "",
             "| Model | Photos | Review cases | Review recall | Review precision | Macro F1 | Balanced accuracy |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, report in reports.items():
        metrics = report.get("application_with_evidence_checks", report)
        lines.append(f"| {name} | {metrics['photos']} | {metrics['review_cases']} | {metrics['review_recall']:.3f} | "
                     f"{metrics['review_precision']:.3f} | {metrics['macro_f1']:.3f} | {metrics['balanced_accuracy']:.3f} |")
    lines += ["", "Review means inspect before delivery, not discard. Zero-support class metrics are reported as zero; "
              "small event/shot groups support limited conclusions. See JSON for confusion counts, per-event and per-shot "
              "metrics, errors, evidence coverage and timings.", "",
              "Model contributions are associations; independent defect diagnoses require their own annotations.", "",
              "Context modes are full-image/global, detected subject/face, and combined. Shot/motion context is shared "
              "where available. Subject-only has zero visual subject evidence when no person is detected; details and "
              "scenes rely on global inputs. ResNet18 is frozen; its classification head is trained.", ""]
    (directory/f"{stage}_results.md").write_text("\n".join(lines), encoding="utf-8")


def write_predictions(path, rows):
    fields = ["model", "image_id", "event_id", "shot_type", "human_label", "predicted_label",
              "probability_keep", "human_creative_style", "predicted_creative_style",
              "human_review_reason", "evidence_check", "review_evidence"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
