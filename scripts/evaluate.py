"""
CLI Evaluation Tool for Context-Aware IQA
Evaluates predictions from pipeline, baseline CSV, or cache against ground truth expert labels.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import EvaluationHarness
from src.pipeline import Phase4Pipeline, PipelineConfig
import csv
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Context-Aware IQA against expert ground-truth labels.")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "dataset_manifest.csv"), help="Path to manifest CSV")
    parser.add_argument("--event", type=str, default="birthday_shoot", help="Event ID or prefix filter (default: birthday_shoot / bday_)")
    parser.add_argument("--threshold", type=float, default=55.0, help="Decision threshold for Keep vs Review (default: 55.0)")
    parser.add_argument("--w-local", type=float, default=0.65, help="Local feature fusion weight (default: 0.65)")
    parser.add_argument("--w-global", type=float, default=0.35, help="Global context fusion weight (default: 0.35)")
    parser.add_argument("--ablation", type=str, default="full", choices=["full", "local_only", "global_only", "no_penalty"])
    parser.add_argument("--baseline-csv", type=str, default=None, help="Optional baseline_results.csv to evaluate baseline metric")
    parser.add_argument("--report-md", type=str, default=str(PROJECT_ROOT / "outputs" / "evaluation_report.md"), help="Path to save markdown report")
    parser.add_argument("--report-json", type=str, default=str(PROJECT_ROOT / "outputs" / "evaluation_report.json"), help="Path to save JSON report")
    return parser.parse_args()


def main():
    args = parse_args()
    harness = EvaluationHarness(ground_truth_manifest=args.manifest)
    gt_mapping = harness.load_ground_truth(event_filter=args.event)
    print(f"[INFO] Loaded {len(gt_mapping)} ground-truth labeled samples for event filter: '{args.event}'")

    predictions = []

    if args.baseline_csv:
        print(f"[INFO] Evaluating baseline predictions from: {args.baseline_csv}")
        with open(args.baseline_csv, mode="r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                img_id = row.get("image_id", "").strip()
                if img_id in gt_mapping:
                    score = float(row.get("predicted_score", 0.0))
                    predictions.append({
                        "image_id": img_id,
                        "predicted_score": score,
                        "predicted_classification": "Keep" if score >= args.threshold else "Review"
                    })
        eval_name = f"Baseline Metric ({Path(args.baseline_csv).name})"
    else:
        print(f"[INFO] Running Phase 4 Pipeline (Ablation: {args.ablation}, w_local={args.w_local}, w_global={args.w_global}, threshold={args.threshold})")
        pipeline = Phase4Pipeline()
        config = PipelineConfig(
            w_local=args.w_local,
            w_global=args.w_global,
            threshold=args.threshold,
            ablation_mode=args.ablation
        )

        for img_id, gt_item in tqdm(gt_mapping.items(), desc="Evaluating Images"):
            fpath = PROJECT_ROOT / gt_item["file_path"]
            if not fpath.is_file():
                print(f"[WARNING] Image file not found: {fpath}")
                continue
            with Image.open(fpath) as img:
                res = pipeline.run(img, config)
                predictions.append({
                    "image_id": img_id,
                    "predicted_score": res.unified_score,
                    "predicted_classification": res.classification,
                    "s_local": res.s_local,
                    "s_global": res.s_global,
                    "framing_penalty": res.framing_penalty_pts,
                    "face_penalty": res.face_penalty_pts,
                    "motion_penalty": res.motion_penalty_pts,
                })
        eval_name = f"Phase 4 Pipeline ({args.ablation.upper()}, Thr={args.threshold})"

    report = harness.evaluate_predictions(predictions, name=eval_name, gt_mapping=gt_mapping)
    harness.print_report(report)

    if args.report_md:
        harness.export_markdown_report(report, args.report_md)

    if args.report_json:
        import json
        out_j = Path(args.report_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[SUCCESS] Exported JSON report to: {args.report_json}")


if __name__ == "__main__":
    main()
