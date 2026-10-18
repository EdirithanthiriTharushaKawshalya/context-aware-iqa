"""
CLI Training Script for Learned Context-Aware IQA Fusion
Trains Logistic Regression and Tiny MLP classifiers on Event 01 ground-truth features.
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.learned_fusion import LearnedFusionEngine, FEATURE_NAMES
from src.evaluation import EvaluationHarness


def parse_args():
    parser = argparse.ArgumentParser(description="Train learned fusion models on ground truth expert labels.")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "dataset_manifest.csv"))
    parser.add_argument("--cache", type=str, default=str(PROJECT_ROOT / "outputs" / ".features_cache.json"))
    parser.add_argument("--model-type", type=str, default="logistic", choices=["logistic", "mlp"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output-model", type=str, default=str(PROJECT_ROOT / "outputs" / "learned_fusion_model.pt"))
    parser.add_argument("--report-md", type=str, default=str(PROJECT_ROOT / "outputs" / "learned_fusion_report.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 78)
    print(f"       CONTEXT-AWARE IQA: LEARNED FUSION TRAINING ({args.model_type.upper()})")
    print("=" * 78)

    harness = EvaluationHarness(ground_truth_manifest=args.manifest)
    gt = harness.load_ground_truth(event_filter="birthday_shoot")

    with open(args.cache, "r", encoding="utf-8") as f:
        cache = json.load(f)

    norm_cache = {str(Path(k).resolve()).lower(): v for k, v in cache.items()}

    samples = []
    for img_id, item in gt.items():
        p = str((PROJECT_ROOT / item["file_path"]).resolve()).lower()
        if p in norm_cache:
            samples.append((img_id, item["expert_score_usability"], norm_cache[p]))

    print(f"[INFO] Loaded {len(samples)} samples for training ({sum(1 for _, l, _ in samples if l == 'Keep')} Keep, {sum(1 for _, l, _ in samples if l == 'Review')} Review).")

    engine = LearnedFusionEngine(model_type=args.model_type, threshold=0.50)
    train_res = engine.train(samples, epochs=args.epochs)

    print("\n" + "=" * 78)
    print("                         TRAINING RESULTS")
    print("=" * 78)
    print(f"Model Type:               {args.model_type.upper()}")
    print(f"Train Macro F1:           {train_res['train_macro_f1']:.4f}")
    print(f"Train Accuracy:           {train_res['train_accuracy']*100:.2f}%")
    print(f"Separation Margin:        {train_res['train_separation_margin']:+.2f} points")

    if train_res.get("feature_weights"):
        print("-" * 78)
        print("                  LEARNED FEATURE WEIGHTS (LOGISTIC)")
        print("-" * 78)
        for name, weight in sorted(train_res["feature_weights"].items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "(Rewards Keep)" if weight > 0 else "(Penalizes Defects)"
            print(f"  • {name:<26}: {weight:+.4f} {direction}")

    print("=" * 78)

    # Export report & save model
    harness.print_report(train_res["report"])
    if args.report_md:
        harness.export_markdown_report(train_res["report"], args.report_md)

    engine.save(args.output_model)


if __name__ == "__main__":
    main()
