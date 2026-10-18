"""
CLI Hyperparameter Tuning Tool for Context-Aware IQA
Optimizes decision threshold, fusion weights, and penalty factors against Event 01 expert labels.
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimizer import HyperparameterOptimizer
from src.pipeline import Phase4Pipeline, PipelineConfig
from src.evaluation import EvaluationHarness
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Context-Aware IQA fusion weights and threshold against ground truth.")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "dataset_manifest.csv"), help="Path to manifest CSV")
    parser.add_argument("--cache", type=str, default=str(PROJECT_ROOT / "outputs" / ".features_cache.json"), help="Path to cached features JSON")
    parser.add_argument("--output-config", type=str, default=str(PROJECT_ROOT / "configs" / "tuned_config.json"), help="Output config path")
    parser.add_argument("--optimize-penalties", action="store_true", help="Also optimize motion and face penalty hyperparameters")
    parser.add_argument("--refresh-cache", action="store_true", help="Re-extract fresh features for all images before tuning")
    return parser.parse_args()


def refresh_features(manifest_path: str, cache_path: str):
    """Re-extracts fresh features from images using current neural network localizer and extractors."""
    from src.semantic_localizer import SemanticLocalizer
    from src.local_features import LocalFeatureExtractor
    from src.global_features import GlobalFeatureExtractor

    localizer = SemanticLocalizer()
    local_extractor = LocalFeatureExtractor()
    global_extractor = GlobalFeatureExtractor()

    harness = EvaluationHarness(ground_truth_manifest=manifest_path)
    gt = harness.load_ground_truth()

    cache = {}
    print(f"[INFO] Extracting fresh features for {len(gt)} images...")
    for img_id, item in tqdm(gt.items(), desc="Extracting Features"):
        full_path = (PROJECT_ROOT / item["file_path"]).resolve()
        if not full_path.is_file():
            continue
        with Image.open(full_path) as img:
            rgb = img.convert("RGB")
            loc = localizer.localize(rgb)
            from src.scene_context import resolve_context
            loc["assessment_context"] = resolve_context(loc)
            lf = local_extractor.extract(rgb, loc)
            gf = global_extractor.extract(rgb, loc)

            cache[str(full_path)] = {
                "localization": loc,
                "local_features": lf,
                "global_features": gf,
            }

    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"[SUCCESS] Refreshed feature cache ({len(cache)} entries) -> {cache_path}")
    return cache


def main():
    args = parse_args()
    print("=" * 78)
    print("      CONTEXT-AWARE IQA - SCIENTIFIC HYPERPARAMETER OPTIMIZATION")
    print("=" * 78)
    print(f"Ground-Truth Manifest: {args.manifest}")
    print(f"Features Cache:        {args.cache}")
    print(f"Output Config:         {args.output_config}")
    print("-" * 78)

    cache_file = Path(args.cache)
    if args.refresh_cache or not cache_file.is_file():
        features_cache = refresh_features(args.manifest, args.cache)
    else:
        with open(cache_file, "r", encoding="utf-8") as f:
            features_cache = json.load(f)
        if any("subjects" not in entry.get("localization", {}) for entry in features_cache.values()):
            raise ValueError("Legacy portrait feature cache: rerun with --refresh-cache before tuning the new policy.")
        print(f"[INFO] Loaded {len(features_cache)} cached feature entries from: {args.cache}")

    optimizer = HyperparameterOptimizer(ground_truth_manifest=args.manifest)
    opt_result = optimizer.grid_search(
        features_cache=features_cache,
        optimize_penalties=args.optimize_penalties,
    )

    # Display comparison
    base = opt_result.baseline_comparison
    best = opt_result.best_config

    print("\n" + "=" * 78)
    print("                     OPTIMIZATION RESULTS COMPARISON")
    print("=" * 78)
    print(f"{'Metric / Parameter':<30} | {'Baseline (Hardcoded)':<20} | {'Tuned (Optimized)':<20}")
    print("-" * 78)
    print(f"{'Decision Threshold':<30} | {55.0:<20.2f} | {best['threshold']:<20.2f}")
    print(f"{'Local Weight (w_local)':<30} | {0.65:<20.2f} | {best['w_local']:<20.2f}")
    print(f"{'Global Weight (w_global)':<30} | {0.35:<20.2f} | {best['w_global']:<20.2f}")
    print(f"{'Face Turned Penalty Mult':<30} | {0.60:<20.2f} | {best.get('face_discount', 0.60):<20.2f}")
    print(f"{'Motion Blur Threshold':<30} | {0.45:<20.2f} | {best.get('motion_thresh', 0.45):<20.2f}")
    print("-" * 78)
    print(f"{'Macro F1-Score':<30} | {base['macro_f1']:<20.4f} | {opt_result.best_macro_f1:<20.4f}")
    print(f"{'Balanced Accuracy':<30} | {base['balanced_acc']*100:<19.2f}% | {opt_result.best_balanced_acc*100:<19.2f}%")
    print(f"{'Review Recall':<30} | {base['recall_review']*100:<19.2f}% | {opt_result.grid_search_summary[0]['recall_review']*100:<19.2f}%")
    sep_b = f"{base['separation_margin']:+.2f}"
    sep_o = f"{opt_result.best_separation_margin:+.2f}"
    print(f"{'Separation Margin':<30} | {sep_b:<20} | {sep_o:<20}")
    print(f"{'Lowest Keep Score':<30} | {base['keep_min']:<20.2f} | {opt_result.grid_search_summary[0]['keep_min']:<20.2f}")
    print(f"{'Highest Review Score':<30} | {base['review_max']:<20.2f} | {opt_result.grid_search_summary[0]['review_max']:<20.2f}")
    print("-" * 78)
    if opt_result.cv_mean_macro_f1 is not None:
        print(f"5-Fold Stratified CV Macro F1: {opt_result.cv_mean_macro_f1:.4f} ± {opt_result.cv_std_macro_f1:.4f}")
    print("=" * 78)

    # Save tuned config
    out_path = optimizer.save_tuned_config(opt_result, args.output_config)
    print(f"[SUCCESS] Optimization complete! Saved configuration to: {out_path}\n")


if __name__ == "__main__":
    main()
