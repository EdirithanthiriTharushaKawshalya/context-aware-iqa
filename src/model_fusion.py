"""
Context-Aware Image Quality Assessment (IQA) - Phase 4: Module 4
Unified Model Fusion & Ablation Engine

This script combines local subject features and global context features into
a unified context-aware image quality score. It references 'dataset_manifest.csv'
and 'baseline_results.csv' to compare predictions against human expert labels
(Keep vs. Review) and supports ablation modes:
  - 'full'       : Complete Context-Aware Model (Local + Global + Defect Penalties)
  - 'local_only' : Ablation evaluating ONLY localized subject/face quality
  - 'global_only': Ablation evaluating ONLY whole-frame global quality (traditional style)
  - 'no_penalty' : Ablation evaluating fusion WITHOUT semantic anomaly penalties
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_localizer import SemanticLocalizer
from src.local_features import LocalFeatureExtractor
from src.global_features import GlobalFeatureExtractor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None):
        total = total or len(iterable)
        for idx, item in enumerate(iterable, start=1):
            if idx % max(1, total // 10) == 0 or idx == total:
                print(f"[{idx}/{total}] {desc}...")
            yield item


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Context-Aware IQA Fusion and Ablation Engine."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "dataset_manifest.csv"),
        help="Path to dataset_manifest.csv (default: root/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(PROJECT_ROOT / "baseline_results.csv"),
        help="Optional path to baseline_results.csv to include baseline scores side-by-side.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "fused_results.csv"),
        help="Output CSV path (default: root/fused_results.csv)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "local_only", "global_only", "no_penalty"],
        help="Model fusion mode for ablation testing: full, local_only, global_only, no_penalty",
    )
    parser.add_argument(
        "--w-local",
        type=float,
        default=0.65,
        help="Weight for local technical subject score (default: 0.65)",
    )
    parser.add_argument(
        "--w-global",
        type=float,
        default=0.35,
        help="Weight for global context score (default: 0.35)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples for fast testing.",
    )
    return parser.parse_args()


class ContextAwareIQAFusion:
    """
    Modular fusion engine combining semantic localization, local features,
    and global context representations.
    """

    def __init__(self, mode: str = "full", w_local: float = 0.65, w_global: float = 0.35):
        self.mode = mode
        self.w_local = w_local
        self.w_global = w_global
        
        print(f"[INFO] Initializing ContextAwareIQAFusion (Mode: {self.mode})...")
        self.localizer = SemanticLocalizer()
        self.local_extractor = LocalFeatureExtractor()
        self.global_extractor = GlobalFeatureExtractor()

    def evaluate_image(self, img_path: str) -> Dict[str, Any]:
        """
        Runs the full context-aware assessment pipeline on a single image.
        """
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            
            # Step 1: Semantic Localisation
            loc = self.localizer.localize(rgb_img)

            # Step 2: Local Feature Extraction
            local_feat = self.local_extractor.extract(rgb_img, loc)

            # Step 3: Global Context Extraction
            global_feat = self.global_extractor.extract(rgb_img, loc)

        # Step 4: Fusion Computation according to ablation mode
        s_local = local_feat["local_technical_score"]
        s_global = global_feat["global_context_score"]
        framing_defect = local_feat["framing_defect_detected"]
        has_face = local_feat["face_present"]
        motion_metric = local_feat["motion_blur_metric"]

        if self.mode == "local_only":
            unified_score = s_local
        elif self.mode == "global_only":
            unified_score = s_global
        elif self.mode == "no_penalty":
            unified_score = (self.w_local * s_local) + (self.w_global * s_global)
        else:  # 'full' context-aware fusion
            base_score = (self.w_local * s_local) + (self.w_global * s_global)
            
            # Semantic Defect Penalties:
            # 1. Missing head / cropped face
            if framing_defect:
                base_score = min(base_score * 0.35, 25.0)
            # 2. Turned away face in portrait
            elif not has_face:
                base_score = base_score * 0.60
            # 3. Severe motion blur
            if motion_metric > 0.45:
                base_score -= (motion_metric - 0.45) * 45.0

            unified_score = max(0.0, min(100.0, base_score))

        return {
            "unified_score": round(unified_score, 4),
            "s_local": round(s_local, 4),
            "s_global": round(s_global, 4),
            "subject_detected": loc["subject_detected"],
            "face_detected": loc["face_detected"],
            "framing_anomaly": loc["framing_anomaly"],
            "motion_blur_metric": local_feat["motion_blur_metric"],
            "dof_separation_ratio": global_feat["dof_separation_ratio"],
        }


def load_manifest_and_baseline(manifest_path: str, baseline_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Loads manifest rows and merges with baseline scores if available."""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = []
    with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Merge baseline predictions if available
    if baseline_path and os.path.isfile(baseline_path):
        baseline_map = {}
        with open(baseline_path, mode="r", encoding="utf-8-sig") as f:
            for b_row in csv.DictReader(f):
                img_id = b_row.get("image_id")
                if img_id:
                    baseline_map[img_id] = b_row.get("predicted_score", "")

        for r in rows:
            img_id = r.get("image_id")
            r["baseline_score"] = baseline_map.get(img_id, "")
    else:
        for r in rows:
            r["baseline_score"] = ""

    return rows


def run_fusion_pipeline():
    args = parse_args()
    print("=" * 72)
    print("  Context-Aware IQA - Phase 4: Model Fusion & Ablation Evaluation ")
    print("=" * 72)
    print(f"Manifest:       {args.manifest}")
    print(f"Ablation Mode:  {args.mode.upper()}")
    print(f"Weights:        Local={args.w_local}, Global={args.w_global}")
    print(f"Output File:    {args.output}")

    # Load data
    rows = load_manifest_and_baseline(args.manifest, args.baseline)
    if args.max_samples:
        rows = rows[:args.max_samples]
    print(f"[INFO] Evaluating {len(rows)} images...\n")

    fusion_engine = ContextAwareIQAFusion(
        mode=args.mode,
        w_local=args.w_local,
        w_global=args.w_global
    )

    results = []
    keep_scores = []
    review_scores = []

    for row in tqdm(rows, desc=f"Evaluating (Mode: {args.mode})"):
        rel_path = row.get("file_path", "").strip()
        img_id = row.get("image_id", "").strip()
        expert_label = row.get("expert_score_usability", "").strip()
        
        full_path = (PROJECT_ROOT / rel_path).resolve() if not os.path.isabs(rel_path) else Path(rel_path)

        if not os.path.isfile(full_path):
            print(f"[WARNING] File not found: {full_path}")
            continue

        try:
            eval_dict = fusion_engine.evaluate_image(str(full_path))
            score = eval_dict["unified_score"]

            out_row = dict(row)
            out_row["fusion_mode"] = args.mode
            out_row["context_aware_score"] = f"{score:.4f}"
            out_row["local_score"] = f"{eval_dict['s_local']:.4f}"
            out_row["global_score"] = f"{eval_dict['s_global']:.4f}"
            out_row["subject_detected"] = str(eval_dict["subject_detected"])
            out_row["face_detected"] = str(eval_dict["face_detected"])
            out_row["framing_anomaly"] = eval_dict["framing_anomaly"]
            out_row["motion_blur_metric"] = f"{eval_dict['motion_blur_metric']:.4f}"
            out_row["dof_separation"] = f"{eval_dict['dof_separation_ratio']:.4f}"

            results.append(out_row)

            if expert_label.lower() == "keep":
                keep_scores.append(score)
            elif expert_label.lower() == "review":
                review_scores.append(score)

        except Exception as e:
            print(f"[ERROR] Failed on {img_id}: {e}")

    # Write output CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[SUCCESS] Wrote {len(results)} rows to: {args.output}")

    # Performance Analysis & Separation Margin
    print("\n" + "=" * 72)
    print(f"               ABLATION / PERFORMANCE SUMMARY (MODE: {args.mode.upper()})")
    print("=" * 72)
    print(f"Total Evaluated: {len(results)}")
    
    if keep_scores:
        avg_keep = sum(keep_scores) / len(keep_scores)
        print(f"Average Unified Score for 'Keep'   (n={len(keep_scores)}): {avg_keep:.4f}")
    if review_scores:
        avg_review = sum(review_scores) / len(review_scores)
        print(f"Average Unified Score for 'Review' (n={len(review_scores)}): {avg_review:.4f}")

    if keep_scores and review_scores:
        separation = avg_keep - avg_review
        print(f"Score Separation Margin (Keep - Review):      +{separation:.4f}")
        
        # Simple optimal threshold separation accuracy
        # Find threshold that best splits Keep and Review
        all_thresholds = sorted(list(set(keep_scores + review_scores)))
        best_acc = 0.0
        best_thresh = 0.0
        for thresh in all_thresholds:
            correct = sum(1 for s in keep_scores if s >= thresh) + sum(1 for s in review_scores if s < thresh)
            acc = correct / (len(keep_scores) + len(review_scores))
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        print(f"Optimal Decision Threshold:                   {best_thresh:.4f}")
        print(f"Ground Truth Binary Classification Accuracy:  {best_acc * 100:.2f}%")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run_fusion_pipeline()
