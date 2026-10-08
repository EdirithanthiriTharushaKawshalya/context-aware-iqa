"""
Context-Aware Image Quality Assessment (IQA) - Phase 3
Baseline No-Reference IQA Evaluation Script

This script reads image paths from 'dataset_manifest.csv', evaluates each image
using a standard open-source No-Reference IQA model (such as NIMA, MUSIQ, or BRISQUE
via 'pyiqa' or standalone PyTorch features) while strictly preserving aspect ratio,
and writes predictions and comparison metadata to 'baseline_results.csv'.
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback progress bar
    def tqdm(iterable, desc="", total=None):
        total = total or len(iterable)
        for idx, item in enumerate(iterable, start=1):
            if idx % max(1, total // 10) == 0 or idx == total:
                print(f"[{idx}/{total}] {desc}...")
            yield item


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run No-Reference IQA baseline models on event photo dataset manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "dataset_manifest.csv"),
        help="Path to dataset_manifest.csv (default: root/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "baseline_results.csv"),
        help="Path to save baseline results CSV (default: root/baseline_results.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="musiq",
        help="Model metric name to use (e.g. 'musiq', 'nima', 'brisque', 'niqe'). Default: musiq",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device: 'auto', 'cuda', or 'cpu'. Default: auto",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of images to process (useful for quick testing).",
    )
    return parser.parse_args()


def resolve_device(requested_device: str):
    """Determine compute device safely."""
    try:
        import torch
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif requested_device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")
    except ImportError:
        return "cpu"


def initialize_iqa_metric(metric_name: str, device: Any):
    """
    Initialize a No-Reference IQA metric.
    Prioritizes 'pyiqa' create_metric.
    Falls back gracefully if pyiqa or weights need alternatives.
    """
    print(f"\n[INFO] Initializing NR-IQA model: '{metric_name}' on device: {device}")
    
    try:
        import pyiqa
        metric = pyiqa.create_metric(metric_name, device=device)
        print(f"[SUCCESS] Successfully loaded pyiqa metric '{metric_name}'.")
        return metric, "pyiqa"
    except ImportError:
        print("[WARNING] 'pyiqa' library is not yet installed or importable.")
    except Exception as e:
        print(f"[WARNING] Could not load pyiqa metric '{metric_name}' ({e}).")

    # Fallback to standard PyTorch / PIL feature evaluation if pyiqa is unavailable
    print("[INFO] Attempting fallback lightweight No-Reference sharpness/quality metric...")
    return FallbackQualityScorer(), "fallback"


class FallbackQualityScorer:
    """
    Lightweight fallback No-Reference image quality estimator.
    Computes spatial frequency / edge gradient variance & dynamic range
    without altering or cropping the aspect ratio. Uses Pillow natively.
    """
    def __init__(self):
        self.name = "laplacian_edge_fallback"

    def __call__(self, img_path: str) -> float:
        if Image is None:
            raise RuntimeError("PIL (Pillow) is required to process images.")
        
        from PIL import ImageFilter, ImageStat

        with Image.open(img_path) as img:
            # Convert to grayscale without resizing to preserve native aspect ratio and spatial detail
            gray = img.convert("L")
            width, height = gray.size
            
            # Subsample gently if resolution is excessively huge (> 2048) while keeping exact aspect ratio
            max_dim = 1600
            if max(width, height) > max_dim:
                scale = max_dim / max(width, height)
                new_size = (int(width * scale), int(height * scale))
                gray = gray.resize(new_size, Image.Resampling.BILINEAR)

            # Edge detection to quantify high-frequency focus/sharpness
            edges = gray.filter(ImageFilter.FIND_EDGES)
            stat = ImageStat.Stat(edges)
            edge_variance = stat.var[0] if stat.var else 0.0

            # Contrast / dynamic range calculation
            orig_stat = ImageStat.Stat(gray)
            std_dev = orig_stat.stddev[0] if orig_stat.stddev else 0.0

            # Composite sharpness + contrast quality metric
            raw_score = (edge_variance ** 0.5) * 1.5 + std_dev * 0.5
            norm_score = float(min(100.0, max(0.0, raw_score)))
            return round(norm_score, 4)


def load_manifest(manifest_path: str) -> List[Dict[str, str]]:
    """Read dataset manifest CSV."""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    records = []
    with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def evaluate_image(
    metric: Any,
    backend: str,
    full_img_path: str
) -> Tuple[Optional[float], str, str]:
    """
    Evaluate quality of a single image while preserving its aspect ratio.
    Returns: (predicted_score, status, error_message)
    """
    if not os.path.isfile(full_img_path):
        return None, "FILE_NOT_FOUND", f"File does not exist: {full_img_path}"

    try:
        # Verify file is readable as an image
        if Image is not None:
            with Image.open(full_img_path) as test_img:
                test_img.verify()

        start_t = time.time()
        
        if backend == "pyiqa":
            # PyIQA metrics can directly take the file path and handle aspect ratio natively
            # (MUSIQ, BRISQUE, NIQE, etc. operate on full scale or aspect-ratio preserved patches)
            score_tensor = metric(full_img_path)
            if hasattr(score_tensor, "item"):
                score = float(score_tensor.item())
            else:
                score = float(score_tensor)
        else:
            score = metric(full_img_path)

        return round(score, 4), "SUCCESS", ""

    except Exception as e:
        return None, "ERROR", str(e)


def run_baseline_pipeline():
    args = parse_args()
    print("=" * 70)
    print("  Context-Aware IQA - Phase 3: Baseline NR-IQA Model Evaluation  ")
    print("=" * 70)
    print(f"Manifest Path: {args.manifest}")
    print(f"Output Path:   {args.output}")
    print(f"Target Model:  {args.model}")

    # 1. Load manifest
    manifest_rows = load_manifest(args.manifest)
    total_manifest = len(manifest_rows)
    print(f"[INFO] Loaded {total_manifest} rows from manifest.")

    if args.max_samples:
        manifest_rows = manifest_rows[:args.max_samples]
        print(f"[INFO] Limiting evaluation to first {len(manifest_rows)} rows as requested.")

    # 2. Setup Device & Model
    device = resolve_device(args.device)
    metric, backend = initialize_iqa_metric(args.model, device)

    # 3. Process Each Image Preserving Aspect Ratio
    print(f"\n[INFO] Starting inference on {len(manifest_rows)} images...\n")
    results = []
    
    keep_scores = []
    review_scores = []

    for row in tqdm(manifest_rows, desc="Evaluating images"):
        rel_path = row.get("file_path", "").strip()
        img_id = row.get("image_id", "").strip()
        expert_label = row.get("expert_score_usability", "").strip()
        defect_flag = row.get("local_defect_flag", "").strip()

        # Handle relative path with respect to project root
        full_path = (PROJECT_ROOT / rel_path).resolve() if not os.path.isabs(rel_path) else Path(rel_path)

        score, status, err = evaluate_image(metric, backend, str(full_path))

        # Copy all original manifest columns and append predictions
        out_row = dict(row)
        out_row["baseline_metric"] = args.model if backend == "pyiqa" else getattr(metric, "name", "fallback")
        out_row["predicted_score"] = f"{score:.4f}" if score is not None else ""
        out_row["status"] = status
        out_row["error_message"] = err

        results.append(out_row)

        if status == "SUCCESS" and score is not None:
            if expert_label.lower() == "keep":
                keep_scores.append(score)
            elif expert_label.lower() == "review":
                review_scores.append(score)

    # 4. Save to baseline_results.csv
    print(f"\n[INFO] Saving evaluation results to: {args.output}")
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[SUCCESS] Wrote {len(results)} rows to {args.output}")

    # 5. Print Summary Statistics vs Expert Labels
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    error_count = len(results) - success_count

    print("\n" + "=" * 70)
    print("                     EVALUATION SUMMARY                      ")
    print("=" * 70)
    print(f"Total Processed: {len(results)}")
    print(f"Successful:      {success_count}")
    print(f"Failed / Missing:{error_count}")

    if keep_scores:
        avg_keep = sum(keep_scores) / len(keep_scores)
        print(f"Average Predicted Score for 'Keep'   (n={len(keep_scores)}): {avg_keep:.4f}")
    if review_scores:
        avg_review = sum(review_scores) / len(review_scores)
        print(f"Average Predicted Score for 'Review' (n={len(review_scores)}): {avg_review:.4f}")

    if keep_scores and review_scores:
        diff = avg_keep - avg_review
        print(f"Score Separation (Keep - Review):      {diff:+.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_baseline_pipeline()
