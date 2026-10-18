"""
CLI Tool for Burst-Aware Context Assessment
Processes image bursts to find the sharpest Hero Frame and identify redundant or degraded shots.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.burst_context import BurstContextEngine
from src.pipeline import Phase4Pipeline, PipelineConfig
from src.evaluation import EvaluationHarness


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate burst context and identify hero frames.")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "dataset_manifest.csv"))
    parser.add_argument("--cache", type=str, default=str(PROJECT_ROOT / "outputs" / ".features_cache.json"))
    parser.add_argument("--event", type=str, default="birthday_shoot")
    parser.add_argument("--output-json", type=str, default=str(PROJECT_ROOT / "outputs" / "burst_analysis.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    harness = EvaluationHarness(ground_truth_manifest=args.manifest)
    gt = harness.load_ground_truth(event_filter=args.event)

    # Group by burst_id
    bursts = defaultdict(list)
    with open(args.cache, "r", encoding="utf-8") as f:
        cache = json.load(f)
    norm_cache = {str(Path(k).resolve()).lower(): v for k, v in cache.items()}

    config = PipelineConfig.from_tuned()

    for img_id, item in gt.items():
        bid = item.get("burst_id", "burst_default") or "burst_default"
        p = str((PROJECT_ROOT / item["file_path"]).resolve()).lower()
        if p in norm_cache:
            entry = norm_cache[p]
            lf = entry["local_features"]
            gf = entry["global_features"]
            loc = entry["localization"]
            sl = lf["local_technical_score"]
            sg = gf["global_context_score"]
            base = config.w_local * sl + config.w_global * sg
            score = base
            if loc.get("face_orientation") == "TURNED_AWAY" or loc.get("framing_anomaly") == "FACE_NOT_FOUND":
                score *= config.face_discount
            elif loc.get("framing_anomaly") == "HEAD_CROPPED":
                score = min(score * 0.35, config.framing_cap)
            if lf.get("motion_blur_metric", 0.0) > config.motion_thresh:
                score -= (lf["motion_blur_metric"] - config.motion_thresh) * config.motion_scale
            score = max(0.0, min(100.0, score))

            bursts[bid].append({
                "image_id": img_id,
                "file_path": item["file_path"],
                "unified_score": score,
                "defect_flag": item.get("local_defect_flag"),
            })

    engine = BurstContextEngine()
    all_summaries = {}

    for bid, frames in sorted(bursts.items()):
        summary = engine.evaluate_burst(burst_id=bid, frame_entries=frames, event_id=args.event)
        engine.print_burst_summary(summary)
        all_summaries[bid] = summary.to_dict()

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[SUCCESS] Exported complete burst analysis to: {args.output_json}")


if __name__ == "__main__":
    main()
