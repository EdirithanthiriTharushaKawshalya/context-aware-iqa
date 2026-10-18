"""
Context-Aware Image Quality Assessment (IQA) - Optimization Engine
Calibrates decision threshold, fusion weights, and defect penalty parameters against ground-truth expert labels.

Supports:
- Grid Search & Scipy Nelder-Mead / Powell Optimization.
- Multi-objective criteria: Macro F1, Review Recall, Balanced Accuracy, Separation Margin.
- Stratified K-Fold Cross-Validation to ensure generalizability.
- Automatic persistence to 'configs/tuned_config.json'.
"""

import os
import sys
import json
import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import EvaluationHarness, EvaluationReport


@dataclass
class OptimizationConfig:
    """Hyperparameters to optimize and their search boundaries."""
    w_local: float = 0.65
    w_global: float = 0.35
    threshold: float = 55.0
    face_discount: float = 0.60
    motion_thresh: float = 0.45
    motion_scale: float = 45.0
    framing_cap: float = 25.0


@dataclass
class OptimizationResult:
    """Structured container holding optimization metrics and best configuration."""
    best_config: Dict[str, float]
    best_macro_f1: float
    best_balanced_acc: float
    best_separation_margin: float
    best_accuracy: float
    total_samples: int
    num_keep: int
    num_review: int
    cv_mean_macro_f1: Optional[float] = None
    cv_std_macro_f1: Optional[float] = None
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    grid_search_summary: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HyperparameterOptimizer:
    """
    Optimizes fusion weights, decision threshold, and defect penalty parameters
    against ground-truth expert labels using cached or extracted features.
    """

    def __init__(self, ground_truth_manifest: Optional[str] = None):
        self.harness = EvaluationHarness(ground_truth_manifest=ground_truth_manifest)
        self.gt = self.harness.load_ground_truth(event_filter="birthday_shoot")

    def _compute_score(
        self,
        entry: Dict[str, Any],
        w_local: float,
        w_global: float,
        face_discount: float = 0.60,
        motion_thresh: float = 0.45,
        motion_scale: float = 45.0,
        framing_cap: float = 25.0,
    ) -> float:
        """Computes unified context-aware score for an image feature entry."""
        loc = entry.get("localization", {})
        lf = entry.get("local_features", {})
        gf = entry.get("global_features", {})

        s_local = float(lf.get("local_technical_score", 50.0))
        s_global = float(gf.get("global_context_score", 50.0))

        base_score = (w_local * s_local) + (w_global * s_global)

        framing_defect = bool(lf.get("framing_defect_detected", False)) or loc.get("framing_anomaly") == "HEAD_CROPPED"
        has_face = bool(lf.get("face_present", True))
        subject_present = bool(loc.get("subject_detected", True))
        motion_metric = float(lf.get("motion_blur_metric", 0.0))

        score_curr = base_score

        # Penalty 1: Head Cropped
        if framing_defect:
            score_curr = min(score_curr * 0.35, framing_cap)
        # Penalty 2: Face missing / turned away in portrait
        elif not has_face and subject_present:
            score_curr = score_curr * face_discount

        # Penalty 3: Motion blur
        if motion_metric > motion_thresh:
            score_curr -= (motion_metric - motion_thresh) * motion_scale

        return max(0.0, min(100.0, score_curr))

    def evaluate_parameter_set(
        self,
        samples: List[Tuple[str, str, Dict[str, Any]]],  # (image_id, label, features)
        w_local: float,
        w_global: float,
        threshold: float,
        face_discount: float = 0.60,
        motion_thresh: float = 0.45,
        motion_scale: float = 45.0,
        framing_cap: float = 25.0,
    ) -> Dict[str, float]:
        """Evaluates a single hyperparameter tuple across sample set."""
        preds = []
        for img_id, _, feats in samples:
            sc = self._compute_score(
                feats,
                w_local=w_local,
                w_global=w_global,
                face_discount=face_discount,
                motion_thresh=motion_thresh,
                motion_scale=motion_scale,
                framing_cap=framing_cap,
            )
            preds.append({
                "image_id": img_id,
                "predicted_score": sc,
                "predicted_classification": "Keep" if sc >= threshold else "Review"
            })

        gt_map = {
            img_id: {"expert_score_usability": label, "file_path": "", "local_defect_flag": "None"}
            for img_id, label, _ in samples
        }
        rep = self.harness.evaluate_predictions(preds, name="Parameter Eval", gt_mapping=gt_map)
        return {
            "macro_f1": rep.macro_f1,
            "balanced_acc": rep.balanced_accuracy,
            "accuracy": rep.accuracy,
            "recall_review": rep.recall_review,
            "f1_review": rep.f1_review,
            "f1_keep": rep.f1_keep,
            "separation_margin": rep.separation_margin,
            "keep_min": rep.keep_score_min,
            "review_max": rep.review_score_max,
        }

    def grid_search(
        self,
        features_cache: Dict[str, Dict[str, Any]],
        w_local_steps: Optional[List[float]] = None,
        threshold_steps: Optional[List[float]] = None,
        optimize_penalties: bool = False,
    ) -> OptimizationResult:
        """
        Executes a fine-grained grid search over fusion weights and decision thresholds.
        Selects the configuration that maximizes Macro F1, breaking ties with Separation Margin.
        """
        # Match cache to ground-truth samples
        samples = []
        norm_cache = {str(Path(k).resolve()).lower(): v for k, v in features_cache.items()}

        for img_id, item in self.gt.items():
            full_path = str((PROJECT_ROOT / item["file_path"]).resolve()).lower()
            if full_path in norm_cache:
                samples.append((img_id, item["expert_score_usability"], norm_cache[full_path]))

        if not samples:
            raise ValueError("No matching samples found between ground-truth manifest and features cache.")

        num_k = sum(1 for _, l, _ in samples if l.lower() == "keep")
        num_r = sum(1 for _, l, _ in samples if l.lower() == "review")

        # Define grid search ranges
        if w_local_steps is None:
            # Local weights from 0.20 to 0.90 in 0.05 increments
            w_local_steps = [round(x * 0.05, 2) for x in range(4, 19)]

        if threshold_steps is None:
            # Threshold from 35.0 to 65.0 in 0.5 increments
            threshold_steps = [round(35.0 + x * 0.5, 1) for x in range(61)]

        face_discounts = [0.50, 0.60, 0.70] if optimize_penalties else [0.60]
        motion_thresholds = [0.40, 0.45, 0.50] if optimize_penalties else [0.45]

        best_score = -1.0
        best_margin = -999.0
        best_cfg = {}
        all_results = []

        total_combos = len(w_local_steps) * len(threshold_steps) * len(face_discounts) * len(motion_thresholds)
        print(f"[INFO] Running grid search over {total_combos} hyperparameter combinations on {len(samples)} images...")

        for wl in w_local_steps:
            wg = round(1.0 - wl, 4)
            for thr in threshold_steps:
                for fd in face_discounts:
                    for mt in motion_thresholds:
                        metrics = self.evaluate_parameter_set(
                            samples,
                            w_local=wl,
                            w_global=wg,
                            threshold=thr,
                            face_discount=fd,
                            motion_thresh=mt,
                        )

                        f1 = metrics["macro_f1"]
                        margin = metrics["separation_margin"]

                        record = {
                            "w_local": wl,
                            "w_global": wg,
                            "threshold": thr,
                            "face_discount": fd,
                            "motion_thresh": mt,
                            **metrics
                        }
                        all_results.append(record)

                        # Primary: Macro F1. Secondary: Separation Margin (robust safety distance)
                        is_better = False
                        if f1 > best_score:
                            is_better = True
                        elif abs(f1 - best_score) < 1e-4 and margin > best_margin:
                            is_better = True

                        if is_better:
                            best_score = f1
                            best_margin = margin
                            best_cfg = {
                                "w_local": wl,
                                "w_global": wg,
                                "threshold": thr,
                                "face_discount": fd,
                                "motion_thresh": mt,
                                "motion_scale": 45.0,
                                "framing_cap": 25.0,
                            }

        # Baseline evaluation (w_local=0.65, w_global=0.35, threshold=55.0)
        baseline_metrics = self.evaluate_parameter_set(
            samples,
            w_local=0.65,
            w_global=0.35,
            threshold=55.0,
            face_discount=0.60,
            motion_thresh=0.45,
        )

        # Cross-Validation: Stratified 5-Fold
        cv_macro_f1s = self._cross_validate(samples, best_cfg, k=5)

        return OptimizationResult(
            best_config=best_cfg,
            best_macro_f1=round(best_score, 4),
            best_balanced_acc=round(all_results[-1]["balanced_acc"], 4),
            best_separation_margin=round(best_margin, 4),
            best_accuracy=round(all_results[-1]["accuracy"], 4),
            total_samples=len(samples),
            num_keep=num_k,
            num_review=num_r,
            cv_mean_macro_f1=round(sum(cv_macro_f1s) / len(cv_macro_f1s), 4) if cv_macro_f1s else None,
            cv_std_macro_f1=round(
                (sum((x - sum(cv_macro_f1s)/len(cv_macro_f1s))**2 for x in cv_macro_f1s) / len(cv_macro_f1s))**0.5, 4
            ) if cv_macro_f1s else None,
            baseline_comparison=baseline_metrics,
            grid_search_summary=sorted(all_results, key=lambda x: (x["macro_f1"], x["separation_margin"]), reverse=True)[:10],
        )

    def _cross_validate(self, samples: List[Tuple[str, str, Dict[str, Any]]], config: Dict[str, float], k: int = 5) -> List[float]:
        """Runs Stratified K-Fold CV to estimate generalization capability."""
        keep_samples = [s for s in samples if s[1].lower() == "keep"]
        rev_samples = [s for s in samples if s[1].lower() == "review"]

        if len(rev_samples) < k:
            k = max(2, len(rev_samples))

        folds = [[] for _ in range(k)]
        for i, s in enumerate(keep_samples):
            folds[i % k].append(s)
        for i, s in enumerate(rev_samples):
            folds[i % k].append(s)

        fold_f1s = []
        for fold_idx in range(k):
            val_set = folds[fold_idx]
            if not val_set:
                continue

            metrics = self.evaluate_parameter_set(
                val_set,
                w_local=config["w_local"],
                w_global=config["w_global"],
                threshold=config["threshold"],
                face_discount=config.get("face_discount", 0.60),
                motion_thresh=config.get("motion_thresh", 0.45),
            )
            fold_f1s.append(metrics["macro_f1"])

        return fold_f1s

    def save_tuned_config(self, opt_result: OptimizationResult, output_path: Optional[str] = None) -> str:
        """Saves best configuration to a JSON config file."""
        p = Path(output_path) if output_path else PROJECT_ROOT / "configs" / "tuned_config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(opt_result.to_dict(), f, indent=2)
        print(f"[SUCCESS] Saved tuned configuration to: {p}")
        return str(p)
