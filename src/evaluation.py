"""
Context-Aware Image Quality Assessment (IQA) - Evaluation Harness
Comprehensive benchmarking and validation suite against ground-truth expert labels.

Supports:
- Binary classification metrics: Accuracy, Precision, Recall, F1 (macro & per-class),
  Balanced Accuracy, Specificity, ROC-AUC, PR-AUC.
- Confusion Matrix (Keep vs. Review).
- Per-defect diagnosis breakdown (Motion Blur, Face Turned Away, Head Cropped Out, etc.).
- Score distribution statistics & separation margin (min Keep - max Review).
- Formatted console display, Markdown export, and structured JSON/dict export.
"""

import os
import sys
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ConfusionMatrix:
    """Confusion matrix representation for binary classification (Positive = Keep, Negative = Review)."""
    tp: int = 0  # True Keep
    fp: int = 0  # False Keep (Review predicted as Keep - critical defect leak)
    tn: int = 0  # True Review
    fn: int = 0  # False Review (Keep predicted as Review - unnecessary reject)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> Dict[str, int]:
        return {"TP": self.tp, "FP": self.fp, "TN": self.tn, "FN": self.fn, "Total": self.total}


@dataclass
class DefectStats:
    """Detection performance for a specific defect category."""
    defect_name: str
    total_count: int
    detected_as_review: int
    falsely_kept: int
    recall: float
    sample_ids: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete structured evaluation results container."""
    name: str
    total_samples: int
    num_keep: int
    num_review: int
    accuracy: float
    balanced_accuracy: float
    precision_keep: float
    recall_keep: float
    f1_keep: float
    precision_review: float
    recall_review: float
    f1_review: float
    macro_f1: float
    specificity: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: ConfusionMatrix
    separation_margin: float
    keep_score_mean: float
    keep_score_std: float
    keep_score_min: float
    keep_score_max: float
    review_score_mean: float
    review_score_std: float
    review_score_min: float
    review_score_max: float
    defect_breakdown: Dict[str, DefectStats] = field(default_factory=dict)
    mismatches: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confusion_matrix"] = self.confusion_matrix.to_dict()
        d["defect_breakdown"] = {k: asdict(v) for k, v in self.defect_breakdown.items()}
        return d


class EvaluationHarness:
    """
    Standardized evaluation harness for Context-Aware IQA pipelines, baseline models,
    and ablation studies against ground truth expert labels.
    """

    def __init__(self, ground_truth_manifest: Optional[str] = None):
        self.manifest_path = Path(ground_truth_manifest) if ground_truth_manifest else PROJECT_ROOT / "dataset_manifest.csv"
        self._ground_truth_cache: Optional[Dict[str, Dict[str, str]]] = None

    def load_ground_truth(self, event_filter: Optional[Union[str, List[str]]] = None) -> Dict[str, Dict[str, str]]:
        """
        Loads ground truth expert annotations from manifest CSV.
        Returns mapping from image_id -> dict(file_path, expert_score_usability, local_defect_flag, ...).
        """
        import csv

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        events = [event_filter] if isinstance(event_filter, str) else event_filter

        gt = {}
        with open(self.manifest_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("image_id", "").strip()
                event_id = row.get("event_id", "").strip()
                label = row.get("expert_score_usability", "").strip()

                if not label:
                    continue

                if events is not None:
                    if event_id not in events and not any(img_id.startswith(e) for e in events):
                        continue

                gt[img_id] = {
                    "image_id": img_id,
                    "file_path": row.get("file_path", "").strip(),
                    "event_id": event_id,
                    "burst_id": row.get("burst_id", "").strip(),
                    "expert_score_usability": label,
                    "local_defect_flag": row.get("local_defect_flag", "None").strip() or "None",
                }

        self._ground_truth_cache = gt
        return gt

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        name: str = "Pipeline Evaluation",
        gt_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> EvaluationReport:
        """
        Evaluates a list of prediction dictionaries.
        Each prediction dictionary should contain:
            - 'image_id': str
            - 'predicted_score': float (continuous quality score, e.g. 0-100)
            - 'predicted_classification': Optional[str] ('Keep' or 'Review')
        """
        if gt_mapping is None:
            if self._ground_truth_cache is None:
                gt_mapping = self.load_ground_truth()
            else:
                gt_mapping = self._ground_truth_cache

        y_true = []
        y_pred = []
        scores = []
        defects = []
        mismatches = []
        sample_ids = []

        keep_scores = []
        review_scores = []

        for p in predictions:
            img_id = p.get("image_id", "").strip()
            if img_id not in gt_mapping:
                continue

            gt_item = gt_mapping[img_id]
            t_label = gt_item["expert_score_usability"]
            d_flag = gt_item["local_defect_flag"]

            score = float(p.get("predicted_score", 0.0))
            pred_label = p.get("predicted_classification")
            if not pred_label:
                pred_label = "Keep" if score >= 55.0 else "Review"

            y_true.append(t_label)
            y_pred.append(pred_label)
            scores.append(score)
            defects.append(d_flag)
            sample_ids.append(img_id)

            if t_label.lower() == "keep":
                keep_scores.append(score)
            else:
                review_scores.append(score)

            if t_label.lower() != pred_label.lower():
                mismatches.append({
                    "image_id": img_id,
                    "file_path": gt_item["file_path"],
                    "ground_truth": t_label,
                    "predicted": pred_label,
                    "score": round(score, 2),
                    "defect_flag": d_flag,
                    "metadata": {k: v for k, v in p.items() if k not in ["image_id", "predicted_score", "predicted_classification"]}
                })

        total = len(y_true)
        if total == 0:
            raise ValueError("No matching ground-truth items found for the provided predictions.")

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt.lower() == "keep" and yp.lower() == "keep")
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt.lower() == "review" and yp.lower() == "keep")
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt.lower() == "review" and yp.lower() == "review")
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt.lower() == "keep" and yp.lower() == "review")
        cm = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)

        accuracy = (tp + tn) / total
        precision_keep = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_keep = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_keep = (2 * precision_keep * recall_keep) / (precision_keep + recall_keep) if (precision_keep + recall_keep) > 0 else 0.0

        precision_review = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        recall_review = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_review = (2 * precision_review * recall_review) / (precision_review + recall_review) if (precision_review + recall_review) > 0 else 0.0

        macro_f1 = (f1_keep + f1_review) / 2.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (recall_keep + recall_review) / 2.0

        roc_auc = self._calc_roc_auc(y_true, scores)
        pr_auc = self._calc_pr_auc(y_true, scores)

        k_mean = float(sum(keep_scores) / len(keep_scores)) if keep_scores else 0.0
        k_std = float(math.sqrt(sum((s - k_mean)**2 for s in keep_scores) / len(keep_scores))) if keep_scores else 0.0
        k_min = float(min(keep_scores)) if keep_scores else 0.0
        k_max = float(max(keep_scores)) if keep_scores else 0.0

        r_mean = float(sum(review_scores) / len(review_scores)) if review_scores else 0.0
        r_std = float(math.sqrt(sum((s - r_mean)**2 for s in review_scores) / len(review_scores))) if review_scores else 0.0
        r_min = float(min(review_scores)) if review_scores else 0.0
        r_max = float(max(review_scores)) if review_scores else 0.0

        separation_margin = (k_min - r_max) if (keep_scores and review_scores) else 0.0

        defect_breakdown = {}
        defect_groups = {}
        for img_id, yt, yp, d in zip(sample_ids, y_true, y_pred, defects):
            if yt.lower() == "review":
                d_key = d if d and d.lower() != "nan" else "Unspecified Defect"
                defect_groups.setdefault(d_key, []).append((img_id, yp))

        for d_name, items in defect_groups.items():
            tot = len(items)
            det = sum(1 for _, yp in items if yp.lower() == "review")
            f_kept = tot - det
            rec = det / tot if tot > 0 else 0.0
            defect_breakdown[d_name] = DefectStats(
                defect_name=d_name,
                total_count=tot,
                detected_as_review=det,
                falsely_kept=f_kept,
                recall=round(rec, 4),
                sample_ids=[i for i, _ in items]
            )

        return EvaluationReport(
            name=name,
            total_samples=total,
            num_keep=len(keep_scores),
            num_review=len(review_scores),
            accuracy=round(accuracy, 4),
            balanced_accuracy=round(balanced_accuracy, 4),
            precision_keep=round(precision_keep, 4),
            recall_keep=round(recall_keep, 4),
            f1_keep=round(f1_keep, 4),
            precision_review=round(precision_review, 4),
            recall_review=round(recall_review, 4),
            f1_review=round(f1_review, 4),
            macro_f1=round(macro_f1, 4),
            specificity=round(specificity, 4),
            roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
            pr_auc=round(pr_auc, 4) if pr_auc is not None else None,
            confusion_matrix=cm,
            separation_margin=round(separation_margin, 4),
            keep_score_mean=round(k_mean, 4),
            keep_score_std=round(k_std, 4),
            keep_score_min=round(k_min, 4),
            keep_score_max=round(k_max, 4),
            review_score_mean=round(r_mean, 4),
            review_score_std=round(r_std, 4),
            review_score_min=round(r_min, 4),
            review_score_max=round(r_max, 4),
            defect_breakdown=defect_breakdown,
            mismatches=mismatches,
        )

    @staticmethod
    def _calc_roc_auc(y_true: List[str], scores: List[float]) -> Optional[float]:
        pos_scores = [s for yt, s in zip(y_true, scores) if yt.lower() == "keep"]
        neg_scores = [s for yt, s in zip(y_true, scores) if yt.lower() != "keep"]
        if not pos_scores or not neg_scores:
            return None

        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        ranks_sum = 0
        all_samples = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
        all_samples.sort(key=lambda x: x[0])

        rank = 1
        i = 0
        n = len(all_samples)
        while i < n:
            j = i
            while j < n - 1 and all_samples[j][0] == all_samples[j + 1][0]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                if all_samples[k][1] == 1:
                    ranks_sum += avg_rank
            i = j + 1

        u = ranks_sum - (n_pos * (n_pos + 1)) / 2.0
        return u / (n_pos * n_neg)

    @staticmethod
    def _calc_pr_auc(y_true: List[str], scores: List[float]) -> Optional[float]:
        pos_indices = {i for i, yt in enumerate(y_true) if yt.lower() == "keep"}
        n_pos = len(pos_indices)
        if n_pos == 0 or n_pos == len(y_true):
            return None

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        num_true_pos = 0
        precisions = []

        for rank, (orig_idx, _) in enumerate(ranked, start=1):
            if orig_idx in pos_indices:
                num_true_pos += 1
                precisions.append(num_true_pos / rank)

        return sum(precisions) / n_pos if precisions else 0.0

    def print_report(self, report: EvaluationReport):
        cm = report.confusion_matrix
        print("\n" + "=" * 78)
        print(f"       CONTEXT-AWARE IQA EVALUATION HARNESS: {report.name.upper()}")
        print("=" * 78)
        print(f"Dataset Size:           {report.total_samples} images ({report.num_keep} Keep, {report.num_review} Review)")
        print(f"Overall Accuracy:       {report.accuracy * 100:.2f}%")
        print(f"Balanced Accuracy:      {report.balanced_accuracy * 100:.2f}%")
        print(f"Macro F1 Score:         {report.macro_f1:.4f}")
        print(f"ROC-AUC / PR-AUC:       {report.roc_auc if report.roc_auc is not None else 'N/A'} / {report.pr_auc if report.pr_auc is not None else 'N/A'}")
        print("-" * 78)
        print("                      CLASSIFICATION METRICS")
        print(f"  [Keep Class]     Precision: {report.precision_keep:.4f} | Recall: {report.recall_keep:.4f} | F1: {report.f1_keep:.4f}")
        print(f"  [Review Class]   Precision: {report.precision_review:.4f} | Recall: {report.recall_review:.4f} | F1: {report.f1_review:.4f}")
        print(f"  Specificity:     {report.specificity:.4f}")
        print("-" * 78)
        print("                      CONFUSION MATRIX")
        print(f"                       Predicted Keep      Predicted Review")
        print(f"  Actual Keep (Pos)     TP: {cm.tp:3d}              FN: {cm.fn:3d}")
        print(f"  Actual Review (Neg)   FP: {cm.fp:3d}              TN: {cm.tn:3d}")
        print("-" * 78)
        print("                      PER-DEFECT RECALL BREAKDOWN")
        if report.defect_breakdown:
            for defect_name, d_stat in report.defect_breakdown.items():
                print(f"  • {defect_name:<20}: {d_stat.detected_as_review}/{d_stat.total_count} flagged Review ({d_stat.recall * 100:.1f}% recall) | Leaked: {d_stat.falsely_kept}")
        else:
            print("  (No defects present in evaluated set)")
        print("-" * 78)
        print("                      SCORE DISTRIBUTION & SEPARATION")
        print(f"  Keep Scores:    Mean={report.keep_score_mean:.2f} ± {report.keep_score_std:.2f} | Range: [{report.keep_score_min:.2f}, {report.keep_score_max:.2f}]")
        print(f"  Review Scores:  Mean={report.review_score_mean:.2f} ± {report.review_score_std:.2f} | Range: [{report.review_score_min:.2f}, {report.review_score_max:.2f}]")
        sep_symbol = "+" if report.separation_margin > 0 else ""
        print(f"  Safety Margin (Min Keep - Max Review): {sep_symbol}{report.separation_margin:.2f} points")
        if report.separation_margin > 0:
            print(f"  Status: 100% LINEAR SEPARABILITY ACHIEVED (Optimal boundary: ~{(report.keep_score_min + report.review_score_max)/2:.2f})")
        else:
            overlap = abs(report.separation_margin)
            print(f"  Status: SCORE OVERLAP DETECTED (Overlap region: {overlap:.2f} points)")

        if report.mismatches:
            print("-" * 78)
            print(f"                      MISCLASSIFICATIONS ({len(report.mismatches)})")
            for m in report.mismatches:
                print(f"  • {m['image_id']}: GT={m['ground_truth']} | Pred={m['predicted']} | Score={m['score']} | Defect={m['defect_flag']}")
        print("=" * 78 + "\n")

    def export_markdown_report(self, report: EvaluationReport, output_path: str):
        cm = report.confusion_matrix
        lines = [
            f"# Context-Aware IQA Evaluation: {report.name}",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Evaluated Samples**: {report.total_samples} ({report.num_keep} Keep, {report.num_review} Review)",
            f"- **Overall Accuracy**: **{report.accuracy * 100:.2f}%**",
            f"- **Balanced Accuracy**: **{report.balanced_accuracy * 100:.2f}%**",
            f"- **Macro F1-Score**: **{report.macro_f1:.4f}**",
            f"- **Keep F1**: {report.f1_keep:.4f} | **Review F1**: {report.f1_review:.4f}",
            f"- **Separation Margin**: `{report.separation_margin:+.2f}` points",
            "",
            "## 2. Classification Performance",
            "",
            "| Class | Precision | Recall | F1-Score | Support |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Keep** | {report.precision_keep:.4f} | {report.recall_keep:.4f} | {report.f1_keep:.4f} | {report.num_keep} |",
            f"| **Review** | {report.precision_review:.4f} | {report.recall_review:.4f} | {report.f1_review:.4f} | {report.num_review} |",
            f"| **Macro Avg** | {((report.precision_keep + report.precision_review)/2):.4f} | {((report.recall_keep + report.recall_review)/2):.4f} | **{report.macro_f1:.4f}** | {report.total_samples} |",
            "",
            "### Confusion Matrix",
            "",
            "| Ground Truth \\ Prediction | Predicted Keep | Predicted Review | Total |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Actual Keep** | **{cm.tp}** (TP) | {cm.fn} (FN) | {report.num_keep} |",
            f"| **Actual Review** | {cm.fp} (FP) | **{cm.tn}** (TN) | {report.num_review} |",
            "",
            "## 3. Defect-Level Diagnosis",
            "",
            "| Defect Category | Total Cases | Flagged Review | False Keeps (Leaked) | Recall |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]

        for d_name, d_stat in report.defect_breakdown.items():
            lines.append(f"| **{d_name}** | {d_stat.total_count} | {d_stat.detected_as_review} | {d_stat.falsely_kept} | **{d_stat.recall * 100:.1f}%** |")

        lines.extend([
            "",
            "## 4. Score Separation Analysis",
            "",
            "| Subset | Mean Score | Std Dev | Min Score | Max Score |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Keep Images** | {report.keep_score_mean:.2f} | {report.keep_score_std:.2f} | {report.keep_score_min:.2f} | {report.keep_score_max:.2f} |",
            f"| **Review Images** | {report.review_score_mean:.2f} | {report.review_score_std:.2f} | {report.review_score_min:.2f} | {report.review_score_max:.2f} |",
            "",
            f"> [!NOTE]\n> **Separation Margin ($S_{{Keep}}^{{min}} - S_{{Review}}^{{max}}$)**: `{report.separation_margin:+.2f}` points.\n",
        ])

        if report.mismatches:
            lines.extend([
                "## 5. Misclassified Samples",
                "",
                "| Image ID | Ground Truth | Predicted | Score | Defect Flag |",
                "| :--- | :---: | :---: | :---: | :--- |",
            ])
            for m in report.mismatches:
                lines.append(f"| `{m['image_id']}` | {m['ground_truth']} | {m['predicted']} | {m['score']:.2f} | {m['defect_flag']} |")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"[SUCCESS] Exported Markdown report to: {output_path}")
