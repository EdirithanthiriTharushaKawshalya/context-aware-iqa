"""
Context-Aware Image Quality Assessment (IQA) - Learned Fusion Module
Learns the optimal multi-modal feature combination using machine learning classifiers
(Regularized Logistic Regression and Tiny Multi-Layer Perceptron) trained on ground-truth expert labels.

Replaces hand-set heuristic weights with data-driven calibrated feature importance.
"""

import os
import sys
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import EvaluationHarness, EvaluationReport


FEATURE_NAMES = [
    "s_local",
    "s_global",
    "face_sharpness",
    "subject_sharpness",
    "face_tenengrad",
    "motion_blur_metric",
    "local_contrast",
    "global_contrast",
    "dof_separation_ratio",
    "face_yaw_asymmetry",
    "framing_defect_flag",
    "face_present_flag",
]


def extract_feature_vector(entry: Dict[str, Any]) -> np.ndarray:
    """
    Extracts a normalized, standardized 12-dimensional feature vector from an image entry.
    """
    loc = entry.get("localization", {})
    lf = entry.get("local_features", {})
    gf = entry.get("global_features", {})

    s_local = float(lf.get("local_technical_score", 50.0)) / 100.0
    s_global = float(gf.get("global_context_score", 50.0)) / 100.0
    face_sharp = float(np.clip(np.log1p(float(lf.get("face_sharpness", 0.0))) / 10.0, 0.0, 2.0))
    subj_sharp = float(np.clip(np.log1p(float(lf.get("subject_sharpness", 0.0))) / 10.0, 0.0, 2.0))
    face_ten = float(np.clip(np.log1p(float(lf.get("face_tenengrad", 0.0))) / 5.0, 0.0, 2.0))
    motion = float(lf.get("motion_blur_metric", 0.25))
    l_contrast = float(lf.get("local_contrast", 40.0)) / 100.0
    g_contrast = float(gf.get("global_contrast", 40.0)) / 100.0
    dof = float(np.clip(float(gf.get("dof_separation_ratio", 1.0)) / 10.0, 0.0, 2.0))

    face_yaw = float(loc.get("face_yaw_asymmetry", 0.0 if lf.get("face_present", False) else 1.0))
    framing_def = 1.0 if (lf.get("framing_defect_detected", False) or loc.get("framing_anomaly") == "HEAD_CROPPED") else 0.0
    face_pres = 1.0 if lf.get("face_present", False) else 0.0

    return np.array([
        s_local,
        s_global,
        face_sharp,
        subj_sharp,
        face_ten,
        motion,
        l_contrast,
        g_contrast,
        dof,
        face_yaw,
        framing_def,
        face_pres,
    ], dtype=np.float32)


class LogisticRegressionModel(nn.Module):
    """Calibrated L2-regularized logistic regression for feature fusion."""
    def __init__(self, in_features: int = 12):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class TinyMLPModel(nn.Module):
    """Compact 2-layer multi-layer perceptron for non-linear feature interactions."""
    def __init__(self, in_features: int = 12, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedFusionEngine:
    """
    Trains, validates, and serves learned feature fusion models for Context-Aware IQA.
    """

    def __init__(self, model_type: str = "logistic", threshold: float = 0.50):
        self.model_type = model_type.lower()
        self.threshold = threshold
        self.model: Optional[nn.Module] = None
        self.feature_weights: Optional[Dict[str, float]] = None

    def train(
        self,
        samples: List[Tuple[str, str, Dict[str, Any]]],  # (image_id, label, raw_features)
        epochs: int = 300,
        lr: float = 0.05,
        weight_decay: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Trains the classifier on labeled samples with class-balanced cross-entropy loss.
        """
        X_list = []
        y_list = []
        ids = []

        for img_id, label, entry in samples:
            vec = extract_feature_vector(entry)
            X_list.append(vec)
            # Positive class = Keep (1.0), Negative class = Review (0.0)
            target = 1.0 if label.lower() == "keep" else 0.0
            y_list.append(target)
            ids.append(img_id)

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)

        # Handle class imbalance (39 Keep vs 5 Review -> Review weight = 39/5)
        num_keep = float(torch.sum(y == 1.0))
        num_rev = float(torch.sum(y == 0.0))
        pos_weight = torch.tensor([num_rev / max(1.0, num_keep)])

        in_dim = X.shape[1]
        if self.model_type == "mlp":
            self.model = TinyMLPModel(in_features=in_dim, hidden_dim=16)
        else:
            self.model = LogisticRegressionModel(in_features=in_dim)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X)
            # Weighted BCE
            weights = torch.where(y == 0.0, torch.tensor(num_keep / max(1.0, num_rev)), torch.tensor(1.0))
            loss = nn.functional.binary_cross_entropy(preds, y, weight=weights)
            loss.backward()
            optimizer.step()

        self.model.eval()

        # Extract interpretable weights if logistic regression
        if self.model_type == "logistic":
            w = self.model.linear.weight.detach().cpu().numpy().flatten()
            b = float(self.model.linear.bias.detach().cpu().item())
            self.feature_weights = {name: round(float(val), 4) for name, val in zip(FEATURE_NAMES, w)}
            self.feature_weights["bias"] = round(b, 4)

        # Evaluate on training set
        with torch.no_grad():
            probs = self.model(X).cpu().numpy().flatten()

        eval_preds = []
        for img_id, prob in zip(ids, probs):
            # Scale probability to 0-100 quality score
            score = float(prob * 100.0)
            eval_preds.append({
                "image_id": img_id,
                "predicted_score": score,
                "predicted_classification": "Keep" if prob >= self.threshold else "Review"
            })

        harness = EvaluationHarness()
        gt_mapping = {img_id: {"expert_score_usability": l, "file_path": "", "local_defect_flag": "None"} for img_id, l, _ in samples}
        report = harness.evaluate_predictions(eval_preds, name=f"Learned Fusion ({self.model_type.upper()})", gt_mapping=gt_mapping)

        return {
            "model_type": self.model_type,
            "train_macro_f1": report.macro_f1,
            "train_accuracy": report.accuracy,
            "train_separation_margin": report.separation_margin,
            "feature_weights": self.feature_weights,
            "report": report,
        }

    def predict(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference on a single image feature dictionary."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        vec = extract_feature_vector(entry)
        tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            prob = float(self.model(tensor).item())

        score = round(prob * 100.0, 2)
        classification = "Keep" if prob >= self.threshold else "Review"
        confidence_pct = round(abs(prob - self.threshold) * 200.0, 1)

        return {
            "learned_score": score,
            "probability_keep": round(prob, 4),
            "classification": classification,
            "confidence_pct": min(100.0, max(0.0, confidence_pct)),
        }

    def save(self, output_path: str):
        """Saves model state and weights to file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_type": self.model_type,
            "threshold": self.threshold,
            "state_dict": self.model.state_dict(),
            "feature_weights": self.feature_weights,
            "feature_names": FEATURE_NAMES,
        }, p)
        print(f"[SUCCESS] Saved learned fusion model to: {output_path}")

    @classmethod
    def load(cls, model_path: str) -> "LearnedFusionEngine":
        """Loads trained model state from file."""
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(p, map_location="cpu")
        engine = cls(model_type=checkpoint["model_type"], threshold=checkpoint["threshold"])
        in_dim = len(checkpoint.get("feature_names", FEATURE_NAMES))

        if engine.model_type == "mlp":
            engine.model = TinyMLPModel(in_features=in_dim, hidden_dim=16)
        else:
            engine.model = LogisticRegressionModel(in_features=in_dim)

        engine.model.load_state_dict(checkpoint["state_dict"])
        engine.model.eval()
        engine.feature_weights = checkpoint.get("feature_weights")
        return engine
