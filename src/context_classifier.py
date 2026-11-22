"""Versioned, standardized logistic fusion for independently labelled photos.

Scores estimate P(Keep); they are not calibrated probabilities of artistic merit.
Linear contributions explain the model's evidence, not the cause of a defect.
"""
from pathlib import Path
import math
import copy

import numpy as np
import torch

from src.scene_context import SHOT_TYPES, MOTION_INTENTS
from src.model_registry import DEFAULT_MODEL

SCHEMA_VERSION = 3
FEATURE_NAMES = (
    "subject_sharpness", "face_sharpness", "face_tenengrad", "subject_motion", "body_motion",
    "background_motion", "background_detail", "subject_background_separation",
    "subject_contrast", "global_contrast", "subject_highlight_clipping",
    "subject_shadow_clipping", "global_highlight_clipping", "global_shadow_clipping",
    "subject_present", "face_assessable", "background_assessable", "detector_available",
    "weakest_subject_score", "subject_count",
    "global_sharpness",
) + tuple("shot_" + s for s in SHOT_TYPES if s != "auto") + tuple("motion_" + s for s in MOTION_INTENTS)


def missing_evidence_reasons(localization, features):
    context = localization.get("assessment_context", {})
    reasons = list(context.get("review_reasons", []))
    if localization.get("unmatched_face_count", 0):
        reasons.append("Some faces could not be matched to people; subject coverage is incomplete.")
    if context.get("faces_required") and (not features.get("assessable_face_count") or
                                          features.get("unassessable_subject_count", 0)):
        reasons.append("Some posed faces could not be assessed reliably.")
    return reasons


def feature_vector(entry):
    loc, lf, gf = (entry.get(k, {}) for k in ("localization", "local_features", "global_features"))
    context = loc.get("assessment_context", {})
    def number(mapping, key, default=0.):
        value = mapping.get(key)
        result = float(default if value is None else value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite feature: {key}")
        return result
    def log(mapping, key):
        return math.log1p(max(0., number(mapping, key)))
    records = lf.get("subject_metrics", [])
    values = [
        log(lf, "subject_sharpness"), log(lf, "face_sharpness"), log(lf, "face_tenengrad"),
        number(lf, "motion_blur_metric"), number(lf, "body_motion_metric", number(lf, "motion_blur_metric")),
        number(gf, "background_motion_metric"),
        log(gf, "background_gradient_energy"), log(gf, "dof_separation_ratio"),
        number(lf, "local_contrast") / 100, number(gf, "global_contrast") / 100,
        number(lf, "subject_highlight_clipping"), number(lf, "subject_shadow_clipping"),
        number(gf, "highlight_clipping_ratio"), number(gf, "shadow_underexposure_ratio"),
        float(bool(loc.get("subject_detected"))), float(lf.get("assessable_face_count", 0) > 0),
        float(bool(gf.get("background_assessable"))),
        float(loc.get("person_detector_status") == "ok"),
        min((float(r["local_technical_score"]) for r in records), default=number(lf, "local_technical_score"))/100,
        math.log1p(max(0, int(context.get("subject_count", 0)))),
        log(gf, "global_sharpness"),
    ]
    values += [float(context.get("shot_type") == s) for s in SHOT_TYPES if s != "auto"]
    values += [float(context.get("motion_intent", "auto") == s) for s in MOTION_INTENTS]
    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Invalid feature vector")
    return vector


def classification_metrics(labels, predictions):
    """Keep=1, Review=0; use explicit confusion names to avoid ambiguity."""
    y, p = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    kk, kr = int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    rk, rr = int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    divide = lambda a, b: a/b if b else 0.
    return {"photos": len(y), "review_cases": int((y == 0).sum()),
            "accuracy": divide(kk+rr, len(y)), "review_recall": divide(rr, rr+rk),
            "review_precision": divide(rr, rr+kr),
            "keep_recall": divide(kk, kk+kr), "keep_precision": divide(kk, kk+rk),
            "keep_f1": divide(2*kk, 2*kk+kr+rk), "review_f1": divide(2*rr, 2*rr+kr+rk),
            "macro_f1": (divide(2*kk, 2*kk+kr+rk)+divide(2*rr, 2*rr+kr+rk))/2,
            "balanced_accuracy": (divide(kk, kk+kr)+divide(rr, rr+rk))/2,
            "confusion": {"keep_as_keep": kk, "keep_as_review": kr,
                          "review_as_keep": rk, "review_as_review": rr}}


class ContextClassifier:
    def __init__(self, model_type="logistic", input_mode="combined", encoder=None):
        if model_type not in ("logistic", "cnn") or input_mode not in ("global", "subject", "combined"):
            raise ValueError("Unsupported classifier type or input mode")
        self.model_type, self.input_mode, self.encoder = model_type, input_mode, encoder
        global_names = {"global_sharpness", "global_contrast", "global_highlight_clipping", "global_shadow_clipping"}
        combined_names = {"background_motion", "background_detail", "subject_background_separation"}
        context_names = {n for n in FEATURE_NAMES if n.startswith(("shot_", "motion_"))}
        selected = set(FEATURE_NAMES)
        if input_mode == "global":
            selected = global_names | context_names
        elif input_mode == "subject":
            selected -= global_names | combined_names
        self.indices = [i for i, name in enumerate(FEATURE_NAMES) if name in selected]
        self.feature_names = [FEATURE_NAMES[i] for i in self.indices]
        if model_type == "cnn":
            if input_mode != "subject":
                self.feature_names += [f"image_visual_{i}" for i in range(512)]
            if input_mode != "global":
                self.feature_names += [f"subject_visual_{i}" for i in range(512)]
        self.model = self._new_head()
        self.mean = np.zeros(len(self.feature_names), dtype=np.float32)
        self.scale = np.ones(len(self.feature_names), dtype=np.float32)
        self.threshold = .5
        self.metadata = {}
        self.creative_classifier = None

    def _new_head(self):
        if self.model_type == "logistic":
            return torch.nn.Linear(len(self.feature_names), 1)
        return torch.nn.Sequential(torch.nn.Linear(len(self.feature_names), 64), torch.nn.ReLU(),
                                   torch.nn.Dropout(.2), torch.nn.Linear(64, 1))

    def prepare_entry(self, image, entry):
        if self.model_type == "logistic":
            return entry
        if self.encoder is None:
            raise ValueError("CNN classifier has no frozen visual encoder")
        return dict(entry, cnn_features=self.encoder.extract(image, entry["localization"], self.input_mode))

    def _vector(self, entry):
        values = feature_vector(entry)[self.indices]
        if self.model_type == "cnn":
            extras = []
            for key in ("image", "subject"):
                if (key == "image" and self.input_mode == "subject") or (key == "subject" and self.input_mode == "global"):
                    continue
                array = np.asarray(entry.get("cnn_features", {}).get(key, []), dtype=np.float32)
                if array.shape != (512,) or not np.isfinite(array).all():
                    raise ValueError(f"Missing or invalid {key} CNN features")
                extras.append(array)
            values = np.concatenate([values] + extras)
        return values

    def fit(self, entries, labels, epochs=400, seed=42, validation_entries=None, validation_labels=None):
        self.creative_classifier = None
        if set(labels) != {0, 1}:
            raise ValueError("Training requires both Keep and Review labels")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        torch.manual_seed(seed)
        if len(entries) != len(labels):
            raise ValueError("Features and labels must have equal lengths")
        self.model = self._new_head()
        x = np.stack([self._vector(e) for e in entries])
        self.mean, self.scale = x.mean(0), np.maximum(x.std(0), .05)
        inputs = torch.from_numpy((x-self.mean)/self.scale)
        targets = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        counts = np.bincount(labels, minlength=2)
        weights = torch.tensor([len(labels)/(2*counts[y]) for y in labels]).reshape(-1, 1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.025, weight_decay=.01)
        validation = None
        if validation_entries is not None:
            if validation_labels is None or set(validation_labels) != {0, 1} or len(validation_entries) != len(validation_labels):
                raise ValueError("Early stopping needs both validation classes and matching labels")
            vx = torch.from_numpy((np.stack([self._vector(e) for e in validation_entries])-self.mean)/self.scale)
            vy = torch.tensor(validation_labels, dtype=torch.float32).reshape(-1, 1)
            counts_val = np.bincount(validation_labels, minlength=2)
            vw = torch.tensor([len(validation_labels)/(2*counts_val[y]) for y in validation_labels]).reshape(-1, 1)
            validation = vx, vy, vw
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.model(inputs), targets, weight=weights)
            loss.backward()
            optimizer.step()
            if validation is not None:
                self.model.eval()
                with torch.no_grad():
                    value = float(torch.nn.functional.binary_cross_entropy_with_logits(
                        self.model(validation[0]), validation[1], weight=validation[2]))
                if value < best_loss-1e-4:
                    best_loss, best_state, best_epoch, stale = value, copy.deepcopy(self.model.state_dict()), epoch+1, 0
                else:
                    stale += 1
                if stale >= 40:
                    break
                self.model.train()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.training_summary = {"epochs_run": epoch+1, "selected_epoch": best_epoch or epoch+1,
                                 "early_stopping": validation is not None}
        self.model.eval()
        return self

    def predict(self, entry):
        self.model.eval()
        standardized = (self._vector(entry)-self.mean)/self.scale
        with torch.no_grad():
            logit = self.model(torch.from_numpy(standardized)).squeeze()
            probability = float(torch.sigmoid(logit))
        if not math.isfinite(probability):
            raise ValueError("Classifier returned a non-finite score")
        if self.model_type == "logistic":
            contributions = standardized*self.model.weight.detach().numpy().reshape(-1)
            evidence = list(zip(self.feature_names, contributions.tolist()))
        else:
            # Group occlusion in standardized space replaces evidence with its
            # training mean. Effects are associations, never defect diagnoses.
            groups = {"image_visual_evidence": [], "subject_visual_evidence": [], "technical_and_context_evidence": []}
            for i, name in enumerate(self.feature_names):
                group = "image_visual_evidence" if name.startswith("image_visual_") else (
                    "subject_visual_evidence" if name.startswith("subject_visual_") else "technical_and_context_evidence")
                groups[group].append(i)
            evidence = []
            with torch.no_grad():
                for name, indices in groups.items():
                    if indices:
                        hidden = standardized.copy()
                        hidden[indices] = 0
                        evidence.append((name, float(logit-self.model(torch.from_numpy(hidden)).squeeze())))
        evidence.sort(key=lambda item: item[1])
        creative_probability, creative_style = None, "Unknown"
        if self.creative_classifier is not None:
            style = self.creative_classifier.predict(entry)
            creative_probability = style["probability_keep"]
            creative_style = "Creative" if creative_probability >= self.creative_classifier.threshold else "Standard"
        return {"classification": "Keep" if probability >= self.threshold else "Review",
                "probability_keep": probability,
                "creative_probability": creative_probability, "creative_style": creative_style,
                "review_evidence": [{"feature": n, "contribution": round(v, 4)} for n, v in evidence if v < 0][:3]}

    def fit_creative(self, entries, labels, validation_entries, validation_labels, epochs=400, seed=42):
        """Learn style independently of delivery suitability; Creative=1, Standard=0.

        A creative label never automatically approves an otherwise defective photo.
        The delivery head must learn which creative effects are worth keeping.
        """
        style = ContextClassifier(self.model_type, self.input_mode, self.encoder)
        style.fit(entries, labels, epochs=epochs, seed=seed,
                  validation_entries=validation_entries, validation_labels=validation_labels)
        style.tune_threshold(validation_entries, validation_labels)
        self.creative_classifier = style
        return self

    def tune_threshold(self, entries, labels):
        if set(labels) != {0, 1}:
            raise ValueError("Validation requires both Keep and Review labels")
        probabilities = np.array([self.predict(e)["probability_keep"] for e in entries])
        candidates = sorted(set([.5] + probabilities.tolist() + [float(np.nextafter(probabilities.max(), 1.))]))
        def objective(t):
            metrics = classification_metrics(labels, probabilities >= t)
            return metrics["macro_f1"], metrics["review_recall"], -abs(t-.5)
        self.threshold = max(candidates, key=objective)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.model_type == "cnn" and self.encoder is None:
            raise ValueError("Cannot save a CNN checkpoint without its frozen encoder")
        torch.save({"schema_version": SCHEMA_VERSION, "feature_names": self.feature_names,
                    "model_type": self.model_type, "input_mode": self.input_mode,
                    "encoder": self.encoder.checkpoint() if self.model_type == "cnn" else None,
                    "state_dict": self.model.state_dict(), "mean": self.mean.tolist(),
                    "scale": self.scale.tolist(), "threshold": self.threshold,
                    "metadata": self.metadata,
                    "creative_head": self.creative_classifier._head_checkpoint()
                    if self.creative_classifier is not None else None}, path)

    def _head_checkpoint(self):
        return {"state_dict": self.model.state_dict(), "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "threshold": self.threshold,
                "training_summary": getattr(self, "training_summary", {})}

    def _restore_head(self, saved):
        self.model.load_state_dict(saved["state_dict"])
        self.mean = np.asarray(saved["mean"], dtype=np.float32)
        self.scale = np.asarray(saved["scale"], dtype=np.float32)
        self.threshold = float(saved["threshold"])
        if (self.mean.shape != (len(self.feature_names),) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or (self.scale <= 0).any() or not 0 <= self.threshold <= 1
                or not all(torch.isfinite(p).all() for p in self.model.parameters())):
            raise ValueError("Invalid classifier parameters")
        self.training_summary = saved.get("training_summary", {})
        self.model.eval()

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Incompatible context classifier; regenerate features and retrain")
        engine = cls(saved.get("model_type", "logistic"), saved.get("input_mode", "combined"))
        if saved.get("feature_names") != engine.feature_names:
            raise ValueError("Incompatible classifier feature schema")
        engine._restore_head(saved)
        engine.metadata = saved.get("metadata", {})
        if engine.model_type == "cnn":
            from src.cnn_features import FrozenResNetEncoder
            engine.encoder = FrozenResNetEncoder.from_checkpoint(saved["encoder"])
            if not all(torch.isfinite(p).all() for p in engine.encoder.backbone.parameters()):
                raise ValueError("Invalid CNN encoder weights")
        if saved.get("creative_head") is not None:
            style = cls(engine.model_type, engine.input_mode, engine.encoder)
            style._restore_head(saved["creative_head"])
            engine.creative_classifier = style
        return engine
