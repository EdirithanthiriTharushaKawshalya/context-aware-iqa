"""Optional PyIQA MUSIQ comparison; never silently substitute another metric."""
import numpy as np
import torch
from PIL import ImageOps
from importlib.metadata import version

from src.context_classifier import classification_metrics


class MusiqBaseline:
    def __init__(self, saved_state=None):
        try:
            import pyiqa
        except ImportError as exc:
            raise RuntimeError("MUSIQ requires the optional pyiqa package; see requirements-research.txt") from exc
        options = {"max_seq_len_from_original_res": 512}
        if saved_state is not None:
            options["pretrained"] = False
        self.metric = pyiqa.create_metric("musiq", device="cpu", **options)
        if saved_state is not None:
            self.metric.load_state_dict(saved_state, strict=True)
        self.threshold = None

    def score(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()/255.
        with torch.inference_mode():
            score = float(self.metric(torch.from_numpy(array).unsqueeze(0)).item())
        if not np.isfinite(score):
            raise ValueError("MUSIQ produced an invalid quality score")
        return score

    def tune(self, scores, labels):
        scores = np.asarray(scores)
        if set(labels) != {0, 1}:
            raise ValueError("MUSIQ threshold tuning requires both validation classes")
        candidates = sorted(set(scores.tolist()+[float(np.nextafter(scores.max(), float('inf')))]))
        def objective(threshold):
            metrics = classification_metrics(labels, scores >= threshold)
            return metrics["macro_f1"], metrics["review_recall"]
        self.threshold = max(candidates, key=objective)
        return self.threshold

    def save(self, path):
        if self.threshold is None:
            raise ValueError("Tune MUSIQ on validation events before saving")
        torch.save({"metric": "musiq", "dataset": "koniq10k", "max_original_patches": 512,
                    "pyiqa_version": version("pyiqa"), "threshold": float(self.threshold),
                    "state_dict": self.metric.state_dict()}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved.get("metric") != "musiq" or saved.get("dataset") != "koniq10k" or
                saved.get("max_original_patches") != 512 or saved.get("pyiqa_version") != version("pyiqa")):
            raise ValueError("Frozen MUSIQ configuration/version mismatch")
        scorer = cls(saved_state=saved["state_dict"])
        scorer.threshold = float(saved["threshold"])
        if not np.isfinite(scorer.threshold):
            raise ValueError("Invalid MUSIQ threshold")
        return scorer
