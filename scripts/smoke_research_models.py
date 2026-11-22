"""Verify pretrained components on a real photo without assigning human labels."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageOps

from src.cnn_features import FrozenResNetEncoder, pretrained_weights_path
from src.semantic_localizer import SemanticLocalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--musiq", action="store_true")
    parser.add_argument("--report", default=str(ROOT/"outputs"/"research_runtime_probe.json"))
    args = parser.parse_args()
    if pretrained_weights_path(download=False) is None:
        raise FileNotFoundError("Download the official ResNet18 weights before running this verification")
    with Image.open(args.image) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    image.thumbnail((1280, 1280))
    start = time.perf_counter()
    loc = SemanticLocalizer().localize(image)
    localization_seconds = time.perf_counter()-start
    encoder = FrozenResNetEncoder()
    start = time.perf_counter()
    embeddings = encoder.extract(image, loc)
    assert len(embeddings["image"]) == len(embeddings["subject"]) == 512
    assert all(np.isfinite(embeddings[key]).all() for key in embeddings)
    report = {"purpose": "Pretrained runtime verification only; no Keep/Review prediction or accuracy claim",
              "probe_image_size": list(image.size), "resnet18_initial_weights": encoder.weights_origin,
              "encoder_frozen": all(not p.requires_grad for p in encoder.backbone.parameters()),
              "image_embedding_dimensions": 512, "subject_embedding_dimensions": 512,
              "person_detector_status": loc["person_detector_status"], "face_detector_status": loc["face_detector_status"],
              "detected_people": len(loc["subjects"]), "detected_faces": len(loc["faces"]),
              "localization_seconds_including_initialization": localization_seconds,
              "cnn_embedding_seconds": time.perf_counter()-start}
    if args.musiq:
        from src.musiq_baseline import MusiqBaseline
        scorer = MusiqBaseline()
        probe = image.copy()
        probe.thumbnail((512, 512))
        report["musiq_raw_quality_score"] = scorer.score(probe)
        report["musiq_probe_image_size"] = list(probe.size)
        report["musiq_note"] = "General quality score on a resized probe; no project Keep/Review threshold has been fitted."
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
