"""Report real data/model readiness without inventing labels or running downloads."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn_features import pretrained_weights_path
from src.training_data import audit_image_contents, load_human_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT/"outputs"/"human_labels.csv"))
    parser.add_argument("--report", default=str(ROOT/"outputs"/"research_readiness.json"))
    parser.add_argument("--probe-image", help="Also verify person/face localization on one source image")
    args = parser.parse_args()
    with open(args.manifest, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    report = {"photos_in_annotation_sheet": len(rows),
              "source_photos_available": sum((ROOT/r["file_path"]).is_file() for r in rows),
              "independent_human_labels": sum(r.get("label_source") == "human" and
                    bool(r.get("labeler_id")) and r.get("expert_score_usability") in ("Keep", "Review") for r in rows),
              "pretrained_resnet18_available": pretrained_weights_path(download=False) is not None,
              "optional_face_package_installed": importlib.util.find_spec("facexlib") is not None,
              "optional_musiq_package_installed": importlib.util.find_spec("pyiqa") is not None,
              "ready_to_train": False}
    human_style_rows = [r for r in rows if r.get("label_source") == "human" and r.get("labeler_id")
                        and r.get("expert_score_usability") in ("Keep", "Review")
                        and r.get("creative_style") in ("Creative", "Standard")]
    report["independent_style_labels"] = len(human_style_rows)
    report["ready_to_train_style"] = False
    try:
        partitions = load_human_partitions(args.manifest, ROOT)
        report["dataset_sha256"] = audit_image_contents(partitions)
        report["ready_to_train"] = True
        report["ready_to_train_style"] = all(
            {r.get("creative_style") for r in partitions[split]} >= {"Creative", "Standard"}
            for split in ("train", "validation"))
        report["partitions"] = {s: {"photos": len(rs), "review": sum(r["target"] == 0 for r in rs),
                                     "events": sorted({r["event_id"] for r in rs})} for s, rs in partitions.items()}
    except (OSError, ValueError) as exc:
        report["remaining_dependency"] = str(exc)
    if args.probe_image:
        from PIL import Image, ImageOps
        from src.semantic_localizer import SemanticLocalizer
        with Image.open(args.probe_image) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1280, 1280))
            localized = SemanticLocalizer().localize(image)
        report["localization_probe"] = {key: localized.get(key) for key in
            ("person_detector_status", "face_detector_status", "subject_detected", "face_detected")}
        report["localization_probe"]["note"] = "One resized source image; confirms runtime loading, not detector accuracy."
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
