"""Annotation validation before any feature extraction or model fitting."""
import csv
import hashlib
import json
from pathlib import Path

from src.scene_context import SHOT_TYPES, MOTION_INTENTS


def load_human_partitions(manifest, root):
    partitions = {name: [] for name in ("train", "validation", "test")}
    events, paths, ids = {}, set(), set()
    with open(manifest, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label = row.get("expert_score_usability", "").strip()
            if not label:
                continue
            image_id = row.get("image_id", "").strip()
            if not image_id or image_id in ids:
                raise ValueError(f"Missing or duplicate image_id: {image_id}")
            ids.add(image_id)
            if label not in ("Keep", "Review"):
                raise ValueError(f"Invalid human label for {image_id}")
            if row.get("label_source", "").strip() != "human" or not row.get("labeler_id", "").strip():
                raise ValueError(f"{image_id}: independent human label_source and labeler_id are required")
            if label == "Review" and not row.get("review_reason", "").strip():
                raise ValueError(f"{image_id}: record the photographer's reason for Review")
            split, event = row.get("split_category", "").strip(), row.get("event_id", "").strip()
            if split not in partitions or not event:
                raise ValueError(f"{image_id}: assign an event to train, validation or test")
            if event in events and events[event] != split:
                raise ValueError(f"Event leakage: {event} occurs in multiple partitions")
            events[event] = split
            if not row.get("file_path", "").strip():
                raise ValueError(f"{image_id}: missing file_path")
            path = (Path(root)/row["file_path"]).resolve()
            if not path.is_file():
                raise ValueError(f"Missing image: {path}")
            if str(path).casefold() in paths:
                raise ValueError(f"Duplicate image path: {path}")
            paths.add(str(path).casefold())
            row["shot_type"] = row.get("shot_type", "").strip() or "auto"
            row["motion_intent"] = row.get("motion_intent", "").strip() or "auto"
            row["creative_style"] = row.get("creative_style", "").strip()
            if row["creative_style"] not in ("", "Creative", "Standard"):
                raise ValueError(f"{image_id}: unsupported creative style")
            if row["shot_type"] not in SHOT_TYPES or row["motion_intent"] not in MOTION_INTENTS:
                raise ValueError(f"{image_id}: unsupported photographic context")
            row["absolute_path"] = str(path)
            row["target"] = int(label == "Keep")
            partitions[split].append(row)
    for split, rows in partitions.items():
        if {r["target"] for r in rows} != {0, 1}:
            raise ValueError(f"{split} needs independently labelled Keep and Review photos from separate events")
    return partitions


def audit_image_contents(partitions):
    seen, records = {}, []
    for split, rows in partitions.items():
        for row in rows:
            with open(row["absolute_path"], "rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate image contents: {row['image_id']} and {seen[digest]}")
            seen[digest] = row["image_id"]
            records.append((split, row["image_id"], digest))
    return hashlib.sha256(json.dumps(sorted(records)).encode()).hexdigest()


IMPLEMENTATION_FILES = ("cnn_features.py", "context_classifier.py", "pipeline.py", "global_features.py",
                        "local_features.py", "quality_policy.py", "scene_context.py", "semantic_localizer.py",
                        "model_registry.py", "musiq_baseline.py", "photographic_intent.py")


def implementation_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in IMPLEMENTATION_FILES:
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    return digest.hexdigest()
