"""Blind human annotations with atomic saves and concurrent-edit detection."""
import csv
import hashlib
import io
import os
from pathlib import Path
import tempfile

from src.scene_context import SHOT_TYPES, MOTION_INTENTS


class AnnotationStore:
    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        self.fields = reader.fieldnames
        self.rows = list(reader)
        required = {"image_id", "file_path", "event_id", "split_category", "label_source", "labeler_id",
                    "expert_score_usability", "review_reason", "shot_type", "motion_intent", "creative_notes"}
        if not self.fields or not required.issubset(self.fields):
            raise ValueError("Create the blind annotation sheet with scripts/prepare_human_labels.py first")
        # Older sheets remain readable; the new field is persisted on next save.
        if "creative_style" not in self.fields:
            self.fields.append("creative_style")
        for row in self.rows:
            row.setdefault("creative_style", "")
        ids = [row["image_id"] for row in self.rows]
        if len(ids) != len(set(ids)) or not all(ids):
            raise ValueError("Annotation image IDs must be unique and nonempty")

    def _write(self, expected_digest):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != expected_digest:
            raise ValueError("Annotations changed in another session. Reload before saving.")
        descriptor, name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp", prefix="annotations-")
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            # A lock file makes the final compare-and-replace exclusive among
            # annotation sessions. It is never held while the user labels.
            lock = self.path.with_suffix(self.path.suffix+".lock")
            acquired = False
            try:
                with lock.open("x"):
                    acquired = True
                    if hashlib.sha256(self.path.read_bytes()).hexdigest() != expected_digest:
                        raise ValueError("Annotations changed in another session. Reload before saving.")
                    new_digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    os.replace(name, self.path)
                    self.digest = new_digest
            except FileExistsError as exc:
                raise ValueError("Another annotation save is in progress; try again.") from exc
            finally:
                # Only remove locks acquired by this operation.
                if acquired and lock.exists():
                    lock.unlink()
        finally:
            if Path(name).exists():
                Path(name).unlink()

    def save_label(self, image_id, label, reviewer, reason="", shot_type="auto", motion_intent="auto",
                   creative_notes="", expected_digest=None, creative_style=""):
        if label not in ("Keep", "Review") or not reviewer.strip():
            raise ValueError("Choose Keep or Review and identify the photographer")
        if label == "Review" and not reason.strip():
            raise ValueError("Give a reason for Review")
        if shot_type not in SHOT_TYPES or motion_intent not in MOTION_INTENTS:
            raise ValueError("Unsupported photographic context")
        if creative_style not in ("", "Creative", "Standard"):
            raise ValueError("Choose Creative, Standard or leave style unspecified")
        matches = [row for row in self.rows if row["image_id"] == image_id]
        if not matches:
            raise ValueError("Unknown photo")
        matches[0].update(expert_score_usability=label, label_source="human", labeler_id=reviewer.strip(),
                          review_reason=reason.strip() if label == "Review" else "", shot_type=shot_type,
                          motion_intent=motion_intent, creative_notes=creative_notes.strip(), creative_style=creative_style)
        self._write(expected_digest or self.digest)

    def assign_event(self, event, split, expected_digest=None):
        if split not in ("train", "validation", "test"):
            raise ValueError("Choose train, validation or test")
        matches = [row for row in self.rows if row["event_id"] == event]
        if not matches:
            raise ValueError("Unknown event")
        for row in matches:
            row["split_category"] = split
        self._write(expected_digest or self.digest)
