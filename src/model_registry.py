"""Lightweight active-model lookup usable before desktop model initialization."""
import hashlib
import json
import os
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parents[1]/"outputs"/"context_fusion_model.pt"
ASSESSMENT_VERSION = 6


def active_model_path(explicit=None):
    return Path(explicit or os.environ.get("CONTEXT_IQA_MODEL") or DEFAULT_MODEL)


def engine_signature(shot_type="auto", motion_intent="auto"):
    path = active_model_path()
    model_hash = "rules"
    if path.is_file():
        with path.open("rb") as handle:
            model_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    record = {"assessment_version": ASSESSMENT_VERSION, "model": model_hash,
              "shot_type": shot_type, "motion_intent": motion_intent}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
