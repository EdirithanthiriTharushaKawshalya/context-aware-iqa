"""Train logistic or CNN fusion on independent human event partitions.

Features are freshly extracted. Test evaluation is opt-in after settings freeze.
The legacy train_fusion_classifier.py is a training-set-only experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from src.context_classifier import ContextClassifier, SCHEMA_VERSION, classification_metrics, missing_evidence_reasons
from src.pipeline import Phase4Pipeline, PipelineConfig
from src.training_data import load_human_partitions, audit_image_contents, implementation_fingerprint


def extract_rows(rows, pipeline, encoder=None):
    entries, timings = [], []
    for row in rows:
        started = time.perf_counter()
        with Image.open(row["absolute_path"]) as image:
            result = pipeline.run(image, PipelineConfig(shot_type=row["shot_type"],
                motion_intent=row["motion_intent"], decision_mode="rules"))
            visual = encoder.extract(image, result.localization) if encoder is not None else None
        entries.append({"localization": result.localization, "local_features": result.local_features,
                        "global_features": result.global_features, "rule_classification": result.classification,
                        "image_id": row["image_id"]})
        if visual is not None:
            entries[-1]["cnn_features"] = visual
        timings.append(time.perf_counter()-started)
        print(f"Extracted {row['image_id']}", flush=True)
    return entries, timings


def fit_creative_head(model, partitions, extracted, epochs=400, seed=42):
    """Only explicit photographer style labels train the optional style head."""
    labelled = {}
    for split in ("train", "validation"):
        pairs = [(entry, int(row["creative_style"] == "Creative"))
                 for row, entry in zip(partitions[split], extracted[split][0])
                 if row.get("creative_style") in ("Creative", "Standard")]
        labelled[split] = pairs
    if not any(labelled.values()):
        return {"trained": False, "reason": "No photographer style labels; automatic creative filtering is unavailable."}
    for split, pairs in labelled.items():
        if {label for _, label in pairs} != {0, 1}:
            return {"trained": False, "reason": f"{split} needs both Creative and Standard style labels."}
    model.fit_creative([e for e, _ in labelled["train"]], [y for _, y in labelled["train"]],
                       [e for e, _ in labelled["validation"]], [y for _, y in labelled["validation"]],
                       epochs=epochs, seed=seed)
    return {"trained": True, "label_counts": {s: len(pairs) for s, pairs in labelled.items()},
            "training_summary": model.creative_classifier.training_summary}


def evaluate(model, rows, entries, timings):
    labels = [r["target"] for r in rows]
    raw, predictions, styles = [], [], []
    started = time.perf_counter()
    for entry in entries:
        prediction = model.predict(entry)
        loc, lf = entry["localization"], entry["local_features"]
        insufficient = bool(missing_evidence_reasons(loc, lf))
        raw.append(int(prediction["classification"] == "Keep"))
        predictions.append(int(raw[-1] and not insufficient))
        styles.append(prediction.get("creative_style", "Unknown"))
    classifier_seconds = (time.perf_counter()-started)/len(rows)
    report = {"learned_classifier": classification_metrics(labels, raw),
              "application_with_evidence_checks": classification_metrics(labels, predictions),
              "always_keep": classification_metrics(labels, [1]*len(rows)),
              "current_rules": classification_metrics(labels, [int(e["rule_classification"] == "Keep") for e in entries]),
              "mean_feature_extraction_seconds": sum(timings)/len(timings),
              "mean_classifier_seconds": classifier_seconds,
              "evidence_coverage": {
                  "person_detector_unavailable": sum(e["localization"].get("person_detector_status") != "ok" for e in entries),
                  "photos_with_assessable_face": sum(e["local_features"].get("assessable_face_count", 0) > 0 for e in entries),
                  "photos_with_assessable_background": sum(bool(e["global_features"].get("background_assessable")) for e in entries)},
              "timing_note": "First extraction includes lazy face initialization; person initialization excluded.",
              "errors": [{"image_id": r["image_id"], "human": r["expert_score_usability"],
                          "predicted": "Keep" if p else "Review", "reason": r.get("review_reason", "")}
                         for r, p in zip(rows, predictions) if r["target"] != p]}
    for field in ("event_id", "shot_type"):
        report[field] = {}
        for value in sorted({r[field] for r in rows}):
            indices = [i for i, r in enumerate(rows) if r[field] == value]
            report[field][value] = classification_metrics([labels[i] for i in indices], [predictions[i] for i in indices])
    creative_keeps = [i for i, row in enumerate(rows) if row.get("creative_style") == "Creative" and labels[i] == 1]
    style_indices = [i for i, row in enumerate(rows) if row.get("creative_style") in ("Creative", "Standard")]
    report["creative_photos"] = {
        "style_head_available": model.creative_classifier is not None,
        "human_creative_keep_count": len(creative_keeps),
        "creative_keeps_sent_to_review": sum(predictions[i] == 0 for i in creative_keeps),
        "creative_keep_recall": sum(predictions[i] == 1 for i in creative_keeps)/len(creative_keeps) if creative_keeps else None,
        "style_labelled_photos": len(style_indices),
        "style_accuracy": sum(styles[i] == rows[i]["creative_style"] for i in style_indices)/len(style_indices)
                          if style_indices and model.creative_classifier is not None else None,
        "note": "Style recognition does not approve a photo; delivery suitability is assessed independently."}
    report["resolved_shot_type"] = {}
    resolved = [e["localization"]["assessment_context"]["shot_type"] for e in entries]
    for shot in sorted(set(resolved)):
        indices = [i for i, value in enumerate(resolved) if value == shot]
        report["resolved_shot_type"][shot] = classification_metrics([labels[i] for i in indices], [predictions[i] for i in indices])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT/"outputs"/"human_labels.csv"))
    parser.add_argument("--output-model", default=str(ROOT/"outputs"/"candidate_context_fusion_model.pt"))
    parser.add_argument("--report-json", default=str(ROOT/"outputs"/"context_fusion_evaluation.json"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", choices=("logistic", "cnn"), default="logistic")
    parser.add_argument("--input-mode", choices=("global", "subject", "combined"), default="combined")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate held-out events after freezing settings")
    parser.add_argument("--evaluate-model", help="Evaluate a frozen checkpoint on test events without retraining")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    partitions = load_human_partitions(args.manifest, ROOT)
    dataset_digest = audit_image_contents(partitions)
    manifest_digest = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if args.evaluate_model:
        model = ContextClassifier.load(args.evaluate_model)
        if (model.metadata.get("training_protocol") != "human_event_holdout_v1" or
                model.metadata.get("manifest_sha256") != manifest_digest or
                model.metadata.get("dataset_sha256") != dataset_digest or
                model.metadata.get("implementation_sha256") != implementation_fingerprint()):
            raise ValueError("Frozen checkpoint does not match this independently labelled manifest")
        pipeline = Phase4Pipeline()
        report = {"metadata": model.metadata, "threshold_keep": model.threshold,
                  "frozen_checkpoint_sha256": hashlib.sha256(Path(args.evaluate_model).read_bytes()).hexdigest(),
                  "test": evaluate(model, partitions["test"], *extract_rows(partitions["test"], pipeline, model.encoder))}
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Evaluated frozen checkpoint without retraining. Saved: {report_path}")
        return
    pipeline = Phase4Pipeline()
    encoder = None
    if args.model_type == "cnn":
        from src.cnn_features import FrozenResNetEncoder
        encoder = FrozenResNetEncoder(image_size=args.image_size)
    extracted = {split: extract_rows(partitions[split], pipeline, encoder) for split in ("train", "validation")}
    model = ContextClassifier(args.model_type, args.input_mode, encoder).fit(
        extracted["train"][0], [r["target"] for r in partitions["train"]], epochs=args.epochs, seed=args.seed,
        validation_entries=extracted["validation"][0], validation_labels=[r["target"] for r in partitions["validation"]])
    model.tune_threshold(extracted["validation"][0], [r["target"] for r in partitions["validation"]])
    creative_training = fit_creative_head(model, partitions, extracted, args.epochs, args.seed)
    model.metadata = {"training_protocol": "human_event_holdout_v1", "schema_version": SCHEMA_VERSION,
                      "epochs": args.epochs, "seed": args.seed,
                      "model_type": args.model_type, "input_mode": args.input_mode,
                      "training_summary": model.training_summary,
                      "creative_training": creative_training,
                      "manifest_sha256": manifest_digest,
                      "dataset_sha256": dataset_digest, "implementation_sha256": implementation_fingerprint(),
                      "events": {s: sorted({r["event_id"] for r in rows}) for s, rows in partitions.items()}}
    report = {"metadata": model.metadata, "threshold_keep": model.threshold,
              "validation": evaluate(model, partitions["validation"], *extracted["validation"]),
              "limitations": ["Human provenance is declared by annotators, not independently verified by software.",
                              "Exact duplicates are rejected; audit near duplicates manually.",
                              "Technical observations and model contributions are not ground-truth defect explanations.",
                              "This is a single candidate; use run_research_experiment.py for model/context comparisons."]}
    model.save(args.output_model)
    if args.evaluate_test:
        report["test"] = evaluate(model, partitions["test"], *extract_rows(partitions["test"], pipeline, encoder))
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved candidate: {args.output_model}\nSaved evaluation: {report_path}")
    print("Creative style head trained." if creative_training["trained"] else creative_training["reason"])
    print("Candidate is not activated. Compare frozen candidates on held-out events before deployment.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot train/evaluate the classifier: {exc}", file=sys.stderr)
        raise SystemExit(2)
