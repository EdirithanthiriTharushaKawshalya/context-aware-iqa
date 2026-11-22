"""Train six context/model candidates, freeze selection, then evaluate unseen events.

Train: python scripts/run_research_experiment.py --run-dir outputs/research_runs/run1
Test:  python scripts/run_research_experiment.py --evaluate-run outputs/research_runs/run1
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import torchvision

from scripts.train_context_classifier import extract_rows, evaluate, fit_creative_head
from src.context_classifier import ContextClassifier, classification_metrics, missing_evidence_reasons
from src.pipeline import Phase4Pipeline, PipelineConfig
from src.research_reporting import write_comparison, write_predictions
from src.training_data import audit_image_contents, implementation_fingerprint, load_human_partitions


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def contained_checkpoint(directory, name):
    path = (directory/name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Checkpoint is missing or outside this experiment")
    return path


def prediction_rows(name, model, rows, entries):
    records = []
    for row, entry in zip(rows, entries):
        prediction = model.predict(entry)
        reasons = missing_evidence_reasons(entry["localization"], entry["local_features"])
        records.append({"model": name, "image_id": row["image_id"], "event_id": row["event_id"],
                        "shot_type": entry["localization"]["assessment_context"]["shot_type"],
                        "human_label": row["expert_score_usability"],
                        "predicted_label": "Review" if reasons else prediction["classification"],
                        "probability_keep": prediction["probability_keep"],
                        "human_creative_style": row.get("creative_style", ""),
                        "predicted_creative_style": prediction.get("creative_style", "Unknown"),
                        "human_review_reason": row.get("review_reason", ""),
                        "evidence_check": " | ".join(reasons),
                        "review_evidence": json.dumps(prediction["review_evidence"])})
    return records


def benchmark(path, rows, pipeline):
    config = PipelineConfig(decision_mode="learned" if path is not None else "rules",
                            model_path=str(path) if path is not None else None)
    # Initialize checkpoint and lazy detectors outside timed samples.
    with Image.open(rows[0]["absolute_path"]) as image:
        pipeline.run(image, PipelineConfig(**{**config.__dict__, "shot_type": rows[0]["shot_type"],
                                             "motion_intent": rows[0]["motion_intent"]}))
    seconds = []
    for row in rows:
        config.shot_type, config.motion_intent = row["shot_type"], row["motion_intent"]
        start = time.perf_counter()
        with Image.open(row["absolute_path"]) as image:
            pipeline.run(image, config)
        seconds.append(time.perf_counter()-start)
    return {"images": len(seconds), "mean_seconds": float(np.mean(seconds)),
            "median_seconds": float(np.median(seconds)), "p95_seconds": float(np.percentile(seconds, 95)),
            "conditions": "CPU, source decoding plus complete normal pipeline; initialization/warm-up excluded"}


def musiq_report(baseline, rows):
    scores, seconds = [], []
    for row in rows:
        start = time.perf_counter()
        with Image.open(row["absolute_path"]) as image:
            scores.append(baseline.score(image))
        seconds.append(time.perf_counter()-start)
    return scores, {"mean_seconds": float(np.mean(seconds)), "images": len(seconds),
                    "conditions": "CPU, image decoding and MUSIQ scoring; no person/face detection"}


def run_training(args, partitions, metadata):
    directory = Path(args.run_dir).resolve()
    if (directory/"selection.json").exists():
        raise ValueError("This experiment is already frozen. Choose a new run directory.")
    directory.mkdir(parents=True, exist_ok=True)
    # A run cannot accidentally consume any test image features or labels in fit.
    pipeline = Phase4Pipeline()
    encoder = None
    if not args.logistic_only:
        from src.cnn_features import FrozenResNetEncoder
        encoder = FrozenResNetEncoder(image_size=args.image_size)
    extracted = {split: extract_rows(partitions[split], pipeline, encoder) for split in ("train", "validation")}
    reports, candidates, predictions = {}, {}, []
    for kind in (["logistic"] if args.logistic_only else ["logistic", "cnn"]):
        for mode in ("global", "subject", "combined"):
            name = f"{kind}_{mode}"
            print(f"Training {name}", flush=True)
            model = ContextClassifier(kind, mode, encoder if kind == "cnn" else None)
            model.fit(extracted["train"][0], [r["target"] for r in partitions["train"]],
                      epochs=args.epochs, seed=args.seed, validation_entries=extracted["validation"][0],
                      validation_labels=[r["target"] for r in partitions["validation"]])
            model.tune_threshold(extracted["validation"][0], [r["target"] for r in partitions["validation"]])
            creative_training = fit_creative_head(model, partitions, extracted, args.epochs, args.seed)
            model.metadata = dict(metadata, model_type=kind, input_mode=mode, seed=args.seed,
                                  training_summary=model.training_summary, creative_training=creative_training,
                                  epochs=args.epochs)
            path = directory/(name+".pt")
            model.save(path)
            reports[name] = evaluate(model, partitions["validation"], *extracted["validation"])
            predictions.extend(prediction_rows(name, model, partitions["validation"], extracted["validation"][0]))
            candidates[name] = {"checkpoint": path.name, "sha256": sha(path), "threshold_keep": model.threshold}
    labels = [r["target"] for r in partitions["validation"]]
    reports["always_keep"] = classification_metrics(labels, [1]*len(labels))
    reports["current_rules"] = classification_metrics(labels, [int(e["rule_classification"] == "Keep") for e in extracted["validation"][0]])
    musiq = None
    if args.musiq:
        from src.musiq_baseline import MusiqBaseline
        scorer = MusiqBaseline()
        scores, timing = musiq_report(scorer, partitions["validation"])
        scorer.tune(scores, labels)
        reports["musiq"] = dict(classification_metrics(labels, np.asarray(scores) >= scorer.threshold), runtime=timing)
        musiq_path = directory/"musiq_baseline.pt"
        scorer.save(musiq_path)
        musiq = {"threshold": scorer.threshold, "metric": "musiq", "checkpoint": musiq_path.name,
                 "sha256": sha(musiq_path)}
    # Selection is based only on validation, including application evidence checks.
    # Prefer the simpler logistic candidate on an exact metric tie.
    def objective(name):
        report = reports[name]["application_with_evidence_checks"]
        return report["macro_f1"], report["review_recall"], int(name.startswith("logistic")), int(name.endswith("combined"))
    selected = max(candidates, key=objective)
    selection = {"metadata": metadata, "candidates": candidates, "selected": selected,
                 "selection_basis": "validation application macro F1, then Review recall, simpler model, combined context",
                 "optional_musiq": musiq}
    (directory/"selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_comparison(directory, "validation", reports, dict(metadata, selected=selected))
    write_predictions(directory/"validation_predictions.csv", predictions)
    print(f"Frozen selection: {selected}. Evaluate unseen events with --evaluate-run {directory}")


def evaluate_run(args, partitions, metadata):
    directory = Path(args.evaluate_run).resolve()
    selection = json.loads((directory/"selection.json").read_text(encoding="utf-8"))
    for field in ("manifest_sha256", "dataset_sha256", "implementation_sha256"):
        if selection["metadata"].get(field) != metadata[field]:
            raise ValueError(f"Frozen experiment mismatch: {field}. Restore the original data/code before test evaluation.")
    if (directory/"test_results.json").exists():
        raise ValueError("Test results already exist for this frozen experiment; do not repeatedly tune against test events")
    pipeline, reports, predictions, extracted = Phase4Pipeline(), {}, [], None
    rows = partitions["test"]
    for name, saved in selection["candidates"].items():
        path = contained_checkpoint(directory, saved["checkpoint"])
        if sha(path) != saved["sha256"]:
            raise ValueError(f"Frozen checkpoint was modified: {name}")
        model = ContextClassifier.load(path)
        if model.metadata.get("manifest_sha256") != metadata["manifest_sha256"]:
            raise ValueError("Candidate provenance does not match this experiment")
        if extracted is None:
            extracted = extract_rows(rows, pipeline)
        entries, times = extracted
        if model.model_type == "cnn":
            cnn_entries, cnn_times = [], []
            for row, entry, seconds in zip(rows, entries, times):
                start = time.perf_counter()
                with Image.open(row["absolute_path"]) as image:
                    cnn_entries.append(model.prepare_entry(image, entry))
                cnn_times.append(seconds+time.perf_counter()-start)
            entries, times = cnn_entries, cnn_times
        reports[name] = evaluate(model, rows, entries, times)
        reports[name]["runtime"] = benchmark(path, rows, pipeline)
        predictions.extend(prediction_rows(name, model, rows, entries))
    labels = [r["target"] for r in rows]
    reports["always_keep"] = classification_metrics(labels, [1]*len(rows))
    reports["current_rules"] = classification_metrics(labels, [int(e["rule_classification"] == "Keep") for e in extracted[0]])
    reports["current_rules"]["runtime"] = benchmark(None, rows, pipeline)
    if selection.get("optional_musiq"):
        from src.musiq_baseline import MusiqBaseline
        saved = selection["optional_musiq"]
        path = contained_checkpoint(directory, saved["checkpoint"])
        if sha(path) != saved["sha256"]:
            raise ValueError("Frozen MUSIQ checkpoint was modified")
        scorer = MusiqBaseline.load(path)
        scores, timing = musiq_report(scorer, rows)
        reports["musiq"] = dict(classification_metrics(labels, np.asarray(scores) >= scorer.threshold), runtime=timing)
    write_comparison(directory, "test", reports, dict(metadata, selected_before_test=selection["selected"]))
    write_predictions(directory/"test_predictions.csv", predictions)
    selected_saved = selection["candidates"][selection["selected"]]
    selected_report = reports[selection["selected"]]["application_with_evidence_checks"]
    receipt = {"selected_before_test": selection["selected"], "checkpoint_sha256": selected_saved["sha256"],
               "test_metrics": selected_report, "held_out_evaluated": True,
               "macro_f1_change_vs_rules": selected_report["macro_f1"]-reports["current_rules"]["macro_f1"],
               "note": "Tested on this labelled dataset; this does not guarantee artistic judgment on other populations."}
    (directory/"evaluation_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if args.activate:
        # Atomic replacement only after the complete evaluation is written.
        destination = ROOT/"outputs"/"context_fusion_model.pt"
        temporary = destination.with_suffix(".pt.tmp")
        shutil.copyfile(contained_checkpoint(directory, selected_saved["checkpoint"]), temporary)
        temporary.replace(destination)
        print(f"Activated validation-selected model: {destination}")
    print(f"Saved frozen test comparisons: {directory/'test_results.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT/"outputs"/"human_labels.csv"))
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--run-dir")
    operation.add_argument("--evaluate-run")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--logistic-only", action="store_true", help="Run the three logistic context ablations only")
    parser.add_argument("--musiq", action="store_true", help="Add the optional pretrained MUSIQ comparison")
    parser.add_argument("--activate", action="store_true", help="Use the validation-selected checkpoint after frozen test evaluation")
    args = parser.parse_args()
    if args.epochs < 1 or not 224 <= args.image_size <= 1024:
        parser.error("Use positive epochs and an image size between 224 and 1024")
    if args.activate and not args.evaluate_run:
        parser.error("--activate requires --evaluate-run")
    partitions = load_human_partitions(args.manifest, ROOT)
    metadata = {"training_protocol": "human_event_holdout_v1", "manifest_sha256": sha(args.manifest),
                "dataset_sha256": audit_image_contents(partitions), "implementation_sha256": implementation_fingerprint(),
                "events": {s: sorted({r["event_id"] for r in rows}) for s, rows in partitions.items()},
                "environment": {"platform": platform.platform(), "python": platform.python_version(),
                                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "device": "cpu"}}
    if args.evaluate_run:
        evaluate_run(args, partitions, metadata)
    else:
        run_training(args, partitions, metadata)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot run the research experiment: {exc}", file=sys.stderr)
        raise SystemExit(2)
