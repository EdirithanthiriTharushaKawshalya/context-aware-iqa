"""Create resumable machine assessments without modifying human ground truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from itertools import islice

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageOps
from src.pipeline import Phase4Pipeline, PipelineConfig
from src.training_data import implementation_fingerprint


def bounded_results(executor, assess, rows, limit):
    iterator = iter(rows)
    pending = deque(executor.submit(assess, row) for row in islice(iterator, limit))
    while pending:
        yield pending.popleft().result()
        row = next(iterator, None)
        if row is not None:
            pending.append(executor.submit(assess, row))


def replace_snapshot(temporary, output):
    # Windows readers/indexers may briefly deny replacement. Keep the previous
    # complete snapshot until replacement succeeds, and bound retries.
    for attempt in range(8):
        try:
            temporary.replace(output)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05*(attempt+1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT/"outputs"/"human_labels.csv"))
    parser.add_argument("--output", default=str(ROOT/"outputs"/"automated_photographic_assessments.jsonl"))
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="Independent assessment engines; each consumes additional memory")
    args = parser.parse_args()
    if args.max_side < 256:
        parser.error("--max-side must be at least 256")
    if args.threads < 1 or args.workers < 1:
        parser.error("--threads and --workers must be positive")
    import torch
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    if output.resolve() == Path(args.manifest).resolve():
        parser.error("The machine output must not overwrite the annotation manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = implementation_fingerprint()
    cached = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                cached[record["image_id"]] = record
            except (ValueError, KeyError):
                # A partial final line from an interrupted write is retried.
                continue
    with open(args.manifest, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    engines = threading.local()
    def assess(row):
        path = ROOT/row["file_path"]
        previous = cached.get(row["image_id"], {})
        record = {"image_id": row["image_id"], "file_path": row["file_path"],
                  "label_source": "automated_rules", "is_human_judgment": False,
                  "implementation_sha256": fingerprint, "max_side": args.max_side}
        try:
            with path.open("rb") as handle:
                record["source_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
            if (all(previous.get(k) == v for k, v in record.items()) and "error" not in previous):
                return previous
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((args.max_side, args.max_side))
            if not hasattr(engines, "pipeline"):
                engines.pipeline = Phase4Pipeline()
            result = engines.pipeline.run(image, PipelineConfig(decision_mode="rules"))
            record.update(predicted_decision=result.classification, score=result.unified_score,
                          photographic_profiles=result.photographic_profiles,
                          possible_creative=result.possible_creative,
                          review_reasons=result.review_reasons, notes=result.assessment_notes,
                          assessed_size=list(image.size),
                          limitation="Preview-resolution rules; not artistic ground truth or original-resolution focus validation.")
        except (OSError, ValueError, RuntimeError) as exc:
            record["error"] = str(exc)
        return record

    records = []
    current_ids = {row["image_id"] for row in rows}
    executor = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = bounded_results(executor, assess, rows, args.workers*2) if executor else map(assess, rows)
        for index, record in enumerate(results, 1):
            records.append(record)
            # Only this coordinator writes; workers never share inference engines.
            visited = {r["image_id"] for r in records}
            snapshot = records + [r for key, r in cached.items() if key not in visited and key in current_ids]
            temporary = output.with_suffix(output.suffix+".tmp")
            temporary.write_text("".join(json.dumps(r)+"\n" for r in snapshot), encoding="utf-8")
            replace_snapshot(temporary, output)
            print(f"{index}/{len(rows)} {record['image_id']}: {record.get('predicted_decision', 'Error')} "
                  f"{', '.join(record.get('photographic_profiles', []))}", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    summary = {"photos": len(records), "source": "automated_rules", "human_labels_added": 0,
               "errors": sum("error" in r for r in records),
               "possible_creative": sum(r.get("possible_creative", False) for r in records),
               "keep": sum(r.get("predicted_decision") == "Keep" for r in records),
               "review": sum(r.get("predicted_decision") == "Review" for r in records),
               "profiles": {p: sum(p in r.get("photographic_profiles", []) for r in records)
                            for p in sorted({p for r in records for p in r.get("photographic_profiles", [])})},
               "implementation_sha256": fingerprint, "max_side": args.max_side,
               "note": "Never use these predictions as independent labels to train or validate the same system."}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
