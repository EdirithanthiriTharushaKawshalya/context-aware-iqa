"""Measure independent reviewer agreement and export disagreements for resolution."""
import argparse
import csv
import json
from pathlib import Path


def compare(first, second):
    def read(path):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        result = {}
        for row in rows:
            if not row.get("expert_score_usability"):
                continue
            if (row.get("label_source") != "human" or not row.get("labeler_id") or
                    row["expert_score_usability"] not in ("Keep", "Review")):
                raise ValueError("Agreement requires independent human annotations")
            if row["image_id"] in result:
                raise ValueError("Duplicate annotation image ID")
            result[row["image_id"]] = row
        return result
    a, b = read(first), read(second)
    shared = sorted(a.keys() & b.keys())
    if not shared:
        raise ValueError("No shared labelled photos")
    disagreements, matches = [], 0
    for image_id in shared:
        left, right = a[image_id], b[image_id]
        if left["labeler_id"] == right["labeler_id"]:
            raise ValueError("Shared photos must be labelled by different reviewers")
        if left.get("file_path") != right.get("file_path") or left.get("event_id") != right.get("event_id"):
            raise ValueError("Shared image IDs refer to different photos or events")
        if left["expert_score_usability"] == right["expert_score_usability"]:
            matches += 1
        else:
            disagreements.append({"image_id": image_id, "first_label": left["expert_score_usability"],
                "second_label": right["expert_score_usability"], "first_reason": left.get("review_reason", ""),
                "second_reason": right.get("review_reason", ""), "resolved_label": "", "resolution_reason": ""})
    count = len(shared)
    expected = sum(sum(a[i]["expert_score_usability"] == label for i in shared)/count *
                   sum(b[i]["expert_score_usability"] == label for i in shared)/count for label in ("Keep", "Review"))
    observed = matches/count
    return {"shared_photos": count, "agreement": observed,
            "cohen_kappa": (observed-expected)/(1-expected) if expected < 1 else None,
            "disagreements": disagreements,
            "note": "Kappa is undefined when both reviewers assign one identical class to every shared photo."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--output", default="outputs/reviewer_agreement.json")
    args = parser.parse_args()
    report = compare(args.first, args.second)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Agreement: {report['agreement']:.1%}; shared photos: {report['shared_photos']}; saved {path}")


if __name__ == "__main__":
    main()
