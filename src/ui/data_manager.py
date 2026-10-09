"""
Dataset Manifest & Baseline Results Data Access Layer
"""

import csv
from pathlib import Path
from typing import List, Dict, Optional


class DataManager:
    """Manages loading and querying dataset manifest entries and baseline scores."""

    BENCHMARK_PRESETS = [
        {"id": "bday_01", "label": "Sharp Portrait (bday_01)"},
        {"id": "bday_17", "label": "Motion Blur (bday_17)"},
        {"id": "bday_36", "label": "Turned Away (bday_36)"},
        {"id": "bday_44", "label": "Head Cropped (bday_44)"},
    ]

    @staticmethod
    def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
        """Loads records from dataset_manifest.csv."""
        if not manifest_path.is_file():
            return []
        records = []
        with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records

    @staticmethod
    def load_baselines(baseline_path: Path) -> Dict[str, str]:
        """Loads baseline scores from baseline_results.csv."""
        if not baseline_path.is_file():
            return {}
        baselines = {}
        with open(baseline_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("image_id")
                if img_id:
                    baselines[img_id] = row.get("predicted_score", "N/A")
        return baselines

    @staticmethod
    def filter_manifest(
        rows: List[Dict[str, str]],
        filter_option: str,
        event_option: str = "All Events",
    ) -> List[Dict[str, str]]:
        """Filters manifest rows by event and research defect scenario."""
        filtered = rows
        if event_option != "All Events":
            if "Event 01" in event_option:
                filtered = [r for r in filtered if r.get("event_id") == "birthday_shoot"]
            elif "Event 02" in event_option:
                filtered = [r for r in filtered if r.get("event_id") == "event_02_birthday"]

        if filter_option == "Keep / Usable Portraits":
            return [r for r in filtered if r.get("expert_score_usability") == "Keep"]
        elif filter_option == "Review / Flagged Photos":
            return [r for r in filtered if r.get("expert_score_usability") == "Review"]
        elif filter_option == "Motion Blur Defects":
            return [r for r in filtered if "Motion Blur" in r.get("local_defect_flag", "")]
        elif filter_option == "Turned Away Faces":
            return [r for r in filtered if "Face Turned" in r.get("local_defect_flag", "")]
        elif filter_option == "Head Cropped Defects":
            return [r for r in filtered if "Head Cropped" in r.get("local_defect_flag", "")]
        return filtered
