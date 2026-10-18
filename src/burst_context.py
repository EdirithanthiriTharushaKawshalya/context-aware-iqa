"""
Context-Aware Image Quality Assessment (IQA) - Burst Context Engine
Analyzes temporal bursts of photographs to assess relative quality, detect moment-level degradation,
identify the "Hero Frame" (best shot in burst), and assist in professional photo culling.
"""

import os
import sys
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BurstFrameResult:
    """Quality and contextual assessment of an individual frame within a burst."""
    image_id: str
    file_path: str
    burst_id: str
    sequence_index: int
    raw_unified_score: float
    burst_rank: int
    burst_percentile: float
    burst_z_score: float
    is_hero_frame: bool
    is_burst_outlier_low: bool
    burst_recommendation: str  # "Hero Frame", "Keep (Good Alternative)", "Cull (Inferior Duplicate)", "Review (Defective)"
    defect_flag: Optional[str] = None
    delta_to_prev_frame: Optional[float] = None
    delta_to_hero_frame: float = 0.0


@dataclass
class BurstSummary:
    """Aggregate statistics and hero frame details for a burst sequence."""
    burst_id: str
    event_id: str
    total_frames: int
    hero_image_id: str
    hero_score: float
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    frames: List[BurstFrameResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["frames"] = [asdict(f) for f in self.frames]
        return d


class BurstContextEngine:
    """
    Evaluates multi-shot bursts to contextualize quality relative to adjacent frames.
    """

    def __init__(self, hero_margin_threshold: float = 2.0, cull_inferior_threshold: float = 5.0):
        self.hero_margin_thresh = hero_margin_threshold
        self.cull_inferior_thresh = cull_inferior_threshold

    def evaluate_burst(
        self,
        burst_id: str,
        frame_entries: List[Dict[str, Any]],  # List of dicts with 'image_id', 'file_path', 'unified_score', 'defect_flag'
        event_id: str = "event",
    ) -> BurstSummary:
        """
        Assesses all frames within a single burst.
        """
        if not frame_entries:
            raise ValueError(f"No frames provided for burst: {burst_id}")

        n = len(frame_entries)
        scores = [float(e.get("unified_score", 0.0)) for e in frame_entries]

        mean_s = sum(scores) / n
        std_s = math.sqrt(sum((s - mean_s) ** 2 for s in scores) / n) if n > 1 else 0.0
        max_s = max(scores)
        min_s = min(scores)

        # Rank frames by score descending
        sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
        ranks = [0] * n
        for rank_pos, idx in enumerate(sorted_indices, start=1):
            ranks[idx] = rank_pos

        hero_idx = sorted_indices[0]
        hero_id = frame_entries[hero_idx].get("image_id", f"frame_{hero_idx}")

        burst_frames = []
        for i, entry in enumerate(frame_entries):
            sc = scores[i]
            rank = ranks[i]
            z_score = (sc - mean_s) / (std_s + 1e-6)
            pct = round((n - rank) / max(1, n - 1) * 100.0, 1) if n > 1 else 100.0

            is_hero = (i == hero_idx)
            is_outlier_low = (z_score < -1.5) or (sc < 45.0)

            delta_prev = (sc - scores[i - 1]) if i > 0 else None
            delta_hero = round(sc - max_s, 2)

            # Determine professional culling recommendation
            d_flag = entry.get("defect_flag")
            if d_flag and d_flag not in ["None", "NaN", None]:
                rec = f"Review ({d_flag})"
            elif is_hero:
                rec = "Hero Frame (Sharpest / Best Moment)"
            elif abs(delta_hero) <= self.hero_margin_thresh:
                rec = "Keep (Close Alternative)"
            elif abs(delta_hero) <= self.cull_inferior_thresh:
                rec = "Candidate (Viable Backup)"
            else:
                rec = "Cull (Inferior Burst Duplicate)"

            burst_frames.append(BurstFrameResult(
                image_id=entry.get("image_id", f"img_{i}"),
                file_path=entry.get("file_path", ""),
                burst_id=burst_id,
                sequence_index=i + 1,
                raw_unified_score=round(sc, 2),
                burst_rank=rank,
                burst_percentile=pct,
                burst_z_score=round(z_score, 2),
                is_hero_frame=is_hero,
                is_burst_outlier_low=is_outlier_low,
                burst_recommendation=rec,
                defect_flag=d_flag if d_flag not in ["None", "NaN", None] else None,
                delta_to_prev_frame=round(delta_prev, 2) if delta_prev is not None else None,
                delta_to_hero_frame=delta_hero,
            ))

        return BurstSummary(
            burst_id=burst_id,
            event_id=event_id,
            total_frames=n,
            hero_image_id=hero_id,
            hero_score=round(max_s, 2),
            mean_score=round(mean_s, 2),
            std_score=round(std_s, 2),
            min_score=round(min_s, 2),
            max_score=round(max_s, 2),
            frames=burst_frames,
        )

    def print_burst_summary(self, summary: BurstSummary):
        """Prints a rich tabular inspection of the burst sequence."""
        print("\n" + "=" * 82)
        print(f"      BURST CONTEXT ANALYSIS: BURST '{summary.burst_id.upper()}' ({summary.total_frames} Frames)")
        print("=" * 82)
        print(f"Hero Frame (Top Pick):  {summary.hero_image_id} (Score: {summary.hero_score})")
        print(f"Burst Score Mean ± Std: {summary.mean_score:.2f} ± {summary.std_score:.2f} (Range: [{summary.min_score:.2f}, {summary.max_score:.2f}])")
        print("-" * 82)
        print(f"{'Seq':<4} | {'Image ID':<10} | {'Score':<7} | {'Rank':<5} | {'Z-Score':<8} | {'Delta Hero':<11} | {'Recommendation'}")
        print("-" * 82)
        for f in summary.frames:
            hero_star = "[*]" if f.is_hero_frame else "   "
            z_str = f"{f.burst_z_score:+.2f}"
            delta_str = f"{f.delta_to_hero_frame:+.2f}" if not f.is_hero_frame else "TOP"
            print(f"{f.sequence_index:<4} | {hero_star}{f.image_id:<8} | {f.raw_unified_score:<7.2f} | #{f.burst_rank:<4} | {z_str:<8} | {delta_str:<11} | {f.burst_recommendation}")
        print("=" * 82 + "\n")
