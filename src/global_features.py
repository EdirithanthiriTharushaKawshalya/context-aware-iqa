"""
Context-Aware Image Quality Assessment (IQA) - Phase 4: Module 3
Global Context Feature Extractor

This module captures whole-image contextual and aesthetic representations, including:
- Full-frame sharpness and luminance distribution
- Exposure quality (clipping in highlights / underexposure in shadows)
- Subject-to-background depth-of-field separation ratio (shallow DOF bokeh modeling)
- Global context score preserving whole-image composition
"""

from typing import Dict, Any, Optional
from PIL import Image, ImageFilter, ImageStat
import numpy as np


class GlobalFeatureExtractor:
    """
    Extracts whole-frame context features, including composition and depth-of-field separation.
    """

    def __init__(self):
        pass

    def extract(
        self,
        img: Image.Image,
        localization: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extracts global scene context features.

        Returns dictionary with:
            - 'global_sharpness': float
            - 'global_contrast': float
            - 'highlight_clipping_ratio': float
            - 'shadow_underexposure_ratio': float
            - 'dof_separation_ratio': float (subject sharpness / background sharpness)
            - 'global_context_score': float (0-100 scale)
        """
        orig_w, orig_h = img.size
        gray = img.convert("L")
        gray_np = np.array(gray)

        # 1. Whole-frame Sharpness & Contrast
        edges = gray.filter(ImageFilter.FIND_EDGES)
        if min(edges.size) > 2:
            edges = edges.crop((1, 1, edges.width-1, edges.height-1))
        edge_stat = ImageStat.Stat(edges)
        global_sharpness = float(edge_stat.var[0]) if edge_stat.var else 0.0

        lum_stat = ImageStat.Stat(gray)
        global_contrast = float(lum_stat.stddev[0]) if lum_stat.stddev else 0.0
        mean_lum = float(lum_stat.mean[0]) if lum_stat.mean else 128.0

        # 2. Exposure Quality (Overexposure / Underexposure)
        total_pixels = orig_w * orig_h
        highlight_clipping = float(np.sum(gray_np >= 250) / total_pixels)
        shadow_clipping = float(np.sum(gray_np <= 5) / total_pixels)

        # Penalty for severe over/under exposure (> 15% blown out or black)
        exposure_penalty = max(0.0, (highlight_clipping - 0.08) * 40.0) + max(0.0, (shadow_clipping - 0.08) * 40.0)

        # 3. Depth-of-Field (DOF) Separation Ratio
        # In professional event photography, a sharp subject on a softly blurred
        # background (bokeh) is a deliberate aesthetic choice, not a defect!
        dof_separation = 1.0
        boxes = []
        if localization and localization.get("subject_detected"):
            boxes = [s["subject_bbox"] for s in localization.get("subjects", [])]
            if not boxes and localization.get("subject_bbox"):
                boxes = [localization["subject_bbox"]]
        if boxes and min(orig_w, orig_h) >= 2:
            subject_mask = np.zeros((orig_h, orig_w), dtype=bool)
            for box in boxes:
                x1, y1, x2, y2 = [int(round(c)) for c in box]
                subject_mask[max(0,y1):min(orig_h,y2), max(0,x1):min(orig_w,x2)] = True
            bg_mask = ~subject_mask
            if bg_mask.sum() > 1000 and subject_mask.any():
                gy, gx = np.gradient(gray_np.astype(np.float32))
                energy = gx**2 + gy**2
                dof_separation = float((energy[subject_mask].mean()+1e-6)/(energy[bg_mask].mean()+1e-6))

        # 4. Composite Global Context Score (0 to 100)
        # Scaled sharpness + contrast + aesthetic DOF reward - exposure penalty
        norm_sharpness = float(np.clip(np.log1p(global_sharpness) * 10.0, 0.0, 100.0))
        norm_contrast = float(np.clip(global_contrast * 1.2, 0.0, 100.0))

        # Reward natural DOF separation (values between 1.2 and 5.0 indicate beautiful subject isolation)
        dof_bonus = float(np.clip((dof_separation - 1.0) * 4.0, 0.0, 15.0))

        context = (localization or {}).get("assessment_context", {})
        if context.get("shot_type") in ("group", "scene"):
            dof_bonus = 0.0
        global_score = (0.50 * norm_sharpness + 0.50 * norm_contrast + dof_bonus) - exposure_penalty
        global_score = float(np.clip(global_score, 0.0, 100.0))

        return {
            "global_sharpness": round(global_sharpness, 4),
            "global_contrast": round(global_contrast, 4),
            "mean_luminance": round(mean_lum, 4),
            "highlight_clipping_ratio": round(highlight_clipping, 4),
            "shadow_underexposure_ratio": round(shadow_clipping, 4),
            "dof_separation_ratio": round(dof_separation, 4),
            "global_context_score": round(global_score, 4),
        }
