"""
Context-Aware Image Quality Assessment (IQA) - Phase 4: Module 2
Local Technical Feature Extractor

This module isolates the localized subject and face regions and computes
high-resolution local technical quality indicators, including:
- Face/Head sharpness and edge clarity (Modified Laplacian & Tenengrad gradient)
- Local contrast and dynamic range
- Motion blur / directional blur indicators
- Semantic presence flags
"""

from typing import Dict, Any, Optional
from PIL import Image, ImageFilter, ImageStat
import numpy as np


class LocalFeatureExtractor:
    """
    Extracts fine-grained technical quality features from localized subject and face regions.
    """

    def __init__(self):
        pass

    def _calc_tenengrad(self, gray_np: np.ndarray) -> float:
        """Computes Tenengrad gradient energy (Sobel-based sharpness indicator)."""
        gx = np.gradient(gray_np.astype(np.float32), axis=1)
        gy = np.gradient(gray_np.astype(np.float32), axis=0)
        grad_norm = gx**2 + gy**2
        return float(np.mean(grad_norm))

    def _calc_laplacian_var(self, gray_pil: Image.Image) -> float:
        """Computes variance of Laplacian using Pillow filter."""
        edges = gray_pil.filter(ImageFilter.FIND_EDGES)
        stat = ImageStat.Stat(edges)
        return float(stat.var[0]) if stat.var else 0.0

    def _calc_motion_blur_ratio(self, gray_np: np.ndarray) -> float:
        """
        Estimates motion blur severity by measuring anisotropy of gradients
        (motion blur causes elongated gradients along one axis and suppressed gradients along the blur axis).
        """
        gx = np.gradient(gray_np.astype(np.float32), axis=1)
        gy = np.gradient(gray_np.astype(np.float32), axis=0)
        var_x = float(np.var(gx))
        var_y = float(np.var(gy))
        total_energy = var_x + var_y + 1e-6
        # Ratio of directional asymmetry
        asymmetry = abs(var_x - var_y) / total_energy
        # Combine with low overall high-frequency energy
        motion_indicator = (1.0 / (1.0 + np.log1p(total_energy))) * (1.0 + asymmetry)
        return float(motion_indicator)

    def extract(
        self,
        img: Image.Image,
        localization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extracts local technical quality metrics for the image based on its localization.

        Returns dictionary containing:
            - 'face_sharpness': float
            - 'subject_sharpness': float
            - 'local_contrast': float
            - 'motion_blur_metric': float
            - 'framing_defect_detected': bool
            - 'local_technical_score': float (0-100 scale)
        """
        orig_w, orig_h = img.size
        framing_anomaly = localization.get("framing_anomaly", "NONE")
        has_face = localization.get("face_detected", False)
        
        # 1. Subject Region Analysis
        subj_box = localization.get("subject_bbox")
        if subj_box is not None:
            sx1, sy1, sx2, sy2 = [int(round(c)) for c in subj_box]
            sx1, sy1 = max(0, sx1), max(0, sy1)
            sx2, sy2 = min(orig_w, sx2), min(orig_h, sy2)
            subj_crop = img.crop((sx1, sy1, sx2, sy2))
        else:
            subj_crop = img

        subj_gray = subj_crop.convert("L")
        subj_np = np.array(subj_gray)
        subj_sharpness = self._calc_laplacian_var(subj_gray)
        subj_contrast = float(ImageStat.Stat(subj_gray).stddev[0]) if ImageStat.Stat(subj_gray).stddev else 0.0
        motion_metric = self._calc_motion_blur_ratio(subj_np)

        # 2. Face / Head Region Analysis
        face_box = localization.get("face_bbox")
        if has_face and face_box is not None:
            fx1, fy1, fx2, fy2 = [int(round(c)) for c in face_box]
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(orig_w, fx2), min(orig_h, fy2)
            face_crop = img.crop((fx1, fy1, fx2, fy2))
            face_gray = face_crop.convert("L")
            face_sharpness = self._calc_laplacian_var(face_gray)
            face_tenengrad = self._calc_tenengrad(np.array(face_gray))
        else:
            face_sharpness = 0.0
            face_tenengrad = 0.0

        # 3. Composite Local Technical Score (0 to 100)
        # In a usable portrait:
        # - Subject must have clear edge definition
        # - Face must be present and in focus
        # - Severe motion blur must heavily penalize the score
        # - Framing defects (head cut off) severely penalize the score
        
        # Base sharpness score (scaled logarithmically)
        norm_subj_sharp = float(np.clip(np.log1p(subj_sharpness) * 11.0, 0.0, 100.0))
        norm_face_sharp = float(np.clip(np.log1p(face_sharpness) * 12.0, 0.0, 100.0)) if has_face else 0.0
        
        # Weighted combination of face sharpness (60%) and subject sharpness (40%)
        if has_face:
            local_score = 0.60 * norm_face_sharp + 0.40 * norm_subj_sharp
        else:
            # If face is missing or turned away, subject sharpness is discounted heavily
            local_score = 0.30 * norm_subj_sharp

        # Penalty for motion blur
        if motion_metric > 0.45:
            # Scale penalty according to blur severity
            blur_penalty = float(np.clip((motion_metric - 0.45) * 50.0, 0.0, 40.0))
            local_score -= blur_penalty

        # Penalty for framing anomaly (e.g. head cropped out)
        framing_defect = (framing_anomaly == "HEAD_CROPPED")
        if framing_defect:
            local_score = min(local_score, 20.0)  # Capped at defective level

        local_score = float(np.clip(local_score, 0.0, 100.0))

        return {
            "face_sharpness": round(face_sharpness, 4),
            "face_tenengrad": round(face_tenengrad, 4),
            "subject_sharpness": round(subj_sharpness, 4),
            "local_contrast": round(subj_contrast, 4),
            "motion_blur_metric": round(motion_metric, 4),
            "framing_defect_detected": framing_defect,
            "face_present": has_face,
            "local_technical_score": round(local_score, 4),
        }
