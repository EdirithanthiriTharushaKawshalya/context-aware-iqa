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
        if min(gray_np.shape) < 2:
            return 0.0
        gx = np.gradient(gray_np.astype(np.float32), axis=1)
        gy = np.gradient(gray_np.astype(np.float32), axis=0)
        grad_norm = gx**2 + gy**2
        return float(np.mean(grad_norm))

    def _calc_laplacian_var(self, gray_pil: Image.Image) -> float:
        """Computes variance of Laplacian using Pillow filter."""
        edges = gray_pil.filter(ImageFilter.FIND_EDGES)
        if min(edges.size) > 2:
            edges = edges.crop((1, 1, edges.width-1, edges.height-1))
        stat = ImageStat.Stat(edges)
        return float(stat.var[0]) if stat.var else 0.0

    def _calc_motion_blur_ratio(self, gray_np: np.ndarray) -> float:
        """
        Estimates motion blur severity by measuring anisotropy of gradients
        (motion blur causes elongated gradients along one axis and suppressed gradients along the blur axis).
        """
        if min(gray_np.shape) < 2:
            return 0.0
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
        if "subjects" in localization:
            return self._extract_multi(img, localization)
        orig_w, orig_h = img.size
        framing_anomaly = localization.get("framing_anomaly", "NONE")
        has_face = localization.get("face_detected", False)
        subject_present = localization.get("subject_detected", True)
        
        # 1. Subject Region Analysis
        subj_box = localization.get("subject_bbox")
        if subj_box is not None:
            sx1, sy1, sx2, sy2 = [int(round(c)) for c in subj_box]
            sx1, sy1 = max(0, sx1), max(0, sy1)
            sx2, sy2 = min(orig_w, sx2), min(orig_h, sy2)
            subj_crop = img.crop((sx1, sy1, sx2, sy2)) if sx2 > sx1 and sy2 > sy1 else img
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
            face_crop = img.crop((fx1, fy1, fx2, fy2)) if fx2 > fx1 and fy2 > fy1 else subj_crop
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
        
        face_orientation = localization.get("face_orientation", "FRONTAL" if has_face else "TURNED_AWAY")
        face_yaw_deg = localization.get("face_yaw_deg", 0.0)
        face_yaw_asym = localization.get("face_yaw_asymmetry", 0.0)

        # Weighted combination of face sharpness and subject sharpness
        if has_face:
            if face_orientation == "PROFILE":
                # Semi-turned profile: slight adjustment
                local_score = 0.50 * norm_face_sharp + 0.50 * norm_subj_sharp
            else:
                # FRONTAL or SEMI_PROFILE (natural portrait): standard 60/40 weighting
                local_score = 0.60 * norm_face_sharp + 0.40 * norm_subj_sharp
        elif subject_present:
            # Missing face evidence does not establish a quality defect.
            local_score = norm_subj_sharp
        else:
            # No subject at all (landscape/background): score on scene sharpness alone.
            local_score = norm_subj_sharp

        # Defect deductions are applied once by the fusion engine.
        framing_defect = (framing_anomaly == "HEAD_CROPPED")

        local_score = float(np.clip(local_score, 0.0, 100.0))

        return {
            "face_sharpness": round(face_sharpness, 4),
            "face_tenengrad": round(face_tenengrad, 4),
            "subject_sharpness": round(subj_sharpness, 4),
            "local_contrast": round(subj_contrast, 4),
            "motion_blur_metric": round(motion_metric, 4),
            "framing_defect_detected": framing_defect,
            "face_present": has_face,
            "face_orientation": face_orientation,
            "face_yaw_deg": face_yaw_deg,
            "face_yaw_asymmetry": face_yaw_asym,
            "local_technical_score": round(local_score, 4),
        }

    def _extract_multi(self, img, localization):
        """Equal-person aggregation with lower-tail protection for couples/groups."""
        context = localization.get("assessment_context", {})
        records = []
        people = localization["subjects"] if context.get("shot_type") not in ("detail", "scene") else []
        for subject in people:
            features = self.extract(img, subject)
            face_box = subject.get("face_bbox")
            assessable = bool(face_box and min(face_box[2]-face_box[0], face_box[3]-face_box[1]) >= 24)
            if not assessable and features["face_present"]:
                # Tiny faces cannot support reliable face focus measurements.
                features = self.extract(img, dict(subject, face_detected=False))
            records.append(dict(features, subject_id=subject["subject_id"], face_assessable=assessable,
                                face_detected=subject.get("face_detected", False)))
        if not records:
            result = self.extract(img, {"subject_detected": False, "face_detected": False})
        else:
            result = dict(records[0])
            for key in ("face_sharpness", "face_tenengrad", "subject_sharpness", "local_contrast"):
                result[key] = round(float(np.mean([r[key] for r in records])), 4)
            scores = [r["local_technical_score"] for r in records]
            # A sharp partner cannot conceal the other partner being out of focus.
            lower = min(scores) if len(scores) == 2 else float(np.percentile(scores, 20))
            result["local_technical_score"] = round(.5*float(np.mean(scores)) + .5*lower, 4)
            result["motion_blur_metric"] = round(float(np.percentile([r["motion_blur_metric"] for r in records], 80)), 4)
            result["face_present"] = any(r["face_present"] for r in records)
        if context.get("motion_intent") == "intentional":
            # Retain a static anchor test: intentional movement is not a blanket
            # pass for an entirely unfocused frame. Evaluate sharp image tiles.
            gray = img.convert("L")
            tiles = []
            for row in range(4):
                for col in range(4):
                    tile = gray.crop((col*img.width//4, row*img.height//4,
                                      (col+1)*img.width//4, (row+1)*img.height//4))
                    if min(tile.size) >= 8:
                        tiles.append(float(np.clip(np.log1p(self._calc_laplacian_var(tile))*11, 0, 100)))
            anchor = float(np.percentile(tiles, 75)) if tiles else 0.
            result["static_anchor_score"] = round(anchor, 4)
            result["local_technical_score"] = round(.5*result["local_technical_score"] + .5*anchor, 4)
        result["subject_metrics"] = records
        result["assessable_face_count"] = sum(r["face_assessable"] for r in records)
        result["unassessable_subject_count"] = sum(not r["face_assessable"] for r in records)
        result["framing_defect_detected"] = False
        return result
