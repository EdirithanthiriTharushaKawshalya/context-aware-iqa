"""
Image Annotation & Visual Markup Module

Provides decoupled rendering of bounding boxes, anomaly warning banners,
and isolated subject/face crop extraction for high-resolution event photographs.
"""

import os
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont


class ImageAnnotator:
    """
    Renders high-contrast bounding boxes, badges, and warning overlays on PIL images.
    """

    COLOR_SUBJECT_BOX = "#06b6d4"  # Cyan
    COLOR_SUBJECT_BADGE = "#0891b2"
    COLOR_FACE_BOX = "#f59e0b"     # Amber/Gold
    COLOR_FACE_BADGE = "#d97706"
    COLOR_DEFECT_BADGE = "#dc2626"   # Red

    @staticmethod
    def get_font(size: int = 24) -> ImageFont.ImageFont:
        """Selects available system font or falls back cleanly."""
        font_candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for path in font_candidates:
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    @classmethod
    def draw_annotations(
        cls,
        img: Image.Image,
        localization: Dict[str, Any],
        show_subject: bool = True,
        show_face: bool = True,
    ) -> Image.Image:
        """
        Draws bounding boxes and metadata badges on an image copy.
        """
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        w, h = img.size

        # Proportional line width and typography scaling
        line_w = max(4, int(min(w, h) * 0.005))
        font_size = max(18, int(min(w, h) * 0.022))
        font = cls.get_font(font_size)

        # 1. Primary Subject Box
        subj_box = localization.get("subject_bbox")
        if show_subject and subj_box is not None and localization.get("subject_detected", False):
            sx1, sy1, sx2, sy2 = subj_box
            conf = localization.get("subject_confidence", 0.0)
            label = f"Primary Subject ({conf*100:.1f}%)"

            draw.rectangle([sx1, sy1, sx2, sy2], outline=cls.COLOR_SUBJECT_BOX, width=line_w)

            # Badge background
            bbox = draw.textbbox((sx1, sy1), label, font=font)
            bw = bbox[2] - bbox[0] + 16
            bh = bbox[3] - bbox[1] + 12
            by1 = max(0, sy1 - bh)
            by2 = by1 + bh
            draw.rectangle([sx1, by1, sx1 + bw, by2], fill=cls.COLOR_SUBJECT_BADGE)
            draw.text((sx1 + 8, by1 + 4), label, fill="#ffffff", font=font)

        # 2. Face / Head Region Box
        face_box = localization.get("face_bbox")
        if show_face and face_box is not None and localization.get("face_detected", False):
            fx1, fy1, fx2, fy2 = face_box
            fconf = localization.get("face_confidence", 0.0)
            flabel = f"Face / Head ({fconf*100:.1f}%)"

            draw.rectangle([fx1, fy1, fx2, fy2], outline=cls.COLOR_FACE_BOX, width=line_w)

            bbox = draw.textbbox((fx1, fy1), flabel, font=font)
            bw = bbox[2] - bbox[0] + 16
            bh = bbox[3] - bbox[1] + 12
            by1 = max(0, fy1 - bh)
            by2 = by1 + bh
            draw.rectangle([fx1, by1, fx1 + bw, by2], fill=cls.COLOR_FACE_BADGE)
            draw.text((fx1 + 8, by1 + 4), flabel, fill="#ffffff", font=font)

        # 3. Framing Anomaly Warning Banner
        if localization.get("framing_anomaly") == "HEAD_CROPPED":
            warn_msg = "⚠️ DEFECT DETECTED: HEAD / FACE CROPPED OUT"
            bbox = draw.textbbox((20, 20), warn_msg, font=font)
            bw = bbox[2] - bbox[0] + 28
            bh = bbox[3] - bbox[1] + 18
            draw.rectangle([20, 20, 20 + bw, 20 + bh], fill=cls.COLOR_DEFECT_BADGE)
            draw.text((34, 28), warn_msg, fill="#ffffff", font=font)

        return annotated

    @staticmethod
    def extract_crops(
        img: Image.Image,
        localization: Dict[str, Any],
        margin: float = 0.05,
    ) -> Dict[str, Optional[Image.Image]]:
        """
        Extracts cropped regions for the primary subject and isolated face.
        """
        w, h = img.size
        crops: Dict[str, Optional[Image.Image]] = {"subject": None, "face": None}

        if localization.get("subject_bbox"):
            sx1, sy1, sx2, sy2 = localization["subject_bbox"]
            dx = (sx2 - sx1) * margin
            dy = (sy2 - sy1) * margin
            crops["subject"] = img.crop((
                max(0, sx1 - dx),
                max(0, sy1 - dy),
                min(w, sx2 + dx),
                min(h, sy2 + dy),
            ))

        if localization.get("face_bbox") and localization.get("face_detected"):
            fx1, fy1, fx2, fy2 = localization["face_bbox"]
            fdx = (fx2 - fx1) * margin
            fdy = (fy2 - fy1) * margin
            crops["face"] = img.crop((
                max(0, fx1 - fdx),
                max(0, fy1 - fdy),
                min(w, fx2 + fdx),
                min(h, fy2 + fdy),
            ))

        return crops
