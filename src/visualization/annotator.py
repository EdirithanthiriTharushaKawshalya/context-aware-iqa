"""
Image Annotation & Visual Markup Module

Provides decoupled rendering of bounding boxes, telemetry tags,
and isolated subject/face crop extraction for high-resolution event photographs.
"""

import os
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont


class ImageAnnotator:
    """
    Renders clean, high-precision bounding boxes and metadata tags on PIL images
    without label collisions.
    """

    COLOR_SUBJECT = "#06b6d4"  # Cyan
    COLOR_FACE = "#f59e0b"     # Amber / Gold
    COLOR_DEFECT = "#ef4444"   # Red

    @staticmethod
    def get_font(size: int = 18) -> ImageFont.ImageFont:
        """Selects clean system font or falls back cleanly."""
        font_candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
        Draws bounding boxes and non-overlapping labels on an image copy.
        """
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        w, h = img.size

        # Crisp proportional line width and font sizing
        line_w = max(2, int(min(w, h) * 0.003))
        font_size = max(14, int(min(w, h) * 0.016))
        font = cls.get_font(font_size)

        # 1. Primary Subject Box
        subj_box = localization.get("subject_bbox")
        if show_subject and subj_box is not None and localization.get("subject_detected", False):
            sx1, sy1, sx2, sy2 = [float(c) for c in subj_box]
            conf = localization.get("subject_confidence", 0.0)
            subj_label = f"Subject [{conf*100:.0f}%]"

            # Draw crisp bounding box
            draw.rectangle([sx1, sy1, sx2, sy2], outline=cls.COLOR_SUBJECT, width=line_w)

            # Position Subject tag at BOTTOM-LEFT to avoid colliding with Face tag at top
            bbox = draw.textbbox((sx1, sy2), subj_label, font=font)
            bw = bbox[2] - bbox[0] + 12
            bh = bbox[3] - bbox[1] + 8

            # Place inside or just above bottom boundary
            tag_y1 = max(0, sy2 - bh)
            tag_y2 = tag_y1 + bh
            draw.rectangle([sx1, tag_y1, sx1 + bw, tag_y2], fill="#0e7490")
            draw.text((sx1 + 6, tag_y1 + 4), subj_label, fill="#ffffff", font=font)

        # 2. Face / Head Region Box
        face_box = localization.get("face_bbox")
        if show_face and face_box is not None and localization.get("face_detected", False):
            fx1, fy1, fx2, fy2 = [float(c) for c in face_box]
            fconf = localization.get("face_confidence", 0.0)
            face_label = f"Face [{fconf*100:.0f}%]"

            draw.rectangle([fx1, fy1, fx2, fy2], outline=cls.COLOR_FACE, width=line_w)

            # Position Face tag at TOP-LEFT
            bbox = draw.textbbox((fx1, fy1), face_label, font=font)
            bw = bbox[2] - bbox[0] + 12
            bh = bbox[3] - bbox[1] + 8
            tag_y1 = max(0, fy1 - bh)
            tag_y2 = tag_y1 + bh
            draw.rectangle([fx1, tag_y1, fx1 + bw, tag_y2], fill="#b45309")
            draw.text((fx1 + 6, tag_y1 + 4), face_label, fill="#ffffff", font=font)

        # 3. Framing Anomaly Warning Banner (Technical HUD style)
        if localization.get("framing_anomaly") == "HEAD_CROPPED":
            warn_msg = "ANOMALY: HEAD_CROPPED (UPPER BOUND OVERFLOW)"
            bbox = draw.textbbox((16, 16), warn_msg, font=font)
            bw = bbox[2] - bbox[0] + 20
            bh = bbox[3] - bbox[1] + 12
            draw.rectangle([16, 16, 16 + bw, 16 + bh], fill="#991b1b")
            draw.text((26, 22), warn_msg, fill="#ffffff", font=font)

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
