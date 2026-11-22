"""Automatic photographic candidates, never human labels or proof of intent.

Thresholds are conservative engineering defaults, not learned artistic judgments.
Person boxes are approximate; ambiguous dark/soft images remain Review.
"""
import math

import numpy as np
from PIL import ImageFilter


def inspect_photographic_intent(image, localization, local, global_features, exposure_time=None):
    # Bound memory on full-resolution camera files. These are composition/
    # contour candidates, not a replacement for full-resolution focus checks.
    original_width, original_height = image.size
    image = image.copy()
    image.thumbnail((1280, 1280))
    scale_x, scale_y = image.width/original_width, image.height/original_height
    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32)
    profiles, notes, silhouettes = [], [], []
    low_light = float(np.mean(gray < 65)) >= .65
    if low_light:
        profiles.append("Low-light / low-key")
        notes.append("Low-light appearance detected. Darkness alone does not establish underexposure.")
    try:
        shutter = float(exposure_time)
        slow_shutter = math.isfinite(shutter) and shutter >= 1/30
    except (TypeError, ValueError, ZeroDivisionError):
        slow_shutter = False

    context = localization.get("assessment_context", {})
    subjects = localization.get("subjects", [])
    if not subjects and localization.get("subject_bbox"):
        subjects = [{"subject_bbox": localization["subject_bbox"]}]
    # Explicit detail/scene assessment must not adopt incidental people as intent.
    if context.get("shot_type") in ("detail", "scene"):
        subjects = []
    scaled_subjects = []
    for subject in subjects:
        scaled = dict(subject)
        for key in ("subject_bbox", "face_bbox"):
            box = subject.get(key)
            if box and len(box) == 4:
                scaled[key] = [v*(scale_x if i % 2 == 0 else scale_y) for i, v in enumerate(box)]
        scaled_subjects.append(scaled)
    subjects = scaled_subjects
    residual = np.abs(gray - np.asarray(gray_image.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32))
    if min(gray.shape) >= 3:
        gy, gx = np.gradient(gray)
        gradient = np.hypot(gx, gy)
    else:
        gradient = np.zeros_like(gray)
    for subject in subjects:
        box = subject.get("subject_bbox")
        if not box or len(box) != 4 or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if min(x2-x1, y2-y1) < 24:
            continue
        # Inner body avoids letting bright sky around a loose box dominate.
        dx, dy = (x2-x1)//4, (y2-y1)//4
        core = gray[y1+dy:y2-dy, x1+dx:x2-dx]
        # A black shirt against a bright wall is not silhouette evidence.
        # Require darkness in the head as well as the body's inner region.
        head = gray[y1+max(1,(y2-y1)//20):y1+max(2,(y2-y1)//4), x1+dx:x2-dx]
        face_box = subject.get("face_bbox")
        if subject.get("face_detected") and face_box and len(face_box) == 4 and np.isfinite(face_box).all():
            a, b, c, d = map(int, face_box)
            head = gray[max(0,b):max(0,min(image.height,d)), max(0,a):max(0,min(image.width,c))]
        pad = max(8, (x2-x1)//3)
        rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(image.width, x2+pad), min(image.height, y2+pad)
        ring = np.ones((ry2-ry1, rx2-rx1), dtype=bool)
        # Exclude every person, so one bright partner cannot act as a bright sky.
        for other in subjects:
            other_box = other.get("subject_bbox")
            if other_box and len(other_box) == 4 and np.isfinite(other_box).all():
                a, b, c, d = map(int, other_box)
                ring[max(0,b-ry1):max(0,min(ry2,d)-ry1), max(0,a-rx1):max(0,min(rx2,c)-rx1)] = False
        background = gray[ry1:ry2, rx1:rx2][ring]
        if background.size < 100 or not core.size:
            continue
        dark = gray[y1:y2, x1:x2] < 45
        edge = (gradient[y1:y2, x1:x2] >= 20) & dark
        contrast = float(np.median(background)-np.median(core))
        crispness = float(np.median(residual[y1:y2, x1:x2][edge]/
                                    gradient[y1:y2, x1:x2][edge])) if edge.any() else 0.
        candidate = (head.size > 0 and np.mean(head < 45) >= .6 and
                     np.mean(core < 40) >= .7 and np.median(background) >= 100 and contrast >= 75)
        if candidate:
            silhouettes.append({"subject_id": subject.get("subject_id"), "edge_pixels": int(edge.sum()),
                                "edge_crispness": round(crispness, 4), "contrast": round(contrast, 2),
                                "supported": bool(edge.sum() >= 24 and crispness >= .18 and
                                                  global_features.get("highlight_clipping_ratio", 0) < .15)})
    if silhouettes:
        profiles.append("Possible silhouette")
        notes.append("Dark subject against a brighter background: possible silhouette, not proof of sunset, beach or intent.")
    supported_silhouette = bool(subjects and len(silhouettes) == len(subjects) and
                                all(record["supported"] for record in silhouettes))
    sharp_subject = (local.get("local_technical_score", 0) >= 55 and
                     local.get("motion_blur_metric", 1) <= .45)
    separated_blur = (sharp_subject and global_features.get("background_assessable") and
                      global_features.get("background_motion_metric", 0) > .45)
    if separated_blur:
        profiles.append("Sharp subject / soft background")
        notes.append("Subject detail is stronger than background detail. Background softness may be bokeh or movement; vehicles are not identified.")
    motion = local.get("motion_blur_metric", 0) > .45
    if (slow_shutter and subjects) or (motion and subjects):
        profiles.append("Possible slow-shutter / dreamy motion")
        notes.append("Slow exposure or subject softness may be deliberate. Artistic success cannot be inferred from shutter speed or blur alone.")
    return {"profiles": profiles, "notes": notes, "silhouette_evidence": silhouettes,
            "supported_silhouette": supported_silhouette, "low_light": low_light,
            "sharp_subject": sharp_subject, "separated_background_blur": bool(separated_blur),
            "slow_shutter_metadata": slow_shutter,
            "possible_creative": bool(silhouettes or separated_blur or (subjects and (slow_shutter or motion)))}
