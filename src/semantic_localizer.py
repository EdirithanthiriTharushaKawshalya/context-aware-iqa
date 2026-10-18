"""
Context-Aware Image Quality Assessment (IQA) - Phase 4: Module 1
Semantic Localisation Module

This module retains detected people and associates each with a face region
using lightweight pretrained deep learning models (torchvision SSDLite / MobileNet and
optional facexlib RetinaFace). It also detects framing anomalies such as cropped heads
without treating missing detections as confirmed defects.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.functional as TF


class SemanticLocalizer:
    """
    Localizes the primary human subject and face/head region in an event photograph.
    Preserves original aspect ratio and coordinates.
    """

    def __init__(self, device: str = "auto", confidence_threshold: float = 0.5):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.conf_thresh = confidence_threshold
        self._subject_detector = None
        self._retinaface_detector = None
        self._face_init_attempted = False
        self._init_subject_detector()

    def _init_subject_detector(self):
        """Initializes torchvision SSDLite MobileNetV3 for COCO person detection."""
        try:
            import torchvision.models.detection as detection
            weights = detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = detection.ssdlite320_mobilenet_v3_large(weights=weights)
            model.to(self.device)
            model.eval()
            self._subject_detector = model
        except Exception as e:
            print(f"[WARNING] Could not initialize SSDLite person detector: {e}")
            self._subject_detector = None

    def _get_face_detector(self):
        """Lazy-loads RetinaFace detector if available."""
        if self._retinaface_detector is not None or self._face_init_attempted:
            return self._retinaface_detector

        self._face_init_attempted = True
        try:
            from facexlib.detection import init_detection_model
            # Use CPU or device
            model = init_detection_model("retinaface_resnet50", half=False, device=str(self.device))
            self._retinaface_detector = model
            return model
        except Exception:
            return None

    def localize(self, image_input: Any) -> Dict[str, Any]:
        """
        Takes image path or PIL Image and returns structured semantic localization result.

        Returns dictionary with:
            - 'subject_detected': bool
            - 'subject_bbox': [x1, y1, x2, y2] in original image coordinates
            - 'subject_confidence': float
            - 'face_detected': bool
            - 'face_bbox': [x1, y1, x2, y2] in original image coordinates or None
            - 'face_confidence': float
            - 'framing_anomaly': str ('NONE', 'HEAD_CROPPED', 'FACE_NOT_FOUND', 'NO_SUBJECT')
            - 'image_size': (width, height)
        """
        # Load and validate PIL Image
        if isinstance(image_input, (str, os.PathLike)):
            with Image.open(image_input) as source:
                img = ImageOps.exif_transpose(source).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = ImageOps.exif_transpose(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        orig_w, orig_h = img.size

        result = {
            "subject_detected": False,
            "subject_bbox": None,
            "subject_confidence": 0.0,
            "face_detected": False,
            "face_bbox": None,
            "face_confidence": 0.0,
            "face_landmarks": None,
            "face_yaw_asymmetry": 0.0,
            "face_yaw_deg": 0.0,
            "face_orientation": "NO_SUBJECT",
            "framing_anomaly": "NO_SUBJECT",
            "image_size": (orig_w, orig_h),
            "subjects": [], "faces": [],
            "person_detector_status": "ok" if self._subject_detector is not None else "unavailable",
            "face_detector_status": "unavailable",
        }

        # 1. Detect Primary Person (Subject)
        if self._subject_detector is not None:
            try:
                tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    preds = self._subject_detector(tensor)[0]

                # COCO class 1 is 'person'
                person_mask = (preds["labels"] == 1) & (preds["scores"] >= self.conf_thresh)
                if person_mask.any():
                    person_boxes = preds["boxes"][person_mask].cpu().numpy()
                    person_scores = preds["scores"][person_mask].cpu().numpy()

                    for box, score in zip(person_boxes, person_scores):
                        x1, y1, x2, y2 = box.tolist()
                        clipped = [max(0., x1), max(0., y1), min(float(orig_w), x2), min(float(orig_h), y2)]
                        if clipped[2]-clipped[0] >= 2 and clipped[3]-clipped[1] >= 2:
                            result["subjects"].append({"subject_bbox": clipped, "subject_confidence": float(score),
                                "face_detected": False, "face_bbox": None, "face_confidence": 0.,
                                "face_orientation": "UNKNOWN", "framing_anomaly": "NONE"})
                    if result["subjects"]:
                        primary = max(result["subjects"], key=lambda s: (s["subject_bbox"][2]-s["subject_bbox"][0]) *
                            (s["subject_bbox"][3]-s["subject_bbox"][1])*s["subject_confidence"])
                        result.update(subject_detected=True, subject_bbox=primary["subject_bbox"],
                                      subject_confidence=primary["subject_confidence"])

            except Exception:
                result["person_detector_status"] = "failed"
                result["subjects"] = []
                result["subject_detected"] = False

        # No fabricated subject: scene measurements use the full image.
        if not result["subject_detected"]:
            result["subject_bbox"] = [0., 0., float(orig_w), float(orig_h)]

        # Detect faces across the full frame, then associate them with people.
        face_detector = self._get_face_detector()

        if face_detector is not None:
            try:
                import cv2
                np_img = np.array(img)[:, :, ::-1]  # RGB to BGR
                # Scale down for fast face inference while preserving exact aspect ratio
                max_d = 1600
                scale = max_d / max(orig_h, orig_w) if max(orig_h, orig_w) > max_d else 1.0
                if scale < 1.0:
                    small = cv2.resize(np_img, (int(orig_w * scale), int(orig_h * scale)))
                else:
                    small = np_img

                with torch.no_grad():
                    bboxes = face_detector.detect_faces(small)
                result["face_detector_status"] = "ok"
                for detected in bboxes:
                    if detected[4] < .6:
                        continue
                    box = [float(v)/scale for v in detected[:4]]
                    box = [max(0., box[0]), max(0., box[1]), min(float(orig_w), box[2]), min(float(orig_h), box[3])]
                    if box[2]-box[0] < 2 or box[3]-box[1] < 2:
                        continue
                    face = {"face_bbox": box, "face_confidence": float(detected[4]), "face_orientation": "UNKNOWN"}
                    if len(detected) >= 15:
                        points = np.asarray(detected[5:15], dtype=float).reshape(5, 2)/scale
                        dl, dr = np.linalg.norm(points[2]-points[0]), np.linalg.norm(points[2]-points[1])
                        asym = float(abs(dl-dr)/(dl+dr+1e-6))
                        face.update(face_yaw_asymmetry=asym, face_yaw_deg=asym*90,
                            face_orientation="PROFILE" if asym > .55 else "FRONTAL")
                    result["faces"].append(face)
            except Exception:
                result["face_detector_status"] = "failed"
                result["faces"] = []

        # Match every face once to the smallest containing upper-body box.
        subjects = result["subjects"]
        subjects.sort(key=lambda s: (s["subject_bbox"][2]-s["subject_bbox"][0]) *
            (s["subject_bbox"][3]-s["subject_bbox"][1]) * s["subject_confidence"], reverse=True)
        candidates = []
        for fi, face in enumerate(result["faces"]):
            x1, y1, x2, y2 = face["face_bbox"]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            for si, subject in enumerate(subjects):
                a, b, c, d = subject["subject_bbox"]
                if a <= cx <= c and b <= cy <= b + .6*(d-b):
                    candidates.append(((c-a)*(d-b), -face["face_confidence"], si, fi))
        used_subjects, used_faces = set(), set()
        for _, _, si, fi in sorted(candidates):
            if si not in used_subjects and fi not in used_faces:
                subjects[si].update(result["faces"][fi], face_detected=True)
                result["faces"][fi]["subject_id"] = si
                used_subjects.add(si)
                used_faces.add(fi)
        for si, subject in enumerate(subjects):
            subject["subject_id"] = si
            subject["face_detector_status"] = result["face_detector_status"]
            subject["possible_head_crop"] = not subject["face_detected"] and subject["subject_bbox"][1] < .015*orig_h
        primary = subjects[0] if subjects else {}
        for key in ("face_detected", "face_bbox", "face_confidence", "face_orientation"):
            result[key] = primary.get(key, {"face_detected": False, "face_bbox": None,
                "face_confidence": 0., "face_orientation": "NO_SUBJECT"}[key])
        result["face_yaw_asymmetry"] = primary.get("face_yaw_asymmetry", 0.)
        result["face_yaw_deg"] = primary.get("face_yaw_deg", 0.)
        result["framing_anomaly"] = "NONE" if subjects else "NO_SUBJECT"
        result["subject_count"] = len(subjects)
        result["face_count"] = len(result["faces"])
        result["unmatched_face_count"] = len(result["faces"])-len(used_faces)

        return result

    def extract_crops(
        self,
        img: Image.Image,
        localization: Dict[str, Any],
        margin: float = 0.05
    ) -> Dict[str, Image.Image]:
        """
        Crops subject and face regions preserving spatial context.
        """
        w, h = img.size
        crops = {}

        if localization.get("subject_bbox"):
            x1, y1, x2, y2 = localization["subject_bbox"]
            dx = (x2 - x1) * margin
            dy = (y2 - y1) * margin
            crops["subject"] = img.crop((
                max(0, x1 - dx),
                max(0, y1 - dy),
                min(w, x2 + dx),
                min(h, y2 + dy),
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
        else:
            crops["face"] = None

        return crops
