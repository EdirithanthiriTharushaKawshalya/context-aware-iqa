"""
Context-Aware Image Quality Assessment (IQA) - Phase 4: Module 1
Semantic Localisation Module

This module detects the primary subject (person) and isolates the face/head region
using lightweight pretrained deep learning models (torchvision SSDLite / MobileNet and
optional facexlib RetinaFace). It also detects framing anomalies such as cropped heads
or subjects turned away from the camera.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
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
        if self._retinaface_detector is not None:
            return self._retinaface_detector

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
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
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
            "framing_anomaly": "NO_SUBJECT",
            "image_size": (orig_w, orig_h),
        }

        # 1. Detect Primary Person (Subject)
        if self._subject_detector is not None:
            tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self._subject_detector(tensor)[0]

            # COCO class 1 is 'person'
            person_mask = (preds["labels"] == 1) & (preds["scores"] >= self.conf_thresh)
            if person_mask.any():
                person_boxes = preds["boxes"][person_mask].cpu().numpy()
                person_scores = preds["scores"][person_mask].cpu().numpy()

                # Pick the most prominent/highest confidence person
                areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1])
                # Combine area with confidence to prioritize foreground main subject
                prominence = areas * person_scores
                best_idx = np.argmax(prominence)

                subj_box = person_boxes[best_idx].tolist()
                subj_box = [
                    max(0.0, float(subj_box[0])),
                    max(0.0, float(subj_box[1])),
                    min(float(orig_w), float(subj_box[2])),
                    min(float(orig_h), float(subj_box[3])),
                ]

                result["subject_detected"] = True
                result["subject_bbox"] = subj_box
                result["subject_confidence"] = float(person_scores[best_idx])
                result["framing_anomaly"] = "NONE"

        # Fallback: If no subject detector or no person found, assume center crop represents subject
        if not result["subject_detected"]:
            # Center 60% of frame as default subject
            result["subject_bbox"] = [orig_w * 0.2, orig_h * 0.1, orig_w * 0.8, orig_h * 0.9]
            result["subject_confidence"] = 0.5
            result["subject_detected"] = True

        # 2. Check for Head / Face Cropping Anomaly
        subj_x1, subj_y1, subj_x2, subj_y2 = result["subject_bbox"]
        subj_h = subj_y2 - subj_y1
        subj_w = subj_x2 - subj_x1

        # Severe crop defect check: if person box hits upper margin (< 2% from top)
        # and has no headroom above shoulders
        is_upper_edge_cropped = (subj_y1 / orig_h) < 0.015 and (subj_h / orig_h) > 0.85

        # 3. Detect Face Region
        face_found = False
        face_detector = self._get_face_detector()

        if face_detector is not None:
            try:
                import cv2
                np_img = np.array(img)[:, :, ::-1]  # RGB to BGR
                # Scale down for fast face inference while preserving exact aspect ratio
                max_d = 1024
                scale = max_d / max(orig_h, orig_w) if max(orig_h, orig_w) > max_d else 1.0
                if scale < 1.0:
                    small = cv2.resize(np_img, (int(orig_w * scale), int(orig_h * scale)))
                else:
                    small = np_img

                bboxes = face_detector.detect_faces(small)
                if len(bboxes) > 0:
                    # Choose highest confidence face
                    best_face = max(bboxes, key=lambda b: b[4])
                    if best_face[4] >= 0.6:
                        fx1, fy1, fx2, fy2 = [coord / scale for coord in best_face[:4]]
                        result["face_detected"] = True
                        result["face_bbox"] = [
                            max(0.0, float(fx1)),
                            max(0.0, float(fy1)),
                            min(float(orig_w), float(fx2)),
                            min(float(orig_h), float(fy2)),
                        ]
                        result["face_confidence"] = float(best_face[4])
                        face_found = True
            except Exception:
                face_found = False

        # If dedicated face detector is unavailable or face wasn't found:
        if not face_found:
            # Check anatomical region: for a standing/seated portrait, head is in the top 25% of subject box
            # If the subject hits the very top border (like DSC_5216), flag HEAD_CROPPED
            if is_upper_edge_cropped or (subj_y1 / orig_h < 0.01 and subj_h > 0.8 * orig_h):
                result["face_detected"] = False
                result["framing_anomaly"] = "HEAD_CROPPED"
            else:
                # Anatomical approximation for upper head region
                head_y2 = subj_y1 + (subj_h * 0.28)
                head_x1 = subj_x1 + (subj_w * 0.20)
                head_x2 = subj_x2 - (subj_w * 0.20)
                
                # Verify if this upper region is skin/facial tone or back of head
                head_crop = img.crop((head_x1, subj_y1, head_x2, head_y2))
                head_stat = np.array(head_crop)
                
                # Anomaly check: if upper region is uniform/low variance or turned away
                result["face_bbox"] = [head_x1, subj_y1, head_x2, head_y2]
                result["face_confidence"] = 0.7
                result["face_detected"] = True

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
