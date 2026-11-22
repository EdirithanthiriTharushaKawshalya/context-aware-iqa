"""Frozen ResNet18 visual features with aspect-preserving image/crop inputs."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.models import ResNet18_Weights, resnet18

EMBEDDING_SIZE = 512


def pretrained_weights_path(download=True):
    filename = "resnet18-f37072fd.pth"
    cached = Path(torch.hub.get_dir())/"checkpoints"/filename
    local = Path(__file__).resolve().parents[1]/"outputs"/"pretrained"/filename
    for path in (local, cached):
        if path.is_file():
            return path
    if not download:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(ResNet18_Weights.IMAGENET1K_V1.url, str(local), hash_prefix="f37072fd")
    return local


class FrozenResNetEncoder:
    def __init__(self, image_size=384, pretrained=True, state_dict=None):
        if not 224 <= image_size <= 1024:
            raise ValueError("CNN image size must be between 224 and 1024")
        if state_dict is not None and pretrained:
            raise ValueError("Use saved weights or pretrained weights, not both")
        self.image_size = int(image_size)
        self.backbone = resnet18(weights=None)
        self.weights_origin = "IMAGENET1K_V1" if pretrained else "untrained"
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained_weights_path(), map_location="cpu", weights_only=True))
        self.backbone.fc = torch.nn.Identity()
        if state_dict is not None:
            self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def _tensor(self, image):
        # Letterboxing preserves the composition and avoids a center crop that
        # can omit peripheral subjects. Normalization uses ImageNet statistics.
        padded = ImageOps.pad(image.convert("RGB"), (self.image_size, self.image_size),
                              method=Image.Resampling.LANCZOS, color=(124, 116, 104))
        array = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1).copy()/255.
        tensor = torch.from_numpy(array)
        return (tensor-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]

    def extract(self, image, localization, input_mode="combined"):
        if input_mode not in ("global", "subject", "combined"):
            raise ValueError("Unsupported CNN input mode")
        image = ImageOps.exif_transpose(image).convert("RGB")
        views, global_index = [], None
        if input_mode != "subject":
            global_index = len(views)
            views.append(image)
        subject_start = len(views)
        if input_mode != "global":
            context = localization.get("assessment_context", {})
            subjects = localization.get("subjects", [])
            if context.get("shot_type") in ("detail", "scene"):
                subjects = []
            for subject in subjects:
                for key in ("subject_bbox", "face_bbox"):
                    box = subject.get(key)
                    if key == "face_bbox" and not subject.get("face_detected"):
                        continue
                    if not box or len(box) != 4 or not np.isfinite(box).all():
                        continue
                    x1, y1, x2, y2 = box
                    box = (max(0, int(x1)), max(0, int(y1)), min(image.width, int(x2)), min(image.height, int(y2)))
                    minimum = 24 if key == "face_bbox" else 2
                    if min(box[2]-box[0], box[3]-box[1]) >= minimum:
                        views.append(image.crop(box))
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(views), 4):
                batch = torch.stack([self._tensor(v) for v in views[start:start+4]])
                embeddings.extend(self.backbone(batch).cpu().numpy())
        zero = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
        # Missing subjects are represented by zero plus explicit missing flags,
        # never by a fabricated full-frame subject crop.
        subject = np.mean(embeddings[subject_start:], axis=0) if len(embeddings) > subject_start else zero
        full = embeddings[global_index] if global_index is not None else zero
        return {"image": np.asarray(full, dtype=np.float32).tolist(),
                "subject": np.asarray(subject, dtype=np.float32).tolist()}

    def checkpoint(self):
        return {"image_size": self.image_size, "state_dict": self.backbone.state_dict(),
                "architecture": "resnet18", "initial_weights": self.weights_origin,
                "preprocessing": "letterbox_imagenet_v1", "frozen": True}

    @classmethod
    def from_checkpoint(cls, saved):
        if (saved.get("architecture") != "resnet18" or saved.get("preprocessing") != "letterbox_imagenet_v1"
                or saved.get("frozen") is not True):
            raise ValueError("Unsupported CNN encoder checkpoint")
        encoder = cls(image_size=saved["image_size"], pretrained=False, state_dict=saved["state_dict"])
        encoder.weights_origin = saved["initial_weights"]
        return encoder
