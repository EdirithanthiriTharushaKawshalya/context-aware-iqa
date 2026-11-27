"""Bounded illumination compensation for RAW focus evidence, never exposure scoring."""
import numpy as np
from PIL import Image

from src.local_features import LocalFeatureExtractor


def focus_gain(gray):
    sample = gray.copy()
    sample.thumbnail((256, 256))
    level = float(np.percentile(np.asarray(sample), 80))
    if level <= 0 or level >= 80:
        return 1.
    return min(4., 96./level)


def brighten(image, gain):
    # Lift shadows gently while preserving highlight headroom. Linear gain
    # would flatten bright faces in a predominantly dark group photograph.
    table = [round(value*gain/(1+(gain-1)*value/255.)) for value in range(256)]
    return image.point(table * len(image.getbands()))


class RawFocusExtractor(LocalFeatureExtractor):
    def __init__(self):
        self._raw = False
        self._depth = 0
        self._gains = []

    def extract(self, img, localization):
        previous = self._raw
        outer = self._depth == 0
        self._raw = bool(img.info.get('raw_render_profile'))
        if outer:
            self._gains = []
        self._depth += 1
        try:
            result = super().extract(img, localization)
            if outer and self._raw:
                result['raw_focus_gain'] = max(self._gains, default=1.)
            return result
        finally:
            self._depth -= 1
            self._raw = previous

    def _focus_gray(self, gray):
        if not self._raw:
            return gray
        gain = focus_gain(gray)
        self._gains.append(gain)
        return brighten(gray, gain) if gain > 1 else gray

    def _calc_laplacian_var(self, gray):
        return super()._calc_laplacian_var(self._focus_gray(gray))

    def _calc_tenengrad(self, array):
        gray = np.asarray(self._focus_gray(Image.fromarray(array))) if self._raw else array
        return super()._calc_tenengrad(gray)

    def _calc_motion_blur_ratio(self, array):
        gray = np.asarray(self._focus_gray(Image.fromarray(array))) if self._raw else array
        return super()._calc_motion_blur_ratio(gray)


class RawAwareLocalizer:
    def __init__(self, localizer):
        self.localizer = localizer

    def localize(self, image):
        gain = focus_gain(image.convert('L')) if image.info.get('raw_render_profile') else 1.
        result = self.localizer.localize(brighten(image, gain) if gain > 1 else image)
        if gain > 1:
            result['raw_detection_gain'] = gain
        return result

    def __getattr__(self, name):
        return getattr(self.localizer, name)
