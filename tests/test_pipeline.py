"""
Unit & Integration Test Suite for Context-Aware IQA Pipeline
Verifies algorithm correctness, penalty equations, decision boundaries, and uncertainty models.
"""

import pytest
import numpy as np
from PIL import Image

from src.pipeline import Phase4Pipeline, PipelineConfig, PipelineResult
from src.visualization.annotator import ImageAnnotator


@pytest.fixture
def dummy_rgb_image():
    """Generates a synthetic RGB image for fast deterministic pipeline testing."""
    arr = np.random.randint(50, 200, (600, 800, 3), dtype=np.uint8)
    # Inject a high contrast center block simulating a sharp subject
    arr[150:450, 250:550] = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_pipeline_output_structure(dummy_rgb_image):
    """Verifies that the pipeline produces all expected fields with valid ranges."""
    pipeline = Phase4Pipeline()
    config = PipelineConfig(threshold=55.0)
    result = pipeline.run(dummy_rgb_image, config)

    assert isinstance(result, PipelineResult)
    assert 0.0 <= result.unified_score <= 100.0
    assert 0.0 <= result.s_local <= 100.0
    assert 0.0 <= result.s_global <= 100.0
    assert result.classification in ["Keep", "Review"]
    assert 0.0 <= result.overall_confidence_pct <= 100.0
    assert result.margin_of_error > 0.0
    assert result.conf_tier in ["high", "med", "low"]


def test_decision_boundary_threshold():
    """Verifies that the threshold parameter accurately governs Keep vs Review."""
    pipeline = Phase4Pipeline()
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))

    # Low threshold forces Keep
    res_low = pipeline.run(img, PipelineConfig(threshold=1.0))
    assert res_low.classification == "Keep"

    # Extreme threshold forces Review
    res_high = pipeline.run(img, PipelineConfig(threshold=99.0))
    assert res_high.classification == "Review"


def test_framing_defect_penalty_capping():
    """Verifies that framing defect (HEAD_CROPPED) caps the final score at <= 25.0."""
    pipeline = Phase4Pipeline()
    img = Image.new("RGB", (400, 400), color=(100, 100, 100))

    # Run pipeline with full context-aware penalties
    res = pipeline.run(img, PipelineConfig(ablation_mode="full"))
    # If a framing defect is detected on synthetic or real data, score cannot exceed 25.0
    if res.framing_penalty_pts > 0:
        assert res.unified_score <= 25.0


def test_ablation_modes(dummy_rgb_image):
    """Verifies that ablation modes evaluate isolated components."""
    pipeline = Phase4Pipeline()

    res_local = pipeline.run(dummy_rgb_image, PipelineConfig(ablation_mode="local_only"))
    assert res_local.unified_score == res_local.s_local

    res_global = pipeline.run(dummy_rgb_image, PipelineConfig(ablation_mode="global_only"))
    assert res_global.unified_score == res_global.s_global

    res_no_pen = pipeline.run(dummy_rgb_image, PipelineConfig(ablation_mode="no_penalty"))
    expected_base = round((0.65 * res_no_pen.s_local) + (0.35 * res_no_pen.s_global), 2)
    assert abs(res_no_pen.unified_score - expected_base) < 0.1


def test_annotator_drawing(dummy_rgb_image):
    """Verifies that the ImageAnnotator draws annotations without modifying original image size."""
    loc = {
        "subject_detected": True,
        "subject_bbox": [100, 100, 400, 500],
        "subject_confidence": 0.95,
        "face_detected": True,
        "face_bbox": [150, 120, 300, 280],
        "face_confidence": 0.85,
        "framing_anomaly": "NONE",
    }
    annotated = ImageAnnotator.draw_annotations(dummy_rgb_image, loc)
    assert annotated.size == dummy_rgb_image.size
    assert annotated is not dummy_rgb_image

    crops = ImageAnnotator.extract_crops(dummy_rgb_image, loc)
    assert crops["subject"] is not None
    assert crops["face"] is not None
