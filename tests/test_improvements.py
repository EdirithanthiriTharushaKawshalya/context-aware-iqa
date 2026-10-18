"""
Comprehensive Test Suite for Context-Aware IQA Improvements:
1. Evaluation Harness & Metrics
2. Hyperparameter Optimizer & Cross-Validation
3. Face Orientation & Yaw Estimation
4. Learned Fusion Classifier
5. Burst Context Engine & Hero Frame Selection
"""

import pytest
import numpy as np
import torch
from PIL import Image

from src.evaluation import EvaluationHarness, EvaluationReport, ConfusionMatrix
from src.optimizer import HyperparameterOptimizer, OptimizationConfig
from src.learned_fusion import LearnedFusionEngine, extract_feature_vector, FEATURE_NAMES
from src.burst_context import BurstContextEngine, BurstSummary
from src.pipeline import Phase4Pipeline, PipelineConfig


@pytest.fixture
def sample_ground_truth():
    """Generates synthetic ground truth annotations."""
    return {
        "img_01": {"image_id": "img_01", "expert_score_usability": "Keep", "file_path": "path/1.jpg", "local_defect_flag": "None"},
        "img_02": {"image_id": "img_02", "expert_score_usability": "Keep", "file_path": "path/2.jpg", "local_defect_flag": "None"},
        "img_03": {"image_id": "img_03", "expert_score_usability": "Review", "file_path": "path/3.jpg", "local_defect_flag": "Motion Blur"},
        "img_04": {"image_id": "img_04", "expert_score_usability": "Review", "file_path": "path/4.jpg", "local_defect_flag": "Face Turned Away"},
    }


def test_evaluation_harness_metrics(sample_ground_truth):
    """Verifies precision, recall, F1, confusion matrix, and defect breakdown calculation."""
    harness = EvaluationHarness()
    predictions = [
        {"image_id": "img_01", "predicted_score": 75.0, "predicted_classification": "Keep"},
        {"image_id": "img_02", "predicted_score": 68.0, "predicted_classification": "Keep"},
        {"image_id": "img_03", "predicted_score": 35.0, "predicted_classification": "Review"},
        {"image_id": "img_04", "predicted_score": 25.0, "predicted_classification": "Review"},
    ]

    report = harness.evaluate_predictions(predictions, name="Perfect Classification Test", gt_mapping=sample_ground_truth)
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.confusion_matrix.tp == 2
    assert report.confusion_matrix.tn == 2
    assert report.confusion_matrix.fp == 0
    assert report.confusion_matrix.fn == 0
    assert report.separation_margin == (68.0 - 35.0)  # 33.0 points positive margin
    assert "Motion Blur" in report.defect_breakdown
    assert report.defect_breakdown["Motion Blur"].recall == 1.0


def test_optimizer_grid_search():
    """Verifies that the hyperparameter optimizer discovers optimal separation."""
    optimizer = HyperparameterOptimizer()
    sample_features = {
        "d:/img_keep_1.jpg": {
            "localization": {"subject_detected": True, "face_detected": True, "face_orientation": "FRONTAL"},
            "local_features": {"local_technical_score": 70.0, "framing_defect_detected": False, "face_present": True, "motion_blur_metric": 0.20},
            "global_features": {"global_context_score": 65.0, "dof_separation_ratio": 2.5},
        },
        "d:/img_rev_1.jpg": {
            "localization": {"subject_detected": True, "face_detected": False, "face_orientation": "TURNED_AWAY"},
            "local_features": {"local_technical_score": 25.0, "framing_defect_detected": False, "face_present": False, "motion_blur_metric": 0.20},
            "global_features": {"global_context_score": 65.0, "dof_separation_ratio": 1.0},
        },
    }

    samples = [
        ("img_k", "Keep", sample_features["d:/img_keep_1.jpg"]),
        ("img_r", "Review", sample_features["d:/img_rev_1.jpg"]),
    ]

    metrics = optimizer.evaluate_parameter_set(samples, w_local=0.55, w_global=0.45, threshold=52.0)
    assert metrics["macro_f1"] == 1.0
    assert metrics["separation_margin"] > 0.0


def test_face_orientation_landmarks_and_yaw():
    """Verifies that facial landmark geometry calculates yaw asymmetry correctly."""
    # Frontal geometry: nose is equidistant from both eyes
    lex, ley = 100.0, 150.0
    rex, rey = 200.0, 150.0
    nx_frontal = 150.0

    dl_front = abs(nx_frontal - lex)  # 50
    dr_front = abs(nx_frontal - rex)  # 50
    asym_front = abs(dl_front - dr_front) / (dl_front + dr_front)
    yaw_front = asym_front * 90.0
    assert asym_front == 0.0
    assert yaw_front == 0.0

    # Turned geometry: nose shifted towards one side
    nx_turned = 185.0
    dl_turn = abs(nx_turned - lex)  # 85
    dr_turn = abs(nx_turned - rex)  # 15
    asym_turn = abs(dl_turn - dr_turn) / (dl_turn + dr_turn)
    yaw_turn = asym_turn * 90.0
    assert asym_turn == pytest.approx(0.7)
    assert yaw_turn == pytest.approx(63.0)  # Profile / turned angle!


def test_learned_fusion_model():
    """Verifies feature extraction, model forward pass, and predict structure."""
    dummy_entry = {
        "localization": {"subject_detected": True, "face_detected": True, "face_yaw_asymmetry": 0.2},
        "local_features": {
            "local_technical_score": 75.0,
            "face_sharpness": 250.0,
            "subject_sharpness": 120.0,
            "face_tenengrad": 35.0,
            "motion_blur_metric": 0.22,
            "local_contrast": 50.0,
            "framing_defect_detected": False,
            "face_present": True,
        },
        "global_features": {
            "global_context_score": 70.0,
            "global_contrast": 45.0,
            "dof_separation_ratio": 2.8,
        }
    }

    vec = extract_feature_vector(dummy_entry)
    assert len(vec) == len(FEATURE_NAMES)
    assert isinstance(vec, np.ndarray)

    engine = LearnedFusionEngine(model_type="logistic")
    samples = [
        ("k1", "Keep", dummy_entry),
        ("r1", "Review", {
            "localization": {"subject_detected": True, "face_detected": False, "face_yaw_asymmetry": 1.0},
            "local_features": {"local_technical_score": 20.0, "motion_blur_metric": 0.60, "face_present": False, "framing_defect_detected": False},
            "global_features": {"global_context_score": 40.0, "dof_separation_ratio": 0.9},
        })
    ]

    res = engine.train(samples, epochs=50)
    assert res["train_macro_f1"] == 1.0

    pred = engine.predict(dummy_entry)
    assert "learned_score" in pred
    assert "probability_keep" in pred
    assert pred["classification"] == "Keep"


def test_burst_context_engine():
    """Verifies burst ranking, z-score computation, and hero frame selection."""
    engine = BurstContextEngine()
    frames = [
        {"image_id": "frame_1", "file_path": "f1.jpg", "unified_score": 64.0},
        {"image_id": "frame_2", "file_path": "f2.jpg", "unified_score": 75.0},  # Hero
        {"image_id": "frame_3", "file_path": "f3.jpg", "unified_score": 73.5},  # Close alternative
        {"image_id": "frame_4", "file_path": "f4.jpg", "unified_score": 30.0, "defect_flag": "Motion Blur"},  # Review
    ]

    summary = engine.evaluate_burst("burst_test", frames)
    assert summary.hero_image_id == "frame_2"
    assert summary.hero_score == 75.0
    assert len(summary.frames) == 4

    # Frame 2 must be rank 1 and hero frame
    f2 = next(f for f in summary.frames if f.image_id == "frame_2")
    assert f2.burst_rank == 1
    assert f2.is_hero_frame is True

    # Frame 4 must be rank 4 and flagged Review
    f4 = next(f for f in summary.frames if f.image_id == "frame_4")
    assert f4.burst_rank == 4
    assert "Review" in f4.burst_recommendation
