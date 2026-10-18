# Context-Aware IQA Evaluation: Baseline Metric (musiq_results.csv)

## 1. Executive Summary

- **Evaluated Samples**: 44 (39 Keep, 5 Review)
- **Overall Accuracy**: **93.18%**
- **Balanced Accuracy**: **70.00%**
- **Macro F1-Score**: **0.7672**
- **Keep F1**: 0.9630 | **Review F1**: 0.5714
- **Separation Margin**: `-2.55` points

## 2. Classification Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Keep** | 0.9286 | 1.0000 | 0.9630 | 39 |
| **Review** | 1.0000 | 0.4000 | 0.5714 | 5 |
| **Macro Avg** | 0.9643 | 0.7000 | **0.7672** | 44 |

### Confusion Matrix

| Ground Truth \ Prediction | Predicted Keep | Predicted Review | Total |
| :--- | :---: | :---: | :---: |
| **Actual Keep** | **39** (TP) | 0 (FN) | 39 |
| **Actual Review** | 3 (FP) | **2** (TN) | 5 |

## 3. Defect-Level Diagnosis

| Defect Category | Total Cases | Flagged Review | False Keeps (Leaked) | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Motion Blur** | 2 | 2 | 0 | **100.0%** |
| **Face Turned Away** | 2 | 0 | 2 | **0.0%** |
| **Head Cropped Out** | 1 | 0 | 1 | **0.0%** |

## 4. Score Separation Analysis

| Subset | Mean Score | Std Dev | Min Score | Max Score |
| :--- | :---: | :---: | :---: | :---: |
| **Keep Images** | 69.39 | 3.00 | 62.79 | 75.88 |
| **Review Images** | 49.20 | 17.18 | 22.95 | 65.34 |

> [!NOTE]
> **Separation Margin ($S_{Keep}^{min} - S_{Review}^{max}$)**: `-2.55` points.

## 5. Misclassified Samples

| Image ID | Ground Truth | Predicted | Score | Defect Flag |
| :--- | :---: | :---: | :---: | :--- |
| `bday_36` | Review | Keep | 60.18 | Face Turned Away |
| `bday_37` | Review | Keep | 63.00 | Face Turned Away |
| `bday_44` | Review | Keep | 65.34 | Head Cropped Out |