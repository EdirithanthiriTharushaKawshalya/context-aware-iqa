# Context-Aware IQA Evaluation: Baseline Metric (baseline_results.csv)

## 1. Executive Summary

- **Evaluated Samples**: 44 (39 Keep, 5 Review)
- **Overall Accuracy**: **20.45%**
- **Balanced Accuracy**: **20.26%**
- **Macro F1-Score**: **0.1839**
- **Keep F1**: 0.3137 | **Review F1**: 0.0541
- **Separation Margin**: `-49.21` points

## 2. Classification Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Keep** | 0.6667 | 0.2051 | 0.3137 | 39 |
| **Review** | 0.0312 | 0.2000 | 0.0541 | 5 |
| **Macro Avg** | 0.3489 | 0.2026 | **0.1839** | 44 |

### Confusion Matrix

| Ground Truth \ Prediction | Predicted Keep | Predicted Review | Total |
| :--- | :---: | :---: | :---: |
| **Actual Keep** | **8** (TP) | 31 (FN) | 39 |
| **Actual Review** | 4 (FP) | **1** (TN) | 5 |

## 3. Defect-Level Diagnosis

| Defect Category | Total Cases | Flagged Review | False Keeps (Leaked) | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Motion Blur** | 2 | 0 | 2 | **0.0%** |
| **Face Turned Away** | 2 | 0 | 2 | **0.0%** |
| **Head Cropped Out** | 1 | 1 | 0 | **100.0%** |

## 4. Score Separation Analysis

| Subset | Mean Score | Std Dev | Min Score | Max Score |
| :--- | :---: | :---: | :---: | :---: |
| **Keep Images** | 44.34 | 10.40 | 21.01 | 65.12 |
| **Review Images** | 55.38 | 18.55 | 19.86 | 70.22 |

> [!NOTE]
> **Separation Margin ($S_{Keep}^{min} - S_{Review}^{max}$)**: `-49.21` points.

## 5. Misclassified Samples

| Image ID | Ground Truth | Predicted | Score | Defect Flag |
| :--- | :---: | :---: | :---: | :--- |
| `bday_03` | Keep | Review | 38.32 | None |
| `bday_04` | Keep | Review | 48.63 | None |
| `bday_05` | Keep | Review | 42.99 | None |
| `bday_06` | Keep | Review | 47.70 | None |
| `bday_07` | Keep | Review | 41.07 | None |
| `bday_09` | Keep | Review | 49.02 | None |
| `bday_10` | Keep | Review | 50.45 | None |
| `bday_11` | Keep | Review | 49.09 | None |
| `bday_12` | Keep | Review | 36.08 | None |
| `bday_14` | Keep | Review | 44.44 | None |
| `bday_17` | Review | Keep | 56.03 | Motion Blur |
| `bday_18` | Review | Keep | 69.68 | Motion Blur |
| `bday_19` | Keep | Review | 34.47 | None |
| `bday_21` | Keep | Review | 44.68 | None |
| `bday_22` | Keep | Review | 21.01 | None |
| `bday_23` | Keep | Review | 43.34 | None |
| `bday_24` | Keep | Review | 37.97 | None |
| `bday_25` | Keep | Review | 45.42 | None |
| `bday_26` | Keep | Review | 44.78 | None |
| `bday_27` | Keep | Review | 39.41 | None |
| `bday_28` | Keep | Review | 24.43 | None |
| `bday_29` | Keep | Review | 34.67 | None |
| `bday_30` | Keep | Review | 28.40 | None |
| `bday_32` | Keep | Review | 35.36 | None |
| `bday_33` | Keep | Review | 42.36 | None |
| `bday_34` | Keep | Review | 40.00 | None |
| `bday_35` | Keep | Review | 46.30 | None |
| `bday_36` | Review | Keep | 70.22 | Face Turned Away |
| `bday_37` | Review | Keep | 61.10 | Face Turned Away |
| `bday_38` | Keep | Review | 42.92 | None |
| `bday_39` | Keep | Review | 33.45 | None |
| `bday_40` | Keep | Review | 41.15 | None |
| `bday_41` | Keep | Review | 43.57 | None |
| `bday_42` | Keep | Review | 33.97 | None |
| `bday_43` | Keep | Review | 40.10 | None |