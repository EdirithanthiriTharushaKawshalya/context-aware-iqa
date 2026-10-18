# Context-Aware IQA Evaluation: Learned Fusion (LOGISTIC)

## 1. Executive Summary

- **Evaluated Samples**: 44 (39 Keep, 5 Review)
- **Overall Accuracy**: **100.00%**
- **Balanced Accuracy**: **100.00%**
- **Macro F1-Score**: **1.0000**
- **Keep F1**: 1.0000 | **Review F1**: 1.0000
- **Separation Margin**: `+60.91` points

## 2. Classification Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Keep** | 1.0000 | 1.0000 | 1.0000 | 39 |
| **Review** | 1.0000 | 1.0000 | 1.0000 | 5 |
| **Macro Avg** | 1.0000 | 1.0000 | **1.0000** | 44 |

### Confusion Matrix

| Ground Truth \ Prediction | Predicted Keep | Predicted Review | Total |
| :--- | :---: | :---: | :---: |
| **Actual Keep** | **39** (TP) | 0 (FN) | 39 |
| **Actual Review** | 0 (FP) | **5** (TN) | 5 |

## 3. Defect-Level Diagnosis

| Defect Category | Total Cases | Flagged Review | False Keeps (Leaked) | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **None** | 5 | 5 | 0 | **100.0%** |

## 4. Score Separation Analysis

| Subset | Mean Score | Std Dev | Min Score | Max Score |
| :--- | :---: | :---: | :---: | :---: |
| **Keep Images** | 95.15 | 5.91 | 77.90 | 99.96 |
| **Review Images** | 5.38 | 6.43 | 0.20 | 16.99 |

> [!NOTE]
> **Separation Margin ($S_{Keep}^{min} - S_{Review}^{max}$)**: `+60.91` points.
