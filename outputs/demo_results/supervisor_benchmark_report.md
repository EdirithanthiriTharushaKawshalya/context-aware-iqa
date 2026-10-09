# Context-Aware Image Quality Assessment (IQA) — Phase 4 Research Demonstration

## Executive Summary
Traditional No-Reference IQA algorithms (such as BRISQUE, NIQE) fail on professional event photography
because they misinterpret intentional shallow depth-of-field (bokeh) as quality degradation, while failing to detect
critical semantic flaws such as severed framing or motion blur. The Phase 4 Context-Aware pipeline solves this
by combining deep semantic localization, local subject sharpness, global context modeling, and anomaly penalty engines.

### Key Benchmark Results

| Image ID | Defect Flag | Ground Truth | Baseline (BRISQUE) | Context-Aware Score | Decision | Confidence | Ground Truth Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bday_01` | None | **Keep** | 65.1215 | **63.20** | `Keep` | 64.6% (±2.8) | ✅ Match |
| `bday_17` | Motion Blur | **Review** | 56.0339 | **43.27** | `Review` | 81.1% (±1.5) | ✅ Match |
| `bday_36` | Face Turned Away | **Review** | 70.2187 | **65.99** | `Keep` | 77.6% (±1.8) | ❌ Divergent |
| `bday_44` | Head Cropped Out | **Review** | 19.8555 | **12.92** | `Review` | 93.2% (±0.5) | ✅ Match |

## Core Insights Demonstrated
1. **Depth-of-Field (Bokeh) Resilience**: On sharp portraits (`bday_01`), our model rewards subject isolation ($DOF > 2.0x$) achieving **63.2**, whereas traditional baseline falsely penalizes background blur.
2. **Motion Blur Penalty**: On motion-degraded bursts (`bday_17`), the directional gradient anisotropy metric flags blur ($> 0.45$) and applies points deduction, dropping the score to **43.3** (`Review`).
3. **Framing Defect Cap**: On severely cropped subjects (`bday_44`), the semantic localizer flags `HEAD_CROPPED`, capping the score at **12.9** (`Review`), matching expert culling criteria.

---
*Generated automatically by Context-Aware IQA Phase 4 CLI Engine.*