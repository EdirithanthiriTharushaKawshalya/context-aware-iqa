"""
Modular UI Components for Streamlit Demonstrator
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import streamlit as st

from src.pipeline import PipelineConfig, PipelineResult
from src.visualization.annotator import ImageAnnotator
from src.ui.data_manager import DataManager
from src.ui.styles import (
    render_html_header,
    render_html_kpi_score,
    render_html_decision_badge,
    render_html_confidence_gauge,
    render_html_ground_truth,
)


def render_hero_header():
    """Renders compact top navigation bar."""
    st.markdown(render_html_header(), unsafe_allow_html=True)


def render_sidebar_controls(
    manifest_rows: List[Dict[str, str]],
    project_root: Path,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, str]], PipelineConfig, Dict[str, bool]]:
    """
    Renders sidebar controls, image selectors, presets, and pipeline configuration.
    """
    with st.sidebar:
        st.markdown("**Input Selection**")

        source_mode = st.radio(
            "Input Mode:",
            ["Dataset Manifest", "Upload Image"],
            index=0,
            horizontal=True,
        )

        selected_image_path = None
        uploaded_image = None
        manifest_meta = None

        if source_mode == "Dataset Manifest":
            if not manifest_rows:
                st.error("Dataset manifest is missing.")
            else:
                # 1. Benchmark Presets Dropdown (clean, non-truncated)
                benchmark_selection = st.selectbox(
                    "Benchmark Scenarios:",
                    [
                        "None (Manual Browsing)",
                        "bday_01: Sharp Portrait (Intentional Bokeh)",
                        "bday_17: Motion Blur Defect",
                        "bday_36: Turned Away Face",
                        "bday_44: Severe Head Cropping",
                    ],
                    index=0,
                )

                active_jump = None
                if "bday_01" in benchmark_selection:
                    active_jump = "bday_01"
                elif "bday_17" in benchmark_selection:
                    active_jump = "bday_17"
                elif "bday_36" in benchmark_selection:
                    active_jump = "bday_36"
                elif "bday_44" in benchmark_selection:
                    active_jump = "bday_44"

                if active_jump:
                    for r in manifest_rows:
                        if r.get("image_id") == active_jump:
                            manifest_meta = r
                            selected_image_path = str(project_root / r.get("file_path", ""))
                            break

                # 2. Category Filter & Selection Dropdown
                preset_filter = st.selectbox(
                    "Filter Category:",
                    [
                        "All Photos",
                        "Keep / Usable Portraits",
                        "Motion Blur Defects",
                        "Turned Away Faces",
                        "Head Cropped Defects",
                    ],
                    disabled=(active_jump is not None),
                )

                filtered_rows = DataManager.filter_manifest(manifest_rows, preset_filter)

                def format_row(r):
                    img_id = r.get("image_id", "")
                    filename = Path(r.get("file_path", "")).name
                    label = r.get("expert_score_usability", "")
                    defect = r.get("local_defect_flag", "None")
                    defect_str = f" | {defect}" if defect and defect != "None" else ""
                    return f"{img_id} ({filename}) [{label}{defect_str}]"

                if not active_jump:
                    selected_row = st.selectbox(
                        f"Select Image ({len(filtered_rows)} available):",
                        filtered_rows,
                        format_func=format_row,
                    )
                    if selected_row:
                        manifest_meta = selected_row
                        rel_path = selected_row.get("file_path", "")
                        full_p = project_root / rel_path
                        if full_p.is_file():
                            selected_image_path = str(full_p)
                        else:
                            st.warning(f"File not found on disk: {rel_path}")
        else:
            uploaded_image = st.file_uploader(
                "Upload Image File:",
                type=["jpg", "jpeg", "png", "webp"],
                help="Supports event photos in RGB format.",
            )

        # 3. Pipeline Configuration Parameters
        st.markdown("---")
        with st.expander("Pipeline Configuration", expanded=False):
            threshold = st.slider(
                "Decision Threshold (Keep vs Review):",
                min_value=30.0,
                max_value=85.0,
                value=55.0,
                step=1.0,
                help="Separation threshold calibrated from Phase 4 ablation experiments.",
            )
            w_local = st.slider(
                "Local Technical Weight (w_local):",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
            )
            w_global = round(1.0 - w_local, 2)
            st.caption(f"Global Context Weight (w_global): {w_global:.2f}")

            ablation_mode = st.selectbox(
                "Ablation Mode:",
                ["full", "local_only", "global_only", "no_penalty"],
                index=0,
                format_func=lambda m: {
                    "full": "Full Context-Aware (Local + Global + Penalties)",
                    "local_only": "Local Technical Only",
                    "global_only": "Global Context Only",
                    "no_penalty": "Fusion without Penalties",
                }[m],
            )

        # 4. Visualization Layers
        st.markdown("---")
        st.markdown("**Visualization Layers**")
        show_subject = st.checkbox("Subject Layer (Cyan)", value=True)
        show_face = st.checkbox("Face Layer (Gold)", value=True)

    # Load PIL image
    raw_img = None
    if selected_image_path and Path(selected_image_path).is_file():
        try:
            raw_img = Image.open(selected_image_path).convert("RGB")
        except Exception as e:
            st.error(f"Error opening image: {e}")
    elif uploaded_image is not None:
        try:
            raw_img = Image.open(uploaded_image).convert("RGB")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

    config = PipelineConfig(
        w_local=w_local,
        w_global=w_global,
        threshold=threshold,
        ablation_mode=ablation_mode,
    )

    visual_flags = {
        "show_subject": show_subject,
        "show_face": show_face,
    }

    return raw_img, manifest_meta, config, visual_flags


def render_kpi_dashboard(
    result: PipelineResult,
    config: PipelineConfig,
    manifest_meta: Optional[Dict[str, str]],
    baseline_map: Dict[str, str],
):
    """Renders top row of 4 KPI metrics cards."""
    is_keep = result.classification == "Keep"
    gt_label = manifest_meta.get("expert_score_usability", "N/A") if manifest_meta else "Custom"
    defect_flag = manifest_meta.get("local_defect_flag", "None") if manifest_meta else "None"
    img_id = manifest_meta.get("image_id", "") if manifest_meta else ""
    baseline_score = baseline_map.get(img_id, "N/A")
    match_flag = (gt_label == result.classification)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.3, 1.3])

    with c1:
        st.markdown(
            render_html_kpi_score(result.unified_score, result.base_score, config.threshold),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_html_decision_badge(is_keep),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_html_confidence_gauge(
                result.overall_confidence_pct,
                result.conf_tier,
                result.unified_score,
                result.margin_of_error,
                result.dist_to_boundary,
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_html_ground_truth(gt_label, match_flag, defect_flag, baseline_score),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)


def render_visual_inspection(
    raw_img: Image.Image,
    result: PipelineResult,
    visual_flags: Dict[str, bool],
):
    """Renders left column visual inspection tabs."""
    st.markdown('<div class="section-header">Visual Semantic Inspection</div>', unsafe_allow_html=True)

    tab_annotated, tab_original, tab_crops = st.tabs([
        "Annotated Image",
        "Original Image",
        "Region Crops",
    ])

    with tab_annotated:
        annotated_img = ImageAnnotator.draw_annotations(
            raw_img,
            result.localization,
            show_subject=visual_flags.get("show_subject", True),
            show_face=visual_flags.get("show_face", True),
        )
        st.image(annotated_img, use_container_width=True)

        leg1, leg2, leg3 = st.columns(3)
        with leg1:
            st.caption("Subject Layer: Cyan")
        with leg2:
            st.caption("Face Layer: Gold")
        with leg3:
            st.caption(f"Resolution: {raw_img.width} × {raw_img.height}")

    with tab_original:
        st.image(raw_img, use_container_width=True)

    with tab_crops:
        crops = ImageAnnotator.extract_crops(raw_img, result.localization)
        crop1, crop2 = st.columns(2)
        with crop1:
            st.markdown("**Subject Body Crop:**")
            if crops.get("subject"):
                st.image(crops["subject"], use_container_width=True)
            else:
                st.info("No subject isolated.")
        with crop2:
            st.markdown("**Face / Head Crop:**")
            if crops.get("face"):
                st.image(crops["face"], use_container_width=True)
            else:
                st.warning("Face not detected or cropped out.")


def render_score_breakdown(result: PipelineResult, config: PipelineConfig):
    """Renders right column multi-stage score breakdown and penalties."""
    st.markdown('<div class="section-header">Multi-Stage Feature & Penalty Breakdown</div>', unsafe_allow_html=True)

    local_f = result.local_features
    global_f = result.global_features

    # 1. Local Technical Card
    st.markdown(
        f"""
        <div class="breakdown-card">
            <div class="breakdown-title">
                <span>Local Technical Evaluation</span>
                <span style="color: #38bdf8; font-family:'JetBrains Mono', monospace;">{result.s_local:.1f} / 100 <span style="font-size:0.72rem; color:#64748b;">({int(config.w_local*100)}% Weight)</span></span>
            </div>
            <div class="metric-row">
                <span>Face Sharpness (Laplacian Var):</span>
                <span class="val">{local_f.get('face_sharpness', 0.0):.1f}</span>
            </div>
            <div class="metric-row">
                <span>Face Edge Energy (Tenengrad):</span>
                <span class="val">{local_f.get('face_tenengrad', 0.0):.1f}</span>
            </div>
            <div class="metric-row">
                <span>Subject Body Sharpness:</span>
                <span class="val">{local_f.get('subject_sharpness', 0.0):.1f}</span>
            </div>
            <div class="metric-row">
                <span>Motion Blur Anisotropy Metric:</span>
                <span class="val" style="color: {'#f87171' if local_f.get('motion_blur_metric', 0.0) > 0.45 else '#34d399'};">
                    {local_f.get('motion_blur_metric', 0.0):.3f} {'(Anisotropic Blur)' if local_f.get('motion_blur_metric', 0.0) > 0.45 else '(Nominal)'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Global Context Card
    dof_val = global_f.get("dof_separation_ratio", 1.0)
    dof_color = "#34d399" if dof_val >= 2.0 else "#94a3b8"
    st.markdown(
        f"""
        <div class="breakdown-card">
            <div class="breakdown-title">
                <span>Global Context Evaluation</span>
                <span style="color: #c084fc; font-family:'JetBrains Mono', monospace;">{result.s_global:.1f} / 100 <span style="font-size:0.72rem; color:#64748b;">({int(config.w_global*100)}% Weight)</span></span>
            </div>
            <div class="metric-row">
                <span>Depth-of-Field (Bokeh) Ratio:</span>
                <span class="val" style="color: {dof_color};">{dof_val:.2f}x {'(Bokeh Reward)' if dof_val >= 2.0 else ''}</span>
            </div>
            <div class="metric-row">
                <span>Global Scene Contrast:</span>
                <span class="val">{global_f.get('global_contrast', 0.0):.1f}</span>
            </div>
            <div class="metric-row">
                <span>Mean Frame Luminance:</span>
                <span class="val">{global_f.get('mean_luminance', 128.0):.1f}</span>
            </div>
            <div class="metric-row">
                <span>Highlight / Shadow Clipping:</span>
                <span class="val">{(global_f.get('highlight_clipping_ratio', 0.0)*100):.1f}% / {(global_f.get('shadow_underexposure_ratio', 0.0)*100):.1f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Defect Penalties Card
    st.markdown(
        """
        <div class="breakdown-card">
            <div class="breakdown-title">
                <span>Semantic Anomaly & Defect Deductions</span>
                <span style="color: #f43f5e; font-family:'JetBrains Mono', monospace;">Deductions</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    penalties_present = False
    if result.framing_penalty_pts > 0:
        penalties_present = True
        st.markdown(
            f"""
            <div class="penalty-item">
                <span class="penalty-tag">ANOMALY</span> <b>Framing Defect</b>: Subject head cropped at boundary (-{result.framing_penalty_pts:.1f} pts, ceiling: 25.0).
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result.face_penalty_pts > 0:
        penalties_present = True
        st.markdown(
            f"""
            <div class="penalty-item">
                <span class="penalty-tag">ANOMALY</span> <b>Orientation Defect</b>: Subject turned away or face occluded (-{result.face_penalty_pts:.1f} pts, -40%).
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result.motion_penalty_pts > 0:
        penalties_present = True
        st.markdown(
            f"""
            <div class="penalty-item">
                <span class="penalty-tag">ANOMALY</span> <b>Motion Blur</b>: Directional gradient anisotropy exceeds threshold (-{result.motion_penalty_pts:.1f} pts).
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not penalties_present:
        st.markdown(
            """
            <div class="penalty-none">
                <span class="clean-tag">NOMINAL</span> Framing geometry intact, face detected, motion within tolerance.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Formula Math Box
    score_color = "#10b981" if result.unified_score >= config.threshold else "#f43f5e"
    deduction_str = f" - {result.total_defect_penalty:.1f} deductions" if result.total_defect_penalty > 0 else ""
    formula_html = (
        f'<div class="formula-container">'
        f'<b>Formula:</b> [{config.w_local:.2f} × {result.s_local:.1f}] + [{config.w_global:.2f} × {result.s_global:.1f}] = {result.base_score:.1f} base'
        f'{deduction_str} = <span style="color: {score_color}; font-weight:700;">{result.unified_score:.1f} final</span>'
        f'</div>'
    )
    st.markdown(formula_html, unsafe_allow_html=True)


def render_diagnostics_and_table(
    result: PipelineResult,
    manifest_rows: List[Dict[str, str]],
    baseline_map: Dict[str, str],
):
    """Renders bottom expandable diagnostic telemetry and manifest comparison table."""
    st.markdown("---")
    with st.expander("Diagnostic Telemetry & Semantic Coordinates", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Semantic Coordinates & Flags:**")
            loc = result.localization
            loc_data = {
                "Subject Detected": loc.get("subject_detected"),
                "Subject Confidence": f"{loc.get('subject_confidence', 0.0):.4f}",
                "Subject Bounding Box [x1, y1, x2, y2]": [round(c, 1) for c in (loc.get("subject_bbox") or [])],
                "Face Detected": loc.get("face_detected"),
                "Face Confidence": f"{loc.get('face_confidence', 0.0):.4f}",
                "Face Bounding Box [x1, y1, x2, y2]": [round(c, 1) for c in (loc.get("face_bbox") or [])],
                "Framing Anomaly Flag": loc.get("framing_anomaly"),
            }
            st.json(loc_data)
        with t2:
            st.markdown("**Extracted Feature Vectors:**")
            raw_features = {
                "Local Features": result.local_features,
                "Global Features": result.global_features,
            }
            st.json(raw_features)

    if manifest_rows:
        with st.expander("Dataset Manifest & Benchmark Comparisons", expanded=False):
            st.caption("Phase 4 Context-Aware pipeline vs. Phase 3 No-Reference baseline across event bursts:")
            preview_data = []
            for r in manifest_rows:
                img_id = r.get("image_id", "")
                preview_data.append({
                    "Image ID": img_id,
                    "Burst": r.get("burst_id", ""),
                    "Filename": Path(r.get("file_path", "")).name,
                    "Expert Label": r.get("expert_score_usability", ""),
                    "Defect Flag": r.get("local_defect_flag", "None"),
                    "Baseline BRISQUE": baseline_map.get(img_id, "N/A"),
                })
            st.dataframe(preview_data, use_container_width=True)
