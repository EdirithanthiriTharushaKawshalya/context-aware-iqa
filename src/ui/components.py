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
    render_html_kpi_score,
    render_html_decision_badge,
    render_html_confidence_gauge,
    render_html_ground_truth,
)


def render_hero_header():
    """Renders main application title banner."""
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-title">Context-Aware Image Quality Assessment (IQA)</div>
            <div class="hero-subtitle">
                Phase 4 Unified Research Demonstrator — Semantic Subject Localization,
                Local Subject Clarity, Global Context Separation, and Anomaly Defect Penalty Engine.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls(
    manifest_rows: List[Dict[str, str]],
    project_root: Path,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, str]], PipelineConfig, Dict[str, bool]]:
    """
    Renders sidebar controls, image selectors, preset buttons, and pipeline configuration.
    """
    with st.sidebar:
        st.header("⚙️ Image Selection & Controls")

        source_mode = st.radio(
            "Select Photo Source:",
            ["Dataset Manifest", "Upload Custom Image"],
            index=0,
            horizontal=True,
        )

        selected_image_path = None
        uploaded_image = None
        manifest_meta = None

        if source_mode == "Dataset Manifest":
            if not manifest_rows:
                st.error("Dataset manifest is empty or missing.")
            else:
                st.markdown("**Quick Preset Filters:**")
                preset_filter = st.selectbox(
                    "Filter Manifest by Scenario:",
                    [
                        "All Photos",
                        "🌟 Keep / Usable Portraits",
                        "⚡ Motion Blur Defects",
                        "👤 Turned Away Faces",
                        "✂️ Head / Face Cropped Defects",
                    ],
                )

                filtered_rows = DataManager.filter_manifest(manifest_rows, preset_filter)

                def format_row(r):
                    img_id = r.get("image_id", "")
                    filename = Path(r.get("file_path", "")).name
                    label = r.get("expert_score_usability", "")
                    defect = r.get("local_defect_flag", "None")
                    defect_str = f" • {defect}" if defect and defect != "None" else ""
                    return f"{img_id} ({filename}) — [{label}{defect_str}]"

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

                # Quick Jump Buttons
                st.markdown("---")
                st.markdown("**⚡ Instant Benchmark Jumps:**")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("💎 Sharp (bday_01)", use_container_width=True):
                        st.session_state["quick_jump"] = "bday_01"
                    if st.button("✂️ Cropped (bday_44)", use_container_width=True):
                        st.session_state["quick_jump"] = "bday_44"
                with col_btn2:
                    if st.button("⚡ Motion (bday_17)", use_container_width=True):
                        st.session_state["quick_jump"] = "bday_17"
                    if st.button("👤 Turned (bday_36)", use_container_width=True):
                        st.session_state["quick_jump"] = "bday_36"

                if "quick_jump" in st.session_state:
                    jump_id = st.session_state.pop("quick_jump")
                    for r in manifest_rows:
                        if r.get("image_id") == jump_id:
                            manifest_meta = r
                            selected_image_path = str(project_root / r.get("file_path", ""))
                            break
        else:
            uploaded_image = st.file_uploader(
                "Upload an Event Photograph:",
                type=["jpg", "jpeg", "png", "webp"],
                help="Supports portrait, burst, or event photography in RGB format.",
            )

        # Pipeline Configuration Parameters
        st.markdown("---")
        with st.expander("🛠️ Pipeline Parameters & Ablation", expanded=False):
            st.caption("Adjust fusion weights and decision boundary:")
            threshold = st.slider(
                "Keep / Review Threshold:",
                min_value=30.0,
                max_value=85.0,
                value=55.0,
                step=1.0,
                help="Calibrated decision separation threshold from Phase 4 ablation experiments.",
            )
            w_local = st.slider(
                "Local Subject Weight (w_local):",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
            )
            w_global = round(1.0 - w_local, 2)
            st.text(f"Global Context Weight (w_global): {w_global:.2f}")

            ablation_mode = st.selectbox(
                "Ablation Mode:",
                ["full", "local_only", "global_only", "no_penalty"],
                index=0,
                format_func=lambda m: {
                    "full": "Full Context-Aware (Local + Global + Penalties)",
                    "local_only": "Local Subject Only (No Global Context)",
                    "global_only": "Global Context Only (Traditional Frame)",
                    "no_penalty": "Fusion without Defect Penalties",
                }[m],
            )

        # Visual overlay toggles
        st.markdown("---")
        st.markdown("**🎨 Visualization Overlays:**")
        show_subject = st.checkbox("Show Subject Bounding Box (Cyan)", value=True)
        show_face = st.checkbox("Show Face / Head Bounding Box (Gold)", value=True)

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
    match_icon = "✅" if (gt_label == result.classification) else ("⚠️" if gt_label != "N/A" else "—")

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
            render_html_ground_truth(gt_label, match_icon, defect_flag, baseline_score),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


def render_visual_inspection(
    raw_img: Image.Image,
    result: PipelineResult,
    visual_flags: Dict[str, bool],
):
    """Renders left column visual inspection tabs."""
    st.subheader("🖼️ Visual Semantic Inspection")

    tab_annotated, tab_original, tab_crops = st.tabs([
        "🎯 Annotated Bounding Boxes",
        "📷 Original Photo",
        "🔍 Localized Subject & Face Crops",
    ])

    with tab_annotated:
        annotated_img = ImageAnnotator.draw_annotations(
            raw_img,
            result.localization,
            show_subject=visual_flags.get("show_subject", True),
            show_face=visual_flags.get("show_face", True),
        )
        st.image(annotated_img, use_container_width=True, caption="Detected Primary Subject & Face Regions")

        leg1, leg2, leg3 = st.columns(3)
        with leg1:
            st.markdown("🟦 **Cyan:** Primary Person")
        with leg2:
            st.markdown("🟨 **Gold:** Face / Head")
        with leg3:
            st.markdown(f"📐 **Res:** `{raw_img.width} × {raw_img.height}`")

    with tab_original:
        st.image(raw_img, use_container_width=True, caption="Original Input Photograph")

    with tab_crops:
        crops = ImageAnnotator.extract_crops(raw_img, result.localization)
        crop1, crop2 = st.columns(2)
        with crop1:
            st.markdown("**Primary Subject Crop:**")
            if crops.get("subject"):
                st.image(crops["subject"], use_container_width=True)
            else:
                st.info("No subject isolated.")
        with crop2:
            st.markdown("**Isolated Face / Head Crop:**")
            if crops.get("face"):
                st.image(crops["face"], use_container_width=True)
            else:
                st.warning("Face not detected or cropped out.")


def render_score_breakdown(result: PipelineResult, config: PipelineConfig):
    """Renders right column multi-stage score breakdown and penalties."""
    st.subheader("📊 Multi-Stage Score Breakdown")

    local_f = result.local_features
    global_f = result.global_features

    # 1. Local Technical Card
    st.markdown(
        f"""
        <div class="breakdown-card">
            <div class="breakdown-title">
                <span>🔬 Local Technical Score</span>
                <span style="color: #38bdf8;">{result.s_local:.1f} / 100 <span style="font-size:0.75rem; color:#94a3b8;">({int(config.w_local*100)}% Weight)</span></span>
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
                <span class="val" style="color: {'#ef4444' if local_f.get('motion_blur_metric', 0.0) > 0.45 else '#10b981'};">
                    {local_f.get('motion_blur_metric', 0.0):.3f} {'(⚠️ Blur Defect)' if local_f.get('motion_blur_metric', 0.0) > 0.45 else '(Clean)'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Global Context Card
    dof_val = global_f.get("dof_separation_ratio", 1.0)
    dof_color = "#10b981" if dof_val >= 2.0 else "#94a3b8"
    st.markdown(
        f"""
        <div class="breakdown-card">
            <div class="breakdown-title">
                <span>🌐 Global Context Score</span>
                <span style="color: #c084fc;">{result.s_global:.1f} / 100 <span style="font-size:0.75rem; color:#94a3b8;">({int(config.w_global*100)}% Weight)</span></span>
            </div>
            <div class="metric-row">
                <span>Subject / Background DOF Separation:</span>
                <span class="val" style="color: {dof_color};">{dof_val:.2f}x {'(✨ Bokeh Reward)' if dof_val >= 2.0 else ''}</span>
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
                <span>⚡ Defect & Framing Penalties</span>
                <span style="color: #f43f5e;">Deductions</span>
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
                <b>✂️ Severe Framing Defect (-{result.framing_penalty_pts:.1f} pts):</b><br/>
                Subject head is cropped at image boundary. Score capped at defective usability level (max 25.0).
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result.face_penalty_pts > 0:
        penalties_present = True
        st.markdown(
            f"""
            <div class="penalty-item">
                <b>👤 Turned Away Face (-{result.face_penalty_pts:.1f} pts):</b><br/>
                Subject is turned away or facial features are not visible in portrait (-40% penalty).
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result.motion_penalty_pts > 0:
        penalties_present = True
        st.markdown(
            f"""
            <div class="penalty-item">
                <b>🌪️ Severe Motion Blur (-{result.motion_penalty_pts:.1f} pts):</b><br/>
                Directional gradient anisotropy exceeds threshold ({local_f.get('motion_blur_metric', 0.0):.3f} > 0.450).
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not penalties_present:
        st.markdown(
            """
            <div class="penalty-none">
                <b>✅ No Anomaly Penalties Applied</b><br/>
                Framing is intact, face is clearly visible, and motion stability is confirmed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Formula Math Box
    score_color = "#10b981" if result.unified_score >= config.threshold else "#f43f5e"
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 10px 14px; font-size: 0.83rem; color: #94a3b8; border: 1px dashed rgba(255,255,255,0.1);">
            <b>Formula:</b> [{config.w_local:.2f} × {result.s_local:.1f}] + [{config.w_global:.2f} × {result.s_global:.1f}] = <b>{result.base_score:.1f}</b> base
            {" - " + str(result.total_defect_penalty) + " penalties" if result.total_defect_penalty > 0 else ""}
            = <b style="color: {score_color}; font-size: 0.95rem;">{result.unified_score:.1f} final</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnostics_and_table(
    result: PipelineResult,
    manifest_rows: List[Dict[str, str]],
    baseline_map: Dict[str, str],
):
    """Renders bottom expandable diagnostic telemetry and manifest comparison table."""
    st.markdown("---")
    with st.expander("🔬 Deep Diagnostic Telemetry & Bounding Box Coordinates", expanded=False):
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
            st.markdown("**Raw Local & Global Feature Vectors:**")
            raw_features = {
                "Local": result.local_features,
                "Global": result.global_features,
            }
            st.json(raw_features)

    if manifest_rows:
        with st.expander("📚 Dataset Manifest Context & Phase 3 Baseline Benchmark Table", expanded=False):
            st.caption("Browse how the Context-Aware Phase 4 pipeline compares against baseline No-Reference IQA across event bursts:")
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
