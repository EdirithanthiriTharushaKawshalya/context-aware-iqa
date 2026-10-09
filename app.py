"""
Context-Aware Image Quality Assessment (IQA) - Phase 4 Research Demonstrator
Application Controller Entrypoint
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Phase4Pipeline
from src.ui.styles import CUSTOM_CSS
from src.ui.data_manager import DataManager
from src.ui.components import (
    render_hero_header,
    render_sidebar_controls,
    render_kpi_dashboard,
    render_visual_inspection,
    render_score_breakdown,
    render_diagnostics_and_table,
)

# -----------------------------------------------------------------------------
# Application Setup & Resource Caching
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Context-Aware IQA | Phase 4 Demonstrator",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Phase 4 AI Pipeline Models...")
def get_pipeline() -> Phase4Pipeline:
    """Initializes and caches the Phase 4 pipeline neural weights."""
    return Phase4Pipeline()


@st.cache_data
def get_manifest_and_baselines():
    """Loads manifest rows and baseline benchmark results."""
    manifest_rows = DataManager.load_manifest(PROJECT_ROOT / "dataset_manifest.csv")
    baseline_map = DataManager.load_baselines(PROJECT_ROOT / "baseline_results.csv")
    return manifest_rows, baseline_map


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    pipeline = get_pipeline()
    manifest_rows, baseline_map = get_manifest_and_baselines()

    # 1. Header Banner
    render_hero_header()

    # 2. Sidebar Controls
    raw_img, manifest_meta, config, visual_flags = render_sidebar_controls(
        manifest_rows=manifest_rows,
        project_root=PROJECT_ROOT,
    )

    if raw_img is None:
        st.info("👈 Please select an image from the dataset manifest or upload a photograph in the sidebar.")
        return

    # 3. Model Inference Execution
    with st.spinner("Running Context-Aware Phase 4 assessment..."):
        result = pipeline.run(raw_img, config)

    # 4. Top KPI Cards
    render_kpi_dashboard(result, config, manifest_meta, baseline_map)

    # 5. Split View: Visual Inspection (Left) vs Multi-Stage Breakdown (Right)
    col_img, col_metrics = st.columns([1.15, 0.95], gap="large")
    with col_img:
        render_visual_inspection(raw_img, result, visual_flags)
    with col_metrics:
        render_score_breakdown(result, config)

    # 6. Deep Telemetry & Manifest Benchmark Table
    render_diagnostics_and_table(result, manifest_rows, baseline_map)


if __name__ == "__main__":
    main()
