"""
UI Design Tokens, CSS Stylesheets, and HTML Component Renderers
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
}
.hero-title {
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(120deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.45;
}

/* KPI Metric Cards */
.kpi-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 18px;
    transition: transform 0.2s ease, border-color 0.2s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
}
.kpi-subtext {
    font-size: 0.8rem;
    color: #cbd5e1;
}

/* Badges */
.badge-keep {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.badge-review {
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #fb7185;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Confidence Tiers */
.confidence-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 4px;
}
.conf-high {
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.conf-med {
    background: rgba(245, 158, 11, 0.2);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.conf-low {
    background: rgba(239, 68, 68, 0.2);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Breakdown Cards */
.breakdown-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.breakdown-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.metric-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #94a3b8;
    padding: 4px 0;
    border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
}
.metric-row:last-child {
    border-bottom: none;
}
.metric-row span.val {
    color: #e2e8f0;
    font-weight: 600;
}

/* Penalties */
.penalty-item {
    background: rgba(239, 68, 68, 0.12);
    border-left: 3px solid #ef4444;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #fca5a5;
}
.penalty-none {
    background: rgba(16, 185, 129, 0.08);
    border-left: 3px solid #10b981;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #86efac;
}
</style>
"""


def render_html_kpi_score(score: float, base_score: float, threshold: float) -> str:
    color = "#10b981" if score >= threshold else "#f43f5e"
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-label">Unified Quality Score</div>
            <div class="kpi-value" style="color: {color};">{score:.1f}<span style="font-size:1.1rem;color:#64748b;"> / 100</span></div>
        </div>
        <div class="kpi-subtext">Base: {base_score:.1f} • Cutoff: {threshold:.0f}</div>
    </div>
    """


def render_html_decision_badge(is_keep: bool) -> str:
    badge = (
        '<span class="badge-keep">✅ KEEP (USABLE)</span>'
        if is_keep
        else '<span class="badge-review">⚠️ REVIEW (CULLED)</span>'
    )
    desc = "Meets album delivery standards" if is_keep else "Defect flagged for culling"
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-label">Decision Classification</div>
            <div style="margin: 10px 0 6px 0;">{badge}</div>
        </div>
        <div class="kpi-subtext">{desc}</div>
    </div>
    """


def render_html_confidence_gauge(conf_pct: float, conf_tier: str, score: float, error_margin: float, dist: float) -> str:
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-label">Confidence & Uncertainty</div>
            <div class="kpi-value" style="color: #60a5fa;">{conf_pct:.1f}%</div>
        </div>
        <div>
            <span class="confidence-pill conf-{conf_tier}">Score: {score:.1f} ± {error_margin:.1f} pts</span>
            <div class="kpi-subtext" style="margin-top:4px;">{dist:.1f} pts from decision boundary</div>
        </div>
    </div>
    """


def render_html_ground_truth(gt_label: str, match_icon: str, defect_flag: str, baseline_score: str) -> str:
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-label">Ground Truth Alignment</div>
            <div style="font-size: 1.15rem; font-weight:700; color: #f8fafc; margin-top:4px;">
                Expert: <span style="color:#a78bfa;">{gt_label}</span> ({match_icon})
            </div>
        </div>
        <div class="kpi-subtext">
            Defect Tag: <b>{defect_flag}</b><br/>
            Baseline (BRISQUE): <b>{baseline_score}</b>
        </div>
    </div>
    """
