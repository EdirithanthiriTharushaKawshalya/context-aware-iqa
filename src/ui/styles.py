"""
Design System — Context-Aware IQA Research Demonstrator
Professional dark-theme CSS with structured typography and component tokens.
"""

# ---------------------------------------------------------------------------
# SVG Icon Library  (Lucide icon set, 16px, stroke-only, no fill)
# ---------------------------------------------------------------------------

ICON_CHECK = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:-2px;margin-right:5px;">'
    '<polyline points="20 6 9 17 4 12"></polyline></svg>'
)

ICON_ALERT = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:-2px;margin-right:5px;">'
    '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>'
    '<line x1="12" y1="9" x2="12" y2="13"></line>'
    '<line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
)

ICON_SHIELD = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:-2px;margin-right:5px;">'
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
)

ICON_TARGET = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:-2px;margin-right:5px;">'
    '<circle cx="12" cy="12" r="10"></circle>'
    '<circle cx="12" cy="12" r="6"></circle>'
    '<circle cx="12" cy="12" r="2"></circle></svg>'
)

# ---------------------------------------------------------------------------
# CSS — Professional Dark Design System
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ─── Reset & Base ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    letter-spacing: -0.012em;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', 'Cascadia Code', monospace !important;
}

/* ─── Layout & Streamlit Header Reset ─────────────────────────────────── */
header[data-testid="stHeader"],
[data-testid="stHeader"],
.stAppHeader {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}

#MainMenu, footer {
    display: none !important;
    visibility: hidden !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px;
}

/* ─── Design Tokens — Dark Theme ─────────────────────────────────────── */
:root {
    /* Surface */
    --surface-base:    #0d1117;
    --surface-raised:  #161b22;
    --surface-overlay: #1c2333;
    --surface-border:  #30363d;

    /* Accent */
    --accent-blue:    #58a6ff;
    --accent-green:   #3fb950;
    --accent-red:     #f85149;
    --accent-amber:   #d29922;
    --accent-purple:  #bc8cff;

    /* Text */
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #6e7681;

    /* Shared */
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
}

/* ─── Top Navigation Bar ─────────────────────────────────────────────── */
.top-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 18px;
    background: var(--surface-raised);
    border: 1px solid var(--surface-border);
    border-radius: var(--radius-md);
    margin-bottom: 20px;
    box-shadow: var(--shadow-sm);
}

.nav-brand {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.nav-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.3;
}

.nav-subtitle {
    font-size: 0.76rem;
    color: var(--text-muted);
    font-weight: 400;
}

.nav-badge {
    display: flex;
    align-items: center;
    gap: 8px;
}

.nav-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--surface-overlay);
    border: 1px solid var(--surface-border);
    padding: 3px 8px;
    border-radius: var(--radius-sm);
}

.nav-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 6px var(--accent-green);
    display: inline-block;
}

/* ─── KPI Cards ──────────────────────────────────────────────────────── */
.kpi-card {
    background: var(--surface-raised);
    border: 1px solid var(--surface-border);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: var(--shadow-sm);
}

.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
}

.kpi-value-unit {
    font-size: 0.9rem;
    color: var(--text-muted);
    font-weight: 400;
}

.kpi-subtext {
    font-size: 0.73rem;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* ─── Status Badges ──────────────────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.status-keep {
    background: rgba(63,185,80,0.12);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.3);
}

.status-review {
    background: rgba(248,81,73,0.12);
    color: #f85149;
    border: 1px solid rgba(248,81,73,0.3);
}

/* ─── Confidence Tags ─────────────────────────────────────────────────── */
.conf-tag {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

.conf-high {
    background: rgba(63,185,80,0.12);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.25);
}

.conf-med {
    background: rgba(210,153,34,0.12);
    color: #d29922;
    border: 1px solid rgba(210,153,34,0.25);
}

.conf-low {
    background: rgba(248,81,73,0.12);
    color: #f85149;
    border: 1px solid rgba(248,81,73,0.25);
}

/* ─── Section Headers ─────────────────────────────────────────────────── */
.section-header {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--surface-border);
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ─── Score Breakdown Cards ──────────────────────────────────────────── */
.breakdown-card {
    background: var(--surface-raised);
    border: 1px solid var(--surface-border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    margin-bottom: 10px;
}

.breakdown-title {
    font-size: 0.80rem;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--surface-border);
}

.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    color: var(--text-secondary);
    padding: 5px 0;
    border-bottom: 1px solid rgba(48,54,61,0.5);
}

.metric-row:last-child {
    border-bottom: none;
}

.metric-row span.val {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    font-size: 0.76rem;
}

/* ─── Defect Penalty Items ───────────────────────────────────────────── */
.penalty-item {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.25);
    border-left: 3px solid #f85149;
    padding: 8px 12px;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin-bottom: 6px;
    font-size: 0.78rem;
    color: #f28b82;
    line-height: 1.5;
}

.penalty-tag {
    display: inline-block;
    background: rgba(248,81,73,0.2);
    color: #f85149;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 3px;
    margin-right: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.penalty-none {
    background: rgba(63,185,80,0.08);
    border: 1px solid rgba(63,185,80,0.25);
    border-left: 3px solid #3fb950;
    padding: 8px 12px;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 0.78rem;
    color: #7ee787;
    line-height: 1.5;
}

.clean-tag {
    display: inline-block;
    background: rgba(63,185,80,0.2);
    color: #3fb950;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 3px;
    margin-right: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ─── Formula / Code Box ──────────────────────────────────────────────── */
.formula-container {
    background: var(--surface-overlay);
    border: 1px solid var(--surface-border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    color: var(--text-secondary);
    margin-top: 10px;
    line-height: 1.7;
}

/* ─── Divider ─────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid var(--surface-border);
    margin: 20px 0;
}
</style>
"""


# ---------------------------------------------------------------------------
# HTML Render Helpers
# ---------------------------------------------------------------------------

def render_html_header() -> str:
    return """
    <div class="top-navbar">
        <div class="nav-brand">
            <span class="nav-title">Context-Aware Image Quality Assessment</span>
            <span class="nav-subtitle">Phase 4 Multimodal Fusion Pipeline</span>
        </div>
        <div class="nav-badge">
            <span class="nav-status-dot"></span>
            <span class="nav-tag">Research Demonstrator v1.0</span>
        </div>
    </div>
    """


def render_html_kpi_score(score: float, base_score: float, threshold: float) -> str:
    color = "#3fb950" if score >= threshold else "#f85149"
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-label">Unified Quality Score</div>
            <div class="kpi-value" style="color:{color};">{score:.1f}<span class="kpi-value-unit"> / 100</span></div>
        </div>
        <div class="kpi-subtext">Base fused: {base_score:.1f}&nbsp;&nbsp;|&nbsp;&nbsp;Threshold: {threshold:.0f}</div>
    </div>
    """


def render_html_decision_badge(is_keep: bool) -> str:
    badge = (
        f'<span class="status-pill status-keep">{ICON_CHECK}KEEP</span>'
        if is_keep
        else f'<span class="status-pill status-review">{ICON_ALERT}REVIEW</span>'
    )
    desc = "Passed technical and composition criteria" if is_keep else "Flagged for manual review or culling"
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-label">Decision Classification</div>
            <div style="margin:8px 0 6px 0;">{badge}</div>
        </div>
        <div class="kpi-subtext">{desc}</div>
    </div>
    """


def render_html_confidence_gauge(conf_pct: float, conf_tier: str, score: float, error_margin: float, dist: float) -> str:
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-label">Confidence Index</div>
            <div class="kpi-value" style="color:#58a6ff;">{conf_pct:.1f}<span class="kpi-value-unit">%</span></div>
        </div>
        <div>
            <span class="conf-tag conf-{conf_tier}">{ICON_SHIELD}{score:.1f} ± {error_margin:.1f}</span>
            <div class="kpi-subtext" style="margin-top:5px;">{dist:.1f} pts from boundary</div>
        </div>
    </div>
    """


def render_html_ground_truth(gt_label: str, match_flag: bool, defect_flag: str, baseline_score: str) -> str:
    match_color = "#3fb950" if match_flag else "#d29922"
    match_text = "Aligned" if match_flag else "Divergent"
    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-label">Ground Truth Alignment</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e6edf3;margin-top:6px;font-family:'JetBrains Mono',monospace;">
                {gt_label}
                <span style="font-size:0.72rem;font-weight:600;color:{match_color};margin-left:8px;text-transform:uppercase;letter-spacing:0.06em;">{match_text}</span>
            </div>
        </div>
        <div class="kpi-subtext" style="margin-top:4px;">
            Defect: <b style="color:#e6edf3;">{defect_flag}</b>&nbsp;&nbsp;|&nbsp;&nbsp;Baseline: <b style="color:#e6edf3;">{baseline_score}</b>
        </div>
    </div>
    """
