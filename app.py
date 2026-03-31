"""
Unified ML Demo App — Fire Detection | Mushroom Classification | Audio CAPTCHA
Redesigned with PresenceX-inspired bright design system.
"""

import os
import sys
import io
import pickle
import pathlib
import tempfile
import time

import streamlit as st
import numpy as np

# ─────────────────────────────────────────────
# Page config  (must be FIRST streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Vision Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
PKL_DIR = BASE_DIR / "pkls"

VIDEO_PKL  = PKL_DIR / "best_video_fire_model.pkl"
IMAGE_PKL  = PKL_DIR / "optimized_champion_package.pkl"
AUDIO_PKL  = PKL_DIR / "audio_captcha_model.pkl"
NUMERIC_PKL = PKL_DIR / "Market_Basket_Model3.pkl"

# ══════════════════════════════════════════════════════════════════════════════
# ◈  BRIGHT DESIGN SYSTEM — PresenceX language, luminous palette
#    Palette: warm cream canvas, rich slate ink, vivid violet accent, sage green.
#    Typography: DM Serif Display (headlines) + DM Sans (body).
#    Mood: premium editorial studio — bright, airy, confident, authoritative.
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

    /* ══════════════════════════════════════════════
       CSS CUSTOM PROPERTIES — BRIGHT DESIGN TOKENS
    ══════════════════════════════════════════════ */
    :root {
        /* Surfaces */
        --bg-canvas:      #faf8f4;
        --bg-card:        #ffffff;
        --bg-card-hover:  #f7f4ef;
        --bg-input:       #f2efe9;
        --bg-glass:       rgba(255, 255, 255, 0.90);

        /* Borders */
        --border-subtle:  rgba(60, 50, 35, 0.08);
        --border-mid:     rgba(60, 50, 35, 0.15);
        --border-accent:  rgba(109, 61, 210, 0.40);

        /* Typography */
        --text-primary:   #1a1610;
        --text-secondary: #5a5040;
        --text-muted:     #a09078;
        --text-inverse:   #ffffff;

        /* Brand accent — vivid violet */
        --accent:         #6d3dd2;
        --accent-mid:     #8b5cf6;
        --accent-dim:     rgba(109, 61, 210, 0.10);
        --accent-glow:    rgba(109, 61, 210, 0.06);

        /* Status */
        --success:        #1e8c5a;
        --success-dim:    rgba(30, 140, 90, 0.10);
        --warning:        #c4773a;
        --warning-dim:    rgba(196, 119, 58, 0.10);
        --error:          #c0392b;
        --error-dim:      rgba(192, 57, 43, 0.10);
        --info:           #2563eb;
        --info-dim:       rgba(37, 99, 235, 0.08);

        /* Radius */
        --radius-sm:      8px;
        --radius-md:      14px;
        --radius-lg:      20px;
        --radius-xl:      28px;

        /* Shadows */
        --shadow-card:    0 1px 3px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.07);
        --shadow-lift:    0 4px 16px rgba(0,0,0,0.10), 0 20px 48px rgba(0,0,0,0.08);
        --shadow-glow:    0 0 40px rgba(109, 61, 210, 0.10);

        /* Fonts */
        --font-display:   'DM Serif Display', Georgia, serif;
        --font-body:      'DM Sans', system-ui, sans-serif;
    }

    /* ══════════════════════════════════════════════
       GLOBAL RESET & BASE
    ══════════════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: var(--font-body) !important;
        background-color: var(--bg-canvas) !important;
        color: var(--text-primary) !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
        background-color: var(--bg-canvas) !important;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(109,61,210,0.05) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 90%, rgba(30,140,90,0.04) 0%, transparent 50%);
        min-height: 100vh;
    }

    .block-container {
        padding: 2rem 3.5rem 4rem !important;
        max-width: 1280px !important;
    }

    /* ══════════════════════════════════════════════
       WORDMARK / HERO
    ══════════════════════════════════════════════ */
    .px-wordmark {
        display: flex;
        align-items: flex-end;
        gap: 1.2rem;
        padding: 3rem 0 2.5rem;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 2.5rem;
    }
    .px-wordmark .logotype {
        font-family: var(--font-display);
        font-size: 4.5rem !important;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1;
        margin: 0;
    }
    .px-wordmark .logotype em {
        color: var(--accent);
        font-style: italic;
    }
    .px-wordmark .tagline {
        font-size: 1.1rem !important;
        font-weight: 300;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-muted);
        padding-bottom: 0.35rem;
        line-height: 1;
    }
    .px-pill {
        margin-left: auto;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.85rem;
        border: 1px solid var(--border-accent);
        border-radius: 99px;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
        background: var(--accent-dim);
        align-self: center;
    }
    .px-pill::before {
        content: '';
        display: inline-block;
        width: 5px; height: 5px;
        border-radius: 50%;
        background: var(--success);
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.7); }
    }

    /* ══════════════════════════════════════════════
       TABS
    ══════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        gap: 0 !important;
        padding: 0 !important;
        margin-bottom: 2.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        color: var(--text-muted) !important;
        font-family: var(--font-body) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.85rem 1.6rem !important;
        margin-right: 0 !important;
        transition: color 0.25s ease, border-color 0.25s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary) !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0 !important;
    }

    /* ══════════════════════════════════════════════
       CARDS
    ══════════════════════════════════════════════ */
    .px-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-card);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .px-card:hover {
        border-color: var(--border-mid);
        box-shadow: var(--shadow-lift);
    }
    .px-card-accent { border-left: 3px solid var(--accent); }
    .px-card-fire   { border-left: 3px solid #ef4444; }
    .px-card-mushroom { border-left: 3px solid var(--success); }
    .px-card-audio  { border-left: 3px solid var(--info); }

    .px-card-title {
        font-family: var(--font-display);
        font-size: 1.35rem;
        color: var(--text-primary);
        letter-spacing: -0.01em;
        margin: 0 0 0.35rem;
        line-height: 1.2;
    }
    .px-card-subtitle {
        font-size: 0.78rem;
        font-weight: 300;
        color: var(--text-muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0 0 1.5rem;
    }

    /* ══════════════════════════════════════════════
       SECTION LABELS
    ══════════════════════════════════════════════ */
    .px-section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .px-section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border-subtle);
    }

    /* ══════════════════════════════════════════════
       HOME CARDS (feature tiles)
    ══════════════════════════════════════════════ */
    .px-feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 2rem 2.2rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-card);
        transition: all 0.3s ease;
        height: 100%;
    }
    .px-feature-card:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-lift), var(--shadow-glow);
        transform: translateY(-2px);
    }
    .px-feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    .px-feature-title {
        font-family: var(--font-display);
        font-size: 1.3rem;
        color: var(--text-primary);
        letter-spacing: -0.01em;
        margin: 0 0 0.6rem;
    }
    .px-feature-title.fire    { color: #c0392b; }
    .px-feature-title.mushroom { color: var(--success); }
    .px-feature-title.audio   { color: var(--info); }

    .px-feature-body {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.65;
        margin-bottom: 1.2rem;
    }
    .px-feature-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.75rem;
        border-radius: 99px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .px-feature-tag.violet {
        background: var(--accent-dim);
        color: var(--accent);
        border: 1px solid rgba(109,61,210,0.20);
    }
    .px-feature-tag.green {
        background: var(--success-dim);
        color: var(--success);
        border: 1px solid rgba(30,140,90,0.20);
    }
    .px-feature-tag.blue {
        background: var(--info-dim);
        color: var(--info);
        border: 1px solid rgba(37,99,235,0.15);
    }
    .px-feature-tag.red {
        background: var(--error-dim);
        color: var(--error);
        border: 1px solid rgba(192,57,43,0.20);
    }

    /* ══════════════════════════════════════════════
       BUTTONS
    ══════════════════════════════════════════════ */
    div.stButton > button {
        background: var(--accent) !important;
        color: var(--text-inverse) !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-family: var(--font-body) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 0.75rem 1.8rem !important;
        width: 100% !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(109, 61, 210, 0.25) !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover {
        background: #5a2cb8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(109, 61, 210, 0.35) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 6px rgba(109, 61, 210, 0.2) !important;
    }

    /* ══════════════════════════════════════════════
       FORM INPUTS
    ══════════════════════════════════════════════ */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] select {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-mid) !important;
        border-radius: var(--radius-sm) !important;
        color: #000000 !important;
        font-family: var(--font-body) !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #7a7060 !important;
        opacity: 0.7 !important;
    }
    div[data-testid="stTextArea"] textarea:disabled {
        -webkit-text-fill-color: #1a1610 !important;
        background: var(--bg-canvas) !important;
        opacity: 1 !important;
        color: #1a1610 !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-dim) !important;
        outline: none !important;
    }
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #1a1610 !important;
        margin-bottom: 0.45rem !important;
    }

    /* ══════════════════════════════════════════════
       METRICS
    ══════════════════════════════════════════════ */
    div[data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.4rem 1.6rem !important;
        box-shadow: var(--shadow-card) !important;
        transition: border-color 0.3s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-accent) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: var(--font-display) !important;
        font-size: 2.4rem !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        color: var(--success) !important;
    }

    /* ══════════════════════════════════════════════
       DATAFRAME / TABLE
    ══════════════════════════════════════════════ */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
    }

    /* ══════════════════════════════════════════════
       ALERTS / MESSAGES
    ══════════════════════════════════════════════ */
    div[data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        border-left-width: 3px !important;
        font-size: 0.87rem !important;
    }
    div[data-testid="stAlert"][data-type="success"] {
        background: var(--success-dim) !important;
        border-left-color: var(--success) !important;
        color: var(--success) !important;
    }
    div[data-testid="stAlert"][data-type="warning"] {
        background: var(--warning-dim) !important;
        border-left-color: var(--warning) !important;
        color: var(--warning) !important;
    }
    div[data-testid="stAlert"][data-type="error"] {
        background: var(--error-dim) !important;
        border-left-color: var(--error) !important;
        color: var(--error) !important;
    }
    div[data-testid="stAlert"][data-type="info"] {
        background: var(--info-dim) !important;
        border-left-color: var(--info) !important;
        color: var(--info) !important;
    }

    /* ══════════════════════════════════════════════
       PROGRESS BAR
    ══════════════════════════════════════════════ */
    div[data-testid="stProgressBar"] > div {
        background: var(--border-subtle) !important;
        border-radius: 99px !important;
        height: 6px !important;
    }
    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-mid)) !important;
        border-radius: 99px !important;
        transition: width 0.4s ease !important;
    }

    /* Progress bar label text - ensure visibility */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stProgressBar"] + div,
    [data-testid="stProgressBar"] span,
    [data-testid="stProgressBar"] p {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        opacity: 1 !important;
    }

    /* ══════════════════════════════════════════════
       CODE BLOCKS
    ══════════════════════════════════════════════ */
    code, pre {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--accent) !important;
        font-size: 0.82rem !important;
    }

    /* ══════════════════════════════════════════════
       DOWNLOAD BUTTON
    ══════════════════════════════════════════════ */
    div[data-testid="stDownloadButton"] button {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-mid) !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-dim) !important;
    }

    /* ══════════════════════════════════════════════
       GUIDE BLOCK
    ══════════════════════════════════════════════ */
    .px-guide {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.6rem 1.8rem;
    }
    .px-guide-title {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1.1rem;
    }
    .px-guide-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 0.9rem;
        font-size: 0.84rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    .px-guide-icon {
        color: var(--accent);
        flex-shrink: 0;
        font-size: 0.8rem;
        margin-top: 0.1rem;
    }

    /* ══════════════════════════════════════════════
       RESULT BOXES
    ══════════════════════════════════════════════ */
    .px-result {
        border-radius: var(--radius-md);
        padding: 1.2rem 1.6rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        font-weight: 500;
        line-height: 1.6;
        color: #000000 !important; /* Force black for contrast */
    }
    .px-result-fire {
        background: var(--error-dim);
        border-left: 4px solid var(--error);
    }
    .px-result-safe {
        background: var(--success-dim);
        border-left: 4px solid var(--success);
    }
    .px-result-info {
        background: var(--accent-dim);
        border-left: 4px solid var(--accent);
    }
    .px-result-blue {
        background: var(--info-dim);
        border-left: 4px solid var(--info);
    }
    .px-result-green {
        background: var(--success-dim);
        border-left: 4px solid var(--success);
    }
    .px-result-red {
        background: var(--error-dim);
        border-left: 4px solid var(--error);
    }
    /* ══════════════════════════════════════════════
   RESPONSIVE BREAKPOINTS — NO DESIGN CHANGES
══════════════════════════════════════════════ */

/* ── Tablet (max 1024px) ── */
@media (max-width: 1024px) {
    .block-container {
        padding: 1.5rem 2rem 3rem !important;
    }
    .px-wordmark .logotype {
        font-size: 3.2rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
}

/* ── Mobile (max 768px) ── */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2.5rem !important;
    }

    /* Wordmark stacks vertically */
    .px-wordmark {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 1.5rem 0 1.5rem;
    }
    .px-wordmark .logotype {
        font-size: 2.6rem !important;
    }
    .px-wordmark .tagline {
        font-size: 0.85rem !important;
    }
    .px-pill {
        margin-left: 0;
        margin-top: 0.3rem;
    }

    /* Cards */
    .px-card,
    .px-feature-card,
    .px-guide {
        padding: 1.2rem 1.2rem !important;
    }
    .px-card-title {
        font-size: 1.1rem !important;
    }
    .px-feature-title {
        font-size: 1.1rem !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        padding: 1rem 1rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }

    /* Tabs — allow horizontal scroll on tiny screens */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 1rem !important;
        font-size: 0.72rem !important;
        white-space: nowrap !important;
    }

    /* Column padding reset */
    div[data-testid="column"] {
        padding: 0 0.25rem !important;
    }

    /* Result boxes */
    .px-result {
        padding: 1rem 1rem !important;
        font-size: 0.85rem !important;
    }

    /* Buttons */
    div.stButton > button {
        padding: 0.65rem 1.2rem !important;
        font-size: 0.78rem !important;
    }
}

    /* ── Small Mobile (max 480px) ── */
    @media (max-width: 480px) {
        .px-wordmark .logotype {
            font-size: 2rem !important;
        }
        .px-feature-icon {
            font-size: 1.6rem !important;
        }
        .px-feature-body {
            font-size: 0.82rem !important;
        }
        .px-card-subtitle {
            font-size: 0.7rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }
        /* Section labels compress */
        .px-section-label {
            font-size: 0.6rem !important;
        }
        /* Guide items */
        .px-guide-item {
            font-size: 0.78rem !important;
        }
    }

    /* ══════════════════════════════════════════════
       STATUS BADGE
    ══════════════════════════════════════════════ */
    .px-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.7rem;
        border-radius: 99px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .px-status-online {
        background: var(--success-dim);
        color: var(--success);
        border: 1px solid rgba(30, 140, 90, 0.25);
    }
    .px-status-dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse-dot 2s ease-in-out infinite;
    }

    /* ══════════════════════════════════════════════
       FILE UPLOADER
    ══════════════════════════════════════════════ */
    div[data-testid="stFileUploader"] {
        border-radius: var(--radius-md) !important;
    }
    div[data-testid="stFileUploader"] > div {
        border-color: var(--border-mid) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-input) !important;
    }

    /* ══════════════════════════════════════════════
       SCROLLBAR
    ══════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ══════════════════════════════════════════════
       HIDE STREAMLIT CHROME
    ══════════════════════════════════════════════ */
    #MainMenu, footer, header { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
    .st-emotion-cache-z5fcl4 { padding-top: 1rem !important; }

    div[data-testid="stImage"] img {
        border-radius: var(--radius-md) !important;
    }
    div[data-testid="column"] {
        padding: 0 0.75rem !important;
    }
    div[data-testid="column"]:first-child { padding-left: 0 !important; }
    div[data-testid="column"]:last-child  { padding-right: 0 !important; }

    /* Radio buttons */
    div[data-testid="stRadio"] label {
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stRadio"] > div > label > div:first-child {
        border-color: var(--border-mid) !important;
    }

    /* Slider */
    div[data-testid="stSlider"] > div > div > div {
        background: var(--accent) !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-card) !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ══════════════════════════════════════════════════════════════════════════════
# ◈  WORDMARK HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="px-wordmark">
    <div>
        <p class="logotype">ML Vision<em>Suite</em></p>
    </div>
    <div class="tagline">Deep Learning<br>Intelligence</div>
    <div class="px-pill">Models Ready</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ◈  MAIN NAVIGATION TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_home, tab_fire, tab_mushroom, tab_audio, tab_numeric, tab_text = st.tabs([
    "◎  Home",
    "🔥  Fire Detection",
    "🍄  Mushroom Classifier",
    "🔊  CAPTCHA Solver",
    "🛒  Market Basket",
    "📝  Sentiment Analysis",
])

# ─────────────────────────────────────────────────────────────────────────────
# Lazy import helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tf():
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return tf

@st.cache_resource(show_spinner=False)
def load_cv2():
    import cv2
    return cv2

@st.cache_resource(show_spinner=False)
def load_librosa():
    import librosa
    return librosa

# ─────────────────────────────────────────────────────────────────────────────
# Model loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_video_model():
    tf = load_tf()
    with open(VIDEO_PKL, "rb") as f:
        pkg = pickle.load(f)
    model = tf.keras.models.model_from_json(pkg["architecture"])
    model.set_weights(pkg["weights"])
    return model

@st.cache_resource(show_spinner=False)
def load_image_model():
    tf = load_tf()
    with open(IMAGE_PKL, "rb") as f:
        pkg = pickle.load(f)
    model = tf.keras.models.model_from_json(pkg["architecture"])
    model.set_weights(pkg["weights"])
    classes = pkg["classes"]
    return model, classes

@st.cache_resource(show_spinner=False)
def load_audio_model():
    import torch
    import torch.nn as nn
    import torchaudio

    class CRNNAudio(nn.Module):
        def __init__(self, num_classes, input_size=64, hidden_size=256, dropout=0.3):
            super(CRNNAudio, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(p=dropout / 2),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(p=dropout)
            )
            self.rnn = nn.GRU(
                input_size=256, hidden_size=hidden_size, num_layers=2,
                batch_first=True, bidirectional=True, dropout=dropout
            )
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(hidden_size * 2, num_classes)
            )
        def forward(self, x, input_lengths):
            x = x.transpose(1, 2)
            x = self.cnn(x)
            lengths = input_lengths // 4
            x = x.transpose(1, 2)
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_x)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            return self.fc(out), lengths

    device = torch.device('cpu')
    checkpoint = torch.load(AUDIO_PKL, map_location=device, weights_only=False)
    idx_to_char  = checkpoint['idx_to_char']
    blank_idx    = checkpoint['blank_idx']
    num_classes  = checkpoint['num_classes']
    model_config = checkpoint['model_config']
    audio_config = checkpoint['audio_config']
    model = CRNNAudio(
        num_classes=num_classes,
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        dropout=model_config['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, idx_to_char, blank_idx, audio_config, device

@st.cache_resource(show_spinner=False)
def load_numeric_model():
    with open(NUMERIC_PKL, "rb") as f:
        rules = pickle.load(f)
    return rules

@st.cache_resource(show_spinner=False)
def load_text_model():
    from transformers import pipeline
    # Using a robust default model as fallback if specific path is missing
    # Optimized for size and performance
    try:
        # Attempt to load from local if exists, else hub
        model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        model = pipeline("sentiment-analysis")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def predict_fire_frame(frame, model):
    cv2 = load_cv2()
    tf  = load_tf()
    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(rgb, (224, 224))
    normalised = resized.astype("float32") / 255.0
    arr       = tf.expand_dims(normalised, axis=0)
    pred      = model.predict(arr, verbose=0)[0][0]
    return float(1.0 - pred)

def predict_mushroom(img_bytes, model, classes):
    tf = load_tf()
    from tensorflow.keras.preprocessing import image as keras_image
    img   = keras_image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    arr   = keras_image.img_to_array(img)
    batch = tf.expand_dims(arr, axis=0)
    preds = model.predict(batch, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return classes[idx], float(preds[idx]) * 100, list(zip(classes, (preds * 100).tolist()))

def predict_audio_captcha(audio_bytes, model, idx_to_char, blank_idx, audio_config, device):
    import torch
    import torchaudio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        import soundfile as sf
        y, sr = sf.read(tmp_path, dtype='float32')
        if y.ndim == 1:
            waveform = torch.tensor(y).unsqueeze(0)
        else:
            waveform = torch.tensor(y.T)
        if sr != audio_config['sample_rate']:
            waveform = torchaudio.functional.resample(waveform, sr, audio_config['sample_rate'])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_config['sample_rate'],
            n_fft=audio_config['n_fft'],
            hop_length=audio_config['hop_length'],
            n_mels=audio_config['n_mels']
        )
        db_transform = torchaudio.transforms.AmplitudeToDB()
        spec = db_transform(mel_transform(waveform))
        spec = spec.squeeze(0).transpose(0, 1).unsqueeze(0).to(device)
        input_len = torch.tensor([spec.shape[1]], dtype=torch.long).to(device)
        with torch.no_grad():
            logits, out_lens = model(spec, input_len)
            preds = logits.argmax(dim=2)[0][:out_lens[0]].cpu().numpy()
        decoded, prev_idx = [], -1
        for idx in preds:
            if idx != blank_idx and idx != prev_idx:
                decoded.append(idx_to_char[idx])
            prev_idx = idx
        return "".join(decoded).upper()
    finally:
        os.unlink(tmp_path)

def get_recommendations(selected_items, rules, top_n=5):
    """Recommend items based on association rules."""
    if not selected_items:
        return []
    
    recommendations = []
    # Filter rules where antecedents are a subset of selected_items
    matching_rules = rules[rules['antecedents'].apply(lambda x: set(x).issubset(set(selected_items)))]
    
    for _, rule in matching_rules.iterrows():
        consequent = list(rule['consequents'])
        for item in consequent:
            if item not in selected_items:
                recommendations.append({
                    'item': item,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support']
                })
    
    # Sort by confidence and take top N
    recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    
    # Remove duplicates
    seen = set()
    unique_recs = []
    for r in recommendations:
        if r['item'] not in seen:
            unique_recs.append(r)
            seen.add(r['item'])
            
    return unique_recs[:top_n]

def analyze_sentiment(text, model):
    """Analyze sentiment using the transformers pipeline."""
    result = model(text)[0]
    return result['label'], result['score']


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — HOME
# ─────────────────────────────────────────────────────────────────────────────
with tab_home:
    st.markdown('<p class="px-section-label">Available Modules</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="px-feature-card">
            <span class="px-feature-icon">🔥</span>
            <p class="px-feature-title fire">Fire Detection</p>
            <p class="px-feature-body">
                Upload a video clip and our MobileNetV2-based model scans every frame for fire.
                Results include per-frame confidence and a summary verdict.
            </p>
            <span class="px-feature-tag red">MobileNetV2</span>
            <span class="px-feature-tag red">MP4 · AVI · MOV</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="px-feature-card">
            <span class="px-feature-icon">🍄</span>
            <p class="px-feature-title mushroom">Mushroom Classifier</p>
            <p class="px-feature-body">
                Drop any mushroom photo and get an instant species prediction with confidence
                scores across all known classes in the dataset.
            </p>
            <span class="px-feature-tag green">Optimised CNN</span>
            <span class="px-feature-tag green">JPG · PNG · WebP</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="px-feature-card">
            <span class="px-feature-icon">🔊</span>
            <p class="px-feature-title audio">CAPTCHA Solver</p>
            <p class="px-feature-body">
                Upload an audio CAPTCHA file and our Deep CRNN model will transcribe the spoken
                characters automatically via CTC decoding.
            </p>
            <span class="px-feature-tag blue">Deep CRNN + CTC</span>
            <span class="px-feature-tag blue">WAV · MP3</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    row2_col1, row2_col2, row2_col3 = st.columns(3, gap="large")

    with row2_col1:
        st.markdown("""
        <div class="px-feature-card">
            <span class="px-feature-icon">🛒</span>
            <p class="px-feature-title mushroom">Market Basket</p>
            <p class="px-feature-body">
                Discover product associations and item recommendations using high-confidence 
                FP-Growth association rules from retail datasets.
            </p>
            <span class="px-feature-tag green">FP-Growth</span>
            <span class="px-feature-tag green">Association Rules</span>
        </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        st.markdown("""
        <div class="px-feature-card">
            <span class="px-feature-icon">📝</span>
            <p class="px-feature-title audio">Sentiment Analysis</p>
            <p class="px-feature-body">
                Analyze the emotional tone of any text. Use state-of-the-art Transformers 
                to determine if a message is positive, negative, or neutral.
            </p>
            <span class="px-feature-tag blue">DistilBERT / RoBERTa</span>
            <span class="px-feature-tag blue">NLP Pipeline</span>
        </div>
        """, unsafe_allow_html=True)

    with row2_col3:
        # Placeholder for future expansion
        st.markdown("""
        <div class="px-feature-card" style="opacity: 0.5; border-style: dashed;">
            <span class="px-feature-icon">✨</span>
            <p class="px-feature-title">More coming soon</p>
            <p class="px-feature-body">
                We are constantly training and integrating new state-of-the-art models 
                into the Vision Suite.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown('<p class="px-section-label">System Status</p>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4, gap="medium")
    s1.metric("Models Available", "5")
    s2.metric("Framework", "TensorFlow/PyTorch")
    s3.metric("Modern Stack", "Transformers + MLxtend")
    s4.metric("Interface", "Streamlit")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.info("◎  Select a module from the tabs above to begin inference.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FIRE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_fire:
    col_main, col_side = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown('<p class="px-section-label">Video Analysis</p>', unsafe_allow_html=True)

        st.markdown("""
        <div class="px-card px-card-fire">
            <p class="px-card-title">Fire Detection</p>
            <p class="px-card-subtitle">Upload a video — every sampled frame is scored for fire presence</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="video_uploader"
        )

        if uploaded:
            threshold   = 0.5
            sample_rate = 5

            with st.expander("⚙️  Advanced Settings"):
                threshold   = st.slider("Detection threshold (%)", 0, 100, 50, 5) / 100.0
                sample_rate = st.slider("Analyse every N-th frame", 1, 30, 5)

            with st.spinner("Loading fire detection model…"):
                model = load_video_model()
            cv2 = load_cv2()

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            st.video(tmp_path)

            if st.button("🔍  Analyse Video", key="run_fire"):
                cap     = cv2.VideoCapture(tmp_path)
                total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_val = cap.get(cv2.CAP_PROP_FPS) or 25

                fire_frames  = []
                frame_idx    = 0
                checked      = 0
                progress     = st.progress(0, text="Analysing frames…")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_rate == 0:
                        conf = predict_fire_frame(frame, model)
                        if conf >= threshold:
                            fire_frames.append((frame_idx, conf))
                        checked += 1
                        progress.progress(min(frame_idx / max(total, 1), 1.0),
                                          text=f"Frame {frame_idx}/{total}")
                    frame_idx += 1

                cap.release()
                os.unlink(tmp_path)
                progress.empty()

                st.markdown('<p class="px-section-label">Results</p>', unsafe_allow_html=True)

                if fire_frames:
                    peak_frame, peak_conf = max(fire_frames, key=lambda x: x[1])
                    peak_ts = peak_frame / fps_val
                    st.markdown(
                        f'<div class="px-result px-result-fire">🔥 <b>FIRE DETECTED</b> — '
                        f'found in <b>{len(fire_frames)}</b> of <b>{checked}</b> sampled frames.<br>'
                        f'<small>Peak confidence: <b>{peak_conf*100:.1f}%</b> at frame <b>{peak_frame}</b> '
                        f'(~{peak_ts:.1f}s)</small></div>',
                        unsafe_allow_html=True
                    )
                    with st.expander("View all fire frames"):
                        for fr, cf in fire_frames[:50]:
                            st.write(f"Frame {fr} — {cf*100:.1f}%")
                else:
                    st.markdown(
                        f'<div class="px-result px-result-safe">✅ <b>NO FIRE DETECTED</b> in '
                        f'<b>{checked}</b> sampled frames.</div>',
                        unsafe_allow_html=True
                    )

    with col_side:
        st.markdown("""
        <div class="px-guide">
            <p class="px-guide-title">Quick Guide</p>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Supports MP4, AVI, MOV, MKV formats</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Every 5th frame is sampled by default for speed</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Adjust threshold in Advanced Settings for sensitivity</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Model: MobileNetV2 trained on fire/non-fire dataset</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="px-guide" style="margin-top:1rem;">
            <p class="px-guide-title">Model Status</p>
            <span class="px-status px-status-online">
                <span class="px-status-dot"></span>MobileNetV2 Ready
            </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MUSHROOM CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
with tab_mushroom:
    st.markdown('<p class="px-section-label">Species Classification</p>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("""
        <div class="px-card px-card-mushroom">
            <p class="px-card-title">Mushroom Classifier</p>
            <p class="px-card-subtitle">Upload a mushroom photo for instant species prediction</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="image_uploader"
        )

        if uploaded:
            img_bytes = uploaded.read()
            st.image(img_bytes, caption="Uploaded image", use_container_width=True)

    with col_result:
        if uploaded:
            with st.spinner("Loading mushroom model…"):
                model, classes = load_image_model()
            with st.spinner("Classifying…"):
                pred_class, confidence, all_scores = predict_mushroom(img_bytes, model, classes)

            st.markdown('<p class="px-section-label">Prediction Results</p>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="px-result px-result-info">🍄 <b>Predicted Species:</b><br>'
                f'<span style="font-family:var(--font-display);font-size:1.4rem;color:var(--accent)">'
                f'{pred_class}</span></div>',
                unsafe_allow_html=True
            )

            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{confidence:.1f}%")
            m2.metric("Classes Scored", f"{len(all_scores)}")

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown('<p class="px-section-label">Top 5 Predictions</p>', unsafe_allow_html=True)

            top5 = sorted(all_scores, key=lambda x: x[1], reverse=True)[:5]
            for cls, score in top5:
                st.progress(int(score), text=f"{cls}: {score:.1f}%")
        else:
            st.markdown("""
            <div class="px-guide" style="min-height:280px;display:flex;flex-direction:column;justify-content:center;">
                <p class="px-guide-title">Awaiting Input</p>
                <div class="px-guide-item">
                    <span class="px-guide-icon">◦</span>
                    <span>Upload an image to see species prediction here</span>
                </div>
                <div class="px-guide-item">
                    <span class="px-guide-icon">◦</span>
                    <span>Accepts JPG, PNG, WebP, BMP formats</span>
                </div>
                <div class="px-guide-item">
                    <span class="px-guide-icon">◦</span>
                    <span>Top 5 candidates shown with confidence scores</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — AUDIO CAPTCHA SOLVER
# ─────────────────────────────────────────────────────────────────────────────
with tab_audio:
    col_main, col_side = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown('<p class="px-section-label">Audio Transcription</p>', unsafe_allow_html=True)

        st.markdown("""
        <div class="px-card px-card-audio">
            <p class="px-card-title">CAPTCHA Solver</p>
            <p class="px-card-subtitle">Upload an audio CAPTCHA — the CRNN model decodes the spoken characters</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload audio CAPTCHA", type=["wav", "mp3", "ogg", "flac"], key="audio_uploader"
        )

        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes, format=uploaded.type)

            with st.spinner("Loading audio model…"):
                try:
                    model, idx_to_char, blank_idx, audio_config, device = load_audio_model()
                    model_ok = True
                except Exception as e:
                    st.error(f"Could not load audio model: {e}")
                    model_ok = False

            if model_ok:
                if st.button("🔍  Decode CAPTCHA", key="run_audio"):
                    with st.spinner("Analysing audio…"):
                        try:
                            result = predict_audio_captcha(
                                audio_bytes, model, idx_to_char, blank_idx, audio_config, device
                            )
                            st.markdown('<p class="px-section-label">Decoded Result</p>', unsafe_allow_html=True)
                            st.markdown(
                                f'<div class="px-result px-result-blue">🔊 <b>Decoded CAPTCHA:</b><br>'
                                f'<span style="font-family:var(--font-display);font-size:2rem;'
                                f'letter-spacing:0.15em;color:var(--info)">{result}</span></div>',
                                unsafe_allow_html=True
                            )
                            st.metric("Model", "Deep CRNN + CTC Decoding")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            st.info(
                                "Tip: Make sure the audio file is a valid WAV/MP3 CAPTCHA recording "
                                "in the same format the model was trained on."
                            )

    with col_side:
        st.markdown("""
        <div class="px-guide">
            <p class="px-guide-title">Quick Guide</p>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Supports WAV, MP3, OGG, FLAC formats</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Model uses CTC decoding for robust transcription</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Audio is resampled automatically if needed</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Output is shown in uppercase CAPTCHA format</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="px-guide" style="margin-top:1rem;">
            <p class="px-guide-title">Model Status</p>
            <span class="px-status px-status-online">
                <span class="px-status-dot"></span>CRNN + CTC Ready
            </span>
        </div>
        """, unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MARKET BASKET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_numeric:
    col_main, col_side = st.columns([2.5, 1], gap="large")

    with col_main:
        st.markdown("""
        <div class="px-card px-card-mushroom">
            <p class="px-card-title">Market Basket Analysis</p>
            <p class="px-card-subtitle">Discover product associations based on historical purchase data</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Loading association rules…"):
            try:
                rules = load_numeric_model()
                # Extract all unique items from antecedents and consequents
                all_items = sorted(list(set([item for sublist in rules['antecedents'] for item in sublist] + 
                                       [item for sublist in rules['consequents'] for item in sublist])))
                model_ok = True
            except Exception as e:
                st.error(f"Could not load numeric model: {e}")
                model_ok = False

        if model_ok:
            selected = st.multiselect(
                "Select items to find recommendations",
                options=all_items,
                help="Start typing an item name (e.g., 'SPACEBOY' or 'CLOCK')"
            )

            if selected:
                with st.spinner("Finding best associations…"):
                    recommendations = get_recommendations(selected, rules)
                    
                    if recommendations:
                        st.markdown('<p class="px-section-label">Top Recommendations</p>', unsafe_allow_html=True)
                        for i, rec in enumerate(recommendations):
                            st.markdown(f"""
                            <div class="px-result px-result-green" style="margin-bottom:1rem;">
                                <div style="display:flex; justify-content:between; align-items:center;">
                                    <div style="flex-grow:1">
                                        <span style="font-size:0.8rem; color:var(--success); text-transform:uppercase; font-weight:600;">Recommendation #{i+1}</span>
                                        <p style="font-size:1.4rem; font-family:var(--font-display); margin:0; color:var(--text-primary);">{rec['item']}</p>
                                    </div>
                                    <div style="text-align:right">
                                        <span style="font-size:1.5rem; font-weight:700; color:var(--success);">{rec['confidence']:.1%}</span><br>
                                        <span style="font-size:0.7rem; color:var(--text-muted);">Confidence</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No strong associations found for this combination. Try adding more items.")

    with col_side:
        st.markdown("""
        <div class="px-guide">
            <p class="px-guide-title">Market Basket Logic</p>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Uses FP-Growth algorithm for efficient rule mining</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Confidence represents the probability of buying the recommendation given the current items</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Lift indicates how much more likely the items are bought together than independent</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_text:
    col_main, col_side = st.columns([2.5, 1], gap="large")

    with col_main:
        st.markdown("""
        <div class="px-card px-card-audio">
            <p class="px-card-title">Sentiment Analysis</p>
            <p class="px-card-subtitle">Analyze the emotional tone of text using Transformer models</p>
        </div>
        """, unsafe_allow_html=True)

        user_text = st.text_area(
            "Enter text to analyze",
            placeholder="Type or paste your message here...",
            height=150
        )

        st.markdown('<p style="font-size:0.8rem; margin-bottom:0.5rem; color:#000000; font-weight:600;">Quick-test Samples:</p>', unsafe_allow_html=True)
        samples = [
            "Viral social media claims that the Prime Minister died and was replaced by AI-generated footage",
            "The first stage of the U.S.-backed peace plan, which established the initial truce last October, is now considered largely complete."
        ]
        
        cols = st.columns(len(samples))
        for i, sample in enumerate(samples):
            if cols[i].button(f"Sample {i+1}", key=f"sample_{i}"):
                st.session_state["sentiment_input"] = sample
                st.rerun()

        if "sentiment_input" in st.session_state:
            user_text = st.session_state["sentiment_input"]
            st.text_area("Selected sample", value=user_text, height=100, disabled=True)
            del st.session_state["sentiment_input"]

        if st.button("✨ Analyze Sentiment", key="run_sentiment") and user_text:
            with st.spinner("Processing text with Transformers…"):
                try:
                    model = load_text_model()
                    label, score = analyze_sentiment(user_text, model)
                    
                    st.markdown('<p class="px-section-label">Analysis Result</p>', unsafe_allow_html=True)
                    
                    color_class = "px-result-blue"
                    if label.upper() in ["POSITIVE", "LABEL_2"]: color_class = "px-result-green"
                    elif label.upper() in ["NEGATIVE", "LABEL_0"]: color_class = "px-result-red"
                    
                    st.markdown(f"""
                    <div class="px-result {color_class}" style="color:#000000 !important;">
                        <p style="font-size:0.8rem; text-transform:uppercase; font-weight:600; margin-bottom:0.5rem; color:#000000;">Tone Identified</p>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:2.5rem; font-family:var(--font-display); color:#000000;">{label.replace('LABEL_0', 'NEGATIVE').replace('LABEL_1', 'NEUTRAL').replace('LABEL_2', 'POSITIVE')}</span>
                            <span style="font-size:1.5rem; font-weight:700; color:#000000;">{score:.1%}</span>
                        </div>
                        <div style="margin-top:1rem; width:100%; height:8px; background:rgba(0,0,0,0.1); border-radius:4px; overflow:hidden;">
                            <div style="width:{score*100}%; height:100%; background:#000000;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Text analysis failed: {e}")

    with col_side:
        st.markdown("""
        <div class="px-guide">
            <p class="px-guide-title">NLP Insights</p>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Powered by DistilBERT/RoBERTa architectures</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Analysis covers semantic tone and emotional weight</span>
            </div>
            <div class="px-guide-item">
                <span class="px-guide-icon">◦</span>
                <span>Scores indicate the model's confidence in the result</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

