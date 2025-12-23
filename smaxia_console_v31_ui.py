# smaxia_console_v31_ui.py
import streamlit as st
import pandas as pd
import numpy as np

from smaxia_granulo_engine_test import run_granulo_test

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(
    page_title="SMAXIA - Console V31 (Saturation Proof)",
    layout="wide"
)

st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# CSS – UI CONTRACTUELLE (INCHANGÉE)
# ==============================================================================
st.markdown("""
<style>
.qc-box {
    background:#f8fafc;
    border-left:6px solid #2563eb;
    padding:16px;
    border-radius:6px;
    margin-bottom:16px;
}
.qc-chap {
    font-size:0.85em;
    font-weight:800;
    color:#475569;
    text-transform:uppercase;
}
.qc-title {
    font-size:1.15em;
    font-weight:900;
    color:#0f172a;
    margin-top:6px;
}
.qc-meta {
    margin-top:8px;
    font-family:monospace;
    font-size:0.85em;
    background:#e5e7eb;
    padding:4px 8px;
    border-radius:4px;
    display:inline-block;
}

/* TRIGGERS */
.trigger {
    background:#fff1f2;
    border-left:4px solid #ef4444;
    padding:6px 10px;
    border-radius:4px;
    margin-bottom:6px;
    font-weight:600;
}

/* ARI */
.ari-step {
    background:#f3f4f6;
    border:1px dashed #cbd5f5;
    padding:6px;
    border-radius:4px;
    margin-bottom:5px;
    font-family:monospace;
}

/* FRT */
.frt {
    padding:12px;
    border-radius:6px;
    margin-bottom:8px;
    border-left:6px solid;
}
.frt-usage { background:#fff7ed; border-color:#f59e0b; }
.frt-method { background:#f0fdf4; border-color:#22c55e; }
.frt-trap { background:#fef2f2; border-color:#ef4444; }
.frt-conc { background:#eff6ff; border-color:#3b82f6; }

.frt-title {
    font-weight:900;
    font-size:0.75em;
    text-transform:uppercase;
    margin-bottom:6px;
}

/* QI */
.file-box {
    border:1px solid #e5e7eb;
    border-radius:6px;
    margin-bottom:10px;
}
.file-header {
    background:#f1f5f9;
    padding:8px 12px;
    font-weight:700;
}
.qi {
    padding:8px 12px;
    border-left:4px solid #8b5cf6;
    font-family:Georgia, serif;
}

/* SATURATION */
.sat-box {
    background:#f0f9ff;
    border:1px solid #bae6fd;
    padding:16px;
    border-radius:8px;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR – PARAMÈTRES ACADÉMIQUES (INCHANGÉE)
# ==============================================================================
with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    st.multiselect(
        "Chapitres",
        ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
        default=["SUITES NUMÉRIQUES"]
    )

# ==============================================================================
# TABS (INCHANGÉ)
# ==============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# ==============================================================================
# ONGLET 1 – USINE
# ==============================================================================
with tab_usine:

    # --------------------------------------------------------------------------
    # ZONE 1 – INJECTION DES SUJETS (INCHANGÉE)
    # --------------------------------------------------------------------------
    st.sub
