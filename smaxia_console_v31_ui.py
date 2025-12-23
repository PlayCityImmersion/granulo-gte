import streamlit as st
import pandas as pd
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    layout="wide",
    page_title="SMAXIA - Console V31 (Saturation Proof)"
)

st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# =============================================================================
# CSS — UI SMAXIA PREMIUM (VERROUILLÉ)
# =============================================================================
st.markdown("""
<style>

/* ==============================
   SIDEBAR
================================*/
section[data-testid="stSidebar"] {
    background-color: #f8fafc;
}

/* ==============================
   HEADER QC — WAHOO VERSION
================================*/
.qc-header-box {
    background: #f8fafc;
    border-left: 6px solid #2563eb;
    padding: 14px 16px;
    margin-bottom: 12px;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}

.qc-chap-line{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;
    font-size: 0.86em;
    font-weight: 800;
    color: #475569;
    margin-bottom: 6px;
}

.qc-title-line{
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 8px;
}

.qc-id-pill{
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #c2410c;
    font-weight: 900;
    font-size: 0.95em;
    padding: 3px 8px;
    border-radius: 999px;
}

.qc-title-text{
    color: #0f172a;
    font-weight: 800;
    font-size: 1.08em;
}

.qc-meta-line{
    font-family: ui-monospace, monospace;
    font-size: 0.86em;
    font-weight: 800;
    color: #334155;
    background: #e2e8f0;
    padding: 6px 10px;
    border-radius: 8px;
    display: inline-block;
}

/* ==============================
   BLOCKS
================================*/
.trigger-item {
    background-color: #fff1f2;
    color: #991b1b;
    padding: 6px 10px;
    margin-bottom: 6px;
    border-radius: 6px;
    border-left: 4px solid #f87171;
    font-weight: 600;
}

.ari-step {
    background-color: #f1f5f9;
    padding: 6px 10px;
    border-radius: 6px;
    font-family: ui-monospace, monospace;
    font-size: 0.85em;
    margin-bottom: 6px;
    border: 1px dashed #cbd5e1;
}

.frt-block {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
}

.frt-title {
    font-weight: 800;
    font-size: 0.75em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.frt-content {
    font-size: 0.95em;
    line-height: 1.6;
    white-space: pre-wrap;
}

.file-block {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    margin-bottom: 10px;
}

.file-header {
    background: #f1f5f9;
    padding: 6px 10px;
    font-weight: 700;
    font-size: 0.85em;
}

.qi-item {
    padding: 8px 12px;
    border-left: 4px solid #9333ea;
    font-family: Georgia, serif;
    font-size: 0.95em;
}

.sat-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — PARAMÈTRES ACADÉMIQUES
# =============================================================================
with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    st.multiselect(
        "Chapitres",
        ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"]
    )

# =============================================================================
# TABS
# =============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# =============================================================================
# ONGLET 1 — USINE (UI ONLY)
# =============================================================================
with tab_usine:
    st.subheader("🔌 Injection des sujets")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.text_area("URLs Sources (références)", "https://apmep.fr", height=70)
    with c2:
        st.number_input("Volume de sujets", 5, 500, 15, step=5)
        st.button("🚀 LANCER L'USINE")

    st.divider()

    # -----------------------------
    # SUJETS TRAITÉS (VIDE PAR DÉFAUT)
    # -----------------------------
    st.markdown("### 📥 Sujets traités")
    st.dataframe(
        pd.DataFrame(columns=["Fichier", "Nature", "Année", "Téléchargement"]),
        use_container_width=True
    )
    st.caption("⚠️ Données affichées uniquement après branchement du moteur réel.")

    st.divider()

    # -----------------------------
    # BASE DE CONNAISSANCE QC (MOCK)
    # -----------------------------
    st.markdown("### 🧠 Base de Connaissance (QC)")
    st.markdown("#### 📘 Chapitre : Suites Numériques")

    # MOCK QC — STRUCTURE UNIQUEMENT
    st.markdown("""
    <div class="qc-header-box">
        <div class="qc-chap-line">────────── Chapitre : Suites Numériques</div>
        <div class="qc-title-line">
            <span class="qc-id-pill">[ QC-03 ]</span>
            <span class="qc-title-text">Comment lever une indétermination (limite)</span>
        </div>
        <div class="qc-meta-line">
            Score(q)=— | n_q=— | Ψ=0.85 | N_tot=— | t_rec=—
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.expander("🔥 Déclencheurs"):
            st.markdown("<div class='trigger-item'>« calculer la limite »</div>", unsafe_allow_html=True)
            st.markdown("<div class='trigger-item'>« limite quand n → +∞ »</div>", unsafe_allow_html=True)

    with c2:
        with st.expander("⚙️ ARI"):
            st.markdown("<div class='ari-step'>1. Identifier le terme dominant</div>", unsafe_allow_html=True)
            st.markdown("<div class='ari-step'>2. Factoriser</div>", unsafe_allow_html=True)

    with c3:
        with st.expander("🧾 FRT"):
            st.markdown("""
            <div class="frt-block">
                <div class="frt-title">Quand utiliser</div>
                <div class="frt-content">Forme indéterminée ∞/∞</div>
            </div>
            """, unsafe_allow_html=True)

    with c4:
        with st.expander("📄 Qi"):
            st.markdown("""
            <div class="file-block">
                <div class="file-header">📁 Sujet_MATHS_BAC_2022.pdf</div>
                <div class="qi-item">Déterminer la limite de la suite.</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # -----------------------------
    # SATURATION — UI ONLY
    # -----------------------------
    st.markdown("### 📈 Analyse de saturation")
    st.caption("Visualisation UI — aucune logique Granulo implémentée ici.")

    st.line_chart(
        pd.DataFrame({
            "Sujets": [10, 30, 50, 70, 100],
            "QC découvertes": [2, 4, 6, 6, 6]
        }),
        x="Sujets",
        y="QC découvertes"
    )

# =============================================================================
# ONGLET 2 — AUDIT (UI ONLY)
# =============================================================================
with tab_audit:
    st.subheader("🔍 Audit du moteur Granulo")

    st.markdown("### ✅ Audit interne (sujet traité)")
    st.info("Objectif : chaque Qi doit mapper vers UNE et UNE SEULE QC.")
    st.metric("Résultat attendu", "100 %")

    st.divider()

    st.markdown("### 🌍 Audit externe (sujet inconnu)")
    st.file_uploader("Importer un sujet PDF externe", type="pdf")
    st.info("Indicateur cible : taux ≥ 95 %")

    st.caption(
        "SMAXIA — Console V31 | UI contractuelle. "
        "Aucune logique métier, aucun calcul Granulo dans ce fichier."
    )
