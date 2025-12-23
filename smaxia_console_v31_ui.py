import streamlit as st
import pandas as pd
import random
from collections import defaultdict
from datetime import datetime

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(layout="wide", page_title="SMAXIA - Console V31")
st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# STYLE PREMIUM SMAXIA
# ==============================================================================
st.markdown("""
<style>
.qc-header {
    background: #f8fafc;
    border-left: 6px solid #2563eb;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 10px;
}
.qc-line1 { font-size: 0.9em; font-weight: 700; color: #475569; }
.qc-line2 { font-size: 1.15em; font-weight: 800; color: #0f172a; margin-top: 2px; }
.qc-line3 {
    margin-top: 6px;
    font-family: monospace;
    background: #e5e7eb;
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 700;
    color: #1f2937;
}
.trigger { background:#fff1f2; border-left:4px solid #ef4444; padding:6px 10px; margin-bottom:4px; font-weight:600; }
.ari { background:#f3f4f6; border:1px dashed #cbd5f5; padding:6px; margin-bottom:4px; font-family:monospace; }
.frt { border:1px solid #e5e7eb; border-left:4px solid #2563eb; padding:10px; margin-bottom:6px; background:white; }
.qi-file { background:#f1f5f9; padding:6px 10px; font-weight:700; }
.qi { border-left:3px solid #9333ea; padding:6px 10px; font-family:Georgia; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# PROGRAMME FRANCE
# ==============================================================================
PROGRAMME = {
    "MATHS": [
        "SUITES NUMÉRIQUES",
        "FONCTIONS",
        "PROBABILITÉS",
        "GÉOMÉTRIE"
    ],
    "PHYSIQUE": [
        "CINÉMATIQUE",
        "DYNAMIQUE",
        "ONDES",
        "ÉLECTRICITÉ"
    ]
}

# ==============================================================================
# DONNÉES QC (STUB UI)
# ==============================================================================
QC_DATA = [
    {
        "Chapitre": "SUITES NUMÉRIQUES",
        "QC_ID": "QC-03",
        "Titre": "Comment lever une indétermination (limite) ?",
        "Score": 212,
        "n_q": 25,
        "Psi": 0.85,
        "N_tot": 60,
        "t_rec": 2.0,
        "Triggers": [
            "calculer la limite",
            "limite quand n tend vers +infini",
            "étudier la convergence"
        ],
        "ARI": [
            "1. Identifier le terme dominant",
            "2. Factoriser",
            "3. Utiliser les limites usuelles",
            "4. Conclure"
        ],
        "FRT": [
            ("Quand utiliser", "Forme indéterminée infini / infini."),
            ("Méthode", "Identifier le terme dominant.\nFactoriser.\nAppliquer les limites usuelles."),
            ("Pièges", "Règle des signes sans factorisation."),
            ("Conclusion", "La suite converge vers une limite finie.")
        ],
        "Qi": {
            "Sujet_MATHS_INTERRO_2021.pdf": [
                "Déterminer la limite. [Ref:94]",
                "Calculer la limite en +∞. [Ref:77]"
            ],
            "Sujet_MATHS_BAC_2024.pdf": [
                "Déterminer la limite. [Ref:71]",
                "Calculer la limite en +∞. [Ref:63]"
            ]
        }
    }
]

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chapitres = st.multiselect(
        "Chapitres",
        PROGRAMME[matiere],
        default=[PROGRAMME[matiere][0]]
    )

# ==============================================================================
# TABS
# ==============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# ==============================================================================
# ONGLET USINE
# ==============================================================================
with tab_usine:

    st.subheader("🧪 Injection des sujets")
    col1, col2 = st.columns([3,1])
    with col1:
        urls = st.text_area("URLs Sources (références)", "https://apmep.fr")
    with col2:
        volume = st.number_input("Volume de sujets", 5, 200, 15)
        lancer = st.button("🚀 LANCER L'USINE")

    st.divider()

    st.subheader("📥 Sujets traités")
    df_sources = pd.DataFrame([
        {"Fichier":"Sujet_MATHS_INTERRO_2021.pdf","Nature":"INTERRO","Année":2021,"Source":"APMEP"},
        {"Fichier":"Sujet_MATHS_BAC_2024.pdf","Nature":"BAC","Année":2024,"Source":"Banque BAC"}
    ])
    st.dataframe(df_sources, use_container_width=True)

    st.divider()
    st.subheader("🧠 Base de connaissance (QC)")

    for qc in QC_DATA:
        if qc["Chapitre"] not in chapitres:
            continue

        st.markdown(f"""
        <div class="qc-header">
            <div class="qc-line1">Chapitre : {qc["Chapitre"]} &nbsp;&nbsp; [ {qc["QC_ID"]} ]</div>
            <div class="qc-line2">{qc["Titre"]}</div>
            <div class="qc-line3">
            Score(q)={qc["Score"]} | n_q={qc["n_q"]} | Ψ={qc["Psi"]} | N_tot={qc["N_tot"]} | t_réc={qc["t_rec"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("**🔥 Déclencheurs**")
            for t in qc["Triggers"]:
                st.markdown(f"<div class='trigger'>{t}</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("**⚙️ ARI**")
            for a in qc["ARI"]:
                st.markdown(f"<div class='ari'>{a}</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("**📘 FRT**")
            for title, txt in qc["FRT"]:
                st.markdown(f"<div class='frt'><b>{title}</b><br>{txt}</div>", unsafe_allow_html=True)

        with c4:
            st.markdown(f"**📄 Qi ({qc['n_q']})**")
            for f, qis in qc["Qi"].items():
                st.markdown(f"<div class='qi-file'>📁 {f}</div>", unsafe_allow_html=True)
                for q in qis:
                    st.markdown(f"<div class='qi'>{q}</div>", unsafe_allow_html=True)

# ==============================================================================
# ONGLET AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("🔍 Audit du moteur Granulo")

    st.markdown("### ✅ Audit interne (sujet déjà traité)")
    st.info("Objectif : vérifier que chaque Qi mappe vers UNE et UNE SEULE QC.\nRésultat attendu : **100 %**")

    st.divider()

    st.markdown("### 🌍 Audit externe (sujet inconnu du moteur)")
    st.file_uploader("Importer un sujet PDF externe", type="pdf")
    st.info("Indicateur clé : **taux ≥ 95 %**")

    st.caption("UI contractuelle — aucune logique métier implémentée ici.")
