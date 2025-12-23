# ============================================================
# SMAXIA – Console V31 (Saturation Proof)
# UI CONTRACTUELLE – SCELLÉE
# Aucune logique métier Granulo ici
# ============================================================

import streamlit as st
import pandas as pd

# 🔌 BRANCHEMENT MOTEUR (TEST)
from smaxia_granulo_engine_test import run_granulo_factory

# -----------------------------
# CONFIG GLOBALE
# -----------------------------
st.set_page_config(
    page_title="SMAXIA – Console V31",
    layout="wide",
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "subjects" not in st.session_state:
    st.session_state.subjects = []

if "qc" not in st.session_state:
    st.session_state.qc = []

# -----------------------------
# SIDEBAR – PARAMÈTRES ACADÉMIQUES
# -----------------------------
with st.sidebar:
    st.markdown("## 📘 Paramètres Académiques")

    classe = st.selectbox("Classe", ["Terminale"], index=0)

    matiere = st.selectbox(
        "Matière",
        ["MATHS", "PHYSIQUE"]
    )

    if matiere == "MATHS":
        chapitres = st.multiselect(
            "Chapitres",
            [
                "SUITES NUMÉRIQUES",
                "FONCTIONS",
                "PROBABILITÉS",
                "GÉOMÉTRIE"
            ]
        )
    else:
        chapitres = st.multiselect(
            "Chapitres",
            [
                "MÉCANIQUE",
                "ONDES",
                "ÉLECTRICITÉ",
                "CHIMIE"
            ]
        )

# -----------------------------
# HEADER
# -----------------------------
st.markdown("## 🛡️ SMAXIA – Console V31 (Saturation Proof)")
st.caption("UI contractuelle – aucune logique métier – moteur branché dynamiquement")

# -----------------------------
# ONGLET PRINCIPAL
# -----------------------------
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "🧪 Onglet 2 : Audit"])

# ============================================================
# ONGLET 1 – USINE
# ============================================================
with tab_usine:

    st.markdown("### 🔌 Injection des sujets")

    col1, col2 = st.columns([4, 1])

    with col1:
        urls_input = st.text_area(
            "URLs Sources (références)",
            value="https://apmep.fr"
        )

    with col2:
        volume = st.number_input(
            "Volume de sujets",
            min_value=1,
            max_value=200,
            value=15,
            step=1
        )

    # -----------------------------
    # LANCEMENT USINE
    # -----------------------------
    if st.button("🚀 LANCER L’USINE"):
        urls = [u.strip() for u in urls_input.split("\n") if u.strip()]

        with st.spinner("Injection et traitement des sujets…"):
            result = run_granulo_factory(
                urls=urls,
                volume=volume,
                classe=classe,
                matiere=matiere,
                chapitres=chapitres
            )

            st.session_state.subjects = result["subjects"]
            st.session_state.qc = result["qc"]

        st.success("Traitement terminé.")

    # -----------------------------
    # TABLEAU DES SUJETS
    # -----------------------------
    st.markdown("### 📥 Sujets traités")

    if st.session_state.subjects:
        df_subjects = pd.DataFrame([
            {
                "Fichier": s["id"],
                "Nature": s["nature"],
                "Année": s["year"],
                "Source": s["source"]
            }
            for s in st.session_state.subjects
        ])

        st.dataframe(df_subjects, use_container_width=True)
    else:
        st.info("Données affichées uniquement après branchement du moteur réel.")

    # -----------------------------
    # BASE DE CONNAISSANCE QC
    # -----------------------------
    if st.session_state.qc:

        st.markdown("## 🧠 Base de connaissance (QC)")

        for qc in st.session_state.qc:

            st.markdown(
                f"""
                ### Chapitre : {", ".join(chapitres) if chapitres else "—"}
                **{qc['qc_id']} : QC générée**
                """
            )

            st.markdown(
                f"""
                **Score(q)** = {qc['score']} |
                **n_q** = {qc['n_q']} |
                **Ψ** = {qc['psi']} |
                **N_tot** = {qc['N_tot']} |
                **t_réc** = {qc['t_rec']}
                """
            )

            colA, colB, colC, colD = st.columns(4)

            with colA:
                st.markdown("🔥 **Déclencheurs**")
                for qi in qc["qi"]:
                    st.write("•", qi["text"])

            with colB:
                st.markdown("⚙️ **ARI**")
                for step in qc["ari"]:
                    st.write("•", step["step"])

            with colC:
                st.markdown("📘 **FRT**")
                st.info("Affichage FRT – moteur en cours de validation")

            with colD:
                st.markdown("📄 **Qi associées**")
                for qi in qc["qi"]:
                    st.write(qi["qi_id"])

            st.divider()

    # -----------------------------
    # COURBE DE SATURATION (PLACEHOLDER)
    # -----------------------------
    st.markdown("### 📈 Courbe de saturation")
    st.warning("Courbe activée lorsque le moteur de saturation sera branché.")

# ============================================================
# ONGLET 2 – AUDIT
# ============================================================
with tab_audit:

    st.markdown("## 🧪 Audit du moteur Granulo")

    st.success("Audit interne : chaque Qi doit mapper vers UNE et UNE SEULE QC (objectif 100 %)")

    st.info("Audit externe : import d’un sujet inconnu → calcul du taux de couverture (≥ 95 %)")

    st.warning("Audit actif après stabilisation complète du moteur Granulo.")
