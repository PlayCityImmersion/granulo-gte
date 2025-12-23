# smaxia_console_v31_ui.py
# SMAXIA GRANULO CONSOLE v3.1 — STREAMLIT

import streamlit as st
from smaxia_granulo_engine_test import run_granulo_test

st.set_page_config(page_title="SMAXIA Granulo GTE", layout="wide")

st.title("🧠 SMAXIA — Granulo Test Engine")
st.caption("Extraction réelle → Qi → QC → FRT (preuves uniquement)")

if st.button("🚀 Lancer le moteur Granulo"):
    with st.spinner("Extraction des PDFs et calcul en cours..."):
        results = run_granulo_test()

    if not results:
        st.error("❌ Aucune QC générée — vérifier les sources")
    else:
        st.success(f"✅ QC générées : {len(results)}")

        for i, r in enumerate(results[:5], 1):
            with st.expander(f"QC {i}"):
                for j, qi in enumerate(r["qc"], 1):
                    st.write(f"**Qi {j}** : {qi}")

                st.markdown("### FRT")
                st.json(r["frt"])
