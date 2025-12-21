import streamlit as st
import pandas as pd
import pdfplumber
import re
import os
from granulo_engine import GranuloEngine

st.set_page_config(page_title="SMAXIA GTE-V4.0", layout="wide")

st.title("🛡️ SMAXIA GRANULO TEST ENGINE (V4.0)")
st.caption("ARCHITECTURE SCELLABLE | MOTEUR INVARIANT + INJECTION P3 | VÉRIFICATION STRICTE")

# VERIFICATION DE L'ENVIRONNEMENT
P3_LIBRARY_FILE = "smaxia_p3_db_fr.json"

if not os.path.exists(P3_LIBRARY_FILE):
    st.error(f"🚨 ERREUR CRITIQUE : La bibliothèque d'invariants '{P3_LIBRARY_FILE}' est introuvable. Le moteur ne peut pas démarrer.")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.header("⚙️ INGESTION & CONFIG")
    st.success(f"Bibliothèque chargée : {P3_LIBRARY_FILE}")
    uploaded_files = st.file_uploader("Sources (PDF/TXT)", type=['pdf', 'txt'], accept_multiple_files=True)

# SPLITTER UNIVERSEL
def smart_split_pdf(text):
    pattern = r"(?:\n|^)\s*(?:Exercice|Exercise|Ex|Problème|Problem|Partie|Part)\s*\d*"
    segments = re.split(pattern, text, flags=re.IGNORECASE)
    valid_qi = [seg.strip() for seg in segments if len(seg.strip()) > 50]
    return valid_qi

if uploaded_files:
    # INJECTION DE DÉPENDANCE : ON PASSE LE CHEMIN DU FICHIER AU MOTEUR
    engine = GranuloEngine(p3_library_path=P3_LIBRARY_FILE)
    
    with st.spinner('⚡ EXÉCUTION DU MOTEUR INVARIANT (LOAD + MATCH)...'):
        for file in uploaded_files:
            text = ""
            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages: text += page.extract_text() or ""
            else: text = file.read().decode("utf-8")
            
            atoms = smart_split_pdf(text)
            for atom in atoms:
                engine.ingest_qi(atom, source_type=file.name)

        qc_results, audit = engine.run_reduction_process()

    # DASHBOARD BOOLÉEN
    st.header("⚖️ VERDICT BOOLÉEN SMAXIA")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    def icon(v): return "✅ PASS" if v else "❌ FAIL"
    
    c1.metric("B1: Mapping Total", icon(audit.b1_all_mapped))
    c2.metric("B2: Structure QC", icon(audit.b2_qc_structure))
    c3.metric("B3: Triggers Valid", icon(audit.b3_triggers_valid))
    c4.metric("B4: Conservation", icon(audit.b4_conservation))
    c5.metric("B5: Black Swan", icon(audit.b5_black_swan))

    if audit.global_pass:
        st.success("🏆 ARCHITECTURE VALIDÉE : Le moteur est conforme à l'Axiome SMAXIA-INV-01.")
    else:
        st.error("⛔ ARCHITECTURE INVALIDÉE : Vérifiez vos invariants.")

    st.divider()

    # VUE ARBORESCENCE
    st.header("🧬 MAPPING P3 INJECTÉ")
    for qc in qc_results:
        count = len(qc.covered_qi_list)
        label = f"📍 {qc.id} : {qc.canonical_text} ({count} Qi)"
        if qc.is_black_swan: label = f"⚠️ {qc.id} : {qc.canonical_text} ({count} Qi)"
        
        with st.expander(label, expanded=qc.is_black_swan):
            st.info(f"**Concept:** {qc.operator_tag}")
            st.success(f"**Déclencheurs (P3):** {', '.join(list(set(qc.triggers_found))[:3])}")
            st.markdown("---")
            for i, txt in enumerate(qc.covered_qi_list):
                st.text_area(f"Qi #{i+1}", txt[:300] + "...", height=70, disabled=True)

else:
    st.info("👈 Chargez vos sujets pour tester l'architecture découplée.")
