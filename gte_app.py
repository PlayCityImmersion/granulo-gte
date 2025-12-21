import streamlit as st
import pandas as pd
import pdfplumber
import re
from granulo_engine import GranuloEngine

st.set_page_config(page_title="SMAXIA GTE-V2.2", layout="wide")

st.title("🛡️ SMAXIA GRANULO TEST ENGINE (V2.2)")
st.caption("BANC DE TEST INDUSTRIEL | SPLITTER UNIVERSEL | MAPPING ARBORESCENT")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ INGESTION")
    uploaded_files = st.file_uploader("Sources (PDF/TXT)", type=['pdf', 'txt'], accept_multiple_files=True)
    st.info("Nouveau : Découpage intelligent (Exercice, Problème, Partie...)")

# =============================================================================
# FONCTION DE DÉCOUPAGE AVANCÉE (SMART SPLITTER)
# =============================================================================
def smart_split_pdf(text):
    """
    Découpe le texte brut en Qi atomiques basées sur les structures d'examens.
    Reconnait : Exercice, Ex, Problème, Problem, Partie, Q1, 1.
    """
    # Regex pour trouver les séparateurs d'exercices (Universel FR/EN)
    pattern = r"(?:\n|^)\s*(?:Exercice|Exercise|Ex|Problème|Problem|Partie|Part)\s*\d*"
    
    # 1. Découpage primaire (Gros blocs)
    segments = re.split(pattern, text, flags=re.IGNORECASE)
    
    # 2. Nettoyage et Filtrage (On garde ce qui ressemble à une question)
    valid_qi = []
    for seg in segments:
        clean_seg = seg.strip()
        # On garde si c'est assez long (> 50 caractères) pour contenir de la sémantique
        if len(clean_seg) > 50: 
            valid_qi.append(clean_seg)
            
    return valid_qi

# =============================================================================
# COEUR DU TEST
# =============================================================================
if uploaded_files:
    engine = GranuloEngine()
    
    with st.spinner('⚡ ANALYSE INTELLIGENTE (Smart Split + Invariance)...'):
        # INGESTION
        total_pages = 0
        for file in uploaded_files:
            text = ""
            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages: 
                        text += page.extract_text() or ""
                        total_pages += 1
            else:
                text = file.read().decode("utf-8")
            
            # UTILISATION DU SMART SPLITTER V2.2
            atoms = smart_split_pdf(text)
            for atom in atoms:
                engine.ingest_qi(atom, source_type=file.name)

        # RÉDUCTION
        qc_results = engine.run_reduction_process()
        audit = engine.check_coverage(qc_results)

    # =========================================================================
    # DASHBOARD
    # =========================================================================
    
    # KPI
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("PDFs / Pages", f"{len(uploaded_files)} / {total_pages}")
    kpi2.metric("Qi Extraites (Smart)", audit['total_qi']) # Doit être >> 13
    kpi3.metric("Clusters QC", len(qc_results))
    kpi4.metric("Couverture", f"{audit['rate']:.1f}%")

    if audit['is_valid']:
        st.success("✅ VERDICT : PASS. La Loi de Réduction est respectée.")
    else:
        st.error("❌ VERDICT : REJECT. Taux de couverture insuffisant.")

    st.divider()

    # =========================================================================
    # VUE ARBORESCENCE (MAPPING PREUVE)
    # =========================================================================
    st.header("🧬 ARBORESCENCE DU MAPPING (QC Mère ➔ Qi Filles)")
    st.markdown("Dépliez une QC pour voir les questions réelles qu'elle a capturées.")

    for qc in qc_results:
        # Titre de l'expander : ID + SIGNATURE + NB CAPTURÉS
        label = f"📍 {qc.id} : {qc.canonical_text} ({len(qc.covered_qi_list)} Qi capturées)"
        
        with st.expander(label, expanded=False):
            # En-tête de la QC
            c1, c2, c3 = st.columns([1, 2, 1])
            c1.markdown(f"**Signature V:** `{qc.signature.verb}`")
            c2.markdown(f"**Signature O:** `{qc.signature.obj}`")
            c3.markdown(f"**Psi Score:** `{qc.psi_score}`")
            
            st.markdown("---")
            st.markdown("**🔽 Qi RÉELLES (SOURCES PDF) :**")
            
            # Liste des Qi filles
            if len(qc.covered_qi_list) > 0:
                for i, qi_text in enumerate(qc.covered_qi_list):
                    st.text_area(f"Qi #{i+1} associée", value=qi_text, height=100, disabled=True)
            else:
                st.caption("⚠️ Aucune Qi capturée par ce cluster (potentiel QC théorique ou Black Swan vide).")

else:
    st.info("👈 Chargez vos sujets pour voir l'arborescence du mapping.")
