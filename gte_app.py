import streamlit as st
import pandas as pd
import pdfplumber
from granulo_engine import GranuloEngine

# =============================================================================
# CONFIGURATION DE L'INTERFACE
# =============================================================================
st.set_page_config(page_title="SMAXIA GTE-T1", layout="wide")

st.title("🛡️ SMAXIA GRANULO TEST ENGINE (GTE-T1)")
st.markdown("""
**Statut :** BANC DE TEST INDUSTRIEL (SANDBOX)  
**Objectif :** Valider la Loi de Réduction Axiomatique (15 QC / Chapitre)  
[cite_start]**Critère Booléen :** Couverture ≥ 95% des Qi injectées [cite: 575]
""")

# =============================================================================
# SIDEBAR : CONFIGURATION
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_files = st.file_uploader(
        "Injecter Sujets (PDF/TXT)", 
        type=['pdf', 'txt'], 
        accept_multiple_files=True
    )
    matiere = st.selectbox("Matière Cible", ["Mathématiques", "Physique", "Chimie"])
    st.info("Le moteur appliquera les formules F1 à F7 et l'Axiome Delta.")

# =============================================================================
# COEUR DU TEST
# =============================================================================

if uploaded_files:
    engine = GranuloEngine()
    
    with st.spinner('🚀 INITIALISATION DU MOTEUR GRANULO...'):
        # 1. INGESTION & HACHAGE ATOMIQUE
        all_text_debug = []
        for file in uploaded_files:
            text = ""
            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            else:
                text = file.read().decode("utf-8")
            
            # Simulation : Découpage par question (simplifié pour le test)
            # Dans la réalité, c'est fait par P5 Harvester
            segments = text.split("Exercice")
            for seg in segments:
                if len(seg) > 20:
                    engine.ingest_qi(seg, source_type="UPLOAD")
    
        # 2. EXÉCUTION DE LA RÉDUCTION
        qc_results = engine.run_reduction_process()
        
        # 3. CALCUL COUVERTURE
        audit = engine.check_coverage(qc_results)

    # =========================================================================
    # RÉSULTATS & AUDIT VISUEL
    # =========================================================================
    
    # KPI GLOBAL
    col1, col2, col3 = st.columns(3)
    col1.metric("Qi Injectées", audit['total_qi'])
    col2.metric("QC Invariantes", len(qc_results))
    col3.metric("Taux Couverture", f"{audit['rate']:.1f}%")

    # VERDICT BOOLEEN
    if audit['is_valid']:
        st.success("✅ VERDICT : PASS (SCELLABLE). La Loi de Réduction est respectée.")
    else:
        st.error("❌ VERDICT : REJECT (FAIL). Trous dans la raquette détectés.")

    st.divider()

    # VISUALISATION DES INVARIANTS
    st.subheader("🧬 TABLE DES INVARIANTS (QC)")
    
    data = [qc.to_dict() for qc in qc_results]
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # DÉTAIL D'UNE QC (INSPECTION F1/ARI)
    st.subheader("🔍 INSPECTION ATOMIQUE")
    selected_qc_id = st.selectbox("Choisir une QC pour inspection F1/ARI", [qc.id for qc in qc_results])
    
    if selected_qc_id:
        target = next(qc for qc in qc_results if qc.id == selected_qc_id)
        st.json(target.to_dict())
        st.markdown(f"**Texte Canonique :** {target.canonical_text}")
        if target.is_black_swan:
            st.warning("⚠️ Ceci est la QC #15 (Transposition). Elle capture les questions hors-cluster.")

else:
    st.write("👈 Veuillez charger des sujets dans la barre latérale pour lancer le test.")

# Signature de validation
st.divider()
st.caption("SMAXIA GTE-T1 | Validé par Panel Élite | Code Scellé A2/A3")
