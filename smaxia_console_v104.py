# smaxia_console_v104.py
# =============================================================================
# SMAXIA CONSOLE V10.4 — UI STREAMLIT
# Compatible avec le Kernel V10.4 scellé (GPT 5.2 corrigé)
# =============================================================================
# Panel: GPT 5.2 | GEMINI 3.0 | CLAUDE OPUS 4.5
# Date: 27 décembre 2025
# =============================================================================

import streamlit as st
import pandas as pd
import io

# Import du moteur V10.4 corrigé
from smaxia_granulo_engine_v104 import (
    run_granulo_v104,
    KernelConstants,
    create_pack_france_terminale_maths
)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="SMAXIA - Console V10.4 (Kernel Scellé)",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .chapter-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .qc-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin-bottom: 1rem;
    }
    .qc-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E3A5F;
    }
    .ari-step {
        background: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-family: monospace;
    }
    .trigger-tag {
        display: inline-block;
        background: #e1e8ed;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
    .gte-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sealed-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
    .not-sealed-badge {
        background: #dc3545;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<p class="main-header">🧠 SMAXIA - Console V10.4</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Kernel Version: {KernelConstants.KERNEL_VERSION} | Status: {KernelConstants.KERNEL_STATUS} | Date: {KernelConstants.KERNEL_DATE}</p>', unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("📁 Sources")
    
    subject_file = st.file_uploader("📄 Sujet (PDF)", type=["pdf"], key="subject")
    correction_file = st.file_uploader("📝 Corrigé (PDF, optionnel)", type=["pdf"], key="correction")
    
    st.divider()
    
    st.header("⚙️ Paramètres")
    year_ref = st.number_input("Année du sujet", min_value=2000, max_value=2030, value=2023)
    
    gte_mode = st.checkbox(
        "Mode GTE (prévisualisation sans corrigé)", 
        value=True,
        help="Génère des QC pour prévisualisation UI. Ces QC ne sont PAS scellées."
    )
    
    if gte_mode:
        st.warning("⚠️ Mode GTE: QC non scellables")
    
    st.divider()
    
    st.header("🔍 Affichage")
    show_qi_list = st.checkbox("Liste des Qi", value=False)
    show_posable = st.checkbox("Décisions POSABLE", value=False)
    show_audit = st.checkbox("Audit Log", value=True)
    show_coverage = st.checkbox("Coverage détaillée", value=False)
    
    st.divider()
    
    st.header("📊 Pack actif")
    pack = create_pack_france_terminale_maths()
    st.info(f"**{pack.pack_id}**\n\nv{pack.pack_version}\n\n{len(pack.chapters)} chapitres")


# =============================================================================
# EXTRACTION PDF
# =============================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    if not pdfplumber:
        st.error("pdfplumber non installé")
        return ""
    
    try:
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        uploaded_file.seek(0)
        return text
    except Exception as e:
        st.error(f"Erreur extraction PDF: {e}")
        return ""


# =============================================================================
# MAIN
# =============================================================================

if subject_file:
    # Extraction
    with st.spinner("📖 Extraction du sujet..."):
        subject_text = extract_text_from_pdf(subject_file)
    
    correction_text = None
    if correction_file:
        with st.spinner("📖 Extraction du corrigé..."):
            correction_text = extract_text_from_pdf(correction_file)
    
    if not subject_text:
        st.error("❌ Impossible d'extraire le texte du sujet")
        st.stop()
    
    st.success(f"✅ Sujet chargé: {len(subject_text):,} caractères")
    if correction_text:
        st.success(f"✅ Corrigé chargé: {len(correction_text):,} caractères")
    
    # Traitement
    with st.spinner("⚙️ Traitement par le moteur V10.4..."):
        results = run_granulo_v104(
            subject_text=subject_text,
            correction_text=correction_text,
            source_id=subject_file.name,
            year_ref=year_ref,
            extracted_at="2025-12-27T00:00:00Z",
            gte_mode=gte_mode,
            pack=pack
        )
    
    # ==========================================================================
    # MÉTRIQUES
    # ==========================================================================
    
    st.header("📊 Métriques")
    
    metrics = results["metrics"]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['total_qi']}</div>
            <div class="metric-label">Qi extraites</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)" if metrics['total_posable'] > 0 else "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
        st.markdown(f"""
        <div class="metric-card" style="background: {color};">
            <div class="metric-value">{metrics['total_posable']}</div>
            <div class="metric-label">POSABLES</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);">
            <div class="metric-value">{metrics['total_qc']}</div>
            <div class="metric-label">QC Sélectionnées</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);">
            <div class="metric-value">{metrics.get('total_gte_preview', 0)}</div>
            <div class="metric-label">QC Preview (GTE)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        coverage_pct = metrics['coverage'] * 100
        color = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)" if coverage_pct >= 95 else "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
        st.markdown(f"""
        <div class="metric-card" style="background: {color};">
            <div class="metric-value">{coverage_pct:.0f}%</div>
            <div class="metric-label">Coverage</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==========================================================================
    # WARNING GTE MODE
    # ==========================================================================
    
    if gte_mode and metrics['total_posable'] == 0:
        st.markdown("""
        <div class="gte-warning">
            <strong>⚠️ Mode GTE actif sans corrigé</strong><br>
            Les QC affichées ci-dessous sont des <em>prévisualisations</em> (gte_qc_preview).<br>
            Elles ne sont <strong>PAS scellables</strong> selon le Kernel V10.4.<br>
            Pour sceller, fournissez un corrigé apparié.
        </div>
        """, unsafe_allow_html=True)
    
    # ==========================================================================
    # CHAPITRES
    # ==========================================================================
    
    st.header("📂 Chapitres détectés")
    
    if results["chapters_detected"]:
        colors = {
            "SUITES NUMÉRIQUES": "#3498db",
            "FONCTIONS": "#2ecc71",
            "INTÉGRALES": "#9b59b6",
            "PROBABILITÉS": "#e74c3c",
            "GÉOMÉTRIE DANS L'ESPACE": "#f39c12",
            "NOMBRES COMPLEXES": "#1abc9c",
            "MATRICES": "#34495e",
            "ARITHMÉTIQUE": "#e67e22"
        }
        
        badges_html = ""
        for chapter, count in sorted(results["chapters_detected"].items(), key=lambda x: x[1], reverse=True):
            color = colors.get(chapter, "#95a5a6")
            sealed = results["sealed_by_chapter"].get(chapter, False)
            seal_badge = '<span class="sealed-badge">SEALED</span>' if sealed else '<span class="not-sealed-badge">NOT SEALED</span>'
            badges_html += f'<span class="chapter-badge" style="background: {color}; color: white;">{chapter}: {count} Qi {seal_badge}</span>'
        
        st.markdown(badges_html, unsafe_allow_html=True)
        
        # Graphique
        df_chapters = pd.DataFrame([
            {"Chapitre": ch, "Qi": count}
            for ch, count in results["chapters_detected"].items()
        ])
        st.bar_chart(df_chapters.set_index("Chapitre"))
    else:
        st.warning("Aucun chapitre détecté")
    
    # ==========================================================================
    # QC SÉLECTIONNÉES (KERNEL)
    # ==========================================================================
    
    if results["selected_qcs"]:
        st.header("📌 Questions Canoniques Sélectionnées (Kernel)")
        
        qc_by_chapter = {}
        for qc in results["selected_qcs"]:
            if qc.chapter_ref not in qc_by_chapter:
                qc_by_chapter[qc.chapter_ref] = []
            qc_by_chapter[qc.chapter_ref].append(qc)
        
        for chapter in sorted(qc_by_chapter.keys()):
            sealed = results["sealed_by_chapter"].get(chapter, False)
            seal_icon = "✅" if sealed else "❌"
            with st.expander(f"{seal_icon} {chapter} ({len(qc_by_chapter[chapter])} QC)", expanded=True):
                for qc in sorted(qc_by_chapter[chapter], key=lambda x: x.n_q, reverse=True):
                    st.markdown(f"""
                    <div class="qc-card">
                        <div class="qc-title">{qc.qc_id}</div>
                        <div>{qc.title}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("n_q", qc.n_q)
                    with col2:
                        st.metric("Score", f"{qc.score:.1f}")
                    with col3:
                        status = "✅ EXTRACTED" if qc.frt.source == "EXTRACTED" else "⚠️ RECONSTRUCTED"
                        st.write(f"**ARI:** {status}")
                    
                    # ARI
                    st.write("**Algorithme de Résolution (ARI):**")
                    for step in qc.ari[:5]:
                        st.markdown(f'<div class="ari-step">{step}</div>', unsafe_allow_html=True)
                    
                    # Triggers
                    triggers_html = "**Triggers:** "
                    for t in qc.triggers[:6]:
                        triggers_html += f'<span class="trigger-tag">{t}</span>'
                    st.markdown(triggers_html, unsafe_allow_html=True)
                    
                    st.divider()
    
    # ==========================================================================
    # QC PREVIEW (GTE)
    # ==========================================================================
    
    if results.get("gte_qc_preview"):
        st.header("👁️ QC Preview (GTE - Non Scellables)")
        
        st.warning("Ces QC sont générées pour prévisualisation uniquement. Elles ne font PAS partie de la sélection Kernel.")
        
        for chapter, qcs in results["gte_qc_preview"].items():
            with st.expander(f"🔍 {chapter} ({len(qcs)} QC preview)", expanded=True):
                for qc in sorted(qcs, key=lambda x: x.n_q, reverse=True):
                    st.markdown(f"""
                    <div class="qc-card" style="border-left-color: #ffc107;">
                        <div class="qc-title">{qc.qc_id} <span style="color: #ffc107;">[GTE]</span></div>
                        <div>{qc.title}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("n_q", qc.n_q)
                    with col2:
                        st.metric("Score", f"{qc.score:.1f}")
                    with col3:
                        st.write(f"**ARI:** ⚠️ {qc.frt.source}")
                    
                    # ARI
                    for step in qc.ari[:3]:
                        st.markdown(f'<div class="ari-step">{step}</div>', unsafe_allow_html=True)
                    
                    st.divider()
    
    # ==========================================================================
    # SECTIONS OPTIONNELLES
    # ==========================================================================
    
    if show_audit:
        st.header("📋 Audit Log")
        for log in results["audit_log"]:
            if "PASS" in log:
                st.success(log)
            elif "FAIL" in log or "BLOCKED" in log:
                st.error(log)
            else:
                st.info(log)
    
    if show_qi_list:
        st.header("📜 Liste des Qi extraites")
        df_qi = pd.DataFrame([
            {
                "ID": a.qi_id,
                "Locator": a.qi_evidence.locator.to_ref(),
                "Verbe": a.verbe_action or "-",
                "Chapitre": a.chapter_detected or "-",
                "Appariée": "✅" if a.rqi_id else "❌",
                "Texte": a.qi_clean[:80] + "..."
            }
            for a in results["atoms"]
        ])
        st.dataframe(df_qi, use_container_width=True)
    
    if show_posable:
        st.header("🔒 Décisions POSABLE")
        
        posable_count = sum(1 for d in results["posable_decisions"] if d.posable)
        non_posable_count = len(results["posable_decisions"]) - posable_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ POSABLE: {posable_count}")
        with col2:
            st.error(f"❌ NON-POSABLE: {non_posable_count}")
        
        # Raisons
        reason_counts = {}
        for d in results["posable_decisions"]:
            for rc in d.reason_codes:
                reason_counts[rc.value] = reason_counts.get(rc.value, 0) + 1
        
        if reason_counts:
            st.write("**Raisons de non-posabilité:**")
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"• {reason}: {count}")
    
    if show_coverage:
        st.header("📈 Coverage détaillée")
        
        for chapter, cov_map in results["coverage_maps"].items():
            sealed = results["sealed_by_chapter"].get(chapter, False)
            seal_icon = "✅" if sealed else "❌"
            with st.expander(f"{seal_icon} {chapter} — {cov_map.coverage_ratio:.0%}"):
                st.metric("Qi couvertes", f"{cov_map.n_covered}/{cov_map.n_total_target}")
                
                if cov_map.orphans:
                    st.warning(f"⚠️ {len(cov_map.orphans)} orphelins")
                    for orphan in list(cov_map.orphans)[:5]:
                        st.write(f"• {orphan}")

else:
    # Page d'accueil
    st.info("👆 Uploadez un sujet PDF pour commencer l'analyse")
    
    st.markdown("""
    ### 🔧 Conformité V10.4 (GPT 5.2 corrigé)
    
    | Règle Kernel | Implémentation | Status |
    |--------------|----------------|--------|
    | POSABLE = corrigé ∧ scope ∧ évaluable | PosableGate.evaluate() | ✅ |
    | Attach = AND strict (4 critères) | AttachOperator.attach() | ✅ |
    | Sélection coverage-driven | CoverageEngine.select_coverage_driven() | ✅ |
    | Déterminisme (list vs set) | Patch 1 appliqué | ✅ |
    | F2 anti-redondance dynamique | Patch 2 appliqué | ✅ |
    | GTE ≠ selected_qcs | Patch 3 appliqué | ✅ |
    
    ### 📦 Fichiers
    
    - `smaxia_granulo_engine_v104.py` — Moteur V10.4 corrigé
    - `smaxia_console_v104.py` — Cette interface
    
    ### 🚀 Mode GTE
    
    Le mode GTE permet de prévisualiser les QC sans corrigé.  
    Ces QC sont stockées dans `gte_qc_preview`, **séparées** de `selected_qcs`.  
    Elles ne sont **jamais scellables** sans corrigé.
    """)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(f"SMAXIA Kernel {KernelConstants.KERNEL_VERSION} ({KernelConstants.KERNEL_STATUS}) | GPT 5.2 Corrigé | 27 décembre 2025")
