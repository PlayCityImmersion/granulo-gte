# smaxia_console_v104_ui.py
# ═══════════════════════════════════════════════════════════════════════════════
# SMAXIA CONSOLE V10.4 — UI STREAMLIT
# Compatible avec le Kernel V10.4 scellé
# ═══════════════════════════════════════════════════════════════════════════════
# Panel: GPT 5.2 | GEMINI 3.0 | CLAUDE OPUS 4.5
# Date: 26 décembre 2025
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np

# Import du moteur V10.4
from smaxia_granulo_engine_v104 import run_granulo_v104, KernelConstants

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(
    page_title="SMAXIA - Console V10.4 (Kernel Scellé)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ SMAXIA - Console V10.4")
st.caption(f"Kernel Version: {KernelConstants.KERNEL_VERSION} | Date: {KernelConstants.KERNEL_DATE} | Panel Scellé")

# ==============================================================================
# CSS – UI CONTRACTUELLE (ENRICHIE V10.4)
# ==============================================================================
st.markdown("""
<style>
/* QC BOX */
.qc-box {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-left: 6px solid #2563eb;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.qc-chap {
    font-size: 0.75em;
    font-weight: 800;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.qc-title {
    font-size: 1.2em;
    font-weight: 900;
    color: #0f172a;
    margin-top: 6px;
}
.qc-meta {
    margin-top: 10px;
    font-family: 'Courier New', monospace;
    font-size: 0.8em;
    background: #1e293b;
    color: #22c55e;
    padding: 8px 12px;
    border-radius: 4px;
    display: inline-block;
}

/* KERNEL INFO */
.kernel-info {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 10px;
    font-size: 0.85em;
}

/* TRIGGERS */
.trigger {
    background: linear-gradient(90deg, #fff1f2 0%, #fecdd3 100%);
    border-left: 4px solid #ef4444;
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 6px;
    font-weight: 600;
    font-size: 0.9em;
}

/* ARI */
.ari-step {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    padding: 8px 10px;
    border-radius: 4px;
    margin-bottom: 5px;
    font-family: 'Courier New', monospace;
    font-size: 0.85em;
}
.ari-step:hover {
    background: #e2e8f0;
}

/* FRT */
.frt {
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    border-left: 6px solid;
}
.frt-usage { background: #fff7ed; border-color: #f59e0b; }
.frt-method { background: #f0fdf4; border-color: #22c55e; }
.frt-trap { background: #fef2f2; border-color: #ef4444; }
.frt-conc { background: #eff6ff; border-color: #3b82f6; }

.frt-title {
    font-weight: 900;
    font-size: 0.7em;
    text-transform: uppercase;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}

/* QI */
.file-box {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin-bottom: 10px;
    overflow: hidden;
}
.file-header {
    background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 100%);
    padding: 8px 12px;
    font-weight: 700;
    font-size: 0.85em;
}
.qi {
    padding: 8px 12px;
    border-left: 4px solid #8b5cf6;
    font-family: Georgia, serif;
    font-size: 0.9em;
    border-bottom: 1px solid #f1f5f9;
}

/* SATURATION */
.sat-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    padding: 16px;
    border-radius: 8px;
    margin-top: 20px;
}

/* COVERAGE */
.coverage-pass {
    background: #dcfce7;
    border: 2px solid #22c55e;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
}
.coverage-fail {
    background: #fef2f2;
    border: 2px solid #ef4444;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
}

/* AUDIT */
.audit-section {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
}
.audit-title {
    font-weight: 800;
    font-size: 0.9em;
    color: #475569;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR – PARAMÈTRES ACADÉMIQUES + KERNEL INFO
# ==============================================================================
with st.sidebar:
    st.header("🔧 Paramètres")
    
    # Kernel Info
    st.markdown(f"""
    <div class="kernel-info">
        <strong>Kernel:</strong> {KernelConstants.KERNEL_VERSION}<br>
        <strong>ε:</strong> {KernelConstants.EPSILON}<br>
        <strong>α:</strong> {KernelConstants.ALPHA_DEFAULT}<br>
        <strong>Coverage:</strong> {KernelConstants.ORPHAN_TOLERANCE_THRESHOLD*100}% tolérance
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Académique")
    classe = st.selectbox("Classe", ["Terminale"], disabled=True)
    matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chapitres = st.multiselect(
        "Chapitres",
        ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
        default=["SUITES NUMÉRIQUES"]
    )
    
    st.divider()
    st.subheader("Constantes Kernel")
    st.code(f"""
EPSILON = {KernelConstants.EPSILON}
ALPHA = {KernelConstants.ALPHA_DEFAULT}
COHERENCE = {KernelConstants.CLUSTER_COHERENCE_THRESHOLD_DEFAULT}
MAX_ITER = {KernelConstants.MAX_IA1_IA2_CORRECTION_ITERATIONS}
ORPHAN_TOL = {KernelConstants.ORPHAN_TOLERANCE_ABSOLUTE}
    """)

# ==============================================================================
# TABS
# ==============================================================================
tab_usine, tab_audit, tab_kernel = st.tabs(["🏭 Usine", "✅ Audit", "⚙️ Kernel"])

# ==============================================================================
# ONGLET 1 – USINE
# ==============================================================================
with tab_usine:
    
    # --------------------------------------------------------------------------
    # ZONE 1 – INJECTION DES SUJETS
    # --------------------------------------------------------------------------
    st.subheader("🧪 Injection des sujets")
    
    c1, c2 = st.columns([4, 1])
    with c1:
        urls = st.text_area(
            "URLs sources (références)",
            value="https://www.apmep.fr/Terminale-S-702-sujets-702-corriges",
            height=80,
            help="Entrez les URLs des sources d'annales (une par ligne)"
        )
    with c2:
        volume = st.number_input(
            "Volume de sujets",
            min_value=5,
            max_value=500,
            value=15,
            step=5
        )
        launch = st.button("🚀 LANCER L'USINE", type="primary", use_container_width=True)
    
    if launch:
        url_list = [u.strip() for u in urls.split("\n") if u.strip()]
        with st.spinner(f"Pipeline V10.4 : Harvesting → Atomisation → POSABLE → Clustering → F1/F2 → Coverage..."):
            result = run_granulo_v104(url_list, int(volume))
        st.session_state["granulo_result"] = result
        st.success(f"✅ Pipeline terminé en {result['audit']['elapsed_s']}s")
    
    # --------------------------------------------------------------------------
    # ZONE 2 – TABLEAU DES SUJETS TRAITÉS
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("📥 Sujets traités")
    
    if "granulo_result" not in st.session_state:
        st.info("⏳ Données affichées après exécution du moteur.")
        st.dataframe(pd.DataFrame(columns=["Fichier", "Nature", "Année", "Source", "Fingerprint"]), use_container_width=True)
    else:
        df_sujets = pd.DataFrame(st.session_state["granulo_result"]["sujets"])
        if df_sujets.empty:
            st.warning("Aucun sujet exploitable récupéré.")
        else:
            st.dataframe(df_sujets, use_container_width=True)
        
        # Métriques rapides
        audit = st.session_state["granulo_result"]["audit"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sujets", audit["n_subjects_ok"])
        col2.metric("Qi POSABLE", audit["n_qi_posable"])
        col3.metric("QC Générées", audit["n_qc_selected"])
        col4.metric("Coverage", f"{audit['coverage_ratio']*100:.1f}%")
    
    # --------------------------------------------------------------------------
    # ZONE 3 – BASE DE CONNAISSANCE (QC)
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("🧠 Base de connaissance (QC)")
    
    if "granulo_result" not in st.session_state:
        st.info("Aucune QC affichée tant que le moteur n'a pas produit de résultats.")
    else:
        qc_list = st.session_state["granulo_result"]["qc"]
        audit = st.session_state["granulo_result"]["audit"]
        
        # Statut couverture
        if audit.get("chapter_sealed"):
            st.markdown("""
            <div class="coverage-pass">
                ✅ CHAPITRE SCELLÉ — Coverage 100% (zéro orphelin significatif)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="coverage-fail">
                ⚠️ CHAPITRE NON SCELLÉ — {audit.get('n_orphans', 0)} orphelins détectés
            </div>
            """, unsafe_allow_html=True)
        
        if not qc_list:
            st.warning("0 QC produite.")
        else:
            # Sélecteur de QC
            qc_options = [f"{qc['qc_id']}: {qc['qc_title'][:50]}..." for qc in qc_list]
            selected_idx = st.selectbox("Sélectionner une QC", range(len(qc_options)), format_func=lambda i: qc_options[i])
            
            qc = qc_list[selected_idx]
            
            st.markdown(f"""
            <div class="qc-box">
                <div class="qc-chap">Chapitre : {qc['chapter']}</div>
                <div class="qc-title">{qc['qc_id']} : {qc['qc_title']}</div>
                <div class="qc-meta">
                    Score(q)={qc['score']} | n_q={qc['n_q']} | Ψ={qc['psi']} | N_tot={qc['n_tot']} | t_réc={qc['t_rec']}y | Σ_Tj={qc.get('sum_tj', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.markdown("### 🔥 Déclencheurs")
                if qc["triggers"]:
                    for t in qc["triggers"][:6]:
                        st.markdown(f"<div class='trigger'>{t}</div>", unsafe_allow_html=True)
                else:
                    st.caption("Aucun déclencheur.")
            
            with c2:
                st.markdown("### ⚙️ ARI")
                for i, s in enumerate(qc["ari"], 1):
                    st.markdown(f"<div class='ari-step'><strong>{i}.</strong> {s}</div>", unsafe_allow_html=True)
            
            with c3:
                st.markdown("### 📘 FRT")
                frt = qc["frt"]
                st.markdown(f"<div class='frt frt-usage'><div class='frt-title'>🎯 Quand utiliser</div>{frt['usage']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='frt frt-method'><div class='frt-title'>📝 Méthode</div>{frt['method']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='frt frt-trap'><div class='frt-title'>⚠️ Pièges</div>{frt['trap']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='frt frt-conc'><div class='frt-title'>✅ Conclusion</div>{frt['conc']}</div>", unsafe_allow_html=True)
            
            with c4:
                st.markdown("### 📄 Qi associées")
                qi_map = qc["qi_by_file"]
                if not qi_map:
                    st.caption("Aucune Qi mappée.")
                else:
                    for f, qs in list(qi_map.items())[:3]:
                        html = f"<div class='file-box'><div class='file-header'>📁 {f}</div>"
                        for q in qs[:5]:
                            q_short = q[:100] + "..." if len(q) > 100 else q
                            html += f"<div class='qi'>{q_short}</div>"
                        if len(qs) > 5:
                            html += f"<div class='qi' style='font-style:italic'>… +{len(qs)-5} autres</div>"
                        html += "</div>"
                        st.markdown(html, unsafe_allow_html=True)
    
    # --------------------------------------------------------------------------
    # ZONE 4 – COURBE DE SATURATION
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("📈 Analyse de saturation")
    
    if "granulo_result" not in st.session_state:
        st.info("Courbe disponible après exécution.")
    else:
        sat = st.session_state["granulo_result"]["saturation"]
        df_sat = pd.DataFrame(sat)
        
        if df_sat.empty:
            st.warning("Aucune donnée de saturation.")
        else:
            st.line_chart(df_sat, x="Nombre de sujets injectés", y="Nombre de QC découvertes")
            
            # Analyse saturation
            if len(df_sat) >= 5:
                tail = df_sat["Nombre de QC découvertes"].tail(5).tolist()
                if len(set(tail)) == 1:
                    st.success("🎯 Seuil de saturation atteint : derniers sujets ⇒ 0 nouvelle QC")
                else:
                    delta = tail[-1] - tail[0]
                    st.info(f"📊 Croissance en cours : +{delta} QC sur les 5 derniers sujets")

# ==============================================================================
# ONGLET 2 – AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("🔍 Audit du moteur Granulo V10.4")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="audit-section">
            <div class="audit-title">📋 Checks IA2 (Kernel V10.4)</div>
            <ul>
                <li>CHK_POSABLE_VALID</li>
                <li>CHK_QC_FORM</li>
                <li>CHK_NO_LOCAL_CONSTANTS</li>
                <li>CHK_FRT_TEMPLATE_OK</li>
                <li>CHK_TRIGGERS_QUALITY</li>
                <li>CHK_ARI_TYPED_ONLY</li>
                <li>CHK_F1_RECALCULABLE</li>
                <li>CHK_F2_TERMS_VISIBLE</li>
                <li>CHK_COVERAGE_BOOL_ZERO_ORPHAN</li>
                <li>CHK_DETERMINISM_LOCK</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="audit-section">
            <div class="audit-title">🔒 Corrections V10.4 Intégrées</div>
            <ul>
                <li>✅ C1: Seuil cohérence cluster = 0.70</li>
                <li>✅ C2: Max itérations IA1↔IA2 = 3</li>
                <li>✅ C3: Tie-break déterministe</li>
                <li>✅ C4: t_rec en années</li>
                <li>✅ C5: Fingerprint SHA256</li>
                <li>✅ C6: Singletons irréductibles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if "granulo_result" in st.session_state:
        st.divider()
        
        result = st.session_state["granulo_result"]
        audit = result["audit"]
        
        # Métriques détaillées
        st.subheader("📊 Métriques d'exécution")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("URLs", audit["n_urls"])
        col2.metric("PDFs trouvés", audit["n_pdf_links"])
        col3.metric("Sujets OK", audit["n_subjects_ok"])
        col4.metric("Qi POSABLE", audit["n_qi_posable"])
        col5.metric("Clusters", audit["n_clusters"])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("QC Candidates", audit["n_qc_candidates"])
        col2.metric("QC Sélectionnées", audit["n_qc_selected"])
        col3.metric("Orphelins", audit["n_orphans"])
        col4.metric("Orphan Cap", audit["orphan_cap"])
        col5.metric("Temps (s)", audit["elapsed_s"])
        
        # Statut scellement
        if audit["chapter_sealed"]:
            st.success(f"✅ CHK_COVERAGE_BOOL_ZERO_ORPHAN: PASS — Chapitre SCELLÉ")
        else:
            st.error(f"❌ CHK_COVERAGE_BOOL_ZERO_ORPHAN: FAIL — {audit['n_orphans']} orphelins")
        
        # Audit Log
        st.subheader("📝 Audit Log")
        if audit.get("audit_log"):
            for msg in audit["audit_log"]:
                if "PASS" in msg:
                    st.success(msg)
                elif "FAIL" in msg or "BLOCAGE" in msg:
                    st.error(msg)
                else:
                    st.info(msg)
        else:
            st.caption("Aucun message d'audit.")
        
        # JSON complet
        with st.expander("🔧 Données brutes (JSON)"):
            st.json(audit)

# ==============================================================================
# ONGLET 3 – KERNEL
# ==============================================================================
with tab_kernel:
    st.subheader("⚙️ Kernel SMAXIA V10.4 — Spécifications")
    
    if "granulo_result" in st.session_state:
        kernel_info = st.session_state["granulo_result"]["kernel_info"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📐 Formule F1 (Ψ_q)")
            st.latex(r"\Psi_q = \frac{\delta_c \times (\epsilon + \sum_j T_j)^2}{\max_{p \in Q_c}(\Psi_p)}")
            
            st.markdown("### 📐 Formule F2 (Score)")
            st.latex(r"Score(q) = \frac{n_q}{N_{total}} \times \left(1 + \frac{\alpha}{t_{rec}}\right) \times \Psi_q \times \prod_{p<q}(1-\sigma) \times 100")
        
        with col2:
            st.markdown("### 🔧 Constantes Scellées")
            st.json(kernel_info["constants"])
            
            st.markdown("### 📋 Métadonnées")
            st.json({
                "version": kernel_info["version"],
                "date": kernel_info["date"],
                "fingerprint_algo": kernel_info["fingerprint_algo"]
            })
    else:
        st.info("Exécutez le moteur pour voir les détails du Kernel.")
    
    st.divider()
    st.markdown("### 📖 Table T_j — Verbes Cognitifs (Bloom)")
    
    tj_data = {
        "Verbe": ["Identifier", "Définir", "Analyser", "Contextualiser", "Simplifier", 
                  "Calculer", "Exprimer", "Comparer", "Appliquer", "Résoudre",
                  "Dériver", "Argumenter", "Démontrer", "Récurrence", "Synthétiser"],
        "T_j": [0.15, 0.15, 0.20, 0.25, 0.25, 0.30, 0.30, 0.35, 0.35, 0.40, 
                0.40, 0.45, 0.50, 0.60, 0.20],
        "Niveau": ["Connaissance", "Connaissance", "Compréhension", "Application", "Application",
                   "Analyse", "Analyse", "Synthèse", "Synthèse", "Évaluation",
                   "Évaluation", "Argumentation", "Démonstration", "Récurrence", "Synthèse"]
    }
    st.dataframe(pd.DataFrame(tj_data), use_container_width=True)
