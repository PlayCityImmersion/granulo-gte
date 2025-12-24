# smaxia_console_v31.py
# =============================================================================
# SMAXIA - Console V31 (Saturation Proof) - VERSION 3.1
# =============================================================================

import streamlit as st
import pandas as pd
from collections import defaultdict

# Import du moteur RÉEL
from smaxia_granulo_engine_real import (
    ingest_real,
    compute_qc_real,
    compute_saturation_real,
    audit_internal_real,
    audit_external_real,
    VERSION
)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V31")

# Afficher la version pour debug
st.sidebar.markdown(f"**Version Moteur:** `{VERSION}`")

st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# 🎨 STYLES CSS
# ==============================================================================
st.markdown("""
<style>
    .qc-header-box {
        background-color: #f8f9fa; border-left: 6px solid #2563eb; 
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
    }
    .qc-id-text { color: #d97706; font-weight: 900; font-size: 1.2em; margin-right: 10px; }
    .qc-title-text { color: #1f2937; font-weight: 700; font-size: 1.15em; }
    .qc-meta-text { 
        font-family: monospace; font-size: 0.85em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 4px 8px; border-radius: 4px; margin-top: 5px; display: inline-block;
    }
    .trigger-item {
        background-color: #fff1f2; color: #991b1b; padding: 5px 10px; margin-bottom: 4px; 
        border-radius: 4px; border-left: 4px solid #f87171; font-weight: 600; display: block;
    }
    .ari-step {
        background-color: #f3f4f6; color: #374151; padding: 4px 8px; margin-bottom: 3px; 
        border-radius: 3px; font-family: monospace; border: 1px dashed #d1d5db; display: block;
    }
    .frt-block { padding: 12px; border-bottom: 1px solid #e2e8f0; background: white; margin-bottom: 5px; border-radius: 4px; }
    .frt-title { font-weight: 800; text-transform: uppercase; font-size: 0.75em; display: block; margin-bottom: 6px; }
    .frt-content { font-size: 0.95em; color: #334155; line-height: 1.6; white-space: pre-wrap; }
    .c-usage { color: #d97706; border-left: 4px solid #d97706; }
    .c-method { color: #059669; border-left: 4px solid #059669; background-color: #f0fdf4; }
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; }
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; }
    .file-block { margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
    .file-header { background-color: #f1f5f9; padding: 8px 12px; font-weight: 700; font-size: 0.85em; color: #475569; }
    .qi-item { background-color: white; padding: 10px 12px; border-bottom: 1px solid #f8fafc; font-size: 0.95em; border-left: 3px solid #9333ea; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps = LISTE_CHAPITRES.get(sel_matiere, [])
    sel_chapitres = st.multiselect("Chapitres", chaps, default=chaps[:1] if chaps else [])

# ==============================================================================
# ONGLETS
# ==============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# ==============================================================================
# ONGLET 1 - USINE
# ==============================================================================
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1:
        urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2:
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        url_list = [u.strip() for u in urls.split('\n') if u.strip()]
        chapter_filter = sel_chapitres[0] if sel_chapitres else None
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.info("🔍 Collecte des sujets en cours...")
            
            df_src, df_atoms = ingest_real(
                url_list, 
                vol, 
                sel_matiere, 
                chapter_filter,
                progress_callback=lambda p: progress.progress(p)
            )
            
            if not df_atoms.empty:
                df_qc = compute_qc_real(df_atoms)
                
                st.session_state['df_src'] = df_src
                st.session_state['df_qc'] = df_qc
                st.session_state['df_atoms'] = df_atoms
                
                status.success(f"✅ Ingestion terminée : {len(df_src)} sujets traités, {len(df_atoms)} Qi extraites.")
            else:
                status.warning("⚠️ Aucune Qi extraite. Vérifiez les URLs ou le filtre chapitre.")
                
        except Exception as e:
            status.error(f"❌ Erreur : {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    st.divider()

    # --- TABLEAU SUJETS ---
    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        df_view = st.session_state['df_src'].copy()
        df_view = df_view.rename(columns={"Annee": "Année", "Telechargement": "Sujet", "Corrige": "Corrigé"})
        
        # Afficher le tableau avec colonnes Sujet et Corrigé
        display_cols = ["Fichier", "Nature", "Année", "Sujet", "Corrigé"]
        if "Corrigé" not in df_view.columns:
            display_cols = ["Fichier", "Nature", "Année", "Sujet"]
        
        st.data_editor(
            df_view[display_cols], 
            column_config={
                "Sujet": st.column_config.LinkColumn("📥 Sujet", display_text="PDF"),
                "Corrigé": st.column_config.LinkColumn("📝 Corrigé", display_text="PDF"),
            },
            hide_index=True, 
            use_container_width=True, 
            disabled=True
        )

        st.divider()

        # --- BASE QC ---
        st.markdown("### 🧠 Base de Connaissance (QC)")
        
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc']
            if sel_chapitres:
                qc_view = qc_view[qc_view["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Aucune QC pour ces chapitres.")
            else:
                chapters = qc_view["Chapitre"].unique()
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    st.markdown(f"#### 📘 {chap} ({len(subset)} QC)")
                    
                    for idx, row in subset.iterrows():
                        st.markdown(f"""
                        <div class="qc-header-box">
                            <span class="qc-id-text">{row['QC_ID']}</span>
                            <span class="qc-title-text">{row['Titre']}</span><br>
                            <span class="qc-meta-text">Score(q)={row['Score']} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        
                        with c1:
                            with st.expander("🔥 Déclencheurs"):
                                triggers = row['Triggers'] if isinstance(row['Triggers'], list) else []
                                for t in triggers:
                                    st.markdown(f"<span class='trigger-item'>\"{t}\"</span>", unsafe_allow_html=True)
                        
                        with c2:
                            with st.expander("⚙️ ARI"):
                                ari = row['ARI'] if isinstance(row['ARI'], list) else []
                                for s in ari:
                                    st.markdown(f"<span class='ari-step'>{s}</span>", unsafe_allow_html=True)
                        
                        with c3:
                            with st.expander("🧾 FRT"):
                                frt_data = row['FRT_DATA'] if isinstance(row['FRT_DATA'], list) else []
                                for block in frt_data:
                                    cls = {"usage": "c-usage", "method": "c-method", "trap": "c-trap", "conc": "c-conc"}.get(block.get('type', ''), "")
                                    st.markdown(f"<div class='frt-block {cls}'><span class='frt-title'>{block.get('title', '')}</span><div class='frt-content'>{block.get('text', '')}</div></div>", unsafe_allow_html=True)
                        
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                evidence = row['Evidence'] if isinstance(row['Evidence'], list) else []
                                qi_by_file = defaultdict(list)
                                for item in evidence:
                                    qi_by_file[item['Fichier']].append(item['Qi'])
                                html = ""
                                for f, qlist in qi_by_file.items():
                                    html += f"<div class='file-block'><div class='file-header'>📁 {f}</div>"
                                    for q in qlist[:5]:
                                        q_disp = q[:100] + "..." if len(q) > 100 else q
                                        html += f"<div class='qi-item'>\"{q_disp}\"</div>"
                                    if len(qlist) > 5:
                                        html += f"<div class='qi-item'>... +{len(qlist)-5} autres</div>"
                                    html += "</div>"
                                st.markdown(html, unsafe_allow_html=True)
                        
                        st.write("")
        
        # --- SATURATION ---
        st.divider()
        st.markdown("### 📈 Analyse de Saturation")
        
        if st.button("Lancer Analyse Saturation"):
            if 'df_atoms' in st.session_state and not st.session_state['df_atoms'].empty:
                with st.spinner("Calcul en cours..."):
                    df_chart = compute_saturation_real(st.session_state['df_atoms'])
                
                if not df_chart.empty:
                    st.line_chart(df_chart, x="Sujets (N)", y="QC Découvertes", color="#2563eb")
                    
                    max_qc = df_chart["QC Découvertes"].max()
                    sat_point = df_chart[df_chart["QC Découvertes"] >= max_qc * 0.9]
                    if not sat_point.empty:
                        sat_n = sat_point.iloc[0]["Sujets (N)"]
                        st.success(f"✅ Seuil de Saturation atteint à ~{sat_n} sujets.")

# ==============================================================================
# ONGLET 2 - AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        st.markdown("#### ✅ 1. Test Interne")
        if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
            t1_file = st.selectbox("Sujet", st.session_state['df_src']["Fichier"])
            
            if st.button("AUDIT INTERNE"):
                row = st.session_state['df_src'][st.session_state['df_src']["Fichier"] == t1_file].iloc[0]
                results = audit_internal_real(row["Qi_Data"], st.session_state['df_qc'])
                
                if results:
                    matched = sum(1 for r in results if r["Statut"] == "✅ MATCH")
                    st.metric("Couverture", f"{(matched/len(results))*100:.0f}%")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        st.divider()
        st.markdown("#### 🌍 2. Test Externe")
        
        up = st.file_uploader("PDF externe", type="pdf")
        if up and st.button("AUDIT EXTERNE"):
            coverage, results = audit_external_real(up.read(), st.session_state['df_qc'], sel_chapitres[0] if sel_chapitres else None)
            st.metric("Couverture", f"{coverage}%")
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Lancez l'usine d'abord.")
