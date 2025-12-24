# smaxia_console_v31.py
# =============================================================================
# SMAXIA - Console V31 (Saturation Proof)
# UI IDENTIQUE à Gemini + Moteur RÉEL (F1/F2)
# =============================================================================

import streamlit as st
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Import du moteur RÉEL
from smaxia_granulo_engine_real import (
    ingest_real,
    compute_qc_real,
    compute_saturation_real,
    audit_internal_real,
    audit_external_real
)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V31")
st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# 🎨 STYLES CSS (GABARIT SMAXIA - IDENTIQUE GEMINI)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
    .qc-header-box {
        background-color: #f8f9fa; border-left: 6px solid #2563eb; 
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .qc-id-text { color: #d97706; font-weight: 900; font-size: 1.2em; margin-right: 10px; }
    .qc-title-text { color: #1f2937; font-weight: 700; font-size: 1.15em; }
    .qc-meta-text { 
        font-family: 'Courier New', monospace; font-size: 0.85em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 4px 8px; border-radius: 4px; margin-top: 5px; display: inline-block;
    }

    /* DETAILS */
    .trigger-item {
        background-color: #fff1f2; color: #991b1b; padding: 5px 10px; margin-bottom: 4px; 
        border-radius: 4px; border-left: 4px solid #f87171; font-weight: 600; font-size: 0.9em; display: block;
    }
    .ari-step {
        background-color: #f3f4f6; color: #374151; padding: 4px 8px; margin-bottom: 3px; 
        border-radius: 3px; font-family: monospace; font-size: 0.85em; border: 1px dashed #d1d5db; display: block;
    }

    /* FRT */
    .frt-block { padding: 12px; border-bottom: 1px solid #e2e8f0; background: white; margin-bottom: 5px; border-radius: 4px; border: 1px solid #e2e8f0;}
    .frt-title { font-weight: 800; text-transform: uppercase; font-size: 0.75em; display: block; margin-bottom: 6px; letter-spacing: 0.5px; }
    .frt-content { font-family: 'Segoe UI', sans-serif; font-size: 0.95em; color: #334155; line-height: 1.6; white-space: pre-wrap; }
    
    .c-usage { color: #d97706; border-left: 4px solid #d97706; }
    .c-method { color: #059669; border-left: 4px solid #059669; background-color: #f0fdf4; }
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; }
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; }

    /* QI CARDS */
    .file-block { margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
    .file-header { background-color: #f1f5f9; padding: 8px 12px; font-weight: 700; font-size: 0.85em; color: #475569; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; }
    .qi-item { background-color: white; padding: 10px 12px; border-bottom: 1px solid #f8fafc; font-family: 'Georgia', serif; font-size: 0.95em; color: #1e293b; border-left: 3px solid #9333ea; margin: 0; }

    /* SATURATION */
    .sat-box { background-color: #f0f9ff; border: 1px solid #bae6fd; padding: 20px; border-radius: 8px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR - PARAMÈTRES ACADÉMIQUES
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
        
        try:
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
                
                st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")
            else:
                st.warning("Aucune Qi extraite. Vérifiez les URLs.")
                
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

    st.divider()

    # --- TABLEAU SUJETS ---
    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        df_view = st.session_state['df_src'].rename(columns={"Annee": "Année", "Telechargement": "Lien"})
        st.data_editor(
            df_view[["Fichier", "Nature", "Année", "Lien"]], 
            column_config={"Lien": st.column_config.LinkColumn("Téléchargement", display_text="📥 PDF")},
            hide_index=True, 
            use_container_width=True, 
            disabled=True
        )

        st.divider()

        # --- BASE QC ---
        st.markdown("### 🧠 Base de Connaissance (QC)")
        
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Aucune QC pour ces chapitres.")
            else:
                chapters = qc_view["Chapitre"].unique()
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    st.markdown(f"#### 📘 {chap} ({len(subset)} QC)")
                    
                    for idx, row in subset.iterrows():
                        # Header QC
                        st.markdown(f"""
                        <div class="qc-header-box">
                            <span class="qc-id-text">{row['QC_ID']}</span>
                            <span class="qc-title-text">{row['Titre']}</span><br>
                            <span class="qc-meta-text">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 4 colonnes
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
        else:
            st.warning("Aucune QC générée.")
        
        # --- SATURATION ---
        st.divider()
        st.markdown("### 📈 Analyse de Saturation (Preuve de Complétude)")
        st.caption("Ce graphique montre à quelle vitesse le moteur découvre l'ensemble des types de questions (QC) possibles.")
        
        col_sim_1, col_sim_2 = st.columns([1, 3])
        with col_sim_1:
            if st.button("Lancer Analyse"):
                with col_sim_2:
                    if 'df_atoms' in st.session_state and not st.session_state['df_atoms'].empty:
                        df_chart = compute_saturation_real(st.session_state['df_atoms'])
                        
                        if not df_chart.empty:
                            # Graphique
                            st.line_chart(df_chart, x="Sujets (N)", y="QC Découvertes", color="#2563eb")
                            
                            # Tableau
                            st.markdown("#### 🔢 Données de Convergence")
                            step = max(1, len(df_chart) // 10)
                            df_display = df_chart.iloc[::step].reset_index(drop=True)
                            st.dataframe(df_display, use_container_width=True)
                            
                            # Analyse
                            max_qc = df_chart["QC Découvertes"].max()
                            sat_point = df_chart[df_chart["QC Découvertes"] >= max_qc * 0.9]
                            if not sat_point.empty:
                                sat_n = sat_point.iloc[0]["Sujets (N)"]
                                st.success(f"✅ **Seuil de Saturation (Granulo 15) atteint à ~{sat_n} sujets.** À partir de ce point, l'ajout de nouveaux sujets n'apporte que des variations mineures (Qi), plus de nouvelles structures (QC).")
                        else:
                            st.warning("Pas assez de données.")
                    else:
                        st.info("Lancez d'abord l'usine.")

# ==============================================================================
# ONGLET 2 - AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        # --- AUDIT INTERNE ---
        st.markdown("#### ✅ 1. Test Interne")
        
        if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
            t1_file = st.selectbox("Sujet", st.session_state['df_src']["Fichier"])
            
            if st.button("AUDIT INTERNE"):
                row = st.session_state['df_src'][st.session_state['df_src']["Fichier"] == t1_file].iloc[0]
                qi_data = row["Qi_Data"]
                
                results = audit_internal_real(qi_data, st.session_state['df_qc'])
                
                if results:
                    matched = sum(1 for r in results if r["Statut"] == "✅ MATCH")
                    coverage = (matched / len(results)) * 100 if results else 0
                    
                    st.metric("Couverture", f"{coverage:.0f}%")
                    
                    df_results = pd.DataFrame(results)
                    
                    def color_status(row):
                        if row['Statut'] == "✅ MATCH":
                            return ['background-color: #dcfce7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(df_results.style.apply(color_status, axis=1), use_container_width=True)
                else:
                    st.warning("Aucune Qi dans ce sujet.")
        
        st.divider()
        
        # --- AUDIT EXTERNE ---
        st.markdown("#### 🌍 2. Test Externe")
        
        up = st.file_uploader("PDF", type="pdf")
        
        if up:
            if st.button("AUDIT EXTERNE"):
                pdf_bytes = up.read()
                chapter_filter = sel_chapitres[0] if sel_chapitres else None
                
                coverage, results = audit_external_real(pdf_bytes, st.session_state['df_qc'], chapter_filter)
                
                if results:
                    st.markdown(f"### Taux : {coverage:.1f}%")
                    
                    df_results = pd.DataFrame(results)
                    
                    def color_status(row):
                        if row['Statut'] == "✅ MATCH":
                            return ['background-color: #dcfce7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(df_results.style.apply(color_status, axis=1), use_container_width=True)
                else:
                    st.warning("Aucune Qi extraite.")
    else:
        st.info("Lancez l'usine.")
