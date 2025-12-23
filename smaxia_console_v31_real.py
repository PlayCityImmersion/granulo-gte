# smaxia_console_v31_real.py
# =============================================================================
# SMAXIA - Console V31 (MOTEUR RÉEL)
# =============================================================================
# UI Gemini V31 connectée au moteur RÉEL (zéro hardcode)
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
    audit_external_real,
    CHAPTER_KEYWORDS
)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V31 (Real Engine)")
st.title("🛡️ SMAXIA - Console V31 (Real Engine)")

# ==============================================================================
# 🎨 STYLES CSS (GABARIT SMAXIA - INCHANGÉ)
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
    
    /* STATUS */
    .status-real { background-color: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-info { background-color: #dbeafe; color: #1e40af; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; }
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
    st.header("📚 Paramètres Académiques")
    
    st.markdown('<span class="status-real">🔴 MOTEUR RÉEL</span>', unsafe_allow_html=True)
    st.caption("Extraction PDF réelle, zéro données fake")
    
    st.divider()
    
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps = LISTE_CHAPITRES.get(sel_matiere, [])
    sel_chapitres = st.multiselect("Chapitres (Filtre)", chaps, default=chaps[:1] if chaps else [])
    
    st.divider()
    
    # Stats en temps réel
    if 'all_qis' in st.session_state and st.session_state['all_qis']:
        st.markdown("### 📊 Statistiques")
        st.metric("Qi extraites", len(st.session_state['all_qis']))
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            st.metric("QC générées", len(st.session_state['df_qc']))

# ==============================================================================
# ONGLETS
# ==============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# ==============================================================================
# ONGLET 1 - USINE
# ==============================================================================
with tab_usine:
    
    st.markdown("""
    <div class="status-info">
        ℹ️ <strong>Moteur Réel</strong> : Ce moteur scrape les URLs, télécharge les PDFs, 
        extrait le texte et les questions, puis cluster les Qi en QC par similarité Jaccard.
    </div>
    """, unsafe_allow_html=True)
    
    # --- ZONE INJECTION ---
    c1, c2 = st.columns([3, 1])
    with c1:
        urls = st.text_area(
            "URLs Sources (sites avec des PDFs de sujets)",
            "https://www.apmep.fr/Terminale-S-702-sujets-702",
            height=80,
            help="Entrez des URLs de sites contenant des liens vers des PDFs de sujets"
        )
    with c2:
        vol = st.number_input("Volume", 5, 100, 15, step=5, help="Nombre de PDFs à traiter")
        run = st.button("🚀 LANCER L'USINE", type="primary", use_container_width=True)

    if run:
        url_list = [u.strip() for u in urls.split('\n') if u.strip()]
        chapter_filter = sel_chapitres[0] if sel_chapitres else None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Scraping des URLs et téléchargement des PDFs...")
        
        try:
            df_src, df_atoms, all_qis = ingest_real(
                url_list, 
                vol, 
                sel_matiere, 
                chapter_filter,
                progress_callback=lambda p: progress_bar.progress(p)
            )
            
            if all_qis:
                status_text.text("🧠 Clustering Qi → QC...")
                df_qc = compute_qc_real(all_qis)
                
                st.session_state['df_src'] = df_src
                st.session_state['df_qc'] = df_qc
                st.session_state['all_qis'] = all_qis
                st.session_state['chapter_filter'] = chapter_filter
                
                status_text.empty()
                st.success(f"✅ Ingestion terminée : {len(df_src)} sujets traités, {len(all_qis)} Qi extraites, {len(df_qc)} QC générées.")
            else:
                status_text.empty()
                st.warning("⚠️ Aucune Qi extraite. Vérifiez les URLs ou le filtre chapitre.")
                
        except Exception as e:
            status_text.empty()
            st.error(f"❌ Erreur : {str(e)}")

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
            df_qc = st.session_state['df_qc']
            
            # Filtrer par chapitres sélectionnés
            if sel_chapitres:
                qc_view = df_qc[df_qc["Chapitre"].isin(sel_chapitres)]
            else:
                qc_view = df_qc
            
            if qc_view.empty:
                st.info("Aucune QC pour les chapitres sélectionnés.")
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
                            <span class="qc-meta-text">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_réc={row['t_rec']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 4 colonnes
                        c1, c2, c3, c4 = st.columns(4)
                        
                        with c1:
                            with st.expander("🔥 Déclencheurs"):
                                triggers = row['Triggers'] if isinstance(row['Triggers'], list) else []
                                if triggers:
                                    for t in triggers:
                                        st.markdown(f"<span class='trigger-item'>\"{t}\"</span>", unsafe_allow_html=True)
                                else:
                                    st.caption("Aucun déclencheur identifié")
                        
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
                                    for q in qlist[:5]:  # Max 5 par fichier
                                        q_display = q[:100] + "..." if len(q) > 100 else q
                                        html += f"<div class='qi-item'>\"{q_display}\"</div>"
                                    if len(qlist) > 5:
                                        html += f"<div class='qi-item'>… +{len(qlist)-5} autres</div>"
                                    html += "</div>"
                                st.markdown(html, unsafe_allow_html=True)
                        
                        st.write("")
        else:
            st.warning("Aucune QC générée. Lancez l'usine d'abord.")
        
        # --- SATURATION ---
        st.divider()
        st.markdown("### 📈 Analyse de Saturation (Preuve de Complétude)")
        st.caption("Ce graphique montre l'évolution du nombre de QC en fonction des sujets traités.")
        
        if 'all_qis' in st.session_state and st.session_state['all_qis']:
            if st.button("📊 Calculer la courbe de saturation"):
                with st.spinner("Calcul de la saturation..."):
                    df_sat = compute_saturation_real(st.session_state['all_qis'])
                    
                    if not df_sat.empty:
                        # Graphique
                        st.line_chart(df_sat, x="Sujets (N)", y="QC Découvertes")
                        
                        # Tableau
                        st.markdown("#### 🔢 Données de Convergence")
                        # Afficher tous les 5 sujets ou moins si petit volume
                        step = max(1, len(df_sat) // 10)
                        df_display = df_sat.iloc[::step].reset_index(drop=True)
                        st.dataframe(df_display, use_container_width=True)
                        
                        # Analyse
                        max_qc = df_sat["QC Découvertes"].max()
                        last_values = df_sat["QC Découvertes"].tail(3).tolist()
                        
                        if len(set(last_values)) == 1:
                            st.success(f"✅ **Saturation atteinte !** Les derniers sujets n'apportent plus de nouvelles QC ({max_qc} QC max).")
                        else:
                            # Trouver le point de saturation approximatif
                            sat_90 = df_sat[df_sat["QC Découvertes"] >= max_qc * 0.9]
                            if not sat_90.empty:
                                sat_point = sat_90.iloc[0]["Sujets (N)"]
                                st.info(f"📈 **Saturation ~90% atteinte à {sat_point} sujets.** Continuez pour confirmer la stabilité.")
                            else:
                                st.warning("⚠️ Saturation non atteinte. Ajoutez plus de sujets.")
                    else:
                        st.warning("Pas assez de données pour la saturation.")
        else:
            st.info("Lancez l'usine pour voir la courbe de saturation.")
    else:
        st.info("⏳ Lancez l'usine pour commencer l'extraction.")

# ==============================================================================
# ONGLET 2 - AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("🔍 Validation Booléenne")
    
    st.markdown("""
    **Objectifs :**
    - **Audit Interne** : Chaque Qi d'un sujet traité → QC = **100%**
    - **Audit Externe** : Sujet inconnu → QC = **≥ 95%**
    """)
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        # --- AUDIT INTERNE ---
        st.divider()
        st.markdown("#### ✅ 1. Test Interne (Sujet Traité)")
        
        if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
            fichiers = st.session_state['df_src']["Fichier"].tolist()
            t1_file = st.selectbox("Choisir un sujet traité", fichiers)
            
            if st.button("🔬 AUDIT INTERNE"):
                # Récupérer les Qi du sujet
                row = st.session_state['df_src'][st.session_state['df_src']["Fichier"] == t1_file].iloc[0]
                qi_data = row["Qi_Data"]
                
                results = audit_internal_real(qi_data, st.session_state['df_qc'])
                
                if results:
                    matched = sum(1 for r in results if r["Statut"] == "✅ MATCH")
                    coverage = (matched / len(results)) * 100 if results else 0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Couverture", f"{coverage:.0f}%")
                    col2.metric("Qi mappées", f"{matched}/{len(results)}")
                    
                    if coverage >= 100:
                        st.success("✅ 100% de couverture - SUCCÈS")
                    elif coverage >= 80:
                        st.warning(f"⚠️ {coverage:.0f}% de couverture - À améliorer")
                    else:
                        st.error(f"❌ {coverage:.0f}% de couverture - INSUFFISANT")
                    
                    # Tableau détaillé
                    df_results = pd.DataFrame(results)
                    
                    def highlight_status(row):
                        if row['Statut'] == "✅ MATCH":
                            return ['background-color: #dcfce7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(
                        df_results.style.apply(highlight_status, axis=1),
                        use_container_width=True
                    )
                else:
                    st.warning("Aucune Qi trouvée dans ce sujet.")
        
        # --- AUDIT EXTERNE ---
        st.divider()
        st.markdown("#### 🌍 2. Test Externe (Sujet Inconnu)")
        
        uploaded = st.file_uploader("Charger un PDF externe", type="pdf")
        
        if uploaded:
            if st.button("🔬 AUDIT EXTERNE"):
                pdf_bytes = uploaded.read()
                chapter_filter = st.session_state.get('chapter_filter', sel_chapitres[0] if sel_chapitres else None)
                
                with st.spinner("Analyse du sujet externe..."):
                    coverage, results = audit_external_real(pdf_bytes, st.session_state['df_qc'], chapter_filter)
                
                if results:
                    matched = sum(1 for r in results if r["Statut"] == "✅ MATCH")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Couverture", f"{coverage:.0f}%")
                    col2.metric("Qi couvertes", f"{matched}/{len(results)}")
                    
                    if coverage >= 95:
                        st.success(f"✅ {coverage:.0f}% de couverture - OBJECTIF ATTEINT (≥95%)")
                    elif coverage >= 80:
                        st.warning(f"⚠️ {coverage:.0f}% de couverture - PROCHE DE L'OBJECTIF")
                    else:
                        st.error(f"❌ {coverage:.0f}% de couverture - INSUFFISANT")
                    
                    # Tableau détaillé
                    df_results = pd.DataFrame(results)
                    
                    def highlight_status(row):
                        if row['Statut'] == "✅ MATCH":
                            return ['background-color: #dcfce7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(
                        df_results.style.apply(highlight_status, axis=1),
                        use_container_width=True
                    )
                else:
                    st.warning("Aucune Qi extraite du PDF externe.")
    else:
        st.info("⏳ Lancez d'abord l'usine pour générer des QC.")
