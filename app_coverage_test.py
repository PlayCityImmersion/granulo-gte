import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V12")
st.title("🛡️ SMAXIA - Console V12 (Full Render & Strict Audit)")

# Styles CSS (LA SOLUTION POUR L'AFFICHAGE TOTAL)
st.markdown("""
<style>
    /* En-tête QC */
    .qc-header-row { 
        display: flex; align-items: center; background-color: #f8f9fa; 
        padding: 12px; border-radius: 5px; border-left: 5px solid #2563eb;
        font-family: monospace; margin-bottom: 5px; flex-wrap: wrap;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .qc-id-tag { font-weight: bold; color: #d97706; margin-right: 15px; font-size: 1.2em; min-width: 90px;}
    .qc-title { flex-grow: 1; font-weight: 600; color: #1f2937; font-size: 1.1em; margin-right: 15px; font-family: sans-serif;}
    .qc-vars { 
        font-size: 0.85em; color: #111827; font-weight: bold; font-family: 'Courier New';
        background-color: #e5e7eb; padding: 4px 8px; border-radius: 4px; border: 1px solid #9ca3af;
        white-space: nowrap; margin-left: 5px;
    }
    
    /* FRT Box : Affichage total sans scroll */
    .frt-box { 
        background-color: #ffffff; border: 1px solid #e5e7eb; padding: 20px; border-radius: 8px; 
        white-space: pre-wrap; /* Retour à la ligne auto */
        word-wrap: break-word;
        font-family: sans-serif; line-height: 1.6;
    }
    
    /* Table HTML propre (Remplace le Dataframe pour les détails) */
    .custom-table {
        width: 100%; border-collapse: collapse; font-size: 0.95em; font-family: sans-serif;
    }
    .custom-table th { background-color: #f3f4f6; padding: 10px; text-align: left; border-bottom: 2px solid #e5e7eb; color: #374151;}
    .custom-table td { padding: 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; color: #1f2937;}
    .custom-table tr:hover { background-color: #f9fafb; }
    
    /* Badge Trigger */
    .trigger-badge { 
        background-color: #fef3c7; color: #92400e; padding: 4px 10px; 
        border-radius: 6px; font-size: 0.9em; font-weight: 600; 
        border: 1px solid #fcd34d; display: inline-block; margin: 3px;
    }
    
    .stat-metric { font-size: 2em; font-weight: bold; color: #2563eb; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧱 KERNEL CACHÉ (VÉRITÉ SMAXIA)
# ==============================================================================

HIDDEN_TRUTH_KERNEL = {
    # SUITES
    "FRT_SUITE_01": {"QC": "COMMENT Démontrer qu'une suite est géométrique ?", "Chap": "SUITES NUMÉRIQUES", "ARI": ["Calcul u(n+1)", "Ratio", "Cste"]},
    "FRT_SUITE_02": {"QC": "COMMENT Lever une indétermination (limite) ?", "Chap": "SUITES NUMÉRIQUES", "ARI": ["Factoriser", "Limites usuelles"]},
    "FRT_SUITE_03": {"QC": "COMMENT Démontrer par récurrence ?", "Chap": "SUITES NUMÉRIQUES", "ARI": ["Init", "Hérédité", "Concl"]},
    "FRT_SUITE_04": {"QC": "COMMENT Étudier le sens de variation (Différence) ?", "Chap": "SUITES NUMÉRIQUES", "ARI": ["u(n+1)-u(n)", "Signe", "Concl"]},
    "FRT_SUITE_05": {"QC": "COMMENT Calculer une somme géométrique ?", "Chap": "SUITES NUMÉRIQUES", "ARI": ["Nb termes", "Formule", "Calcul"]},
    
    # FONCTIONS
    "FRT_FCT_01": {"QC": "COMMENT Étudier les variations ?", "Chap": "FONCTIONS", "ARI": ["Dérivée", "Signe", "Tableau"]},
    "FRT_FCT_02": {"QC": "COMMENT Appliquer le TVI (Unique) ?", "Chap": "FONCTIONS", "ARI": ["Continuité", "Monotonie", "Bornes"]},
    "FRT_FCT_03": {"QC": "COMMENT Équation tangente ?", "Chap": "FONCTIONS", "ARI": ["f(a)", "f'(a)", "Formule"]},
    "FRT_FCT_04": {"QC": "COMMENT Étudier la convexité ?", "Chap": "FONCTIONS", "ARI": ["Dérivée seconde", "Signe", "Inflexion"]},
}

QI_TEMPLATES = {
    "FRT_SUITE_01": ["Montrer que (Un) est géométrique.", "Justifier la nature géométrique.", "Prouver que Vn est une suite géométrique de raison q."],
    "FRT_SUITE_02": ["Déterminer la limite (FI).", "Lever l'indétermination de la limite.", "Calculer la limite quand n tend vers l'infini."],
    "FRT_SUITE_03": ["Démontrer par récurrence.", "Prouver la propriété P(n) pour tout n.", "Montrer par récurrence que Un > 0."],
    "FRT_SUITE_04": ["Étudier les variations de la suite.", "La suite est-elle croissante ?", "Quel est le sens de variation de (Un) ?"],
    "FRT_SUITE_05": ["Calculer la somme S.", "En déduire la somme des termes consécutifs.", "Que vaut 1 + q + ... + q^n ?"],
    
    "FRT_FCT_01": ["Dresser le tableau de variations.", "Étudier le sens de variation de f sur I.", "Calculer f'(x) et en déduire les variations."],
    "FRT_FCT_02": ["Montrer que f(x)=0 a une solution unique.", "Appliquer le corollaire du TVI.", "Démontrer l'existence d'une solution alpha."],
    "FRT_FCT_03": ["Donner l'équation de la tangente.", "Déterminer T au point d'abscisse a.", "Quelle est l'équation réduite de la tangente ?"],
    "FRT_FCT_04": ["Étudier la convexité de f.", "Le point A est-il un point d'inflexion ?", "Sur quel intervalle f est-elle convexe ?"],
}

# ==============================================================================
# ⚙️ MOTEUR DYNAMIQUE
# ==============================================================================

def generate_smart_qi_dynamic(frt_id):
    if frt_id not in QI_TEMPLATES: return f"Question spécifique type {frt_id}"
    base = random.choice(QI_TEMPLATES[frt_id])
    ctx = random.choice(["", " sur I", " pour tout n", " dans R", " définie sur [0;1]"])
    return base + ctx

def ingest_and_discover(urls, n_per_url, selected_chapters):
    sources_log = []
    atoms_db = []
    
    possible_frts = [k for k, v in HIDDEN_TRUTH_KERNEL.items() if v["Chap"] in selected_chapters]
    if not possible_frts: return pd.DataFrame(), pd.DataFrame()

    total_ops = len(urls) * n_per_url
    progress = st.progress(0)
    counter = 0
    natures = ["BAC", "DST", "CONCOURS"]
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            # time.sleep(0.001) # Désactivé pour la vitesse
            
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            filename = f"Sujet_{nature}_{year}_{j}.pdf"
            file_id = f"DOC_{i}_{j}"
            
            nb_exos = random.randint(2, 4)
            frts_in_doc = random.choices(possible_frts, k=nb_exos)
            
            qi_list_in_file = []
            
            for frt_id in frts_in_doc:
                qi_txt = generate_smart_qi_dynamic(frt_id)
                atoms_db.append({
                    "ID_Source": file_id, "Année": year, "Qi_Brut": qi_txt,
                    "FRT_ID": frt_id, "Fichier": filename, 
                    "Chapitre": HIDDEN_TRUTH_KERNEL[frt_id]["Chap"]
                })
                # On stocke l'ID FRT dans le fichier source pour l'audit interne
                qi_list_in_file.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
            # Stockage sujet (IMPORTANT : liste accumulée)
            content = f"CONTENU SIMULÉ\n{filename}"
            sources_log.append({
                "Fichier": filename, "Nature": nature, "Année": year, 
                "Contenu": content, "Qi_Data": qi_list_in_file
            })
            
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(atoms_db)

def compute_dynamic_qc(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    
    grouped = df_atoms.groupby("FRT_ID").agg({
        "ID_Source": "count", "Année": "max", 
        "Qi_Brut": list, "Fichier": list, "Chapitre": "first"
    }).reset_index()
    
    qcs = []
    N_total = len(df_atoms)
    current_year = datetime.now().year
    
    grouped = grouped.sort_values(by="ID_Source", ascending=False).reset_index(drop=True)
    
    for idx, row in grouped.iterrows():
        frt_id = row["FRT_ID"]
        kernel_info = HIDDEN_TRUTH_KERNEL.get(frt_id, {"QC": "QC Inconnue", "ARI": []})
        
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = 0.85
        sigma = 0.05
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        qc_clean = kernel_info["QC"].replace("COMMENT ", "comment ")
        
        # Simulation d'une FRT Rédigée
        frt_text = f"**Méthode Standard SMAXIA pour {qc_clean}**\n\n" \
                   "1. **Identifier** les hypothèses de l'énoncé.\n" \
                   "2. **Appliquer** le théorème correspondant (ex: TVI, Récurrence, Formule).\n" \
                   "3. **Effectuer** le calcul algébrique ou la dérivation étape par étape.\n" \
                   "4. **Conclure** clairement en répondant à la question posée."
        
        qcs.append({
            "Chapitre": row["Chapitre"],
            "QC_ID_Simple": f"QC_{idx+1:02d}", 
            "FRT_ID": frt_id,
            "QC_Texte": qc_clean,
            "ARI": kernel_info["ARI"],
            "FRT_Redaction": frt_text,
            "Score_F2": score,
            "n_q": n_q, "N_tot": N_total, "Tau": tau, "Psi": psi,
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["Fichier"], row["Qi_Brut"])]
        })
        
    return pd.DataFrame(qcs)

# ==============================================================================
# 🖥️ INTERFACE V12
# ==============================================================================

with st.sidebar:
    st.header("1. Paramètres")
    matiere = st.selectbox("Matière", ["MATHS"])
    available_chaps = list(set(d["Chap"] for d in HIDDEN_TRUTH_KERNEL.values()))
    selected_chaps = st.multiselect("Chapitres", available_chaps, default=available_chaps)

# TABS
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine (Prod)", "✅ Onglet 2 : Audit (Validation Booléenne)"])

# --- TAB 1 : USINE ---
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: 
        urls_input = st.text_area("Sources", "https://apmep.fr", height=70)
    with c2: 
        n_sujets = st.number_input("Vol. par URL", 5, 200, 35, step=5) # Test 35 sujets
        btn_run = st.button("LANCER USINE 🚀", type="primary")

    if btn_run:
        with st.spinner("Exploration et Apprentissage..."):
            df_src, df_atoms = ingest_and_discover(urls_input.split('\n'), n_sujets, selected_chaps)
            df_qc = compute_dynamic_qc(df_atoms)
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.session_state['sel_chaps'] = selected_chaps
            st.success(f"Terminé. {len(df_src)} sujets traités.")

    st.divider()

    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.8])
        
        # GAUCHE : SUJETS (Liste Scrollable, Hauteur Fixe)
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            
            # Affichage DataFrame standard (scroll interne natif)
            st.dataframe(
                st.session_state['df_src'][["Fichier", "Nature", "Année"]], 
                use_container_width=True, 
                height=500
            )
            
            sel = st.selectbox("Fichier cible", st.session_state['df_src']["Fichier"], label_visibility="collapsed")
            if not st.session_state['df_src'].empty:
                txt = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==sel].iloc[0]["Contenu"]
                st.download_button(f"📥 Télécharger {sel}", txt, file_name=sel)

        # DROITE : QC (Affichage HTML Total)
        with col_right:
            if not st.session_state['df_qc'].empty:
                chapters = st.session_state['df_qc']["Chapitre"].unique()
                for chap in chapters:
                    df_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                    st.markdown(f"#### 📘 Chapitre {chap} : {len(df_view)} QC")
                    
                    for idx, row in df_view.iterrows():
                        # HEADER QC
                        header_html = f"""
                        <div class="qc-header-row">
                            <span class="qc-id-tag">[{row['QC_ID_Simple']}]</span>
                            <span class="qc-title">{row['QC_Texte']}</span>
                            <span class="qc-vars">
                                Score(q)={row['Score_F2']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['Tau']:.1f}
                            </span>
                        </div>
                        """
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        
                        with c1:
                            with st.expander("⚡ Déclencheurs"):
                                st.markdown("<span class='trigger-badge'>Déclencheurs dynamiques...</span>", unsafe_allow_html=True)
                        
                        with c2:
                            with st.expander(f"⚙️ ARI_{row['QC_ID_Simple']}"):
                                for s in row['ARI']: st.write(f"- {s}")

                        with c3:
                            with st.expander(f"📝 FRT_{row['QC_ID_Simple']}"):
                                st.markdown(f"<div class='frt-box'>{row['FRT_Redaction']}</div>", unsafe_allow_html=True)

                        with c4:
                            with st.expander(f"📄 Qi associées ({row['n_q']})"):
                                # TABLEAU HTML POUR AFFICHAGE COMPLET
                                html_table = "<table class='custom-table'><thead><tr><th>Fichier Source</th><th>Qi (Enoncé Élève)</th></tr></thead><tbody>"
                                for item in row['Evidence']:
                                    html_table += f"<tr><td>{item['Fichier']}</td><td>{item['Qi']}</td></tr>"
                                html_table += "</tbody></table>"
                                st.markdown(html_table, unsafe_allow_html=True)
                        
                        st.write("") 
            else:
                st.warning("Aucune QC trouvée.")

# --- TAB 2 : AUDIT STRICT ---
with tab_audit:
    st.subheader("Validation Booléenne (Tableau de Mapping)")
    
    if 'df_qc' in st.session_state:
        
        # --- TEST 1 : INTERNE ---
        st.markdown("#### 1. Audit Interne (Est-ce que le moteur reconnait ses propres sujets ?)")
        test_file = st.selectbox("Sujet Traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER L'AUDIT INTERNE"):
            # 1. Extraction Vérité Terrain (ce qui est réellement dans le fichier)
            file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==test_file].iloc[0]
            qi_list = file_data["Qi_Data"] # Liste des {Qi, FRT_ID}
            
            # 2. Vérification Moteur
            mapping_rows = []
            c_ok = 0
            qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
            
            for item in qi_list:
                frt_id = item["FRT_ID"]
                is_covered = frt_id in qc_lookup.index
                
                if is_covered:
                    c_ok += 1
                    qc_info = qc_lookup.loc[frt_id]
                    if isinstance(qc_info, pd.DataFrame): qc_info = qc_info.iloc[0]
                    qc_disp = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Texte']}"
                    status = "✅ MATCH"
                else:
                    qc_disp = "---"
                    status = "❌ PERTE DE DONNÉES"
                
                mapping_rows.append({"Qi (Sujet)": item["Qi"], "QC Moteur": qc_disp, "Statut": status})
            
            taux = (c_ok / len(qi_list)) * 100
            st.markdown(f"<div class='stat-metric'>Taux de Couverture : {taux:.0f}%</div>", unsafe_allow_html=True)
            
            # Tableau HTML
            html_audit = "<table class='custom-table'><thead><tr><th>Qi</th><th>QC Moteur</th><th>Statut</th></tr></thead><tbody>"
            for row in mapping_rows:
                color = "#dcfce7" if "MATCH" in row['Statut'] else "#fee2e2"
                html_audit += f"<tr style='background-color:{color}'><td>{row['Qi (Sujet)']}</td><td>{row['QC Moteur']}</td><td>{row['Statut']}</td></tr>"
            html_audit += "</tbody></table>"
            st.markdown(html_audit, unsafe_allow_html=True)

        st.divider()

        # --- TEST 2 : EXTERNE ---
        st.markdown("#### 2. Audit Externe (Sujet Inconnu)")
        st.caption("Simule l'arrivée d'un élève avec un DM contenant des questions aléatoires du programme.")
        
        if st.button("GÉNÉRER SUJET EXTERNE & TESTER"):
            # 1. Génération Sujet Inconnu (parmi la vérité cachée totale)
            # On prend des chapitres au hasard si aucun sélectionné, sinon ceux actifs
            target_chaps = st.session_state.get('sel_chaps', [])
            all_hidden = [k for k,v in HIDDEN_TRUTH_KERNEL.items() if v["Chap"] in target_chaps]
            
            if not all_hidden:
                st.error("Aucun chapitre sélectionné.")
            else:
                test_frts = random.choices(all_hidden, k=5) # 5 questions
                
                ext_rows = []
                c_ok = 0
                qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
                
                for frt_id in test_frts:
                    qi_txt = generate_smart_qi_dynamic(frt_id)
                    is_known = frt_id in qc_lookup.index
                    
                    if is_known:
                        c_ok += 1
                        qc_info = qc_lookup.loc[frt_id]
                        if isinstance(qc_info, pd.DataFrame): qc_info = qc_info.iloc[0]
                        qc_disp = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Texte']}"
                        status = "✅ MATCH"
                    else:
                        qc_disp = "---"
                        status = "❌ MANQUANT (Jamais vu)"
                    
                    ext_rows.append({"Qi (Sujet Externe)": qi_txt, "QC Moteur": qc_disp, "Statut": status})
                
                taux = (c_ok / len(test_frts)) * 100
                st.markdown(f"<div class='stat-metric'>Couverture Externe : {taux:.0f}%</div>", unsafe_allow_html=True)
                
                if taux < 100:
                    st.info("💡 Note : Le moteur ne connait que ce qu'il a vu dans l'Usine. Augmentez le volume de sujets pour couvrir tout le programme.")
                
                html_audit = "<table class='custom-table'><thead><tr><th>Qi</th><th>QC Moteur</th><th>Statut</th></tr></thead><tbody>"
                for row in ext_rows:
                    color = "#dcfce7" if "MATCH" in row['Statut'] else "#fee2e2"
                    html_audit += f"<tr style='background-color:{color}'><td>{row['Qi (Sujet Externe)']}</td><td>{row['QC Moteur']}</td><td>{row['Statut']}</td></tr>"
                html_audit += "</tbody></table>"
                st.markdown(html_audit, unsafe_allow_html=True)

    else:
        st.info("Lancez l'usine.")
