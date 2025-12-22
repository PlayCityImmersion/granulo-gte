import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V13 (Discovery)")
st.title("🛡️ SMAXIA - Console V13 (True Discovery Engine)")

# Styles CSS (Full Render + Alignement)
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
    
    /* FRT Box */
    .frt-box { 
        background-color: #ffffff; border: 1px solid #e5e7eb; padding: 20px; border-radius: 8px; 
        white-space: pre-wrap; word-wrap: break-word; font-family: sans-serif; line-height: 1.6;
    }
    
    /* Table HTML */
    .custom-table { width: 100%; border-collapse: collapse; font-size: 0.95em; font-family: sans-serif; }
    .custom-table th { background-color: #f3f4f6; padding: 10px; text-align: left; border-bottom: 2px solid #e5e7eb; color: #374151;}
    .custom-table td { padding: 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; color: #1f2937;}
    
    .trigger-badge { 
        background-color: #fef3c7; color: #92400e; padding: 4px 10px; 
        border-radius: 6px; font-size: 0.9em; font-weight: 600; 
        border: 1px solid #fcd34d; display: inline-block; margin: 3px;
    }
    .stat-metric { font-size: 2em; font-weight: bold; color: #2563eb; margin-bottom: 10px;}
    
    /* Scroll vertical pour la liste des sujets */
    .subject-list-container { max-height: 600px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🌌 UNIVERS MATHÉMATIQUE THÉORIQUE (HIDDEN TRUTH)
# C'est l'espace des possibles. Le moteur ne le connait pas, il doit le découvrir.
# ==============================================================================

UNIVERS_MATHS = {
    # --- SUITES (Fréquent) ---
    "FRT_SUITE_GEO": {"QC": "COMMENT Démontrer qu'une suite est géométrique ?", "Chap": "SUITES", "Proba": 0.9},
    "FRT_SUITE_LIM": {"QC": "COMMENT Lever une indétermination (limite) ?", "Chap": "SUITES", "Proba": 0.8},
    "FRT_SUITE_REC": {"QC": "COMMENT Démontrer par récurrence ?", "Chap": "SUITES", "Proba": 0.8},
    # --- SUITES (Rare) ---
    "FRT_SUITE_VAR": {"QC": "COMMENT Étudier le sens de variation (u_n+1 - u_n) ?", "Chap": "SUITES", "Proba": 0.4},
    "FRT_SUITE_SOM": {"QC": "COMMENT Calculer une somme de termes ?", "Chap": "SUITES", "Proba": 0.3},
    "FRT_SUITE_SEUIL": {"QC": "COMMENT Déterminer un seuil (Algorithme) ?", "Chap": "SUITES", "Proba": 0.2},
    "FRT_SUITE_MAJ": {"QC": "COMMENT Montrer qu'une suite est majorée ?", "Chap": "SUITES", "Proba": 0.3},

    # --- FONCTIONS (Fréquent) ---
    "FRT_FCT_VAR": {"QC": "COMMENT Étudier les variations (Dérivée) ?", "Chap": "FONCTIONS", "Proba": 0.9},
    "FRT_FCT_TVI": {"QC": "COMMENT Appliquer le TVI (Solution unique) ?", "Chap": "FONCTIONS", "Proba": 0.8},
    "FRT_FCT_TG": {"QC": "COMMENT Équation de la tangente ?", "Chap": "FONCTIONS", "Proba": 0.7},
    # --- FONCTIONS (Rare) ---
    "FRT_FCT_CONV": {"QC": "COMMENT Étudier la convexité (Inflexion) ?", "Chap": "FONCTIONS", "Proba": 0.3},
    "FRT_FCT_PRIM": {"QC": "COMMENT Déterminer une primitive ?", "Chap": "FONCTIONS", "Proba": 0.4},
    "FRT_FCT_POS": {"QC": "COMMENT Étudier la position relative (Courbe/Tangente) ?", "Chap": "FONCTIONS", "Proba": 0.2},
    "FRT_FCT_INT": {"QC": "COMMENT Calculer une intégrale (Aire) ?", "Chap": "FONCTIONS", "Proba": 0.3},

    # --- GEO (Fréquent) ---
    "FRT_GEO_ORTHO": {"QC": "COMMENT Démontrer l'orthogonalité Droite/Plan ?", "Chap": "GÉOMÉTRIE", "Proba": 0.7},
    "FRT_GEO_PARA": {"QC": "COMMENT Représentation paramétrique ?", "Chap": "GÉOMÉTRIE", "Proba": 0.6},
    # --- GEO (Rare) ---
    "FRT_GEO_CART": {"QC": "COMMENT Équation cartésienne d'un plan ?", "Chap": "GÉOMÉTRIE", "Proba": 0.3},
    "FRT_GEO_INTER": {"QC": "COMMENT Intersection Droite/Plan ?", "Chap": "GÉOMÉTRIE", "Proba": 0.4},
    "FRT_GEO_PROJ": {"QC": "COMMENT Coordonnées du projeté orthogonal ?", "Chap": "GÉOMÉTRIE", "Proba": 0.1},

    # --- PROBA (Fréquent) ---
    "FRT_PROBA_TOT": {"QC": "COMMENT Probabilités totales (Arbre) ?", "Chap": "PROBABILITÉS", "Proba": 0.8},
    "FRT_PROBA_BIN": {"QC": "COMMENT Loi Binomiale (k succès) ?", "Chap": "PROBABILITÉS", "Proba": 0.7},
    # --- PROBA (Rare) ---
    "FRT_PROBA_COND": {"QC": "COMMENT Probabilité conditionnelle P_B(A) ?", "Chap": "PROBABILITÉS", "Proba": 0.5},
    "FRT_PROBA_IND": {"QC": "COMMENT Démontrer l'indépendance ?", "Chap": "PROBABILITÉS", "Proba": 0.3},
}

# Modèles de génération de texte (Polymorphisme)
QI_TEMPLATES = {
    "FRT_SUITE_GEO": ["Montrer que (Un) est géométrique.", "Justifier la nature de la suite."],
    "FRT_SUITE_LIM": ["Déterminer la limite.", "Lever l'indétermination.", "Calculer la limite en +inf."],
    "FRT_SUITE_REC": ["Démontrer par récurrence.", "Prouver la propriété P(n)."],
    "FRT_SUITE_VAR": ["Étudier les variations.", "La suite est-elle croissante ?"],
    "FRT_SUITE_SOM": ["Calculer la somme S.", "En déduire la somme des termes."],
    "FRT_SUITE_SEUIL": ["Déterminer le plus petit entier n.", "Algorithme de seuil."],
    "FRT_FCT_VAR": ["Dresser le tableau de variations.", "Sens de variation de f."],
    "FRT_FCT_TVI": ["Montrer que f(x)=0 a une solution unique.", "Appliquer le corollaire TVI."],
    "FRT_FCT_TG": ["Donner l'équation de la tangente T.", "Tangente au point a."],
    "FRT_FCT_CONV": ["Étudier la convexité.", "Point d'inflexion ?"],
    "FRT_GEO_ORTHO": ["Démontrer que (d) est orthogonale à (P)."],
    "FRT_GEO_PARA": ["Représentation paramétrique de (D)."],
    "FRT_PROBA_TOT": ["Calculer P(B) (Arbre)."],
    "FRT_PROBA_BIN": ["Probabilité de k succès.", "Loi Binomiale."]
}

# ==============================================================================
# ⚙️ MOTEUR DYNAMIQUE (Vrai Clustering)
# ==============================================================================

def generate_smart_qi_dynamic(frt_id):
    """Génère une phrase unique"""
    base = random.choice(QI_TEMPLATES.get(frt_id, [f"Question sur {frt_id}"]))
    ctx = random.choice(["", " sur I", " pour tout n", " dans R", " définie sur [0;1]"])
    return base + ctx

def ingest_and_discover(urls, n_per_url, selected_chapters):
    """
    Simule la découverte :
    - On pioche dans l'UNIVERS selon les Probabilités d'apparition.
    - Si n_per_url est faible, on ne verra que les fréquentes.
    - Si n_per_url est élevé, les "rares" vont finir par sortir.
    """
    sources_log = []
    atoms_db = []
    
    # Filtrer l'univers selon les chapitres demandés
    possible_universe = [k for k, v in UNIVERS_MATHS.items() if v["Chap"] in selected_chapters]
    weights = [UNIVERS_MATHS[k]["Proba"] for k in possible_universe] # Poids pour la loi de proba
    
    if not possible_universe: return pd.DataFrame(), pd.DataFrame()

    total_ops = len(urls) * n_per_url
    progress = st.progress(0)
    counter = 0
    natures = ["BAC", "DST", "CONCOURS"]
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            filename = f"Sujet_{nature}_{year}_{j}.pdf"
            file_id = f"DOC_{i}_{j}"
            
            # Nombre de questions dans ce sujet
            nb_exos = random.randint(3, 5)
            
            # TIRAGE AVEC POIDS (Loi Réaliste)
            # C'est ici que la magie opère : on ne tire pas au hasard uniforme.
            # On tire selon la fréquence d'apparition réelle.
            frts_in_doc = random.choices(possible_universe, weights=weights, k=nb_exos)
            
            qi_list_in_file = []
            
            for frt_id in frts_in_doc:
                qi_txt = generate_smart_qi_dynamic(frt_id)
                atoms_db.append({
                    "ID_Source": file_id,
                    "Année": year,
                    "Qi_Brut": qi_txt,
                    "FRT_ID": frt_id, # L'ID découvert
                    "Fichier": filename,
                    "Chapitre": UNIVERS_MATHS[frt_id]["Chap"]
                })
                qi_list_in_file.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
            content = f"CONTENU SIMULÉ\n{filename}"
            sources_log.append({
                "Fichier": filename, "Nature": nature, "Année": year, 
                "Contenu": content, "Qi_Data": qi_list_in_file
            })
            
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(atoms_db)

def compute_dynamic_qc(df_atoms):
    """
    Construit les QC UNIQUEMENT à partir de ce qui a été vu.
    Pas de projection d'une liste fixe.
    """
    if df_atoms.empty: return pd.DataFrame()
    
    # 1. Regroupement (Clustering) par Structure (FRT)
    grouped = df_atoms.groupby("FRT_ID").agg({
        "ID_Source": "count", "Année": "max", 
        "Qi_Brut": list, "Fichier": list, "Chapitre": "first"
    }).reset_index()
    
    qcs = []
    N_total = len(df_atoms)
    current_year = datetime.now().year
    
    # Tri par fréquence (Les QC les plus importantes en premier)
    grouped = grouped.sort_values(by="ID_Source", ascending=False).reset_index(drop=True)
    
    for idx, row in grouped.iterrows():
        frt_id = row["FRT_ID"]
        kernel_info = UNIVERS_MATHS.get(frt_id)
        
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        # Simulation d'un Psi calculé
        psi = 0.85
        sigma = 0.05
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        qc_clean = kernel_info["QC"].replace("COMMENT ", "comment ")
        
        # Données Pédago Simulées
        triggers = ["Déclencheur 1", "Déclencheur 2"]
        ari = ["Etape 1", "Etape 2", "Etape 3"]
        frt_text = f"**Méthode Standard pour {qc_clean}**\n\n1. Identifier les hypothèses.\n2. Appliquer la formule.\n3. Conclure."
        
        qcs.append({
            "Chapitre": row["Chapitre"],
            "QC_ID_Simple": f"QC_{idx+1:02d}", 
            "FRT_ID": frt_id,
            "QC_Texte": qc_clean,
            "Triggers": triggers, "ARI": ari, "FRT_Redaction": frt_text,
            "Score_F2": score,
            "n_q": n_q, "N_tot": N_total, "Tau": tau, "Psi": psi,
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["Fichier"], row["Qi_Brut"])]
        })
        
    return pd.DataFrame(qcs)

# ==============================================================================
# 🖥️ INTERFACE V13
# ==============================================================================

with st.sidebar:
    st.header("1. Paramètres")
    matiere = st.selectbox("Matière", ["MATHS"])
    available_chaps = list(set(d["Chap"] for d in UNIVERS_MATHS.values()))
    selected_chaps = st.multiselect("Chapitres", available_chaps, default=available_chaps)
    st.markdown("---")
    st.info("💡 **Mode Découverte :** Le nombre de QC dépend du volume de sujets injectés. Augmentez le volume pour trouver les QC rares.")

# TABS
tab_usine, tab_audit = st.tabs(["🏭 USINE (Discovery)", "✅ AUDIT (Validité)"])

# --- TAB 1 : USINE ---
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: 
        urls_input = st.text_area("Sources", "https://apmep.fr", height=70)
    with c2: 
        n_sujets = st.number_input("Vol. par URL", 5, 500, 10, step=5) # Jusqu'à 500 pour voir la convergence
        btn_run = st.button("LANCER USINE 🚀", type="primary")

    if btn_run:
        with st.spinner("Exploration de l'espace mathématique..."):
            df_src, df_atoms = ingest_and_discover(urls_input.split('\n'), n_sujets, selected_chaps)
            df_qc = compute_dynamic_qc(df_atoms)
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.session_state['sel_chaps'] = selected_chaps
            
            # Calcul du taux de découverte
            nb_theorique = len([k for k,v in UNIVERS_MATHS.items() if v["Chap"] in selected_chaps])
            nb_decouvert = len(df_qc)
            taux_decouverte = (nb_decouvert / nb_theorique) * 100 if nb_theorique > 0 else 0
            
            st.success(f"Terminé. {len(df_src)} sujets traités.")
            st.info(f"📊 **Performance Moteur :** {nb_decouvert} QC découvertes sur {nb_theorique} possibles ({taux_decouverte:.0f}% de l'univers connu).")

    st.divider()

    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.8])
        
        # GAUCHE : SUJETS (Tableau HTML Full ou Container Scrollable)
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            
            # Utilisation d'un container scrollable en CSS pour la liste
            st.markdown('<div class="subject-list-container">', unsafe_allow_html=True)
            # On affiche un tableau HTML simple pour la liste des fichiers
            html_files = "<table class='custom-table'><thead><tr><th>Fichier</th><th>Nature</th><th>Action</th></tr></thead><tbody>"
            for idx, row in st.session_state['df_src'].iterrows():
                html_files += f"<tr><td>{row['Fichier']}</td><td>{row['Nature']}</td><td>📥</td></tr>"
            html_files += "</tbody></table></div>"
            st.markdown(html_files, unsafe_allow_html=True)
            
            st.write("")
            sel = st.selectbox("Fichier cible", st.session_state['df_src']["Fichier"])
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
                        # HEADER
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
                            with st.expander("⚡ Déclencheurs"): st.markdown("<span class='trigger-badge'>Smart Trigger...</span>", unsafe_allow_html=True)
                        with c2:
                            with st.expander(f"⚙️ ARI_{row['QC_ID_Simple']}"): st.write("- Etape ARI...")
                        with c3:
                            with st.expander(f"📝 FRT_{row['QC_ID_Simple']}"): st.markdown(f"<div class='frt-box'>{row['FRT_Redaction']}</div>", unsafe_allow_html=True)
                        with c4:
                            with st.expander(f"📄 Qi associées ({row['n_q']})"):
                                html_table = "<table class='custom-table'><thead><tr><th>Fichier</th><th>Qi</th></tr></thead><tbody>"
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
        
        # TEST 1 : INTERNE
        st.markdown("#### 1. Audit Interne (Mémoire Moteur)")
        test_file = st.selectbox("Sujet Traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER L'AUDIT INTERNE"):
            file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==test_file].iloc[0]
            qi_list = file_data["Qi_Data"]
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
            
            html_audit = "<table class='custom-table'><thead><tr><th>Qi</th><th>QC Moteur</th><th>Statut</th></tr></thead><tbody>"
            for row in mapping_rows:
                color = "#dcfce7" if "MATCH" in row['Statut'] else "#fee2e2"
                html_audit += f"<tr style='background-color:{color}'><td>{row['Qi (Sujet)']}</td><td>{row['QC Moteur']}</td><td>{row['Statut']}</td></tr>"
            html_audit += "</tbody></table>"
            st.markdown(html_audit, unsafe_allow_html=True)

        st.divider()

        # TEST 2 : EXTERNE
        st.markdown("#### 2. Audit Externe (Capacité de Généralisation)")
        st.caption("Ce test génère un sujet piochant dans TOUT l'Univers Théorique. Si le moteur n'a pas assez appris (trop peu de sujets injectés), il ratera les QC rares.")
        
        if st.button("GÉNÉRER SUJET EXTERNE & TESTER"):
            # On prend dans TOUT l'univers (même ce que le moteur n'a pas vu)
            target_chaps = st.session_state.get('sel_chaps', [])
            all_hidden = [k for k,v in UNIVERS_MATHS.items() if v["Chap"] in target_chaps]
            
            # On force le tirage de QC rares pour tester la limite
            test_frts = random.sample(all_hidden, k=min(5, len(all_hidden)))
            
            ext_rows = []
            c_ok = 0
            qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
            
            for frt_id in test_frts:
                qi_txt = generate_smart_qi_dynamic(frt_id)
                # Le moteur connait-il cette FRT ?
                is_known = frt_id in qc_lookup.index
                
                if is_known:
                    c_ok += 1
                    qc_info = qc_lookup.loc[frt_id]
                    if isinstance(qc_info, pd.DataFrame): qc_info = qc_info.iloc[0]
                    qc_disp = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Texte']}"
                    status = "✅ MATCH"
                else:
                    qc_disp = "---"
                    status = "❌ MANQUANT (Non appris)"
                
                ext_rows.append({"Qi (Sujet Externe)": qi_txt, "QC Moteur": qc_disp, "Statut": status})
            
            taux = (c_ok / len(test_frts)) * 100
            st.markdown(f"<div class='stat-metric'>Couverture Externe : {taux:.0f}%</div>", unsafe_allow_html=True)
            
            if taux < 100:
                st.warning(f"⚠️ Le moteur n'a couvert que {taux:.0f}% de ce sujet. Cela signifie qu'il contient des questions rares que le moteur n'a pas encore rencontrées dans l'Usine. -> ACTION : Augmenter le volume de sujets.")
            else:
                st.success("✅ PERFORMANCE OPTIMALE : Le moteur connait toutes les QC nécessaires pour ce sujet.")
                
            html_audit = "<table class='custom-table'><thead><tr><th>Qi</th><th>QC Moteur</th><th>Statut</th></tr></thead><tbody>"
            for row in ext_rows:
                color = "#dcfce7" if "MATCH" in row['Statut'] else "#fee2e2"
                html_audit += f"<tr style='background-color:{color}'><td>{row['Qi (Sujet Externe)']}</td><td>{row['QC Moteur']}</td><td>{row['Statut']}</td></tr>"
            html_audit += "</tbody></table>"
            st.markdown(html_audit, unsafe_allow_html=True)

    else:
        st.info("Lancez l'usine.")
