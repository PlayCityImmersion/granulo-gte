import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V10.7")
st.title("🛡️ SMAXIA - Console V10.7 (Full Height)")

# Styles CSS (Optimisé pour Full Height sans scroll interne)
st.markdown("""
<style>
    .qc-header-row { 
        display: flex; align-items: center; background-color: #f8f9fa; 
        padding: 10px; border-radius: 5px; border-left: 5px solid #2563eb;
        font-family: monospace; margin-bottom: 5px; flex-wrap: wrap;
    }
    .qc-id-tag { font-weight: bold; color: #d97706; margin-right: 10px; font-size: 1.2em; min-width: 80px;}
    .qc-title { flex-grow: 1; font-weight: 600; color: #1f2937; font-size: 1.1em; margin-right: 15px;}
    .qc-vars { 
        font-size: 0.9em; color: #111827; font-weight: bold; font-family: 'Courier New';
        background-color: #e5e7eb; padding: 6px 12px; border-radius: 4px; border: 1px solid #9ca3af;
        white-space: nowrap;
    }
    /* FRT Full Height */
    .frt-box { 
        background-color: #ffffff; border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; 
        height: auto !important; overflow: visible !important;
    }
    .trigger-badge { 
        background-color: #fef3c7; color: #92400e; padding: 4px 10px; 
        border-radius: 6px; font-size: 0.95em; font-weight: 600; 
        border: 1px solid #fcd34d; display: block; margin-bottom: 4px;
        line-height: 1.4; white-space: normal;
    }
    .stat-metric { font-size: 1.5em; font-weight: bold; color: #2563eb; }
    
    /* Force Table Full Height */
    [data-testid="stDataFrame"] > div { height: auto !important; max-height: none !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧱 KERNEL SMAXIA
# ==============================================================================

KERNEL_MAPPING = {
    "FRT_SUITE_01": "SUITES NUMÉRIQUES", "FRT_SUITE_02": "SUITES NUMÉRIQUES", "FRT_SUITE_03": "SUITES NUMÉRIQUES",
    "FRT_FCT_01": "FONCTIONS", "FRT_FCT_02": "FONCTIONS", "FRT_FCT_03": "FONCTIONS",
    "FRT_GEO_01": "GÉOMÉTRIE", "FRT_GEO_02": "GÉOMÉTRIE",
    "FRT_PROBA_01": "PROBABILITÉS", "FRT_PROBA_02": "PROBABILITÉS"
}

SMAXIA_KERNEL = {
    # --- SUITES NUMÉRIQUES ---
    "FRT_SUITE_01": {
        "QC": "COMMENT Démontrer qu'une suite est géométrique ?",
        "Triggers": ["Relation de récurrence multiplicative $u_{n+1} = f(u_n)$", "Demande explicite sur la nature de la suite définie par un produit"],
        "FRT_Redaction": """
        **🔔 Quand utiliser cette méthode ?**
        Lorsque l'énoncé demande la nature de la suite et que l'expression lie $u_{n+1}$ à $u_n$ par un facteur multiplicatif.

        **✅ Méthode Standard :**
        1.  **Exprimer** le rapport $\\frac{u_{n+1}}{u_n}$ pour tout entier $n$.
        2.  **Remplacer** $u_{n+1}$ par sa définition en fonction de $u_n$.
        3.  **Simplifier** l'expression algébrique jusqu'à éliminer tous les termes en $u_n$ ou $n$.
        4.  **Identifier** le résultat obtenu comme une constante réelle $q$.

        **⚠️ Pièges à éviter :**
        * Ne pas vérifier que $u_n \\neq 0$ avant de diviser.
        * Confondre avec une suite arithmétique (différence constante).

        **✍️ Modèle de Rédaction Examen :**
        > "Pour tout entier naturel $n$, on calcule le rapport $\\frac{u_{n+1}}{u_n} = \\dots = q$. Ce rapport étant constant, la suite $(u_n)$ est géométrique de raison $q$."
        """,
        "ARI": ["Calcul Ratio", "Simplification", "Identification Constante"],
        "Weights": [0.2, 0.3, 0.2, 0.1], "Delta": 1.2
    },

    "FRT_SUITE_02": {
        "QC": "COMMENT Lever une indétermination sur une limite ?",
        "Triggers": ["Présence simultanée de plusieurs termes en $n$ de même ordre (polynômes ou fractions) créant un conflit à l'infini"],
        "FRT_Redaction": """
        **🔔 Quand utiliser cette méthode ?**
        L'expression de $u_n$ contient plusieurs termes en $n$ qui "s'affrontent" (forme $\\infty - \\infty$ ou $\\frac{\\infty}{\\infty}$).

        **✅ Méthode Standard :**
        1.  **Identifier** le terme dominant (plus grande puissance de $n$ ou exponentielle).
        2.  **Factoriser** toute l'expression par ce terme dominant (force brute).
            * $u_n = n^k \\times (\\dots)$
        3.  **Simplifier** les termes intérieurs en utilisant les limites usuelles (ex: $1/n \\to 0$).
        4.  **Conclure** par produit ou somme de limites.

        **⚠️ Pièges à éviter :**
        * Utiliser la règle des signes sans factoriser.
        * Oublier de justifier les limites usuelles ($1/n$).

        **✍️ Modèle de Rédaction Examen :**
        > "On factorise par le terme de plus haut degré : $u_n = n^k(\\dots)$. Or $\\lim \\frac{1}{n} = 0$, donc par produit, $\\lim u_n = \\dots$"
        """,
        "ARI": ["Identifier Dominant", "Factorisation Forcée", "Limites Usuelles"],
        "Weights": [0.2, 0.3, 0.3, 0.2], "Delta": 1.1
    },

    "FRT_SUITE_03": {
        "QC": "COMMENT Démontrer par récurrence ?",
        "Triggers": ["Propriété $P(n)$ dépendant de $n$ à valider pour tout entier naturel (souvent inégalité ou divisibilité)"],
        "FRT_Redaction": """
        **🔔 Quand utiliser cette méthode ?**
        Dès que l'énoncé contient "pour tout entier naturel $n$" et une propriété qui se propage (inégalité, suite définie par récurrence).

        **✅ Méthode Standard :**
        1.  **Initialisation :** Vérifier que la propriété est vraie au premier rang (souvent $n=0$ ou $n=1$).
        2.  **Hérédité :**
            * Supposer la propriété vraie au rang $k$ (Hypothèse de Récurrence - HR).
            * Démontrer qu'elle est vraie au rang $k+1$ en utilisant **explicitement** l'HR.
        3.  **Conclusion :** Rappeler le principe de récurrence.

        **⚠️ Pièges à éviter :**
        * Oublier l'initialisation.
        * Ne pas utiliser l'hypothèse de récurrence dans l'hérédité (signe que la démonstration est fausse).

        **✍️ Modèle de Rédaction Examen :**
        > "Initialisation : pour $n=0$... Hérédité : Supposons $P(k)$ vraie. Montrons $P(k+1)$. On a... (utilisation HR)... Donc $P(k+1)$ est vraie. Conclusion : Par récurrence, la propriété est vraie pour tout $n$."
        """,
        "ARI": ["Initialisation", "Hérédité", "Conclusion"],
        "Weights": [0.1, 0.2, 0.6, 0.1], "Delta": 1.5
    },

    # --- FONCTIONS ---
    "FRT_FCT_01": {
        "QC": "COMMENT Étudier les variations d'une fonction ?",
        "Triggers": ["Demande explicite du sens de variation", "Nécessité de dresser le tableau de variations"],
        "FRT_Redaction": """
        **🔔 Quand utiliser cette méthode ?**
        Systématiquement quand on veut connaître la croissance/décroissance d'une fonction dérivable.

        **✅ Méthode Standard :**
        1.  **Justifier** la dérivabilité sur l'intervalle.
        2.  **Calculer** la dérivée $f'(x)$.
        3.  **Étudier le signe** de $f'(x)$ (factorisation, racines, tableau de signes).
        4.  **Conclure :**
            * $f'(x) > 0 \\Rightarrow f$ croissante.
            * $f'(x) < 0 \\Rightarrow f$ décroissante.

        **⚠️ Pièges à éviter :**
        * Confondre le signe de $f(x)$ et le signe de $f'(x)$.
        * Oublier les valeurs interdites dans le tableau.

        **✍️ Modèle de Rédaction Examen :**
        > "$f$ est dérivable sur $I$. Pour tout $x$, $f'(x) = \\dots$. Comme $f'(x) > 0$ sur cet intervalle, la fonction $f$ est strictement croissante."
        """,
        "ARI": ["Dérivabilité", "Calcul f'", "Signe f'", "Conclusion"],
        "Weights": [0.3, 0.3, 0.2, 0.1], "Delta": 1.3
    },
    "FRT_FCT_02": {
        "QC": "COMMENT Appliquer le TVI (Solution unique) ?",
        "Triggers": ["Montrer que l'équation $f(x)=k$ admet une unique solution", "Encadrement d'une solution alpha"],
        "FRT_Redaction": """
        **🔔 Quand utiliser cette méthode ?**
        Pour prouver l'existence et l'unicité d'une solution sans pouvoir la calculer explicitement.

        **✅ Méthode Standard :**
        1.  Vérifier la **Continuité** de $f$ sur l'intervalle.
        2.  Vérifier la **Stricte Monotonie** (strictement croissante ou décroissante).
        3.  Calculer les **Images aux bornes** (ou limites) pour montrer que la valeur cible $k$ est atteinte.
        4.  Invoquer le **Corollaire du TVI**.

        **⚠️ Pièges à éviter :**
        * Oublier la condition "stricte monotonie" (nécessaire pour l'unicité).
        * Oublier la condition "continuité" (nécessaire pour l'existence).

        **✍️ Modèle de Rédaction Examen :**
        > "La fonction est continue et strictement monotone sur $I$. Or $k$ est compris entre les images des bornes. D'après le corollaire du TVI, l'équation admet une unique solution $\\alpha$."
        """,
        "ARI": ["Continuité", "Monotonie", "Images Bornes", "Invocation"],
        "Weights": [0.1, 0.2, 0.2, 0.4], "Delta": 1.4
    },
    "FRT_FCT_03": {
        "QC": "COMMENT Déterminer l'équation d'une tangente ?",
        "Triggers": ["Déterminer l'équation de la tangente au point d'abscisse $a$", "Équation réduite de la tangente"],
        "FRT_Redaction": """
        **🔔 Quand utiliser cette méthode ?**
        Dès que le mot "tangente" apparaît avec un point de contact donné.

        **✅ Méthode Standard :**
        1.  **Identifier** l'abscisse $a$ du point de contact.
        2.  **Calculer** l'image $f(a)$.
        3.  **Calculer** le nombre dérivé $f'(a)$.
        4.  **Appliquer** la formule : $y = f'(a)(x-a) + f(a)$.

        **⚠️ Pièges à éviter :**
        * Confondre $f(a)$ et $f'(a)$.
        * Laisser l'expression non réduite (il faut la forme $y=mx+p$).

        **✍️ Modèle de Rédaction Examen :**
        > "L'équation de la tangente $T$ au point d'abscisse $a$ est donnée par $y = f'(a)(x-a) + f(a)$. On a $f(a)=...$ et $f'(a)=...$, d'où $y = ...$"
        """,
        "ARI": ["Formule", "Calcul f(a)", "Calcul f'(a)", "Substitution"],
        "Weights": [0.1, 0.2, 0.2, 0.1], "Delta": 0.9
    },

    # --- GEOMETRIE & PROBA ---
    "FRT_GEO_01": {
        "QC": "COMMENT Démontrer l'orthogonalité Droite/Plan ?",
        "Triggers": ["Montrer que la droite (d) est orthogonale au plan (P)"],
        "FRT_Redaction": """
        **Méthode :**
        1. Extraire vecteur $\\vec{u}$ de la droite.
        2. Extraire deux vecteurs non colinéaires $\\vec{v_1}, \\vec{v_2}$ du plan.
        3. Montrer que $\\vec{u}.\\vec{v_1}=0$ et $\\vec{u}.\\vec{v_2}=0$.
        """,
        "ARI": ["Vecteur u", "Base Plan", "Produits Scalaires"],
        "Weights": [0.1, 0.1, 0.4, 0.2], "Delta": 1.3
    },
    "FRT_GEO_02": {
        "QC": "COMMENT Déterminer une représentation paramétrique ?",
        "Triggers": ["Donner une représentation paramétrique de la droite passant par A et de vecteur u"],
        "FRT_Redaction": "**Méthode :** Utiliser la condition $\\vec{AM} = t\\vec{u}$ pour écrire le système.",
        "ARI": ["Point A", "Vecteur u", "Système"],
        "Weights": [0.2, 0.2, 0.4], "Delta": 1.0
    },
    "FRT_PROBA_01": {
        "QC": "COMMENT Calculer une probabilité totale ?",
        "Triggers": ["Calculer P(B) dans une expérience à plusieurs étapes (Arbre)"],
        "FRT_Redaction": "**Méthode :** Identifier les chemins de l'arbre et sommer les probabilités.",
        "ARI": ["Arbre", "Chemins", "Somme"],
        "Weights": [0.1, 0.3, 0.2, 0.2], "Delta": 1.1
    },
    "FRT_PROBA_02": {
        "QC": "COMMENT Utiliser la Loi Binomiale ?",
        "Triggers": ["Calculer la probabilité d'obtenir exactement k succès"],
        "FRT_Redaction": "**Méthode :** Justifier Bernoulli, donner (n,p), appliquer formule.",
        "ARI": ["Justification", "Paramètres", "Formule"],
        "Weights": [0.3, 0.1, 0.3, 0.1], "Delta": 1.2
    }
}

# --- GÉNÉRATEUR ---
QI_TEMPLATES = {
    "FRT_SUITE_01": ["Montrer que (Un) est géométrique.", "Démontrer que Vn est une suite géométrique.", "Justifier la nature géométrique de la suite."],
    "FRT_SUITE_02": ["Déterminer la limite de la suite Un.", "Calculer la limite quand n tend vers l'infini (forme indéterminée).", "Étudier la convergence de Un = (n^2+1)/(n-3)."],
    "FRT_SUITE_03": ["Démontrer par récurrence que Un > 0.", "Montrer par récurrence la propriété P(n).", "Prouver par récurrence que Un < 5."],
    "FRT_FCT_01": ["Étudier les variations de la fonction f.", "Dresser le tableau de variations complet.", "Quel est le sens de variation de f ?"],
    "FRT_FCT_02": ["Montrer que l'équation f(x)=0 admet une unique solution.", "Démontrer l'existence d'une solution alpha sur [0;1]."],
    "FRT_FCT_03": ["Déterminer l'équation de la tangente T au point A.", "Donner l'équation réduite de la tangente en 0."],
    "FRT_GEO_01": ["Démontrer que la droite (d) est orthogonale au plan (P)."],
    "FRT_GEO_02": ["Déterminer une représentation paramétrique de (D)."],
    "FRT_PROBA_01": ["Calculer la probabilité de l'événement B (Total)."],
    "FRT_PROBA_02": ["Calculer la probabilité d'obtenir exactement 3 succès."]
}

def generate_smart_qi(frt_id):
    if frt_id not in QI_TEMPLATES: return "Question Standard"
    text = random.choice(QI_TEMPLATES[frt_id])
    context = random.choice(["", " sur l'intervalle I", " dans le repère Oijk", " pour tout entier n"])
    return text + context

# ==============================================================================
# ⚙️ MOTEUR
# ==============================================================================

def calculate_psi_real(frt_id):
    data = SMAXIA_KERNEL[frt_id]
    psi_raw = data["Delta"] * (0.1 + sum(data["Weights"]))**2
    return round(min(psi_raw / 4.0, 1.0), 4)

def ingest_and_process(urls, n_per_url, selected_chapters):
    sources_log = []
    atoms_db = []
    progress = st.progress(0)
    
    active_frts = [k for k, v in KERNEL_MAPPING.items() if v in selected_chapters]
    if not active_frts: return pd.DataFrame(), pd.DataFrame()

    total_ops = len(urls) * n_per_url if len(urls) > 0 else 1
    counter = 0
    natures = ["BAC", "DST", "CONCOURS"]
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.002)
            
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            filename = f"Sujet_{nature}_{year}_{j}.pdf"
            file_id = f"DOC_{i}_{j}"
            
            nb_exos = random.randint(2, 4)
            frts_in_doc = random.sample(active_frts, k=min(nb_exos, len(active_frts)))
            
            qi_list_in_file = []
            
            for frt_id in frts_in_doc:
                qi_txt = generate_smart_qi(frt_id)
                atoms_db.append({
                    "ID_Source": file_id, "Année": year, "Qi_Brut": qi_txt,
                    "FRT_ID": frt_id, "Fichier": filename, 
                    "Chapitre": KERNEL_MAPPING[frt_id]
                })
                qi_list_in_file.append({"Qi": qi_txt, "FRT_ID": frt_id})
                
            content = f"CONTENU SIMULÉ PDF\nFICHIER: {filename}\n" + "\n".join([f"- {q['Qi']}" for q in qi_list_in_file])
            sources_log.append({
                "Fichier": filename, "Nature": nature, "Année": year, 
                "Contenu": content, "Qi_Data": qi_list_in_file 
            })
            
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(atoms_db)

def compute_engine_metrics(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    grouped = df_atoms.groupby("FRT_ID").agg({
        "ID_Source": "count", "Année": "max", 
        "Qi_Brut": list, "Fichier": list, "Chapitre": "first"
    }).reset_index()
    
    qcs = []
    N_total = len(df_atoms)
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        frt_id = row["FRT_ID"]
        kernel = SMAXIA_KERNEL[frt_id]
        
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = calculate_psi_real(frt_id)
        sigma = 0.05
        
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        qc_clean = kernel["QC"].replace("COMMENT ", "comment ")
        
        qcs.append({
            "Chapitre": row["Chapitre"],
            "QC_ID_Simple": f"QC_{idx+1:02d}", 
            "FRT_ID": frt_id,
            "QC_Texte": qc_clean,
            "Triggers": kernel["Triggers"],
            "FRT_Redaction": kernel["FRT_Redaction"],
            "ARI": kernel["ARI"],
            "Score_F2": score,
            "n_q": n_q, "N_tot": N_total, "Tau": tau, "Alpha": alpha, "Psi": psi, "Sigma": sigma,
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["Fichier"], row["Qi_Brut"])]
        })
        
    return pd.DataFrame(qcs).sort_values(by=["Chapitre", "Score_F2"], ascending=[True, False])

# ==============================================================================
# 🖥️ INTERFACE V10.7
# ==============================================================================

with st.sidebar:
    st.header("1. Paramètres")
    matiere = st.selectbox("Matière", ["MATHS"])
    all_chaps = list(set(KERNEL_MAPPING.values()))
    selected_chaps = st.multiselect("Chapitres", all_chaps, default=all_chaps)
    st.info(f"ℹ️ Kernel Actuel : {len(SMAXIA_KERNEL)} FRT définies.")

# TABS
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine (Prod)", "✅ Onglet 2 : Audit (Validation Booléenne)"])

# --- TAB 1 : USINE ---
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: urls_input = st.text_area("Sources", "https://apmep.fr", height=70)
    with c2: 
        n_sujets = st.number_input("Vol. par URL", 5, 100, 10, step=5)
        btn_run = st.button("LANCER USINE 🚀", type="primary")

    if btn_run:
        with st.spinner("Traitement..."):
            df_src, df_atoms = ingest_and_process(urls_input.split('\n'), n_sujets, selected_chaps)
            df_qc = compute_engine_metrics(df_atoms)
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.session_state['sel_chaps'] = selected_chaps
            st.success("Traitement terminé.")

    st.divider()

    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.8])
        
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            df_display = st.session_state['df_src'][["Fichier", "Nature", "Année"]].copy()
            df_display["Action"] = "📥 PDF" 
            st.dataframe(df_display, use_container_width=True, height=500)
            
            st.caption("Sélectionner pour télécharger :")
            sel = st.selectbox("Fichier cible", st.session_state['df_src']["Fichier"], label_visibility="collapsed")
            if not st.session_state['df_src'].empty:
                txt = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==sel].iloc[0]["Contenu"]
                st.download_button(f"Télécharger {sel}", txt, file_name=sel)

        with col_right:
            if not st.session_state['df_qc'].empty:
                chapters = st.session_state['df_qc']["Chapitre"].unique()
                for chap in chapters:
                    df_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                    st.markdown(f"#### 📘 {chap} : {len(df_view)} QC")
                    
                    for idx, row in df_view.iterrows():
                        # HEADER
                        header_html = f"""
                        <div class="qc-header-row">
                            <span class="qc-id-tag">[{row['QC_ID_Simple']}]</span>
                            <span class="qc-title">{row['QC_Texte']}</span>
                            <span class="qc-vars">
                                Score(q)={row['Score_F2']:.0f} | 
                                Ψ={row['Psi']} | 
                                n_q={row['n_q']} | 
                                N_tot={row['N_tot']} | 
                                t_rec={row['Tau']:.1f}
                            </span>
                        </div>
                        """
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        
                        with c1:
                            with st.expander("⚡ Déclencheurs"):
                                for t in row['Triggers']: 
                                    st.markdown(f"<div class='trigger-badge'>{t}</div>", unsafe_allow_html=True)
                        
                        with c2:
                            with st.expander(f"⚙️ ARI_{row['QC_ID_Simple']}"):
                                for s in row['ARI']: st.markdown(f"- {s}")

                        with c3:
                            with st.expander(f"📝 FRT_{row['QC_ID_Simple']}"):
                                st.markdown(f"<div class='frt-box'>{row['FRT_Redaction']}</div>", unsafe_allow_html=True)

                        with c4:
                            with st.expander(f"📄 Qi associées ({row['n_q']})"):
                                st.dataframe(
                                    pd.DataFrame(row['Evidence']), 
                                    hide_index=True, 
                                    use_container_width=True, 
                                    column_config={"Qi": st.column_config.TextColumn("Questions Élèves", width="large")}
                                )
                        
                        st.write("") 
            else:
                st.warning("Aucune QC.")

# --- TAB 2 : AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne (Tableau de Mapping Unifié)")
    
    if 'df_qc' in st.session_state and 'df_src' in st.session_state:
        
        # TEST 1
        st.markdown("#### 1. Audit Interne (Sujet Traité)")
        test_file = st.selectbox("Choisir un sujet traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER L'AUDIT DE COUVERTURE (INTERNE)", type="primary"):
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
                    qc_display = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Texte']}"
                    frt_display = f"FRT_{qc_info['QC_ID_Simple']}"
                    status = "✅ MATCH"
                else:
                    qc_display = "---"
                    frt_display = "---"
                    status = "❌ GAP"
                
                mapping_rows.append({
                    "1. Qi (Question Élève)": item["Qi"],
                    "2. QC (ID + Titre)": qc_display,
                    "3. FRT (ID Technique)": frt_display,
                    "Statut": status
                })
            
            taux = (c_ok / len(qi_list)) * 100
            st.markdown(f"<div class='stat-metric'>Taux de Couverture : {taux:.0f}%</div>", unsafe_allow_html=True)
            
            def color_map(val): return f'background-color: {"#dcfce7" if val == "✅ MATCH" else "#fee2e2"}; color: black'
            st.dataframe(pd.DataFrame(mapping_rows).style.map(color_map, subset=['Statut']), use_container_width=True)

        st.divider()

        # TEST 2
        st.markdown("#### 2. Audit Externe (Nouveau Sujet)")
        if st.button("LANCER L'AUDIT DE COUVERTURE (EXTERNE)"):
            all_kernel_frts = list(SMAXIA_KERNEL.keys())
            test_frts = random.sample(all_kernel_frts, 4)
            ext_rows = []
            c_ok_ext = 0
            qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
            
            for frt_id in test_frts:
                qi_txt = generate_smart_qi(frt_id)
                is_covered = frt_id in qc_lookup.index
                if is_covered:
                    c_ok_ext += 1
                    qc_info = qc_lookup.loc[frt_id]
                    if isinstance(qc_info, pd.DataFrame): qc_info = qc_info.iloc[0]
                    qc_display = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Texte']}"
                    status = "✅ MATCH"
                else:
                    qc_display = "---"
                    status = "❌ HORS PÉRIMÈTRE"
                ext_rows.append({"1. Qi (Nouveau Sujet)": qi_txt, "2. QC Trouvée": qc_display, "3. FRT Requise": frt_id, "Statut": status})
            
            taux_ext = (c_ok_ext / 4) * 100
            st.markdown(f"<div class='stat-metric'>Taux de Couverture : {taux_ext:.0f}%</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(ext_rows).style.map(color_map, subset=['Statut']), use_container_width=True)

    else:
        st.info("Veuillez lancer l'usine dans l'onglet 1.")
