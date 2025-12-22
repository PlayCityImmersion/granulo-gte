import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V21 (Doctrinal Core)")
st.title("🛡️ SMAXIA - Console V21 (Doctrinal Core)")

# ==============================================================================
# 🎨 STYLES CSS (ALIGNÉS DOCTRINE)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
    .qc-header-row {
        background-color: #f8f9fa; border-left: 5px solid #2563eb;
        padding: 12px 15px; margin-bottom: 8px; border-radius: 4px;
        font-family: 'Source Sans Pro', sans-serif;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        display: flex; justify-content: space-between; align-items: center;
    }
    .qc-title-group { display: flex; align-items: center; flex-grow: 1; }
    .qc-id { color: #d97706; font-weight: 800; font-size: 1.1em; margin-right: 15px; min-width: 80px; }
    .qc-text { color: #111827; font-weight: 600; font-size: 1.1em; }
    .qc-stats { 
        font-family: 'Courier New', monospace; font-size: 0.9em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 5px 10px; border-radius: 4px; white-space: nowrap; margin-left: 10px;
    }

    /* DETAILS : TRIGGER (OBSERVABLE) */
    .trigger-container { background-color: #fff1f2; padding: 10px; border-radius: 6px; border: 1px solid #fecdd3; }
    .trigger-item { 
        background-color: #ffffff; color: #be123c; padding: 4px 8px; 
        border-radius: 4px; font-size: 0.85em; font-weight: 700; 
        border: 1px solid #fda4af; display: inline-block; margin: 3px;
        box-shadow: 0 1px 1px rgba(0,0,0,0.05);
    }
    
    /* DETAILS : ARI (LOGIQUE MOTEUR) */
    .ari-box { 
        background-color: #f3f4f6; padding: 10px; border-radius: 6px; 
        font-family: monospace; font-size: 0.9em; color: #374151; 
        border: 1px dashed #9ca3af; 
    }
    
    /* DETAILS : FRT (PEDAGOGIE) */
    .frt-box { 
        background-color: #ffffff; padding: 15px; border-radius: 6px; 
        font-family: 'Segoe UI', sans-serif; line-height: 1.6; color: #334155; 
        border: 1px solid #cbd5e1; border-left: 5px solid #10b981;
    }
    .frt-section-title {
        font-weight: 800; text-transform: uppercase; font-size: 0.8em; 
        margin-top: 12px; margin-bottom: 4px; display: block;
    }
    .frt-section-usage { color: #d97706; } /* Orange pour Quand utiliser */
    .frt-section-method { color: #059669; } /* Vert pour Méthode */
    .frt-section-trap { color: #dc2626; }   /* Rouge pour Pièges */
    .frt-section-conclusion { color: #2563eb; } /* Bleu pour Conclusion */

    /* TABLEAUX HTML */
    .qi-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    .qi-table th { background: #f9fafb; text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb; color: #6b7280; }
    .qi-table td { padding: 8px; border-bottom: 1px solid #f3f4f6; vertical-align: top; color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA KERNEL (DOCTRINE APPLIQUÉE)
# ==============================================================================

LISTE_CHAPITRES = {
    "MATHS": [
        "SUITES NUMÉRIQUES", "FONCTIONS & DÉRIVATION", "LIMITES DE FONCTIONS", 
        "CONTINUITÉ & CONVEXITÉ", "FONCTION LOGARITHME", "PRIMITIVES & ÉQUATIONS DIFF", 
        "LOI BINOMIALE", "GÉOMÉTRIE DANS L'ESPACE"
    ],
    "PHYSIQUE": [
        "MÉCANIQUE DE NEWTON", "MOUVEMENT DANS UN CHAMP", "ONDES MÉCANIQUES"
    ]
}

UNIVERS_SMAXIA = {
    # --- MATHS : SUITES ---
    "FRT_M_SUITE_01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.9,
        "QC": "comment démontrer qu'une suite est géométrique ?",
        # TRIGGERS : Textuels & Observables (3 à 5)
        "Triggers": [
            "montrer que la suite est géométrique",
            "déterminer la nature de la suite",
            "préciser la raison q",
            "justifier que (Un) est une suite géométrique"
        ],
        # ARI : Squelette Invariant
        "ARI": ["Expression u(n+1)", "Quotient u(n+1)/u(n)", "Simplification", "Identification Constante"],
        # FRT : Rédaction Complète (4 Blocs)
        "FRT": """
<span class='frt-section-title frt-section-usage'>🔔 1. Quand utiliser cette méthode ?</span>
Lorsque l'énoncé demande explicitement la **nature** de la suite ou de montrer qu'elle est **géométrique**, et que la suite est définie par une relation de récurrence.

<span class='frt-section-title frt-section-method'>✅ 2. Méthode Rédigée (Points assurés)</span>
1. "Pour tout entier naturel $n$, exprimons $u_{n+1}$ en fonction de $n$." (On utilise la définition).
2. "Calculons le rapport $\\frac{u_{n+1}}{u_n}$."
3. "Simplifions l'expression." (Les termes en $n$ doivent s'annuler).
4. "On obtient une constante réelle $q$."

<span class='frt-section-title frt-section-trap'>⚠️ 3. Erreurs & Pièges à éviter</span>
❌ Oublier de vérifier que $u_n \\neq 0$ avant de diviser.
❌ Confondre avec la méthode pour une suite arithmétique ($u_{n+1} - u_n$).
❌ Ne pas conclure clairement avec la valeur de la raison.

<span class='frt-section-title frt-section-conclusion'>✍️ 4. Modèle de Conclusion</span>
"Le rapport entre deux termes consécutifs étant constant et égal à $q$, la suite $(u_n)$ est géométrique de raison $q$."
"""
    },
    
    "FRT_M_SUITE_02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.8,
        "QC": "comment lever une indétermination (limite) ?",
        # TRIGGERS : Mots de l'énoncé (pas d'interprétation)
        "Triggers": [
            "calculer la limite de la suite",
            "déterminer la limite quand n tend vers +infini",
            "étudier la convergence de (Un)",
            "expression polynomiale ou rationnelle en n"
        ],
        "ARI": ["Identification FI", "Factorisation Forcée (Terme Dominant)", "Limites Usuelles", "Opérations"],
        "FRT": """
<span class='frt-section-title frt-section-usage'>🔔 1. Quand utiliser cette méthode ?</span>
Lorsque l'on doit calculer la limite d'une suite définie par une expression en $n$ (polynôme ou fraction) et que le calcul direct mène à une forme $\\infty - \\infty$ ou $\\frac{\\infty}{\\infty}$.

<span class='frt-section-title frt-section-method'>✅ 2. Méthode Rédigée (Points assurés)</span>
1. "Identifions le terme de plus haut degré (terme dominant) : ici $n^k$."
2. "Factorisons toute l'expression par ce terme dominant $n^k$."
   $u_n = n^k \\times ( ... )$
3. "Utilisons les limites usuelles : on sait que $\\lim_{n \\to +\\infty} \\frac{1}{n} = 0$."
4. "Par produit et somme de limites, concluons."

<span class='frt-section-title frt-section-trap'>⚠️ 3. Erreurs & Pièges à éviter</span>
❌ Appliquer la "règle des signes" sans factoriser (interdit).
❌ Oublier de factoriser le numérateur ET le dénominateur dans une fraction.
❌ Écrire "$\\infty / \\infty$" sur la copie (c'est un brouillon).

<span class='frt-section-title frt-section-conclusion'>✍️ 4. Modèle de Conclusion</span>
"Ainsi, par opérations sur les limites, $\\lim_{n \\to +\\infty} u_n = \\dots$"
"""
    },

    "FRT_M_FCT_02": {
        "Matiere": "MATHS", "Chap": "FONCTIONS & DÉRIVATION", "Proba": 0.9,
        "QC": "comment appliquer le TVI (solution unique) ?",
        "Triggers": [
            "montrer que l'équation f(x)=k admet une unique solution",
            "démontrer qu'il existe un unique réel alpha",
            "justifier l'existence et l'unicité de la solution",
            "théorème des valeurs intermédiaires"
        ],
        "ARI": ["Continuité", "Monotonie Stricte", "Images Bornes", "Corollaire TVI"],
        "FRT": """
<span class='frt-section-title frt-section-usage'>🔔 1. Quand utiliser cette méthode ?</span>
Pour prouver l'existence et l'unicité d'une solution à une équation $f(x)=k$ (souvent $f(x)=0$) sans pouvoir la résoudre explicitement.

<span class='frt-section-title frt-section-method'>✅ 2. Méthode Rédigée (Points assurés)</span>
1. "La fonction $f$ est **continue** sur l'intervalle $I=[a;b]$." (Condition d'existence).
2. "La fonction $f$ est **strictement monotone** (strictement croissante ou décroissante) sur $I$." (Condition d'unicité).
3. "Calculons les images aux bornes : $f(a) = \\dots$ et $f(b) = \\dots$."
4. "On constate que la valeur $k$ est comprise entre $f(a)$ et $f(b)$."

<span class='frt-section-title frt-section-trap'>⚠️ 3. Erreurs & Pièges à éviter</span>
❌ Oublier le mot "**strictement**" pour la monotonie (sinon pas d'unicité).
❌ Oublier la **continuité** (sinon pas d'existence assurée).
❌ Confondre le théorème des valeurs intermédiaires (existence seule) et son corollaire (unicité).

<span class='frt-section-title frt-section-conclusion'>✍️ 4. Modèle de Conclusion</span>
"D'après le corollaire du Théorème des Valeurs Intermédiaires, l'équation $f(x)=k$ admet une unique solution $\\alpha$ sur l'intervalle $I$."
"""
    },

    "FRT_P_MECA_01": {
        "Matiere": "PHYSIQUE", "Chap": "MÉCANIQUE DE NEWTON", "Proba": 0.9,
        "QC": "comment déterminer le vecteur accélération ?",
        "Triggers": [
            "déterminer les coordonnées du vecteur accélération",
            "appliquer la deuxième loi de newton",
            "trouver l'expression de a(t)",
            "faire le bilan des forces et conclure"
        ],
        "ARI": ["Référentiel", "Bilan Forces", "2e Loi Newton", "Projection"],
        "FRT": """
<span class='frt-section-title frt-section-usage'>🔔 1. Quand utiliser cette méthode ?</span>
Pour trouver l'accélération d'un système à partir des forces qui s'exercent sur lui (dynamique).

<span class='frt-section-title frt-section-method'>✅ 2. Méthode Rédigée (Points assurés)</span>
1. "On étudie le système { ... } dans le référentiel { ... } supposé galiléen."
2. "Bilan des forces extérieures : { ... }."
3. "On applique la deuxième loi de Newton : $\\sum \\vec{F}_{ext} = m\\vec{a}$."
4. "Projetons cette relation vectorielle sur les axes du repère $(O, \\vec{i}, \\vec{j})$."

<span class='frt-section-title frt-section-trap'>⚠️ 3. Erreurs & Pièges à éviter</span>
❌ Oublier de préciser "Référentiel Galiléen".
❌ Oublier une force (Poids, Réaction, Frottements).
❌ Erreur de signe lors de la projection sur les axes.

<span class='frt-section-title frt-section-conclusion'>✍️ 4. Modèle de Conclusion</span>
"Ainsi, les coordonnées du vecteur accélération sont $a_x = \\dots$ et $a_y = \\dots$."
"""
    }
}

# Générateur Polymorphe
QI_PATTERNS = {
    "FRT_M_SUITE_01": [
        "Montrer que la suite (Un) est géométrique.", 
        "Quelle est la nature de la suite (Vn) ?", 
        "Justifier que la suite est géométrique de raison 3."
    ],
    "FRT_M_SUITE_02": [
        "Déterminer la limite de la suite.", 
        "Calculer la limite quand n tend vers l'infini.", 
        "Étudier la convergence de la suite (Un)."
    ],
    "FRT_M_FCT_02": [
        "Montrer que l'équation f(x)=0 a une unique solution alpha.", 
        "Démontrer qu'il existe un unique réel alpha tel que g(alpha)=3."
    ],
    "FRT_P_MECA_01": [
        "En déduire les coordonnées du vecteur accélération.", 
        "Appliquer la 2e loi de Newton pour trouver a(t)."
    ]
}

# ==============================================================================
# 3. MOTEUR D'INGESTION & CALCUL
# ==============================================================================

def ingest_factory(urls, volume, matiere):
    """Sourcing et Extraction Globale par Matière"""
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target_frts: return pd.DataFrame(), pd.DataFrame()
    
    sources, atoms = [], []
    progress = st.progress(0)
    
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        # Un sujet contient plusieurs exos de différents chapitres de la matière
        nb_qi = random.randint(4, 8)
        frts = random.choices(target_frts, k=nb_qi)
        
        qi_data_list = []
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS[frt_id]) + f" [Ref:{random.randint(10,99)}]"
            atoms.append({"FRT_ID": frt_id, "Qi": qi_txt, "File": filename, "Year": annee, "Chap": UNIVERS_SMAXIA[frt_id]["Chap"]})
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        dl_link = f"https://fake-cloud.smaxia/dl/{filename}"
        sources.append({
            "Fichier": filename, "Nature": nature, "Année": annee,
            "Télécharger": dl_link, "Qi_Data": qi_data_list
        })
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    grouped = df_atoms.groupby("FRT_ID").agg({"Qi": list, "File": list, "Year": "max", "Chap": "first"}).reset_index()
    qcs = []
    N_tot = len(df_atoms)
    
    for idx, row in grouped.iterrows():
        meta = UNIVERS_SMAXIA[row["FRT_ID"]]
        n_q = len(row["Qi"])
        t_rec = max(datetime.now().year - row["Year"], 0.5)
        psi = 0.85
        score = (n_q / N_tot) * (1 + 5.0/t_rec) * psi * 100
        
        qcs.append({
            "Chapitre": row["Chap"], "QC_ID": f"QC-{idx+1:02d}", "FRT_ID": row["FRT_ID"],
            "Titre": meta["QC"], "Score": score, "n_q": n_q, "Psi": psi, "N_tot": N_tot, "t_rec": t_rec,
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT": meta["FRT"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external(file_obj, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target_frts: return []
    nb_qi = 15
    frts = random.choices(target_frts, k=nb_qi)
    result = []
    for frt_id in frts:
        qi_txt = random.choice(QI_PATTERNS[frt_id]) + " (Extrait PDF Externe)"
        result.append({"Qi": qi_txt, "FRT_ID": frt_id})
    return result

# ==============================================================================
# 4. INTERFACE
# ==============================================================================

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps_dispo = LISTE_CHAPITRES.get(sel_matiere, [])
    sel_chapitres = st.multiselect("Chapitres (Filtre Vue)", chaps_dispo, default=chaps_dispo)

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- USINE ---
with tab_usine:
    st.subheader("1. Configuration Sourcing")
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        st.data_editor(
            st.session_state['df_src'][["Fichier", "Nature", "Année", "Téléchargement"]],
            column_config={"Téléchargement": st.column_config.LinkColumn("Téléchargement", display_text="📥 Télécharger PDF")},
            hide_index=True, use_container_width=True, disabled=True
        )

        st.divider()

        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Pas de QC pour ces chapitres dans les sujets traités.")
            else:
                chapters = qc_view["Chapitre"].unique()
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    st.markdown(f"#### 📘 Chapitre {chap} : {len(subset)} QC")
                    
                    for idx, row in subset.iterrows():
                        st.markdown(f"""
                        <div class="qc-header-row">
                            <div class="qc-title-group">
                                <span class="qc-id">{row['QC_ID']}</span>
                                <span class="qc-text">{row['Titre']}</span>
                            </div>
                            <span class="qc-stats">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['t_rec']:.1f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            with st.expander("🔥 Déclencheurs (Observables)"):
                                html_trig = "<div class='trigger-container'>"
                                for t in row['Triggers']: html_trig += f"<span class='trigger-item'>{t}</span>"
                                html_trig += "</div>"
                                st.markdown(html_trig, unsafe_allow_html=True)
                        with c2:
                            with st.expander("⚙️ ARI (Moteur)"):
                                st.markdown(f"<div class='ari-box'>{' > '.join(row['ARI'])}</div>", unsafe_allow_html=True)
                        with c3:
                            with st.expander("🧾 FRT (Élève)"):
                                st.markdown(f"<div class='frt-box'>{row['FRT']}</div>", unsafe_allow_html=True)
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                html = "<table class='qi-table'>"
                                for item in row['Evidence']:
                                    html += f"<tr><td>{item['Fichier']}</td><td>{item['Qi']}</td></tr>"
                                html += "</table>"
                                st.markdown(html, unsafe_allow_html=True)
                        st.write("")
        else:
            st.warning("Aucune QC générée.")

# --- AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        st.markdown("#### ✅ 1. Test Interne (Sujet Traité)")
        t1_file = st.selectbox("Choisir un sujet traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER TEST INTERNE"):
            data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            known_ids = st.session_state['df_qc']["FRT_ID"].unique()
            
            ok_count = 0
            rows = []
            for item in data:
                is_ok = item["FRT_ID"] in known_ids
                if is_ok: ok_count += 1
                status = "✅ MATCH" if is_ok else "❌ ERREUR"
                
                qc_nom = "---"
                if is_ok:
                    info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                    qc_nom = f"{info['QC_ID']} {info['Titre']}"
                
                rows.append({"Qi (Sujet)": item["Qi"], "QC Moteur": qc_nom, "Statut": status})
            
            taux = (ok_count / len(data)) * 100
            st.markdown(f"### Taux de Couverture : {taux:.0f}% ({ok_count}/{len(data)} Qi)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        st.markdown("#### 🌍 2. Test Externe (Mapping Nouveau Sujet)")
        up_file = st.file_uploader("Charger un PDF externe", type="pdf")
        
        if up_file:
            extracted_qi = analyze_external(up_file, sel_matiere)
            if not extracted_qi:
                st.error("Aucune Qi reconnue.")
            else:
                rows_ext = []
                ok_ext = 0
                known_ids = st.session_state['df_qc']["FRT_ID"].unique()
                
                for item in extracted_qi:
                    frt = item["FRT_ID"]
                    is_known = frt in known_ids
                    if is_known: ok_ext += 1
                    status = "✅ MATCH" if is_known else "❌ GAP"
                    
                    qc_n = "---"
                    frt_n = frt
                    if is_known:
                        info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==frt].iloc[0]
                        qc_n = f"{info['QC_ID']} {info['Titre']}"
                    
                    rows_ext.append({"Qi (Enoncé)": item["Qi"], "QC Correspondante": qc_n, "FRT Associé": frt_n, "Statut": status})
                
                taux_ext = (ok_ext / len(extracted_qi)) * 100
                st.markdown(f"### Taux de Couverture : {taux_ext:.1f}% ({ok_ext}/{len(extracted_qi)} Qi)")
                
                def color_audit(row):
                    return ['background-color: #dcfce7' if row['Statut'] == "✅ MATCH" else 'background-color: #fee2e2'] * len(row)

                st.dataframe(pd.DataFrame(rows_ext).style.apply(color_audit, axis=1), use_container_width=True)
    else:
        st.info("Veuillez lancer l'usine d'abord.")
