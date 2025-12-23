# smaxia_console_v31_ui.py
import streamlit as st
import pandas as pd
import numpy as np

from smaxia_granulo_engine_test import run_granulo_test

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(
    page_title="SMAXIA - Console V31 (Saturation Proof)",
    layout="wide"
)

st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# CSS – UI CONTRACTUELLE (INCHANGÉE)
# ==============================================================================
st.markdown("""
<style>
.qc-box {
    background:#f8fafc;
    border-left:6px solid #2563eb;
    padding:16px;
    border-radius:6px;
    margin-bottom:16px;
}
.qc-chap {
    font-size:0.85em;
    font-weight:800;
    color:#475569;
    text-transform:uppercase;
}
.qc-title {
    font-size:1.15em;
    font-weight:900;
    color:#0f172a;
    margin-top:6px;
}
.qc-meta {
    margin-top:8px;
    font-family:monospace;
    font-size:0.85em;
    background:#e5e7eb;
    padding:4px 8px;
    border-radius:4px;
    display:inline-block;
}

/* TRIGGERS */
.trigger {
    background:#fff1f2;
    border-left:4px solid #ef4444;
    padding:6px 10px;
    border-radius:4px;
    margin-bottom:6px;
    font-weight:600;
}

/* ARI */
.ari-step {
    background:#f3f4f6;
    border:1px dashed #cbd5f5;
    padding:6px;
    border-radius:4px;
    margin-bottom:5px;
    font-family:monospace;
}

/* FRT */
.frt {
    padding:12px;
    border-radius:6px;
    margin-bottom:8px;
    border-left:6px solid;
}
.frt-usage { background:#fff7ed; border-color:#f59e0b; }
.frt-method { background:#f0fdf4; border-color:#22c55e; }
.frt-trap { background:#fef2f2; border-color:#ef4444; }
.frt-conc { background:#eff6ff; border-color:#3b82f6; }

.frt-title {
    font-weight:900;
    font-size:0.75em;
    text-transform:uppercase;
    margin-bottom:6px;
}

/* QI */
.file-box {
    border:1px solid #e5e7eb;
    border-radius:6px;
    margin-bottom:10px;
}
.file-header {
    background:#f1f5f9;
    padding:8px 12px;
    font-weight:700;
}
.qi {
    padding:8px 12px;
    border-left:4px solid #8b5cf6;
    font-family:Georgia, serif;
}

/* SATURATION */
.sat-box {
    background:#f0f9ff;
    border:1px solid #bae6fd;
    padding:16px;
    border-radius:8px;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR – PARAMÈTRES ACADÉMIQUES (INCHANGÉE)
# ==============================================================================
with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    st.multiselect(
        "Chapitres",
        ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
        default=["SUITES NUMÉRIQUES"]
    )

# ==============================================================================
# TABS (INCHANGÉ)
# ==============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# ==============================================================================
# ONGLET 1 – USINE
# ==============================================================================
with tab_usine:

    # --------------------------------------------------------------------------
    # ZONE 1 – INJECTION DES SUJETS (INCHANGÉE)
    # --------------------------------------------------------------------------
    st.subheader("🧪 Injection des sujets")

    c1, c2 = st.columns([4, 1])
    with c1:
        urls = st.text_area(
            "URLs sources (références)",
            value="https://apmep.fr",
            height=80
        )
    with c2:
        volume = st.number_input(
            "Volume de sujets",
            min_value=5,
            max_value=500,
            value=15,
            step=5
        )

        launch = st.button("🚀 LANCER L’USINE")

    if launch:
        # exécution moteur
        url_list = [u.strip() for u in urls.split("\n") if u.strip()]
        with st.spinner("Granulo Test Engine : récupération PDFs → extraction Qi → clustering QC…"):
            result = run_granulo_test(url_list, int(volume))
        st.session_state["granulo_result"] = result

    # --------------------------------------------------------------------------
    # ZONE 2 – TABLEAU DES SUJETS TRAITÉS (ZÉRO HARDCODE)
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("📥 Sujets traités")

    if "granulo_result" not in st.session_state:
        st.caption("⚠️ Données affichées uniquement après branchement du moteur réel.")
        st.dataframe(pd.DataFrame(columns=["Fichier", "Nature", "Année", "Source"]), use_container_width=True)
    else:
        df_sujets = pd.DataFrame(st.session_state["granulo_result"]["sujets"])
        if df_sujets.empty:
            st.warning("Aucun sujet exploitable récupéré depuis ces URLs (0 PDF traité).")
            st.dataframe(pd.DataFrame(columns=["Fichier", "Nature", "Année", "Source"]), use_container_width=True)
        else:
            st.dataframe(df_sujets, use_container_width=True)
        st.caption(f"Audit moteur: {st.session_state['granulo_result']['audit']}")

    # --------------------------------------------------------------------------
    # ZONE 3 – BASE DE CONNAISSANCE (QC) (ZÉRO HARDCODE)
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("🧠 Base de connaissance (QC)")

    if "granulo_result" not in st.session_state:
        st.info("Aucune QC affichée tant que le moteur n’a pas produit de résultats.")
    else:
        qc_list = st.session_state["granulo_result"]["qc"]
        if not qc_list:
            st.warning("0 QC produite : soit 0 Qi extraite, soit filtrage trop strict (Suites).")
        else:
            # Afficher la 1ère QC (même layout que scellé)
            qc = qc_list[0]

            st.markdown(f"""
            <div class="qc-box">
                <div class="qc-chap">Chapitre : {qc['chapter']}</div>
                <div class="qc-title">{qc['qc_id']} : {qc['qc_title']}</div>
                <div class="qc-meta">
                    Score(q)={qc['score']} | n_q={qc['n_q']} | Ψ={qc['psi']} | N_tot={qc['n_tot']} | t_réc={qc['t_rec']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("### 🔥 Déclencheurs")
                if qc["triggers"]:
                    for t in qc["triggers"]:
                        st.markdown(f"<div class='trigger'>{t}</div>", unsafe_allow_html=True)
                else:
                    st.caption("Aucun déclencheur extrait.")

            with c2:
                st.markdown("### ⚙️ ARI")
                for s in qc["ari"]:
                    st.markdown(f"<div class='ari-step'>{s}</div>", unsafe_allow_html=True)

            with c3:
                st.markdown("### 📘 FRT")
                frt = qc["frt"]
                st.markdown(f"<div class='frt frt-usage'><div class='frt-title'>Quand utiliser</div>{frt['usage']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='frt frt-method'><div class='frt-title'>Méthode rédigée</div>{frt['method']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='frt frt-trap'><div class='frt-title'>Pièges</div>{frt['trap']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='frt frt-conc'><div class='frt-title'>Conclusion</div>{frt['conc']}</div>", unsafe_allow_html=True)

            with c4:
                st.markdown("### 📄 Qi associées")
                qi_map = qc["qi_by_file"]
                if not qi_map:
                    st.caption("Aucune Qi mappée.")
                else:
                    for f, qs in qi_map.items():
                        html = f"<div class='file-box'><div class='file-header'>{f}</div>"
                        for q in qs[:12]:
                            html += f"<div class='qi'>{q}</div>"
                        if len(qs) > 12:
                            html += f"<div class='qi'>… +{len(qs)-12} autres</div>"
                        html += "</div>"
                        st.markdown(html, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # ZONE 4 – COURBE DE SATURATION (ZÉRO HARDCODE)
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("📈 Analyse de saturation (preuve de complétude)")

    if "granulo_result" not in st.session_state:
        st.info("Courbe de saturation disponible après exécution du moteur.")
    else:
        sat = st.session_state["granulo_result"]["saturation"]
        df_sat = pd.DataFrame(sat)
        if df_sat.empty:
            st.warning("Aucune donnée de saturation (0 sujet traité).")
        else:
            st.line_chart(df_sat, x="Nombre de sujets injectés", y="Nombre de QC découvertes")
            st.dataframe(df_sat, use_container_width=True)

        # message de saturation uniquement si stabilité observée
        if len(df_sat) >= 5:
            tail = df_sat["Nombre de QC découvertes"].tail(5).tolist()
            if len(set(tail)) == 1:
                st.success("Seuil de saturation probable : derniers sujets ⇒ 0 nouvelle QC")
            else:
                st.info("Saturation non atteinte : QC encore en croissance.")

# ==============================================================================
# ONGLET 2 – AUDIT (structure inchangée + affichage si dispo)
# ==============================================================================
with tab_audit:
    st.subheader("🔍 Audit du moteur Granulo")

    st.info("Audit interne : chaque Qi d’un sujet traité doit mapper vers UNE et UNE SEULE QC (100 %).")
    st.info("Audit externe : couverture attendue ≥ 95 % sur sujet inconnu.")
    st.caption("Aucune logique métier implémentée dans cette version UI.")

    if "granulo_result" in st.session_state:
        st.divider()
        st.subheader("Résultats moteur (test)")
        st.json(st.session_state["granulo_result"]["audit"])
