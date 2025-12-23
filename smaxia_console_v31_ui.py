import streamlit as st
import pandas as pd
from datetime import datetime

# =========================================================
# CONFIG GÉNÉRALE
# =========================================================
st.set_page_config(
    page_title="SMAXIA - Console V31 (Saturation Proof)",
    layout="wide"
)

st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# =========================================================
# SIDEBAR – PARAMÈTRES ACADÉMIQUES (FRANCE)
# =========================================================
with st.sidebar:
    st.header("Paramètres académiques")

    st.selectbox("Classe", ["Terminale"], disabled=True)

    matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])

    PROGRAMMES = {
        "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
        "PHYSIQUE": ["MÉCANIQUE", "ONDES", "ÉLECTRICITÉ"]
    }

    chapitres = st.multiselect(
        "Chapitres",
        PROGRAMMES[matiere],
        default=PROGRAMMES[matiere][:1]
    )

# =========================================================
# ONGLET PRINCIPAL
# =========================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# =========================================================
# ONGLET 1 — USINE
# =========================================================
with tab_usine:

    # -------------------------------
    # INJECTION DES SUJETS
    # -------------------------------
    st.subheader("🧪 Injection des sujets")

    col1, col2 = st.columns([4, 1])

    with col1:
        urls = st.text_area(
            "URLs Sources (références)",
            "https://apmep.fr",
            height=80
        )

    with col2:
        volume = st.number_input(
            "Volume de sujets",
            min_value=1,
            max_value=500,
            value=15,
            step=1
        )

    lancer = st.button("🚀 LANCER L'USINE")

    # -------------------------------
    # APPEL DU MOTEUR (HOOK)
    # -------------------------------
    if lancer:
        try:
            from smaxia_granulo_engine_test import run_granulo_engine

            result = run_granulo_engine(
                urls=urls.splitlines(),
                volume=volume,
                matiere=matiere,
                chapitres=chapitres
            )

            st.session_state["sources"] = result["sources"]
            st.session_state["qcs"] = result["qcs"]

            st.success("Usine lancée – moteur Granulo branché (mode test).")

        except Exception as e:
            st.error("Moteur non branché ou erreur détectée.")
            st.code(str(e))

    # -------------------------------
    # TABLEAU DES SUJETS TRAITÉS
    # -------------------------------
    st.divider()
    st.subheader("📥 Sujets traités")

    if "sources" in st.session_state:
        df_sources = pd.DataFrame(st.session_state["sources"])
        st.dataframe(
            df_sources,
            use_container_width=True
        )
    else:
        st.info("Données affichées uniquement après branchement du moteur réel.")

    # -------------------------------
    # BASE DE CONNAISSANCE (QC)
    # -------------------------------
    st.divider()
    st.subheader("🧠 Base de connaissance (QC)")

    if "qcs" in st.session_state:
        for qc in st.session_state["qcs"]:
            st.markdown(f"""
            ### Chapitre : {qc['chapitre']}
            **{qc['qc_id']} : {qc['titre']}**

            `Score(q)={qc['score']} | n_q={qc['n_q']} | Ψ={qc['psi']} | N_tot={qc['n_tot']} | t_réc={qc['t_rec']}`
            """)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("🔥 **Déclencheurs**")
                for d in qc["declencheurs"]:
                    st.markdown(f"- {d}")

            with c2:
                st.markdown("⚙️ **ARI**")
                for a in qc["ari"]:
                    st.markdown(f"- {a}")

            with c3:
                st.markdown("📘 **FRT**")
                for bloc in qc["frt"]:
                    st.info(bloc)

            with c4:
                st.markdown("📄 **Qi associées**")
                for qi in qc["qi"]:
                    st.markdown(f"- {qi}")

            st.divider()
    else:
        st.info("Aucune QC affichée – moteur non exécuté.")

    # -------------------------------
    # COURBE DE SATURATION (PASSIVE)
    # -------------------------------
    st.subheader("📈 Courbe de saturation (QC / volume de sujets)")
    st.info("La courbe sera activée une fois les équations F1/F2 validées.")

# =========================================================
# ONGLET 2 — AUDIT
# =========================================================
with tab_audit:
    st.subheader("🔍 Audit du moteur Granulo")

    st.markdown("""
    **Audit interne**  
    Objectif : chaque Qi → UNE et UNE SEULE QC  
    Résultat attendu : **100 %**
    """)

    st.di
