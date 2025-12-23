# smaxia_console_v31_ui.py
# UI SAFE PATCH — aucune régression: l'app ne crash plus au chargement

import streamlit as st

st.set_page_config(page_title="SMAXIA Console v3.1", layout="wide")
st.title("SMAXIA — Console Granulo v3.1")

st.info(
    "Mode sécurisé : l'application ne plante plus au chargement. "
    "Le moteur est importé uniquement au clic (anti-régression)."
)

run_clicked = st.button("🚀 Lancer le moteur")

if run_clicked:
    try:
        # Import retardé (le chargement UI ne dépend plus du moteur)
        from smaxia_granulo_engine_test import run_granulo_test  # noqa: WPS433

        with st.spinner("Exécution du moteur..."):
            results = run_granulo_test()

        if not results:
            st.warning("Aucun résultat renvoyé par le moteur.")
        else:
            st.success(f"QC générées : {len(results)}")
            for i, r in enumerate(results[:10], 1):
                with st.expander(f"QC {i}"):
                    qc = r.get("qc", [])
                    frt = r.get("frt", {})
                    for j, qi in enumerate(qc, 1):
                        st.write(f"Qi {j} : {qi}")
                    st.markdown("### FRT")
                    st.json(frt)

    except Exception as e:
        # IMPORTANT : on n'écrase pas l'UI, on affiche l'erreur proprement
        st.error("Le moteur n’a pas pu être importé/exécuté. UI intacte (aucune régression).")
        st.code(repr(e))
        st.stop()
