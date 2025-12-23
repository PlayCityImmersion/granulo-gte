# smaxia_console_v31_ui.py
# SMAXIA GRANULO CONSOLE v3.1

from smaxia_granulo_engine_test import run_engine

def display_qc(qc_data):
    print("\n==============================")
    print(" QC — QUESTION CLÉ")
    print("==============================")

    for i, q in enumerate(qc_data["qc"], 1):
        print(f"Qi {i}: {q}")

    frt = qc_data["frt"]

    print("\n--- FRT ---")
    print(f"Déclencheurs : {', '.join(frt['declencheurs'])}")
    print(f"ARI : {frt['ari']}")
    print(f"Nombre Qi : {frt['n_q']}")
    print(f"Score : {frt['score']}")

def main():
    print("=== SMAXIA GRANULO ENGINE — TEST ===")
    results = run_engine()

    if not results:
        print("❌ Aucune QC générée — vérifier les sources")
        return

    print(f"✅ QC totales générées : {len(results)}")

    for qc in results[:3]:
        display_qc(qc)

if __name__ == "__main__":
    main()
