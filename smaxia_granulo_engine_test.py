# ============================================================
# SMAXIA – Granulo Test Engine (GTE)
# Phase : TEST / R&D
# Aucun hardcoding de QC
# ============================================================

import requests
import hashlib
import time
from collections import defaultdict
from math import sqrt

# -----------------------------
# 1. RÉCUPÉRATION DES SUJETS
# -----------------------------

def fetch_subjects_from_urls(urls: list, volume: int):
    """
    Télécharge les PDFs depuis les URLs fournies
    (TEST : ici on simule le contenu texte)
    """
    subjects = []

    for i in range(volume):
        url = urls[i % len(urls)]
        content = f"SUJET_SIMULÉ_{i}_FROM_{url}"

        subject_hash = hashlib.sha256(content.encode()).hexdigest()

        subjects.append({
            "id": f"S{i}",
            "source": url,
            "year": 2020 + (i % 5),
            "nature": ["INTERRO", "DST", "BAC"][i % 3],
            "raw_text": content,
            "hash": subject_hash,
            "timestamp": time.time()
        })

    return subjects


# -----------------------------
# 2. EXTRACTION DES Qi
# -----------------------------

def extract_qi(subject):
    """
    TEST : on découpe artificiellement un sujet en Qi
    """
    qi_list = []

    for i in range(3):
        qi_list.append({
            "qi_id": f"{subject['id']}_QI{i}",
            "text": f"Question {i} du {subject['id']}",
            "subject_id": subject["id"]
        })

    return qi_list


# -----------------------------
# 3. EXTRACTION ARI (STRUCTURE)
# -----------------------------

def extract_ari(qi):
    """
    ARI = structure de résolution (TEST SIMPLIFIÉ)
    """
    return [
        {"step": "identifier_terme_dominant", "T_j": 0.3},
        {"step": "factoriser", "T_j": 0.25},
        {"step": "limites_usuelles", "T_j": 0.2}
    ]


# -----------------------------
# 4. F1 – PSI
# -----------------------------

def compute_psi(ari_steps):
    epsilon = 0.01
    delta_c = len(ari_steps)

    sum_T = sum(step["T_j"] for step in ari_steps)
    psi_raw = delta_c * (epsilon + sum_T) ** 2

    return round(min(1.0, psi_raw), 2)


# -----------------------------
# 5. SIMILARITÉ σ (COSINUS)
# -----------------------------

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# -----------------------------
# 6. F2 – SCORE GRANULO
# -----------------------------

def compute_score(n_q, N_tot, psi, t_rec, sigmas):
    density = n_q / max(1, N_tot)
    recency = 1 / (1 + t_rec)

    redundancy_penalty = 1
    for s in sigmas:
        redundancy_penalty *= (1 - s)

    score = density * recency * psi * redundancy_penalty * 1000
    return int(score)


# -----------------------------
# 7. GÉNÉRATION DES QC
# -----------------------------

def generate_qc(qi_list):
    """
    Groupement par structure ARI
    """
    qc_map = defaultdict(list)

    for qi in qi_list:
        ari = extract_ari(qi)
        signature = tuple(step["step"] for step in ari)
        qc_map[signature].append((qi, ari))

    qc_list = []

    for idx, (sig, items) in enumerate(qc_map.items(), start=1):
        ari_steps = items[0][1]
        psi = compute_psi(ari_steps)

        n_q = len(items)
        N_tot = len(qi_list)
        t_rec = 1.0

        sigmas = []  # TEST : pas encore de redondance inter-QC
        score = compute_score(n_q, N_tot, psi, t_rec, sigmas)

        qc_list.append({
            "qc_id": f"QC-{idx:02d}",
            "ari": ari_steps,
            "psi": psi,
            "n_q": n_q,
            "N_tot": N_tot,
            "t_rec": t_rec,
            "score": score,
            "qi": [q[0] for q in items]
        })

    return qc_list


# -----------------------------
# 8. POINT D’ENTRÉE UI
# -----------------------------

def run_granulo_factory(urls, volume, classe, matiere, chapitres):
    subjects = fetch_subjects_from_urls(urls, volume)

    all_qi = []
    for subject in subjects:
        all_qi.extend(extract_qi(subject))

    qc_list = generate_qc(all_qi)

    return {
        "subjects": subjects,
        "qc": qc_list
    }
