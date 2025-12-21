# granulo_engine.py
# GRANULO 15 ENGINE — P6
# Rôle : Moteur de réduction Qi -> QC (Auditeur, pas applicatif)

from dataclasses import dataclass
from typing import List, Dict
import hashlib
import uuid
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# =========================
# DATA STRUCTURES (SCELLÉES)
# =========================

@dataclass
class TriggerSignature:
    verb: str
    obj: str
    context: str

    def hash(self) -> str:
        raw = f"{self.verb}|{self.obj}|{self.context}".lower()
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class QuestionCle:
    qc_id: str
    signature: TriggerSignature
    psi_score: float
    sigma_class: int
    canonical_text: str
    qi_covered: int
    is_black_swan: bool = False


# =========================
# GRANULO ENGINE
# =========================

class GranuloEngine:

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # -------------------------
    # EXTRACTION Qi (BRUTE)
    # -------------------------
    def extract_qi(self, texts: List[str]) -> List[str]:
        qi = []
        for t in texts:
            for line in t.split("\n"):
                line = line.strip()
                if len(line) > 20:
                    qi.append(line)
        return qi

    # -------------------------
    # SIGNATURE V,O,C (HEURISTIQUE)
    # -------------------------
    def infer_signature(self, text: str) -> TriggerSignature:
        tokens = text.lower().split()
        verb = tokens[0] if tokens else "resoudre"
        obj = tokens[1] if len(tokens) > 1 else "objet"
        context = "chapitre"
        return TriggerSignature(verb, obj, context)

    # -------------------------
    # PSI (F1 — DENSITÉ)
    # -------------------------
    def compute_psi(self, qi_count: int) -> float:
        return round(min(1.0, np.log1p(qi_count) / 3), 3)

    # -------------------------
    # SIGMA (DIFFICULTÉ)
    # -------------------------
    def compute_sigma(self, qi_count: int) -> int:
        if qi_count < 3:
            return 1
        if qi_count < 7:
            return 2
        if qi_count < 12:
            return 3
        return 4

    # -------------------------
    # CORE PROCESS
    # -------------------------
    def process(self, texts: List[str]) -> Dict:
        qi = self.extract_qi(texts)

        if len(qi) == 0:
            return {"qcs": [], "orphans": [], "coverage": 0.0}

        embeddings = self.model.encode(qi)
        sim_matrix = cosine_similarity(embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1 - self.similarity_threshold
        )

        labels = clustering.fit_predict(1 - sim_matrix)

        clusters: Dict[int, List[str]] = {}
        for label, q in zip(labels, qi):
            clusters.setdefault(label, []).append(q)

        qcs: List[QuestionCle] = []

        for cluster_qi in clusters.values():
            sig = self.infer_signature(cluster_qi[0])
            qc = QuestionCle(
                qc_id=str(uuid.uuid4())[:8],
                signature=sig,
                psi_score=self.compute_psi(len(cluster_qi)),
                sigma_class=self.compute_sigma(len(cluster_qi)),
                canonical_text=cluster_qi[0],
                qi_covered=len(cluster_qi)
            )
            qcs.append(qc)

        # -------------------------
        # NORMALISATION À 15 QC
        # -------------------------
        qcs = sorted(qcs, key=lambda x: x.psi_score, reverse=True)

        if len(qcs) > 14:
            qcs = qcs[:14]

        # QC #15 — BLACK SWAN
        black_swan = QuestionCle(
            qc_id="QC-15",
            signature=TriggerSignature("transposer", "concept", "inattendu"),
            psi_score=1.0,
            sigma_class=4,
            canonical_text="Question de transposition hors cadre standard",
            qi_covered=0,
            is_black_swan=True
        )
        qcs.append(black_swan)

        total_covered = sum(q.qi_covered for q in qcs)
        coverage = round(min(1.0, total_covered / max(1, len(qi))), 3)

        orphans = qi[total_covered:]

        return {
            "qcs": qcs,
            "orphans": orphans,
            "coverage": coverage
        }

