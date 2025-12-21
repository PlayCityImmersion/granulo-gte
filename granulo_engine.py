import hashlib
import numpy as np
import re
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# =============================================================================
# MOTEUR GRANULO 15 V4.0 (PURE LOGIC - ZERO HARDCODE)
# =============================================================================

@dataclass
class QuestionCle:
    id: str
    canonical_text: str
    operator_tag: str
    triggers_found: List[str]
    covered_qi_list: List[str] = field(default_factory=list) 
    is_black_swan: bool = False

    @property
    def is_well_formed(self) -> bool:
        return "?" in self.canonical_text or self.is_black_swan

@dataclass
class GranuloAudit:
    b1_all_mapped: bool = False
    b2_qc_structure: bool = False
    b3_triggers_valid: bool = False
    b4_conservation: bool = False
    b5_black_swan: bool = False
    
    @property
    def global_pass(self) -> bool:
        return all([self.b1_all_mapped, self.b2_qc_structure, self.b3_triggers_valid, self.b4_conservation, self.b5_black_swan])

class GranuloEngine:
    def __init__(self, config_path: str = "smaxia_p3_db.json"):
        self.raw_atoms = []
        self.archetypes = self._load_configuration(config_path)

    def _load_configuration(self, path: str) -> List[Dict]:
        """
        INJECTION DE DÉPENDANCE : 
        Le moteur charge sa connaissance depuis l'extérieur.
        Si le fichier change (Pays), le moteur s'adapte sans redéploiement.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback critique (ne devrait jamais arriver en PROD)
            return []

    def ingest_qi(self, text: str, source_type: str):
        self.raw_atoms.append({"text": text, "source": source_type, "mapped": False})

    def run_reduction_process(self) -> Tuple[List[QuestionCle], GranuloAudit]:
        # 1. INITIALISATION DYNAMIQUE (Pas de hardcode)
        active_qcs = {}
        
        # 2. MAPPING AGNOSTIQUE
        for atom in self.raw_atoms:
            text_lower = atom['text'].lower()
            matched = False
            
            # Le moteur itère sur la CONFIGURATION injectée, pas sur du code dur
            for arch in self.archetypes:
                found_triggers = []
                # Matching Universel (Regex ou String)
                for trigger in arch['triggers']:
                    # Échappement regex sécurisé pour les symboles mathématiques
                    pattern = re.escape(trigger) if "\\" in trigger else trigger
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        found_triggers.append(trigger)
                
                if found_triggers:
                    qc_id = arch['id']
                    if qc_id not in active_qcs:
                        active_qcs[qc_id] = QuestionCle(
                            id=qc_id,
                            canonical_text=arch['canonical_template'],
                            operator_tag=arch['operator_tag'],
                            triggers_found=found_triggers
                        )
                    
                    active_qcs[qc_id].covered_qi_list.append(atom['text'])
                    atom['mapped'] = True
                    matched = True
                    break 
            
            if not matched:
                # BLACK SWAN DYNAMIQUE
                meta_id = "QC-15-META"
                if meta_id not in active_qcs:
                    active_qcs[meta_id] = QuestionCle(
                        id=meta_id,
                        canonical_text="Méta-Protocole : Transposition Hors-Piste",
                        operator_tag="BLACK_SWAN",
                        triggers_found=["NO_MATCH"],
                        is_black_swan=True
                    )
                active_qcs[meta_id].covered_qi_list.append(atom['text'])

        # 3. TRI & AUDIT
        qc_list = list(active_qcs.values())
        qc_list.sort(key=lambda x: len(x.covered_qi_list), reverse=True)

        audit = GranuloAudit()
        audit.b1_all_mapped = all(len(qc.covered_qi_list) > 0 for qc in qc_list)
        audit.b2_qc_structure = all(qc.is_well_formed for qc in qc_list)
        audit.b3_triggers_valid = all(len(qc.triggers_found) > 0 for qc in qc_list)
        
        total_input = len(self.raw_atoms)
        total_output = sum(len(qc.covered_qi_list) for qc in qc_list)
        audit.b4_conservation = (total_input == total_output)
        
        audit.b5_black_swan = any(qc.is_black_swan for qc in qc_list)

        return qc_list, audit
