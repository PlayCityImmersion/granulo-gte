import json
import re
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# =============================================================================
# STRUCTURES DE DONNÉES INVARIANTES
# =============================================================================

@dataclass
class QuestionCle:
    id: str
    canonical_text: str     # La QUESTION (chargée depuis P3)
    operator_tag: str       # Le Concept (chargé depuis P3)
    triggers_found: List[str]
    covered_qi_list: List[str] = field(default_factory=list) 
    is_black_swan: bool = False

    @property
    def is_well_formed(self) -> bool:
        """B2: Vérifie que c'est une structure interrogative valide"""
        return "?" in self.canonical_text or self.is_black_swan

@dataclass
class GranuloAudit:
    """Structure du Verdict Booléen"""
    b1_all_mapped: bool = False       
    b2_qc_structure: bool = False     
    b3_triggers_valid: bool = False   
    b4_conservation: bool = False     
    b5_black_swan: bool = False       
    
    @property
    def global_pass(self) -> bool:
        return all([self.b1_all_mapped, self.b2_qc_structure, self.b3_triggers_valid, self.b4_conservation, self.b5_black_swan])

# =============================================================================
# MOTEUR GRANULO 15 V4.0 (ARCHITECTURE P3 INJECTÉE)
# =============================================================================

class GranuloEngine:
    def __init__(self, p3_library_path: str):
        self.raw_atoms = []
        self.archetypes = self._load_invariant_library(p3_library_path)

    def _load_invariant_library(self, path: str) -> List[Dict]:
        """
        INJECTION P3 : Charge la connaissance externe (JSON).
        Le moteur est agnostique du contenu.
        """
        if not os.path.exists(path):
            return [] # Ou lever une erreur critique en PROD
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def ingest_qi(self, text: str, source_type: str):
        self.raw_atoms.append({"text": text, "source": source_type})

    def run_reduction_process(self) -> Tuple[List[QuestionCle], GranuloAudit]:
        # 1. INITIALISATION DYNAMIQUE (Aucun hardcode)
        active_qcs = {} 
        
        # 2. MÉCANISME DE MATCHING ABSTRAIT
        for atom in self.raw_atoms:
            text_lower = atom['text'].lower()
            matched = False
            
            # Itération sur la Bibliothèque injectée
            for arch in self.archetypes:
                found_triggers = []
                
                # Vérification des Regex injectées
                for trigger_pattern in arch['triggers']:
                    # Compilation à la volée pour robustesse
                    if re.search(trigger_pattern, text_lower, re.IGNORECASE):
                        found_triggers.append(trigger_pattern)
                
                if found_triggers:
                    qc_id = arch['id']
                    if qc_id not in active_qcs:
                        active_qcs[qc_id] = QuestionCle(
                            id=qc_id,
                            canonical_text=arch['canonical_text'],
                            operator_tag=arch['operator_tag'],
                            triggers_found=found_triggers
                        )
                    
                    # Règle de Conservation
                    active_qcs[qc_id].covered_qi_list.append(atom['text'])
                    matched = True
                    break # Une Qi -> Une QC (Principe de convergence)
            
            if not matched:
                # GESTION BLACK SWAN (Invariant universel)
                meta_id = "QC-15-META"
                if meta_id not in active_qcs:
                    active_qcs[meta_id] = QuestionCle(
                        id=meta_id,
                        canonical_text="Méta-Protocole : Transposition Hors-Piste (Black Swan)",
                        operator_tag="BLACK_SWAN",
                        triggers_found=["NO_MATCH"],
                        is_black_swan=True
                    )
                active_qcs[meta_id].covered_qi_list.append(atom['text'])

        # 3. CONSOLIDATION
        qc_list = list(active_qcs.values())
        qc_list.sort(key=lambda x: len(x.covered_qi_list), reverse=True)

        # 4. AUDIT BOOLÉEN (Le Juge)
        audit = GranuloAudit()
        
        # B1: All Mapped (Garanti par le fallback Black Swan, mais vérifié)
        audit.b1_all_mapped = all(len(qc.covered_qi_list) > 0 for qc in qc_list)
        
        # B2: Structure QC (Doit venir du JSON propre)
        audit.b2_qc_structure = all(qc.is_well_formed for qc in qc_list)
        
        # B3: Triggers Valid
        audit.b3_triggers_valid = all(len(qc.triggers_found) > 0 for qc in qc_list)
        
        # B4: Conservation (Entrée == Sortie)
        total_input = len(self.raw_atoms)
        total_output = sum(len(qc.covered_qi_list) for qc in qc_list)
        audit.b4_conservation = (total_input == total_output)
        
        # B5: Black Swan Exists (Sécurité)
        audit.b5_black_swan = any(qc.is_black_swan for qc in qc_list)

        return qc_list, audit
