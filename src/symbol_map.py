"""
symbol_map.py -- alphabet for the DIABETIC-cohort diabetes-care model.
Built against the real 5000-patient MA general-population run.

Strategy: anchor on diabetes meds + complications + relevant encounters.
Vitals observations are DELIBERATELY excluded (24.7M rows of BP/weight/HR
would drown the signal). Only HbA1c/glucose labs are kept, IF present.
FREEZE once experiments start.
"""

SYMBOL_MAP = {
    # --- diabetes medications (anchors) ---
    "Metformin":                         "MED_METFORMIN",
    "insulin isophane":                  "MED_INSULIN",
    "insulin":                           "MED_INSULIN",       # substr catches Humulin line
    # --- common comorbid cardio meds (diabetes travels with these) ---
    "lisinopril":                        "MED_ACE",
    "Hydrochlorothiazide":               "MED_DIURETIC",
    "amLODIPine":                        "MED_CCB",
    "Simvastatin":                       "MED_STATIN",
    # --- diabetes labs (HbA1c is the meaningful monitoring event) ---
    "Hemoglobin A1c":                    "HBA1C_LAB",
    # NOTE: Glucose deliberately EXCLUDED -- measured at nearly every visit,
    #       floods strings without marking a care-pathway step.
    # --- complications / diabetes-specific procedures ---
    "Ophthalmic examination":            "RETINAL_EXAM",
    "tropicamide":                       "RETINAL_EXAM",      # dilating agent for eye exam
    # NOTE: Renal dialysis deliberately EXCLUDED -- 3x/week for years produces
    #       thousands of repeats per patient; collapse+exclude keeps signal clean.
    # --- encounters (care structure) ---
    "Emergency room admission":          "ENCOUNTER_ED",
    "Emergency hospital admission":      "ENCOUNTER_ED",
    "Encounter for check up":            "ENCOUNTER_AMB",
    "General examination of patient":    "ENCOUNTER_AMB",
    "Follow-up encounter":               "ENCOUNTER_FU",
    "Outpatient procedure":              "ENCOUNTER_OP",
    # --- diagnosis marker ---
    "Diabetes":                          "DX_DIABETES",
}

MATCH_MODE = "substr"       # dosed drug names -> match on the ingredient substring
OTHER = "OTHER"
DROP_OTHER = True           # essential here: drops all the dental/social/vitals noise


def to_symbol(description: str) -> str:
    if not isinstance(description, str):
        return OTHER
    if MATCH_MODE == "exact":
        return SYMBOL_MAP.get(description, OTHER)
    for key, sym in SYMBOL_MAP.items():
        if key.lower() in description.lower():
            return sym
    return OTHER


def gap_symbol(days: float) -> str:
    if days <= 30:  return "GAP_LE30"
    if days <= 180: return "GAP_LE180"
    return "GAP_GT180"


USE_GAP_SYMBOLS = True