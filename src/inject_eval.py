"""
inject_eval.py  --  the experiment.

Take held-out NORMAL strings. Copy a fraction and CORRUPT them with each
fraud edit. Score all. Label injected=1, clean=0. Compute ROC-AUC / PR-AUC
overall and per fraud type, and save curves.

    python src/inject_eval.py

Fraud edits:
  substitution -> upcoding        (swap a symbol for a higher-value one)
  insertion    -> phantom billing (insert a service that didn't happen)
  expansion    -> unbundling      (replace 1 symbol with several components)
  transposition-> sequence viol.  (swap adjacent clinical events)
"""
import pickle, random, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sklearn.metrics import roc_auc_score, average_precision_score
from score import score_many
from mc_io import load_mc

random.seed(1)

# symbols the injector uses; adjust to your real alphabet
HIGH_VALUE = "ENCOUNTER_ED"          # for upcoding substitution
PHANTOM    = "MRI_SCAN"              # for phantom insertion (unseen symbol)
BUNDLE_SRC = "HBA1C_LAB"            # expand this...
BUNDLE_OUT = ["GLUCOSE_LAB", "GLUCOSE_LAB", "NEPHRO_SCREEN"]  # ...into these
CLINICAL   = {"ENCOUNTER_AMB", "HBA1C_LAB", "MED_METFORMIN",
              "MED_INSULIN", "RETINAL_SCREEN", "FOOT_EXAM"}


def substitute(seq):
    s = list(seq)
    idx = [i for i, x in enumerate(s) if x in CLINICAL]
    if not idx:
        return None
    s[random.choice(idx)] = HIGH_VALUE
    return s


def insert(seq):
    s = list(seq)
    # never insert before START (position 0); keep the shared root intact
    s.insert(random.randint(1, len(s)), PHANTOM)
    return s


def expand(seq):
    s = list(seq)
    idx = [i for i, x in enumerate(s) if x == BUNDLE_SRC]
    if not idx:
        return None
    i = random.choice(idx)
    return s[:i] + BUNDLE_OUT + s[i+1:]


def transpose(seq):
    s = list(seq)
    idx = [i for i in range(len(s)-1) if s[i] in CLINICAL and s[i+1] in CLINICAL]
    if not idx:
        return None
    i = random.choice(idx)
    s[i], s[i+1] = s[i+1], s[i]
    return s


EDITS = {"upcoding": substitute, "phantom": insert,
         "unbundling": expand, "sequence": transpose}


def evaluate(model, clean, frac=0.5):
    results = {}
    for name, fn in EDITS.items():
        pool = random.sample(clean, int(len(clean) * frac))
        corrupted = [c for c in (fn(s) for s in pool) if c is not None]
        if not corrupted:
            print(f"[skip] {name}: no applicable strings")
            continue

        y = [0] * len(clean) + [1] * len(corrupted)
        raw = score_many(model, clean) + score_many(model, corrupted)
        # map inf -> large finite for AUC math; keep ordering
        finite = [r for r in raw if r != float("inf")]
        big = (max(finite) * 2 + 10) if finite else 1e6
        s = [big if r == float("inf") else r for r in raw]

        roc = roc_auc_score(y, s)
        pr  = average_precision_score(y, s)
        infrate = sum(1 for r in score_many(model, corrupted) if r == float("inf")) / len(corrupted)
        results[name] = dict(roc=roc, pr=pr, n=len(corrupted), inf_rate=infrate)
        print(f"{name:11s}  ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  "
              f"n={len(corrupted):4d}  hard-break={infrate:.0%}")
    return results


if __name__ == "__main__":
    model = load_mc("results/model.pkl")
    test  = pickle.load(open("results/test.pkl", "rb"))
    print(f"[eval] clean held-out = {len(test)}\n")
    res = evaluate(model, test)
    pathlib.Path("results").mkdir(exist_ok=True)
    pickle.dump(res, open("results/eval.pkl", "wb"))
    print("\n[saved] results/eval.pkl")