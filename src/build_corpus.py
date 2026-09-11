"""
build_corpus.py -- CSVs -> per-patient symbol strings, DIABETIC COHORT ONLY.

    python src/build_corpus.py --csv data --out results/corpus.pkl

Key behaviours for the real MA general-population run:
  * cohort filter: keep only patients with a diabetes diagnosis in conditions.csv
  * observations: read only rows that map to a real symbol (HbA1c/glucose),
    so the 24.7M vitals rows don't drown the signal
  * per-file timestamp column: observations use DATE, others use START
"""
import argparse, pickle, pathlib, sys, random
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from symbol_map import to_symbol, gap_symbol, USE_GAP_SYMBOLS, DROP_OTHER, OTHER

EVENT_FILES = ["conditions", "medications", "procedures", "observations", "encounters"]
TIME_COL = {"observations": "DATE"}          # default "START" otherwise


def time_col(name):
    return TIME_COL.get(name, "START")


def diabetic_ids(csv_dir):
    cond = pd.read_csv(csv_dir / "conditions.csv", low_memory=False,
                       usecols=["PATIENT", "DESCRIPTION"])
    desc = cond["DESCRIPTION"].fillna("")
    # require a genuine diabetes DISORDER, not prediabetes / risk / history
    is_dm = desc.str.contains("diabetes", case=False) & desc.str.contains("disorder", case=False)
    is_dm &= ~desc.str.contains("prediabet", case=False)
    ids = set(cond.loc[is_dm, "PATIENT"])
    print(f"[cohort] {len(ids)} true-diabetes patients "
          f"(of {cond['PATIENT'].nunique()} total)")
    return ids


def load_events(csv_dir, keep_ids):
    """Return long DataFrame: PATIENT, when, symbol -- pre-filtered to cohort & mapped."""
    frames = []
    for name in EVENT_FILES:
        f = csv_dir / f"{name}.csv"
        if not f.exists():
            continue
        tc = time_col(name)
        df = pd.read_csv(f, low_memory=False, usecols=["PATIENT", tc, "DESCRIPTION"])
        df = df[df["PATIENT"].isin(keep_ids)].copy()
        df["symbol"] = df["DESCRIPTION"].map(to_symbol)
        if DROP_OTHER:
            df = df[df["symbol"] != OTHER]
        # Synthea writes a consistent timestamp format per column, but mixes
        # tz-aware and tz-naive ACROSS files. utc=True unifies both cleanly;
        # tz_localize(None) then makes the whole column sortable (tz-naive).
        parsed = pd.to_datetime(df[tc], errors="coerce", utc=True).dt.tz_localize(None)
        n_bad = parsed.isna().sum()
        if n_bad:
            print(f"    [note] {name}: {n_bad} rows had unparseable {tc} -> dropped")
        df["when"] = parsed
        df = df.dropna(subset=["when"])
        frames.append(df[["PATIENT", "when", "symbol"]])
        print(f"  {name}: {len(df)} mapped events")
    return pd.concat(frames, ignore_index=True)


def to_strings(events, max_events=60):
    corpus = []
    for pid, g in events.groupby("PATIENT"):
        g = g.sort_values("when")
        syms = list(g["symbol"])
        dates = list(g["when"])

        # collapse consecutive duplicate events BEFORE adding gaps:
        # dialysis 3x/week for years -> one DIALYSIS token, not thousands
        c_syms, c_dates = [], []
        for s, d in zip(syms, dates):
            if not c_syms or c_syms[-1] != s:
                c_syms.append(s)
                c_dates.append(d)

        # cap very long histories: keep first max_events clinical events
        c_syms, c_dates = c_syms[:max_events], c_dates[:max_events]

        seq, prev = [], None
        for s, d in zip(c_syms, c_dates):
            if prev is not None and USE_GAP_SYMBOLS:
                seq.append(gap_symbol((d - prev).days))
            seq.append(s)
            prev = d

        if len(seq) >= 2:
            corpus.append(["START"] + seq)
    return corpus


def build(csv_dir):
    ids = diabetic_ids(csv_dir)
    if not ids:
        print("[warn] no diabetic patients found -> toy fallback")
        return toy_corpus()
    events = load_events(csv_dir, ids)
    corpus = to_strings(events)
    print(f"[ok] built {len(corpus)} diabetic-patient strings")
    return corpus


def toy_corpus(n=400, seed=1):
    rng = random.Random(seed)
    base = ["ENCOUNTER_AMB", "HBA1C_LAB", "MED_METFORMIN"]
    out = []
    for _ in range(n):
        s = list(base)
        for _ in range(rng.randint(0, 3)):
            s += ["ENCOUNTER_AMB", "HBA1C_LAB"]
            if rng.random() < 0.4: s.append("RETINAL_EXAM")
        if rng.random() < 0.3: s.append("MED_INSULIN")
        if USE_GAP_SYMBOLS:
            g = []
            for i, sym in enumerate(s):
                if i > 0: g.append(rng.choice(["GAP_LE30", "GAP_LE180"]))
                g.append(sym)
            s = g
        out.append(["START"] + s)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data")
    ap.add_argument("--out", default="results/corpus.pkl")
    ap.add_argument("--toy", action="store_true")
    a = ap.parse_args()

    corpus = toy_corpus() if a.toy else build(pathlib.Path(a.csv))
    lengths = sorted(len(s) for s in corpus)
    print(f"    n={len(corpus)}  len min/med/max = "
          f"{lengths[0]}/{lengths[len(lengths)//2]}/{lengths[-1]}")
    # show a couple of real examples
    for s in corpus[:2]:
        print("    e.g.", " ".join(s[:18]), "...")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(corpus, open(a.out, "wb"))
    print(f"[saved] {a.out}")