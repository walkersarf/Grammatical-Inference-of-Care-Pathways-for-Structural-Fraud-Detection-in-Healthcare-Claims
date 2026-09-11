"""Run FIRST once CSV exists: dumps real DESCRIPTIONs to design SYMBOL_MAP."""
import pandas as pd, pathlib, sys
CSV = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
for name in ["conditions","medications","procedures","observations","encounters"]:
    f = CSV / f"{name}.csv"
    if not f.exists():
        print(f"[missing] {f}"); continue
    df = pd.read_csv(f, low_memory=False)
    print(f"\n===== {name}.csv ({len(df)} rows) =====")
    print("columns:", list(df.columns))
    if "DESCRIPTION" in df.columns:
        print(df["DESCRIPTION"].value_counts().head(25).to_string())
