"""
learn.py  --  corpus -> ALERGIA model (+ RPNI optional) + figure.

    python src/learn.py --corpus results/corpus.pkl --eps 0.05

Holds out a test split BEFORE learning (never leak test patients).
Saves: results/model.pkl, results/train.pkl, results/test.pkl,
       figures/induced_automaton.pdf
"""
import argparse, pickle, pathlib, random
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from aalpy.learning_algs import run_Alergia
from mc_io import save_mc


def split(corpus, test_frac=0.2, seed=1):
    rng = random.Random(seed)
    idx = list(range(len(corpus)))
    rng.shuffle(idx)
    k = int(len(idx) * test_frac)
    test = [corpus[i] for i in idx[:k]]
    train = [corpus[i] for i in idx[k:]]
    return train, test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="results/corpus.pkl")
    ap.add_argument("--eps", type=float, default=0.05)   # ALERGIA merge tolerance (alpha)
    ap.add_argument("--test_frac", type=float, default=0.2)
    a = ap.parse_args()

    corpus = pickle.load(open(a.corpus, "rb"))
    train, test = split(corpus, a.test_frac)
    print(f"train={len(train)} test={len(test)}  eps={a.eps}")

    model = run_Alergia(train, automaton_type="mc", eps=a.eps)
    print(f"[ok] learned MC with {len(model.states)} states")

    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("figures").mkdir(exist_ok=True)
    save_mc(model, "results/model.pkl")
    pickle.dump(train, open("results/train.pkl", "wb"))
    pickle.dump(test,  open("results/test.pkl",  "wb"))

    try:
        model.visualize(path="figures/induced_automaton", file_type="pdf")
        print("[ok] figure -> figures/induced_automaton.pdf")
    except Exception as e:
        print(f"[warn] visualize failed ({e}); need graphviz on PATH. Model still saved.")
