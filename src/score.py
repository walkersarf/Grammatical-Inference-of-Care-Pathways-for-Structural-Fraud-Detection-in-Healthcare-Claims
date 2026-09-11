"""
score.py  --  anomaly score = -log P(string) under the ALERGIA Markov chain.

Works on the dict form produced by mc_io.save_mc:
    {"initial": id, "states": {id: {"output": sym, "trans": [(id, p), ...]}}}

Each state emits a symbol (output). A transition (tgt, p) means: with prob p
the next emitted symbol is states[tgt]["output"]. Scoring walks the chain
matching each string symbol to a transition whose target output == symbol.
No matching transition (prob 0) => structurally impossible => +inf.
"""
import math


def _step(states, sid, symbol):
    for tgt, p in states[sid]["trans"]:
        if states[tgt]["output"] == symbol:
            return tgt, p
    return None, 0.0


def neg_log_prob(mc, seq):
    if not seq:
        return float("inf")
    states = mc["states"]
    sid = mc["initial"]
    logp = 0.0
    if states[sid]["output"] == seq[0]:
        rest = seq[1:]
    else:
        sid, p = _step(states, sid, seq[0])
        if p == 0:
            return float("inf")
        logp += math.log(p)
        rest = seq[1:]
    for sym in rest:
        sid, p = _step(states, sid, sym)
        if p == 0:
            return float("inf")
        logp += math.log(p)
    return -logp


def score_many(mc, seqs):
    return [neg_log_prob(mc, s) for s in seqs]


if __name__ == "__main__":
    import pickle, pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from mc_io import load_mc
    mc = load_mc("results/model.pkl")
    test = pickle.load(open("results/test.pkl", "rb"))
    for s in test[:10]:
        sc = neg_log_prob(mc, s)
        tag = " INF " if sc == float("inf") else f"{sc:6.2f}"
        print(tag, " ".join(s)[:70])
