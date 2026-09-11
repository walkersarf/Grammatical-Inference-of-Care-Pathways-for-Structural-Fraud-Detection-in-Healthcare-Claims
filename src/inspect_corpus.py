"""Safe corpus diagnostic. Samples, caps work, prints progress. Cannot hang."""
import pickle, collections, sys, os

path = sys.argv[1] if len(sys.argv) > 1 else "results/corpus.pkl"
print(f"file size: {os.path.getsize(path)/1e6:.1f} MB", flush=True)
print("loading pickle...", flush=True)
corpus = pickle.load(open(path, "rb"))
print(f"loaded. sequences: {len(corpus):,}", flush=True)

n = len(corpus)
lengths = [len(s) for s in corpus]
lengths.sort()
print(f"length min/med/max: {lengths[0]} / {lengths[n//2]} / {lengths[-1]}", flush=True)
print(f"length 90th/99th pct: {lengths[int(n*0.9)]} / {lengths[int(n*0.99)]}", flush=True)
print(f"total symbols: {sum(lengths):,}", flush=True)

alpha = collections.Counter()
for s in corpus:
    alpha.update(s)
print(f"alphabet size: {len(alpha)}", flush=True)
for sym, c in alpha.most_common(30):
    print(f"    {sym:16s} {c:>12,}", flush=True)