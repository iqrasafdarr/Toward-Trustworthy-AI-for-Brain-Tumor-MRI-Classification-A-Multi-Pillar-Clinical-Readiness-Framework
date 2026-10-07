import argparse, json
import numpy as np

def accuracy(p, y):
    return float((p.argmax(1) == y).mean())

def weighted_f1(p, y):
    pred, C, f1s, w = p.argmax(1), p.shape[1], [], []
    for c in range(C):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0); w.append(np.sum(y == c))
    return float(np.average(f1s, weights=w))

def ece(p, y, n_bins=10):
    conf, correct = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1); out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)

def hce(p, y, tau=0.9):
    err = p.argmax(1) != y
    return float(((p.max(1) > tau) & err).sum() / max(err.sum(), 1))

def hce_over_n(p, y, tau=0.9):
    return float(((p.max(1) > tau) & (p.argmax(1) != y)).mean())

def bootstrap_ci(p, y, fn, B=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); vals = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); vals[b] = fn(p[i], y[i])
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]

def aurc_confidence(p, y):
    order = np.argsort(-p.max(1)); err = (p.argmax(1) != y)[order]
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    return float(np.sum((cov[1:] - cov[:-1]) * (risk[1:] + risk[:-1]) / 2))

def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_temperature(p, y, grid=np.linspace(0.05, 10, 400)):
    lg = np.log(np.clip(p, 1e-8, 1.0)); best, bt = np.inf, 1.0
    for T in grid:
        q = _softmax(lg / T); nll = -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best: best, bt = nll, T
    return float(bt)

def rescale(p, T):
    return _softmax(np.log(np.clip(p, 1e-8, 1.0)) / T)

def conformal_lac(p, y, alpha=0.10, n_splits=20, seed=1000):
    n = len(y); h = n // 2; cov, refer, empty, size = [], [], [], []
    for s in range(n_splits):
        idx = np.random.default_rng(seed + s).permutation(n); c, t = idx[:h], idx[h:]
        T = fit_temperature(p[c], y[c]); pc, pt = rescale(p[c], T), rescale(p[t], T)
        scores = 1 - pc[np.arange(len(c)), y[c]]
        q = np.quantile(scores, min(np.ceil((len(c) + 1) * (1 - alpha)) / len(c), 1.0), method="higher")
        sets = pt >= (1 - q); sz = sets.sum(1)
        cov.append(sets[np.arange(len(t)), y[t]].mean()); size.append(sz.mean())
        refer.append((sz != 1).mean()); empty.append((sz == 0).mean())
    f = lambda a: [float(np.mean(a)), float(np.std(a))]
    return {"coverage": f(cov), "mean_set_size": float(np.mean(size)),
            "referral_rate": f(refer), "empty_rate": float(np.mean(empty)),
            "alpha": alpha, "n_splits": n_splits}

def cri(acc, ece_, hce_, gen, w=(0.40, 0.25, 0.20, 0.15)):
    return float(w[0] * acc + w[1] * (1 - ece_) + w[2] * (1 - hce_) + w[3] * gen)

def evaluate(p, y, gen=1.0, B=10000):
    a, e, h = accuracy(p, y), ece(p, y), hce(p, y)
    return {"n": int(len(y)), "accuracy": a, "accuracy_ci95": bootstrap_ci(p, y, accuracy, B),
            "weighted_f1": weighted_f1(p, y), "ece": e, "hce_frac_of_errors": h,
            "hce_frac_of_all": hce_over_n(p, y), "aurc_confidence": aurc_confidence(p, y),
            "conformal_lac": conformal_lac(p, y), "gen": gen, "cri": cri(a, e, h, gen)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--out", default="report.json")
    a = ap.parse_args()
    data = np.load(a.probs); y = np.load(a.labels).astype(int)
    if y.ndim == 2: y = y.argmax(1)
    gen = {k: float(v) for k, v in (s.split("=") for s in a.gen)}
    report = {m: evaluate(data[m], y, gen.get(m, 1.0), a.boot) for m in data.files}
    json.dump(report, open(a.out, "w"), indent=2)
    for m, r in report.items():
        print(f"{m:15s} acc={r['accuracy']:.4f} ece={r['ece']:.4f} hce={r['hce_frac_of_errors']:.4f} "
              f"cri={r['cri']:.4f} lac_referral={r['conformal_lac']['referral_rate'][0]:.3f}")

if __name__ == "__main__":
    main()
