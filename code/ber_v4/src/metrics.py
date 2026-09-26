"""Evaluation that mirrors the official metric, plus blocking diagnostics and F0.5-optimal set selection."""
import numpy as np


def parse_ids(x):
    if not isinstance(x, str) or not x.strip():
        return set()
    return {t.strip() for t in x.split(",") if t.strip()}


def fbeta_entity(pred: set, gold: set, beta=0.5):
    """Per-S1 F-beta with the official singleton rule: empty/empty = 1, and any mismatch on empty = 0."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_fbeta(pred: dict, gold: dict, beta=0.5):
    """Mean over EVERY S1 in gold. An S1 missing from pred counts as an empty prediction."""
    s = [fbeta_entity(pred.get(e, set()), g, beta) for e, g in gold.items()]
    return float(np.mean(s)) if s else 0.0


def macro_breakdown(pred, gold, groups: dict, beta=0.5):
    """Macro F0.5 per group label (e.g. country), plus the singleton and non-singleton split."""
    out = {}
    by = {}
    for e, g in gold.items():
        by.setdefault(groups.get(e, "?"), []).append(fbeta_entity(pred.get(e, set()), g, beta))
        by.setdefault("_singletons" if not g else "_matched", []).append(fbeta_entity(pred.get(e, set()), g, beta))
    for k, v in sorted(by.items()):
        out[k] = (round(float(np.mean(v)), 4), len(v))
    return out


def blocking_report(cands: dict, gold: dict, n_right: int):
    """PC (pair completeness), RR (reduction ratio), PQ, and the oracle macro-F0.5 ceiling."""
    tot_c = sum(len(v) for v in cands.values())
    tot_m = sum(len(v) for v in gold.values())
    hit = sum(len(cands.get(e, set()) & g) for e, g in gold.items())
    oracle = []
    for e, g in gold.items():
        if not g:
            oracle.append(1.0)
            continue
        r = len(cands.get(e, set()) & g) / len(g)
        oracle.append(0.0 if r == 0 else 1.25 * r / (0.25 + r))
    sizes = np.array([len(cands.get(e, ())) for e in gold]) if gold else np.zeros(1)
    return dict(PC=round(hit / max(tot_m, 1), 4), RR=round(1 - tot_c / max(len(gold) * n_right, 1), 6),
                PQ=round(hit / max(tot_c, 1), 4), oracle_macroF05=round(float(np.mean(oracle)), 4),
                mean_cands=round(float(sizes.mean()), 2), p95_cands=int(np.percentile(sizes, 95)),
                max_cands=int(sizes.max()), n_pairs=tot_c)


# ---- expected-F0.5-optimal set selection (GFM-style, independence assumption) ------------------
def _pb_pmf(ps):
    d = np.zeros(len(ps) + 1)
    d[0] = 1.0
    for q in ps:
        d[1:] = d[1:] * (1 - q) + d[:-1] * q
        d[0] *= (1 - q)
    return d


def best_set_fbeta(p, beta=0.5, max_m=12):
    """p: calibrated match probabilities of one S1's candidates. Returns (indices to predict, expected F).
    Under independence the optimum is the empty set or a top-k by probability (Dembczynski et al. 2011).
    Checked against brute force over all subsets."""
    p = np.asarray(p, float)
    order = np.argsort(-p)[:max_m]
    ps = p[order]
    m = len(ps)
    b2 = beta * beta
    best_val, best_k = float(np.prod(1 - ps)), 0
    for k in range(1, m + 1):
        tp, fn = _pb_pmf(ps[:k]), _pb_pmf(ps[k:])
        t = np.arange(k + 1)[:, None]
        f = np.arange(m - k + 1)[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(t > 0, (1 + b2) * t / (b2 * (t + f) + k), 0.0)
        val = float((val * tp[:, None] * fn[None, :]).sum())
        if val > best_val:
            best_val, best_k = val, k
    return order[:best_k], best_val
