"""S1 enrichment: a second scoring round for UNCERTAIN pairs against S1 text enriched by confident matches.

The corruption is per record: every S2/S3 record descends from one S1. Once some records are confidently
assigned, the entity's true surface forms are the union of its records' tokens. A record that lost its house
number, its script, or its whole name still overlaps its siblings-via-S1 heavily. So:
  1) confident set C = pairs with p >= hi that are their record's best S1 by a clear margin
  2) enriched S1 text = S1 name/address tokens  ∪  tokens of its records in C (capped, rare-first)
  3) for pairs with lo <= p < hi, recompute cheap similarities against the enriched text
  4) a small LightGBM (trained on OOF with the same construction) maps (p, enriched sims) -> p'
Only step 3-4 touch the uncertain band, so the extra cost is small. Uses only provided data.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

HI, LO = 0.97, 0.05
CAP = 40                                                          # max tokens an entity accumulates per field


def _cp(a, b, sc):
    return process.cpdist(a, b, scorer=sc, workers=-1, dtype=np.float32)


def confident_mask(s1_i, r_i, p, hi=HI, margin=0.3):
    """p >= hi and the record's second-best S1 is at least `margin` lower."""
    d = pd.DataFrame({"r": r_i, "p": p})
    mx = d.groupby("r")["p"].transform("max").values
    second = d.assign(q=np.where(p >= mx, -1.0, p)).groupby("r")["q"].transform("max").values
    return (p >= hi) & (p >= mx) & (mx - np.maximum(second, 0) >= margin)


def build_enriched(s1, r, s1_i, r_i, conf, n_s1):
    """Enriched name/address token strings per S1 row (index-aligned lists)."""
    name = [set(x.split()) for x in s1.name_core.values]
    addr = [set(x.split()) for x in s1.addr_norm.values]
    order = np.flatnonzero(conf)
    for k in order:
        a, b = s1_i[k], r_i[k]
        if len(name[a]) < CAP:
            name[a] |= set(r.name_core.values[b].split())
        if len(addr[a]) < CAP:
            addr[a] |= set(r.addr_norm.values[b].split())
    return [" ".join(sorted(x)) for x in name], [" ".join(sorted(x)) for x in addr]


ENRICH_COLS = ["p0", "e_n_tset", "e_a_tset", "e_n_partial", "e_a_partial", "e_gain_n", "e_gain_a"]


def enrich_features(pairs_s1, pairs_r, p, s1, r, band, ename, eaddr):
    """Similarities of the banded pairs against enriched S1 text, plus the gain over the plain S1 text."""
    idx = np.flatnonzero(band)
    a_n = [ename[i] for i in pairs_s1[idx]]
    a_a = [eaddr[i] for i in pairs_s1[idx]]
    b_n = r.name_core.values[pairs_r[idx]].tolist()
    b_a = r.addr_norm.values[pairs_r[idx]].tolist()
    p_n = s1.name_core.values[pairs_s1[idx]].tolist()
    p_a = s1.addr_norm.values[pairs_s1[idx]].tolist()
    F = pd.DataFrame({
        "p0": p[idx],
        "e_n_tset": _cp(a_n, b_n, fuzz.token_set_ratio),
        "e_a_tset": _cp(a_a, b_a, fuzz.token_set_ratio),
        "e_n_partial": _cp(a_n, b_n, fuzz.partial_ratio),
        "e_a_partial": _cp(a_a, b_a, fuzz.partial_ratio),
    })
    F["e_gain_n"] = F.e_n_tset.values - _cp(p_n, b_n, fuzz.token_set_ratio)
    F["e_gain_a"] = F.e_a_tset.values - _cp(p_a, b_a, fuzz.token_set_ratio)
    return idx, F[ENRICH_COLS]


def apply_enrichment(s1_i, r_i, p, s1, r, n_s1, model=None, y=None, oof_groups=None, lgb_params=None):
    """Returns p' (p overwritten on the uncertain band) and, in training mode (y given), the fitted model.
    Training mode builds the band features from OOF p, fits LightGBM with grouped OOF, and returns OOF p'."""
    from . import model as M
    conf = confident_mask(s1_i, r_i, p)
    band = (p >= LO) & (p < HI) & ~conf
    ename, eaddr = build_enriched(s1, r, s1_i, r_i, conf, n_s1)
    idx, F = enrich_features(s1_i, r_i, p, s1, r, band, ename, eaddr)
    out = p.copy()
    if y is not None:
        pE, it = M.oof_predict(F, y[idx].astype(int), oof_groups[idx], 3,
                               params=lgb_params or {"learning_rate": 0.1, "num_leaves": 63})
        fitted = M.fit_full(F, y[idx].astype(int), it, params=lgb_params or {"learning_rate": 0.1,
                                                                             "num_leaves": 63})
        out[idx] = pE
        return out, fitted[0]
    out[idx] = model.predict(F)
    return out, None
