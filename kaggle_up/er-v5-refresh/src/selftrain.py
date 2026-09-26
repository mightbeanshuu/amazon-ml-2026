"""One gated self-training round: retrain the matcher with high-confidence TEST pairs added.

The organisers explicitly allow self-training on the provided test records. Evidence from peers is
asymmetric (+0.0022 toward the unseen country, −0.0017 the other way), so the round is GATED:
  1) pseudo-labels are taken only under a hard filter (p >= hi, the record's best S1 by a wide margin);
     pseudo-NEGATIVES come free: the other candidates of a pseudo-matched record.
  2) the gate is validated inside train: leave-one-country-out, self-train on the held-out country's
     unlabelled pairs, and keep the flag only if LOCO improves. The train phase writes the verdict to meta.
"""
import numpy as np
import pandas as pd

HI = 0.995
MARGIN = 0.5
NEG_BELOW = 0.3
W_PSEUDO = 0.5


def pseudo_labels(s1_i, r_i, p, w_country=None):
    """Rows and labels for a self-training round. Returns (idx, y_pseudo, weight)."""
    d = pd.DataFrame({"r": r_i, "p": p})
    mx = d.groupby("r")["p"].transform("max").values
    second = d.assign(q=np.where(p >= mx, -1.0, p)).groupby("r")["q"].transform("max").values
    pos = (p >= HI) & (p >= mx) & (mx - np.maximum(second, 0) >= MARGIN)
    # free negatives: another S1's candidate for a record that is confidently owned elsewhere
    owned = pd.Series(pos).groupby(r_i).transform("max").values.astype(bool)
    neg = owned & ~pos & (p <= NEG_BELOW)
    idx = np.flatnonzero(pos | neg)
    y = pos[idx].astype(int)
    w = np.full(len(idx), W_PSEUDO, np.float32)
    if w_country is not None:
        w *= w_country[idx]
    return idx, y, w


def retrain_with_pseudo(model_mod, X_train, y_train, X_pseudo, y_pseudo, w_pseudo, rounds, params=None):
    """Fit the stage model on labelled train plus weighted pseudo rows."""
    import lightgbm as lgb
    X = pd.concat([X_train, X_pseudo], ignore_index=True)
    y = np.concatenate([y_train, y_pseudo])
    w = np.concatenate([np.ones(len(y_train), np.float32), w_pseudo])
    prm = dict(model_mod.PARAMS, **(params or {}))
    dtr = lgb.Dataset(X, y, weight=w, free_raw_data=False)
    return [lgb.train(prm, dtr, max(rounds, 50))]
