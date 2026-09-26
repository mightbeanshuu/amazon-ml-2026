"""LightGBM matcher: grouped out-of-fold training, leave-one-country-out (LOCO) and collective stacking."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=63, verbose=-1, num_threads=8, seed=0)
MAX_ROUNDS = 800


def _train(X, y, Xv=None, yv=None, rounds=MAX_ROUNDS, params=None, seed=0):
    p = dict(PARAMS, **(params or {}), seed=seed)
    dtr = lgb.Dataset(X, y, free_raw_data=False)
    if Xv is not None:
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        m = lgb.train(p, dtr, rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        m = lgb.train(p, dtr, rounds)
    return m


STAGE1_PARAMS = dict(learning_rate=0.1, num_leaves=31, min_data_in_leaf=100)


def oof_predict(X, y, groups, n_splits=5, params=None, seeds=(0,)):
    """Out-of-fold probabilities, grouped by S1 so that no S1 leaks across folds.
    Returns (oof, mean best iteration)."""
    oof = np.zeros(len(y), np.float64)
    iters = []
    for seed in seeds:
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = _train(X.iloc[tr], y[tr], X.iloc[va], y[va], params=params, seed=seed)
            oof[va] += m.predict(X.iloc[va], num_iteration=m.best_iteration) / len(seeds)
            iters.append(m.best_iteration)
    return oof, int(np.mean(iters))


def fit_full(X, y, rounds, params=None, seeds=(0,)):
    return [_train(X, y, rounds=max(rounds, 50), params=params, seed=s) for s in seeds]


def predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def loco_predict(X, y, country, rounds, params=None):
    """Train on all countries except c and predict c. Proxy for the unseen test country (France)."""
    out = np.full(len(y), np.nan)
    for c in np.unique(country):
        tr, va = country != c, country == c
        if tr.sum() == 0 or y[tr].sum() == 0:
            continue
        m = _train(X[tr], y[tr], rounds=rounds, params=params)
        out[va] = m.predict(X[va])
    return out


def _second_max(keys, p):
    """Second-highest p within each key group, aligned to rows (0 when the group has one row). Vectorised."""
    order = np.lexsort((-p, keys))
    k, v = keys[order], p[order]
    first = np.r_[True, k[1:] != k[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(k)), 0))
    nxt = start + 1
    ok = (nxt < len(k)) & (k[np.minimum(nxt, len(k) - 1)] == k)
    sec_sorted = np.where(ok, v[np.minimum(nxt, len(k) - 1)], 0.0)
    out = np.empty(len(p))
    out[order] = sec_sorted
    return out


def collective_features(s1_i, r_i, p):
    """Features of a pair relative to its competitors, computed from stage-2 probabilities."""
    s1_i, r_i, p = np.asarray(s1_i), np.asarray(r_i), np.asarray(p, float)
    d = pd.DataFrame({"a": s1_i, "b": r_i, "p": p, "h": (p > 0.5).astype(float)})
    ga, gb = d.groupby("a"), d.groupby("b")
    f = pd.DataFrame({
        "p": p,
        "p_rk1": ga["p"].rank(ascending=False, method="min").values,
        "p_gap1": ga["p"].transform("max").values - p,
        "p_second1": _second_max(s1_i, p),
        "p_sum1": ga["p"].transform("sum").values,
        "p_n50_1": ga["h"].transform("sum").values,
        "p_rk2": gb["p"].rank(ascending=False, method="min").values,
        "p_gap2": gb["p"].transform("max").values - p,
        "p_n50_2": gb["h"].transform("sum").values,
    })
    return f.astype(np.float32)
