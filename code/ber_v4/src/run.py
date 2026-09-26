"""End-to-end pipeline: data -> normalise -> block -> prune (stage 1) -> match (stage 2) -> collective (stage 3)
-> decision -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python -m src.run --data <dir containing train/ and test/> --out <output dir> [--eda-only]

Everything is fitted on the provided training data. Test-time statistics (TF-IDF/IDF) come from the provided
test records. There are no external data, API or network calls.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import blocking, decide, features, model
from .io_utils import load_split, validate, write_lists
from .metrics import blocking_report, macro_breakdown, parse_ids
from .normalize import normalize_frame

CFG = dict(k_name=15, k_addr=10, k_comb=20, k_rev=10, k_loc=25, partition_by_country=True, K_final=12, n_splits=5,
           seeds=(0,), unseen_country_shift=0.05)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def prepare(frames):
    s1, s2, s3 = (normalize_frame(f.copy()).reset_index(drop=True) for f in frames)
    r = pd.concat([s2, s3], ignore_index=True)
    return s1, r


def gold_dict(gt, s1):
    g = {e: parse_ids(v) for e, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    for e in s1.entity_id:           # every S1 is scored, even if absent from the GT file
        g.setdefault(e, set())
    return g


def eda(s1, r, gold, name):
    rep = {"split": name, "n_s1": len(s1), "n_r": len(r),
           "n_s2": int(r.entity_id.str.startswith("S2").sum()), "n_s3": int(r.entity_id.str.startswith("S3").sum()),
           "s1_by_country": s1.country.value_counts().to_dict(), "r_by_country": r.country.value_counts().to_dict()}
    if gold is not None:
        ng = pd.Series({e: len(v) for e, v in gold.items()})
        rep["singleton_rate"] = round(float((ng == 0).mean()), 4)
        rep["matches_per_s1_hist"] = ng.value_counts().sort_index().to_dict()
        per_src = pd.DataFrame([(e, sum(x.startswith("S2") for x in v), sum(x.startswith("S3") for x in v))
                                for e, v in gold.items()], columns=["e", "s2", "s3"])
        rep["max_s2_per_s1"] = int(per_src.s2.max())
        rep["max_s3_per_s1"] = int(per_src.s3.max())
        owners = pd.Series([x for v in gold.values() for x in v]).value_counts()
        rep["r_ids_matched_to_multiple_s1"] = int((owners > 1).sum())
        cmap = dict(zip(r.entity_id, r.country_norm))
        c1 = dict(zip(s1.entity_id, s1.country_norm))
        pairs = [(e, x) for e, v in gold.items() for x in v]
        rep["gt_pairs"] = len(pairs)
        rep["gt_pairs_unknown_r"] = sum(1 for e, x in pairs if x not in cmap)
        rep["gt_pairs_cross_country"] = sum(1 for e, x in pairs if x in cmap and cmap[x] != c1.get(e))
    rep["script_non_ascii_names"] = int(s1.business_name.str.contains(r"[^\x00-\x7f]").sum()
                                        + r.business_name.str.contains(r"[^\x00-\x7f]").sum())
    return rep


def label_pairs(pairs, s1, r, gold):
    a = s1.entity_id.values[pairs.s1_i.values]
    b = r.entity_id.values[pairs.r_i.values]
    return np.fromiter((y in gold[x] for x, y in zip(a, b)), bool, len(a))


def cands_dict(pairs, s1, r):
    d = {}
    for a, b in zip(s1.entity_id.values[pairs.s1_i.values], r.entity_id.values[pairs.r_i.values]):
        d.setdefault(a, set()).add(b)
    return d


def report_by_country(pairs, s1, r, gold):
    out = {"ALL": blocking_report(cands_dict(pairs, s1, r), gold, len(r))}
    cd = cands_dict(pairs, s1, r)
    for c in s1.country_norm.unique():
        ids = set(s1.entity_id[s1.country_norm == c])
        out[c] = blocking_report({e: v for e, v in cd.items() if e in ids}, {e: v for e, v in gold.items() if e in ids},
                                 int((r.country_norm == c).sum()))
    return out


def prune(pairs, score, K):
    rk = pd.Series(score).groupby(pairs.s1_i.values).rank(ascending=False, method="first").values
    keep = rk <= K
    out = pairs[keep].reset_index(drop=True)
    blocking.add_rank_features(out, blocking.RANK_COLS)   # recompute on the pruned set
    return out, keep


S1_COLS = None


def stage1_cols(pairs):
    return [c for c in pairs.columns if c not in ("s1_i", "r_i")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eda-only", action="store_true")
    ap.add_argument("--cfg", default="{}", help="JSON overrides for CFG")
    a = ap.parse_args()
    cfg = dict(CFG, **json.loads(a.cfg))
    os.makedirs(a.out, exist_ok=True)
    runlog = {"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}}
    seeds = tuple(cfg["seeds"])

    # ---------------- train ----------------
    (tr_frames, gt) = load_split(a.data, "train")
    s1, r = prepare(tr_frames)
    gold = gold_dict(gt, s1)
    runlog["eda_train"] = eda(s1, r, gold, "train")
    te_frames, te_gt = load_split(a.data, "test")
    s1t, rt = prepare(te_frames)
    runlog["eda_test"] = eda(s1t, rt, None, "test")
    log("EDA", json.dumps({k: runlog[k] for k in ("eda_train", "eda_test")}, default=str)[:3000])
    if a.eda_only:
        json.dump(runlog, open(os.path.join(a.out, "run_log.json"), "w"), indent=1, default=str)
        return

    log("blocking train")
    pairs = blocking.generate(s1, r, cfg["k_name"], cfg["k_addr"], cfg["k_comb"], cfg["k_rev"],
                              cfg["partition_by_country"], k_loc=cfg["k_loc"])
    y_all = label_pairs(pairs, s1, r, gold)
    runlog["block_union"] = report_by_country(pairs, s1, r, gold)
    log("union", runlog["block_union"]["ALL"])

    # stage 1: cheap pruner on blocking features -> top-K per S1 (this is candidate_pairs.tsv)
    c1 = stage1_cols(pairs)
    grp = pairs.s1_i.values
    p1_oof, it1 = model.oof_predict(pairs[c1], y_all.astype(int), grp, 3, params=model.STAGE1_PARAMS)
    pr_pairs, keep = prune(pairs, p1_oof, cfg["K_final"])
    y = y_all[keep]
    runlog["block_pruned"] = report_by_country(pr_pairs, s1, r, gold)
    log("pruned", runlog["block_pruned"]["ALL"])
    m1 = model.fit_full(pairs[c1], y_all.astype(int), it1, params=model.STAGE1_PARAMS)

    # stage 2: full pair features
    log("features train", len(pr_pairs))
    idf_n = features.idf_table(s1.name_core, r.name_core)
    idf_a = features.idf_table(s1.addr_norm, r.addr_norm)
    t0 = time.time()
    X = features.build(pr_pairs, s1, r, idf_n, idf_a)
    log(f"features built in {time.time() - t0:.1f}s, shape {X.shape}")
    g2 = pr_pairs.s1_i.values
    p2_oof, it2 = model.oof_predict(X, y.astype(int), g2, cfg["n_splits"], seeds=seeds)
    country = s1.country_norm.values[g2]
    log(f"stage2 oof done, best iter {it2}")
    p2_loco = model.loco_predict(X, y.astype(int), country, it2)
    log("stage2 loco done")

    # stage 3: collective re-scoring
    C_oof = pd.concat([X, model.collective_features(g2, pr_pairs.r_i.values, p2_oof)], axis=1)
    p3_oof, it3 = model.oof_predict(C_oof, y.astype(int), g2, cfg["n_splits"], seeds=seeds)
    C_loco = pd.concat([X, model.collective_features(g2, pr_pairs.r_i.values, np.nan_to_num(p2_loco))], axis=1)
    p3_loco = np.full(len(y), np.nan)
    for c in np.unique(country):
        tr_, va_ = country != c, country == c
        if tr_.sum() and y[tr_].sum():
            mm = model.fit_full(C_oof[tr_], y[tr_].astype(int), it3)
            p3_loco[va_] = model.predict(mm, C_loco[va_])

    # decision tuning
    s1_codes = g2
    ng = np.array([len(gold[e]) for e in s1.entity_id])
    ev = decide.Evaluator(s1_codes, y, ng, len(s1))
    res = {}
    for nm, p in (("stage2_oof", p2_oof), ("stage3_oof", p3_oof)):
        res[nm] = decide.tune_rule(ev, s1_codes, pr_pairs.r_i.values, p)
    iso = decide.fit_calibrator(p3_oof, y)
    sel_gfm = decide.gfm_select(s1_codes, pr_pairs.r_i.values, iso.predict(p3_oof))
    res["stage3_gfm"] = (ev.score(sel_gfm), {"rule": "gfm"})
    best_rule = res["stage3_oof"][1]
    # LOCO: apply the OOF-tuned rule to LOCO probabilities = proxy for an unseen country
    for nm, p in (("stage2_loco", p2_loco), ("stage3_loco", p3_loco)):
        pp = np.nan_to_num(p)
        sel = decide.rule_select(s1_codes, pp, best_rule["t1"], best_rule["t2"], best_rule["delta"],
                                 decide.exclusive_mask(pr_pairs.r_i.values, pp) if best_rule["excl"] else None)
        res[nm + "_with_oof_rule"] = ev.score(sel)
        for shift in (0.05, 0.1):
            sel = decide.rule_select(s1_codes, pp, best_rule["t1"], best_rule["t2"], best_rule["delta"],
                                     decide.exclusive_mask(pr_pairs.r_i.values, pp) if best_rule["excl"] else None,
                                     t_shift=np.full(len(pp), shift))
            res[f"{nm}_shift{shift}"] = ev.score(sel)
    sel = decide.rule_select(s1_codes, p3_oof, best_rule["t1"], best_rule["t2"], best_rule["delta"],
                             decide.exclusive_mask(pr_pairs.r_i.values, p3_oof) if best_rule["excl"] else None)
    pred = {}
    for a_, b_ in zip(s1.entity_id.values[g2[sel]], r.entity_id.values[pr_pairs.r_i.values[sel]]):
        pred.setdefault(a_, set()).add(b_)
    res["oof_breakdown"] = macro_breakdown(pred, gold, dict(zip(s1.entity_id, s1.country_norm)))
    # error analysis dump: false merges (FP) and missed matches (FN) at the chosen rule, hardest first
    err = pd.DataFrame({"s1": s1.entity_id.values[g2], "r": r.entity_id.values[pr_pairs.r_i.values], "p": p3_oof,
                        "y": y, "sel": sel})
    err = err[(err.sel != err.y)].copy()
    err["type"] = np.where(err.sel, "FP", "FN")
    for side, frame, col in (("a", s1, "s1"), ("b", r, "r")):
        m = frame.set_index("entity_id")
        for c in ("business_name", "business_address", "country"):
            err[f"{side}_{c}"] = m.loc[err[col], c].values
    err.sort_values("p", ascending=False).to_csv(os.path.join(a.out, "oof_errors.tsv"), sep="\t", index=False)
    runlog["cv"] = res
    log("CV", json.dumps(res, default=str))

    # ---------------- test ----------------
    log("fit full models")
    m2 = model.fit_full(X, y.astype(int), it2, seeds=seeds)
    m3 = model.fit_full(C_oof, y.astype(int), it3, seeds=seeds)
    log("blocking test")
    tpairs = blocking.generate(s1t, rt, cfg["k_name"], cfg["k_addr"], cfg["k_comb"], cfg["k_rev"],
        cfg["partition_by_country"], k_loc=cfg["k_loc"])
    tp1 = model.predict(m1, tpairs[c1])
    tpr, _ = prune(tpairs, tp1, cfg["K_final"])
    idf_nt = features.idf_table(s1t.name_core, rt.name_core)
    idf_at = features.idf_table(s1t.addr_norm, rt.addr_norm)
    Xt = features.build(tpr, s1t, rt, idf_nt, idf_at)
    tp2 = model.predict(m2, Xt)
    Ct = pd.concat([Xt, model.collective_features(tpr.s1_i.values, tpr.r_i.values, tp2)], axis=1)
    tp3 = model.predict(m3, Ct)
    seen = set(s1.country_norm)
    shift = np.where(np.isin(s1t.country_norm.values[tpr.s1_i.values], list(seen)), 0.0,
                     cfg["unseen_country_shift"])
    tsel = decide.rule_select(tpr.s1_i.values, tp3, best_rule["t1"], best_rule["t2"], best_rule["delta"],
                              decide.exclusive_mask(tpr.r_i.values, tp3) if best_rule["excl"] else None, t_shift=shift)
    cand = cands_dict(tpr, s1t, rt)
    tpred = {}
    for a_, b_ in zip(s1t.entity_id.values[tpr.s1_i.values[tsel]], rt.entity_id.values[tpr.r_i.values[tsel]]):
        tpred.setdefault(a_, set()).add(b_)
    out_dir = os.path.join(a.out, "output")
    os.makedirs(out_dir, exist_ok=True)
    mp, cp = os.path.join(out_dir, "matching_results.tsv"), os.path.join(out_dir, "candidate_pairs.tsv")
    write_lists(mp, "source1_entity_id", "matched_entity_ids", s1t.entity_id, tpred)
    write_lists(cp, "source1_entity_id", "candidate_entity_ids", s1t.entity_id, cand)
    runlog["test"] = {"n_candidates": int(len(tpr)), "n_pred_links": int(tsel.sum()),
                      "frac_s1_nonempty": round(len(tpred) / len(s1t), 4),
                      "by_country_nonempty": {c: round(float(np.mean([e in tpred for e in s1t.entity_id[
                          s1t.country_norm == c]])), 4) for c in s1t.country_norm.unique()},
                      "candidate_file_MB": round(os.path.getsize(cp) / 1e6, 2)}
    issues = validate(mp, cp, os.path.join(a.data, "test"))
    runlog["validation"] = issues or "PASS"
    hid = os.path.join(a.data, "test", "_hidden_test_ground_truth.tsv")   # synthetic smoke-test only
    if os.path.exists(hid):
        from .io_utils import read_tsv
        tg = gold_dict(read_tsv(hid), s1t)
        runlog["synthetic_test_score"] = macro_breakdown(tpred, tg, dict(zip(s1t.entity_id, s1t.country_norm)))
        runlog["synthetic_test_block"] = report_by_country(tpr, s1t, rt, tg)
    log("TEST", json.dumps(runlog["test"]), "validation:", runlog["validation"])
    if "synthetic_test_score" in runlog:
        log("SYNTH TEST", runlog["synthetic_test_score"])
    json.dump(runlog, open(os.path.join(a.out, "run_log.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
