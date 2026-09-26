"""Full-scale pipeline (millions of records), memory-bounded and restartable.

    python -m src.run_big --prep <prep> --data <dataset> --out <run> --phase all

Phases (each writes its results to disk, so a crash loses at most one step):
  train   For each train country, block ALL of that country's S1 against all its S2+S3 (the reverse search
          needs every S1), keep a random S1 sample's candidate pairs, build pair features and label them.
          Saved as train_features.parquet and reused if present. LightGBM with GroupKFold-by-S1 OOF and
          leave-one-country-out (LOCO). The decision rule is tuned for macro F0.5 on OOF. Models and meta are
          saved to models/.
  test    ONE country per process: block, cap the candidates, build features in chunks, score, apply the
          exclusivity-aware decision rule (stricter thresholds for country labels unseen in training). Writes
          test_<c>.parquet with (s1_id, r_id, p, sel). The candidates are exactly these rows.
  final   Merge all countries, write output/matching_results.tsv and output/candidate_pairs.tsv, validate.
  all     train (if models are missing), then each test country in a fresh subprocess, then final.
"""
import argparse
import csv
import gc
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import decide, features, model
from .block_big import KEY_FIELDS, block_country, group_rank

from .metrics import blocking_report, parse_ids
from .normalize import phon_line, renormalize_names
from .siblings import expand_pairs, sibling_edges, vouch_features

CFG = dict(train_per_country=60_000, K=40, max_df_keys=3000, kscale=1.0, n_splits=3, unseen_shift=0.05,
           feat_workers=4, test_chunk_s1=40_000, seeds=[0], lgb={"learning_rate": 0.1},
           prune_k=20, hide_frac=0.19, sib_k=3, sib_min_sim=0.55, sib_top_s1=5, sib_weak_q=0.30)

VOUCH_COLS = ["p", "vouch", "n_vouch", "sib_best", "p_rk1", "p_gap1", "p_second1", "p_sum1", "p_n50_1",
              "p_rk2", "p_gap2", "p_n50_2"]


def stage_b_frame(s1_i, r_i, pA, sib, n_r):
    """Stage-B features: stage-A p, sibling vouching, and the pair's standing among its competitors."""
    f = model.collective_features(s1_i, r_i, pA)
    vouch, n_vouch, sib_best = vouch_features(np.asarray(s1_i), np.asarray(r_i), np.asarray(pA), sib, n_r)
    f["vouch"], f["n_vouch"], f["sib_best"] = vouch, n_vouch, sib_best
    return f[VOUCH_COLS]


def load_sib(path):
    z = np.load(path)
    return z["q"], z["s"], z["v"]


def add_expansion(pairs, sib, n_r, cfg, log):
    """Deterministic sibling candidate expansion (part of candidate generation, before the first scoring model).
    Expanded pairs carry sib_sim; their blocking columns stay NaN and the similarity prune judges them by text."""
    ext = expand_pairs(pairs, sib, n_r, top_s1=cfg["sib_top_s1"], min_sim=cfg["sib_min_sim"],
                       weak_q=cfg["sib_weak_q"])
    pairs["sib_sim"] = np.float32(0)
    if ext is None:
        return pairs
    have = set(zip(pairs.s1_i.values.tolist(), pairs.r_i.values.tolist()))
    m = [(a, b) not in have for a, b in zip(ext.s1_i.values.tolist(), ext.r_i.values.tolist())]
    ext = ext[np.array(m, bool)]
    log(f"  sibling expansion: +{len(ext)} pairs")
    if not len(ext):
        return pairs
    extra = pd.DataFrame({c: np.full(len(ext), np.nan, np.float32) for c in pairs.columns if c not in
                          ("s1_i", "r_i", "sib_sim")})
    extra["s1_i"] = ext.s1_i.values.astype(pairs.s1_i.dtype)
    extra["r_i"] = ext.r_i.values.astype(pairs.r_i.dtype)
    extra["sib_sim"] = ext.sib_sim.values
    out = pd.concat([pairs, extra], ignore_index=True).sort_values("s1_i", kind="stable").reset_index(drop=True)
    out["n_cand_s1"] = np.bincount(out.s1_i.values)[out.s1_i.values].astype(np.float32)
    return out

# Blocking output columns (fused per-field cosines, ranks, R-side margins, table-level ambiguity counts).
EXTRA_PAIR = ["b_xkey", "b_xkey_rk", "b_key_rmarg", "b_nkey_rmarg", "b_akey_rmarg", "b_xkey_rmarg", "s1_dup_name",
              "s1_dup_addr", "r_named_s1", "r_name_oov", "sib_sim"]
PAIR_COLS = ["s1_i", "r_i", "b_key", "b_nkey", "b_akey", "b_key_rk", "b_nkey_rk", "b_akey_rk", "b_best", "b_rank",
             "n_cand_s1"] + EXTRA_PAIR + ["prune_score"]        # candidate-generation output fed to the matcher


def prune_by_similarity(pairs, s1, r, k, chunk=2_000_000):
    """The LAST candidate-generation step, a fixed rule (no learned model): score each blocked pair with
    max(name token-set, no-space name ratio, phonetic token-set) + address token-set + 50 x combined-key cosine,
    and keep each S1's top-k. The survivors are exactly what the matcher scores, i.e. candidate_pairs.tsv.
    pairs must be sorted by s1_i; order is preserved."""
    from rapidfuzz import fuzz, process
    cp = lambda x, y, sc: process.cpdist(x, y, scorer=sc, workers=-1, dtype=np.float32)
    score = np.empty(len(pairs), np.float32)
    si, ri = pairs.s1_i.values, pairs.r_i.values
    for lo in range(0, len(pairs), chunk):
        a, b = si[lo:lo + chunk], ri[lo:lo + chunk]
        A = {c: s1[c].values[a].tolist() for c in ("name_core", "name_phon", "addr_norm")}
        B = {c: r[c].values[b].tolist() for c in ("name_core", "name_phon", "addr_norm")}
        ns_a = [x.replace(" ", "") for x in A["name_core"]]
        ns_b = [x.replace(" ", "") for x in B["name_core"]]
        name = np.maximum.reduce([cp(A["name_core"], B["name_core"], fuzz.token_set_ratio), cp(ns_a, ns_b, fuzz.ratio),
                                  cp(A["name_phon"], B["name_phon"], fuzz.token_set_ratio)])
        score[lo:lo + chunk] = name + cp(A["addr_norm"], B["addr_norm"], fuzz.token_set_ratio) \
            + 50 * np.nan_to_num(pairs.b_key.values[lo:lo + chunk])
    keep = group_rank(si, score) <= k
    out = pairs[keep].reset_index(drop=True)
    out["prune_score"] = score[keep]
    return out


def train_split(n, cfg, seed=0):
    """Train S1 rows visible to blocking (the rest are hidden, so their S2/S3 records become unmatched
    distractors: test has ~40% unmatched S2/S3 records against ~26% in train), and the labelled sample among the
    visible rows. Deterministic, so both train passes see the same split."""
    rs = np.random.RandomState(seed)
    visible = np.sort(np.flatnonzero(rs.rand(n) >= cfg["hide_frac"]))
    samp = np.sort(rs.choice(len(visible), min(cfg["train_per_country"], len(visible)), replace=False))
    return visible, samp


def add_dup_counts(pairs, s1, r):
    """Table-level ambiguity, counted over ALL S1 of the country: how many S1 share this S1's exact name and its
    address signature, and how many S1 carry exactly this S2/S3 record's name. A unique exact name with a missing
    address is almost surely a match; a shared one is a namesake trap."""
    def cnt(keys, table, empty):
        c = np.array(keys.map(table).values, np.float32)            # a writable copy (pandas 3 is copy-on-write)
        c[(keys == empty).values] = np.nan
        return c
    nc = s1.name_core.value_counts()
    ak1 = s1.addr_alpha + "|" + s1.addr_nums
    s1_name = cnt(s1.name_core, nc, "")
    s1_addr = cnt(ak1, ak1.value_counts(), "|")
    r_name = np.nan_to_num(cnt(r.name_core, nc, ""), nan=0.0)
    r_name[(r.name_core == "").values] = np.nan
    pairs["s1_dup_name"] = s1_name[pairs.s1_i.values]
    pairs["s1_dup_addr"] = s1_addr[pairs.s1_i.values]
    pairs["r_named_s1"] = r_name[pairs.r_i.values]
    # name randomness: share of the record's name tokens that no two S1 names of the country use ('lumhalo',
    # handles, garbled transliterations). High means the name carries no evidence and the address must decide.
    vocab = pd.Series([t for x in s1.name_core.values for t in x.split()]).value_counts()
    known = set(vocab.index[vocab.values >= 2])
    oov = np.array([np.nan if not x else sum(t not in known for t in x.split()) / len(x.split())
                    for x in r.name_core.values], np.float32)
    pairs["r_name_oov"] = oov[pairs.r_i.values]
    return pairs


def atomic_parquet(df, path):
    """Write then rename, so a killed process never leaves a truncated checkpoint behind."""
    df.to_parquet(path + ".tmp", compression="zstd", index=False)
    os.replace(path + ".tmp", path)


def atomic_npy(arr, path):
    with open(path + ".tmp", "wb") as fh:
        np.save(fh, arr)
    os.replace(path + ".tmp", path)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def countries(prep, split):
    return sorted({os.path.basename(p).split("_", 1)[1] for p in glob.glob(f"{prep}/{split}/s1_*")})


BLOCK_COLS = ["entity_id", "name_clean", "name_core", "name_trade", "name_phon", "name_ns", "addr_alpha",
              "addr_nums"]


_LEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lexicon.json")
_LEX = json.load(open(_LEX_PATH)) if os.path.exists(_LEX_PATH) else {}   # learned from train GT (tools/mine_lexicon)


def load(prep, split, tag, c, columns=None):
    """Load a normalised table. Name columns derived from name_core are rebuilt here: the token lexicon and the
    phonetic key were improved after prep ran, and both train and test go through this same path."""
    p = f"{prep}/{split}/{tag}_{c}"
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=columns)
    if _LEX and "name_clean" in df.columns:
        renormalize_names(df, _LEX)
    if "name_core" in df.columns:
        if "name_phon" in df.columns:
            df["name_phon"] = df["name_core"].map(phon_line)
        if "name_ns" in df.columns:
            df["name_ns"] = df["name_core"].str.replace(" ", "", regex=False)
    return df


def load_r(prep, split, c, columns=None):
    parts = [x for x in (load(prep, split, "s2", c, columns), load(prep, split, "s3", c, columns)) if x is not None]
    return pd.concat(parts, ignore_index=True)


def prune_topk(pairs, K):
    """Cap candidates per S1 at K, ranked by the combined-key cosine (pairs proposed only by a single-aspect search
    get half their best cosine as a stand-in)."""
    alt = np.fmax(np.fmax(pairs["b_nkey"].values, pairs["b_akey"].values), pairs["b_xkey"].values)
    score = np.where(np.isnan(pairs["b_key"].values), np.nan_to_num(alt) * 0.5, pairs["b_key"].values)
    rk = group_rank(pairs["s1_i"].values, score.astype(np.float32))
    out = pairs[rk <= K].copy()
    out["b_rank"] = rk[rk <= K]
    out["n_cand_s1"] = np.bincount(out.s1_i.values)[out.s1_i.values].astype(np.float32)
    return out.sort_values(["s1_i", "b_rank"]).reset_index(drop=True)


def block(s1, r, cfg, keep_s1=None, post=None):
    return block_country(s1, r, fields=KEY_FIELDS, kscale=cfg["kscale"], max_df_keys=cfg["max_df_keys"], log=log,
                         keep_s1=keep_s1, post=post, r_parts=cfg.get("r_parts", 3), s1_parts=cfg.get("s1_parts", 4))


def idf_for(s1, r):
    return features.idf_table(s1.name_core, r.name_core), features.idf_table(s1.addr_norm, r.addr_norm)


# ================================================================== TRAIN
def block_train(a, cfg):
    """Pass 1: for each train country, block all VISIBLE S1 (hide_frac of S1 are hidden so that their S2/S3
    records act as unmatched distractors, as in test) against all its S2+S3, keep the labelled sample's top-K
    candidates. Saved as train_block.parquet."""
    path = f"{a.out}/train_block.parquet"
    if os.path.exists(path):
        log("reusing", path)
        return pd.read_parquet(path)
    gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE)
    gold_all = dict(zip(gt.source1_entity_id, gt.matched_entity_ids))
    del gt
    parts, s1_off, r_off, gold_sizes, blk = [], 0, 0, {}, {}
    for c in countries(a.prep, "train"):
        cpath = f"{a.out}/train_block_{c}.parquet"
        if os.path.exists(cpath):                                  # computed by a train_block subprocess
            log(f"[train {c}] reusing {cpath}")
            pairs = pd.read_parquet(cpath)
            meta_c = json.load(open(f"{a.out}/train_block_{c}.json"))
            gold_sizes[c], blk[c] = meta_c["gold_sizes"], meta_c["blk"]
            pairs["s1_g"] = pairs.s1_i.values + s1_off
            pairs["r_g"] = pairs.r_i.values + r_off
            parts.append(pairs)
            s1_off += meta_c["n_samp"]
            r_off += meta_c["n_r"]
            continue
        s1_all = load(a.prep, "train", "s1", c, BLOCK_COLS).reset_index(drop=True)
        visible, samp = train_split(len(s1_all), cfg)
        s1_full = s1_all.iloc[visible].reset_index(drop=True)
        del s1_all
        r = load_r(a.prep, "train", c, BLOCK_COLS)
        log(f"[train {c}] visible s1 {len(s1_full)} r {len(r)} sample {len(samp)}")
        keep = np.zeros(len(s1_full), bool)
        keep[samp] = True
        pairs = block(s1_full, r, cfg, keep_s1=keep)
        pairs = add_dup_counts(pairs, s1_full, r)
        remap = np.full(len(s1_full), -1)
        remap[samp] = np.arange(len(samp))
        pairs["s1_i"] = remap[pairs.s1_i.values].astype(np.int32)
        s1_ids = s1_full.entity_id.values[samp]
        del s1_full
        pairs = prune_topk(pairs, cfg["K"])
        sib = sibling_edges(r, k=cfg["sib_k"], max_df_keys=cfg["max_df_keys"], log=log)
        np.savez_compressed(f"{a.out}/train_{c}_sib.npz", q=sib[0], s=sib[1], v=sib[2])
        pairs = add_expansion(pairs, sib, len(r), cfg, log)
        del sib
        g = {e: parse_ids(gold_all.get(e, "")) for e in s1_ids}
        e1, e2 = s1_ids[pairs.s1_i.values], r.entity_id.values[pairs.r_i.values]
        cd = {}
        for x, y in zip(e1, e2):
            cd.setdefault(x, set()).add(y)
        blk[c] = blocking_report(cd, g, len(r))
        log(f"[train {c}] blocking after top-{cfg['K']}", blk[c])
        pairs["y"] = [y in g[x] for x, y in zip(e1, e2)]
        pairs["s1_id"] = e1
        pairs["country"] = c
        gold_sizes[c] = {e: len(v) for e, v in g.items()}
        blk[c]["r_off"], blk[c]["n_r"] = r_off, len(r)
        json.dump({"gold_sizes": gold_sizes[c], "blk": blk[c], "n_samp": len(samp), "n_r": len(r)},
                  open(f"{a.out}/train_block_{c}.json", "w"))
        atomic_parquet(pairs, cpath)
        pairs["s1_g"] = pairs.s1_i.values + s1_off
        pairs["r_g"] = pairs.r_i.values + r_off
        parts.append(pairs)
        s1_off += len(samp)
        r_off += len(r)
        del r
        gc.collect()
        if getattr(a, "country", None):                            # one-country subprocess mode
            return None
    B = pd.concat(parts, ignore_index=True)
    atomic_parquet(B, path)
    json.dump(gold_sizes, open(f"{a.out}/train_sample_gold_sizes.json", "w"))
    json.dump(blk, open(f"{a.out}/train_blocking.json", "w"), indent=1)
    return B


def build_train(a, cfg):
    """Pass 2: per country, the similarity prune (exactly as at test time), then pair features for the survivors."""
    path = f"{a.out}/train_features.parquet"
    if os.path.exists(path):
        log("reusing", path)
        return pd.read_parquet(path)
    B = block_train(a, cfg)
    parts, rep = [], {}
    for c in countries(a.prep, "train"):
        Bc = B[B.country == c].sort_values(["s1_i", "b_rank"]).reset_index(drop=True)
        s1_all = load(a.prep, "train", "s1", c).reset_index(drop=True)
        visible, samp = train_split(len(s1_all), cfg)
        s1_vis = s1_all.iloc[visible].reset_index(drop=True)
        del s1_all
        r = load_r(a.prep, "train", c)
        idf_n, idf_a = idf_for(s1_vis, r)
        s1 = s1_vis.iloc[samp].reset_index(drop=True)
        del s1_vis
        n0, pos0 = len(Bc), int(Bc.y.sum())
        Bc = prune_by_similarity(Bc, s1, r, cfg["prune_k"])
        rep[c] = dict(pairs_before=n0, pairs_after=len(Bc), per_s1=round(len(Bc) / len(samp), 2),
                      positives_kept=round(float(Bc.y.sum()) / max(pos0, 1), 4))
        log(f"[train {c}] similarity prune to top-{cfg['prune_k']}", rep[c])
        t = time.time()
        X = features.build_parallel(Bc[PAIR_COLS], s1, r, idf_n, idf_a, workers=cfg["feat_workers"])
        log(f"[train {c}] features {X.shape} in {time.time() - t:.0f}s")
        X["y"] = Bc.y.values
        X["s1_i"] = Bc.s1_g.values
        X["r_i"] = Bc.r_g.values
        X["s1_id"] = Bc.s1_id.values
        X["country"] = c
        parts.append(X)
        del r, idf_n, idf_a
        gc.collect()
    json.dump(rep, open(f"{a.out}/train_prune.json", "w"), indent=1)
    T = pd.concat(parts, ignore_index=True)
    atomic_parquet(T, path)
    return T


def train_phase(a, cfg):
    rl = {"cfg": cfg}
    T = build_train(a, cfg)
    gold_sizes = json.load(open(f"{a.out}/train_sample_gold_sizes.json"))
    rl["blocking"] = json.load(open(f"{a.out}/train_blocking.json"))
    feat_cols = [c for c in T.columns if c not in ("y", "s1_i", "r_i", "s1_id", "country")]
    y = T.y.values.astype(int)
    grp = T.s1_i.values
    log(f"train pairs {len(T)}, positives {y.mean():.3f}, features {len(feat_cols)}")
    pA, itA = model.oof_predict(T[feat_cols], y, grp, cfg["n_splits"], params=cfg["lgb"], seeds=tuple(cfg["seeds"]))
    log(f"stage A oof done (iter {itA})")
    pA_loco = np.nan_to_num(model.loco_predict(T[feat_cols], y, T.country.values, itA, params=cfg["lgb"]))
    log("stage A loco done")
    atomic_npy(pA, f"{a.out}/oof_pA.npy")                         # for decision-rule labs (tools/rule_lab)
    atomic_npy(pA_loco, f"{a.out}/loco_pA.npy")
    # evaluator over ALL sampled S1 (including those whose true matches were never blocked)
    s1_ids = [e for c in gold_sizes for e in gold_sizes[c]]
    code = {e: i for i, e in enumerate(s1_ids)}
    ng = np.array([gold_sizes[c][e] for c in gold_sizes for e in gold_sizes[c]])
    s1c = np.array([code[e] for e in T.s1_id.values])
    ev = decide.Evaluator(s1c, T.y.values, ng, len(s1_ids))
    res = {"A_oof": decide.tune_rule(ev, s1c, T.r_i.values, pA)}
    # ---- stage B: sibling vouching + competitor standing on the OOF probabilities (leak-safe)
    blk_meta = rl["blocking"]
    XB_parts, orderB = [], []
    for c in gold_sizes:
        m = np.flatnonzero((T.country == c).values)
        r_local = T.r_i.values[m] - blk_meta[c]["r_off"]
        sib = load_sib(f"{a.out}/train_{c}_sib.npz")
        XB_parts.append(stage_b_frame(T.s1_i.values[m], r_local, pA[m], sib, blk_meta[c]["n_r"]))
        orderB.append(m)
    orderB = np.concatenate(orderB)
    XB = pd.concat(XB_parts, ignore_index=True).iloc[np.argsort(orderB)].reset_index(drop=True)
    del XB_parts
    pB, itB = model.oof_predict(XB, y, T.s1_i.values, cfg["n_splits"], params={"learning_rate": 0.1,
                                                                               "num_leaves": 63})
    log(f"stage B oof done (iter {itB})")
    res["B_oof"] = decide.tune_rule(ev, s1c, T.r_i.values, pB)
    log("stage B rule", res["B_oof"])
    use_b = res["B_oof"][0] > res["A_oof"][0] + 1e-4
    mB = model.fit_full(XB, y, itB, params={"learning_rate": 0.1, "num_leaves": 63})
    best = (res["B_oof"] if use_b else res["A_oof"])[1]
    pSel = pB if use_b else pA
    for sh in (0.0, 0.05, 0.1):
        sel = decide.rule_select(s1c, pA_loco, best["t1"], best["t2"], best["delta"],
                                 decide.exclusive_mask(T.r_i.values, pA_loco) if best["excl"] else None,
                                 t_shift=np.full(len(pA_loco), sh))
        res[f"A_loco_shift{sh}"] = round(ev.score(sel), 5)
    res["A_loco_tuned"] = decide.tune_rule(ev, s1c, T.r_i.values, pA_loco)
    sel = decide.rule_select(s1c, pSel, best["t1"], best["t2"], best["delta"],
                             decide.exclusive_mask(T.r_i.values, pSel) if best["excl"] else None)
    fe = ev.score(sel, per_entity=True)
    res["oof_by_country"] = {c: round(float(fe[[code[e] for e in gold_sizes[c]]].mean()), 5) for c in gold_sizes}
    rl["cv"] = res
    log("CV", json.dumps(res, default=str))
    err = T.loc[sel != T.y.values, ["s1_id", "r_i", "country", "y"]].copy()
    err["p"] = pSel[sel != T.y.values]
    err.sort_values("p", ascending=False).head(200000).to_parquet(f"{a.out}/oof_errors.parquet")
    log("fit full model")
    mA = model.fit_full(T[feat_cols], y, itA, params=cfg["lgb"], seeds=tuple(cfg["seeds"]))
    os.makedirs(f"{a.out}/models", exist_ok=True)
    for i, m in enumerate(mA):
        m.save_model(f"{a.out}/models/A_{i}.txt")
    for i, m in enumerate(mB):
        m.save_model(f"{a.out}/models/B_{i}.txt")
    meta = dict(feat_cols=feat_cols, rule=best, seen=sorted(set(T.country)), n_models=len(mA), cfg=cfg,
                use_b=bool(use_b), n_models_b=len(mB))
    json.dump(meta, open(f"{a.out}/models/meta.json", "w"), indent=1)
    json.dump(rl, open(f"{a.out}/run_log_train.json", "w"), indent=1, default=str)
    log("saved models")


# ================================================================== TEST (two processes per country)
def test_block_phase(a, cfg, c):
    """Process 1: candidate generation only, with only the blocking columns loaded. Saves the checkpoint."""
    ppath = f"{a.out}/test_{c}_pairs.parquet"
    s1 = load(a.prep, "test", "s1", c, BLOCK_COLS).reset_index(drop=True)
    r = load_r(a.prep, "test", c, BLOCK_COLS)
    log(f"[test {c}] s1 {len(s1)} r {len(r)}")
    pairs = block(s1, r, cfg, post=lambda d: prune_topk(d, cfg["K"]))      # merged + pruned per S1 range
    pairs = add_dup_counts(pairs, s1, r)
    sib = sibling_edges(r, k=cfg["sib_k"], max_df_keys=cfg["max_df_keys"], log=log)
    np.savez_compressed(f"{a.out}/test_{c}_sib.npz", q=sib[0], s=sib[1], v=sib[2])
    pairs = add_expansion(pairs, sib, len(r), cfg, log)                    # expanded rows: NaN dup cols, as in train
    del sib
    pairs = pairs.sort_values(["s1_i", "b_rank"]).reset_index(drop=True)
    atomic_parquet(pairs, ppath)
    log(f"[test {c}] candidates {len(pairs)} ({len(pairs) / len(s1):.1f} per S1)")


def test_phase(a, cfg, c):
    """Process 2: features, scoring and decision for one country (resumes from checkpoints)."""
    import lightgbm as lgb
    meta = json.load(open(f"{a.out}/models/meta.json"))
    models = [lgb.Booster(model_file=f"{a.out}/models/A_{i}.txt") for i in range(meta["n_models"])]
    feat_cols, rule = meta["feat_cols"], meta["rule"]
    ppath, spath = f"{a.out}/test_{c}_pairs.parquet", f"{a.out}/test_{c}_scores.npy"
    pairs = pd.read_parquet(ppath)
    s1 = load(a.prep, "test", "s1", c, features.NEEDED).reset_index(drop=True)
    r = load_r(a.prep, "test", c, features.NEEDED)
    n0 = len(pairs)
    pairs = prune_by_similarity(pairs, s1, r, meta["cfg"]["prune_k"])[PAIR_COLS]
    log(f"[test {c}] similarity prune keeps {len(pairs)} of {n0} ({len(pairs) / len(s1):.2f} per S1)")
    log(f"[test {c}] scoring {len(pairs)} candidates")
    idf_n, idf_a = idf_for(s1, r)
    fpool = features.FeaturePool(idf_n, idf_a, cfg["feat_workers"])
    del idf_n, idf_a
    p = np.full(len(pairs), np.nan, np.float32)
    if os.path.exists(spath):                                   # checkpoint: resume scoring
        p = np.load(spath)
        log(f"[test {c}] resuming scores, {int(np.isfinite(p).sum())} already done")
    s1v = pairs.s1_i.values
    bounds = np.searchsorted(s1v, np.arange(0, len(s1) + cfg["test_chunk_s1"], cfg["test_chunk_s1"]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo or np.isfinite(p[lo:hi]).all():
            continue
        Xc = fpool.build(pairs.iloc[lo:hi], s1, r)
        p[lo:hi] = model.predict(models, Xc[feat_cols])
        del Xc
        atomic_npy(p, spath)
        log(f"[test {c}] scored {hi}/{len(pairs)}")
    fpool.close()
    if meta.get("use_b"):
        mB = [lgb.Booster(model_file=f"{a.out}/models/B_{i}.txt") for i in range(meta["n_models_b"])]
        sib = load_sib(f"{a.out}/test_{c}_sib.npz")
        XB = stage_b_frame(pairs.s1_i.values, pairs.r_i.values, p, sib, len(r))
        p = model.predict(mB, XB).astype(np.float32)
        del XB, sib
        log(f"[test {c}] stage B applied")
    shift = 0.0 if c in meta["seen"] else cfg["unseen_shift"]
    sel = decide.rule_select(pairs.s1_i.values, p, rule["t1"], rule["t2"], rule["delta"],
                             decide.exclusive_mask(pairs.r_i.values, p) if rule["excl"] else None,
                             t_shift=np.full(len(p), shift, np.float32))
    out = pd.DataFrame({"s1_id": s1.entity_id.values[pairs.s1_i.values],
                        "r_id": r.entity_id.values[pairs.r_i.values], "p": p, "sel": sel})
    atomic_parquet(out, f"{a.out}/test_{c}.parquet")
    nl = np.bincount(pairs.s1_i.values[sel], minlength=len(s1))
    stats = {"n_s1": len(s1), "n_cand": len(pairs), "links": int(sel.sum()), "mean_links": round(float(nl.mean()), 3),
             "frac_empty": round(float((nl == 0).mean()), 4), "shift": shift}
    json.dump(stats, open(f"{a.out}/test_{c}_stats.json", "w"))
    for f in (ppath, spath):                                     # checkpoints no longer needed
        if os.path.exists(f):
            os.remove(f)
    log(f"[test {c}] done", stats)


# ================================================================== FINAL
def _stream_country(path, fm, fc, seen):
    """Writes one matching row and one candidate row per S1 found in test_<c>.parquet, streaming record batches.
    Rows are contiguous per S1 (pairs are sorted by s1_i), so only the current S1's ids are held in memory."""
    import pyarrow.parquet as pq
    cur, cand, match = None, [], []

    def flush():
        if cur in seen:
            raise ValueError(f"{path}: S1 {cur} is not contiguous")
        seen.add(cur)
        fm.write(f"{cur}\t{','.join(dict.fromkeys(match))}\n")
        fc.write(f"{cur}\t{','.join(dict.fromkeys(cand))}\n")

    for b in pq.ParquetFile(path).iter_batches(batch_size=1_000_000, columns=["s1_id", "r_id", "sel"]):
        for e, x, s in zip(b.column(0).to_pylist(), b.column(1).to_pylist(), b.column(2).to_pylist()):
            if e != cur:
                if cur is not None:
                    flush()
                cur, cand, match = e, [], []
            cand.append(x)
            if s:
                match.append(x)
    if cur is not None:
        flush()


def final_phase(a):
    """Streams every country's decisions into the two submission files (the whole candidate set is ~60M ids,
    far too many to hold as Python objects), then runs the official validator on the leaderboard file."""
    od = f"{a.out}/output"
    os.makedirs(od, exist_ok=True)
    seen, n_s1 = set(), 0
    with open(f"{od}/matching_results.tsv.tmp", "w", encoding="utf-8", newline="") as fm, \
            open(f"{od}/candidate_pairs.tsv.tmp", "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for c in countries(a.prep, "test"):
            _stream_country(f"{a.out}/test_{c}.parquet", fm, fc, seen)
            ids = load(a.prep, "test", "s1", c, ["entity_id"])["entity_id"].values
            n_s1 += len(ids)
            for e in ids:                                     # S1 rows blocking found no candidates for
                if e not in seen:
                    seen.add(e)
                    fm.write(f"{e}\t\n")
                    fc.write(f"{e}\t\n")
            log(f"[final] {c} written, {len(seen)} S1 so far")
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        os.replace(f"{od}/{f}.tmp", f"{od}/{f}")
    official = os.path.join(a.data, "..", "utils", "validate_submission.py")
    v = subprocess.run([sys.executable, official, "--matching", f"{od}/matching_results.tsv",
                        "--test-dir", f"{a.data}/test"], capture_output=True, text=True)
    stats = {c: json.load(open(f"{a.out}/test_{c}_stats.json")) for c in countries(a.prep, "test")}
    rl = {"test": stats, "n_s1": n_s1, "n_written": len(seen),
          "validation": "PASS" if v.returncode == 0 else "FAIL", "validator_tail": v.stdout[-1500:]}
    json.dump(rl, open(f"{a.out}/run_log_final.json", "w"), indent=1, default=str)
    log("FINAL", json.dumps(rl, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="all", choices=["train", "train_block", "test_block", "test", "final",
                                                       "all"])
    ap.add_argument("--country")
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    cfg = dict(CFG, **json.loads(a.cfg))
    os.makedirs(a.out, exist_ok=True)
    if a.phase == "train":
        train_phase(a, cfg)
    elif a.phase == "train_block":
        block_train(a, cfg)
    elif a.phase == "test_block":
        test_block_phase(a, cfg, a.country)
    elif a.phase == "test":
        test_phase(a, cfg, a.country)
    elif a.phase == "final":
        final_phase(a)
    else:
        base = [sys.executable, "-X", "faulthandler", "-u", "-m", "src.run_big", "--prep", a.prep, "--data", a.data,
                "--out", a.out, "--cfg", a.cfg]
        if not os.path.exists(f"{a.out}/models/meta.json"):
            if not os.path.exists(f"{a.out}/train_block.parquet"):
                for c in countries(a.prep, "train"):               # one process per country: memory resets
                    if not os.path.exists(f"{a.out}/train_block_{c}.parquet"):
                        subprocess.run(base + ["--phase", "train_block", "--country", c], check=True)
            subprocess.run(base + ["--phase", "train"], check=True)
        for c in countries(a.prep, "test"):
            if not os.path.exists(f"{a.out}/test_{c}.parquet"):
                if not os.path.exists(f"{a.out}/test_{c}_pairs.parquet"):
                    subprocess.run(base + ["--phase", "test_block", "--country", c], check=True)
                subprocess.run(base + ["--phase", "test", "--country", c], check=True)
        subprocess.run(base + ["--phase", "final"], check=True)


if __name__ == "__main__":
    main()
