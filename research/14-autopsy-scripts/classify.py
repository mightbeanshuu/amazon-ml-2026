"""Join the exact miss set with per-family diagnostics and raw/normalised text; assign a mechanical cause and
noise-mode flags; print cause x frequency tables and knob recovery curves."""
import csv, glob, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
sys.path.insert(0, "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy")
from src.block_big import KEY_FIELDS
from rapidfuzz import fuzz
from fam_diag import gather, iter_s1, iter_r, AU

REAL = AU + "/../real"
FAMS = list(KEY_FIELDS)
import os
SRC = os.environ.get("SRC", "missed")
M = pd.read_parquet(f"{AU}/{SRC}.parquet")
if len(sys.argv) > 1 and sys.argv[1] == "diag":
    M = pd.read_parquet(f"{AU}/{SRC}_text.parquet")
n = len(M)
TEXT = not (len(sys.argv) > 1 and sys.argv[1] == "diag")
if TEXT:
    # ---- text: normalised fields for both sides + raw strings
    us, ur = np.unique(M.s1_vis_i.values), np.unique(M.r_i.values)
    TC = ["name_core", "name_phon", "name_ns", "addr_norm", "addr_nums", "addr_alpha"]
    import fam_diag
    fam_diag.COLS = sorted(set(fam_diag.COLS) | set(TC))
    FS = gather(us, iter_s1).set_axis(us)
    FR = gather(ur, iter_r).set_axis(ur)
    for c in TC:
        M["s_" + c] = FS[c].reindex(M.s1_vis_i.values).values
        M["r_" + c] = FR[c].reindex(M.r_i.values).values
    ids_s, ids_r = set(M.s1_id), set(M.r_id)
    raw = {}
    for fn, ids in (("train_source1.tsv", ids_s), ("train_source2.tsv", ids_r), ("train_source3.tsv", ids_r)):
        for ch in pd.read_csv(f"{REAL}/{fn}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                              chunksize=500000):
            ch = ch[ch.entity_id.isin(ids)]
            for e, a, b in zip(ch.entity_id, ch.business_name, ch.business_address):
                raw[e] = (a, b)
    M["s_raw_name"] = M.s1_id.map(lambda e: raw[e][0]); M["s_raw_addr"] = M.s1_id.map(lambda e: raw[e][1])
    M["r_raw_name"] = M.r_id.map(lambda e: raw[e][0]); M["r_raw_addr"] = M.r_id.map(lambda e: raw[e][1])

    # namesake density over VISIBLE S1 (what blocking competes against)
    nc = pd.concat([d.name_core for d, _ in iter_s1()], ignore_index=True).value_counts()
    M["s1_dup_name"] = M.s_name_core.map(nc).fillna(0).astype(int)
    M["r_named_s1"] = M.r_name_core.map(nc).fillna(0).astype(int)
    vocab = {}
    for k, v in nc.items():
        for t in k.split():
            vocab[t] = vocab.get(t, 0) + v
    INDIC = re.compile("[ऀ-ൿ]")
    M["r_addr_empty"] = M.r_addr_norm.str.strip() == ""
    M["s_addr_empty"] = M.s_addr_norm.str.strip() == ""
    M["r_native"] = M.r_raw_name.map(lambda s: bool(INDIC.search(s)))
    M["r_native_addr"] = M.r_raw_addr.map(lambda s: bool(INDIC.search(s)))
    M["r_handle"] = M.r_raw_name.str.contains(r"^\s*[#@]|www\.|\.(?:com|in|co|net|org)\b", case=False, regex=True)
    M["r_leet"] = M.r_raw_name.map(lambda s: any(re.search(r"[a-zA-Z]", t) and re.search(r"[0158]", t) for t in s.split()))
    M["r_name_empty"] = M.r_name_core.str.strip() == ""
    M["r_oov"] = M.r_name_core.map(lambda s: np.mean([vocab.get(t, 0) < 2 for t in s.split()]) if s.split() else np.nan)
    nsr = [fuzz.ratio(a, b) for a, b in zip(M.s_name_ns, M.r_name_ns)]
    tsr = [fuzz.token_set_ratio(a, b) for a, b in zip(M.s_name_core, M.r_name_core)]
    phr = [fuzz.token_set_ratio(a, b) for a, b in zip(M.s_name_phon, M.r_name_phon)]
    M["name_ns_ratio"], M["name_tset"], M["name_phon_tset"] = nsr, tsr, phr
    M["addr_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(M.s_addr_norm, M.r_addr_norm)]
    M["name_shared_tok"] = [len(set(a.split()) & set(b.split())) for a, b in zip(M.s_name_core, M.r_name_core)]


    M["scramble"] = [(a != b) and sorted(a) == sorted(b) for a, b in zip(M.s_name_ns, M.r_name_ns)]
    M["scramble_near"] = [fuzz.ratio("".join(sorted(a)), "".join(sorted(b))) for a, b in zip(M.s_name_ns, M.r_name_ns)]
    M["r_name_short"] = M.r_name_ns.str.len() <= 3
    M["r_addr_ntok"] = M.r_addr_alpha.str.split().map(len)
    M["addr_nums_shared"] = [len(set(a.split()) & set(b.split())) for a, b in zip(M.s_addr_nums, M.r_addr_nums)]


    def name_rel(r):
        if r.r_name_empty:
            return "r_name_empty"
        if r.s_name_ns == r.r_name_ns:
            return "exact"
        if sorted(r.s_name_core.split()) == sorted(r.r_name_core.split()):
            return "shuffle"
        if r.scramble or (r.scramble_near >= 90 and r.name_ns_ratio < 80):
            return "letter-scramble"
        if r.r_name_short:
            return "abbrev(<=3 chars)"
        if r.name_ns_ratio >= 80 or r.name_tset >= 90:
            return "typo/near"
        if r.name_phon_tset >= 80:
            return "phonetic/translit"
        if r.name_shared_tok > 0 or r.name_tset >= 60:
            return "partial"
        return "different(renamed/random)"


    M["name_rel"] = M.apply(name_rel, axis=1)
    M["addr_rel"] = np.where(M.r_addr_empty | M.s_addr_empty, "empty(either)",
                    np.where(M.addr_tset >= 80, "high>=80", np.where(M.addr_tset >= 50, "mid50-80", "low<50")))

    M.to_parquet(f"{AU}/{SRC}_text.parquet", index=False)
    print("text flags saved", flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "text":
    sys.exit(0)
D = {f: pd.read_parquet(f"{AU}/diag_{f}.parquet") for f in FAMS}
D = {f: d[d.ctl == 0].reset_index(drop=True) for f, d in D.items()}
for f in FAMS:
    kf, kr = KEY_FIELDS[f][2], KEY_FIELDS[f][3]
    d = D[f]
    M[f + "_cos"] = d.cos.values
    M[f + "_fr"] = d.fwd_rank.values
    M[f + "_rr"] = d.rev_rank.values
    M[f + "_in"] = d.in_f.values
    M[f + "_inge"] = d.in_f_ge.values
    M[f + "_fsat"] = d.fwd_sat.values
    M[f + "_rsat"] = d.rev_sat.values
    M[f + "_frge"] = d.fwd_rank_ge.values
    M[f + "_rrge"] = d.rev_rank_ge.values
    M[f + "_ratio"] = np.minimum(d.fwd_rank.values / kf, d.rev_rank.values / kr)
    M[f + "_nsh"] = d.n_sh.values
    M[f + "_alive"] = d.n_alive.values
    M[f + "_maxdf"] = d.n_maxdf.values
M["any_in"] = M[[f + "_in" for f in FAMS]].any(axis=1)
M["any_inge"] = M[[f + "_inge" for f in FAMS]].any(axis=1)
M["any_cos"] = (M[[f + "_cos" for f in FAMS]] > 0).any(axis=1)
M["any_sh"] = (M[[f + "_nsh" for f in FAMS]] > 0).any(axis=1)
M["best_ratio"] = M[[f + "_ratio" for f in FAMS]].min(axis=1)
M["best_fam"] = M[[f + "_ratio" for f in FAMS]].idxmin(axis=1).str.replace("_ratio", "")
cause = np.where(M.any_inge, "A_capped_K40",
          np.where(M.any_in, "A2_tie_straddles_k(or capped)",
          np.where(~M.any_sh, "B_no_shared_key",
          np.where(~M.any_cos, "C_shared_keys_all_df_killed",
          np.where(M.best_ratio <= 2, "D_near_rank(<=2x k)",
          np.where(M.best_ratio <= 5, "E_mid_rank(2-5x k)", "F_far_rank(>5x k)"))))))
M["cause"] = cause

M.to_parquet(f"{AU}/missed_classified.parquet", index=False)

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60)
print("n missed", n)
print(M.cause.value_counts().sort_index().to_string())
print(pd.crosstab(M.cause, M.name_rel, margins=True).to_string())
print(pd.crosstab(M.cause, M.addr_rel, margins=True).to_string())
for flag in ["r_addr_empty", "s_addr_empty", "r_native", "r_handle", "r_leet", "r_name_empty"]:
    print(flag, round(M[flag].mean(), 4), "| by cause:", M.groupby("cause")[flag].mean().round(3).to_dict())
print("best family among D/E/F:", M[M.cause.str[0].isin(list("DEF"))].best_fam.value_counts().to_dict())
