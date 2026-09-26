"""Mock-test world from train (Tony/goldmine-style): per country keep a random fraction f of entities (S1 + its GT
records) and the same fraction of unowned records, so S1 count ~ test's; then hide h of kept S1 (their records stay).
Recompute the shift.py statistics on the visible S1. Tests whether hide-style density simulation reproduces test's
excess same-name/different-house records."""
import sys, csv, numpy as np, pandas as pd
sys.path.insert(0, '/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/top1')
from shift import load_train, R
def stats(s1, pool, label):
    kh = pool.groupby(['h', 'hn']).size()
    pool_ne = pool[~pool.hn_empty]
    k_ne = pool_ne.groupby('h').size(); kh_ne = pool_ne.groupby(['h', 'hn']).size(); k_all = pool.groupby('h').size()
    ne = ~s1.hn_empty.values
    key = pd.Series(list(zip(s1.h, s1.hn)))
    same_hn = key.map(kh).fillna(0).values
    diff = s1.h.map(k_ne).fillna(0).values - key.map(kh_ne).fillna(0).values
    s1dup = s1.h.map(s1.groupby('h').size()).values - 1
    rare = ne & (s1dup == 0)
    d = diff[rare]; sh = same_hn[rare]; tot = s1.h.map(k_all).fillna(0).values[rare]
    print(f"{label:22s} S1 {len(s1):>9,} pool/S1 {len(pool)/len(s1):.2f} | unique-name S1 share {rare.mean():.3f} | "
          f"on unique-name S1: same-name recs mean {tot.mean():.3f}, same-name&SAME-house {sh.mean():.3f}, "
          f"same-name&DIFF-house mean {d.mean():.3f} P(>=1) {(d>=1).mean():.4f} P(>=2) {(d>=2).mean():.4f} | "
          f"all S1 w/ house: P(diff>=1) {(diff[ne]>=1).mean():.3f}", flush=True)
if __name__ == '__main__':
    tr1 = load_train('train_source1.tsv')
    ids1 = pd.read_csv(R + 'train_source1.tsv', sep='\t', dtype=str, usecols=[0], quoting=csv.QUOTE_NONE).entity_id.values
    trp = pd.concat([load_train('train_source2.tsv'), load_train('train_source3.tsv')], ignore_index=True)
    idsp = np.concatenate([pd.read_csv(R + f, sep='\t', dtype=str, usecols=[0], quoting=csv.QUOTE_NONE).entity_id.values
                           for f in ['train_source2.tsv', 'train_source3.tsv']])
    gt = pd.read_csv(R + 'train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    s1pos = pd.Series(np.arange(len(ids1)), index=ids1)
    owner = np.full(len(idsp), -1, np.int64)
    rpos = pd.Series(np.arange(len(idsp)), index=idsp)
    ex = gt.assign(m=gt.matched_entity_ids.str.split(',')).explode('m'); ex = ex[ex.m != '']
    owner[rpos.loc[ex.m.values].values] = s1pos.loc[ex.source1_entity_id.values].values
    del gt, ex, rpos, s1pos, idsp
    rs = np.random.RandomState(0)
    for cc, name, f in [(0, 'us', 663106 / 1323633), (1, 'india', 809986 / 883188)]:
        for h in [0.0, 0.19]:
            keep_ent = rs.rand(len(ids1)) < f
            vis = keep_ent & (rs.rand(len(ids1)) >= h)
            m1 = (tr1.c.values == cc)
            pm = (trp.c.values == cc) & np.where(owner >= 0, keep_ent[np.maximum(owner, 0)], rs.rand(len(owner)) < f)
            stats(tr1[m1 & vis], trp[pm], f'mock {name} f{f:.2f} h{h}')
