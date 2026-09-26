"""Anchor-merge re-retrieval: after confident assignment, search the still-unmatched S2/S3 records against
ENRICHED entity text (S1 tokens ∪ its confident matches' tokens).

A record whose surface form shares almost nothing with its S1 (random name, truncated address) often shares
plenty with a SIBLING that was matched confidently — the enriched entity carries that sibling's tokens, so a
plain lexical search now finds the link (the Foursquare-winner stage: +0.0056 CV there).

Deterministic candidate generation (rule-based retrieval), so the new pairs belong in candidate_pairs.tsv;
they are scored by the same first-stage model as every other candidate.
"""
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.preprocessing import normalize

from .block_big import _HV, _topk_chunks, group_rank


def _entity_docs(s1, s1_i, r_i, conf, cap=60):
    """Enriched key-doc per S1 row: S1 name/addr tokens plus its confident matches' tokens (n:/a: prefixed)."""
    name = [set("n:" + t for t in x.split()) for x in s1.name_core.values]
    addr = [set("a:" + t for t in x.split()) for x in s1.addr_norm.values]
    return name, addr


def anchor_retrieve(s1, r, s1_i, r_i, p, conf, k=4, max_df_abs=50_000, log=print):
    """Search UNRESOLVED records against enriched entity docs. Returns (s1_idx, r_idx, cosine) for new pairs.
    conf: boolean per pair marking confident assignments (the anchors)."""
    import time
    t = time.time()
    name, addr = _entity_docs(s1, s1_i, r_i, conf)
    ci = np.flatnonzero(conf)
    for kk in ci:
        a, b = s1_i[kk], r_i[kk]
        if len(name[a]) < 60:
            name[a] |= set("n:" + t for t in r.name_core.values[b].split())
        if len(addr[a]) < 60:
            addr[a] |= set("a:" + t for t in r.addr_norm.values[b].split())
    docs = [" ".join(sorted(nm | ad)) for nm, ad in zip(name, addr)]
    del name, addr
    # unresolved records: not the confident member of any pair
    resolved = np.zeros(len(r), bool)
    resolved[r_i[ci]] = True
    un = np.flatnonzero(~resolved)
    rdocs = ["n:" + " n:".join(x.split()) + " a:" + " a:".join(y.split()) if x or y else ""
             for x, y in zip(r.name_core.values[un], r.addr_norm.values[un])]
    A = _HV.transform(pd.Series(rdocs).values)                      # queries: unresolved records
    B = _HV.transform(pd.Series(docs).values)                       # index: enriched entities
    df = np.bincount(B.indices, minlength=B.shape[1]) + np.bincount(A.indices, minlength=A.shape[1])
    w = (np.log((A.shape[0] + B.shape[0] + 1) / (df + 1)) + 1).astype(np.float32)
    w[(df > max_df_abs) | (df < 2)] = 0
    for M in (A, B):
        M.data = w[M.indices]
        M.eliminate_zeros()
    A, B = normalize(A, copy=False), normalize(B, copy=False)
    qs, es, vs = [], [], []
    step = 400_000
    for lo in range(0, A.shape[0], step):
        i, j, v = _topk_chunks(A[lo:lo + step], B, k)
        qs.append(i + lo)
        es.append(j)
        vs.append(v)
    q, e, v = np.concatenate(qs), np.concatenate(es), np.concatenate(vs)
    log(f"  anchor retrieve: {len(q)} pairs for {len(un)} unresolved records, {time.time() - t:.0f}s")
    return e.astype(np.int32), un[q].astype(np.int32), v.astype(np.float32)
