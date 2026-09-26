"""Learn a token lexicon from the TRAIN ground truth (full cleaned names, legal words included): S2/S3 name tokens that are not S1 vocabulary (phonetic
renderings of names written in Indian scripts, e.g. 'praivet', 'teknolojis', 'laiph') mapped to the S1 token they
stand for ('private', 'technologies', 'life'). Uses only provided train data.

    python -m tools.mine_lexicon --prep prep --data student_resource/dataset --out src/lexicon.json

For each true pair, each out-of-vocabulary R token is aligned to the S1 name token with the most similar phonetic
key (rapidfuzz ratio >= min_sim). A mapping is kept when it was seen at least min_support times and is the token's
dominant alignment (share >= min_share).
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import parse_ids  # noqa: E402
from src.normalize import phon_key  # noqa: E402
from src.run_big import countries, load, load_r  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument("--min_share", type=float, default=0.6)
    ap.add_argument("--min_sim", type=float, default=70)
    ap.add_argument("--countries", default="india")
    a = ap.parse_args()
    gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE)
    counts = defaultdict(Counter)
    for c in [x for x in countries(a.prep, "train") if x in a.countries.split(",")]:
        s1 = load(a.prep, "train", "s1", c, ["entity_id", "name_clean"])
        r = load_r(a.prep, "train", c, ["entity_id", "name_clean"])
        vocab = Counter(t for x in s1.name_clean.values for t in x.split())
        s1n = dict(zip(s1.entity_id.values, s1.name_clean.values))
        rn = dict(zip(r.entity_id.values, r.name_clean.values))
        del s1, r
        n = 0
        for e, m in zip(gt.source1_entity_id.values, gt.matched_entity_ids.values):
            a_name = s1n.get(e)
            if a_name is None or not m:
                continue
            at = a_name.split()
            if not at:
                continue
            ap_ = [phon_key(t) for t in at]
            for x in parse_ids(m):
                b_name = rn.get(x)
                if not b_name:
                    continue
                for t in b_name.split():
                    if vocab.get(t, 0) >= 3 or len(t) < 3 or t.isdigit():   # known S1 word: nothing to learn
                        continue
                    pt = phon_key(t)
                    best, bs = None, 0.0
                    for s, ps in zip(at, ap_):
                        sc = 100.0 if ps == pt else fuzz.ratio(ps, pt)
                        if sc > bs:
                            best, bs = s, sc
                    ratio = len(t) / max(len(best or "x"), 1)
                    concat = best is not None and best in t and len(t) > len(best) + 2      # 'silvertechnologies'
                    if best is not None and bs >= a.min_sim and best != t and 0.6 <= ratio <= 1.4 and not concat:
                        counts[t][best] += 1
            n += 1
        log(f"[{c}] aligned {n} S1, {len(counts)} candidate tokens so far")
        del s1n, rn, vocab
    lex = {}
    for t, cnt in counts.items():
        s, k = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if k >= a.min_support and k / tot >= a.min_share:
            lex[t] = s
    json.dump(lex, open(a.out, "w"), indent=0, sort_keys=True)
    log(f"lexicon {len(lex)} entries -> {a.out}")
    for t in ["praivet", "praibhet", "teknolojis", "laiph", "limiteda", "elaelapi", "sarvisej", "indastris"]:
        print(t, "->", lex.get(t))


if __name__ == "__main__":
    main()
