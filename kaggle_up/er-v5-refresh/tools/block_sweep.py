"""Blocking recall sweep: python tools/block_sweep.py <data_dir> '<json list of cfg overrides>'"""
import json, sys, time
sys.path.insert(0, ".")
from src import blocking
from src.io_utils import load_split
from src.run import prepare, gold_dict, cands_dict, CFG
from src.metrics import blocking_report
frames, gt = load_split(sys.argv[1], "train")
s1, r = prepare(frames)
gold = gold_dict(gt, s1)
mats = blocking.build_matrices(s1, r)
for ov in json.loads(sys.argv[2]):
    c = dict(dict(k_name=15, k_addr=10, k_comb=20, k_rev=5, k_loc=15), **ov)
    t = time.time()
    p = blocking.generate(s1, r, c["k_name"], c["k_addr"], c["k_comb"], c["k_rev"], True, mats, c["k_loc"])
    print(ov, blocking_report(cands_dict(p, s1, r), gold, len(r)), f"{time.time()-t:.0f}s", flush=True)
