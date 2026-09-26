"""Dump GT pairs that blocking misses, for error analysis. Usage: python tools/block_misses.py <data_dir> <out.tsv>"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from src import blocking
from src.io_utils import load_split
from src.run import prepare, gold_dict, CFG

data, out = sys.argv[1], sys.argv[2]
frames, gt = load_split(data, "train")
s1, r = prepare(frames)
gold = gold_dict(gt, s1)
pairs = blocking.generate(s1, r, CFG["k_name"], CFG["k_addr"], CFG["k_comb"], CFG["k_rev"], CFG["partition_by_country"])
have = set(zip(s1.entity_id.values[pairs.s1_i], r.entity_id.values[pairs.r_i]))
ri = r.set_index("entity_id")
s1i = s1.set_index("entity_id")
rows = [(e, x) for e, v in gold.items() for x in v if (e, x) not in have]
cols = ["business_name", "business_address", "name_core", "addr_norm"]
df = pd.DataFrame([dict(s1=e, r=x, **{f"a_{c}": s1i.at[e, c] for c in cols}, **{f"b_{c}": ri.at[x, c] for c in cols})
                   for e, x in rows])
df.to_csv(out, sep="\t", index=False)
print(len(rows), "missed of", sum(len(v) for v in gold.values()))
