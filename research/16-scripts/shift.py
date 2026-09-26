"""Train vs test: per-S1 counts of pool records sharing the S1's name_core, split by house_no agree / differ.
Same normaliser primitives as prep (normalize.py lines 1-298 identical across versions)."""
import sys, csv, glob, numpy as np, pandas as pd, time
from multiprocessing import Pool
sys.path.insert(0, '/Users/mac/amazon-ml-2026/code/ber_v5')
from src.normalize import clean, _DBA, split_legal, canon_tokens, numbers, postcode_like, house_number
R = '/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/real/'
P = '/Users/mac/amazon-ml-2026/prep/test/'
def nc_hn(df):
    nm = df.business_name.map(clean).map(lambda s: _DBA.split(s)[0].strip())
    core = nm.map(lambda s: " ".join(canon_tokens(split_legal(s)[0])))
    nums = df.business_address.map(clean).map(numbers)
    hn = [house_number(n, postcode_like(n)) for n in nums]
    return pd.DataFrame({'c': df.country.map({'US': 0, 'India': 1}).fillna(2).astype(np.int8).values,
                         'h': pd.util.hash_array(core.values.astype(object)),
                         'hn': pd.util.hash_array(np.array(hn, dtype=object)), 'hn_empty': np.array([x == '' for x in hn])})
def load_train(f):
    rd = pd.read_csv(R + f, sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, chunksize=100_000)
    with Pool(2) as pool:
        return pd.concat(pool.imap(nc_hn, rd), ignore_index=True)
def load_test(tag, c):
    fs = sorted(glob.glob(f'{P}{tag}_{c}/*.parquet'))
    d = pd.concat([pd.read_parquet(f, columns=['name_core', 'house_no']) for f in fs], ignore_index=True)
    return pd.DataFrame({'h': pd.util.hash_array(d.name_core.values.astype(object)),
                         'hn': pd.util.hash_array(d.house_no.values.astype(object)), 'hn_empty': (d.house_no == '').values})
def stats(s1, pool, label):
    kh = pool.groupby(['h', 'hn']).size()
    pool_ne = pool[~pool.hn_empty]
    k_ne = pool_ne.groupby('h').size(); kh_ne = pool_ne.groupby(['h', 'hn']).size()
    k_all = pool.groupby('h').size()
    ne = ~s1.hn_empty.values
    key = pd.Series(list(zip(s1.h, s1.hn)))
    same_hn = key.map(kh).fillna(0).values
    diff = s1.h.map(k_ne).fillna(0).values - key.map(kh_ne).fillna(0).values
    s1dup = s1.h.map(s1.groupby('h').size()).values - 1
    rare = ne & (s1dup == 0)          # S1 name unique among S1s of the country, house number present
    d = diff[rare]; sh = same_hn[rare]; tot = s1.h.map(k_all).fillna(0).values[rare]
    print(f"{label:13s} S1 {len(s1):>9,} pool/S1 {len(pool)/len(s1):.2f} | unique-name S1 share {rare.mean():.3f} | "
          f"on unique-name S1: same-name recs mean {tot.mean():.3f}, same-name&SAME-house {sh.mean():.3f}, "
          f"same-name&DIFF-house mean {d.mean():.3f} P(>=1) {(d>=1).mean():.4f} P(>=2) {(d>=2).mean():.4f} | "
          f"all S1 w/ house: P(diff>=1) {(diff[ne]>=1).mean():.3f} median {np.median(diff[ne]):.0f}", flush=True)
if __name__ == '__main__':
    t = time.time()
    tr1 = load_train('train_source1.tsv'); print('train s1', time.time() - t, flush=True)
    trp = pd.concat([load_train('train_source2.tsv'), load_train('train_source3.tsv')], ignore_index=True)
    print('train pool', time.time() - t, flush=True)
    for c, cc in [('us', 0), ('india', 1)]:
        stats(tr1[tr1.c == cc], trp[trp.c == cc], f'train {c}')
        te1 = load_test('s1', c); tep = pd.concat([load_test('s2', c), load_test('s3', c)], ignore_index=True)
        stats(te1, tep, f'test {c}')
        del te1, tep
    te1 = load_test('s1', 'france'); tep = pd.concat([load_test('s2', 'france'), load_test('s3', 'france')], ignore_index=True)
    stats(te1, tep, 'test france')
