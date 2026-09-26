"""K-stage at test under per-quarter record-top-2, competitor K-score built exactly like 15's testmode.py:
only families that list the competitor within their v10 reverse top-kr count; b_key cosine if b_key lists it,
else 0.5 x best such other-family cosine. Reports the direct realised loss, and the IPW expectation incl. the
inflow of positives the cap cut in train."""
import sys; sys.path.insert(0,'/Users/mac/amazon-ml-2026/code/ber_v5')
import numpy as np, pandas as pd
from scipy.stats import binom
from src.block_big import KEY_FIELDS
FAMS=list(KEY_FIELDS); TOT=415630; NVIS=715441; Q_TR=119999/715440
AR=pd.read_parquet('atrisk.parquet'); PP=pd.read_parquet('prune_pos.parquet')
alt=np.fmax.reduce([PP[c].values for c in FAMS[1:]])
PP['kscore']=np.where(np.isnan(PP.b_key.values), np.nan_to_num(alt)*0.5, PP.b_key.values)
ar=AR[AR.keep & AR.kprot].merge(PP[['s1_i','r_i','kscore']],on=['s1_i','r_i'])
S=pd.concat([pd.read_parquet(f'atrisk_suitors_{f}.parquet').assign(fam=f) for f in FAMS],ignore_index=True)
S=S[S.r_i.isin(set(ar.r_i))]
S=S[S.rrk <= S.fam.map({f:min(int(sys.argv[1]),KEY_FIELDS[f][3]) for f in FAMS})]
W=S.pivot_table(index=['r_i','s1_vis_i'],columns='fam',values='cos',aggfunc='max').reindex(columns=FAMS).reset_index()
bk=W.b_key.values; al=np.fmax.reduce([W[c].values for c in FAMS[1:]])
ks=np.where(np.isfinite(bk), bk, np.nan_to_num(al)*0.5)
T=pd.DataFrame({'r_i':W.r_i.values,'comp':W.s1_vis_i.values,'ks':ks})
J=ar[['r_i','s1_vis_i','kscore']].merge(T,on='r_i'); J=J[J.comp!=J.s1_vis_i]
rng=lambda idx,p: np.searchsorted(np.linspace(0,NVIS,p+1).astype(np.int64), idx, side='right')-1
nb=ar.r_i.map((J.ks>J.kscore+1e-6).groupby(J.r_i).sum()).fillna(0).values
w=1/binom.cdf(1,nb,Q_TR)
print('competitor pairs',len(T),'| nb quantiles',np.quantile(nb,[.25,.5,.75,.9]).tolist(),
      f'| IPW-implied positives cut by the cap in train {int((w-1).sum())} (observed A 3396; A+A2 5473)')
for p in (1,4):
    Jq=J[rng(J.comp.values,p)==rng(J.s1_vis_i.values,p)]
    nbq=ar.r_i.map((Jq.ks>Jq.kscore+1e-6).groupby(Jq.r_i).sum()).fillna(0)
    print(f'direct realised, parts={p}: at-risk lost at K {int((nbq>1).sum())} of {len(ar)}')
for q,lab in ((0.25,'quarters q=.25'),(0.283,'quarters @ test density q=.283'),(1.0,'global')):
    pte=binom.cdf(1,nb,q); loss=(1-pte).sum(); inflow=(pte*(w-1)).sum()
    print(f'{lab:32s}: expected at-risk loss {loss:6.0f}, inflow of train-cut {inflow:6.0f} -> net {inflow-loss:+6.0f} = {(inflow-loss)/TOT*100:+.2f} PC pts at K')
