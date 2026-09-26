"""Expected K-stage outcome at test under per-range record-top-2, INCLUDING positives the cap cut in train that a
different random pool rescues. Each K-protected positive has nb = #suitors (all visible S1) with a higher K-score.
Train keeps it w.p. p(nb, 0.1677) = P(Bin(nb,q)<=1); the kept set is IPW-reweighted to the full population."""
import sys; sys.path.insert(0,'/Users/mac/amazon-ml-2026/code/ber_v5')
import numpy as np, pandas as pd
from scipy.stats import binom
from src.block_big import KEY_FIELDS
FAMS=list(KEY_FIELDS); TOT=415630
AR=pd.read_parquet('atrisk.parquet'); PP=pd.read_parquet('prune_pos.parquet')
alt=np.fmax.reduce([PP[c].values for c in FAMS[1:]])
PP['kscore']=np.where(np.isnan(PP.b_key.values), np.nan_to_num(alt)*0.5, PP.b_key.values)
ar=AR[AR.keep & AR.kprot].merge(PP[['s1_i','r_i','kscore']],on=['s1_i','r_i'])
S=pd.concat([pd.read_parquet(f'atrisk_suitors_{f}.parquet').assign(fam=f, strict=lambda d,f=f: d.rrk<=KEY_FIELDS[f][3]) for f in FAMS],ignore_index=True)
S=S[S.r_i.isin(set(ar.r_i))]
W=S.pivot_table(index=['r_i','s1_vis_i'],columns='fam',values='cos',aggfunc='max').reindex(columns=FAMS)
st=S.groupby(['r_i','s1_vis_i']).strict.any().reindex(W.index).values
bst=S[(S.fam=='b_key')&S.strict].set_index(['r_i','s1_vis_i']).index
W=W.reset_index(); bk=W.b_key.values; al=np.fmax.reduce([W[c].values for c in FAMS[1:]])
p=lambda nb,q: binom.cdf(1, nb, q)
Q_TR=119999/715440
for variant in ('strict','loose'):
    pb = pd.MultiIndex.from_frame(W[['r_i','s1_vis_i']]).isin(bst) if variant=='strict' else np.isfinite(bk)
    mem = st if variant=='strict' else np.ones(len(W),bool)
    ks=np.where(pb,np.nan_to_num(bk),np.nan_to_num(al)*0.5)
    T=pd.DataFrame({'r_i':W.r_i.values,'comp':W.s1_vis_i.values,'ks':ks})[mem]
    J=ar[['r_i','s1_vis_i','kscore']].merge(T,on='r_i'); J=J[J.comp!=J.s1_vis_i]
    nb=ar.r_i.map((J.ks>J.kscore+1e-6).groupby(J.r_i).sum()).fillna(0).values
    ptr=p(nb,Q_TR); w=1/ptr
    print(f'[{variant}] kept K-protected {len(ar)} | nb quantiles', np.quantile(nb,[.1,.25,.5,.75,.9]).tolist(),
          f'| IPW-implied positives CUT by the cap in train: {int((w-1).sum())} (observed A {3396}, A+A2 {5473})')
    for q,lab in ((0.25,'quarters (q=.25)'),(0.283,'quarters at test density (q=.283)'),(1.0,'global (q=1)'),(Q_TR,'train pool (check)')):
        exp_surv=(p(nb,q)*w).sum()
        print(f'   {lab:36s} expected K survivors {exp_surv:8.0f} | vs train-kept {len(ar)}: net {exp_surv-len(ar):+7.0f} = {(exp_surv-len(ar))/TOT*100:+.2f} PC pts'
              f'  [direct at-risk-only loss would be {int(((1-p(nb,q))).sum())}]')
