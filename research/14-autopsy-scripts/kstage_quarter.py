import sys; sys.path.insert(0,'/Users/mac/amazon-ml-2026/code/ber_v5')
import numpy as np, pandas as pd
from src.block_big import KEY_FIELDS
FAMS=list(KEY_FIELDS); NVIS=715441
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
rng=lambda idx,p: np.searchsorted(np.linspace(0,NVIS,p+1).astype(np.int64), idx, side='right')-1
out=[]
for variant in ('strict','loose'):
    pb = pd.MultiIndex.from_frame(W[['r_i','s1_vis_i']]).isin(bst) if variant=='strict' else np.isfinite(bk)
    mem = st if variant=='strict' else np.ones(len(W),bool)
    ks=np.where(pb,np.nan_to_num(bk),np.nan_to_num(al)*0.5)
    T=pd.DataFrame({'r_i':W.r_i.values,'comp':W.s1_vis_i.values,'ks':ks})[mem]
    J=ar[['r_i','s1_vis_i','kscore']].merge(T,on='r_i'); J=J[J.comp!=J.s1_vis_i]
    for p in (1,3,4):
        Jq=J[rng(J.comp.values,p)==rng(J.s1_vis_i.values,p)]
        nb=ar.r_i.map((Jq.ks>Jq.kscore+1e-6).groupby(Jq.r_i).sum()).fillna(0)
        out.append((variant,p,round(len(Jq)/len(J),3),len(ar),int((nb>1).sum()),round((nb>1).mean(),3)))
print(pd.DataFrame(out,columns=['suitors','S1 parts at K','share of suitors in range','K-protected','lost at K','frac']).to_string(index=False))
