import sys; sys.path.insert(0,'/Users/mac/amazon-ml-2026/code/ber_v5')
import numpy as np, pandas as pd
from src.block_big import KEY_FIELDS
FAMS=list(KEY_FIELDS)
AR=pd.read_parquet('atrisk.parquet'); PP=pd.read_parquet('prune_pos.parquet')
alt=np.fmax.reduce([PP[c].values for c in FAMS[1:]])
PP['kscore']=np.where(np.isnan(PP.b_key.values), np.nan_to_num(alt)*0.5, PP.b_key.values)
ar=AR[AR.keep & AR.kprot].merge(PP[['s1_i','r_i','kscore']],on=['s1_i','r_i'])
S=pd.concat([pd.read_parquet(f'atrisk_suitors_{f}.parquet').assign(fam=f, strict=lambda d,f=f: d.rrk<=KEY_FIELDS[f][3]) for f in FAMS],ignore_index=True)
S=S[S.r_i.isin(set(ar.r_i))]
W=S.pivot_table(index=['r_i','s1_vis_i'],columns='fam',values='cos',aggfunc='max').reindex(columns=FAMS)
st=S.groupby(['r_i','s1_vis_i']).strict.any().reindex(W.index).values
bst=S[(S.fam=='b_key')&S.strict].set_index(['r_i','s1_vis_i']).index
W=W.reset_index()
bk=W.b_key.values; al=np.fmax.reduce([W[c].values for c in FAMS[1:]])
out=[]
for variant in ('strict','loose'):
    if variant=='strict':
        pb=pd.MultiIndex.from_frame(W[['r_i','s1_vis_i']]).isin(bst); mem=st
    else:
        pb=np.isfinite(bk); mem=np.ones(len(W),bool)
    ks=np.where(pb,np.nan_to_num(bk),np.nan_to_num(al)*0.5)
    T=pd.DataFrame({'r_i':W.r_i.values,'comp':W.s1_vis_i.values,'ks':ks})[mem]
    for G in (1,6):
        J=ar[['r_i','s1_vis_i','kscore']].merge(T,on='r_i')
        J=J[J.comp!=J.s1_vis_i]
        if G>1: J=J[(J.comp%G)==(J.s1_vis_i%G)]
        kb=(J.ks>J.kscore+1e-6).groupby([J.r_i]).sum()
        nb=ar.r_i.map(kb).fillna(0)
        out.append((variant,'global' if G==1 else f'grouped G={G}',len(ar),int((nb>1).sum()),round((nb>1).mean(),3)))
print(pd.DataFrame(out,columns=['suitors','protection','K-protected positives','lost at K','frac']).to_string(index=False))
