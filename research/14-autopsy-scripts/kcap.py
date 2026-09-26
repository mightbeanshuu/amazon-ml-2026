"""K=40 cap (prune_topk) knobs for the A/A2 misses, against the real v10 India table."""
import sys; sys.path.insert(0,'/Users/mac/amazon-ml-2026/code/ber_v5')
import pandas as pd, numpy as np
from src.block_big import KEY_FIELDS
FAMS=list(KEY_FIELDS); TOT=415630
M0=pd.read_parquet('missed_full.parquet',columns=['s1_i','cause'])
S1A=sorted(set(M0[M0.cause.str.startswith('A')].s1_i.tolist()))
P=pd.read_parquet('box/train_block_india.parquet',columns=['s1_i','r_i','y']+FAMS+['b_rank'],filters=[('s1_i','in',S1A)])
print('pairs of S1s with A/A2 misses', len(P), 'S1s', len(S1A))
fam=P[FAMS].notna().any(axis=1).values
P=P[fam].reset_index(drop=True)
alt=np.fmax.reduce([P[c].values for c in FAMS[1:]])
bk=P.b_key.values
P['k_old']=np.where(np.isnan(bk), np.nan_to_num(alt)*0.5, bk)
P['k_nohalf']=np.where(np.isnan(bk), np.nan_to_num(alt), bk)
P['k_max']=np.fmax(np.nan_to_num(bk,nan=0), np.nan_to_num(alt))
nf=P[FAMS].notna().sum(axis=1).values
print('retained family pairs (these S1s)', len(P), 'per S1', round(len(P)/len(S1A),2), '| mean #families proposing a retained pair', round(nf.mean(),2))
M=pd.read_parquet('missed_full.parquet')
A=M[M.cause.isin(['A_capped_K40','A2_tie_straddles_k(or capped)'])].copy()
cert=A.cause.eq('A_capped_K40').values
# which families proposed the missed pair (certain -> ge; tie -> strict)
prop={f:np.where(cert, A[f+'_inge'].values, A[f+'_in'].values) for f in FAMS}
cos={f:A[f+'_cos'].values for f in FAMS}
a_alt=np.fmax.reduce([np.where(prop[f],cos[f],np.nan) for f in FAMS[1:]])
a_bk=np.where(prop['b_key'],cos['b_key'],np.nan)
A['k_old']=np.where(np.isnan(a_bk), np.nan_to_num(a_alt)*0.5, a_bk)
A['k_nohalf']=np.where(np.isnan(a_bk), np.nan_to_num(a_alt), a_bk)
A['k_max']=np.fmax(np.nan_to_num(a_bk,nan=0), np.nan_to_num(a_alt))
A['by_bkey']=prop['b_key']
g=P.groupby('s1_i')
def higher(col):
    srt={k:np.sort(v)[::-1] for k,v in g[col]}
    return np.array([ (srt.get(s,np.array([]))>x+1e-7).sum() for s,x in zip(A.s1_i.values,A[col].values)])
for col in ['k_old','k_nohalf','k_max']:
    A['hi_'+col]=higher(col)
print('A/A2 misses', len(A), '(certain', int(cert.sum()), ') proposed by b_key:', int(A.by_bkey.sum()))
print('under the v10 K-score, #retained pairs of the S1 scoring higher: quantiles', A.hi_k_old.quantile([.1,.25,.5,.75,.9]).to_dict())
rows=[]
for K2 in (40,60,80,120):
    for col in ['k_old','k_nohalf','k_max']:
        ok=A['hi_'+col]<K2
        rows.append((K2,col,int(ok[cert].sum()),int(ok.sum())))
print(pd.DataFrame(rows,columns=['K','kscore','recov_certainA(upper bd)','recov_A+A2(upper bd)']).to_string(index=False))
# exempt-family-top-m from the cap
for m in (1,2,3,5):
    ok=np.zeros(len(A),bool)
    for f in FAMS:
        fr=np.where(cert,A[f+'_frge'],A[f+'_fr']); rr=np.where(cert,A[f+'_rrge'],A[f+'_rr'])
        ok|=(fr<=m)|(rr<=m)
    print(f'exempt every family fwd/rev top-{m} from the K cap: recovers {int(ok[cert].sum())} certain A, {int(ok.sum())} A+A2')
A.to_parquet('kcap_A.parquet',index=False)
