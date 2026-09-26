import sys; sys.path.insert(0,'/Users/mac/amazon-ml-2026/code/ber_v5')
import pandas as pd, numpy as np
from src.block_big import KEY_FIELDS
FAMS=list(KEY_FIELDS); TOT=415630
M=pd.read_parquet('missed_classified.parquet')
M=M.merge(pd.read_parquet('ayan_rank_missed.parquet'),on=['s1_id','r_id'])
rs=np.random.RandomState(0); n_s1=883188
vis=np.flatnonzero(rs.rand(n_s1)>=0.19); samp=np.sort(rs.choice(len(vis),120000,replace=False))
M['s1_i']=np.searchsorted(samp, M.s1_vis_i.values)
assert (samp[M.s1_i.values]==M.s1_vis_i.values).all()
PP=pd.read_parquet('prune_pos.parquet',columns=['s1_i','thr28','n_cand'])
thr=PP.groupby('s1_i').thr28.first(); ncand=PP.groupby('s1_i').n_cand.first()
M['thr28']=M.s1_i.map(thr); M['s1_ncand']=M.s1_i.map(ncand)
nm=np.maximum.reduce([M.name_tset.values,M.name_ns_ratio.values,M.name_phon_tset.values]).astype(float)
bk=np.where(M.b_key_inge|M.b_key_in, M.b_key_cos, 0.0)
M['pscore']=nm+M.addr_tset.values+50*bk
M['prune_ok']=np.where(M.thr28.isna(), np.nan, (M.pscore>=M.thr28).astype(float))   # NaN: S1 has <28 cands or no positive row
M.loc[M.s1_ncand<28,'prune_ok']=1.0
M.to_parquet('missed_full.parquet',index=False)
pd.set_option('display.width',250)
g=M.groupby('cause')
T=pd.DataFrame({'n':g.size(),'share':(g.size()/len(M)).round(3),'PC_pts':(g.size()/TOT*100).round(3),
  'ayan@8':g.apply(lambda d:(d.ayan_rank<=8).mean()).round(3),'ayan@40':g.apply(lambda d:(d.ayan_rank<=40).mean()).round(3),
  'prune_ok(test-like)':g.prune_ok.mean().round(3),'prune_unknown':g.apply(lambda d:d.prune_ok.isna().mean()).round(3),
  'r_addr_empty':g.r_addr_empty.mean().round(3),'r_native':g.r_native.mean().round(3),'oov_name':g.apply(lambda d:(d.r_oov>=0.99).mean()).round(3),
  'med_dup_name':g.s1_dup_name.median()})
print(T.to_string())
print('ALL: ayan@8', round((M.ayan_rank<=8).mean(),4), 'ayan@8 among not-retrieved(B-F)', round((M[~M.any_in].ayan_rank<=8).mean(),4))
# union of doubling every family's k (strict / ge), over all misses not certainly retrieved
NR=M[~M.any_inge]
for mult in (2,3,5):
    s=np.zeros(len(NR),bool); g_=np.zeros(len(NR),bool)
    for f in FAMS:
        kf,kr=KEY_FIELDS[f][2],KEY_FIELDS[f][3]
        s|=(NR[f+'_fr']<=mult*kf)|(NR[f+'_rr']<=mult*kr); g_|=(NR[f+'_frge']<=mult*kf)|(NR[f+'_rrge']<=mult*kr)
    print(f'all families k x{mult}: recovers (strict) {s.sum()} / (ge) {g_.sum()} of {len(NR)} not-certainly-retrieved')
for f in ['b_cgram','b_xkey','b_key']:
    kf,kr=KEY_FIELDS[f][2],KEY_FIELDS[f][3]
    for kk in (16,24,40):
        print(f, f'rev k {kr}->{kk}: +{int(((NR[f+"_rrge"]<=kk)).sum())} (ge)')
