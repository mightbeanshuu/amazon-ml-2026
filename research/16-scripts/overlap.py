import csv, sys
R='/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/real/'
T='/Users/mac/amazon-ml-2026/student_resource/dataset/test/test_source1.tsv'
def rows(p):
    with open(p, newline='') as f:
        r=csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
        for x in r: yield x
tr_na=set(); tr_n=set(); tr_a=set()
for x in rows(R+'train_source1.tsv'):
    n=x[1].strip().lower(); a=x[2].strip().lower()
    tr_na.add(hash((n,a))); tr_n.add(hash(n)); 
    if a: tr_a.add(hash(a))
from collections import Counter
c=Counter()
for x in rows(T):
    n=x[1].strip().lower(); a=x[2].strip().lower(); co=x[3]
    c[(co,'tot')]+=1
    c[(co,'na')]+= hash((n,a)) in tr_na
    c[(co,'n')]+= hash(n) in tr_n
    c[(co,'a')]+= (a!='' and hash(a) in tr_a)
for co in ['India','US','France']:
    t=c[(co,'tot')]
    print(co, t, 'exact(name,addr) in train S1: %.4f'%(c[(co,'na')]/t), 'name: %.4f'%(c[(co,'n')]/t), 'addr: %.4f'%(c[(co,'a')]/t))
